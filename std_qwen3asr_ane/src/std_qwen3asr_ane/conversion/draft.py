"""Build the optional draft bundle: the 0.6B checkpoint plus a verify head.

The verify head is the target bundle's vocabulary projection rebuilt for one
token block, returning only each chunk's maximum and its index. It is built
from the same source weights and compressed with the target's own settings so
its argmax reproduces the target ``lm_head`` (checked at load time by probing
both heads and, on real decoder states, by ``experiments/benchmark_compact_head.py``).
"""

from __future__ import annotations

import importlib.metadata
import json
import platform
import shutil
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from ..bundle import clone, digest
from ..draft import DRAFT_BUNDLE_KIND, DRAFT_MODEL_ID, DRAFT_REVISION


def build_compact_head(source: Path, output: Path, *, token_batch_size: int) -> dict:
    """Trace a per-chunk (max, argmax) head from the target's source weights."""
    import coremltools as ct
    import numpy as np
    import torch

    from .decoder import LanguageHead, SourceWeights

    class CompactHead(LanguageHead):
        def forward(self, hidden_states):
            normalized = self.norm(hidden_states)
            values, indices = [], []
            for head in self.heads:
                value, index = torch.max(head(normalized).squeeze(2), dim=1)
                values.append(value)
                # Local indices stay int32; FP16 cannot carry exact integers above 2048.
                indices.append(index.to(torch.int32))
            return torch.stack(values, dim=1), torch.stack(indices, dim=1)

    torch.set_num_threads(4)
    config = json.loads((source / "config.json").read_text())["thinker_config"]["text_config"]
    weights = SourceWeights(source)
    module = CompactHead(config).eval()
    module.norm.weight.data.copy_(weights.get("thinker.model.norm.weight"))
    embedding = weights.get("thinker.model.embed_tokens.weight")
    offset = 0
    for head in module.heads:
        count = head.out_channels
        head.weight.data.copy_(embedding[offset : offset + count, :, None, None])
        offset += count
    del embedding
    example = torch.zeros(1, config["hidden_size"], 1, token_batch_size)
    model = ct.convert(
        torch.jit.trace(module, example, check_trace=False),
        inputs=[ct.TensorType(name="hidden_states", shape=example.shape, dtype=np.float16)],
        outputs=[
            ct.TensorType(name="max_values", dtype=np.float16),
            ct.TensorType(name="max_indices", dtype=np.int32),
        ],
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        skip_model_load=True,
    )
    model.save(str(output))
    return {"vocabulary_chunk": module.heads[0].out_channels, "chunks": len(module.heads)}


def build_draft_bundle(
    target: Path, source: Path, output: Path, *, draft_source: Path | None = None
) -> dict:
    """Write ``output`` with the pinned 0.6B checkpoint and a compiled verify head."""
    import coremltools as ct

    from .build import download_source
    from .compress import compress_model, weight_bytes

    target, source, output = target.resolve(), source.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError("Use a new directory to preserve previous artifacts")
    manifest = json.loads((target / "manifest.json").read_text())
    if manifest.get("schema_version") != 1 or manifest.get("model_id") != "Qwen/Qwen3-ASR-1.7B":
        raise ValueError("The target must be a Qwen3-ASR 1.7B bundle")
    provenance = json.loads((source / "source.json").read_text())
    if (
        provenance.get("model_id") != manifest["model_id"]
        or provenance.get("revision") != manifest["source_revision"]
    ):
        raise ValueError("The source checkpoint must match the target bundle's revision")
    width = int(manifest.get("token_batch_size", 1))
    if width < 2:
        raise ValueError("The target must use a token batch size above 1 to verify proposals")
    compression = manifest.get("weight_compression")
    output.mkdir(parents=True)
    draft_path = output / "Qwen3-ASR-0.6B"
    if draft_source is not None:
        record = json.loads((draft_source / "source.json").read_text())
        if record != {"model_id": DRAFT_MODEL_ID, "revision": DRAFT_REVISION}:
            raise ValueError(f"The draft checkpoint must be {DRAFT_MODEL_ID}@{DRAFT_REVISION}")
        clone(draft_source, draft_path)
    else:
        download_source(draft_path, revision=DRAFT_REVISION, model_id=DRAFT_MODEL_ID)
    started = perf_counter()
    fp16 = output / "verify_head_fp16.mlpackage"
    head = build_compact_head(source, fp16, token_batch_size=width)
    if compression is None:
        package = fp16
        counts = None
    else:
        package = output / "verify_head.mlpackage"
        counts = compress_model(
            fp16, package, compression["scheme"], compression["bits"], compression["group_size"]
        )
    compiled = output / "verify_head.mlmodelc"
    ct.models.utils.compile_model(str(package), destination_path=str(compiled))
    # Only the compiled head is loaded; drop the intermediate packages (about 0.9 GB).
    for intermediate in {fp16, package}:
        shutil.rmtree(intermediate)
    tokenizer = target / manifest["files"]["tokenizer"]
    record = {
        "schema_version": 1,
        "kind": DRAFT_BUNDLE_KIND,
        "created_at": datetime.now(UTC).isoformat(),
        "host": platform.platform(),
        "draft": {"model_id": DRAFT_MODEL_ID, "revision": DRAFT_REVISION, "path": draft_path.name},
        "verify_head": {
            "path": compiled.name,
            "token_batch_size": width,
            **head,
            "weight_compression": None
            if compression is None
            else {key: compression[key] for key in ("scheme", "bits", "group_size")},
            "compressed_weight_counts": counts,
            "weight_bytes": weight_bytes(compiled),
            "weight_sha256": [digest(path) for path in sorted(compiled.rglob("weight.bin"))],
            "build_seconds": perf_counter() - started,
        },
        "target": {
            "model_id": manifest["model_id"],
            "source_revision": manifest["source_revision"],
            "token_batch_size": width,
            "tokenizer_sha256": digest(tokenizer),
            "weight_compression": {
                key: (None if compression is None else compression.get(key))
                for key in ("scheme", "bits", "group_size")
            },
            "manifest_sha256": digest(target / "manifest.json"),
        },
        "versions": {
            name: importlib.metadata.version(name) for name in ("torch", "coremltools", "numpy")
        },
    }
    (output / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    return record
