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

from ..bundle import SUPPORTED_SCHEMA_VERSIONS, clone, digest, lm_head_compression
from ..draft import DRAFT_BUNDLE_KIND, DRAFT_MODEL_ID, DRAFT_REVISION, weight_digests


def build_compact_head(
    source: Path, output: Path, *, token_batch_size: int, residual_scale: float = 1.0
) -> dict:
    """Trace a per-chunk (max, argmax) head from the target's source weights."""
    import coremltools as ct
    import numpy as np
    import torch

    from .decoder import CompactLanguageHead, SourceWeights

    torch.set_num_threads(4)
    config = json.loads((source / "config.json").read_text())["thinker_config"]["text_config"]
    weights = SourceWeights(source)
    module = CompactLanguageHead(config, residual_scale=residual_scale).eval()
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
    if (
        manifest.get("schema_version") not in SUPPORTED_SCHEMA_VERSIONS
        or manifest.get("model_id") != "Qwen/Qwen3-ASR-1.7B"
    ):
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
    # The head must mirror lm_head exactly: compressed only if lm_head was.
    compression = lm_head_compression(manifest)
    compressed_head = compression["scheme"] is not None
    lm_head = target / manifest["files"]["lm_head"]
    if lm_head.suffix != ".mlmodelc":
        raise ValueError("The target must be a compiled bundle (qwen3-asr-ane compile)")
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
    head = build_compact_head(
        source,
        fp16,
        token_batch_size=width,
        residual_scale=float(manifest.get("residual_scale", 1.0)),
    )
    if not compressed_head:
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
    digests = weight_digests(compiled)
    if digests != weight_digests(lm_head):
        raise RuntimeError(
            "The verify head weights do not match the target lm_head weights; refusing the bundle"
        )
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
            "weight_compression": compression if compressed_head else None,
            "compressed_weight_counts": counts,
            "weight_bytes": weight_bytes(compiled),
            "weight_sha256": digests,
            "build_seconds": perf_counter() - started,
        },
        "target": {
            "model_id": manifest["model_id"],
            "source_revision": manifest["source_revision"],
            "token_batch_size": width,
            "tokenizer_sha256": digest(tokenizer),
            "weight_compression": compression,
            "manifest_sha256": digest(target / "manifest.json"),
        },
        "versions": {
            name: importlib.metadata.version(name) for name in ("torch", "coremltools", "numpy")
        },
    }
    (output / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    return record
