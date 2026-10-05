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

from ..artifact_validation import (
    DRAFT_BUNDLE_KIND,
    DRAFT_MODEL_ID,
    DRAFT_REVISION,
    validate_target_manifest,
)
from ..bundle import clone, digest, lm_head_compression
from ..draft import weight_digests
from ..source_validation import VerifiedSource, verify_source_checkpoint


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
    target: Path,
    source: Path,
    output: Path,
    *,
    draft_source: Path | None = None,
    verified_target_source: VerifiedSource | None = None,
    verified_draft_source: VerifiedSource | None = None,
) -> dict:
    """Write ``output`` with the pinned 0.6B checkpoint and a compiled verify head."""
    import coremltools as ct

    from .build import download_verified_source
    from .compress import compress_model, weight_bytes

    target, source, output = target.resolve(), source.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError("Use a new directory to preserve previous artifacts")
    manifest = json.loads((target / "manifest.json").read_text())
    target_info = validate_target_manifest(manifest)
    if verified_target_source is None:
        verified_target_source = verify_source_checkpoint(
            source,
            expected_model_id=target_info.model_id,
            expected_revision=target_info.source_revision,
            expected_content_sha256=target_info.source_content_sha256,
        )
    elif verified_target_source.root != source:
        raise ValueError("verified_target_source belongs to a different source directory")
    if (
        verified_target_source.model_id != target_info.model_id
        or verified_target_source.revision != target_info.source_revision
        or (
            target_info.source_content_sha256 is not None
            and verified_target_source.content_sha256 != target_info.source_content_sha256
        )
    ):
        raise ValueError("verified_target_source differs from the selected target")
    width = target_info.token_batch_size
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
        draft_source = draft_source.resolve()
        if verified_draft_source is None:
            verified_draft_source = verify_source_checkpoint(
                draft_source,
                expected_model_id=DRAFT_MODEL_ID,
                expected_revision=DRAFT_REVISION,
            )
        elif verified_draft_source.root != draft_source:
            raise ValueError("verified_draft_source belongs to a different source directory")
        if (
            verified_draft_source.model_id != DRAFT_MODEL_ID
            or verified_draft_source.revision != DRAFT_REVISION
        ):
            raise ValueError("verified_draft_source is not the pinned draft checkpoint")
        clone(draft_source, draft_path)
        # Bind the private copy even when the operator supplied a legacy
        # identity-only source.json. The original source remains untouched.
        (draft_path / "source.json").write_text(
            json.dumps(verified_draft_source.provenance(), indent=2) + "\n"
        )
    else:
        if verified_draft_source is not None:
            raise ValueError("verified_draft_source requires draft_source")
        verified_draft_source = download_verified_source(
            draft_path, revision=DRAFT_REVISION, model_id=DRAFT_MODEL_ID
        )
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
        "draft": {
            "model_id": DRAFT_MODEL_ID,
            "revision": DRAFT_REVISION,
            "content_sha256": verified_draft_source.content_sha256,
            "path": draft_path.name,
        },
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
            "source_content_sha256": verified_target_source.content_sha256,
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
