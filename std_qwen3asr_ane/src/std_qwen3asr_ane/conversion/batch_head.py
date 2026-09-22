"""Build a target-bound compact vocabulary head for offline request packing."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import shutil
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from ..batching import BATCH_HEAD_BUNDLE_KIND, _head_weight_digests
from ..bundle import digest, lm_head_compression
from ..source_validation import VerifiedSource, verify_source_checkpoint


def build_batch_head(
    target: Path,
    source: Path,
    output: Path,
    *,
    verified_source: VerifiedSource | None = None,
) -> dict:
    """Build a compact width-T head whose weights match ``target/lm_head``.

    This auxiliary artifact contains no checkpoint and cannot be used with an
    arbitrary same-shaped target: its manifest and every packed weight payload
    are checked again by :func:`validate_target_bound_batch_head` before inference.
    """
    import coremltools as ct

    from ..artifact_validation import validate_target_manifest
    from .compress import compress_model, weight_bytes
    from .draft import build_compact_head

    target, source, output = target.resolve(), source.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError("Use a new directory to preserve previous batch-head artifacts")
    manifest = json.loads((target / "manifest.json").read_text())
    info = validate_target_manifest(manifest)
    if info.token_batch_size < 2:
        raise ValueError("The target must use a token batch size above one for batch recognition")
    if verified_source is None:
        verified_source = verify_source_checkpoint(
            source,
            expected_model_id=info.model_id,
            expected_revision=info.source_revision,
            expected_content_sha256=info.source_content_sha256,
        )
    elif verified_source.root != source:
        raise ValueError("verified_source belongs to a different source directory")
    if (
        verified_source.model_id != info.model_id
        or verified_source.revision != info.source_revision
        or (
            info.source_content_sha256 is not None
            and verified_source.content_sha256 != info.source_content_sha256
        )
    ):
        raise ValueError("verified_source differs from the selected target")
    lm_head = target / info.files["lm_head"]
    if lm_head.suffix != ".mlmodelc":
        raise ValueError("The target must be compiled before building a compact batch head")
    output.mkdir(parents=True)
    started = perf_counter()
    fp16 = output / "batch_head_fp16.mlpackage"
    head = build_compact_head(
        source,
        fp16,
        token_batch_size=info.token_batch_size,
        residual_scale=info.residual_scale,
    )
    compression = lm_head_compression(manifest)
    if compression["scheme"] is None:
        package = fp16
        counts = None
    else:
        package = output / "batch_head.mlpackage"
        counts = compress_model(
            fp16,
            package,
            compression["scheme"],
            compression["bits"],
            compression["group_size"],
        )
    compiled = output / "batch_head.mlmodelc"
    ct.models.utils.compile_model(str(package), destination_path=str(compiled))
    for intermediate in {fp16, package}:
        shutil.rmtree(intermediate)
    weights = _head_weight_digests(compiled)
    if weights != _head_weight_digests(lm_head):
        raise RuntimeError("Batch-head weights do not match the selected target lm_head weights")
    record = {
        "schema_version": 1,
        "kind": BATCH_HEAD_BUNDLE_KIND,
        "created_at": datetime.now(UTC).isoformat(),
        "host": platform.platform(),
        "head": {
            "path": compiled.name,
            "token_batch_size": info.token_batch_size,
            **head,
            "weight_compression": compression,
            "compressed_weight_counts": counts,
            "weight_bytes": weight_bytes(compiled),
            "weight_sha256": list(weights),
            "build_seconds": perf_counter() - started,
        },
        "target": {
            "model_id": info.model_id,
            "source_revision": info.source_revision,
            "source_content_sha256": verified_source.content_sha256,
            "token_batch_size": info.token_batch_size,
            "tokenizer_sha256": digest(target / info.files["tokenizer"]),
            "weight_compression": compression,
            "manifest_sha256": digest(target / "manifest.json"),
        },
        "versions": {
            name: importlib.metadata.version(name) for name in ("torch", "coremltools", "numpy")
        },
    }
    (output / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    return record
