"""Verify a managed standalone batch-head pull without downloading artifacts.

The verifier checks the worker receipt, the target/head binding, source
content provenance, and (when requested) a real Core ML prediction against the
target language head.  It reads an existing validation directory and emits one
portable JSON record to stdout; it never acquires or modifies an artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from std_qwen3asr_ane.acquisition import inspect_conversion_worker
from std_qwen3asr_ane.artifact_validation import validate_target_manifest
from std_qwen3asr_ane.batching import validate_target_bound_batch_head
from std_qwen3asr_ane.bundle import language_head_output
from std_qwen3asr_ane.runtime import PersistentInputModel, compact_token, logits_token
from std_qwen3asr_ane.source_validation import verify_source_checkpoint


def _sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _display_path(path: Path) -> str:
    """Prefer a repository-relative path while remaining usable elsewhere."""
    try:
        return path.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return str(path)


def verify(root: Path, *, predict: bool) -> dict[str, Any]:
    root = root.expanduser().resolve()
    target_root = root / "model/qwen3-asr-1.7b"
    head_root = root / "model/qwen3-asr-1.7b-batch-head"
    source_root = root / "source/Qwen3-ASR-1.7B"

    target_manifest_path = target_root / "manifest.json"
    head_manifest_path = head_root / "manifest.json"
    target_info = validate_target_manifest(_json(target_manifest_path))
    source = verify_source_checkpoint(
        source_root,
        expected_model_id=target_info.model_id,
        expected_revision=target_info.source_revision,
    )
    head = validate_target_bound_batch_head(
        head_root,
        target=target_info,
        target_root=target_root,
    )

    config = type("WorkerConfig", (), {"source_dir": source_root})()
    worker = inspect_conversion_worker(config)
    if not worker.ready or worker.python_executable is None or worker.receipt is None:
        raise RuntimeError(f"Managed conversion worker is not ready: {worker.reason}")

    head_manifest = _json(head_manifest_path)
    declared_source_digest = head_manifest.get("target", {}).get("source_content_sha256")
    if declared_source_digest != source.content_sha256:
        raise ValueError("Batch head does not record the verified source content digest")

    report: dict[str, Any] = {
        "schema_version": 1,
        "root": _display_path(root),
        "target": {
            "manifest_sha256": _sha256(target_manifest_path),
            "model_id": target_info.model_id,
            "source_revision": target_info.source_revision,
            "token_batch_size": target_info.token_batch_size,
        },
        "source": {
            "model_id": source.model_id,
            "revision": source.revision,
            "content_sha256": source.content_sha256,
            "total_bytes": sum(item.size_bytes for item in source.files),
            "files": [item.to_json() for item in source.files],
        },
        "batch_head": {
            "manifest_sha256": _sha256(head_manifest_path),
            "path": str(head.path.relative_to(root)),
            "token_batch_size": head.token_batch_size,
            "vocabulary_chunk": head.vocabulary_chunk,
            "weight_sha256": head_manifest["head"]["weight_sha256"],
            "build_seconds": head_manifest["head"]["build_seconds"],
            "versions": head_manifest["versions"],
        },
        "worker": {
            "root": str(worker.python_executable.parent.parent.relative_to(root)),
            "receipt": worker.receipt,
        },
        "prediction": {"requested": predict},
    }

    if predict:
        import coremltools as ct

        compute_unit = ct.ComputeUnit.CPU_AND_NE
        serial_model = PersistentInputModel(
            ct.models.CompiledMLModel(
                str(target_root / target_info.files["lm_head"]),
                compute_units=compute_unit,
            )
        )
        packed_model = PersistentInputModel(
            ct.models.CompiledMLModel(str(head.path), compute_units=compute_unit)
        )
        try:
            embeddings = np.load(
                target_root / target_info.files["embedding"],
                mmap_mode="r",
                allow_pickle=False,
            )
            generator = np.random.default_rng(20261002)
            hidden = generator.standard_normal(
                (1, embeddings.shape[1], 1, 3)
            ).astype(np.float32)
            padded = np.zeros((*hidden.shape[:-1], head.token_batch_size), np.float32)
            padded[..., : hidden.shape[-1]] = hidden
            packed = packed_model.predict({"hidden_states": padded})
            values = packed["max_values"][0]
            indices = packed["max_indices"][0]
            chunks = np.argmax(values, axis=0)
            packed_tokens = [
                int(chunks[index] * head.vocabulary_chunk + indices[chunks[index], index])
                for index in range(hidden.shape[-1])
            ]

            serial_kind = language_head_output(dict(target_info.manifest))
            serial_tokens = []
            for index in range(hidden.shape[-1]):
                outputs = serial_model.predict(
                    {"hidden_states": hidden[..., index : index + 1].astype(np.float16)}
                )
                if serial_kind["kind"] == "chunk_max":
                    token = compact_token(
                        outputs,
                        vocabulary_size=embeddings.shape[0],
                        chunk_size=serial_kind["vocabulary_chunk"],
                    )
                else:
                    token = logits_token(outputs, vocabulary_size=embeddings.shape[0])
                serial_tokens.append(token)
            if packed_tokens != serial_tokens:
                raise RuntimeError("Packed and serial language heads selected different tokens")
            report["prediction"] = {
                "requested": True,
                "compute_units": "cpu_and_ne",
                "seed": 20261002,
                "rows": hidden.shape[-1],
                "packed_tokens": packed_tokens,
                "serial_tokens": serial_tokens,
                "equal": True,
            }
        finally:
            PersistentInputModel.close_many((serial_model, packed_model))

    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "--predict",
        action="store_true",
        help="Load the target and new batch head and compare real Core ML predictions.",
    )
    args = parser.parse_args()
    print(json.dumps(verify(args.root, predict=args.predict), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
