"""Hardware evidence must close against actual offline prediction counts."""

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments"))
from bind_trace_evidence import expected_prediction_counts, verify_bundle_identity


def test_batched_frontend_trace_counts_include_b1_tail_and_b4():
    manifest = {
        "frontend": {"offline_batch_size": 4},
        "files": {
            "frontend": "frontend.mlmodelc",
            "frontend_batched": "frontend-b4.mlmodelc",
            "encoder": "encoder.mlmodelc",
            "lm_head": "lm_head.mlmodelc",
        },
        "decoder_partitions": ["decoder_00.mlmodelc", "decoder_14.mlmodelc"],
    }
    row = {
        "phase": "measured",
        "error": None,
        "backend_timings": {
            "computed_feature_frames": 420,
            "frontend_calls": 2,
            "encoder_calls": 1,
            "head_calls": 3,
            "prefill_calls": 5,
            "generation_decoder_calls": 2,
        },
    }
    assert expected_prediction_counts(manifest, [row]) == {
        "frontend": 1,
        "frontend-b4": 1,
        "encoder": 1,
        "lm_head": 3,
        "decoder_00": 7,
        "decoder_14": 7,
    }
    row["backend_timings"]["frontend_calls"] = 5
    with pytest.raises(ValueError, match="offline frontend"):
        expected_prediction_counts(manifest, [row])
    row["error"] = "failed prediction"
    with pytest.raises(ValueError, match="failed workload"):
        expected_prediction_counts(manifest, [row])


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bundle(root: Path, name: str, weights: bytes, graphs=("frontend",)) -> Path:
    bundle = root / name
    for graph in graphs:
        (bundle / f"{graph}.mlmodelc" / "weights").mkdir(parents=True)
        (bundle / f"{graph}.mlmodelc" / "weights" / "weight.bin").write_bytes(weights)
    manifest = {"model_id": name, "files": {graph: f"{graph}.mlmodelc" for graph in graphs}}
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    return bundle


def _record_trace(root: Path, bundle: Path, *, with_payloads: bool = True) -> tuple[Path, Path]:
    """Write the placement summary, workload summary and commands as recorded for a bundle."""
    placement = root / f"{bundle.name}-placement"
    placement.mkdir()
    weight = bundle / "frontend.mlmodelc" / "weights" / "weight.bin"
    summary = {"manifest_sha256": _sha256(bundle / "manifest.json")}
    if with_payloads:
        summary["models"] = [
            {
                "model": "frontend.mlmodelc",
                "files": [{"path": str(weight.relative_to(bundle)), "sha256": _sha256(weight)}],
            }
        ]
    else:
        # Round-two inspections recorded the manifest under the binder's key and no payloads.
        summary = {"bundle_manifest_sha256": summary["manifest_sha256"]}
    (placement / "summary.json").write_text(json.dumps(summary))
    prefix = root / f"{bundle.name}-trace"
    Path(f"{prefix}-workload.jsonl.summary.json").write_text(
        json.dumps(
            {
                "model_metadata": {
                    "metadata_files": {
                        "manifest.json": {"sha256": _sha256(bundle / "manifest.json")}
                    }
                }
            }
        )
    )
    Path(f"{prefix}-commands.json").write_text(
        json.dumps([{"argv": ["evaluate.py", "--model-dir", str(bundle)], "exit_code": 0}])
    )
    return placement, prefix


def test_binding_requires_the_recorded_bundle(tmp_path: Path):
    a = _bundle(tmp_path, "a", b"weights-a")
    b = _bundle(tmp_path, "b", b"weights-b")
    placement, prefix = _record_trace(tmp_path, a)
    checks = verify_bundle_identity(a, placement, prefix)
    assert checks["placement_manifest"] == checks["workload_manifest"] == "verified"
    assert checks["placement_payloads"] == "1 files verified"
    assert checks["commands_model_dir"] == "verified"
    # Same graph names, different manifest and weights: every recorded input disagrees.
    with pytest.raises(RuntimeError, match="different bundle manifest"):
        verify_bundle_identity(b, placement, prefix)
    # The manifest still matches but a weight payload was swapped after inspection.
    (a / "frontend.mlmodelc" / "weights" / "weight.bin").write_bytes(b"weights-b")
    with pytest.raises(RuntimeError, match="payload differs"):
        verify_bundle_identity(a, placement, prefix)


def test_binding_checks_workload_and_commands_separately(tmp_path: Path):
    a = _bundle(tmp_path, "a", b"weights-a")
    b = _bundle(tmp_path, "b", b"weights-b")
    placement, prefix = _record_trace(tmp_path, a, with_payloads=False)
    checks = verify_bundle_identity(a, placement, prefix)
    assert checks["placement_payloads"] == "not recorded"
    workload = Path(f"{prefix}-workload.jsonl.summary.json")
    recorded = workload.read_text()
    workload.write_text(
        recorded.replace(_sha256(a / "manifest.json"), _sha256(b / "manifest.json"))
    )
    with pytest.raises(RuntimeError, match="Workload was recorded"):
        verify_bundle_identity(a, placement, prefix)
    workload.write_text(recorded)
    commands = Path(f"{prefix}-commands.json")
    commands.write_text(json.dumps([{"argv": ["evaluate.py", "--model-dir", str(b)]}]))
    with pytest.raises(RuntimeError, match="commands named"):
        verify_bundle_identity(a, placement, prefix)
    commands.write_text(
        json.dumps([{"argv": ["evaluate.py", "--model-dir", str(tmp_path / "gone")]}])
    )
    assert verify_bundle_identity(a, placement, prefix)["commands_model_dir"].startswith(
        "path no longer"
    )
    (placement / "summary.json").unlink()
    with pytest.raises(RuntimeError, match="no summary.json"):
        verify_bundle_identity(a, placement, prefix)


def test_binding_requires_every_graph_in_the_placement_summary(tmp_path: Path):
    a = _bundle(tmp_path, "a", b"weights-a", graphs=("frontend", "encoder"))
    # The recorded inspection lists only the frontend, as an interrupted rerun would.
    placement, prefix = _record_trace(tmp_path, a)
    with pytest.raises(RuntimeError, match="does not list every bundle graph.*encoder"):
        verify_bundle_identity(a, placement, prefix)
