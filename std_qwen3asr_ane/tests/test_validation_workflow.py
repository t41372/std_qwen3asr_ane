"""Evidence gates must not promote missing, stale, partial, or smoke data to release claims."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
spec = importlib.util.spec_from_file_location(
    "run_validation", ROOT / "experiments/run_validation.py"
)
workflow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workflow)


def placement(operations):
    return {
        "schema_version": 1,
        "evidence_kind": "anticipated_compute_plan",
        "operations": operations,
    }


def test_placement_excludes_constants_but_not_unknown_compute():
    constant = {"operator": "const", "preferred_device": None}
    ane = {"operator": "ios18.conv", "preferred_device": "ane"}
    unknown = {"operator": "ios18.matmul", "preferred_device": None}
    assert workflow.placement_gate(placement([constant, ane]))["status"] == "pass"
    result = workflow.placement_gate(placement([constant, ane, unknown]))
    assert result["status"] == "inconclusive"
    assert result["nonconstant_operations"] == 2
    assert result["unknown_nonconstant_operations"] == 1
    assert result["actual_execution_verified"] is False


def test_known_cpu_placement_below_threshold_fails():
    operations = [{"operator": "conv", "preferred_device": "ane"}] * 8
    operations += [{"operator": "matmul", "preferred_device": "cpu"}] * 2
    assert workflow.placement_gate(placement(operations))["status"] == "fail"
    assert workflow.placement_gate(placement([]))["status"] == "inconclusive"
    assert workflow.placement_gate({})["status"] == "inconclusive"


def rows(language, count, duration, errors=1):
    reference = " ".join(["word"] * 100)
    hypothesis = " ".join(["wrong"] * errors + ["word"] * (100 - errors))
    return [
        {
            "id": f"{language}-{index}",
            "phase": "measured",
            "repeat": 0,
            "language": language,
            "audio_seconds": duration,
            "reference": reference,
            "hypothesis": hypothesis,
        }
        for index in range(count)
    ]


def comparison(**intervals):
    return {
        "valid_comparison": True,
        "by_language": {
            language: {"wer": {"ci95": interval}, "cer": {"ci95": interval}}
            for language, interval in intervals.items()
        },
    }


def test_smoke_functionality_does_not_pass_quality():
    baseline = rows("en", 2, 2)
    result = workflow.quality_gate(
        comparison(en=[0, 0]), baseline, baseline, [r["id"] for r in baseline]
    )
    assert result["status"] == "inconclusive"
    assert result["functional_status"] == "pass"
    assert result["release_coverage_eligible"] is False


def test_per_language_relative_gate_cannot_hide_behind_aggregate():
    baseline = rows("en", 500, 36, errors=2) + rows("de", 500, 36, errors=2)
    result = workflow.quality_gate(
        comparison(en=[0, 0.0005], de=[0.0015, 0.002]),
        baseline,
        baseline,
        [r["id"] for r in baseline],
    )
    assert result["status"] == "fail"
    assert result["by_language"]["en"]["status"] == "pass"
    assert result["by_language"]["de"]["status"] == "fail"
    assert result["by_language"]["de"]["noninferiority_margin"] == pytest.approx(0.001)


def test_paired_subset_is_not_complete_quality_evidence():
    baseline = rows("en", 2, 2)
    result = workflow.quality_gate(comparison(en=[0, 0]), baseline, baseline, ["missing"])
    assert result["status"] == "fail"
    assert result["functional_status"] == "fail"


def trace_report():
    return {
        "evidence_kind": "instruments_ane_hardware_intervals",
        "ane_prediction_rows": 2,
        "target_exit_status": "0",
        "target_pid": 1234,
        "ane_pid_attribution_available": False,
        "gpu_hardware_rows_target_pid": 0,
        "ane_labels": [
            {
                "label": "encoder_main__Op1_AneInference Prediction",
                "count": 2,
                "is_prediction": True,
                "duration_sum_ns": 1000,
            }
        ],
    }


def test_trace_requires_schema_model_binding_and_all_phase_labels():
    phases = {"encoder": Path("encoder.mlpackage")}
    assert workflow.trace_gate({}, phases, "abc")["status"] == "inconclusive"
    report = trace_report()
    assert workflow.trace_gate(report, phases, "abc")["status"] == "inconclusive"
    report["model_manifest_sha256"] = "abc"
    report["isolated_trace_verified"] = True
    result = workflow.trace_gate(report, phases, "abc")
    assert result["status"] == "pass"
    assert result["ane_pid_attribution_available"] is False
    phases["decoder"] = Path("decoder_00.mlpackage")
    assert workflow.trace_gate(report, phases, "abc")["status"] == "inconclusive"


def test_inconsistent_trace_counts_are_not_hardware_proof():
    report = trace_report()
    report["ane_prediction_rows"] = 999
    report["model_manifest_sha256"] = "abc"
    assert (
        workflow.trace_gate(report, {"encoder": Path("encoder.mlpackage")}, "abc")["status"]
        == "inconclusive"
    )


def test_timeout_is_failure_and_preserves_both_logs(tmp_path):
    result = workflow.run_command(
        [sys.executable, "-c", "import time; time.sleep(30)"], tmp_path, "timeout", 0.05
    )
    assert result["status"] == "fail"
    assert result["reason"] == "subprocess_timeout"
    assert Path(result["stdout"]).is_file()
    assert Path(result["stderr"]).is_file()


def test_resume_rejects_modified_evidence(tmp_path):
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"status": "pass"}))
    step = {"evidence_hashes": {str(report): workflow.file_hash(report)}}
    assert workflow.evidence_intact(step)
    report.write_text("{}")
    assert not workflow.evidence_intact(step)


def test_unrun_steps_and_unmeasured_energy_cannot_pass():
    result = workflow.claims({name: {"status": "not_run"} for name in workflow.STEPS})
    assert result["protocol"]["status"] == "not_run"
    assert result["quality"]["status"] == "not_run"
    assert result["energy"]["status"] == "unavailable"
    assert result["release_ready"] is False
    assert result["performance_improvement_verified"] is False


def test_missing_protocol_cases_cannot_pass_pytest_gate(tmp_path):
    junit = tmp_path / "pytest.xml"
    junit.write_text(
        '<testsuites><testsuite><testcase classname="test_other" /></testsuite></testsuites>'
    )
    assert workflow.protocol_result(junit)["status"] == "inconclusive"


def test_resume_cli_rejects_changed_manifest_without_running_models(tmp_path, monkeypatch):
    model, source, output = (tmp_path / name for name in ("model", "source", "output"))
    model.mkdir()
    source.mkdir()
    package = model / "encoder.mlpackage"
    package.mkdir()
    (package / "Manifest.json").write_text("{}")
    model_manifest = model / "manifest.json"
    model_manifest.write_text(json.dumps({"files": {"encoder": "encoder.mlpackage"}}))
    (source / "config.json").write_text("{}")
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"content hashing only; trace-only test never decodes audio")
    manifest = tmp_path / "corpus.jsonl"
    manifest.write_text(
        json.dumps({"id": "sample", "audio_path": str(audio), "reference": "hi"}) + "\n"
    )
    arguments = [
        "run_validation.py",
        "--model-dir",
        str(model),
        "--source-dir",
        str(source),
        "--manifest",
        str(manifest),
        "--output-dir",
        str(output),
        "--steps",
        "trace",
    ]
    monkeypatch.setattr(sys, "argv", arguments)
    assert workflow.main() == 0
    result = json.loads((output / "claims.json").read_text())
    assert result["actual_device"]["status"] == "unavailable"
    assert result["protocol"]["status"] == "not_run"
    monkeypatch.setattr(sys, "argv", [*arguments, "--resume"])
    assert workflow.main() == 0
    assert len(list(output.glob("trace-attempt-*"))) == 1
    model_manifest.write_text(
        json.dumps({"files": {"encoder": "encoder.mlpackage"}, "revision": 2})
    )
    with pytest.raises(SystemExit) as error:
        workflow.main()
    assert error.value.code == 2
    assert len(list(output.glob("trace-attempt-*"))) == 1


def test_attribution_cannot_promote_background_or_mismatched_trace():
    report = trace_report()
    phases = {"encoder": Path("encoder.mlpackage")}
    attribution = {
        "schema_version": 1,
        "bundle_manifest_sha256": "abc",
        "trace_summary_sha256": "tracehash",
        "target_pid": 1234,
        "isolated_trace_verified": False,
        "background_workload_confirmed": True,
        "candidate_models": [
            {
                "model": "encoder.mlpackage",
                "compiled_label": report["ane_labels"][0]["label"],
                "prediction_count": 2,
            }
        ],
    }
    result = workflow.trace_gate(report, phases, "abc", attribution, "tracehash")
    assert result["status"] == "inconclusive"
    assert "background_workload_confirmed" in result["reasons"]
    attribution.update(isolated_trace_verified=True, background_workload_confirmed=False)
    assert workflow.trace_gate(report, phases, "abc", attribution, "tracehash")["status"] == "pass"
    assert (
        workflow.trace_gate(report, phases, "abc", attribution, "different")["status"]
        == "inconclusive"
    )
    attribution["candidate_models"][0]["prediction_count"] = 999
    assert (
        workflow.trace_gate(report, phases, "abc", attribution, "tracehash")["status"]
        == "inconclusive"
    )
