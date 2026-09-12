"""Ensure incomplete compiler evidence cannot masquerade as actual ANE usage."""

import json
from types import SimpleNamespace as NS

import pytest

from std_qwen3asr_ane.diagnostics import _inspect_plan, inspect_compute_plan, main


def test_nested_pipeline_keeps_unknown_devices_and_invalid_costs():
    ane = type("MLNeuralEngineComputeDevice", (), {})()
    cpu = type("MLCPUComputeDevice", (), {})()
    child = NS(operator_name="matmul", outputs=[NS(name="y")], blocks=[])
    parent = NS(operator_name="cond", outputs=[], blocks=[NS(operations=[child])])
    second = NS(operator_name="identity", outputs=[], blocks=[])
    program = NS(
        functions={
            "main": NS(block=NS(operations=[parent])),
            "other": NS(block=NS(operations=[second])),
        }
    )
    structure = NS(program=program, neuralnetwork=None, pipeline=None)
    pipeline = NS(
        program=None, neuralnetwork=None, pipeline=NS(sub_models=[("decoder", structure)])
    )
    usages = {
        id(parent): NS(preferred_compute_device=cpu, supported_compute_devices=[cpu]),
        id(child): NS(preferred_compute_device=ane, supported_compute_devices=[cpu, ane]),
        id(second): None,
    }
    costs = {id(parent): NS(weight=0.1), id(child): None, id(second): NS(weight=float("nan"))}
    plan = NS(
        model_structure=pipeline,
        get_compute_device_usage_for_mlprogram_operation=lambda op: usages[id(op)],
        get_estimated_cost_for_mlprogram_operation=lambda op: costs[id(op)],
    )

    report = _inspect_plan(plan)

    assert len(report["operations"]) == 3
    assert len(report["scope_summaries"]) == 2
    assert report["operations"][1]["nested_block_depth"] == 1
    assert report["operations"][1]["supported_devices"] == ["cpu", "ane"]
    assert report["summary"]["missing_cost_count"] == 2
    assert report["summary"]["invalid_cost_count"] == 1
    assert report["summary"]["unknown_device_count"] == 1
    assert report["summary"]["by_preferred_device"]["ane"]["missing_cost_count"] == 1
    assert report["actual_execution_verified"] is False
    json.dumps(report, allow_nan=False)


def test_unsupported_structure_is_explicit():
    plan = NS(model_structure=NS(program=None, neuralnetwork=None, pipeline=None))
    report = _inspect_plan(plan)
    assert report["unsupported_structure_paths"] == ["model"]
    assert report["operations"] == []
    assert report["actual_execution_verified"] is False


def test_bad_configuration_fails_before_importing_coreml():
    with pytest.raises(ValueError, match="compute_units"):
        inspect_compute_plan("missing.mlpackage", compute_units="ane_only")


def test_missing_model_cli_returns_failure_without_disclosing_path(tmp_path, capsys):
    missing = tmp_path / "sensitive-user-name" / "missing.mlpackage"
    assert main([str(missing)]) == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert str(tmp_path) not in output.err
    assert "sensitive-user-name" not in output.err
    assert json.loads(output.err)["status"] == "unavailable"


def test_cli_handles_coreml_bare_exception(monkeypatch, capsys):
    def fail(*args):
        raise Exception("Core ML model was not compiled")  # noqa: TRY002 -- Matches Apple's API.

    monkeypatch.setattr("std_qwen3asr_ane.diagnostics.inspect_compute_plan", fail)
    assert main(["model.mlpackage"]) == 1
    output = capsys.readouterr()
    assert json.loads(output.err)["error_type"] == "Exception"
