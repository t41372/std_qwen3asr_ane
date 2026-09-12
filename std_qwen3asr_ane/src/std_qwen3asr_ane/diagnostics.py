"""Inspect anticipated Core ML placement without claiming hardware execution."""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

COMPUTE_UNITS = ("cpu_and_ne", "cpu_only", "cpu_and_gpu", "all")
_DEVICE_NAMES = {
    "MLCPUComputeDevice": "cpu",
    "MLGPUComputeDevice": "gpu",
    "MLNeuralEngineComputeDevice": "ane",
}


def inspect_environment() -> dict[str, Any]:
    """Return software/platform facts without hostnames or hardware identifiers."""
    packages = {}
    for name in ("coremltools", "numpy", "standard-asr", "std-qwen3asr-ane"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {
        "system": platform.system(),
        "macos_version": platform.mac_ver()[0] or None,
        "kernel_release": platform.release(),
        "architecture": platform.machine(),
        "python_version": platform.python_version(),
        "packages": packages,
    }


def _device_name(device: Any) -> str | None:
    if device is None:
        return None
    name = type(device).__name__
    return _DEVICE_NAMES.get(name, name)


def _operation_row(plan: Any, operation: Any, path: str, scope: str, depth: int) -> dict:
    usage = plan.get_compute_device_usage_for_mlprogram_operation(operation)
    cost = plan.get_estimated_cost_for_mlprogram_operation(operation)
    weight = float(cost.weight) if cost is not None else None
    valid = weight is not None and math.isfinite(weight) and 0 <= weight <= 1
    return {
        "path": path,
        "scope": scope,
        "nested_block_depth": depth,
        "operator": operation.operator_name,
        "outputs": [output.name for output in operation.outputs],
        "preferred_device": _device_name(usage.preferred_compute_device) if usage else None,
        "supported_devices": [_device_name(device) for device in usage.supported_compute_devices]
        if usage
        else None,
        "estimated_cost_weight": weight if valid else None,
        "cost_status": "available" if valid else "missing" if cost is None else "invalid",
    }


def _collect_structure(plan: Any) -> tuple[list[dict], list[str]]:
    rows: list[dict] = []
    unsupported: list[str] = []

    def visit_block(block: Any, path: str, scope: str, depth: int) -> None:
        for index, operation in enumerate(block.operations):
            operation_path = f"{path}/operations/{index}"
            rows.append(_operation_row(plan, operation, operation_path, scope, depth))
            for child_index, child in enumerate(operation.blocks):
                visit_block(child, f"{operation_path}/blocks/{child_index}", scope, depth + 1)

    def visit_structure(structure: Any, path: str) -> None:
        if structure.program is not None:
            for name, function in structure.program.functions.items():
                scope = f"{path}/functions/{name}"
                visit_block(function.block, scope, scope, 0)
        elif structure.neuralnetwork is not None:
            for index, layer in enumerate(structure.neuralnetwork.layers):
                usage = plan.get_compute_device_usage_for_neuralnetwork_layer(layer)
                rows.append(
                    {
                        "path": f"{path}/layers/{index}",
                        "scope": path,
                        "nested_block_depth": 0,
                        "operator": layer.type,
                        "name": layer.name,
                        "outputs": list(layer.output_names),
                        "preferred_device": _device_name(usage.preferred_compute_device)
                        if usage
                        else None,
                        "supported_devices": [
                            _device_name(device) for device in usage.supported_compute_devices
                        ]
                        if usage
                        else None,
                        "estimated_cost_weight": None,
                        "cost_status": "unsupported_model_type",
                    }
                )
        elif structure.pipeline is not None:
            for index, (name, child) in enumerate(structure.pipeline.sub_models):
                visit_structure(child, f"{path}/models/{index}:{name}")
        else:
            unsupported.append(path)

    visit_structure(plan.model_structure, "model")
    return rows, unsupported


def _summarize(rows: list[dict]) -> dict[str, Any]:
    by_device: dict[str, dict] = {}
    for row in rows:
        device = row["preferred_device"] or "unknown"
        counts = by_device.setdefault(
            device,
            {
                "operation_count": 0,
                "known_cost_count": 0,
                "missing_cost_count": 0,
                "estimated_cost_weight_sum": 0.0,
            },
        )
        counts["operation_count"] += 1
        weight = row["estimated_cost_weight"]
        if weight is None:
            counts["missing_cost_count"] += 1
        else:
            counts["known_cost_count"] += 1
            counts["estimated_cost_weight_sum"] += weight
    return {
        "operation_count": len(rows),
        "unknown_device_count": sum(row["preferred_device"] is None for row in rows),
        "missing_cost_count": sum(row["estimated_cost_weight"] is None for row in rows),
        "invalid_cost_count": sum(row["cost_status"] == "invalid" for row in rows),
        "contains_nested_blocks": any(row["nested_block_depth"] > 0 for row in rows),
        "by_preferred_device": by_device,
    }


def _inspect_plan(plan: Any) -> dict[str, Any]:
    rows, unsupported = _collect_structure(plan)
    scopes: dict[str, list[dict]] = {}
    for row in rows:
        scopes.setdefault(row["scope"], []).append(row)
    return {
        "evidence_kind": "anticipated_compute_plan",
        "actual_execution_verified": False,
        "interpretation": (
            "Preferred and supported devices describe anticipated placement, not actual execution. "
            "Cost sums are raw compiler estimates, not timing, energy, or utilization. "
            "Do not add normalized weights across functions/models or assume all nested branches "
            "execute. Missing/invalid costs remain unknown; no ANE percentage is inferred."
        ),
        "operations": rows,
        "summary": _summarize(rows),
        "scope_summaries": {scope: _summarize(items) for scope, items in scopes.items()},
        "unsupported_structure_paths": unsupported,
    }


def inspect_compute_plan(
    model_path: str | Path, compute_units: str = "cpu_and_ne"
) -> dict[str, Any]:
    """Inspect a Core ML model on this Mac; compilation may write Core ML caches.

    Accepts .mlpackage, .mlmodel, or .mlmodelc. Errors propagate to callers, so a
    missing model, unsupported host, or denied cache access cannot look like a
    successful placement check. No prediction is executed.
    """
    compute_units = compute_units.lower()
    if compute_units not in COMPUTE_UNITS:
        raise ValueError(f"compute_units must be one of {COMPUTE_UNITS}")
    path = Path(model_path).expanduser().resolve(strict=True)
    if path.suffix not in (".mlpackage", ".mlmodel", ".mlmodelc"):
        raise ValueError("Expected a .mlpackage, .mlmodel, or .mlmodelc path")

    import coremltools as ct
    from coremltools.models.compute_plan import MLComputePlan

    units = getattr(ct.ComputeUnit, compute_units.upper())
    # Keep the owning MLModel alive until collection finishes: its compiled path
    # can be temporary and is removed when the Python model is released.
    model = None
    compiled_path = str(path)
    if path.suffix != ".mlmodelc":
        model = ct.models.MLModel(str(path), compute_units=units)
        compiled_path = model.get_compiled_model_path()
    plan = MLComputePlan.load_from_path(path=compiled_path, compute_units=units)
    result = _inspect_plan(plan)
    result.update(
        {
            "schema_version": 1,
            "model_name": path.name,
            "compute_units": compute_units,
            "environment": inspect_environment(),
        }
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", type=Path)
    parser.add_argument("--compute-units", choices=COMPUTE_UNITS, default="cpu_and_ne")
    parser.add_argument("--environment", action="store_true", help="Inspect environment only")
    args = parser.parse_args(argv)
    if not args.environment and args.model is None:
        parser.error("provide a model path or --environment")
    try:
        result = (
            inspect_environment()
            if args.environment
            else inspect_compute_plan(args.model, args.compute_units)
        )
    except Exception as error:  # noqa: BLE001 -- Core ML also raises bare Exception at this CLI boundary.
        # Native errors may contain local cache paths. Keep public diagnostic
        # output free of host/user identifiers; Python callers retain the error.
        print(
            json.dumps(
                {
                    "status": "unavailable",
                    "error_type": type(error).__name__,
                    "actual_execution_verified": False,
                }
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
