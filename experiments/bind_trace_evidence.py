"""Bind an isolated ANE trace to the bundle, placement reports and workload it measured.

Produces the schema-1 attribution sidecar the validation workflow expects: hashes
of the bundle manifest, per-graph placement reports, the trace file tree, the
workload output and the recorded commands, plus per-model ANE prediction rows
recomputed from the exported hardware interval table. Nothing is inferred about
FLOPs, energy or CPU utilization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET

LABEL = re.compile(r"^(?P<model>[A-Za-z0-9_-]+?)_main__Op(?P<op>\d+)_AneInference\s+Prediction$")
# Runtime counters the call-count closure needs; workloads recorded before the
# runtime reported them (round 2 and earlier) bind without closure.
CALL_COUNTERS = (
    "computed_feature_frames",
    "frontend_calls",
    "encoder_calls",
    "head_calls",
    "prefill_calls",
    "generation_decoder_calls",
)


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def canonical_sha256(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def tree_manifest(root: Path) -> list[dict]:
    return [
        {
            "path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]


def prediction_rows(xml_path: Path) -> tuple[dict[str, dict], int, list[str]]:
    root = ET.parse(xml_path).getroot()
    ids = {node.get("id"): node for node in root.iter() if node.get("id")}

    def resolve(node):
        while node.get("ref"):
            node = ids[node.get("ref")]
        return node

    per_model: dict[str, dict] = defaultdict(
        lambda: {
            "prediction_count": 0,
            "duration_sum_ns": 0,
            "first_start_ns": None,
            "last_end_ns": 0,
        }
    )
    unmatched = []
    total_rows = 0
    for row in root.findall(".//row"):
        cells = {cell.tag: resolve(cell) for cell in row}
        total_rows += 1
        label = cells["formatted-label"].get("fmt", "")
        if not label.endswith("Prediction"):
            continue
        match = LABEL.match(label)
        if match is None:
            unmatched.append(label)
            continue
        start = int(cells["start-time"].text)
        duration = int(cells["duration"].text)
        entry = per_model[match.group("model")]
        entry["prediction_count"] += 1
        entry["duration_sum_ns"] += duration
        entry["first_start_ns"] = (
            start if entry["first_start_ns"] is None else min(entry["first_start_ns"], start)
        )
        entry["last_end_ns"] = max(entry["last_end_ns"], start + duration)
        entry["compiled_label"] = label
    return dict(per_model), total_rows, unmatched


def expected_prediction_counts(manifest: dict, workload: list[dict]) -> dict[str, int] | None:
    """Close serial offline model calls against recorded runtime counters.

    Returns None when the workload rows carry no call counters, so retained
    workloads still bind; the sidecar then records that closure was skipped.
    """
    counts = Counter()
    files = manifest["files"]
    batch = manifest.get("frontend", {}).get("offline_batch_size", 1)
    rows = [row for row in workload if row.get("phase") in ("measured", "warmup")]
    for row in rows:
        if row.get("error"):
            raise ValueError("Cannot bind a failed workload as execution evidence")
    if not all(key in row.get("backend_timings", {}) for row in rows for key in CALL_COUNTERS):
        return None
    for row in rows:
        timing = row["backend_timings"]
        chunks = (int(timing["computed_feature_frames"]) + 99) // 100
        batched = chunks // batch if batch > 1 else 0
        single = chunks % batch if batch > 1 else chunks
        if single + batched != timing["frontend_calls"]:
            raise ValueError("Workload is not the declared serial offline frontend path")
        counts[Path(files["frontend"]).stem] += single
        if batch > 1:
            counts[Path(files["frontend_batched"]).stem] += batched
        counts[Path(files["encoder"]).stem] += timing["encoder_calls"]
        counts[Path(files["lm_head"]).stem] += timing["head_calls"]
        for partition in manifest["decoder_partitions"]:
            counts[Path(partition).stem] += (
                timing["prefill_calls"] + timing["generation_decoder_calls"]
            )
    return dict(counts)


def verify_bundle_identity(bundle: Path, placement_dir: Path, prefix: Path) -> dict[str, str]:
    """Refuse to bind artifacts that were recorded against a different bundle.

    A matching call count does not prove model identity: two bundles with the
    same graph names but different weights close identically. Every input that
    recorded the bundle it saw must name this manifest, and payload hashes are
    re-checked when the placement inspection recorded them.
    """
    manifest_sha256 = sha256(bundle / "manifest.json")
    checks = {}
    summary_path = placement_dir / "summary.json"
    if not summary_path.is_file():
        raise RuntimeError("Placement directory has no summary.json naming its bundle")
    summary = json.loads(summary_path.read_text())
    recorded = summary.get("manifest_sha256", summary.get("bundle_manifest_sha256"))
    if recorded != manifest_sha256:
        raise RuntimeError("Placement reports were inspected for a different bundle manifest")
    checks["placement_manifest"] = "verified"
    if summary.get("models") is not None:
        # The inspector rewrites its summary after each model, so an interrupted
        # inspection lists a subset while stale plan files remain readable.
        manifest = json.loads((bundle / "manifest.json").read_text())
        graphs = {
            Path(name).stem
            for name in [
                *manifest.get("files", {}).values(),
                *manifest.get("decoder_partitions", []),
            ]
            if Path(name).suffix in (".mlpackage", ".mlmodelc")
        }
        listed = {Path(entry["model"]).stem for entry in summary["models"]}
        if not graphs <= listed:
            raise RuntimeError(
                f"Placement summary does not list every bundle graph: {sorted(graphs - listed)}"
            )
    payloads = 0
    for entry in summary.get("models") or []:
        for item in entry.get("files") or []:
            path = bundle / item["path"]
            if not path.is_file() or sha256(path) != item["sha256"]:
                raise RuntimeError(
                    f"Bundle payload differs from the inspected placement: {item['path']}"
                )
            payloads += 1
    checks["placement_payloads"] = f"{payloads} files verified" if payloads else "not recorded"
    workload_summary_path = Path(f"{prefix}-workload.jsonl.summary.json")
    if not workload_summary_path.is_file():
        raise RuntimeError("Workload has no summary recording the bundle it loaded")
    loaded = (
        json.loads(workload_summary_path.read_text())
        .get("model_metadata", {})
        .get("metadata_files", {})
        .get("manifest.json", {})
        .get("sha256")
    )
    if loaded != manifest_sha256:
        raise RuntimeError("Workload was recorded against a different bundle manifest")
    checks["workload_manifest"] = "verified"
    commands = json.loads(Path(f"{prefix}-commands.json").read_text())
    named = {
        command["argv"][index + 1]
        for command in commands
        for index, argument in enumerate(command["argv"][:-1])
        if argument == "--model-dir"
    }
    if not named:
        checks["commands_model_dir"] = "not recorded"
    elif all((Path(name) / "manifest.json").is_file() for name in named):
        if any(sha256(Path(name) / "manifest.json") != manifest_sha256 for name in named):
            raise RuntimeError("Trace commands named a different bundle")
        checks["commands_model_dir"] = "verified"
    else:
        # A bundle moved after tracing cannot be re-read; the workload summary
        # above already proved which manifest the traced process loaded.
        checks["commands_model_dir"] = "path no longer present; workload manifest verified"
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prefix", type=Path, required=True, help="trace prefix used by trace_ane.py"
    )
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--placement-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    prefix = args.prefix
    # Verify identity before writing anything, so a mismatch leaves no sidecar.
    identity = verify_bundle_identity(args.bundle, args.placement_dir, prefix)
    manifest = json.loads((args.bundle / "manifest.json").read_text())
    graphs = sorted(set(manifest["files"].values()) | set(manifest["decoder_partitions"]))
    graph_names = {
        Path(graph).stem for graph in graphs if Path(graph).suffix in (".mlpackage", ".mlmodelc")
    }
    placement = {}
    for name in sorted(graph_names):
        report_path = args.placement_dir / f"{name}.json"
        report = json.loads(report_path.read_text())
        summary = report["summary"]
        placement[name] = {
            "report_sha256": sha256(report_path),
            "operation_count": summary["operation_count"],
            "by_preferred_device": {
                device: counts["operation_count"]
                for device, counts in summary["by_preferred_device"].items()
            },
            "known_cost_not_ane": sum(
                counts["known_cost_count"]
                for device, counts in summary["by_preferred_device"].items()
                if device != "ane"
            ),
            "unknown_operator_types": sorted(
                {row["operator"] for row in report["operations"] if row["preferred_device"] is None}
            ),
        }
    placement_summary = {
        "schema_version": 1,
        "bundle_manifest_sha256": sha256(args.bundle / "manifest.json"),
        "graphs": placement,
        "all_known_cost_operations_prefer_ane": all(
            item["known_cost_not_ane"] == 0 for item in placement.values()
        ),
    }
    # Keep the original placement evidence intact; this sidecar belongs to this trace.
    summary_path = args.output.with_suffix(".placement.json")
    if summary_path.exists():
        raise FileExistsError(summary_path)
    summary_path.write_text(json.dumps(placement_summary, indent=2) + "\n")
    per_model, total_rows, unmatched = prediction_rows(Path(f"{prefix}-ane-hw-intervals.xml"))
    workload_path = Path(f"{prefix}-workload.jsonl")
    workload = [json.loads(line) for line in workload_path.read_text().splitlines() if line.strip()]
    expected = expected_prediction_counts(manifest, workload)
    if expected is not None:
        # Every graph gets a count; a graph the workload never called expects 0.
        expected = {name: expected.get(name, 0) for name in sorted(graph_names)}
    candidate_models = []
    normalized = {name: name.replace("-", "_") for name in graph_names}
    if len(set(normalized.values())) != len(normalized):
        raise ValueError("Ambiguous compiled model label names")
    for name in sorted(graph_names):
        entry = per_model.pop(name, None)
        if entry is None and normalized[name] != name:
            entry = per_model.pop(normalized[name], None)
        if entry is None and expected is not None and expected[name] == 0:
            # No rows is correct for a graph the workload never called, such as
            # the B1 frontend when every clip fills complete batches.
            entry = {
                "prediction_count": 0,
                "duration_sum_ns": 0,
                "first_start_ns": None,
                "last_end_ns": 0,
            }
        if entry is None:
            raise RuntimeError(f"No ANE prediction rows for graph {name}")
        candidate_models.append({"model": name, **entry})
    observed = {entry["model"]: entry["prediction_count"] for entry in candidate_models}
    if expected is not None and expected != observed:
        raise RuntimeError(
            f"ANE prediction call-count closure failed: expected={expected}, observed={observed}"
        )
    wall = sum(row["seconds"] for row in workload if row.get("phase") == "measured")
    prediction_total = sum(item["duration_sum_ns"] for item in candidate_models)
    # ANE intervals carry no phase, so a workload with warmup rows cannot be
    # split into a measured-only share.
    if not wall:
        share, share_reason = None, "the workload recorded no measured wall time"
    elif any(row.get("phase") == "warmup" for row in workload):
        share, share_reason = None, "warmup rows share the trace with measured rows"
    else:
        share, share_reason = prediction_total / 1e9 / wall, None
    limitations = [
        "ANE hardware rows carry no PID; attribution uses exact compiled labels"
        + (" and the call-count closure." if expected is not None else "."),
        "Shares are Instruments ANE-active intervals over transcribe() wall time, not FLOPs, energy or CPU utilization.",
        f"Recorded batch workload only; bundle cache has {manifest['max_sequence_length']} positions. Do not generalize to streaming or unrecorded shapes.",
    ]
    if expected is None:
        limitations.append(
            "Workload rows carry no runtime call counters; per-model call-count closure was not verified."
        )
    if share_reason is not None:
        limitations.append(f"ANE-active share not reported: {share_reason}.")
    trace_files = tree_manifest(Path(f"{prefix}.trace"))
    # The target process is recorded once in the trace table of contents as
    # <process pid=... return-exit-status=...>; time-info carries no PID.
    toc = ET.parse(f"{prefix}-toc.xml").getroot()
    target_node = toc.find("./run/info/target/process")
    target = dict(target_node.attrib) if target_node is not None else {}
    sidecar = {
        "schema_version": 1,
        "bundle_name": args.bundle.name,
        "bundle_manifest_sha256": placement_summary["bundle_manifest_sha256"],
        "bundle_identity_checks": identity,
        "placement_summary_path": str(summary_path),
        "placement_summary_sha256": sha256(summary_path),
        "trace_prefix": str(prefix),
        "trace_tree_sha256": canonical_sha256(trace_files),
        "trace_file_count": len(trace_files),
        "ane_table_sha256": sha256(Path(f"{prefix}-ane-hw-intervals.xml")),
        "workload_output_sha256": sha256(workload_path),
        "commands_sha256": sha256(Path(f"{prefix}-commands.json")),
        "target_pid": int(target["pid"]) if target.get("pid") else None,
        "target_exit_status": target.get("return-exit-status"),
        "attribution_method": "isolated_workload_exact_model_labels"
        + ("_and_call_count_closure" if expected is not None else ""),
        "candidate_models": candidate_models,
        "expected_prediction_counts": expected,
        "call_count_closure_verified": expected is not None,
        "candidate_prediction_count": sum(item["prediction_count"] for item in candidate_models),
        "candidate_prediction_duration_sum_ns": prediction_total,
        "ane_hardware_rows_total": total_rows,
        "unattributed_or_background_prediction_count": len(unmatched)
        + sum(item["prediction_count"] for item in per_model.values()),
        "background_prediction_models": sorted(per_model),
        "workload_measured_wall_seconds": wall,
        "ane_active_share_of_measured_wall": share,
        "ane_active_share_reason": share_reason,
        "limitations": limitations,
    }
    sidecar["run_fingerprint_sha256"] = canonical_sha256(
        {
            key: sidecar[key]
            for key in (
                "bundle_manifest_sha256",
                "placement_summary_sha256",
                "trace_tree_sha256",
                "workload_output_sha256",
                "commands_sha256",
            )
        }
    )
    args.output.write_text(json.dumps(sidecar, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: value
                for key, value in sidecar.items()
                if key not in ("candidate_models", "limitations")
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
