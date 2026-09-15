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
from collections import defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET

LABEL = re.compile(
    r"^(?P<model>[A-Za-z0-9_]+?)_main__Op(?P<op>\d+)_AneInference\s+Prediction$"
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
            start
            if entry["first_start_ns"] is None
            else min(entry["first_start_ns"], start)
        )
        entry["last_end_ns"] = max(entry["last_end_ns"], start + duration)
        entry["compiled_label"] = label
    return dict(per_model), total_rows, unmatched


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
    manifest = json.loads((args.bundle / "manifest.json").read_text())
    graphs = sorted(
        set(manifest["files"].values()) | set(manifest["decoder_partitions"])
    )
    graph_names = {
        Path(graph).stem
        for graph in graphs
        if Path(graph).suffix in (".mlpackage", ".mlmodelc")
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
                {
                    row["operator"]
                    for row in report["operations"]
                    if row["preferred_device"] is None
                }
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
    summary_path = args.placement_dir / "summary.json"
    summary_path.write_text(json.dumps(placement_summary, indent=2) + "\n")
    per_model, total_rows, unmatched = prediction_rows(
        Path(f"{prefix}-ane-hw-intervals.xml")
    )
    candidate_models = []
    for name in sorted(graph_names):
        entry = per_model.pop(name, None)
        if entry is None:
            raise RuntimeError(f"No ANE prediction rows for graph {name}")
        candidate_models.append({"model": name, **entry})
    workload_path = Path(f"{prefix}-workload.jsonl")
    workload = [
        json.loads(line)
        for line in workload_path.read_text().splitlines()
        if line.strip()
    ]
    wall = sum(row["seconds"] for row in workload if row.get("phase") == "measured")
    trace_files = tree_manifest(Path(f"{prefix}.trace"))
    # The target process is recorded once in the trace table of contents as
    # <process pid=... return-exit-status=...>; time-info carries no PID.
    toc = ET.parse(f"{prefix}-toc.xml").getroot()
    target_node = toc.find("./run/info/target/process")
    target = dict(target_node.attrib) if target_node is not None else {}
    prediction_total = sum(item["duration_sum_ns"] for item in candidate_models)
    sidecar = {
        "schema_version": 1,
        "bundle_name": args.bundle.name,
        "bundle_manifest_sha256": placement_summary["bundle_manifest_sha256"],
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
        "attribution_method": "isolated_workload_exact_model_labels_and_call_count_closure",
        "candidate_models": candidate_models,
        "candidate_prediction_count": sum(
            item["prediction_count"] for item in candidate_models
        ),
        "candidate_prediction_duration_sum_ns": prediction_total,
        "ane_hardware_rows_total": total_rows,
        "unattributed_or_background_prediction_count": len(unmatched)
        + sum(item["prediction_count"] for item in per_model.values()),
        "background_prediction_models": sorted(per_model),
        "workload_measured_wall_seconds": wall,
        "ane_active_share_of_measured_wall": prediction_total / 1e9 / wall
        if wall
        else None,
        "limitations": [
            "ANE hardware rows carry no PID; attribution uses exact compiled labels and the call-count closure.",
            "Shares are Instruments ANE-active intervals over transcribe() wall time, not FLOPs, energy or CPU utilization.",
            f"Recorded batch workload only; bundle cache has {manifest['max_sequence_length']} positions. Do not generalize to streaming or unrecorded shapes.",
        ],
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
