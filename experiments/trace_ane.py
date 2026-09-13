"""Record/export ANE hardware evidence with public Instruments command-line tools.

Examples (run with normal Instruments/Core ML cache access):
  python experiments/trace_ane.py export --prefix artifacts/telemetry/full-asr
  python experiments/trace_ane.py record --prefix artifacts/telemetry/new-run \
    --seconds 20 -- /absolute/python /absolute/workload.py

Raw trace/XML files contain Instruments host and process metadata. The summary
contains only target PID, model labels, counts, and trace-relative durations.
"""

import argparse
import json
import os
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET

SCHEMAS = ("ane-hw-intervals", "coreml-os-signpost", "metal-gpu-intervals", "time-info")


def read_table(path):
    root = ET.parse(path).getroot()
    ids = {node.get("id"): node for node in root.iter() if node.get("id")}

    def resolve(node):
        while node.get("ref"):
            node = ids[node.get("ref")]
        return node

    tables = []
    for node in root.findall(".//node"):
        schema = node.find("schema")
        if schema is None:
            continue
        columns = [column.findtext("mnemonic") for column in schema.findall("col")]
        rows = [
            dict(zip(columns, (resolve(value) for value in row), strict=True))
            for row in node.findall("row")
        ]
        tables.extend(rows)
    return tables, resolve


def union_duration(intervals):
    total = 0
    end = 0
    for start, stop in sorted(intervals):
        total += max(0, stop - max(start, end))
        end = max(end, stop)
    return total


def summarize(prefix):
    toc = ET.parse(f"{prefix}-toc.xml").getroot()
    target = toc.find("./run/info/target/process")
    target_pid = target.get("pid") if target is not None else None
    ane_rows, _ = read_table(f"{prefix}-ane-hw-intervals.xml")
    gpu_rows, resolve_gpu = read_table(f"{prefix}-metal-gpu-intervals.xml")
    coreml_rows, _ = read_table(f"{prefix}-coreml-os-signpost.xml")
    counts = Counter()
    grouped = defaultdict(list)
    for row in ane_rows:
        label = row["event-label"].get("fmt", "")
        counts[label] += 1
        start = int(row["start"].text)
        duration = int(row["duration"].text)
        grouped[label].append((start, start + duration))
    target_gpu = []
    unknown_gpu_process = 0
    for row in gpu_rows:
        pid_node = row["process"].find("pid")
        if pid_node is None:
            unknown_gpu_process += 1
        elif resolve_gpu(pid_node).text == target_pid:
            target_gpu.append(row)
    labels = [
        {
            "label": label,
            "count": count,
            "duration_sum_ns": sum(end - start for start, end in grouped[label]),
            "interval_union_ns": union_duration(grouped[label]),
            "first_start_ns": min(start for start, _ in grouped[label]),
            "last_end_ns": max(end for _, end in grouped[label]),
            "is_prediction": label.endswith(" Prediction"),
        }
        for label, count in counts.most_common()
    ]
    report = {
        "evidence_kind": "instruments_ane_hardware_intervals",
        "target_pid": int(target_pid) if target_pid else None,
        "target_exit_status": target.get("return-exit-status")
        if target is not None
        else None,
        "trace_duration_seconds": float(toc.findtext("./run/info/summary/duration")),
        "instruments_version": toc.findtext("./run/info/summary/instruments-version"),
        "ane_hardware_rows": len(ane_rows),
        "ane_prediction_rows": sum(
            item["count"] for item in labels if item["is_prediction"]
        ),
        "ane_labels": labels,
        "ane_pid_attribution_available": False,
        "coreml_summary_rows": len(coreml_rows),
        "gpu_hardware_rows_all_processes": len(gpu_rows),
        "gpu_hardware_rows_target_pid": len(target_gpu),
        "gpu_rows_unknown_process": unknown_gpu_process,
        "limitations": [
            "ANE hardware schema has no PID: attribute by compiled model label and workload timing.",
            "Other processes may use ANE/GPU; background labels and events remain in raw evidence.",
            "Hardware durations are not operation FLOPs shares, energy, or unprofiled latency.",
            "Zero Core ML summary rows means unavailable signposts, not absence of Core ML execution.",
            "Zero target GPU rows is scoped to this trace and instrument coverage.",
        ],
    }
    Path(f"{prefix}-summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("record", "export", "summarize"))
    parser.add_argument("--prefix", required=True, type=Path)
    parser.add_argument(
        "--developer-dir", default="/Applications/Xcode-beta.app/Contents/Developer"
    )
    parser.add_argument("--seconds", type=int, default=20)
    args, command = parser.parse_known_args()
    if command and command[0] == "--":
        command = command[1:]
    if not 1 <= args.seconds <= 90:
        parser.error("--seconds must be between 1 and 90")
    prefix = args.prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, DEVELOPER_DIR=args.developer_dir)
    commands = []

    def run(arguments, log_suffix):
        invocation = ["xcrun", "xctrace", *arguments]
        with Path(f"{prefix}-{log_suffix}.log").open("w") as output:
            completed = subprocess.run(
                invocation,
                env=env,
                stdout=output,
                stderr=subprocess.STDOUT,
                timeout=180,
                check=False,
            )
        commands.append({"argv": invocation, "exit_code": completed.returncode})
        Path(f"{prefix}-commands.json").write_text(
            json.dumps(commands, indent=2) + "\n"
        )
        return completed.returncode

    if args.action == "record":
        if not command:
            parser.error("record requires a workload command after --")
        if Path(f"{prefix}.trace").exists():
            parser.error("trace already exists; choose a new prefix")
        run(["list", "templates"], "templates")
        run(["list", "instruments"], "instruments")
        run(
            [
                "record",
                "--template",
                "Core ML",
                "--instrument",
                "Neural Engine",
                "--instrument",
                "GPU",
                "--time-limit",
                f"{args.seconds}s",
                "--no-prompt",
                "--output",
                f"{prefix}.trace",
                "--target-stdout",
                f"{prefix}-stdout.log",
                "--launch",
                "--",
                *command,
            ],
            "record",
        )
    if args.action in ("record", "export"):
        if run(
            [
                "export",
                "--input",
                f"{prefix}.trace",
                "--toc",
                "--output",
                f"{prefix}-toc.xml",
            ],
            "export-toc",
        ):
            raise RuntimeError("Trace TOC export failed; inspect the saved log")
        toc = ET.parse(f"{prefix}-toc.xml").getroot()
        available = {table.get("schema") for table in toc.findall(".//table")}
        for schema in SCHEMAS:
            # Xcode 26 names the same Neural Engine interval table (identical
            # columns) with an "-internal" suffix; export it under the stable name.
            source_schema = schema
            if schema not in available and f"{schema}-internal" in available:
                source_schema = f"{schema}-internal"
            if source_schema not in available:
                raise RuntimeError(f"Required table unavailable: {schema}")
            query = f'/trace-toc/run[@number="1"]/data/table[@schema="{source_schema}"]'
            if run(
                [
                    "export",
                    "--input",
                    f"{prefix}.trace",
                    "--xpath",
                    query,
                    "--output",
                    f"{prefix}-{schema}.xml",
                ],
                f"export-{schema}",
            ):
                raise RuntimeError(f"Table export failed: {schema}")
        query = (
            '/trace-toc/run[@number="1"]/data/table[@schema="os-signpost" and '
            '(@category="coreml" or @category="DynamicTracing")]'
        )
        run(
            [
                "export",
                "--input",
                f"{prefix}.trace",
                "--xpath",
                query,
                "--output",
                f"{prefix}-coreml-raw-signpost.xml",
            ],
            "export-coreml-raw-signpost",
        )
    report = summarize(prefix)
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "ane_labels"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
