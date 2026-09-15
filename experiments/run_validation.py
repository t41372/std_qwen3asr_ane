"""Serial, resumable evidence gates. Never turns an omitted measurement into a pass.

--manifest is the evaluation JSONL; the model manifest is MODEL_DIR/manifest.json.
A timeout terminates the entire subprocess group. Every attempt retains separate
stdout/stderr and reports. Resume requires identical inputs, code, dependencies,
model payloads, manifests, and intact previously recorded evidence files.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import signal
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parents[1]
STEPS = ("tests", "quality", "placement", "trace")
GATES = {
    "absolute_margin": 0.003,
    "relative_margin": 0.05,
    "relative_baseline_minimum": 0.01,
    "release_samples": 1000,
    "release_audio_seconds": 36000,
    "minimum_language_samples": 30,
    "anticipated_ane_operation_ratio": 0.9,
}


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def model_phases(model_dir):
    manifest = read_json(model_dir / "manifest.json")
    phases = {}
    for role, relative in manifest["files"].items():
        path = (model_dir / relative).resolve()
        if not path.is_relative_to(model_dir.resolve()):
            raise ValueError(f"Bundle path escapes model directory: {role}")
        if path.suffix == ".mlpackage":
            phases[role] = path
    if not phases:
        raise ValueError("Bundle contains no .mlpackage phases")
    return phases


def fingerprint(args):
    """Hash payloads before timed evaluation, including referenced audio contents."""
    from evaluate import manifest_rows

    paths = {args.manifest, args.model_dir / "manifest.json"}
    for relative in read_json(args.model_dir / "manifest.json")["files"].values():
        target = (args.model_dir / relative).resolve()
        if not target.is_relative_to(args.model_dir):
            raise ValueError("Model payload escapes bundle directory")
        if target.is_dir():
            paths.update(target.rglob("*"))
        else:
            paths.add(target)
    paths.update(Path(row["audio_path"]) for row in manifest_rows(args.manifest))
    paths.update(args.source_dir.glob("*.json"))
    paths.update(args.source_dir.glob("*.safetensors"))
    paths.update(args.source_dir.glob("*.txt"))
    paths.update((ROOT / "experiments").glob("*.py"))
    paths.update((ROOT / "std_qwen3asr_ane/src").rglob("*.py"))
    paths.update((ROOT / "std_qwen3asr_ane/tests").glob("*.py"))
    paths.add(ROOT / "uv.lock")
    if args.trace_summary:
        paths.add(args.trace_summary)
        provenance = args.trace_summary.with_suffix(".provenance.json")
        if provenance.is_file():
            paths.add(provenance)
    if args.trace_attribution:
        paths.add(args.trace_attribution)
        metadata = read_json(args.trace_attribution)
        for key in ("trace_summary_path", "placement_summary_path"):
            if isinstance(metadata.get(key), str):
                linked = (ROOT / metadata[key]).resolve()
                if linked.is_file():
                    paths.add(linked)
    hashes = {str(path): file_hash(path) for path in sorted(paths) if not path.is_dir()}
    packages = sorted(
        [item.metadata["Name"], item.version]
        for item in importlib.metadata.distributions()
    )
    settings = {
        "python": sys.version,
        "packages": packages,
        "gates": GATES,
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "timeout_seconds": args.timeout_seconds,
    }
    return {
        "input_hash": content_hash({"files": hashes, "settings": settings}),
        "model_manifest_sha256": file_hash(args.model_dir / "manifest.json"),
        "files": hashes,
        "settings": settings,
    }


def run_command(argv, directory, name, timeout):
    stdout, stderr = directory / f"{name}.stdout.log", directory / f"{name}.stderr.log"
    result = {
        "argv": list(map(str, argv)),
        "stdout": str(stdout),
        "stderr": str(stderr),
    }
    start = time.monotonic()
    with stdout.open("wb") as out, stderr.open("wb") as err:
        try:
            process = subprocess.Popen(
                argv, cwd=ROOT, stdout=out, stderr=err, start_new_session=True
            )
            try:
                result["returncode"] = process.wait(timeout=timeout)
                result["status"] = "pass" if process.returncode == 0 else "fail"
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                result.update(
                    status="fail",
                    reason="subprocess_timeout",
                    returncode=process.returncode,
                )
        except OSError as error:
            result.update(
                status="unavailable", reason=f"{type(error).__name__}: {error}"
            )
    result["seconds"] = time.monotonic() - start
    return result


def protocol_result(junit):
    cases = ElementTree.parse(junit).getroot().findall(".//testcase")
    protocol = [case for case in cases if "test_plugin" in case.get("classname", "")]
    failed = [
        case
        for case in cases
        if case.find("failure") is not None or case.find("error") is not None
    ]
    skipped_protocol = [case for case in protocol if case.find("skipped") is not None]
    status = (
        "fail"
        if failed
        else "inconclusive"
        if not protocol or skipped_protocol
        else "pass"
    )
    return {
        "status": status,
        "executed_test_cases": len(cases),
        "protocol_cases": len(protocol),
        "skipped_protocol_cases": len(skipped_protocol),
        "scope": "Standard ASR adapter contract tests; fake runtimes do not establish model quality",
    }


def quality_gate(comparison, baseline_rows, candidate_rows, expected_ids):
    from evaluate import character_language, score

    if not comparison.get("valid_comparison"):
        return {
            "status": "fail",
            "functional_status": "fail",
            "reason": "invalid_paired_comparison",
        }
    baseline = [
        row
        for row in baseline_rows
        if row.get("phase") == "measured" and row.get("repeat") == 0
    ]
    candidate = [
        row
        for row in candidate_rows
        if row.get("phase") == "measured" and row.get("repeat") == 0
    ]
    if {row["id"] for row in baseline} != set(expected_ids) or {
        row["id"] for row in candidate
    } != set(expected_ids):
        return {
            "status": "fail",
            "functional_status": "fail",
            "reason": "incomplete_manifest_coverage",
        }
    grouped = defaultdict(list)
    for row in baseline:
        grouped[row.get("language") or "unspecified"].append(row)
    duration = sum(row.get("audio_seconds") or 0 for row in baseline)
    coverage = (
        len(baseline) >= GATES["release_samples"]
        and duration >= GATES["release_audio_seconds"]
    )
    languages = {}
    for language, rows in sorted(grouped.items()):
        metric = "cer" if character_language(language) else "wer"
        counts = [score(row["reference"], row["hypothesis"])[metric] for row in rows]
        units = sum(row["reference_units"] for row in counts)
        base_rate = sum(row["errors"] for row in counts) / units if units else None
        interval = (
            comparison.get("by_language", {})
            .get(language, {})
            .get(metric, {})
            .get("ci95")
        )
        margin = GATES["absolute_margin"]
        if base_rate is not None and base_rate >= GATES["relative_baseline_minimum"]:
            margin = min(margin, base_rate * GATES["relative_margin"])
        eligible = (
            coverage
            and len(rows) >= GATES["minimum_language_samples"]
            and language != "unspecified"
        )
        if not interval or base_rate is None:
            status = "inconclusive"
        elif interval[0] > margin:
            status = "fail"
        elif interval[1] <= margin and eligible:
            status = "pass"
        else:
            status = "inconclusive"
        languages[language] = {
            "status": status,
            "metric": metric,
            "baseline_rate": base_rate,
            "ci95_delta": interval,
            "noninferiority_margin": margin,
            "samples": len(rows),
            "release_coverage_eligible": eligible,
        }
    statuses = [row["status"] for row in languages.values()]
    status = (
        "fail"
        if "fail" in statuses
        else "pass"
        if statuses and set(statuses) == {"pass"}
        else "inconclusive"
    )
    return {
        "status": status,
        "functional_status": "pass",
        "by_language": languages,
        "samples": len(baseline),
        "audio_seconds": duration,
        "release_coverage_eligible": coverage,
        "interpretation": "Per-language utterance bootstrap; positive delta is regression. No pooled quality gate.",
        "baseline_attention": "official FP32 eager; multi-window encoder differs from intended FA2 window semantics",
    }


def placement_gate(report):
    if (
        report.get("schema_version") != 1
        or report.get("evidence_kind") != "anticipated_compute_plan"
    ):
        return {"status": "inconclusive", "reason": "unrecognized_diagnostics_schema"}
    operations = report.get("operations")
    if not isinstance(operations, list) or not all(
        isinstance(row, dict) for row in operations
    ):
        return {"status": "inconclusive", "reason": "missing_operations"}
    active = [
        row for row in operations if row.get("operator", "").split(".")[-1] != "const"
    ]
    unknown = [
        row
        for row in active
        if row.get("preferred_device") not in ("ane", "cpu", "gpu")
    ]
    ane = sum(row.get("preferred_device") == "ane" for row in active)
    ratio = ane / len(active) if active else None
    if not active or unknown or report.get("unsupported_structure_paths"):
        status = "inconclusive"
    else:
        status = "pass" if ratio >= GATES["anticipated_ane_operation_ratio"] else "fail"
    return {
        "status": status,
        "nonconstant_operations": len(active),
        "unknown_nonconstant_operations": len(unknown),
        "preferred_ane_operations": ane,
        "preferred_ane_operation_ratio": ratio,
        "actual_execution_verified": False,
        "scope": "Anticipated operation-count ratio; not FLOPs, execution time, utilization, or energy",
    }


def trace_gate(report, phases, manifest_hash, provenance=None, summary_hash=None):
    required = {
        "evidence_kind",
        "ane_prediction_rows",
        "ane_labels",
        "target_exit_status",
        "target_pid",
        "ane_pid_attribution_available",
        "gpu_hardware_rows_target_pid",
    }
    if (
        not required.issubset(report)
        or report["evidence_kind"] != "instruments_ane_hardware_intervals"
    ):
        return {"status": "inconclusive", "reason": "unrecognized_trace_schema"}
    if type(report["target_pid"]) is not int or report["target_pid"] <= 0:
        return {"status": "inconclusive", "reason": "missing_valid_trace_target_pid"}
    labels = report["ane_labels"]
    if not isinstance(labels, list) or not all(
        isinstance(row, dict)
        and isinstance(row.get("label"), str)
        and isinstance(row.get("count"), int)
        and isinstance(row.get("is_prediction"), bool)
        for row in labels
    ):
        return {"status": "inconclusive", "reason": "invalid_trace_label_schema"}
    predictions = [row for row in labels if row["is_prediction"]]
    if (
        type(report["ane_prediction_rows"]) is not int
        or any(type(row["count"]) is not int or row["count"] < 0 for row in labels)
        or sum(row["count"] for row in predictions) != report["ane_prediction_rows"]
        or any(
            not isinstance(row.get("duration_sum_ns"), int)
            or row["duration_sum_ns"] <= 0
            for row in predictions
            if row["count"] > 0
        )
    ):
        return {
            "status": "inconclusive",
            "reason": "inconsistent_trace_interval_counts_or_durations",
        }
    coverage = {
        role: [
            row["label"]
            for row in labels
            if row["is_prediction"]
            and row["count"] > 0
            and row["label"].startswith(path.stem + "_")
        ]
        for role, path in phases.items()
    }
    metadata = provenance or report
    binding = (
        metadata.get("bundle_manifest_sha256", metadata.get("model_manifest_sha256"))
        == manifest_hash
    )
    reasons = []
    if provenance is not None:
        if provenance.get("target_pid") != report["target_pid"]:
            reasons.append("attribution_target_pid_mismatch")
        if provenance.get("schema_version") != 1:
            reasons.append("unrecognized_attribution_schema")
        if not summary_hash or provenance.get("trace_summary_sha256") != summary_hash:
            reasons.append("attribution_not_bound_to_trace_summary")
        candidates = provenance.get("candidate_models", [])
        coverage = {role: [] for role in phases}
        observed = {
            row["label"].removesuffix(" Prediction"): row["count"]
            for row in predictions
        }
        if not isinstance(candidates, list):
            candidates = []
            reasons.append("invalid_candidate_model_schema")
        for candidate in candidates:
            if (
                not isinstance(candidate, dict)
                or not isinstance(candidate.get("model"), str)
                or not isinstance(candidate.get("compiled_label"), str)
            ):
                reasons.append("invalid_candidate_model_schema")
                continue
            label = candidate["compiled_label"].removesuffix(" Prediction")
            if observed.get(label, 0) <= 0 or candidate.get(
                "prediction_count"
            ) != observed.get(label):
                reasons.append("candidate_count_does_not_match_trace")
                continue
            for role, path in phases.items():
                if Path(candidate["model"]).stem == path.stem:
                    coverage[role].append(label)
    if metadata.get("isolated_trace_verified") is not True:
        reasons.append("isolated_trace_not_verified")
    if metadata.get("background_workload_confirmed") is True:
        reasons.append("background_workload_confirmed")
    if not binding:
        reasons.append("trace_not_bound_to_current_model_manifest")
    if str(report["target_exit_status"]) != "0":
        reasons.append("target_did_not_exit_successfully")
    if not all(coverage.values()):
        reasons.append("missing_prediction_labels_for_model_phases")
    if (
        not isinstance(report["ane_prediction_rows"], int)
        or report["ane_prediction_rows"] <= 0
    ):
        reasons.append("no_ane_prediction_intervals")
    return {
        "status": "inconclusive" if reasons else "pass",
        "reasons": reasons,
        "phase_prediction_labels": coverage,
        "manifest_binding_verified": binding,
        "ane_pid_attribution_available": report["ane_pid_attribution_available"],
        "gpu_hardware_rows_target_pid": report["gpu_hardware_rows_target_pid"],
        "scope": "Observed ANE prediction activity attributed by model label; no workload-share or energy claim",
        "raw_evidence_limitations": report.get("limitations", []),
    }


def evidence_intact(step):
    return bool(step.get("evidence_hashes")) and all(
        Path(path).is_file() and file_hash(path) == digest
        for path, digest in step["evidence_hashes"].items()
    )


def execute_step(name, args, directory, identity, phases):
    commands = []
    if name == "tests":
        junit = directory / "pytest.xml"
        command = run_command(
            [
                sys.executable,
                "-m",
                "pytest",
                str(ROOT / "std_qwen3asr_ane/tests"),
                "-q",
                f"--junitxml={junit}",
            ],
            directory,
            "pytest",
            args.timeout_seconds,
        )
        commands.append(command)
        result = (
            protocol_result(junit)
            if junit.exists()
            else {"status": command["status"], "reason": "missing_junit"}
        )
        if command["status"] != "pass":
            result["status"] = command["status"]
    elif name == "quality":
        from evaluate import manifest_rows, read_jsonl

        outputs = {}
        for backend, model_dir in (
            ("official", args.source_dir),
            ("coreml", args.model_dir),
        ):
            outputs[backend] = directory / f"{backend}.jsonl"
            command = run_command(
                [
                    sys.executable,
                    str(ROOT / "experiments/evaluate.py"),
                    "--backend",
                    backend,
                    "--model-dir",
                    str(model_dir),
                    "--manifest",
                    str(args.manifest),
                    "--output",
                    str(outputs[backend]),
                    "--warmups",
                    "0",
                    "--repeats",
                    "1",
                    "--torch-threads",
                    "4",
                    "--seed",
                    str(args.seed),
                    "--max-new-tokens",
                    str(args.max_new_tokens),
                ],
                directory,
                backend,
                args.timeout_seconds,
            )
            commands.append(command)
        if any(command["status"] != "pass" for command in commands):
            result = {
                "status": "fail",
                "functional_status": "fail",
                "reason": "evaluation_subprocess_failed",
            }
        else:
            comparison = directory / "comparison.json"
            command = run_command(
                [
                    sys.executable,
                    str(ROOT / "experiments/evaluate.py"),
                    "--compare",
                    str(outputs["official"]),
                    str(outputs["coreml"]),
                    "--output",
                    str(comparison),
                    "--bootstrap-samples",
                    str(args.bootstrap_samples),
                    "--seed",
                    str(args.seed),
                ],
                directory,
                "compare",
                args.timeout_seconds,
            )
            commands.append(command)
            result = (
                quality_gate(
                    read_json(comparison),
                    read_jsonl(outputs["official"]),
                    read_jsonl(outputs["coreml"]),
                    [row["id"] for row in manifest_rows(args.manifest)],
                )
                if command["status"] == "pass"
                else {
                    "status": "fail",
                    "functional_status": "fail",
                    "reason": "paired_comparison_failed",
                }
            )
    elif name == "placement":
        results = {}
        for index, (role, path) in enumerate(phases.items()):
            command = run_command(
                [
                    sys.executable,
                    "-m",
                    "std_qwen3asr_ane.diagnostics",
                    str(path),
                    "--compute-units",
                    "cpu_and_ne",
                ],
                directory,
                f"phase-{index:02d}",
                args.timeout_seconds,
            )
            commands.append(command)
            try:
                results[role] = (
                    placement_gate(read_json(command["stdout"]))
                    if command["status"] == "pass"
                    else {
                        "status": "unavailable",
                        "reason": "compute_plan_subprocess_failed",
                    }
                )
            except (OSError, ValueError, TypeError, KeyError) as error:
                results[role] = {
                    "status": "inconclusive",
                    "reason": f"Invalid phase report: {error}",
                }
            results[role]["model"] = str(path)
            results[role]["diagnostics_stdout"] = command["stdout"]
        statuses = {row["status"] for row in results.values()}
        result = {
            "status": "pass"
            if statuses == {"pass"}
            else "fail"
            if "fail" in statuses
            else "inconclusive",
            "phases": results,
            "actual_execution_verified": False,
        }
    else:
        if not args.trace_summary:
            result = {
                "status": "unavailable",
                "reason": "No --trace-summary supplied; no recording was requested",
            }
        else:
            report = read_json(args.trace_summary)
            write_json(directory / "trace-summary.json", report)
            provenance = args.trace_attribution or args.trace_summary.with_suffix(
                ".provenance.json"
            )
            metadata = read_json(provenance) if provenance.is_file() else None
            result = trace_gate(
                report,
                phases,
                identity["model_manifest_sha256"],
                metadata,
                file_hash(args.trace_summary),
            )
            if metadata:
                write_json(directory / "trace-attribution.json", metadata)
                placement_path = metadata.get("placement_summary_path")
                if placement_path is not None:
                    placement_path = (ROOT / placement_path).resolve()
                    valid = placement_path.is_file() and file_hash(
                        placement_path
                    ) == metadata.get("placement_summary_sha256")
                    result["placement_summary_binding_verified"] = valid
                    if not valid:
                        result["status"] = "inconclusive"
                        result.setdefault("reasons", []).append(
                            "placement_summary_binding_failed"
                        )
    result["commands"] = commands
    return result


def claims(steps):
    result = {
        "protocol": steps["tests"],
        "quality": steps["quality"],
        "anticipated_placement": steps["placement"],
        "actual_device": steps["trace"],
        "energy": {
            "status": "unavailable",
            "reason": "No validated CPU/GPU/ANE energy evidence is available; inaccessible or stalled counters must not be treated as zero consumption",
            "joules": None,
            "power_reduction_verified": False,
        },
    }
    result["release_ready"] = all(
        result[name]["status"] == "pass"
        for name in (
            "protocol",
            "quality",
            "anticipated_placement",
            "actual_device",
            "energy",
        )
    )
    result["performance_improvement_verified"] = False
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model-dir", "source-dir", "manifest", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--steps", nargs="+", choices=STEPS, default=list(STEPS))
    parser.add_argument("--timeout-seconds", type=float, default=3600)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--trace-summary", type=Path)
    parser.add_argument(
        "--trace-attribution", type=Path, help="Versioned trace attribution sidecar"
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()
    if (
        args.timeout_seconds <= 0
        or not 100 <= args.bootstrap_samples <= 100000
        or args.max_new_tokens <= 0
    ):
        parser.error(
            "Timeout/tokens must be positive; bootstrap-samples must be 100..100000"
        )
    for name in (
        "model_dir",
        "source_dir",
        "manifest",
        "output_dir",
        "trace_summary",
        "trace_attribution",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.output_dir / "validation.json"
    phases = model_phases(args.model_dir)
    identity = fingerprint(args)
    if state_path.exists():
        if not args.resume:
            parser.error(
                "Output already contains a validation run; use --resume with identical inputs"
            )
        state = read_json(state_path)
        if state.get("identity") != identity:
            parser.error(
                "Resume refused: inputs/model manifests/payloads/code/settings changed; choose a new output directory"
            )
    else:
        state = {
            "schema_version": 1,
            "identity": identity,
            "gates": GATES,
            "steps": {
                name: {"status": "not_run", "reason": "not_selected"} for name in STEPS
            },
        }
    for name in STEPS:
        if name not in args.steps:
            continue
        previous = state["steps"][name]
        if (
            args.resume
            and previous["status"] in ("pass", "inconclusive", "unavailable")
            and evidence_intact(previous)
        ):
            print(f"{name}: reuse verified evidence ({previous['status']})", flush=True)
            continue
        if name == "quality" and state["steps"]["tests"]["status"] in (
            "fail",
            "unavailable",
            "inconclusive",
        ):
            state["steps"][name] = {
                "status": "not_run",
                "reason": "protocol_gate_not_passed",
            }
            continue
        index = len(list(args.output_dir.glob(f"{name}-attempt-*"))) + 1
        directory = args.output_dir / f"{name}-attempt-{index:03d}"
        directory.mkdir()
        print(f"{name}: starting", flush=True)
        try:
            result = execute_step(name, args, directory, identity, phases)
        except (
            OSError,
            ValueError,
            TypeError,
            KeyError,
            ElementTree.ParseError,
        ) as error:
            result = {"status": "fail", "reason": f"{type(error).__name__}: {error}"}
        write_json(directory / "result.json", result)
        result["evidence_hashes"] = {
            str(path): file_hash(path)
            for path in directory.rglob("*")
            if path.is_file()
        }
        state["steps"][name] = result
        state["claims"] = claims(state["steps"])
        write_json(state_path, state)
        print(f"{name}: {result['status']}", flush=True)
    state["claims"] = claims(state["steps"])
    write_json(state_path, state)
    write_json(args.output_dir / "claims.json", state["claims"])
    return 1 if any(step["status"] == "fail" for step in state["steps"].values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
