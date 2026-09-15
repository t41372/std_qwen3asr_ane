"""Warm, alternating end-to-end corpus timing with evaluator-compatible rows."""

import argparse
import json
from pathlib import Path
from time import perf_counter

from baseline_runtime import load_baseline_runtime, prediction_models
from evaluate import (
    NORMALIZER,
    aggregate,
    audio_samples,
    environment,
    manifest_rows,
    score,
)

from std_qwen3asr_ane.bundle import digest
from std_qwen3asr_ane.runtime import CoreMLRuntime, PersistentInputModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--baseline-source",
        type=Path,
        help="Immutable package directory for measuring source changes as well as artifacts",
    )
    args = parser.parse_args()
    if args.output.exists() or min(args.repeats, args.max_new_tokens) < 1:
        parser.error("Use a fresh output and positive repetitions/token budget")
    args.output.mkdir(parents=True)
    inputs = [
        (row, *audio_samples(Path(row["audio_path"]))) for row in manifest_rows(args.manifest)
    ]
    for row, _, audio_hash in inputs:
        if row.get("audio_sha256") != audio_hash:
            raise ValueError("Manifest audio identity changed")
    runtimes, streams, records = {}, {}, {"baseline": [], "candidate": []}
    factories = {
        "baseline": load_baseline_runtime(args.baseline_source)
        if args.baseline_source
        else CoreMLRuntime,
        "candidate": CoreMLRuntime,
    }
    report = {
        "complete": False,
        "close_succeeded": False,
        "method": "one global warmup per language/model; measured repetitions with alternating model order per utterance",
        "manifest_sha256": digest(args.manifest),
        "environment": environment(),
        "repeats": args.repeats,
        "max_new_tokens": args.max_new_tokens,
        "baseline_runtime_sha256": digest(args.baseline_source / "runtime.py")
        if args.baseline_source
        else None,
        "models": {
            name: {
                "path": str(path.resolve()),
                "manifest_sha256": digest(path / "manifest.json"),
            }
            for name, path in (
                ("baseline", args.baseline),
                ("candidate", args.candidate),
            )
        },
    }
    try:
        for name, path in (("baseline", args.baseline), ("candidate", args.candidate)):
            runtimes[name] = factories[name](path)
            streams[name] = (args.output / f"{name}.jsonl").open("x")
            warmed = set()
            for row, audio, _ in inputs:
                if row["language"] not in warmed:
                    runtimes[name].transcribe(
                        audio, language=None, max_new_tokens=args.max_new_tokens
                    )
                    warmed.add(row["language"])
        for ordinal, (row, audio, audio_hash) in enumerate(inputs):
            for repeat in range(args.repeats):
                order = (
                    ("baseline", "candidate")
                    if (ordinal + repeat) % 2 == 0
                    else ("candidate", "baseline")
                )
                for name in order:
                    record = {
                        "id": row["id"],
                        "repeat": repeat,
                        "phase": "measured",
                        "reference": row["reference"],
                        "language": row["language"],
                        "audio_sha256": audio_hash,
                        "audio_seconds": len(audio) / 16000,
                        "forced_language": None,
                        "language_mode": "auto",
                        "max_new_tokens": args.max_new_tokens,
                        "normalizer": NORMALIZER,
                        "error": None,
                    }
                    started = perf_counter()
                    try:
                        result = runtimes[name].transcribe(
                            audio, language=None, max_new_tokens=args.max_new_tokens
                        )
                        elapsed = perf_counter() - started
                        record.update(
                            seconds=elapsed,
                            hypothesis=result.text,
                            raw_text=result.raw_text,
                            detected_language=result.language,
                            token_ids=list(result.token_ids),
                            backend_timings=result.timings,
                            rtf=elapsed / (len(audio) / 16000),
                            scores=score(row["reference"], result.text),
                        )
                    except Exception as error:  # noqa: BLE001 — retain failed attempts in the gate.
                        record.update(
                            seconds=perf_counter() - started,
                            error=f"{type(error).__name__}: {error}",
                            hypothesis=None,
                        )
                    records[name].append(record)
                    streams[name].write(
                        json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
                    )
                    streams[name].flush()
            print(
                json.dumps(
                    {
                        "completed_utterances": ordinal + 1,
                        "total": len(inputs),
                        "id": row["id"],
                    }
                ),
                flush=True,
            )
        report["complete"] = all(not row["error"] for rows in records.values() for row in rows)
    finally:
        for stream in streams.values():
            stream.close()
        try:
            models = [
                model for runtime in runtimes.values() for model in prediction_models(runtime)
            ]
            PersistentInputModel.close_many(models)
            report["close_succeeded"] = True
        finally:
            for name, rows in records.items():
                if rows:
                    (args.output / f"{name}.summary.json").write_text(
                        json.dumps(aggregate(rows), indent=2) + "\n"
                    )
            (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
