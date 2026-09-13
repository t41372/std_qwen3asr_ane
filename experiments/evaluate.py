"""Reproducible, serial ASR evaluation; no model acquisition or power claims.

Run with the project's uv environment and convert dependency group. Results are
JSONL; metadata and aggregate statistics live in <output>.summary.json.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
import unicodedata
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from math import gcd
from pathlib import Path
from typing import Any

import jiwer
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

NORMALIZER = "nfkc_casefold_unicode_punctuation_to_space_v1"
CHARACTER_LANGUAGES = {
    "zh",
    "chinese",
    "cmn",
    "yue",
    "cantonese",
    "ja",
    "japanese",
    "th",
    "thai",
}


def normalize(text: str, *, characters: bool = False) -> str:
    """Deliberately simple: no number expansion, script conversion or tokenization."""
    text = unicodedata.normalize("NFKC", text).casefold()
    text = "".join(" " if unicodedata.category(c).startswith("P") else c for c in text)
    text = " ".join(text.split())
    return text.replace(" ", "") if characters else text


def character_language(language: str | None) -> bool:
    return (language or "").lower().replace("_", "-").split("-")[
        0
    ] in CHARACTER_LANGUAGES


def score(reference: str, hypothesis: str) -> dict[str, Any]:
    result = {}
    for metric, characters in (("wer", False), ("cer", True)):
        ref, hyp = (
            normalize(s, characters=characters) for s in (reference, hypothesis)
        )
        alignment = (
            jiwer.process_characters(ref, hyp)
            if characters
            else jiwer.process_words(ref, hyp)
        )
        errors = alignment.substitutions + alignment.deletions + alignment.insertions
        units = alignment.hits + alignment.substitutions + alignment.deletions
        result[metric] = {
            "errors": errors,
            "reference_units": units,
            "substitutions": alignment.substitutions,
            "deletions": alignment.deletions,
            "insertions": alignment.insertions,
            # Empty-reference hallucinations retain their counts; no artificial denominator.
            "rate": errors / units if units else None,
        }
    return result


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise TypeError(f"{path}:{number}: expected a JSON object")
        rows.append(row)
    if not rows:
        raise ValueError(f"{path}: empty input")
    return rows


def manifest_rows(path: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    seen = set()
    for row in rows:
        for key in ("id", "audio_path", "reference"):
            if not isinstance(row.get(key), str):
                raise TypeError(f"Manifest requires string {key!r}: {row!r}")
        if not row["id"] or row["id"] in seen:
            raise ValueError(f"Empty or duplicate sample ID: {row['id']!r}")
        seen.add(row["id"])
        if row.get("language") is not None and not isinstance(row["language"], str):
            raise ValueError(f"{row['id']}: language must be a string or null")
        audio_path = Path(row["audio_path"]).expanduser()
        row["audio_path"] = str((path.parent / audio_path).resolve())
    return rows


def audio_samples(path: Path) -> tuple[np.ndarray, str]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    if not len(audio) or not np.isfinite(audio).all() or rate <= 0:
        raise ValueError("Audio must contain finite samples at a positive sample rate")
    audio = audio.mean(axis=1, dtype=np.float32)
    if rate != 16000:
        common = gcd(rate, 16000)
        audio = resample_poly(audio, 16000 // common, rate // common)
    return np.ascontiguousarray(audio, dtype=np.float32), digest


@dataclass(frozen=True)
class Backend:
    transcribe: Callable[[np.ndarray, str | None], tuple[str, str | None, Any]]
    close: Callable[[], None] | None = None
    metadata: dict[str, Any] | None = None
    redact_errors: bool = False


def backend_error(exc: Exception, *, redact: bool) -> str:
    if redact:
        return f"{type(exc).__name__}: Standard plugin operation failed; configuration details omitted."
    return f"{type(exc).__name__}: {exc}"


DTYPE_LABELS = {"float32": "fp32", "bfloat16": "bf16", "float16": "fp16"}


def compute_label(args: argparse.Namespace) -> str:
    if args.backend == "standard":
        return "plugin_managed"
    if args.backend == "coreml":
        return args.compute_units
    return f"{args.device}_{DTYPE_LABELS[args.dtype]}"


def make_backend(args: argparse.Namespace) -> Backend:
    if args.backend == "standard":
        from standard_asr import RuntimeParams, discover_models

        engine = discover_models(strict=True).create(
            args.model_key, **args.engine_config
        )

        def transcribe(
            audio: np.ndarray, language: str | None
        ) -> tuple[str, str | None, Any]:
            result = engine.transcribe((audio, 16000), RuntimeParams(language=language))
            return result.text, result.detected_language, None

        close = getattr(engine, "close", None)
        return Backend(
            transcribe,
            close=close if callable(close) else None,
            metadata={
                "model_key": engine.properties.model_id,
                "engine_config": engine.config.public_dump(),
            },
            redact_errors=True,
        )
    if args.backend == "coreml":
        from std_qwen3asr_ane.runtime import CoreMLRuntime

        runtime = CoreMLRuntime(
            args.model_dir.resolve(), compute_units=args.compute_units
        )

        def transcribe(
            audio: np.ndarray, language: str | None
        ) -> tuple[str, str | None, Any]:
            result = runtime.transcribe(
                audio, language=language, max_new_tokens=args.max_new_tokens
            )
            return result.text, result.language, getattr(result, "timings", None)

        return Backend(transcribe, close=runtime.close)

    # Optional baseline imports are lazy. No MLX package is installed or imported.
    import torch
    from qwen_asr import Qwen3ASRModel
    from std_qwen3asr_ane.languages import LANGUAGE_NAMES

    if args.torch_threads:
        torch.set_num_threads(args.torch_threads)
    torch.manual_seed(args.seed)
    model = Qwen3ASRModel.from_pretrained(
        str(args.model_dir.resolve()),
        dtype=getattr(torch, args.dtype),
        device_map=args.device,
        attn_implementation=args.attn_implementation,
        max_inference_batch_size=1,
        max_new_tokens=args.max_new_tokens,
        local_files_only=True,
    )
    model.model.eval()

    def transcribe(
        audio: np.ndarray, language: str | None
    ) -> tuple[str, str | None, Any]:
        name = LANGUAGE_NAMES.get(language, language) if language else None
        with torch.inference_mode():
            result = model.transcribe(audio=(audio, 16000), language=name)[0]
        if args.device == "mps":
            torch.mps.synchronize()  # the timed window must include GPU completion
        return result.text, result.language, None

    return Backend(transcribe)


def close_backend(backend: Backend | None, output: Path) -> dict[str, Any]:
    """Record cleanup separately from sample scores, including on an interrupted run."""
    cleanup: dict[str, Any] = {
        "schema_version": 1,
        "lifecycle_revision": "explicit_close_v1",
        "status": "not_loaded" if backend is None else "not_applicable",
        "seconds": None,
        "error": None,
    }
    if backend is not None and backend.close is not None:
        begin = time.perf_counter()
        try:
            backend.close()
            cleanup["status"] = "succeeded"
        except Exception as exc:  # noqa: BLE001 — a failed close is separate lifecycle evidence.
            cleanup["status"] = "failed"
            cleanup["error"] = backend_error(exc, redact=backend.redact_errors)
            print(
                f"Backend close failed: {cleanup['error']}", file=sys.stderr, flush=True
            )
        cleanup["seconds"] = time.perf_counter() - begin
    cleanup["finished_at"] = datetime.now(UTC).isoformat()
    write_json(Path(str(output) + ".cleanup.json"), cleanup)
    return cleanup


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Micro-average edit counts; quality uses repeat zero once per utterance."""
    measured = [r for r in rows if r["phase"] == "measured"]
    first = [r for r in measured if r["repeat"] == 0]
    valid = [r for r in first if r["error"] is None]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    groups["all"] = valid
    for row in valid:
        groups[f"language:{row.get('language') or 'unspecified'}"].append(row)
    quality = {}
    for name, group in groups.items():
        entry = {"samples": len(group)}
        for metric in ("wer", "cer"):
            selected = [
                r
                for r in group
                if metric == "cer" or not character_language(r.get("language"))
            ]
            errors = sum(r["scores"][metric]["errors"] for r in selected)
            units = sum(r["scores"][metric]["reference_units"] for r in selected)
            entry[metric] = {
                "rate": errors / units if units else None,
                "errors": errors,
                "reference_units": units,
                "samples": len(selected),
            }
        quality[name] = entry
    successful = [r for r in measured if r["error"] is None]
    times = [r["seconds"] for r in successful]
    durations = sum(r["audio_seconds"] for r in successful)
    nondeterministic = []
    by_id: dict[str, set[str]] = defaultdict(set)
    for row in successful:
        by_id[row["id"]].add(row["hypothesis"])
    for sample_id, hypotheses in by_id.items():
        if len(hypotheses) > 1:
            nondeterministic.append(sample_id)
    return {
        "expected_samples": len(first),
        "scored_samples": len(valid),
        "quality_complete": len(valid) == len(first),
        "failed_sample_ids": [r["id"] for r in first if r["error"]],
        "failed_measured_attempts": sum(r["error"] is not None for r in measured),
        "failed_warmup_attempts": sum(
            r["error"] is not None for r in rows if r["phase"] == "warmup"
        ),
        "nondeterministic_sample_ids": nondeterministic,
        "quality": quality,
        "latency": {
            "successful_attempts": len(times),
            "total_seconds": sum(times),
            "total_audio_seconds": durations,
            "corpus_rtf": sum(times) / durations if durations else None,
            "median_seconds": float(np.median(times)) if times else None,
            "p95_seconds": float(np.percentile(times, 95)) if times else None,
        },
        "energy": {"measured": False, "joules": None},
    }


def environment() -> dict[str, Any]:
    versions = {}
    for name in (
        "std-qwen3asr-ane",
        "qwen-asr",
        "coremltools",
        "torch",
        "numpy",
        "jiwer",
        "soundfile",
        "scipy",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "versions": versions,
    }


def model_metadata(root: Path) -> dict[str, Any]:
    """Record small manifests/configs; do not rescan gigabytes during a timed run."""
    files = {}
    for name in (
        "manifest.json",
        "encoder-manifest.json",
        "config.json",
        "generation_config.json",
    ):
        path = root / name
        if path.is_file():
            data = path.read_bytes()
            files[name] = {
                "sha256": hashlib.sha256(data).hexdigest(),
                "content": json.loads(data),
            }
    return {"metadata_files": files, "weights_independently_rehashed": False}


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


def run(args: argparse.Namespace) -> int:
    rows = manifest_rows(args.manifest.resolve())
    if args.backend != "standard" and not args.model_dir.is_dir():
        raise ValueError(f"Model directory must already exist: {args.model_dir}")
    model_dir = str(args.model_dir.resolve()) if args.model_dir is not None else None
    token_budget = None if args.backend == "standard" else args.max_new_tokens
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if any(
        Path(str(args.output) + suffix).exists()
        for suffix in ("", ".summary.json", ".cleanup.json")
    ):
        raise ValueError(f"Refusing to overwrite run: {args.output}")
    started = datetime.now(UTC).isoformat()
    setup_error = None
    start = time.perf_counter()
    try:
        backend = make_backend(args)
    except Exception as exc:  # noqa: BLE001 — persist model-load failures for every sample.
        setup_error = backend_error(exc, redact=args.backend == "standard")
        backend = None
    load_seconds = time.perf_counter() - start
    results = []
    try:
        with args.output.open("x") as output:
            for item in rows:
                audio = None
                audio_error = None
                audio_hash = None
                try:
                    audio, audio_hash = audio_samples(Path(item["audio_path"]))
                    if item.get("audio_sha256") and item["audio_sha256"] != audio_hash:
                        raise ValueError("Audio SHA256 does not match manifest")
                except Exception as exc:  # noqa: BLE001 — malformed audio must remain in results.
                    audio_error = f"{type(exc).__name__}: {exc}"
                language = (
                    item.get("language") if args.language_mode == "manifest" else None
                )
                for index in range(-args.warmups, args.repeats):
                    record = {
                        "id": item["id"],
                        "reference": item["reference"],
                        "language": item.get("language"),
                        "split": item.get("split"),
                        "forced_language": language,
                        "language_mode": args.language_mode,
                        "audio_path": item["audio_path"],
                        "audio_sha256": audio_hash,
                        "audio_seconds": len(audio) / 16000
                        if audio is not None
                        else None,
                        "backend": args.backend,
                        "model_dir": model_dir,
                        "compute_units": compute_label(args),
                        "max_new_tokens": token_budget,
                        "normalizer": NORMALIZER,
                        "phase": "warmup" if index < 0 else "measured",
                        "repeat": index,
                        "hypothesis": None,
                        "detected_language": None,
                        "seconds": None,
                        "rtf": None,
                        "scores": None,
                        "error": setup_error or audio_error,
                    }
                    if args.backend == "standard":
                        record["model_key"] = args.model_key
                    if record["error"] is None:
                        begin = time.perf_counter()
                        try:
                            hypothesis, detected, timings = backend.transcribe(
                                audio, language
                            )
                            record["seconds"] = time.perf_counter() - begin
                            if not isinstance(hypothesis, str):
                                raise TypeError(
                                    "Backend returned non-string hypothesis"
                                )
                            record.update(
                                hypothesis=hypothesis,
                                detected_language=detected,
                                rtf=record["seconds"] / record["audio_seconds"],
                                scores=score(item["reference"], hypothesis),
                            )
                            if isinstance(timings, dict):
                                record["backend_timings"] = timings
                        except Exception as exc:  # noqa: BLE001 — backend failures are benchmark data.
                            record["seconds"] = time.perf_counter() - begin
                            record["error"] = backend_error(
                                exc, redact=args.backend == "standard"
                            )
                    output.write(
                        json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
                    )
                    output.flush()
                    results.append(record)
                    print(
                        f"{item['id']} {record['phase']} {index}: {record['error'] or 'ok'}",
                        flush=True,
                    )
    finally:
        cleanup = close_backend(backend, args.output)
    summary = {
        "schema_version": 1,
        "started_at": started,
        "finished_at": datetime.now(UTC).isoformat(),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "backend": args.backend,
        "model_dir": model_dir,
        "model_metadata": (backend.metadata if backend is not None else {})
        if args.backend == "standard"
        else model_metadata(args.model_dir),
        "model_load_seconds": None if args.backend == "standard" else load_seconds,
        "model_load_error": setup_error,
        "model_close_seconds": cleanup["seconds"],
        "model_close_error": cleanup["error"],
        "cleanup_record": str(Path(str(args.output) + ".cleanup.json").resolve()),
        "normalizer": NORMALIZER,
        "quality_repeat": 0,
        "timing_scope": "synchronous standard engine.transcribe; includes any lazy plugin initialization; excludes audio file decode/resampling and engine construction"
        if args.backend == "standard"
        else "synchronous transcribe only; excludes audio decode/resample and model load; includes mel/prefill/decode",
        "warmups_per_sample": args.warmups,
        "repeats": args.repeats,
        "configuration": {
            "compute_units": compute_label(args),
            "max_new_tokens": token_budget,
            "language_mode": args.language_mode,
            "seed": None if args.backend == "standard" else args.seed,
            "torch_threads_requested": None
            if args.backend == "standard"
            else args.torch_threads,
            "torch_threads_effective": sys.modules["torch"].get_num_threads()
            if args.backend == "official" and "torch" in sys.modules
            else None,
        },
        "environment": environment(),
        **aggregate(results),
    }
    if args.backend == "standard":
        summary["model_key"] = args.model_key
        summary["engine_create_seconds"] = load_seconds
        summary["configuration"]["language_resolution"] = (
            "manifest_override"
            if args.language_mode == "manifest"
            else "plugin_default"
        )
    write_json(Path(str(args.output) + ".summary.json"), summary)
    return 1 if cleanup["error"] or any(r["error"] for r in results) else 0


def paired_bootstrap(
    base: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    *,
    metric: str,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    """Paired utterance bootstrap of the corpus micro-average, candidate minus base."""
    differences, units = [], []
    for left, right in zip(base, candidate, strict=True):
        if metric == "wer" and character_language(left.get("language")):
            continue
        a = score(left["reference"], left["hypothesis"])[metric]
        b = score(right["reference"], right["hypothesis"])[metric]
        differences.append(b["errors"] - a["errors"])
        units.append(a["reference_units"])
    n = len(units)
    if n == 0 or sum(units) == 0:
        return {
            "samples": n,
            "delta": None,
            "ci95": None,
            "reason": "no reference units",
        }
    delta = sum(differences) / sum(units)
    if n < 2:
        return {
            "samples": n,
            "delta": delta,
            "ci95": None,
            "reason": "at least two utterances required",
        }
    rng = np.random.default_rng(seed)
    differences, units = np.array(differences), np.array(units)
    values = []
    # One resample at a time bounds memory independently of corpus size.
    for _ in range(samples):
        indices = rng.integers(0, n, size=n)
        denominator = units[indices].sum()
        if denominator:
            values.append(float(differences[indices].sum() / denominator))
    return {
        "samples": n,
        "delta": delta,
        "ci95": np.quantile(values, [0.025, 0.975]).tolist() if values else None,
        "bootstrap_resamples": samples,
        "valid_resamples": len(values),
        "zero_denominator_resamples": samples - len(values),
        "seed": seed,
        "exploratory_small_sample": n < 30,
    }


def compare(args: argparse.Namespace) -> int:
    paths = [p.resolve() for p in args.compare]
    runs = []
    reliability_problems = []
    for path in paths:
        all_rows = read_jsonl(path)
        rows = [
            r for r in all_rows if r.get("phase") == "measured" and r.get("repeat") == 0
        ]
        mapping = {r["id"]: r for r in rows}
        if not mapping or len(mapping) != len(rows):
            raise ValueError(f"{path}: missing or duplicate repeat-zero measured rows")
        observed: dict[str, set[str]] = defaultdict(set)
        for row in all_rows:
            if row.get("phase") == "measured":
                if row.get("error"):
                    reliability_problems.append(
                        {
                            "run": str(path),
                            "id": row["id"],
                            "repeat": row.get("repeat"),
                            "error": row["error"],
                        }
                    )
                elif isinstance(row.get("hypothesis"), str):
                    observed[row["id"]].add(row["hypothesis"])
        for sample_id, hypotheses in observed.items():
            if len(hypotheses) > 1:
                reliability_problems.append(
                    {
                        "run": str(path),
                        "id": sample_id,
                        "nondeterministic_repeats": True,
                    }
                )
        runs.append(mapping)
    base, candidate = runs
    problems = reliability_problems
    if base.keys() != candidate.keys():
        problems.append(
            {
                "id_set_mismatch": {
                    "only_baseline": sorted(base.keys() - candidate.keys()),
                    "only_candidate": sorted(candidate.keys() - base.keys()),
                }
            }
        )
    common = sorted(base.keys() & candidate.keys())
    for sample_id in common:
        a, b = base[sample_id], candidate[sample_id]
        if a.get("error") or b.get("error"):
            problems.append(
                {
                    "id": sample_id,
                    "baseline_error": a.get("error"),
                    "candidate_error": b.get("error"),
                }
            )
        for key in (
            "reference",
            "language",
            "audio_sha256",
            "forced_language",
            "language_mode",
            "max_new_tokens",
            "normalizer",
        ):
            if key not in a or key not in b or a[key] != b[key]:
                problems.append({"id": sample_id, "mismatch": key})
        if (
            not a.get("audio_sha256")
            or not isinstance(a.get("hypothesis"), str)
            or not isinstance(b.get("hypothesis"), str)
        ):
            problems.append(
                {"id": sample_id, "invalid_audio_identity_or_hypothesis": True}
            )
        if a.get("normalizer") != NORMALIZER:
            problems.append(
                {"id": sample_id, "unsupported_normalizer": a.get("normalizer")}
            )
    report = {
        "baseline": str(paths[0]),
        "candidate": str(paths[1]),
        "normalizer": NORMALIZER,
        "direction": "candidate minus baseline; positive means regression",
        "paired_samples": len(common),
        "valid_comparison": not problems,
        "problems": problems,
    }
    if not problems:
        a, b = ([mapping[i] for i in common] for mapping in runs)
        report["quality"] = {
            m: paired_bootstrap(
                a, b, metric=m, samples=args.bootstrap_samples, seed=args.seed
            )
            for m in ("wer", "cer")
        }
        report["by_language"] = {}
        for language in sorted({row.get("language") or "unspecified" for row in a}):
            indices = [
                i
                for i, row in enumerate(a)
                if (row.get("language") or "unspecified") == language
            ]
            report["by_language"][language] = {
                metric: paired_bootstrap(
                    [a[i] for i in indices],
                    [b[i] for i in indices],
                    metric=metric,
                    samples=args.bootstrap_samples,
                    seed=args.seed,
                )
                for metric in ("wer", "cer")
            }
        report["interpretation"] = (
            "Utterance bootstrap, not speaker-cluster bootstrap. Small samples cannot establish WER non-inferiority."
        )
    if args.output.exists():
        raise ValueError(f"Refusing to overwrite comparison: {args.output}")
    write_json(args.output, report)
    return 1 if problems else 0


def engine_config_json(value: str) -> dict[str, Any]:
    try:
        config = json.loads(value)
    except (ValueError, TypeError):
        raise argparse.ArgumentTypeError(
            "Engine configuration must be a JSON object."
        ) from None
    if not isinstance(config, dict):
        raise argparse.ArgumentTypeError("Engine configuration must be a JSON object.")
    return config


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--backend", choices=("coreml", "official", "standard"))
    result.add_argument("--manifest", type=Path)
    result.add_argument("--model-dir", type=Path)
    result.add_argument("--model-key", help="Installed Standard ASR entry-point key.")
    result.add_argument(
        "--engine-config",
        type=engine_config_json,
        default={},
        help="JSON object passed to the standard plugin's typed config; only public_dump is recorded.",
    )
    result.add_argument("--output", type=Path, required=True)
    result.add_argument(
        "--compute-units", choices=("cpu_and_ne", "cpu_only"), default="cpu_and_ne"
    )
    result.add_argument("--repeats", type=int, default=1)
    result.add_argument("--warmups", type=int, default=1)
    result.add_argument("--max-new-tokens", type=int, default=256)
    result.add_argument("--language-mode", choices=("auto", "manifest"), default="auto")
    result.add_argument("--torch-threads", type=int, default=0)
    result.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    result.add_argument("--dtype", choices=tuple(DTYPE_LABELS), default="float32")
    result.add_argument("--attn-implementation", choices=("eager", "sdpa"), default="eager")
    result.add_argument("--seed", type=int, default=20260912)
    result.add_argument(
        "--compare", nargs=2, type=Path, metavar=("BASELINE_JSONL", "CANDIDATE_JSONL")
    )
    result.add_argument("--bootstrap-samples", type=int, default=2000)
    return result


def main() -> int:
    cli = parser()
    args = cli.parse_args()
    if (
        args.repeats < 1
        or args.warmups < 0
        or args.max_new_tokens < 1
        or args.torch_threads < 0
    ):
        cli.error(
            "repeats/max-new-tokens must be positive; warmups/torch-threads nonnegative"
        )
    if not 100 <= args.bootstrap_samples <= 100000:
        cli.error("bootstrap-samples must be between 100 and 100000")
    if not args.compare:
        if not all((args.backend, args.manifest)):
            cli.error("run mode requires --backend and --manifest")
        if args.backend == "standard":
            if not args.model_key:
                cli.error("standard backend requires --model-key")
            if args.model_dir is not None:
                cli.error("For standard plugins, pass model_dir inside --engine-config")
        elif args.model_dir is None:
            cli.error("coreml/official backends require --model-dir")
    try:
        return compare(args) if args.compare else run(args)
    except (ValueError, TypeError, OSError, KeyError) as exc:
        cli.exit(2, f"Evaluation configuration error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
