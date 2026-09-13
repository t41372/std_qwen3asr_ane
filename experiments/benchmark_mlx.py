"""Isolated MLX GPU reference with the same audio and scoring protocol as evaluate.py.

No acquisition occurs during a benchmark. --inspect-only reads model headers and
hashes files without importing MLX or loading a model.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from evaluate import (
    NORMALIZER,
    aggregate,
    audio_samples,
    manifest_rows,
    score,
    write_json,
)

WORKSPACE = Path(__file__).resolve().parents[1]
REFERENCE_ENV = WORKSPACE / "experiments/mlx_reference"
OFFICIAL_REVISION = "7278e1e70fe206f11671096ffdd38061171dd6e5"
MLX_REVISION = "e1f6c266914abc5a46e8756e02580f834a6cf8a7"
# mlx-community quantized conversions (decoder-only affine quantization, group 64).
MLX_QUANTIZED = {
    "q4": ("mlx-community/Qwen3-ASR-1.7B-4bit", "78a389c776a5483b2d0d4ea5494e11012e0d6159"),
    "q8": ("mlx-community/Qwen3-ASR-1.7B-8bit", "a8379a2e2f9e313c9292cdf1af4055ab56d50d55"),
}
WEIGHT_DTYPES = ("bf16", "q4", "q8")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def configure_local_caches() -> None:
    """Keep runtime caches local and require all model assets to already exist."""
    os.environ["HF_HOME"] = str(WORKSPACE / ".cache/huggingface")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["XDG_CACHE_HOME"] = str(WORKSPACE / ".cache")
    os.environ["UV_CACHE_DIR"] = str(WORKSPACE / ".cache/uv")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def provenance(model_dir: Path, model_repo: str, revision: str, weights_dtype: str = "bf16") -> dict:
    """Fingerprint the declared weights (BF16 or quantized) and the installed reference code."""
    source_record = model_dir / "source.json"
    if source_record.is_file():
        acquired = json.loads(source_record.read_text())
        if (
            acquired.get("model_id") != model_repo
            or acquired.get("revision") != revision
        ):
            raise ValueError(
                "Requested model provenance differs from the local acquisition record"
            )
    weights = []
    dtypes = Counter()
    for path in sorted(model_dir.glob("*.safetensors")):
        with path.open("rb") as stream:
            header_size = int.from_bytes(stream.read(8), "little")
            if not 0 < header_size < min(path.stat().st_size, 100_000_000):
                raise ValueError(f"Invalid safetensors header: {path}")
            header = json.loads(stream.read(header_size))
        for key, tensor in header.items():
            if key != "__metadata__":
                dtypes[tensor["dtype"]] += 1
        weights.append(
            {"path": path.name, "bytes": path.stat().st_size, "sha256": file_hash(path)}
        )
    expected_dtypes = {"BF16"} if weights_dtype == "bf16" else {"BF16", "U32"}
    if not weights or set(dtypes) != expected_dtypes:
        raise ValueError(
            f"Weights for {weights_dtype} must have tensor dtypes {expected_dtypes}; found {dict(dtypes)}"
        )
    configs = {}
    for path in sorted(model_dir.glob("*.json")):
        configs[path.name] = {"sha256": file_hash(path)}
    config = json.loads((model_dir / "config.json").read_text())
    quantization = config.get("quantization") or config.get("quantization_config")
    if weights_dtype == "bf16" and quantization:
        raise ValueError("Quantized model configuration cannot be labeled BF16")
    if weights_dtype != "bf16":
        expected_bits = int(weights_dtype[1:])
        if not quantization or int(quantization.get("bits", 0)) != expected_bits:
            raise ValueError(f"Expected a {expected_bits}-bit quantization block in config.json")
        # The audio encoder stays BF16 in these conversions; record that scope.
        quantization = {**quantization, "scope": "text decoder and embeddings only"}
    versions, sources = {}, {}
    for name in (
        "mlx-audio",
        "mlx",
        "mlx-metal",
        "transformers",
        "tokenizers",
        "huggingface-hub",
        "numpy",
        "scipy",
        "soundfile",
        "jiwer",
    ):
        versions[name] = importlib.metadata.version(name)
    distribution = importlib.metadata.distribution("mlx-audio")
    for relative in (
        "mlx_audio/utils.py",
        "mlx_audio/stt/utils.py",
        "mlx_audio/lm/generate.py",
        "mlx_audio/stt/models/qwen3_asr/qwen3_asr.py",
        "mlx_audio/stt/models/qwen3_asr/config.py",
    ):
        sources[relative] = file_hash(Path(distribution.locate_file(relative)))
    return {
        "model_repo": model_repo,
        "source_revision": revision,
        "revision_evidence": "Recorded acquisition revision; local payload independently SHA256-hashed",
        "weights_dtype": weights_dtype,
        "quantization": quantization if weights_dtype != "bf16" else None,
        "weight_tensor_dtypes": dict(dtypes),
        "weight_files": weights,
        "metadata_files": configs,
        "weights_independently_rehashed": True,
        "reference_code_sha256": sources,
        "benchmark_script_sha256": file_hash(Path(__file__)),
        "evaluation_helpers_sha256": file_hash(WORKSPACE / "experiments/evaluate.py"),
        "uv_lock_sha256": file_hash(REFERENCE_ENV / "uv.lock"),
        "environment": {
            "versions": versions,
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "official_source_loading": {
            "supported_by_static_inspection": True,
            "behavior": "MLX-Audio accepts thinker_config and sharded safetensors, removes thinker prefixes, and transposes convolution weights in memory; source files remain unchanged",
            "strict_weight_loading": True,
        },
        "fallback": {
            "model_repo": "mlx-community/Qwen3-ASR-1.7B-bf16",
            "revision": MLX_REVISION,
        },
    }


def load_backend(model_dir: Path, seed: int):
    """Load only a local model; leave preprocessing and decoding to MLX-Audio."""
    import mlx.core as mx
    from mlx_audio.stt import load

    sys.path.insert(0, str(WORKSPACE / "std_qwen3asr_ane/src"))
    from std_qwen3asr_ane.languages import LANGUAGE_NAMES, normalize_model_language

    mx.set_default_device(mx.gpu)
    mx.random.seed(seed)
    mx.synchronize()
    started = time.perf_counter()
    model = load(model_dir, strict=True, lazy=False)
    mx.synchronize()
    load_seconds = time.perf_counter() - started
    from mlx_audio.lm.generate import generation_stream

    def synchronize() -> None:
        # MLX-Audio dispatches decoder work on a separate generation stream.
        # A bare mx.synchronize() only waits for the default device stream.
        mx.synchronize(generation_stream)
        mx.synchronize()

    def transcribe(audio, language: str | None, max_tokens: int) -> dict:
        if language is not None and language not in LANGUAGE_NAMES:
            raise ValueError(f"Unsupported manifest language: {language}")
        synchronize()
        started = time.perf_counter()
        try:
            result = model.generate(
                audio,
                language=LANGUAGE_NAMES.get(language),
                max_tokens=max_tokens,
                temperature=0.0,
                batch_size=1,
                verbose=False,
            )
        finally:
            # Include all enqueued Metal work, including on an exception path.
            synchronize()
        seconds = time.perf_counter() - started
        if not isinstance(result.text, str):
            raise TypeError("MLX-Audio returned a non-string transcript")
        # MLX-Audio's public result omits EOS. At the cap, completion cannot be
        # established; preserve the attempt as a failure instead of scoring truncation.
        if result.generation_tokens >= max_tokens:
            raise RuntimeError(
                "MLX generation reached the token budget; completion is unverified"
            )
        languages = getattr(result, "language", None)
        if isinstance(languages, str):
            languages = [languages]
        detected = {normalize_model_language(item) for item in (languages or [])}
        detected.discard(None)
        return {
            "hypothesis": result.text,
            "detected_language": next(iter(detected)) if len(detected) == 1 else None,
            "seconds": seconds,
            "backend_timings": {
                "upstream_total_seconds": float(result.total_time),
                "prompt_tokens": int(result.prompt_tokens),
                "generation_tokens": int(result.generation_tokens),
            },
        }

    return transcribe, load_seconds


def run(args: argparse.Namespace) -> int:
    configure_local_caches()
    args.model_dir = args.model_dir.resolve()
    if args.output.exists() or Path(str(args.output) + ".summary.json").exists():
        raise ValueError(f"Refusing to overwrite run: {args.output}")
    metadata = provenance(args.model_dir, args.model_repo, args.model_revision, args.weights_dtype)
    if args.inspect_only:
        write_json(
            args.output,
            {
                "schema_version": 1,
                "model_loaded": False,
                "model_dir": str(args.model_dir),
                **metadata,
            },
        )
        print(f"Read-only model preflight written to {args.output}")
        return 0
    items = manifest_rows(args.manifest.resolve())
    started_at = datetime.now(UTC).isoformat()
    setup_error, transcribe, load_seconds = None, None, None
    try:
        transcribe, load_seconds = load_backend(args.model_dir, args.seed)
    except Exception as exc:  # noqa: BLE001 — record setup failures for every sample.
        setup_error = f"{type(exc).__name__}: {exc}"
    results = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        for item in items:
            audio, audio_hash, audio_error = None, None, None
            try:
                audio, audio_hash = audio_samples(Path(item["audio_path"]))
                if item.get("audio_sha256") and audio_hash != item["audio_sha256"]:
                    raise ValueError("Audio SHA256 does not match manifest")
            except Exception as exc:  # noqa: BLE001 — preserve malformed audio as benchmark data.
                audio_error = f"{type(exc).__name__}: {exc}"
            language = (
                item.get("language") if args.language_mode == "manifest" else None
            )
            for repeat in range(-args.warmups, args.repeats):
                record = {
                    "id": item["id"],
                    "reference": item["reference"],
                    "language": item.get("language"),
                    "split": item.get("split"),
                    "forced_language": language,
                    "language_mode": args.language_mode,
                    "audio_path": item["audio_path"],
                    "audio_sha256": audio_hash,
                    "audio_seconds": len(audio) / 16000 if audio is not None else None,
                    "backend": "mlx",
                    "model_dir": str(args.model_dir),
                    "compute_units": "mlx_gpu",
                    "weights_dtype": args.weights_dtype,
                    "max_new_tokens": args.max_new_tokens,
                    "normalizer": NORMALIZER,
                    "phase": "warmup" if repeat < 0 else "measured",
                    "repeat": repeat,
                    "hypothesis": None,
                    "detected_language": None,
                    "seconds": None,
                    "rtf": None,
                    "scores": None,
                    "error": setup_error or audio_error,
                }
                if record["error"] is None:
                    begin = time.perf_counter()
                    try:
                        record.update(transcribe(audio, language, args.max_new_tokens))
                        record["rtf"] = record["seconds"] / record["audio_seconds"]
                        record["scores"] = score(
                            item["reference"], record["hypothesis"]
                        )
                    except Exception as exc:  # noqa: BLE001 — retain backend failures in the run.
                        record["seconds"] = time.perf_counter() - begin
                        record["error"] = f"{type(exc).__name__}: {exc}"
                output.write(
                    json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
                )
                output.flush()
                results.append(record)
                print(
                    f"{record['id']} {record['phase']} {repeat}: {record['error'] or 'ok'}",
                    flush=True,
                )
    summary = {
        "schema_version": 1,
        "started_at": started_at,
        "finished_at": datetime.now(UTC).isoformat(),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": file_hash(args.manifest),
        "backend": "mlx",
        "model_dir": str(args.model_dir),
        "model_metadata": metadata,
        "model_load_seconds": load_seconds,
        "model_load_error": setup_error,
        "normalizer": NORMALIZER,
        "quality_repeat": 0,
        "timing_scope": "mx.synchronize on generation and default streams before timer and after generate; excludes audio file decode/resampling, provenance hashing and model load; includes MLX-Audio feature extraction/prefill/decode",
        "warmups_per_sample": args.warmups,
        "repeats": args.repeats,
        "configuration": {
            "compute_units": "mlx_gpu",
            "weights_dtype": args.weights_dtype,
            "max_new_tokens": args.max_new_tokens,
            "language_mode": args.language_mode,
            "seed": args.seed,
            "temperature": 0.0,
            "batch_size": 1,
        },
        "environment": metadata["environment"],
        "comparability": f"MLX GPU with {args.weights_dtype} weights and its own kernels/frontend; not an interchangeable device setting for Core ML FP16/ANE. No energy measurement is inferred from latency.",
        **aggregate(results),
    }
    write_json(Path(str(args.output) + ".summary.json"), summary)
    return int(any(record["error"] is not None for record in results))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--model-dir", type=Path, default=WORKSPACE / "artifacts/source/Qwen3-ASR-1.7B"
    )
    parser.add_argument("--model-repo", default="Qwen/Qwen3-ASR-1.7B")
    parser.add_argument("--model-revision", default=OFFICIAL_REVISION)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--language-mode", choices=("auto", "manifest"), default="auto")
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--weights-dtype", choices=WEIGHT_DTYPES, default="bf16")
    args = parser.parse_args()
    if args.weights_dtype != "bf16":
        repo, revision = MLX_QUANTIZED[args.weights_dtype]
        if args.model_repo == "Qwen/Qwen3-ASR-1.7B":
            args.model_repo, args.model_revision = repo, revision
        if (args.model_repo, args.model_revision) != (repo, revision):
            parser.error(f"{args.weights_dtype} requires {repo}@{revision}")
    if not args.inspect_only and args.manifest is None:
        parser.error("--manifest is required for inference")
    if args.warmups < 0 or args.repeats < 1 or args.max_new_tokens < 1:
        parser.error(
            "warmups must be nonnegative; repeats/max-new-tokens must be positive"
        )
    if len(args.model_revision) != 40 or any(
        c not in "0123456789abcdef" for c in args.model_revision
    ):
        parser.error("--model-revision must be a pinned 40-character SHA")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
