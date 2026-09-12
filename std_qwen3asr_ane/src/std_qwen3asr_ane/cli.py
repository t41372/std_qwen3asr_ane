"""Build, inspect, and transcribe with the local ANE engine."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="Download the pinned official checkpoint")
    download.add_argument("--output", type=Path, default=Path("artifacts/source/Qwen3-ASR-1.7B"))
    download.add_argument("--revision", default="main")
    build = commands.add_parser(
        "build", help="Convert a local checkpoint; requires the convert group"
    )
    build.add_argument("--source", type=Path, default=Path("artifacts/source/Qwen3-ASR-1.7B"))
    build.add_argument("--output", type=Path, default=Path("artifacts/qwen3-asr-1.7b"))
    build.add_argument("--cache-length", type=int, choices=(512, 1024, 2048), default=1024)
    build.add_argument("--reuse-encoder", action="store_true")
    build.add_argument("--token-batch-size", type=int, choices=(1, 8, 16, 32), default=1)
    inspect = commands.add_parser("inspect", help="Report anticipated compute placement")
    inspect.add_argument("model", type=Path)
    inspect.add_argument(
        "--compute-units", choices=("cpu_and_ne", "cpu_only"), default="cpu_and_ne"
    )
    transcribe = commands.add_parser("transcribe")
    transcribe.add_argument("audio", type=Path)
    transcribe.add_argument("--model-dir", type=Path, default=Path("artifacts/qwen3-asr-1.7b"))
    transcribe.add_argument("--language", default="auto")
    transcribe.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args(argv)
    if args.command == "download":
        from .conversion.build import download_source

        result = download_source(args.output, revision=args.revision)
    elif args.command == "build":
        from .conversion.build import build_bundle

        result = build_bundle(
            args.source,
            args.output,
            cache_length=args.cache_length,
            reuse_encoder=args.reuse_encoder,
            token_batch_size=args.token_batch_size,
        )
    elif args.command == "inspect":
        from .diagnostics import inspect_compute_plan

        result = inspect_compute_plan(args.model, args.compute_units)
    else:
        from standard_asr.engine import RuntimeParams

        from .plugin import Qwen3ASREngine

        engine = Qwen3ASREngine(model_dir=args.model_dir, max_new_tokens=args.max_new_tokens)
        try:
            result = engine.transcribe(
                args.audio, RuntimeParams(language=args.language)
            ).model_dump(mode="json")
        finally:
            engine.close()
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
