"""Build, inspect, and transcribe with the local ANE engine."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .conversion.build import SOURCE_REVISION
from .profiles import PROFILES


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="Download the pinned official checkpoint")
    download.add_argument("--output", type=Path, default=Path("artifacts/source/Qwen3-ASR-1.7B"))
    download.add_argument(
        "--revision",
        default=SOURCE_REVISION,
        help="checkpoint revision; the default is the revision every measurement used",
    )
    build = commands.add_parser(
        "build", help="Convert a local checkpoint; requires the convert group"
    )
    build.add_argument("--source", type=Path, default=Path("artifacts/source/Qwen3-ASR-1.7B"))
    build.add_argument("--output", type=Path, default=None)
    build.add_argument("--profile", choices=tuple(PROFILES), default=None)
    build.add_argument("--cache-length", type=int, choices=(512, 1024, 2048), default=None)
    build.add_argument("--reuse-encoder", action="store_true")
    build.add_argument("--token-batch-size", type=int, choices=(1, 8, 16, 32, 64), default=None)
    build.add_argument(
        "--layers-per-partition",
        type=int,
        choices=(4, 7, 14),
        default=14,
        help="decoder layers per Core ML model; 14 measured fastest, 28 fails to load on macOS 27",
    )
    compress = commands.add_parser(
        "compress", help="Write a weight-compressed copy of an uncompiled bundle"
    )
    compress.add_argument("--source", type=Path, required=True)
    compress.add_argument("--output", type=Path, required=True)
    compress.add_argument("--scheme", choices=("palette", "linear"), default="palette")
    compress.add_argument("--bits", type=int, choices=(4, 6, 8), default=8)
    compress.add_argument(
        "--group-size",
        type=int,
        default=32,
        help="palette: output channels sharing one lookup table; linear: input channels per scale",
    )
    compress.add_argument(
        "--roles", nargs="+", choices=("decoder", "lm_head"), default=["decoder", "lm_head"]
    )
    compile_command = commands.add_parser(
        "compile", help="Prepare a separate host-compiled bundle for faster subsequent loads"
    )
    compile_command.add_argument("--source", type=Path, required=True)
    compile_command.add_argument("--output", type=Path, required=True)
    build_draft = commands.add_parser(
        "build-draft",
        help="Build the optional GPU-draft bundle (0.6B checkpoint + verify head) for a target bundle",
    )
    build_draft.add_argument("--target", type=Path, default=Path("artifacts/qwen3-asr-1.7b"))
    build_draft.add_argument("--source", type=Path, default=Path("artifacts/source/Qwen3-ASR-1.7B"))
    build_draft.add_argument(
        "--draft-source",
        type=Path,
        default=None,
        help="An already downloaded Qwen3-ASR-0.6B checkpoint; downloaded when omitted",
    )
    build_draft.add_argument("--output", type=Path, default=Path("artifacts/qwen3-asr-1.7b-draft"))
    inspect = commands.add_parser("inspect", help="Report anticipated compute placement")
    inspect.add_argument("model", type=Path)
    inspect.add_argument(
        "--compute-units", choices=("cpu_and_ne", "cpu_only"), default="cpu_and_ne"
    )
    transcribe = commands.add_parser(
        "transcribe",
        help="Transcribe one audio file with a local bundle (default artifacts/qwen3-asr-1.7b)",
    )
    transcribe.add_argument("audio", type=Path)
    transcribe.add_argument("--model-dir", type=Path, default=None)
    transcribe.add_argument("--profile", choices=tuple(PROFILES), default="general")
    transcribe.add_argument("--language", default="auto")
    transcribe.add_argument("--max-new-tokens", type=int, default=None)
    transcribe.add_argument(
        "--draft-dir", type=Path, default=None, help="Use the GPU-draft bundle at this path"
    )
    args = parser.parse_args(argv)
    if args.command == "download":
        from .conversion.build import download_source

        result = download_source(args.output, revision=args.revision)
    elif args.command == "build":
        from .conversion.build import build_bundle

        result = build_bundle(
            args.source,
            args.output
            or Path(
                "artifacts/qwen3-asr-1.7b-short-dictation"
                if args.profile == "short-dictation"
                else "artifacts/qwen3-asr-1.7b"
            ),
            cache_length=args.cache_length,
            reuse_encoder=args.reuse_encoder,
            token_batch_size=args.token_batch_size,
            layers_per_partition=args.layers_per_partition,
            profile=args.profile,
        )
    elif args.command == "compress":
        from .conversion.compress import compress_bundle, validate_settings

        try:
            validate_settings(args.scheme, args.bits, args.group_size)
        except ValueError as error:
            parser.error(str(error))
        result = compress_bundle(
            args.source,
            args.output,
            scheme=args.scheme,
            bits=args.bits,
            group_size=args.group_size,
            roles=tuple(args.roles),
        )
    elif args.command == "build-draft":
        from .conversion.draft import build_draft_bundle

        result = build_draft_bundle(
            args.target, args.source, args.output, draft_source=args.draft_source
        )
    elif args.command == "compile":
        from .compiled import compile_bundle

        result = compile_bundle(args.source, args.output)
    elif args.command == "inspect":
        from .diagnostics import inspect_compute_plan

        result = inspect_compute_plan(args.model, args.compute_units)
    else:
        from standard_asr.contract.exceptions import StructuredError
        from standard_asr.engine import RuntimeParams

        from .plugin import Qwen3ASREngine

        settings = {"profile": args.profile, "draft_dir": args.draft_dir}
        if args.model_dir is not None:
            settings["model_dir"] = args.model_dir
        if args.max_new_tokens is not None:
            settings["max_new_tokens"] = args.max_new_tokens
        engine = Qwen3ASREngine(**settings)
        try:
            result = engine.transcribe(
                args.audio, RuntimeParams(language=args.language)
            ).model_dump(mode="json")
        except StructuredError as error:
            # Framework errors carry the remedy; a traceback would hide it.
            print(f"error: {error}", file=sys.stderr)
            hint = getattr(error, "hint", None)
            if hint:
                print(f"hint: {hint}", file=sys.stderr)
            return 2
        finally:
            engine.close()
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
