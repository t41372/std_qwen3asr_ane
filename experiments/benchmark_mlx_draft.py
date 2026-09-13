"""Neural Engine target + GPU draft: exact speculative decoding, end to end.

Runs the package's own speculative path (``CoreMLRuntime.transcribe_speculative``
with a ``DraftRuntime``) next to the serial path in the same process on the
same audio, and requires token-identical output; a record is invalid
otherwise. Timings include drafting, verification, both prefills and the MLX
audio encoder.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from evaluate import audio_samples, manifest_rows
from std_qwen3asr_ane.bundle import digest as file_digest
from std_qwen3asr_ane.draft import DraftRuntime
from std_qwen3asr_ane.runtime import CoreMLRuntime


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument(
        "--draft-bundle",
        type=Path,
        default=Path("artifacts/qwen3-asr-1.7b-draft"),
        help="bundle from qwen3-asr-ane build-draft",
    )
    parser.add_argument("--draft-bits", type=int, choices=(4, 8), default=4)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lookahead", type=int, default=15)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--skip-serial",
        action="store_true",
        help="run only the speculative path (for tracing); parity is not checked",
    )
    args = parser.parse_args()
    if args.output.exists() or args.repeats < 1:
        parser.error("Use a fresh output path and positive repeats")
    import coremltools as ct
    import mlx.core
    import mlx_audio

    started = perf_counter()
    target = CoreMLRuntime(args.target)
    target_loaded = perf_counter()
    draft = DraftRuntime(args.draft_bundle, target, quantize_bits=args.draft_bits)
    draft_loaded = perf_counter()
    provenance = {
        "target_manifest_sha256": file_digest(args.target / "manifest.json"),
        "draft_bundle": str(args.draft_bundle),
        "draft_manifest_sha256": file_digest(args.draft_bundle / "manifest.json"),
        "draft_manifest": draft.manifest,
        "manifest": str(args.manifest),
        "manifest_sha256": file_digest(args.manifest),
        "mlx": mlx.core.__version__,
        "mlx_audio": getattr(mlx_audio, "__version__", None),
        "coremltools": ct.__version__,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.output.open("w") as output:
            for row in manifest_rows(args.manifest):
                samples, digest = audio_samples(Path(row["audio_path"]))
                for repeat in range(-1, args.repeats):
                    serial = (
                        None
                        if args.skip_serial
                        else target.transcribe(
                            samples, language=None, max_new_tokens=256
                        )
                    )
                    draft.model.step_calls, draft.model.step_seconds = 0, 0.0
                    begin = perf_counter()
                    result = target.transcribe_speculative(
                        samples,
                        draft,
                        language=None,
                        max_new_tokens=256,
                        lookahead=args.lookahead,
                    )
                    end = perf_counter()
                    record = {
                        "id": row["id"],
                        "repeat": repeat,
                        "audio_sha256": digest,
                        "audio_seconds": len(samples) / 16000,
                        "target": str(args.target),
                        "draft_bits": args.draft_bits,
                        "lookahead": args.lookahead,
                        "provenance": provenance,
                        "load_seconds": {
                            "target": target_loaded - started,
                            "draft_and_verify_head": draft_loaded - target_loaded,
                        },
                        "phase": "measured" if repeat >= 0 else "warmup",
                        "seconds": end - begin,
                        "serial_seconds": None
                        if serial is None
                        else serial.timings["total_seconds"],
                        "timings": result.timings,
                        "draft_step_calls": draft.model.step_calls,
                        "draft_step_seconds": draft.model.step_seconds,
                        "exact_token_parity": None
                        if serial is None
                        else result.token_ids == serial.token_ids,
                        "text": result.text,
                        "serial_tokens": None
                        if serial is None
                        else list(serial.token_ids),
                        "speculative_tokens": list(result.token_ids),
                    }
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    output.flush()
                    print(
                        json.dumps(
                            {
                                k: v
                                for k, v in record.items()
                                if k
                                not in (
                                    "serial_tokens",
                                    "speculative_tokens",
                                    "text",
                                    "provenance",
                                )
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                    if record["exact_token_parity"] is False:
                        raise AssertionError(
                            "Speculative output differs from serial ANE greedy"
                        )
    finally:
        draft.close()
        target.close()


if __name__ == "__main__":
    main()
