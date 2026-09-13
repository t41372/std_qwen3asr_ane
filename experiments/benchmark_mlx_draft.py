"""ANE target + MLX GPU draft: exact greedy speculative decoding, end to end.

Hypothesis under test: a small model on the GPU proposes tokens cheaply while
the 1.7B model on the Neural Engine verifies up to 15 of them per pass. Output
tokens are required to equal serial ANE greedy decoding; the record is invalid
otherwise. Timings include drafting, verification, both prefills and the MLX
audio encoder; the serial baseline runs in the same process on the same audio.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from benchmark_speculative import DecoderCursor
from evaluate import audio_samples, manifest_rows
from mlx_draft import MLXDraft
from std_qwen3asr_ane.bundle import digest as file_digest
from std_qwen3asr_ane.runtime import CoreMLRuntime, PersistentInputModel
from std_qwen3asr_ane.speculative import greedy_speculative_decode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument(
        "--draft-dir", type=Path, default=Path("artifacts/source/Qwen3-ASR-0.6B")
    )
    parser.add_argument("--draft-bits", type=int, choices=(4, 8), default=None)
    parser.add_argument("--verify-head", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lookahead", type=int, default=7)
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

    started = perf_counter()
    target = CoreMLRuntime(args.target)
    target_loaded = perf_counter()
    head = PersistentInputModel(
        ct.models.MLModel(
            str(args.verify_head), compute_units=ct.ComputeUnit.CPU_AND_NE
        )
    )
    head_loaded = perf_counter()
    draft = MLXDraft(args.draft_dir, quantize_bits=args.draft_bits)
    import mlx.core
    import mlx_audio

    source_file = args.draft_dir / "source.json"
    provenance = {
        "target_manifest_sha256": file_digest(args.target / "manifest.json"),
        "verify_head": str(args.verify_head),
        "verify_head_sha256": {
            str(child.relative_to(args.verify_head)): file_digest(child)
            for child in sorted(args.verify_head.rglob("*"))
            if child.is_file()
        },
        "manifest": str(args.manifest),
        "manifest_sha256": file_digest(args.manifest),
        "draft_source": json.loads(source_file.read_text())
        if source_file.exists()
        else None,
        "mlx": mlx.core.__version__,
        "mlx_audio": getattr(mlx_audio, "__version__", None),
        "coremltools": ct.__version__,
    }
    if not 0 <= args.lookahead < target.token_batch_size:
        parser.error("lookahead must leave room for the held token in the target block")
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
                    begin = perf_counter()
                    prompt = target.prepare_prompt(
                        samples, language=None, max_new_tokens=256
                    )
                    target_ready = perf_counter()
                    draft_timings = draft.prepare(samples, list(prompt.token_ids))
                    draft_ready = perf_counter()
                    draft.step_calls, draft.step_seconds = 0, 0.0
                    result = greedy_speculative_decode(
                        DecoderCursor(target, prompt, head),
                        draft,
                        prompt.hidden,
                        target_position=len(prompt.token_ids),
                        draft_position=len(prompt.token_ids),
                        eos_token_ids=frozenset(target.eos_token_ids),
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
                        "draft_dir": str(args.draft_dir),
                        "draft_bits": args.draft_bits,
                        "lookahead": args.lookahead,
                        "provenance": provenance,
                        "load_seconds": {
                            "target": target_loaded - started,
                            "verify_head": head_loaded - target_loaded,
                            "draft": draft.load_seconds,
                        },
                        "phase": "measured" if repeat >= 0 else "warmup",
                        "seconds": end - begin,
                        "serial_seconds": None
                        if serial is None
                        else serial.timings["total_seconds"],
                        "speculative_seconds": end - begin,
                        "target_prepare_seconds": target_ready - begin,
                        "draft_prepare_seconds": draft_ready - target_ready,
                        "draft_prepare_breakdown": draft_timings,
                        "generation_seconds": end - draft_ready,
                        "draft_step_calls": draft.step_calls,
                        "draft_step_seconds": draft.step_seconds,
                        "exact_token_parity": None
                        if serial is None
                        else result.token_ids == serial.token_ids,
                        "text": target.tokenizer.decode(
                            list(result.token_ids), skip_special_tokens=True
                        ),
                        "serial_tokens": None
                        if serial is None
                        else list(serial.token_ids),
                        "speculative": asdict(result),
                    }
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    output.flush()
                    print(
                        json.dumps(
                            {
                                k: v
                                for k, v in record.items()
                                if k not in ("serial_tokens", "text", "provenance")
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
        prompt = None
        head.close()
        target.close()


if __name__ == "__main__":
    main()
