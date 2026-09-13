"""Run equal useful ASR work with independently sampled whole-machine power.

PSTR is a responsive SMC estimate, not calibrated wall-meter or per-device data.
Run paired ABBA blocks in separate processes, with no concurrent model work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

from evaluate import audio_samples, manifest_rows
from power_v2.integrate import integrate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", choices=("coreml", "mlx", "official", "specdraft"), required=True
    )
    parser.add_argument("--draft-dir", type=Path, help="specdraft: MLX draft checkpoint")
    parser.add_argument("--draft-bits", type=int, choices=(4, 8), default=None)
    parser.add_argument("--verify-head", type=Path, help="specdraft: T16 compact head")
    parser.add_argument("--lookahead", type=int, default=15)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=60)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="float32")
    args = parser.parse_args()
    if args.repeats < 1 or args.output.exists():
        parser.error("Require positive repeats and a fresh output directory")
    args.output.mkdir(parents=True)
    inputs = []
    for row in manifest_rows(args.manifest):
        samples, digest = audio_samples(Path(row["audio_path"]))
        inputs.append((row, samples, digest))
    started = time.monotonic()
    close = None
    if args.backend == "mlx":
        from benchmark_mlx import configure_local_caches, load_backend

        configure_local_caches()
        predict, _ = load_backend(args.model_dir.resolve(), 20260912)

        def transcribe(samples):
            return predict(samples, None, 256)["hypothesis"]
    elif args.backend == "specdraft":
        import coremltools as ct
        from benchmark_speculative import DecoderCursor
        from mlx_draft import MLXDraft
        from std_qwen3asr_ane.runtime import (
            CoreMLRuntime,
            PersistentInputModel,
            parse_output,
        )
        from std_qwen3asr_ane.speculative import greedy_speculative_decode

        runtime = CoreMLRuntime(args.model_dir)
        head = PersistentInputModel(
            ct.models.MLModel(str(args.verify_head), compute_units=ct.ComputeUnit.CPU_AND_NE)
        )
        draft_model = MLXDraft(args.draft_dir, quantize_bits=args.draft_bits)

        def close():
            head.close()
            runtime.close()

        def transcribe(samples):
            prompt = runtime.prepare_prompt(samples, language=None, max_new_tokens=256)
            draft_model.prepare(samples, list(prompt.token_ids))
            result = greedy_speculative_decode(
                DecoderCursor(runtime, prompt, head),
                draft_model,
                prompt.hidden,
                target_position=len(prompt.token_ids),
                draft_position=len(prompt.token_ids),
                eos_token_ids=frozenset(runtime.eos_token_ids),
                max_new_tokens=256,
                lookahead=args.lookahead,
            )
            decoded = runtime.tokenizer.decode(list(result.token_ids), skip_special_tokens=True)
            return parse_output(decoded, None)[0]
    elif args.backend == "official":
        import torch
        from qwen_asr import Qwen3ASRModel

        torch.manual_seed(20260912)
        model = Qwen3ASRModel.from_pretrained(
            str(args.model_dir.resolve()),
            dtype=getattr(torch, args.dtype),
            device_map=args.device,
            attn_implementation="sdpa" if args.device == "mps" else "eager",
            max_inference_batch_size=1,
            max_new_tokens=256,
            local_files_only=True,
        )
        model.model.eval()

        def transcribe(samples):
            with torch.inference_mode():
                text = model.transcribe(audio=(samples, 16000), language=None)[0].text
            if args.device == "mps":
                torch.mps.synchronize()
            return text
    else:
        from std_qwen3asr_ane.runtime import CoreMLRuntime

        runtime = CoreMLRuntime(args.model_dir)
        close = runtime.close

        def transcribe(samples):
            return runtime.transcribe(samples, language=None, max_new_tokens=256).text

    load_seconds = time.monotonic() - started
    collector = None
    report = {
        "backend": args.backend,
        "device": args.device if args.backend == "official" else None,
        "dtype": args.dtype if args.backend == "official" else None,
        "draft": {
            "dir": str(args.draft_dir),
            "bits": args.draft_bits,
            "lookahead": args.lookahead,
            "verify_head": str(args.verify_head),
        }
        if args.backend == "specdraft"
        else None,
        "model_dir": str(args.model_dir.resolve()),
        "load_seconds": load_seconds,
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "repeats": args.repeats,
    }
    try:
        expected = [transcribe(samples) for _, samples, _ in inputs]
        trace = args.output / "power.jsonl"
        with trace.open("w") as stream:
            collector = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).parent / "power_v2/whole_machine.py"),
                    "--seconds",
                    "3600",
                    "--battery-every",
                    "5",
                ],
                stdout=stream,
                stderr=(args.output / "collector.log").open("w"),
            )
            # Bracket work and observe idle after models are fully warmed. The
            # independent collector has its own interpreter/GIL and timestamps.
            time.sleep(12)
            if collector.poll() is not None:
                raise RuntimeError("Power collector stopped before workload")
            begin = time.monotonic()
            usage_begin = resource.getrusage(resource.RUSAGE_SELF)
            audio_seconds, completed = 0.0, 0
            with (args.output / "predictions.jsonl").open("w") as predictions:
                for repeat in range(args.repeats):
                    for (row, samples, digest), reference_output in zip(
                        inputs, expected, strict=True
                    ):
                        prediction_begin = time.monotonic()
                        text = transcribe(samples)
                        prediction_end = time.monotonic()
                        if text != reference_output:
                            raise RuntimeError(
                                "Repeated workload changed its output; benchmark invalid"
                            )
                        record = {
                            "id": row["id"],
                            "repeat": repeat,
                            "start_monotonic_s": prediction_begin,
                            "end_monotonic_s": prediction_end,
                            "audio_seconds": len(samples) / 16000,
                            "audio_sha256": digest,
                            "text": text,
                        }
                        predictions.write(json.dumps(record, ensure_ascii=False) + "\n")
                        predictions.flush()
                        audio_seconds += record["audio_seconds"]
                        completed += 1
                    print(
                        json.dumps({"repeat": repeat, "completed": completed}),
                        flush=True,
                    )
            end = time.monotonic()
            usage_end = resource.getrusage(resource.RUSAGE_SELF)
            time.sleep(12)
            collector.terminate()
            collector.wait(timeout=10)
        rows = [json.loads(line) for line in trace.read_text().splitlines()]
        power = integrate(rows, begin, end)
        cpu_user = usage_end.ru_utime - usage_begin.ru_utime
        cpu_system = usage_end.ru_stime - usage_begin.ru_stime
        report.update(
            start_monotonic_s=begin,
            end_monotonic_s=end,
            completed=completed,
            audio_seconds=audio_seconds,
            rtf=(end - begin) / audio_seconds,
            power=power,
            gross_j_per_audio_second=power["gross_j_estimate"] / audio_seconds,
            # This process only: excludes the Core ML/ANE daemons and kernel work
            # done on its behalf, so it bounds the Python-side host cost from below.
            process_cpu={
                "user_seconds": cpu_user,
                "system_seconds": cpu_system,
                "mean_cores_busy": (cpu_user + cpu_system) / (end - begin),
            },
        )
        # Show sensor lag sensitivity without selecting a favorable boundary.
        report["boundary_shift_j"] = {
            str(shift): integrate(rows, begin + shift, end + shift)["gross_j_estimate"]
            for shift in (-2, -1, 0, 1, 2)
        }
        print(json.dumps(report, indent=2), flush=True)
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if collector is not None and collector.poll() is None:
            collector.terminate()
            collector.wait(timeout=10)
        if close is not None:
            close()
        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
