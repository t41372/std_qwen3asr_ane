"""Four post-hoc precision controls; official full FP16 model on CPU.

Each case runs in its own process with a 180-second deadline, four PyTorch
threads, and a 256-token generation budget. Official RMSNorm implementations
retain their internal FP32 accumulation. This diagnostic is not a quality gate.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IDS = [f"fleurs-cmn_hans_cn-test-{number}" for number in (861, 331, 192, 554)]


def measured(path):
    return {
        row["id"]: row
        for row in map(json.loads, Path(path).read_text().splitlines())
        if row.get("phase") == "measured" and row.get("repeat") == 0
    }


def write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


def worker(args):
    import torch
    from evaluate import audio_samples, normalize, score
    from qwen_asr import Qwen3ASRModel
    from std_qwen3asr_ane.languages import LANGUAGE_NAMES

    baseline = measured(args.baseline)[args.worker_id]
    candidate = measured(args.candidate)[args.worker_id]
    torch.set_num_threads(4)
    torch.manual_seed(20260912)
    start = time.monotonic()
    model = Qwen3ASRModel.from_pretrained(
        str(args.source_dir),
        dtype=torch.float16,
        device_map="cpu",
        attn_implementation="eager",
        max_inference_batch_size=1,
        max_new_tokens=256,
        local_files_only=True,
    )
    model.model.eval()
    load_seconds = time.monotonic() - start
    numerics = {}

    def observe(name, tensor):
        tensor = tensor.detach()
        row = numerics.setdefault(
            name,
            {
                "calls": 0,
                "max_absolute_finite": 0.0,
                "nan_count": 0,
                "infinity_count": 0,
            },
        )
        row["calls"] += 1
        row["nan_count"] += int(torch.isnan(tensor).sum())
        row["infinity_count"] += int(torch.isinf(tensor).sum())
        finite = tensor[torch.isfinite(tensor)]
        if finite.numel():
            row["max_absolute_finite"] = max(
                row["max_absolute_finite"], float(finite.abs().max())
            )

    def layer_hook(index):
        def hook(module, inputs, output):
            observe(f"decoder_layer_{index}", output)

        return hook

    handles = [
        layer.register_forward_hook(layer_hook(index))
        for index, layer in enumerate(model.model.thinker.model.layers)
    ]
    handles.append(
        model.model.thinker.audio_tower.register_forward_hook(
            lambda module, inputs, output: observe(
                "audio_encoder", output.last_hidden_state
            )
        )
    )
    audio, digest = audio_samples(Path(baseline["audio_path"]))
    if digest != baseline["audio_sha256"]:
        raise ValueError("Diagnostic audio differs from recorded baseline")
    language = baseline["forced_language"]
    language = LANGUAGE_NAMES.get(language, language) if language else None
    begin = time.monotonic()
    with torch.inference_mode():
        result = model.transcribe(audio=(audio, 16000), language=language)[0]
    inference_seconds = time.monotonic() - begin
    for handle in handles:
        handle.remove()
    normalized = normalize(result.text, characters=True)
    row = {
        "id": args.worker_id,
        "status": "pass",
        "reference": baseline["reference"],
        "audio_seconds": len(audio) / 16000,
        "audio_sha256": digest,
        "official_fp32": {
            "hypothesis": baseline["hypothesis"],
            "cer": baseline["scores"]["cer"],
        },
        "candidate_ane": {
            "hypothesis": candidate["hypothesis"],
            "cer": candidate["scores"]["cer"],
        },
        "official_fp16": {
            "hypothesis": result.text,
            "cer": score(baseline["reference"], result.text)["cer"],
            "equals_fp32_normalized": normalized
            == normalize(baseline["hypothesis"], characters=True),
            "equals_ane_normalized": normalized
            == normalize(candidate["hypothesis"], characters=True),
        },
        "numerics": numerics,
        "model_load_seconds": load_seconds,
        "inference_seconds": inference_seconds,
        "scope": "Entire official model weights FP16 on CPU; official RMSNorm internally accumulates in FP32",
    }
    write_json(args.worker_output, row)
    print(
        json.dumps(
            {"id": args.worker_id, "fp16": row["official_fp16"]}, ensure_ascii=False
        ),
        flush=True,
    )


def main():
    corpus = ROOT / "artifacts/evaluation/fleurs-zh-balanced-100"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir", type=Path, default=ROOT / "artifacts/source/Qwen3-ASR-1.7B"
    )
    parser.add_argument("--baseline", type=Path, default=corpus / "official.jsonl")
    parser.add_argument(
        "--candidate", type=Path, default=corpus / "coreml-t16-quality.jsonl"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts/diagnostics/official-fp16-controls",
    )
    parser.add_argument("--worker-id", choices=IDS)
    parser.add_argument("--worker-output", type=Path)
    args = parser.parse_args()
    if args.worker_id:
        worker(args)
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "evidence_kind": "post_hoc_official_fp16_diagnostic",
        "heldout_quality_gate": False,
        "case_timeout_seconds": 180,
        "torch_threads": 4,
        "max_new_tokens": 256,
        "hypothesis": "If official CPU FP16 reproduces ANE regressions, generic precision sensitivity is implicated; otherwise investigate ANE flush/fusion and rewritten computation separately",
        "cases": [],
    }
    for sample_id in IDS:
        slug = sample_id.rsplit("-", 1)[-1]
        output = args.output_dir / f"{slug}.json"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker-id",
            sample_id,
            "--worker-output",
            str(output),
            "--source-dir",
            str(args.source_dir),
            "--baseline",
            str(args.baseline),
            "--candidate",
            str(args.candidate),
        ]
        start = time.monotonic()
        with (
            (args.output_dir / f"{slug}.stdout.log").open("wb") as stdout,
            (args.output_dir / f"{slug}.stderr.log").open("wb") as stderr,
        ):
            process = subprocess.Popen(
                command, stdout=stdout, stderr=stderr, start_new_session=True
            )
            try:
                code = process.wait(timeout=180)
                row = (
                    json.loads(output.read_text())
                    if code == 0 and output.exists()
                    else {"id": sample_id, "status": "fail", "returncode": code}
                )
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                row = {
                    "id": sample_id,
                    "status": "fail",
                    "reason": "180_second_timeout",
                }
        row["subprocess_seconds"] = time.monotonic() - start
        report["cases"].append(row)
        write_json(args.output_dir / "summary.json", report)
        print(
            json.dumps(
                {
                    "id": sample_id,
                    "status": row["status"],
                    "official_fp16": row.get("official_fp16"),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    valid = [row for row in report["cases"] if row["status"] == "pass"]
    report["diagnostic_totals"] = {
        label: sum(row[label]["cer"]["errors"] for row in valid)
        for label in ("official_fp32", "official_fp16", "candidate_ane")
    }
    report["all_intermediates_finite"] = (
        all(
            value["nan_count"] == 0 and value["infinity_count"] == 0
            for row in valid
            for value in row["numerics"].values()
        )
        if len(valid) == len(IDS)
        else None
    )
    report["interpretation"] = {
        "official_fp16_preserves_all_four_fp32_transcripts": len(valid) == len(IDS)
        and all(row["official_fp16"]["equals_fp32_normalized"] for row in valid),
        "direction": "On these post-hoc controls, generic official FP16 does not explain the ANE transcript changes. This does not prove FP16 non-inferiority on a held-out corpus.",
        "next_hypotheses": [
            "ANE flush-to-zero effects on subnormal weights and activations",
            "Rewritten FP16 normalization versus official internal FP32 reductions",
            "Fused projection/accumulation and cancellation in the final decoder residual layer",
            "Control host mel preprocessing and encoder embeddings independently before assigning decoder blame",
        ],
    }
    write_json(args.output_dir / "summary.json", report)
    print(
        json.dumps(
            {
                "totals": report["diagnostic_totals"],
                "all_intermediates_finite": report["all_intermediates_finite"],
            }
        ),
        flush=True,
    )
    return 0 if len(valid) == len(IDS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
