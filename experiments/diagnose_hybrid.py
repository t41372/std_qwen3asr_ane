"""Four post-hoc hybrid controls isolate encoder and decoder arithmetic.

An external supervisor watches flushed call-start/end events and kills the
worker process group after a 180-second call deadline, including native hangs.
Models are never converted. Both hybrids use the official source weights and
the explicit stable-SiLU Core ML bundle; no alternative backend fallback.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import MethodType

ROOT = Path(__file__).resolve().parents[1]
IDS = [f"fleurs-cmn_hans_cn-test-{number}" for number in (861, 331, 192, 554)]


def read_rows(path):
    rows = {}
    # A concurrent evaluator may still be appending; only consume complete lines.
    for line in Path(path).read_text().splitlines(keepends=True):
        if not line.endswith("\n"):
            continue
        row = json.loads(line)
        if row.get("phase") == "measured" and row.get("repeat") == 0:
            rows[row["id"]] = row
    return rows


def write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


def digest(value):
    return hashlib.sha256(value).hexdigest()


def worker(args):
    import numpy as np
    import torch
    from compare_encoder_semantics import WindowAttentionProbe
    from evaluate import audio_samples, normalize, score
    from qwen_asr import Qwen3ASRModel
    from std_qwen3asr_ane.audio import log_mel_spectrogram
    from std_qwen3asr_ane.languages import LANGUAGE_NAMES
    from std_qwen3asr_ane.runtime import CoreMLRuntime
    from transformers.modeling_outputs import BaseModelOutput

    snapshot = json.loads((args.output_dir / "input-snapshot.json").read_text())
    previous = snapshot["cases"][args.worker_id]
    baseline, candidate = previous["official_fp32"], previous["stable_ane"]
    directory = args.output_dir / args.worker_id.rsplit("-", 1)[-1]
    events = (directory / "calls.jsonl").open("a", buffering=1)
    durations = []

    def emit(row):
        events.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        events.flush()
        os.fsync(events.fileno())

    def call(label, operation):
        start = time.monotonic()
        emit({"event": "call_start", "call": label, "monotonic": start})
        try:
            result = operation()
        except Exception as error:
            emit(
                {
                    "event": "call_error",
                    "call": label,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            raise
        seconds = time.monotonic() - start
        durations.append({"call": label, "seconds": seconds})
        emit({"event": "call_end", "call": label, "seconds": seconds})
        return result

    def comparison(left, right):
        left, right = (
            np.asarray(left, dtype=np.float64),
            np.asarray(right, dtype=np.float64),
        )
        if left.shape != right.shape:
            return {
                "shape_match": False,
                "left_shape": list(left.shape),
                "right_shape": list(right.shape),
            }
        flat_left, flat_right = left.reshape(-1), right.reshape(-1)
        difference = left - right
        denominator = max(np.linalg.norm(flat_left) * np.linalg.norm(flat_right), 1e-30)
        return {
            "shape_match": True,
            "shape": list(left.shape),
            "max_absolute_error": float(np.abs(difference).max()),
            "rmse": float(np.sqrt(np.mean(difference**2))),
            "relative_l2_error": float(
                np.linalg.norm(difference) / max(np.linalg.norm(right), 1e-30)
            ),
            "cosine_similarity": float(np.dot(flat_left, flat_right) / denominator),
        }

    torch.set_num_threads(4)
    torch.manual_seed(20260912)
    official = call(
        "load_official_fp32",
        lambda: Qwen3ASRModel.from_pretrained(
            str(args.source_dir),
            dtype=torch.float32,
            device_map="cpu",
            attn_implementation="eager",
            max_inference_batch_size=1,
            max_new_tokens=256,
            local_files_only=True,
        ),
    )
    official.model.eval()
    core = call(
        "load_coreml_stable_silu",
        lambda: CoreMLRuntime(args.model_dir, compute_units="cpu_and_ne"),
    )
    tower = official.model.thinker.audio_tower
    original_forward, original_encode = tower.forward, core._encode_audio
    probe = WindowAttentionProbe(tower)
    probe.reset("window")
    caches = {"official": {}, "ane": {}}

    def features_key(features):
        return digest(np.asarray(features, dtype=np.float32).tobytes())

    def official_encode(features):
        array = np.array(features, dtype=np.float32, copy=True, order="C")
        key = features_key(array)
        if key not in caches["official"]:
            with torch.inference_mode():
                encoded = original_forward(
                    torch.from_numpy(array),
                    feature_lens=torch.tensor([array.shape[1]], dtype=torch.long),
                ).last_hidden_state
            caches["official"][key] = (
                encoded.T[None, :, None].contiguous().numpy().copy()
            )
        return caches["official"][key].copy()

    def ane_encode(features):
        array = np.array(features, dtype=np.float32, copy=True, order="C")
        key = features_key(array)
        if key not in caches["ane"]:
            caches["ane"][key] = np.array(
                original_encode(array), dtype=np.float32, copy=True, order="C"
            )
        return caches["ane"][key].copy()

    audio, audio_hash = call(
        "read_audio", lambda: audio_samples(Path(baseline["audio_path"]))
    )
    if (
        audio_hash != baseline["audio_sha256"]
        or audio_hash != candidate["audio_sha256"]
    ):
        raise ValueError("Diagnostic audio content differs from saved evaluations")
    language = baseline["forced_language"]
    official_language = LANGUAGE_NAMES.get(language, language) if language else None
    prompt = official._build_text_prompt(context="", force_language=official_language)
    inputs = call(
        "official_processor",
        lambda: official.processor(
            text=[prompt], audio=[audio], return_tensors="pt", padding=True
        ),
    )
    frames = int(inputs["feature_attention_mask"].sum())
    official_features = inputs["input_features"][0, :, :frames].float().numpy().copy()
    host_features = call(
        "host_mel", lambda: log_mel_spectrogram(audio, core.mel_filters)
    )
    embedding_pairs = {}
    for frontend, features in (
        ("official_mel", official_features),
        ("host_mel", host_features),
    ):
        for encoder, function in (
            ("official_window_fp32", official_encode),
            ("ane", ane_encode),
        ):
            embedding_pairs[f"{frontend}_{encoder}"] = call(
                f"encode_{frontend}_{encoder}",
                lambda function=function, features=features: function(features),
            )
    report = {
        "id": args.worker_id,
        "evidence_kind": "post_hoc_encoder_decoder_hybrid_diagnostic",
        "heldout_quality_gate": False,
        "timings_are_benchmarks": False,
        "reference": baseline["reference"],
        "audio_sha256": audio_hash,
        "audio_seconds": len(audio) / 16000,
        "bundle_manifest_sha256": snapshot["bundle_manifest_sha256"],
        "source_config_sha256": snapshot["source_config_sha256"],
        "official_encoder_attention": "explicit window mask from upstream cu_seqlens",
        "mel_comparison_host_vs_official": comparison(host_features, official_features),
        "encoder_embedding_comparisons": {
            "ane_vs_official_same_official_mel": comparison(
                embedding_pairs["official_mel_ane"],
                embedding_pairs["official_mel_official_window_fp32"],
            ),
            "ane_vs_official_same_host_mel": comparison(
                embedding_pairs["host_mel_ane"],
                embedding_pairs["host_mel_official_window_fp32"],
            ),
            "official_encoder_host_vs_official_mel": comparison(
                embedding_pairs["host_mel_official_window_fp32"],
                embedding_pairs["official_mel_official_window_fp32"],
            ),
        },
        "existing_official_fp32": {
            "hypothesis": baseline["hypothesis"],
            "cer": baseline["scores"]["cer"],
        },
        "existing_stable_ane": {
            "hypothesis": candidate["hypothesis"],
            "cer": candidate["scores"]["cer"],
        },
        "hybrids": {},
    }
    np.savez(directory / "encoder-embeddings.npz", **embedding_pairs)
    write_json(directory / "result.json", report)

    def ane_tower_forward(
        module, input_features, feature_lens=None, aftercnn_lens=None
    ):
        del module, aftercnn_lens
        array = input_features.detach().float().cpu().numpy()
        if feature_lens is not None:
            array = array[:, : int(feature_lens[0])]
        encoded = ane_encode(array)
        return BaseModelOutput(
            last_hidden_state=torch.from_numpy(encoded[0, :, 0, :].T.copy())
        )

    def record_hybrid(name, result):
        text = result.text
        row = {
            "hypothesis": text,
            "cer": score(baseline["reference"], text)["cer"],
            "equals_official_fp32_normalized": normalize(text, characters=True)
            == normalize(baseline["hypothesis"], characters=True),
            "equals_stable_ane_normalized": normalize(text, characters=True)
            == normalize(candidate["hypothesis"], characters=True),
        }
        report["hybrids"][name] = row
        emit({"event": "hybrid_result", "hybrid": name, **row})
        write_json(directory / "result.json", report)
        print(
            json.dumps(
                {"id": args.worker_id, "hybrid": name, **row}, ensure_ascii=False
            ),
            flush=True,
        )

    with torch.inference_mode():
        tower.forward = MethodType(ane_tower_forward, tower)
        try:
            result = call(
                "hybrid_ane_encoder_official_fp32_decoder",
                lambda: official.transcribe(
                    audio=(audio, 16000), language=official_language
                )[0],
            )
            record_hybrid("ane_encoder_official_fp32_decoder", result)
        finally:
            tower.forward = original_forward
        core._encode_audio = official_encode
        try:
            result = call(
                "hybrid_official_window_fp32_encoder_ane_decoder",
                lambda: core.transcribe(audio, language=language, max_new_tokens=256),
            )
            record_hybrid("official_window_fp32_encoder_ane_decoder", result)
        finally:
            core._encode_audio = original_encode
    probe.close()
    call("close_coreml", lambda: core.close(timeout=5))
    report["status"] = "pass"
    report["diagnostic_call_seconds"] = durations
    write_json(directory / "result.json", report)
    events.close()


def supervise(args, sample_id):
    directory = args.output_dir / sample_id.rsplit("-", 1)[-1]
    directory.mkdir()
    progress = directory / "calls.jsonl"
    progress.touch()
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-id",
        sample_id,
        "--model-dir",
        str(args.model_dir),
        "--source-dir",
        str(args.source_dir),
        "--output-dir",
        str(args.output_dir),
    ]
    deadline, live_call = time.monotonic() + args.timeout, "worker_startup"
    timeout_reason = None
    with (
        (directory / "stdout.log").open("wb") as out,
        (directory / "stderr.log").open("wb") as err,
        progress.open() as event_stream,
    ):
        process = subprocess.Popen(
            command, cwd=ROOT, stdout=out, stderr=err, start_new_session=True
        )
        while process.poll() is None:
            while True:
                offset = event_stream.tell()
                line = event_stream.readline()
                if not line:
                    break
                if not line.endswith("\n"):
                    event_stream.seek(offset)
                    break
                event = json.loads(line)
                if event["event"] == "call_start":
                    live_call = event["call"]
                    deadline = event["monotonic"] + args.timeout
                elif event["event"] in ("call_end", "call_error"):
                    live_call = "between_calls"
                    deadline = time.monotonic() + args.timeout
            if time.monotonic() > deadline:
                timeout_reason = (
                    f"Hard timeout during {live_call}: {args.timeout} seconds"
                )
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                break
            time.sleep(0.1)
    result = (
        json.loads((directory / "result.json").read_text())
        if (directory / "result.json").exists()
        else {"id": sample_id}
    )
    result["worker_returncode"] = process.returncode
    if timeout_reason or process.returncode:
        result.update(
            status="fail", reason=timeout_reason or "worker_failed; inspect stderr.log"
        )
    write_json(directory / "supervised-result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir", type=Path, default=ROOT / "artifacts/qwen3-asr-1.7b-stable-silu"
    )
    parser.add_argument(
        "--source-dir", type=Path, default=ROOT / "artifacts/source/Qwen3-ASR-1.7B"
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=ROOT / "artifacts/evaluation/fleurs-zh-balanced-100/official.jsonl",
    )
    parser.add_argument(
        "--candidate",
        type=Path,
        default=ROOT / "artifacts/evaluation/silu-validation/coreml-stable-silu.jsonl",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "artifacts/diagnostics/hybrid"
    )
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--worker-id", choices=IDS)
    args = parser.parse_args()
    if args.worker_id:
        worker(args)
        return 0
    if args.timeout <= 0:
        parser.error("timeout must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "input-snapshot.json").exists():
        parser.error("Diagnostic output exists; choose a new --output-dir")
    baseline, candidate = read_rows(args.baseline), read_rows(args.candidate)
    cases = {}
    for sample_id in IDS:
        left, right = baseline[sample_id], candidate[sample_id]
        for field in (
            "reference",
            "audio_sha256",
            "forced_language",
            "language_mode",
            "normalizer",
            "max_new_tokens",
        ):
            if left[field] != right[field]:
                raise ValueError(f"Saved evaluations differ on {sample_id}: {field}")
        if Path(right["model_dir"]).resolve() != args.model_dir.resolve():
            raise ValueError("Candidate report belongs to a different bundle")
        cases[sample_id] = {"official_fp32": left, "stable_ane": right}
    snapshot = {
        "cases": cases,
        "bundle_manifest_sha256": digest(
            (args.model_dir / "manifest.json").read_bytes()
        ),
        "source_config_sha256": digest((args.source_dir / "config.json").read_bytes()),
        "selection": "Four post-hoc precision controls: 861/331 numeric regressions and short 192/554 controls",
    }
    write_json(args.output_dir / "input-snapshot.json", snapshot)
    report = {
        "evidence_kind": "post_hoc_encoder_decoder_hybrid_diagnostic",
        "heldout_quality_gate": False,
        "timings_are_benchmarks": False,
        "hard_call_timeout_seconds": args.timeout,
        "cases": [],
    }
    for sample_id in IDS:
        result = supervise(args, sample_id)
        report["cases"].append(result)
        write_json(args.output_dir / "summary.json", report)
        print(
            json.dumps(
                {
                    "id": sample_id,
                    "status": result.get("status"),
                    "hybrids": result.get("hybrids"),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    return 0 if all(row.get("status") == "pass" for row in report["cases"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
