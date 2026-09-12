"""Diagnostic selection: distinguish encoder attention semantics from precision drift.

Samples are deliberately selected AFTER viewing candidate errors, so this is
neither held-out evidence nor a quality gate. Only an instance-local pre-hook
changes audio encoder attention; all other official FP32 model behavior remains
unchanged. No reference source edits or Core ML compilation occur.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import time
from pathlib import Path

import torch
from evaluate import audio_samples, normalize, read_jsonl, score
from qwen_asr import Qwen3ASRModel
from std_qwen3asr_ane.languages import LANGUAGE_NAMES

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def measured_rows(path):
    rows = [
        row
        for row in read_jsonl(path)
        if row.get("phase") == "measured" and row.get("repeat") == 0
    ]
    return {row["id"]: row for row in rows}


def select_diagnostics(baseline, candidate):
    available = []
    for sample_id in sorted(baseline.keys() & candidate.keys()):
        left, right = baseline[sample_id], candidate[sample_id]
        if left.get("error") or right.get("error"):
            continue
        delta = right["scores"]["cer"]["errors"] - left["scores"]["cer"]["errors"]
        available.append(
            {"id": sample_id, "delta": delta, "seconds": left["audio_seconds"]}
        )
    selected = []
    chosen = set()

    def add(rows, count, reason):
        for row in rows:
            if row["id"] not in chosen:
                selected.append({**row, "selection_reason": reason})
                chosen.add(row["id"])
                count -= 1
                if count == 0:
                    break

    add(
        sorted(
            (row for row in available if row["delta"] > 0 and row["seconds"] > 8),
            key=lambda row: (-row["delta"], row["id"]),
        ),
        6,
        "largest multi-window CER regressions",
    )
    add(
        sorted(
            (row for row in available if row["delta"] > 0 and row["seconds"] <= 8),
            key=lambda row: (-row["delta"], row["id"]),
        ),
        1,
        "single-window regression control",
    )
    add(
        sorted(
            (row for row in available if row["delta"] < 0 and row["seconds"] <= 8),
            key=lambda row: (row["delta"], row["id"]),
        ),
        1,
        "single-window improvement control",
    )
    add(
        sorted(
            (row for row in available if row["delta"] < 0 and row["seconds"] > 8),
            key=lambda row: (row["delta"], row["id"]),
        ),
        1,
        "multi-window improvement contrast",
    )
    add(
        sorted(
            (row for row in available if row["delta"] == 0 and row["seconds"] > 8),
            key=lambda row: row["id"],
        ),
        1,
        "multi-window equal-error control",
    )
    return selected


class WindowAttentionProbe:
    def __init__(self, tower):
        self.tower = tower
        self.mode = "global"
        self.calls = []
        self.embeddings = []
        self.handles = []
        for index, layer in enumerate(tower.layers):
            parameters = inspect.signature(layer.forward).parameters
            if not {"hidden_states", "cu_seqlens", "attention_mask"}.issubset(
                parameters
            ):
                raise TypeError("Unrecognized official audio encoder layer signature")
            self.handles.append(
                layer.register_forward_pre_hook(self.make_hook(index), with_kwargs=True)
            )
        self.handles.append(tower.register_forward_hook(self.capture_embeddings))

    def make_hook(self, index):
        def hook(module, args, kwargs):
            hidden = args[0] if args else kwargs["hidden_states"]
            lengths = args[1] if len(args) > 1 else kwargs["cu_seqlens"]
            prior_mask = args[2] if len(args) > 2 else kwargs.get("attention_mask")
            if index == 0:
                self.calls.append(
                    {
                        "tokens": hidden.shape[0],
                        "cu_seqlens": lengths.tolist(),
                        "windows": len(lengths) - 1,
                        "upstream_mask_supplied": prior_mask is not None,
                    }
                )
            if self.mode == "global":
                return None
            mask = self.tower._prepare_attention_mask(hidden, lengths)
            if mask is None:
                raise ValueError(
                    "Diagnostic requires eager attention and a real additive window mask"
                )
            if len(args) > 2:
                return (*args[:2], mask, *args[3:]), kwargs
            return args, {**kwargs, "attention_mask": mask}

        return hook

    def capture_embeddings(self, module, inputs, outputs):
        self.embeddings.append(outputs.last_hidden_state.detach().float().cpu().clone())

    def reset(self, mode):
        self.mode, self.calls, self.embeddings = mode, [], []

    def close(self):
        for handle in self.handles:
            handle.remove()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    corpus = ROOT / "artifacts/evaluation/fleurs-zh-balanced-100"
    parser.add_argument("--baseline", type=Path, default=corpus / "official.jsonl")
    parser.add_argument(
        "--candidate", type=Path, default=corpus / "coreml-t16-quality.jsonl"
    )
    parser.add_argument(
        "--source-dir", type=Path, default=ROOT / "artifacts/source/Qwen3-ASR-1.7B"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/diagnostics/encoder-semantics-fleurs.json",
    )
    parser.add_argument(
        "--ids",
        nargs="*",
        help="Explicit diagnostic IDs; otherwise apply documented post-hoc selection",
    )
    args = parser.parse_args()
    baseline, candidate = measured_rows(args.baseline), measured_rows(args.candidate)
    selected = select_diagnostics(baseline, candidate)
    if args.ids:
        selected = [
            {"id": sample_id, "selection_reason": "explicit diagnostic selection"}
            for sample_id in args.ids
        ]
    if not selected:
        raise ValueError("No diagnostic samples selected")
    for selection in selected:
        left, right = baseline[selection["id"]], candidate[selection["id"]]
        for field in (
            "reference",
            "audio_sha256",
            "language_mode",
            "forced_language",
            "max_new_tokens",
            "normalizer",
        ):
            if left.get(field) != right.get(field):
                raise ValueError(f"Input mismatch for {selection['id']}: {field}")
    config = json.loads((args.source_dir / "config.json").read_text())
    max_tokens = max(baseline[row["id"]]["max_new_tokens"] for row in selected)
    torch.set_num_threads(4)
    torch.manual_seed(20260912)
    start = time.perf_counter()
    model = Qwen3ASRModel.from_pretrained(
        str(args.source_dir.resolve()),
        dtype=torch.float32,
        device_map="cpu",
        attn_implementation="eager",
        max_inference_batch_size=1,
        max_new_tokens=max_tokens,
        local_files_only=True,
    )
    model.model.eval()
    tower = model.model.thinker.audio_tower
    probe = WindowAttentionProbe(tower)
    source_file = Path(inspect.getfile(type(tower)))
    mlx_file = (
        ROOT
        / "experiments/mlx_reference/.venv/lib/python3.12/site-packages/mlx_audio/stt/models/qwen3_asr/qwen3_asr.py"
    )
    report = {
        "schema_version": 1,
        "evidence_kind": "post_hoc_encoder_semantics_diagnostic",
        "heldout_quality_gate": False,
        "selection": selected,
        "baseline_sha256": sha256(args.baseline),
        "candidate_sha256": sha256(args.candidate),
        "source_config_sha256": sha256(args.source_dir / "config.json"),
        "official_source": {
            "path": str(source_file),
            "sha256": sha256(source_file),
            "semantics": "eager audio encoder passes cu_seqlens but omits attention_mask",
        },
        "mlx_source": {
            "path": str(mlx_file),
            "sha256": sha256(mlx_file) if mlx_file.exists() else None,
            "semantics": "AudioEncoder creates block attention mask from cu_seqlens and passes it to every layer",
            "relevant_lines": [424, 449],
        },
        "intervention": "Instance-local audio layer forward pre-hooks supply upstream _prepare_attention_mask; all weights and decoder remain FP32 official",
        "model_load_seconds": time.perf_counter() - start,
        "audio_config": {
            key: config["thinker_config"]["audio_config"][key]
            for key in ("n_window", "n_window_infer")
        },
        "corpus_existing_error_delta": sum(
            candidate[key]["scores"]["cer"]["errors"]
            - baseline[key]["scores"]["cer"]["errors"]
            for key in baseline.keys() & candidate.keys()
        ),
        "samples": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for selection in selected:
            sample_id = selection["id"]
            left, right = baseline[sample_id], candidate[sample_id]
            audio, digest = audio_samples(Path(left["audio_path"]))
            if digest != left["audio_sha256"]:
                raise ValueError(f"Audio content changed: {sample_id}")
            language = (
                LANGUAGE_NAMES.get(left["forced_language"], left["forced_language"])
                if left["forced_language"]
                else None
            )
            row = {
                "id": sample_id,
                "selection_reason": selection["selection_reason"],
                "audio_seconds": len(audio) / 16000,
                "reference": left["reference"],
                "original_report_global": {
                    "hypothesis": left["hypothesis"],
                    "cer": left["scores"]["cer"],
                },
                "candidate_coreml": {
                    "hypothesis": right["hypothesis"],
                    "cer": right["scores"]["cer"],
                },
            }
            encoded = {}
            for mode in ("global", "window"):
                probe.reset(mode)
                begin = time.perf_counter()
                result = model.transcribe(audio=(audio, 16000), language=language)[0]
                row[f"official_fp32_{mode}"] = {
                    "hypothesis": result.text,
                    "cer": score(left["reference"], result.text)["cer"],
                    "seconds": time.perf_counter() - begin,
                    "encoder_calls": probe.calls,
                    "equals_candidate_normalized": normalize(
                        result.text, characters=True
                    )
                    == normalize(right["hypothesis"], characters=True),
                    "equals_original_global_normalized": normalize(
                        result.text, characters=True
                    )
                    == normalize(left["hypothesis"], characters=True),
                }
                encoded[mode] = torch.cat(probe.embeddings)
            delta = encoded["window"] - encoded["global"]
            row["encoder_intervention"] = {
                "max_absolute_change": float(delta.abs().max()),
                "relative_l2_change": float(delta.norm() / encoded["global"].norm()),
            }
            global_errors = row["official_fp32_global"]["cer"]["errors"]
            window_errors = row["official_fp32_window"]["cer"]["errors"]
            row["error_decomposition"] = {
                "existing_candidate_minus_original": right["scores"]["cer"]["errors"]
                - left["scores"]["cer"]["errors"],
                "window_fp32_minus_global_fp32": window_errors - global_errors,
                "candidate_minus_window_fp32": right["scores"]["cer"]["errors"]
                - window_errors,
            }
            report["samples"].append(row)
            report["diagnostic_totals"] = {
                key: sum(item["error_decomposition"][key] for item in report["samples"])
                for key in row["error_decomposition"]
            }
            args.output.write_text(
                json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
            )
            print(
                json.dumps(
                    {
                        "id": sample_id,
                        "decomposition": row["error_decomposition"],
                        "global": row["official_fp32_global"]["hypothesis"],
                        "window": row["official_fp32_window"]["hypothesis"],
                        "candidate": right["hypothesis"],
                        "encoder_change": row["encoder_intervention"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    probe.close()
    report["elapsed_seconds"] = time.perf_counter() - start
    report["global_rerun_matches_saved_baseline"] = all(
        row["official_fp32_global"]["equals_original_global_normalized"]
        for row in report["samples"]
    )
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "diagnostic_totals": report["diagnostic_totals"],
                "global_rerun_matches_saved_baseline": report[
                    "global_rerun_matches_saved_baseline"
                ],
                "elapsed_seconds": report["elapsed_seconds"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
