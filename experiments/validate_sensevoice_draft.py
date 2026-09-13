"""Compare native-fbank preprocessing with the published Core ML frontend."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import coremltools as ct
import numpy as np
from evaluate import audio_samples
from sensevoice_draft import SenseVoiceDraft
from std_qwen3asr_ane.runtime import PersistentInputModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("artifacts/drafts/sensevoice-small/models"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--encoder", default="SenseVoiceSmall_int8.mlmodelc")
    parser.add_argument("--padding", choices=("zero", "repeat"), default="zero")
    args = parser.parse_args()
    started = perf_counter()
    draft = SenseVoiceDraft(
        args.model_dir, encoder_name=args.encoder, padding=args.padding
    )
    report = {"load_seconds": perf_counter() - started, "samples": []}
    try:
        for language in ("en", "zh"):
            samples, digest = audio_samples(
                Path(f"artifacts/evaluation/smoke/qwen_official_{language}.wav")
            )
            actual = draft.frontend(samples)
            frontend = PersistentInputModel(
                ct.models.CompiledMLModel(
                    str(args.model_dir / "SenseVoicePreprocessor.mlmodelc"),
                    compute_units=ct.ComputeUnit.CPU_ONLY,
                )
            )
            try:
                expected = frontend.predict({"waveform": samples[None] * 32768})[
                    "features"
                ]
            finally:
                frontend.close()
            if actual.shape[0::2] != expected.shape[0::2] or expected.shape[1] not in (
                actual.shape[1],
                actual.shape[1] + 1,
            ):
                raise AssertionError(
                    f"Feature shapes differ: {actual.shape}, {expected.shape}"
                )
            extra_rows = expected.shape[1] - actual.shape[1]
            # The published export appends seven right-edge frames then applies
            # valid stride-six convolution without truncating to ceil(T/6).
            # Some lengths therefore expose one extra trailing LFR row. Compare
            # the common upstream-defined rows and report this discrepancy.
            expected = expected[:, : actual.shape[1]]
            record = {
                "language": language,
                "audio_sha256": digest,
                "feature_shape": list(actual.shape),
                "published_frontend_extra_rows": extra_rows,
                "frontend_max_abs": float(np.max(np.abs(actual - expected))),
                "frontend_relative_l2": float(
                    np.linalg.norm(actual - expected) / np.linalg.norm(expected)
                ),
                "predictions": [],
            }
            report["samples"].append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
            for repeat in range(4):
                result = draft.transcribe(samples)
                record["predictions"].append(
                    {
                        "repeat": repeat,
                        "text": result.text,
                        "detected_language": result.language,
                        "timings": result.timings,
                    }
                )
            print(json.dumps(record, ensure_ascii=False), flush=True)
    finally:
        draft.close()
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
