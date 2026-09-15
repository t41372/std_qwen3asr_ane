"""Create deterministic gain, noise and non-speech cases separately from clean ASR.

Speech retains the source reference. Empty-reference cases test hallucinations;
they must be reported by emitted text/tokens, not pooled into a clean WER score.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from evaluate import audio_samples, manifest_rows
from std_qwen3asr_ane.bundle import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh output directory")
    args.output.mkdir(parents=True)
    rng = np.random.default_rng(20260914)
    rows = []

    def save(identifier, samples, reference, language, condition, parent=None):
        path = args.output / f"{identifier}.wav"
        sf.write(path, samples, 16000, subtype="FLOAT")
        rows.append(
            {
                "id": identifier,
                "audio_path": str(path.resolve()),
                "audio_sha256": digest(path),
                "reference": reference,
                "language": language,
                "duration_seconds": len(samples) / 16000,
                "condition": condition,
                "source_audio_sha256": parent,
                "evaluation_role": "round2_robustness_separate_from_clean",
            }
        )

    for row in manifest_rows(args.manifest):
        samples, parent = audio_samples(Path(row["audio_path"]))
        if len(samples) > 12 * 16000:
            raise ValueError("Source audio must fit the 12-second profile")
        noise = rng.standard_normal(samples.shape).astype(np.float32)
        noise *= np.sqrt(np.mean(samples.astype(np.float64) ** 2)) / (
            10 * np.sqrt(np.mean(noise.astype(np.float64) ** 2))
        )
        variants = {
            "gain-quarter": samples * np.float32(0.25),
            "gain-double-clipped": np.clip(samples * 2, -1, 1),
            "white-noise-snr20db": np.clip(samples + noise, -1, 1),
        }
        for condition, values in variants.items():
            save(
                f"{row['id']}-{condition}",
                values,
                row["reference"],
                row["language"],
                condition,
                parent,
            )
    for seconds in (0.1, 0.5, 5.0, 11.99):
        save(
            f"silence-{seconds:g}s",
            np.zeros(round(seconds * 16000), np.float32),
            "",
            "und",
            "silence",
        )
    save(
        "noise-only-5s",
        rng.normal(0, 0.001, 80000).astype(np.float32),
        "",
        "und",
        "noise-only",
    )
    impulse = np.zeros(80000, np.float32)
    impulse[40000] = 0.5
    save("single-impulse-5s", impulse, "", "und", "impulse")
    path = args.output / "manifest.jsonl"
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    (args.output / "provenance.json").write_text(
        json.dumps(
            {
                "seed": 20260914,
                "source_manifest_sha256": digest(args.manifest),
                "manifest_sha256": digest(path),
                "script_sha256": digest(Path(__file__)),
                "cases": len(rows),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
