"""Create a labelled long-stream diagnostic from complete existing speech clips.

This is a synthetic concatenation test, not an independent natural long-form
quality corpus. Each source utterance is retained completely, with short gaps.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from evaluate import audio_samples, manifest_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=180)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    budget = round(args.seconds * 16000)
    parts, segments, offset = [], [], 0
    for row in manifest_rows(args.manifest):
        samples, digest = audio_samples(Path(row["audio_path"]))
        if offset + len(samples) > budget - 8000:
            continue
        parts.append(samples)
        segments.append(
            {
                "source_id": row["id"],
                "audio_sha256": digest,
                "start_seconds": offset / 16000,
                "end_seconds": (offset + len(samples)) / 16000,
                "reference": row["reference"],
            }
        )
        offset += len(samples)
        gap = np.zeros(min(4000, budget - offset), np.float32)
        parts.append(gap)
        offset += len(gap)
        if budget - offset <= 5 * 16000:
            break
    if not segments:
        raise ValueError("No complete utterance fits the requested duration")
    parts.append(np.zeros(budget - offset, np.float32))
    audio = np.concatenate(parts)
    path = args.output / "audio.wav"
    sf.write(path, audio, 16000, subtype="FLOAT")
    record = {
        "id": f"concatenated-{args.seconds:g}s",
        "audio_path": "audio.wav",
        "reference": " ".join(segment["reference"] for segment in segments),
        "language": "en",
        "split": "synthetic_long_stream_diagnostic",
        "audio_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    (args.output / "manifest.jsonl").write_text(
        json.dumps(record, ensure_ascii=False) + "\n"
    )
    (args.output / "segments.json").write_text(json.dumps(segments, indent=2) + "\n")
    print(
        json.dumps(
            {
                "audio_seconds": len(audio) / 16000,
                "complete_utterances": len(segments),
                "trailing_silence_seconds": (budget - offset) / 16000,
            }
        )
    )


if __name__ == "__main__":
    main()
