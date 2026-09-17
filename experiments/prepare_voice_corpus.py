"""Freeze voice-command eligibility from inputs and prompt capacity, never outputs."""

import argparse
import json
from pathlib import Path

import soundfile as sf
from tokenizers import Tokenizer

from std_qwen3asr_ane.audio import HOP_LENGTH, MIN_SAMPLES, audio_token_count
from std_qwen3asr_ane.bundle import digest
from std_qwen3asr_ane.runtime import DEFAULT_PROMPT, build_prompt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    bundle = json.loads((args.bundle / "manifest.json").read_text())
    tokenizer = Tokenizer.from_file(str(args.bundle / bundle["files"]["tokenizer"]))
    selected, excluded = [], []
    for line in args.manifest.read_text().splitlines():
        row = json.loads(line)
        path = Path(row["audio_path"])
        if not path.is_absolute():
            path = (args.manifest.parent / path).resolve()
        info = sf.info(path)
        samples = (info.frames * 16000 + info.samplerate - 1) // info.samplerate
        frames = max(samples, MIN_SAMPLES) // HOP_LENGTH
        prompt = build_prompt(
            tokenizer,
            audio_token_count(frames),
            None,
            bundle.get("prompt_template", DEFAULT_PROMPT),
        )
        reason = (
            "audio_duration"
            if samples > 6 * 16000
            else ("prompt_capacity" if len(prompt) + 64 - 1 > 256 else None)
        )
        if reason:
            excluded.append({"id": row["id"], "reason": reason})
        else:
            selected.append({**row, "audio_path": str(path)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected))
    report = {
        "source_sha256": digest(args.manifest),
        "output_sha256": digest(args.output),
        "selected": len(selected),
        "excluded": excluded,
        "max_audio_seconds": 6,
        "max_new_tokens": 64,
        "cache_length": 256,
        "output_used_for_selection": False,
    }
    args.output.with_suffix(".freeze.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"selected": len(selected), "excluded": len(excluded)}))


if __name__ == "__main__":
    main()
