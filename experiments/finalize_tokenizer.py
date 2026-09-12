"""Match the official tokenizer option and prove default batch input equivalence."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from std_qwen3asr_ane.audio import HOP_LENGTH, SAMPLE_RATE, audio_token_count
from std_qwen3asr_ane.languages import LANGUAGE_NAMES
from std_qwen3asr_ane.runtime import build_prompt
from tokenizers import Tokenizer
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new immutable bundle directory")
    manifest = json.loads((args.base / "manifest.json").read_text())
    old = Tokenizer.from_file(str(args.base / manifest["files"]["tokenizer"]))
    new = AutoTokenizer.from_pretrained(
        args.source, local_files_only=True, fix_mistral_regex=True
    ).backend_tokenizer
    old_spec, new_spec = json.loads(old.to_str()), json.loads(new.to_str())
    if any(
        old_spec[key] != new_spec[key] for key in old_spec if key != "pre_tokenizer"
    ):
        raise ValueError("Tokenizer components other than the pre-tokenizer changed")
    maximum = audio_token_count(
        int(manifest["max_audio_seconds"] * SAMPLE_RATE) // HOP_LENGTH
    )
    checked = 0
    for tokens in range(1, maximum + 1):
        for language in [None, *LANGUAGE_NAMES]:
            if build_prompt(old, tokens, language) != build_prompt(
                new, tokens, language
            ):
                raise ValueError(
                    f"Default input changed at {tokens} audio tokens, {language}"
                )
            checked += 1
    subprocess.run(["/bin/cp", "-cR", str(args.base), str(args.output)], check=True)
    (args.output / "manifest.json").unlink()
    new.save(str(args.output / manifest["files"]["tokenizer"]))
    parent = hashlib.sha256((args.base / "manifest.json").read_bytes()).hexdigest()
    proof = {
        "schema_version": 1,
        "checked_default_prompt_variants": checked,
        "audio_token_counts": [1, maximum],
        "languages": [None, *LANGUAGE_NAMES],
        "all_default_input_ids_identical": True,
        "all_non_pretokenizer_components_identical": True,
        "parent_manifest_sha256": parent,
        "scope": "Batch without context: identical model inputs and output decoding. Context and stream prefix require their own tests.",
    }
    (args.output / "tokenizer-equivalence.json").write_text(
        json.dumps(proof, indent=2) + "\n"
    )
    manifest.update(
        tokenizer_fix_mistral_regex=True,
        tokenizer_parent_manifest_sha256=parent,
        tokenizer_equivalence_file="tokenizer-equivalence.json",
    )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(proof, indent=2))


if __name__ == "__main__":
    main()
