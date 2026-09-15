"""Expose visible formatting/content changes separately from normalized WER/CER.

This creates a review queue, not an automatic semantic-equivalence or named-
entity score. Reference corpora often lack reliable punctuation/case labels.
"""

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

from evaluate import normalize


def measured(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows = [row for row in rows if row.get("phase") == "measured"]
    result = {(row["id"], row["repeat"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError("Duplicate measured attempts")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    a, b = measured(args.baseline), measured(args.candidate)
    if a.keys() != b.keys():
        raise ValueError("Different measured attempt sets")
    changed, failures, counts = [], [], Counter()
    for key, first in a.items():
        second = b[key]
        if first["audio_sha256"] != second["audio_sha256"]:
            raise ValueError(f"Different audio for {key}")
        if first.get("error") or second.get("error"):
            failures.append(
                {
                    "id": key[0],
                    "repeat": key[1],
                    "baseline": first.get("error"),
                    "candidate": second.get("error"),
                }
            )
            continue
        left, right = first["hypothesis"], second["hypothesis"]
        language_changed = first.get("detected_language") != second.get("detected_language")
        if left == right and not language_changed:
            continue
        flags = {
            "normalized_text_changed": normalize(left) != normalize(right),
            "numbers_changed": re.findall(r"\d+(?:[.,:/-]\d+)*", left)
            != re.findall(r"\d+(?:[.,:/-]\d+)*", right),
            "acronyms_changed": re.findall(r"\b[A-Z][A-Z0-9]{1,}\b", left)
            != re.findall(r"\b[A-Z][A-Z0-9]{1,}\b", right),
            "punctuation_changed": [c for c in left if unicodedata.category(c).startswith("P")]
            != [c for c in right if unicodedata.category(c).startswith("P")],
            "language_changed": language_changed,
        }
        counts.update(name for name, value in flags.items() if value)
        changed.append(
            {
                "id": key[0],
                "repeat": key[1],
                "language": first["language"],
                "reference": first["reference"],
                "baseline": left,
                "candidate": right,
                "baseline_raw": first.get("raw_text"),
                "candidate_raw": second.get("raw_text"),
                "flags": flags,
            }
        )
    report = {
        "paired_attempts": len(a),
        "changed_attempts": len(changed),
        "failures": failures,
        "flags": dict(counts),
        "changes": changed,
        "semantic_review": "required" if changed else "visible_text_and_language_unchanged",
        "limitation": "Flags are review aids, not reference-based punctuation/entity accuracy.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "changes"}))


if __name__ == "__main__":
    main()
