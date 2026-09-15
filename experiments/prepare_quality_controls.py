"""Reuse frozen target transcripts for screening only; never reuse their timings."""

import json
from pathlib import Path

from std_qwen3asr_ane.bundle import digest


def main():
    output = Path("artifacts/evaluation/round3/quality-controls")
    output.mkdir(parents=True, exist_ok=False)
    expected = digest(Path("artifacts/qwen3-asr-1.7b-p14-lut8-g32-compiled/manifest.json"))
    report = {
        "usage": "quality screening only; latency must be remeasured in paired runs",
        "sources": {},
    }
    for name, directory, count in (
        ("regression", "lut6-paired-3", 400),
        ("multilingual", "multilingual-paired", 300),
        ("robustness", "robustness-paired", 12),
    ):
        root = Path("artifacts/evaluation/round2") / directory
        summary = json.loads((root / "summary.json").read_text())
        if not summary["complete"] or summary["models"]["baseline"]["manifest_sha256"] != expected:
            raise ValueError(f"Invalid target control at {root}")
        source = root / "baseline.jsonl"
        rows = [json.loads(line) for line in source.read_text().splitlines()]
        rows = [row for row in rows if row["phase"] == "measured" and row["repeat"] == 0]
        if (
            len(rows) != count
            or len({row["id"] for row in rows}) != count
            or any(row.get("error") for row in rows)
        ):
            raise ValueError(f"Incomplete control rows at {root}")
        path = output / f"{name}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        report["sources"][name] = {
            "raw_path": str(source),
            "raw_sha256": digest(source),
            "summary_sha256": digest(root / "summary.json"),
            "output_sha256": digest(path),
            "count": count,
            "max_new_tokens": summary["max_new_tokens"],
        }
    (output / "provenance.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
