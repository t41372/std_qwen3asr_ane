"""Collect small round-2 reports with hashes of their retained local originals.

Missing files are recorded as missing, never interpreted as a passed gate.
Weights, audio, per-utterance transcripts and raw Instruments host metadata
remain under artifacts/. Re-run after additional measurements finish.
"""

import argparse
import json
from pathlib import Path

from std_qwen3asr_ane.bundle import digest

REPORTS = {
    "t64_t1": "selection-t64-t1-vs-baseline.json",
    "t64_t16_compact": "selection-t64-t16-compact-vs-baseline.json",
    "compact_head": "compact-t1-probe.json",
    "enumerated_real": "enumerated-native.json",
    "enumerated_tiny": "mil-no-output-alias-enumerated.json",
    "enumerated_fixed_control": "mil-no-output-alias-fixed.json",
    "short_paired": "short-paired-3/comparison-v2.json",
    "short_heldout": "short-heldout-comparison.json",
    "short_eos": "short-eos-parity/comparison.json",
    "short_memory_baseline": "short-memory-baseline.json",
    "short_memory_candidate": "short-memory-candidate.json",
    "native_python": "native-python-v2.json",
    "native_swift": "native-swift.json",
    "native_swift_asan": "native-swift-asan.json",
    "native_quiet": "native-quiet-summary.json",
    "fused_head": "fused-head-probe-v2/summary.json",
    "w8a8_head": "w8a8-head-paired.summary.json",
    "w8a8_decoder": "w8a8-decoder-probe/summary.json",
    "lut6_selection": "selection-lut6-comparison.json",
    "lut6_memory": "lut6-memory-paired/summary.json",
    "lut6_paired": "lut6-paired-3/comparison.json",
    "lut6_heldout": "lut6-heldout/comparison.json",
    "lut6_multilingual": "lut6-multilingual/comparison.json",
    "mixed_components": "mixed-components.json",
    "multilingual": "multilingual-paired/comparison.json",
    "robustness": "robustness-paired/comparison.json",
    "robustness_outcomes": "robustness-outcomes.json",
    "short_kv": "short-smoke-kv-parity.json",
    "robustness_kv": "robustness-kv-parity.json",
    "short_trace": "short-trace-summary.json",
    "lut6_trace": "lut6-trace-summary.json",
    "short_trace_attribution": "short-trace-attribution-v2.json",
    "lut6_trace_attribution": "lut6-trace-attribution-v2.json",
    "profile_streaming": "profile-streaming-comparison.json",
    "installed_short_profile": "installed-short-profile.json",
    "energy_summary": "energy-short/summary.json",
    "energy_A1": "energy-short/A1/summary.json",
    "energy_B1": "energy-short/B1/summary.json",
    "energy_B2": "energy-short/B2/summary.json",
    "energy_A2": "energy-short/A2/summary.json",
    "idle_target": "idle-target/summary.json",
    "idle_draft": "idle-draft/summary.json",
    "idle_summary": "idle-comparison.json",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("artifacts/evaluation/round2")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("research/evidence/round2/measurements.json"),
    )
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]

    def portable(value):
        if isinstance(value, dict):
            return {key: portable(item) for key, item in value.items()}
        if isinstance(value, list):
            return [portable(item) for item in value]
        if isinstance(value, str):
            return value.replace(str(repository) + "/", "")
        return value

    report = {
        "schema_version": 1,
        "promotion_claim": False,
        "reports": {},
        "missing": [],
        "pending": [],
    }
    for name, relative in REPORTS.items():
        path = args.root / relative
        if not path.is_file():
            report["missing"].append(relative)
            continue
        data = json.loads(path.read_text())
        if (
            name in ("idle_target", "idle_draft")
            and not data.get("complete")
            and not data.get("error")
        ):
            report["pending"].append(relative)
            continue
        if name in ("short_kv", "robustness_kv"):
            # Keep the full per-layer state hashes in the referenced local report.
            data["cases"] = [
                {
                    **{
                        key: row[key]
                        for key in (
                            "id",
                            "audio_sha256",
                            "exact_decisions",
                            "exact_consumed_kv",
                            "differences",
                        )
                    },
                    "finite_states": all(
                        row[side]["finite_states"] for side in ("baseline", "candidate")
                    ),
                    "consumed_positions": {
                        side: row[side]["consumed_positions"]
                        for side in ("baseline", "candidate")
                    },
                }
                for row in data["cases"]
            ]
        if name == "mixed_components":
            data["cases"] = [
                {
                    **{key: row[key] for key in ("id", "audio_sha256", "metric")},
                    "variants": {
                        variant: {
                            "errors": value["errors"],
                            "exact_tokens_and_eos_to_lut8": (
                                value["tokens"] == row["variants"]["lut8"]["tokens"]
                                and value["eos_token_id"]
                                == row["variants"]["lut8"]["eos_token_id"]
                            ),
                        }
                        for variant, value in row["variants"].items()
                    },
                }
                for row in data["cases"]
            ]
        report["reports"][name] = {
            "source": str(path.relative_to(repository) if path.is_absolute() else path),
            "source_sha256": digest(path),
            "data": portable(data),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
