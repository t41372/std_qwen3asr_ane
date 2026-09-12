"""Verify discovery and the actual plugin result, beyond metadata compliance."""

import argparse
import json
from pathlib import Path

from standard_asr import discover_models
from standard_asr.compliance import check_transcription_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    registry = discover_models(strict=True)
    engine = registry.create("std-qwen3asr-ane/1.7b", model_dir=args.model_dir)
    result = engine.transcribe(args.audio)
    report = check_transcription_result(
        result, capabilities=engine.declared_capabilities
    )
    payload = {
        "result": result.model_dump(mode="json"),
        "compliance_passed": report.passed,
        "issues": [str(issue) for issue in report.issues],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
