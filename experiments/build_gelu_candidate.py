"""Replace only encoder GELU graphs, preserving all decoder/weight evidence."""

import argparse
import json
import os
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import torch
from build_silu_candidate import digest
from std_qwen3asr_ane.conversion.encoder import build_encoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base", type=Path, default=Path("artifacts/qwen3-asr-1.7b-stable-silu")
    )
    parser.add_argument(
        "--source", type=Path, default=Path("artifacts/source/Qwen3-ASR-1.7B")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/qwen3-asr-1.7b-precise")
    )
    args = parser.parse_args()
    base, source, output = (
        args.base.resolve(),
        args.source.resolve(),
        args.output.resolve(),
    )
    if output.exists():
        raise FileExistsError("Use a new candidate directory")
    manifest = json.loads((base / "manifest.json").read_text())
    if (
        manifest["source_revision"]
        != json.loads((source / "source.json").read_text())["revision"]
    ):
        raise ValueError("Source differs from ablation baseline")
    subprocess.run(["/bin/cp", "-cR", str(base), str(output)], check=True)
    (output / "manifest.json").unlink()
    for role in ("frontend", "encoder"):
        shutil.rmtree(output / manifest["files"][role])
    # Keep human-readable conversion source alongside this immutable candidate.
    conversion = (
        Path(__file__).resolve().parents[1]
        / "std_qwen3asr_ane/src/std_qwen3asr_ane/conversion"
    )
    snapshot = output / "conversion_source"
    snapshot.mkdir(exist_ok=True)
    hashes = {}
    for script in conversion.glob("*.py"):
        shutil.copy2(script, snapshot / script.name)
        hashes[script.name] = digest(script)
    torch.set_num_threads(4)
    encoder = build_encoder(source, output)
    for name, expected in hashes.items():
        if digest(conversion / name) != expected:
            raise RuntimeError(
                "Conversion source changed during the build; no completion manifest was written"
            )
    shared = []
    for entry in encoder["files"]:
        for binary in (output / entry["path"]).rglob("weight.bin"):
            original = base / binary.relative_to(output)
            if original.is_file() and digest(original) == digest(binary):
                binary.unlink()
                os.link(original, binary)
                shared.append(str(binary.relative_to(output)))
    manifest.update(
        created_at=datetime.now(UTC).isoformat(),
        validation_status="unvalidated",
        encoder_activation="unfused_erf_gelu",
        encoder=encoder["encoder"],
        ablation_baseline_manifest_sha256=digest(base / "manifest.json"),
        conversion_source_sha256=hashes,
        shared_encoder_weight_files=shared,
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {"output": str(output), "shared_identical_encoder_payloads": len(shared)}
        )
    )


if __name__ == "__main__":
    main()
