"""Localize LUT6 regressions by swapping existing compiled p14 components.

The manifest intentionally contains selected failing cases, so these results
cannot estimate corpus quality. Weight bytes are a storage proxy, not measured
system memory. No candidate bundle is modified or promoted by this diagnostic.
"""

import argparse
import json
from pathlib import Path

from evaluate import audio_samples, manifest_rows, score
from std_qwen3asr_ane.bundle import digest
from std_qwen3asr_ane.runtime import CoreMLRuntime, PersistentInputModel

VARIANTS = {
    "lut8": (8, 8, 8),
    "lut6": (6, 6, 6),
    "decoder6_head8": (6, 6, 8),
    "first14_8_rest6": (8, 6, 6),
    "last14_8_rest6": (6, 8, 6),
}


def weight_bytes(path):
    return sum(child.stat().st_size for child in path.rglob("weight.bin"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lut8", type=Path, required=True)
    parser.add_argument("--lut6", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh output")
    roots = {8: args.lut8, 6: args.lut6}
    manifests = {
        bits: json.loads((path / "manifest.json").read_text())
        for bits, path in roots.items()
    }
    for key in (
        "model_id",
        "source_revision",
        "max_sequence_length",
        "max_audio_seconds",
        "residual_scale",
        "token_batch_size",
        "head_dim",
        "rope_theta",
    ):
        if manifests[8][key] != manifests[6][key]:
            raise ValueError(f"Incompatible component geometry: {key}")
    for bits, manifest in manifests.items():
        compression = manifest["weight_compression"]
        if len(manifest["decoder_partitions"]) != 2 or (
            compression["scheme"],
            compression["bits"],
            compression["group_size"],
        ) != ("palette", bits, 32):
            raise ValueError("Expected matching p14 LUT8/LUT6 g32 bundles")
    for role in ("embedding", "tokenizer", "mel_filters"):
        if (
            len(
                {digest(roots[bits] / manifests[bits]["files"][role]) for bits in roots}
            )
            != 1
        ):
            raise ValueError(f"Non-quantized asset differs: {role}")
    fixed_bytes = sum(
        weight_bytes(args.lut8 / manifests[8]["files"][role])
        for role in ("frontend", "encoder")
    )
    component_bytes = {
        bits: [
            weight_bytes(path / name)
            for name in (
                *manifests[bits]["decoder_partitions"],
                manifests[bits]["files"]["lm_head"],
            )
        ]
        for bits, path in roots.items()
    }
    sizes = {
        name: fixed_bytes
        + sum(component_bytes[bits][index] for index, bits in enumerate(pattern))
        for name, pattern in VARIANTS.items()
    }
    report = {
        "complete": False,
        "purpose": "selected-case component sensitivity only; no corpus quality or latency claim",
        "manifest_sha256": digest(args.manifest),
        "models": {
            bits: digest(path / "manifest.json") for bits, path in roots.items()
        },
        "script_sha256": digest(Path(__file__)),
        "variants": {
            name: {
                "first14_last14_head_bits": pattern,
                "weight_payload_bytes": sizes[name],
                "weight_payload_reduction": 1 - sizes[name] / sizes["lut8"],
            }
            for name, pattern in VARIANTS.items()
        },
        "cases": [],
    }
    runtimes, original = {}, None
    try:
        for bits, path in roots.items():
            runtimes[bits] = CoreMLRuntime(path)
        runtime = runtimes[8]
        original = (runtime.decoders, runtime.lm_head)
        components = {
            bits: (*value.decoders, value.lm_head) for bits, value in runtimes.items()
        }
        for item in manifest_rows(args.manifest):
            audio, audio_hash = audio_samples(Path(item["audio_path"]))
            row = {
                "id": item["id"],
                "audio_sha256": audio_hash,
                "metric": item["diagnostic_metric"],
                "variants": {},
            }
            for name, pattern in VARIANTS.items():
                runtime.decoders = [
                    components[pattern[index]][index] for index in (0, 1)
                ]
                runtime.lm_head = components[pattern[2]][2]
                result = runtime.transcribe(audio, language=None, max_new_tokens=256)
                row["variants"][name] = {
                    "text": result.text,
                    "tokens": list(result.token_ids),
                    "eos_token_id": result.timings["eos_token_id"],
                    "errors": score(item["reference"], result.text)[row["metric"]][
                        "errors"
                    ],
                }
            report["cases"].append(row)
            print(
                json.dumps(
                    {
                        "id": item["id"],
                        "errors": {
                            name: value["errors"]
                            for name, value in row["variants"].items()
                        },
                    }
                ),
                flush=True,
            )
        report["complete"] = True
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if original is not None:
            runtimes[8].decoders, runtimes[8].lm_head = original
        try:
            PersistentInputModel.close_many(
                [
                    model
                    for value in runtimes.values()
                    for model in (
                        value.frontend,
                        value.encoder,
                        *value.decoders,
                        value.lm_head,
                    )
                ]
            )
            report["close_succeeded"] = True
        finally:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(report, indent=2, ensure_ascii=False) + "\n"
            )


if __name__ == "__main__":
    main()
