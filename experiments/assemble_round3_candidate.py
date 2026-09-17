"""Assemble a minimal immutable candidate from independently measured components."""

import argparse
import json
from pathlib import Path

import numpy as np

from std_qwen3asr_ane.bundle import clone, digest, language_head_output, validate_bundle_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--frontend", type=Path)
    parser.add_argument("--encoder", type=Path)
    parser.add_argument("--embedding", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = json.loads((args.base / "manifest.json").read_text())
    validate_bundle_paths(
        args.base.resolve(), set(manifest["files"].values()) | set(manifest["decoder_partitions"])
    )
    sources = {role: args.base / relative for role, relative in manifest["files"].items()}
    components = {
        "base": {"path": str(args.base), "manifest_sha256": digest(args.base / "manifest.json")}
    }
    for role, bundle in (("frontend", args.frontend), ("encoder", args.encoder)):
        if bundle is None:
            continue
        component = json.loads((bundle / "manifest.json").read_text())
        validate_bundle_paths(bundle.resolve(), set(component["files"].values()))
        if component["source_revision"] != manifest["source_revision"]:
            raise ValueError("Component source revision differs from base")
        settings = component["audio_candidate"]
        if settings["role"] != role:
            raise ValueError("Wrong audio component role")
        if role == "frontend":
            if settings["batch"] != 4:
                raise ValueError("Only the measured B4 frontend is supported")
            sources["frontend_batched"] = bundle / component["files"]["frontend_batched"]
            manifest["frontend"] = {**manifest["frontend"], "offline_batch_size": 4}
        else:
            if settings["batch"] != 1:
                raise ValueError("Only a B1 encoder can replace the base encoder")
            sources["encoder"] = bundle / component["files"]["encoder"]
            if settings.get("lut8_group"):
                manifest["audio_weight_compression"] = {
                    "scheme": "palette",
                    "bits": 8,
                    "group_size": settings["lut8_group"],
                    "roles": ["encoder"],
                }
            else:
                # An FP16 encoder component carries no compression claim.
                manifest.pop("audio_weight_compression", None)
        components[role] = {
            "path": str(bundle),
            "manifest_sha256": digest(bundle / "manifest.json"),
        }
    if args.embedding is not None:
        quantization = json.loads((args.embedding / "manifest.json").read_text())
        if digest(sources["embedding"]) != quantization["source_sha256"]:
            raise ValueError("Quantized embedding does not match the base table")
        original = np.load(sources["embedding"], mmap_mode="r", allow_pickle=False)
        if list(original.shape) != quantization["shape"]:
            raise ValueError("Quantized embedding shape differs from base")
        files = quantization["files"]
        validate_bundle_paths(args.embedding.resolve(), set(files.values()))
        # Exports made before the package quantizer existed named the scales "scales".
        scale_key = "embedding_scales" if "embedding_scales" in files else "scales"
        sources["embedding"] = args.embedding / files["embedding"]
        sources["embedding_scales"] = args.embedding / files[scale_key]
        for path in (sources["embedding"], sources["embedding_scales"]):
            if digest(path) != quantization["payload_sha256"][path.name]:
                raise ValueError("Quantized embedding payload changed")
        manifest["head_output"] = language_head_output(manifest)
        manifest["schema_version"] = 3
        manifest["embedding_quantization"] = {
            key: quantization[key]
            for key in (
                "scheme",
                "axis",
                "scale_dtype",
                "shape",
                "quantizer_version",
                "source_sha256",
                "max_abs_error",
                "mean_abs_error",
                "payload_sha256",
                "bytes_saved",
            )
        }
        components["embedding"] = {
            "path": str(args.embedding),
            "manifest_sha256": digest(args.embedding / "manifest.json"),
        }
    args.output.mkdir(parents=True)
    destination_sources = {}
    files = {}
    for role, source in sources.items():
        name = source.name
        if name in destination_sources and source.resolve() != destination_sources[name]:
            raise ValueError(f"Conflicting component filenames: {name}")
        if name not in destination_sources:
            clone(source, args.output / name)
            destination_sources[name] = source.resolve()
        files[role] = name
    mapping = {
        relative: files[role] for role, relative in manifest["files"].items() if role in files
    }
    manifest["decoder_partitions"] = [
        mapping[relative] for relative in manifest["decoder_partitions"]
    ]
    manifest["files"] = files
    manifest.pop("compiled_from", None)
    manifest.pop("round3_audio_batch", None)
    manifest["assembled_from"] = components
    manifest["validation_status"] = "unvalidated"
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "manifest_sha256": digest(args.output / "manifest.json"),
                "components": list(components),
            }
        )
    )


if __name__ == "__main__":
    main()
