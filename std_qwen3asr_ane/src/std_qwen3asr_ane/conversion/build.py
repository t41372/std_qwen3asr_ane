"""Build a locally reproducible bundle from the official pinned checkpoint."""

from __future__ import annotations

import importlib.metadata
import json
from datetime import UTC, datetime
from pathlib import Path

from ..profiles import PROFILES, ProfileName

SUPPORTED_CHECKPOINTS = {"Qwen/Qwen3-ASR-1.7B", "Qwen/Qwen3-ASR-0.6B"}
# The 1.7B revision every measurement in research/ used.
SOURCE_REVISION = "7278e1e70fe206f11671096ffdd38061171dd6e5"


def build_bundle(
    source: Path,
    output: Path,
    *,
    cache_length=None,
    reuse_encoder=False,
    token_batch_size=None,
    layers_per_partition=4,
    profile: ProfileName | None = None,
    frontend_batch_size: int | None = None,
):
    import numpy as np
    import torch
    from transformers import AutoTokenizer, WhisperFeatureExtractor

    from .decoder import build_decoder
    from .encoder import build_encoder

    settings = PROFILES[profile] if profile is not None else None
    if frontend_batch_size is not None and (
        type(frontend_batch_size) is not int or frontend_batch_size not in (1, 4)
    ):
        raise ValueError("Supported offline frontend batches are 1 and 4")
    if cache_length is None:
        cache_length = settings.cache_length if settings else 1024
    if settings is not None and cache_length != settings.cache_length:
        raise ValueError("Explicit cache length conflicts with the selected profile")
    if token_batch_size is None:
        token_batch_size = 16 if settings else 1

    source, output = source.resolve(), output.resolve()
    if not (source / "config.json").is_file():
        raise FileNotFoundError(f"Download the official checkpoint first: {source}")
    provenance = json.loads((source / "source.json").read_text())
    if provenance.get("model_id") not in SUPPORTED_CHECKPOINTS or not provenance.get("revision"):
        raise ValueError("Source metadata must identify a supported pinned Qwen3-ASR checkpoint")
    torch.set_num_threads(4)
    output.mkdir(parents=True, exist_ok=True)
    # The manifest is a completion marker: an interrupted build must not expose
    # old readiness metadata for partly replaced graphs.
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"A completed bundle already exists at {output}; use a new directory")
    if reuse_encoder:
        encoder = json.loads((output / "encoder-manifest.json").read_text())
        if (
            frontend_batch_size is not None
            and encoder["frontend"].get("offline_batch_size", 1) != frontend_batch_size
        ):
            raise ValueError("Reused frontend batch differs from the requested batch")
    else:
        encoder = build_encoder(
            source,
            output,
            frontend_batch_size=1 if frontend_batch_size is None else frontend_batch_size,
        )
    decoder = build_decoder(
        source,
        output,
        cache_length=cache_length,
        token_batch_size=token_batch_size,
        layers_per_partition=layers_per_partition,
    )
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True, fix_mistral_regex=True)
    tokenizer.backend_tokenizer.save(str(output / "tokenizer.json"))
    processor = WhisperFeatureExtractor.from_pretrained(source, local_files_only=True)
    np.save(output / "mel_filters.npy", processor.mel_filters.astype(np.float32))
    chat_template = json.loads((source / "chat_template.json").read_text())["chat_template"]
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": ""},
            {"role": "user", "content": [{"type": "audio", "audio": ""}]},
        ],
        chat_template=chat_template,
        add_generation_prompt=True,
        tokenize=False,
    )
    files = {entry["role"]: entry["path"] for entry in encoder["files"]}
    files.update(decoder.pop("files"))
    files.update(tokenizer="tokenizer.json", mel_filters="mel_filters.npy")
    manifest = {
        "schema_version": 1,
        "model_id": provenance["model_id"],
        "source_revision": provenance["revision"],
        "files": files,
        "created_at": datetime.now(UTC).isoformat(),
        "max_audio_seconds": settings.max_audio_seconds if settings else 30,
        **(
            {"profile": profile, "default_max_new_tokens": settings.max_new_tokens}
            if settings
            else {}
        ),
        "frontend": encoder["frontend"],
        "encoder": encoder["encoder"],
        "prompt_template": prompt,
        "eos_token_ids": [151643, 151645],
        "conversion_versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "coremltools", "transformers", "numpy")
        },
        "validation_status": "unvalidated",
        "activation": "stable_exp_silu",
        "encoder_activation": "unfused_erf_gelu",
        "tokenizer_fix_mistral_regex": True,
        "layers_per_partition": layers_per_partition,
        **decoder,
    }
    temporary = output / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(manifest_path)
    return manifest


def download_source(destination: Path, *, revision="main", model_id="Qwen/Qwen3-ASR-1.7B"):
    from huggingface_hub import HfApi, snapshot_download
    from standard_asr.contract.exceptions import ArtifactAcquisitionError
    from standard_asr.engine import allow_downloads

    if model_id not in SUPPORTED_CHECKPOINTS:
        raise ValueError("Unsupported Qwen3-ASR checkpoint")
    if not allow_downloads():
        raise ArtifactAcquisitionError(
            "Checkpoint download is disabled by STANDARD_ASR_ALLOW_DOWNLOAD.",
            reason="downloads_disabled",
        )
    commit = HfApi().model_info(model_id, revision=revision).sha
    if not allow_downloads():
        raise ArtifactAcquisitionError(
            "Checkpoint download is disabled by STANDARD_ASR_ALLOW_DOWNLOAD.",
            reason="downloads_disabled",
        )
    snapshot_download(
        model_id,
        revision=commit,
        local_dir=destination,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "README.md"],
    )
    provenance = {"model_id": model_id, "revision": commit}
    (destination / "source.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance
