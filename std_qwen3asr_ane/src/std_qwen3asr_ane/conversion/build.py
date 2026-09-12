"""Build a locally reproducible bundle from the official pinned checkpoint."""

from __future__ import annotations

import importlib.metadata
import json
from datetime import UTC, datetime
from pathlib import Path


def build_bundle(
    source: Path, output: Path, *, cache_length=1024, reuse_encoder=False, token_batch_size=1
):
    import numpy as np
    import torch
    from transformers import AutoTokenizer, WhisperFeatureExtractor

    from .decoder import build_decoder
    from .encoder import build_encoder

    source, output = source.resolve(), output.resolve()
    if not (source / "config.json").is_file():
        raise FileNotFoundError(f"Download the official checkpoint first: {source}")
    torch.set_num_threads(4)
    output.mkdir(parents=True, exist_ok=True)
    # The manifest is a completion marker: an interrupted build must not expose
    # old readiness metadata for partly replaced graphs.
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"A completed bundle already exists at {output}; use a new directory")
    if reuse_encoder:
        encoder = json.loads((output / "encoder-manifest.json").read_text())
    else:
        encoder = build_encoder(source, output)
    decoder = build_decoder(
        source, output, cache_length=cache_length, token_batch_size=token_batch_size
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
    source_metadata = source / "source.json"
    if not source_metadata.is_file():
        raise FileNotFoundError("source.json is required to record the exact source revision")
    provenance = json.loads(source_metadata.read_text())
    manifest = {
        "schema_version": 1,
        "model_id": "Qwen/Qwen3-ASR-1.7B",
        "source_revision": provenance["revision"],
        "files": files,
        "created_at": datetime.now(UTC).isoformat(),
        "max_audio_seconds": 30,
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
        **decoder,
    }
    temporary = output / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(manifest_path)
    return manifest


def download_source(destination: Path, *, revision="main"):
    from huggingface_hub import HfApi, snapshot_download

    model_id = "Qwen/Qwen3-ASR-1.7B"
    commit = HfApi().model_info(model_id, revision=revision).sha
    snapshot_download(
        model_id,
        revision=commit,
        local_dir=destination,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "README.md"],
    )
    provenance = {"model_id": model_id, "revision": commit}
    (destination / "source.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance
