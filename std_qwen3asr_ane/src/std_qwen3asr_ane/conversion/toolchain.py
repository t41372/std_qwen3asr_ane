"""The conversion versions used by the validated bundle recipe.

The packaged worker declares these same versions in its PEP 723 metadata.
Inference dependencies may evolve independently; conversion must not silently
change just because an application already has another PyTorch/Core ML stack.
"""

from importlib.metadata import PackageNotFoundError, version

CONVERSION_VERSIONS = {
    "coremltools": "9.0",
    "torch": "2.14.0",
    "numpy": "2.5.3",
    "transformers": "4.57.6",
    "huggingface-hub": "0.36.2",
    "safetensors": "0.8.0",
    "scipy": "1.18.1",
    "tokenizers": "0.22.2",
}


def conversion_toolchain_available() -> bool:
    """Read distribution metadata without importing native libraries."""
    try:
        return all(version(name) == expected for name, expected in CONVERSION_VERSIONS.items())
    except PackageNotFoundError:
        return False
