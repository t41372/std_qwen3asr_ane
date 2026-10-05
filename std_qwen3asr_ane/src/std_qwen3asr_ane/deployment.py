"""Read-only host requirements for the Apple Neural Engine plugin."""

import platform
import sys

MINIMUM_MACOS_MAJOR = 15


def unsupported_host_reason() -> str | None:
    """Return a deployment remedy before acquisition or native model loading."""
    if sys.platform != "darwin" or platform.machine().casefold() not in {"arm64", "aarch64"}:
        return "This engine requires an Apple Silicon Mac."
    version = platform.mac_ver()[0].split(".", 1)[0]
    if not version.isdecimal() or int(version) < MINIMUM_MACOS_MAJOR:
        return "This engine requires macOS 15 or later."
    return None
