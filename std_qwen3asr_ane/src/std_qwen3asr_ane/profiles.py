"""Explicit artifact recipes; actual prompt capacity is always checked at runtime."""

from dataclasses import dataclass
from typing import Literal

ProfileName = Literal["general", "short-dictation"]


@dataclass(frozen=True)
class BundleProfile:
    cache_length: int
    max_audio_seconds: float
    max_new_tokens: int


PROFILES = {
    "general": BundleProfile(1024, 30.0, 256),
    "short-dictation": BundleProfile(512, 12.0, 128),
}
