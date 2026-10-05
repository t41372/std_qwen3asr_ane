"""Keep measured transcript ranges relative to the containing result text."""

from __future__ import annotations

from collections.abc import Mapping


def needs_join_space(left: str, right: str) -> bool:
    """Separate independent word-delimited windows without adding CJK spaces."""
    if left.isspace() or right.isspace():
        return False
    return not any(
        "\u2e80" <= character <= "\ua4cf" or "\uac00" <= character <= "\ud7af"
        for character in (left, right)
    )


def shift_source_offsets(extra: Mapping, offset: int) -> dict:
    """Copy extra fields, moving known character offsets into the merged text.

    Only the plugin's exact source ranges change. Measured times and raw model
    evidence keep their own documented timelines and coordinates.
    """
    result = dict(extra)
    for key in ("source_start", "source_end"):
        if key in result:
            value = result[key]
            if type(value) is not int or value < 0:
                raise ValueError(f"{key} must be a non-negative character offset")
            result[key] = value + offset
    return result
