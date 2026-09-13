"""BCP-47 tags for Qwen3-ASR's 30 explicit language controls.

The upstream language vocabulary is published in
https://github.com/QwenLM/Qwen3-ASR/blob/main/qwen_asr/inference/utils.py.
Chinese dialect recognition does not imply additional explicit language controls.
"""

LANGUAGE_NAMES: dict[str, str] = {
    "zh": "Chinese",
    "en": "English",
    "yue": "Cantonese",
    "ar": "Arabic",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "pt": "Portuguese",
    "id": "Indonesian",
    "it": "Italian",
    "ko": "Korean",
    "ru": "Russian",
    "th": "Thai",
    "vi": "Vietnamese",
    "ja": "Japanese",
    "tr": "Turkish",
    "hi": "Hindi",
    "ms": "Malay",
    "nl": "Dutch",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "pl": "Polish",
    "cs": "Czech",
    "fil": "Filipino",
    "fa": "Persian",
    "el": "Greek",
    "ro": "Romanian",
    "hu": "Hungarian",
    "mk": "Macedonian",
}

_NAME_TO_CODE = {name.casefold(): code for code, name in LANGUAGE_NAMES.items()}


def normalize_model_language(value: str | None) -> str | None:
    """Map a model language name or known tag to BCP-47; unknown means None."""
    if value is None:
        return None
    normalized = value.strip().casefold()
    if normalized in LANGUAGE_NAMES:
        return normalized
    return _NAME_TO_CODE.get(normalized)


def classify_model_language(
    model_language: str | None, requested: str | None
) -> tuple[str | None, str | None]:
    """Return ``(detected_bcp47, unmapped_name)`` for a model language line.

    A forced language reports no detection. In ``auto`` mode a name outside the
    published list yields ``(None, name)`` so callers can disclose it rather
    than silently dropping the model's answer.
    """
    if requested is not None:
        return None, None
    detected = normalize_model_language(model_language)
    if detected is not None or not (model_language or "").strip():
        return detected, None
    return None, model_language.strip()
