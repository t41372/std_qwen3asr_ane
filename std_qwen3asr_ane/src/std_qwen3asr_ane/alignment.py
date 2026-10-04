"""Optional offline forced alignment backed by Qwen3-ForcedAligner-0.6B."""

from __future__ import annotations

import hashlib
import json
import math
import os
import select
import subprocess
import tempfile
import threading
import time
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Self

import numpy as np

from .worker_environment import (
    WorkerEnvironmentInspection,
    WorkerEnvironmentSpec,
    ensure_worker_environment,
    inspect_worker_environment,
    select_worker_environment_root,
)

ALIGNMENT_MODEL_ID = "Qwen/Qwen3-ForcedAligner-0.6B"
ALIGNMENT_MODEL_REVISION = "c7cbfc2048c462b0d63a45797104fc9db3ad62b7"
ALIGNMENT_WEIGHTS_SIZE = 1_835_544_544
ALIGNMENT_WEIGHTS_SHA256 = "47831d0e82f96b20e9034dba01a075ee06436654719f6a68289e49f1b65ce0e7"
ALIGNMENT_REQUIRED_FILES = (
    "config.json",
    "chat_template.json",
    "generation_config.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "model.safetensors",
)
SUPPORTED_ALIGNMENT_LANGUAGES: dict[str, str] = {
    "zh": "Chinese",
    "en": "English",
    "yue": "Cantonese",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "pt": "Portuguese",
    "ru": "Russian",
    "es": "Spanish",
}
MAX_ALIGNMENT_SECONDS = 300.0
ALIGNMENT_SAMPLE_RATE = 16_000

AlignmentGranularity = Literal["word", "char"]
AlignmentState = Literal["missing", "incomplete", "corrupt", "unknown", "ready"]

_MODEL_DIRECTORY = f"model-{ALIGNMENT_MODEL_REVISION[:8]}"
_LEGACY_RUNTIME_DIRECTORY = "runtime-v1"
_MODEL_RECEIPT = "alignment-model.json"
_WORKER_READY_TIMEOUT = 180.0
_ALIGNMENT_ENVIRONMENT = WorkerEnvironmentSpec(
    name="qwen3-forced-aligner-cpu",
    version=1,
    dependencies=(
        "qwen-asr==0.0.6",
        "torch==2.14.0",
        "transformers==4.57.6",
        "numpy==2.5.3",
        "huggingface-hub==0.36.2",
        "safetensors==0.8.0",
    ),
)


class AlignmentError(RuntimeError):
    """The optional aligner could not produce a valid measured alignment."""


class AlignmentCancelled(AlignmentError):
    """Alignment was cancelled and its isolated worker was stopped."""


@dataclass(frozen=True)
class AlignmentSpan:
    """One measured unit and its exact half-open range in the source text."""

    text: str
    start_time: float
    end_time: float
    source_start: int
    source_end: int


@dataclass(frozen=True)
class ForcedAlignerInspection:
    """Read-only state for an optional aligner bundle."""

    state: AlignmentState
    location: Path
    model_path: Path
    runtime_path: Path
    model_state: AlignmentState
    runtime: WorkerEnvironmentInspection
    provenance: dict[str, Any] | None
    detail: str | None = None


def alignment_model_path(root: str | Path) -> Path:
    return Path(root).expanduser() / _MODEL_DIRECTORY


def alignment_runtime_path(root: str | Path) -> Path:
    return select_worker_environment_root(
        Path(root).expanduser(),
        "runtime",
        _ALIGNMENT_ENVIRONMENT,
        legacy_names=(_LEGACY_RUNTIME_DIRECTORY,),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _inspect_model(model_path: Path) -> tuple[AlignmentState, dict[str, Any] | None, str | None]:
    if not model_path.exists():
        return "missing", None, "Pinned alignment model is absent"
    if not model_path.is_dir():
        return "corrupt", None, "Alignment model path is not a directory"
    missing = [name for name in ALIGNMENT_REQUIRED_FILES if not (model_path / name).is_file()]
    if missing:
        return "incomplete", None, f"Missing alignment model files: {', '.join(missing)}"
    receipt_path = model_path / _MODEL_RECEIPT
    try:
        provenance = json.loads(receipt_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return "unknown", None, "Alignment model provenance is absent"
    except (OSError, UnicodeError, json.JSONDecodeError):
        return "corrupt", None, "Alignment model provenance is invalid"
    if not isinstance(provenance, dict):
        return "corrupt", None, "Alignment model provenance must be a JSON object"
    expected_source = {
        "schema_version": 1,
        "model_id": ALIGNMENT_MODEL_ID,
        "revision": ALIGNMENT_MODEL_REVISION,
        "weights_size": ALIGNMENT_WEIGHTS_SIZE,
        "weights_sha256": ALIGNMENT_WEIGHTS_SHA256,
        "required_files": list(ALIGNMENT_REQUIRED_FILES),
        "license": "Apache-2.0",
    }
    if (
        any(provenance.get(key) != value for key, value in expected_source.items())
        or not isinstance(provenance.get("source"), str)
        or not provenance["source"]
    ):
        return "unknown", provenance, "Alignment model provenance does not match the pin"
    weights = model_path / "model.safetensors"
    if weights.stat().st_size != ALIGNMENT_WEIGHTS_SIZE:
        return "corrupt", provenance, "Alignment model weight size does not match the pin"
    return "ready", provenance, None


def verify_alignment_model(root: str | Path) -> None:
    """Explicitly hash the large weight file against the immutable model pin."""
    model_path = alignment_model_path(root)
    state, _, detail = _inspect_model(model_path)
    if state != "ready":
        raise AlignmentError(detail or f"Alignment model is {state}")
    if _sha256(model_path / "model.safetensors") != ALIGNMENT_WEIGHTS_SHA256:
        raise AlignmentError("Alignment model weight digest does not match the pin")


def inspect_forced_aligner(root: str | Path) -> ForcedAlignerInspection:
    """Inspect model provenance and worker availability without changing state."""
    location = Path(root).expanduser()
    model_path = alignment_model_path(location)
    runtime_path = alignment_runtime_path(location)
    model_state, provenance, model_detail = _inspect_model(model_path)
    runtime = inspect_worker_environment(runtime_path, _ALIGNMENT_ENVIRONMENT)
    if model_state == "ready" and runtime.ready:
        state: AlignmentState = "ready"
        detail = None
    elif model_state == "corrupt":
        state = "corrupt"
        detail = model_detail
    elif model_state == "unknown":
        state = "unknown"
        detail = model_detail
    elif model_state == "missing" and runtime.reason == "missing" and not location.exists():
        state = "missing"
        detail = "Alignment bundle is absent"
    else:
        state = "incomplete"
        detail = model_detail or f"Alignment worker runtime is {runtime.reason}"
    return ForcedAlignerInspection(
        state=state,
        location=location,
        model_path=model_path,
        runtime_path=runtime_path,
        model_state=model_state,
        runtime=runtime,
        provenance=provenance,
        detail=detail,
    )


def _worker_command(python: Path, *arguments: str) -> list[str]:
    """Expose plugin code without exposing the parent environment's dependencies."""
    package_dir = Path(__file__).resolve().parent
    package_init = package_dir / "__init__.py"
    bootstrap = (
        "import importlib.util,runpy,sys;"
        "spec=importlib.util.spec_from_file_location("
        f"'std_qwen3asr_ane',{str(package_init)!r},"
        f"submodule_search_locations=[{str(package_dir)!r}]);"
        "package=importlib.util.module_from_spec(spec);"
        "sys.modules[spec.name]=package;"
        "spec.loader.exec_module(package);"
        "runpy.run_module('std_qwen3asr_ane.alignment_worker',run_name='__main__')"
    )
    return [str(python), "-I", "-u", "-c", bootstrap, *arguments]


def _worker_environment(*, offline: bool) -> dict[str, str]:
    environment = os.environ.copy()
    # The bootstrap exposes only this package. Adding its site-packages parent
    # to PYTHONPATH would let the server environment shadow the worker's pinned
    # third-party dependencies and defeat the isolation this process provides.
    environment.pop("PYTHONPATH", None)
    if offline:
        environment["HF_HUB_OFFLINE"] = "1"
        environment["TRANSFORMERS_OFFLINE"] = "1"
    return environment


def acquire_forced_aligner(
    root: str | Path,
    *,
    allow_downloads: bool,
    progress: Callable[[str], None] | None = None,
) -> ForcedAlignerInspection:
    """Explicitly acquire the pinned model and isolated CPU worker runtime."""
    root = Path(root).expanduser().resolve()
    python = ensure_worker_environment(
        alignment_runtime_path(root),
        _ALIGNMENT_ENVIRONMENT,
        allow_downloads=allow_downloads,
        progress=progress,
    )
    if _inspect_model(alignment_model_path(root))[0] != "ready":
        if progress is not None:
            progress("Acquiring pinned forced-alignment model")
        command = _worker_command(
            python,
            "--acquire",
            str(alignment_model_path(root)),
            "--allow-downloads" if allow_downloads else "--offline",
        )
        subprocess.run(command, check=True, env=_worker_environment(offline=not allow_downloads))
    else:
        verify_alignment_model(root)
    inspection = inspect_forced_aligner(root)
    if inspection.state != "ready":
        raise AlignmentError(inspection.detail or f"Alignment bundle is {inspection.state}")
    return inspection


def _is_kept_character(character: str) -> bool:
    return character == "'" or unicodedata.category(character)[0] in {"L", "N"}


def _character_input(text: str) -> str:
    return " ".join(character for character in text if _is_kept_character(character))


def _map_source_ranges(text: str, item_texts: list[str]) -> list[tuple[int, int]]:
    kept = [
        (index, character) for index, character in enumerate(text) if _is_kept_character(character)
    ]
    item_characters = [
        character for item in item_texts for character in item if _is_kept_character(character)
    ]
    if item_characters != [character for _, character in kept]:
        raise AlignmentError("Aligned items do not cover the exact source transcript")
    cursor = 0
    ranges: list[tuple[int, int]] = []
    for item in item_texts:
        expected = [character for character in item if _is_kept_character(character)]
        if not expected:
            raise AlignmentError("The aligner returned an item without alignable characters")
        selected = kept[cursor : cursor + len(expected)]
        ranges.append((selected[0][0], selected[-1][0] + 1))
        cursor += len(expected)
    return ranges


def _cancelled(check: Callable[[], bool] | threading.Event | None) -> bool:
    if check is None:
        return False
    return check.is_set() if isinstance(check, threading.Event) else bool(check())


class ForcedAligner:
    """Synchronous owner of one lazily started, persistent alignment worker."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser()
        self._lock = threading.Lock()
        self._process: subprocess.Popen[bytes] | None = None
        self._stderr: tempfile._TemporaryFileWrapper[bytes] | Any | None = None
        self._closed = False

    def _failure_detail(self) -> str:
        if self._stderr is None:
            return ""
        try:
            self._stderr.seek(0)
            return self._stderr.read().decode("utf-8", errors="replace")[-4000:]
        except OSError:
            return ""

    def _stop_worker(self) -> None:
        process, self._process = self._process, None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        if process is not None:
            for stream in (process.stdin, process.stdout):
                if stream is not None:
                    stream.close()
        if self._stderr is not None:
            self._stderr.close()
            self._stderr = None

    def _read_response(
        self,
        *,
        timeout: float | None,
        cancelled: Callable[[], bool] | threading.Event | None,
    ) -> dict[str, Any]:
        assert self._process is not None and self._process.stdout is not None
        deadline = None if timeout is None else time.monotonic() + timeout
        descriptor = self._process.stdout.fileno()
        while True:
            if _cancelled(cancelled):
                self._stop_worker()
                raise AlignmentCancelled("Forced alignment was cancelled")
            if self._process.poll() is not None:
                detail = self._failure_detail()
                self._stop_worker()
                raise AlignmentError(f"Alignment worker exited unexpectedly. {detail}".strip())
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                self._stop_worker()
                raise AlignmentError("Alignment worker did not become ready before its timeout")
            wait = 0.05 if remaining is None else min(0.05, remaining)
            readable, _, _ = select.select([descriptor], [], [], wait)
            if readable:
                line = self._process.stdout.readline()
                if not line:
                    continue
                try:
                    response = json.loads(line)
                except (UnicodeError, json.JSONDecodeError) as error:
                    self._stop_worker()
                    raise AlignmentError("Alignment worker returned an invalid response") from error
                if not isinstance(response, dict):
                    self._stop_worker()
                    raise AlignmentError("Alignment worker returned a non-object response")
                return response

    def _ensure_worker(
        self, cancelled: Callable[[], bool] | threading.Event | None
    ) -> subprocess.Popen[bytes]:
        if self._closed:
            raise AlignmentError("Forced aligner is closed")
        if self._process is not None and self._process.poll() is None:
            return self._process
        inspection = inspect_forced_aligner(self.root)
        if inspection.state != "ready" or inspection.runtime.python_executable is None:
            raise AlignmentError(inspection.detail or f"Alignment bundle is {inspection.state}")
        # This file belongs to the persistent worker and is closed in _stop_worker.
        self._stderr = tempfile.TemporaryFile(mode="w+b")  # noqa: SIM115
        self._process = subprocess.Popen(
            _worker_command(
                inspection.runtime.python_executable,
                "--serve",
                str(inspection.model_path),
            ),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr,
            env=_worker_environment(offline=True),
        )
        response = self._read_response(timeout=_WORKER_READY_TIMEOUT, cancelled=cancelled)
        if response.get("status") != "ready":
            detail = response.get("message") or self._failure_detail()
            self._stop_worker()
            raise AlignmentError(f"Alignment worker failed to start: {detail}")
        return self._process

    def align(
        self,
        audio: np.ndarray,
        text: str,
        language: str,
        *,
        granularity: AlignmentGranularity = "word",
        cancelled: Callable[[], bool] | threading.Event | None = None,
    ) -> list[AlignmentSpan]:
        """Align one canonical 16 kHz mono float32 array to its exact transcript."""
        if granularity not in ("word", "char"):
            raise ValueError("Alignment granularity must be 'word' or 'char'")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Alignment transcript must be non-empty")
        language_code = language.split("-", 1)[0].casefold()
        try:
            language_name = SUPPORTED_ALIGNMENT_LANGUAGES[language_code]
        except KeyError as error:
            raise ValueError(f"Forced alignment does not support language: {language}") from error
        samples = np.asarray(audio)
        if samples.dtype != np.float32 or samples.ndim != 1:
            raise ValueError("Alignment audio must be a mono float32 array")
        if not samples.size or not np.isfinite(samples).all():
            raise ValueError("Alignment audio must be non-empty and finite")
        duration = samples.size / ALIGNMENT_SAMPLE_RATE
        if duration > MAX_ALIGNMENT_SECONDS:
            raise ValueError(
                f"Alignment audio is {duration:.3f}s; the model limit is {MAX_ALIGNMENT_SECONDS:.0f}s"
            )
        model_text = text if granularity == "word" else _character_input(text)
        if not model_text:
            raise ValueError("Alignment transcript has no characters supported by the model")
        contiguous = np.ascontiguousarray(samples)

        with self._lock:
            process = self._ensure_worker(cancelled)
            assert process.stdin is not None
            request = {
                "operation": "align",
                "sample_rate": ALIGNMENT_SAMPLE_RATE,
                "sample_count": int(contiguous.size),
                "audio_bytes": int(contiguous.nbytes),
                "text": model_text,
                "language": language_name,
            }
            try:
                process.stdin.write(json.dumps(request, separators=(",", ":")).encode() + b"\n")
                process.stdin.write(memoryview(contiguous).cast("B"))
                process.stdin.flush()
            except (BrokenPipeError, OSError) as error:
                detail = self._failure_detail()
                self._stop_worker()
                raise AlignmentError(
                    f"Could not send alignment request. {detail}".strip()
                ) from error
            response = self._read_response(timeout=None, cancelled=cancelled)
            if response.get("status") != "ok":
                raise AlignmentError(str(response.get("message") or "Alignment worker failed"))

        raw_items = response.get("items")
        if not isinstance(raw_items, list):
            raise AlignmentError("Alignment worker response has no item list")
        item_texts = [str(item.get("text", "")) for item in raw_items if isinstance(item, dict)]
        if len(item_texts) != len(raw_items):
            raise AlignmentError("Alignment worker returned a malformed item")
        ranges = _map_source_ranges(text, item_texts)
        spans: list[AlignmentSpan] = []
        previous_end = 0.0
        for item, item_text, source_range in zip(raw_items, item_texts, ranges, strict=True):
            try:
                start = float(item["start_time"])
                end = float(item["end_time"])
            except (KeyError, TypeError, ValueError) as error:
                raise AlignmentError("Alignment worker returned malformed times") from error
            if not all(math.isfinite(value) for value in (start, end)):
                raise AlignmentError("Alignment worker returned non-finite times")
            if start < 0 or end < start or start < previous_end or end > duration + 1e-3:
                raise AlignmentError("Alignment worker returned unordered or out-of-bounds times")
            source_start, source_end = source_range
            spans.append(
                AlignmentSpan(
                    text=text[source_start:source_end],
                    start_time=start,
                    end_time=min(end, duration),
                    source_start=source_start,
                    source_end=source_end,
                )
            )
            previous_end = end
        return spans

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._stop_worker()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
