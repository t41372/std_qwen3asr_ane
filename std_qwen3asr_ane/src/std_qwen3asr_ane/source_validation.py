"""Checkpoint source inspection and content-bound provenance.

Status callers use :func:`inspect_source_checkpoint`, which reads only small
metadata and file sizes. Explicit acquisition/conversion uses
:func:`verify_source_checkpoint` to hash every source payload before attributing
an immutable upstream revision to generated artifacts.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

SourceState = Literal["missing", "incomplete", "corrupt", "ready"]
HashProgress = Callable[[int, int], None]

_REQUIRED_METADATA = (
    "config.json",
    "chat_template.json",
    "generation_config.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
)
_PROVENANCE_SCHEMA = 1
_BUFFER_SIZE = 1024 * 1024
_HEX = frozenset("0123456789abcdef")


class SourceValidationError(ValueError):
    """A checkpoint source cannot safely be attributed or converted."""


@dataclass(frozen=True)
class SourceFileRecord:
    path: str
    size_bytes: int
    sha256: str

    def to_json(self) -> dict[str, object]:
        return {
            "path": self.path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class SourceInspection:
    state: SourceState
    reason: str
    model_id: str | None = None
    revision: str | None = None
    payload_paths: tuple[Path, ...] = ()
    content_sha256: str | None = None
    content_recorded: bool = False

    @property
    def usable(self) -> bool:
        return self.state == "ready"


@dataclass(frozen=True)
class VerifiedSource:
    root: Path
    model_id: str
    revision: str
    files: tuple[SourceFileRecord, ...]
    content_sha256: str

    def provenance(self) -> dict[str, object]:
        return {
            "schema_version": _PROVENANCE_SCHEMA,
            "model_id": self.model_id,
            "revision": self.revision,
            "content": {
                "algorithm": "sha256",
                "digest": self.content_sha256,
                "files": [record.to_json() for record in self.files],
            },
        }


def _safe_path(root: Path, relative: object, field: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise SourceValidationError(f"{field} must be a nonempty relative path")
    path = (root / relative).resolve()
    if path == root or not path.is_relative_to(root):
        raise SourceValidationError(f"{field} escapes the source directory")
    return path


def _read_json(path: Path, field: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SourceValidationError(f"cannot read {field}") from error
    if not isinstance(value, Mapping):
        raise SourceValidationError(f"{field} must contain a JSON object")
    return value


def _required_payloads(root: Path) -> tuple[Path, ...]:
    required = [_safe_path(root, name, name) for name in _REQUIRED_METADATA]
    tokenizer_json = root / "tokenizer.json"
    if tokenizer_json.is_file():
        required.append(tokenizer_json.resolve())
    else:
        required.extend(
            (_safe_path(root, "vocab.json", "vocab.json"), _safe_path(root, "merges.txt", "merges.txt"))
        )

    index_path = root / "model.safetensors.index.json"
    single_path = root / "model.safetensors"
    if index_path.is_file():
        index = _read_json(index_path, "model.safetensors.index.json")
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, Mapping) or not weight_map:
            raise SourceValidationError("model.safetensors.index.json has no weight_map")
        shards: set[Path] = set()
        for name, relative in weight_map.items():
            if not isinstance(name, str) or not name:
                raise SourceValidationError("weight_map contains an invalid tensor name")
            shard = _safe_path(root, relative, f"weight_map.{name}")
            if shard.suffix != ".safetensors":
                raise SourceValidationError("weight_map entries must name safetensors shards")
            shards.add(shard)
        required.append(index_path.resolve())
        required.extend(sorted(shards))
    elif single_path.is_file():
        required.append(single_path.resolve())
    else:
        raise SourceValidationError("checkpoint has no safetensors file or index")

    for path in required:
        try:
            if not path.is_file() or path.stat().st_size == 0:
                raise SourceValidationError(f"required source payload is missing or empty: {path.name}")
        except OSError as error:
            raise SourceValidationError(f"cannot inspect required source payload: {path.name}") from error
    return tuple(dict.fromkeys(required))


def _parse_recorded_content(
    root: Path, provenance: Mapping[str, Any], required: Iterable[Path]
) -> tuple[str | None, bool]:
    content = provenance.get("content")
    if content is None:
        return None, False
    if not isinstance(content, Mapping) or content.get("algorithm") != "sha256":
        raise SourceValidationError("source content provenance is malformed")
    digest = content.get("digest")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in _HEX for character in digest)
    ):
        raise SourceValidationError("source content digest is malformed")
    records = content.get("files")
    if not isinstance(records, list) or not records:
        raise SourceValidationError("source content file list is missing")
    recorded_paths: set[Path] = set()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise SourceValidationError(f"source content file {index} is malformed")
        path = _safe_path(root, record.get("path"), f"content.files[{index}].path")
        size = record.get("size_bytes")
        sha256 = record.get("sha256")
        if type(size) is not int or size < 0:
            raise SourceValidationError(f"content.files[{index}].size_bytes is invalid")
        if (
            not isinstance(sha256, str)
            or len(sha256) != 64
            or any(character not in _HEX for character in sha256)
        ):
            raise SourceValidationError(f"content.files[{index}].sha256 is invalid")
        try:
            if not path.is_file() or path.stat().st_size != size:
                raise SourceValidationError(f"recorded source payload changed: {path.name}")
        except OSError as error:
            raise SourceValidationError(f"cannot inspect recorded source payload: {path.name}") from error
        if path in recorded_paths:
            raise SourceValidationError(f"content provenance repeats a path: {path.name}")
        recorded_paths.add(path)
    missing = set(required) - recorded_paths
    if missing:
        names = ", ".join(sorted(path.name for path in missing))
        raise SourceValidationError(f"content provenance omits required payloads: {names}")
    return digest, True


def inspect_source_checkpoint(
    root: Path,
    *,
    expected_model_id: str | None = None,
    expected_revision: str | None = None,
    expected_content_sha256: str | None = None,
) -> SourceInspection:
    """Inspect source identity and file closure without hashing large payloads."""
    root = Path(root).expanduser().resolve()
    provenance_path = root / "source.json"
    if not provenance_path.is_file():
        return SourceInspection(
            "incomplete" if root.exists() else "missing",
            "source.json is missing",
        )
    try:
        provenance = _read_json(provenance_path, "source.json")
        model_id = provenance.get("model_id")
        revision = provenance.get("revision")
        if not isinstance(model_id, str) or not model_id:
            raise SourceValidationError("source model_id is missing")
        if not isinstance(revision, str) or not revision:
            raise SourceValidationError("source revision is missing")
        if expected_model_id is not None and model_id != expected_model_id:
            raise SourceValidationError("source model_id differs from the required checkpoint")
        if expected_revision is not None and revision != expected_revision:
            raise SourceValidationError("source revision differs from the required checkpoint")
        required = _required_payloads(root)
        content_sha256, recorded = _parse_recorded_content(root, provenance, required)
        if (
            expected_content_sha256 is not None
            and content_sha256 is not None
            and content_sha256 != expected_content_sha256
        ):
            raise SourceValidationError("source content differs from the required checkpoint")
    except SourceValidationError as error:
        return SourceInspection("corrupt", str(error))
    return SourceInspection(
        "ready",
        "source checkpoint has complete local metadata and payload closure",
        model_id=model_id,
        revision=revision,
        payload_paths=required,
        content_sha256=content_sha256,
        content_recorded=recorded,
    )


def _content_paths(root: Path) -> tuple[Path, ...]:
    paths: list[Path] = []
    for path in root.rglob("*"):
        if path.name in {"source.json", "source.json.tmp"} or any(
            part.startswith(".") for part in path.relative_to(root).parts
        ):
            continue
        resolved = path.resolve()
        if not resolved.is_relative_to(root):
            raise SourceValidationError("source payload symlink escapes the source directory")
        if path.is_file():
            paths.append(resolved)
    if not paths:
        raise SourceValidationError("source directory has no content payloads")
    return tuple(sorted(set(paths)))


def _recorded_content_paths(
    root: Path, provenance: Mapping[str, Any]
) -> tuple[Path, ...] | None:
    content = provenance.get("content")
    if not isinstance(content, Mapping):
        return None
    records = content.get("files")
    if not isinstance(records, list):
        return None
    return tuple(
        _safe_path(root, record["path"], f"content.files[{index}].path")
        for index, record in enumerate(records)
    )


def _content_digest(records: Iterable[SourceFileRecord]) -> str:
    payload = json.dumps(
        [record.to_json() for record in records],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _hash_content(
    root: Path,
    paths: Iterable[Path],
    *,
    progress: HashProgress | None = None,
) -> tuple[tuple[SourceFileRecord, ...], str]:
    paths = tuple(paths)
    snapshots = {path: path.stat() for path in paths}
    total = sum(stat.st_size for stat in snapshots.values())
    completed = 0
    records: list[SourceFileRecord] = []
    if progress is not None:
        progress(0, total)
    for path in paths:
        sha256 = hashlib.sha256()
        size = 0
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(_BUFFER_SIZE), b""):
                sha256.update(chunk)
                size += len(chunk)
                completed += len(chunk)
                if completed > total:
                    raise SourceValidationError("source payload changed during content verification")
                if progress is not None:
                    progress(completed, total)
        final_stat = path.stat()
        initial_stat = snapshots[path]
        if (
            size != initial_stat.st_size
            or final_stat.st_size != initial_stat.st_size
            or final_stat.st_mtime_ns != initial_stat.st_mtime_ns
            or final_stat.st_ctime_ns != initial_stat.st_ctime_ns
        ):
            raise SourceValidationError("source payload changed during content verification")
        records.append(
            SourceFileRecord(
                path=path.relative_to(root).as_posix(),
                size_bytes=size,
                sha256=sha256.hexdigest(),
            )
        )
    if completed != total:
        raise SourceValidationError("source payload changed during content verification")
    record_tuple = tuple(records)
    return record_tuple, _content_digest(record_tuple)


def verify_source_checkpoint(
    root: Path,
    *,
    expected_model_id: str | None = None,
    expected_revision: str | None = None,
    expected_content_sha256: str | None = None,
    progress: HashProgress | None = None,
) -> VerifiedSource:
    """Hash the complete local source and verify any recorded provenance."""
    root = Path(root).expanduser().resolve()
    inspection = inspect_source_checkpoint(
        root,
        expected_model_id=expected_model_id,
        expected_revision=expected_revision,
        expected_content_sha256=expected_content_sha256,
    )
    if not inspection.usable or inspection.model_id is None or inspection.revision is None:
        raise SourceValidationError(inspection.reason)
    provenance = _read_json(root / "source.json", "source.json")
    paths = _recorded_content_paths(root, provenance) or _content_paths(root)
    records, content_sha256 = _hash_content(root, paths, progress=progress)
    if expected_content_sha256 is not None and content_sha256 != expected_content_sha256:
        raise SourceValidationError("source content differs from the required checkpoint")
    recorded_content = provenance.get("content")
    if isinstance(recorded_content, Mapping):
        if recorded_content.get("digest") != content_sha256:
            raise SourceValidationError("source content digest does not match local payloads")
        expected_records = recorded_content.get("files")
        if expected_records != [record.to_json() for record in records]:
            raise SourceValidationError("source content file records do not match local payloads")
    return VerifiedSource(
        root=root,
        model_id=inspection.model_id,
        revision=inspection.revision,
        files=records,
        content_sha256=content_sha256,
    )


def write_source_provenance(
    root: Path,
    *,
    model_id: str,
    revision: str,
    progress: HashProgress | None = None,
) -> VerifiedSource:
    """Hash a completed download and atomically publish its provenance marker."""
    root = Path(root).expanduser().resolve()
    _required_payloads(root)
    records, content_sha256 = _hash_content(root, _content_paths(root), progress=progress)
    verified = VerifiedSource(root, model_id, revision, records, content_sha256)
    destination = root / "source.json"
    temporary = destination.with_name("source.json.tmp")
    temporary.write_text(json.dumps(verified.provenance(), indent=2) + "\n")
    temporary.replace(destination)
    return verified
