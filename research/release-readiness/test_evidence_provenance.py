"""Model-free tests for release evidence's source identity boundary."""

from __future__ import annotations

import base64
import hashlib
import json
from datetime import UTC, datetime
from types import SimpleNamespace

import evidence_provenance as provenance
import pytest


class RecordedPath(str):
    def __new__(cls, name: str, content: bytes):
        result = super().__new__(cls, name)
        digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).decode().rstrip("=")
        result.hash = SimpleNamespace(mode="sha256", value=digest)
        return result


@pytest.fixture
def installed(monkeypatch, tmp_path):
    root = tmp_path / "site-packages"
    package = root / "example"
    package.mkdir(parents=True)
    source = package / "__init__.py"
    source.write_text("VALUE = 1\n")
    direct_url = {
        "url": "https://example.org/repo.git",
        "vcs_info": {"vcs": "git", "commit_id": "actual-installed-commit"},
    }
    distribution = SimpleNamespace(
        version="1.2.3",
        files=[RecordedPath("example/__init__.py", source.read_bytes())],
        locate_file=lambda file: root / file,
        read_text=lambda name: json.dumps(direct_url) if name == "direct_url.json" else None,
    )
    module = SimpleNamespace(__file__=str(source))
    monkeypatch.setattr(provenance.importlib.metadata, "distribution", lambda _: distribution)
    monkeypatch.setattr(provenance.importlib, "import_module", lambda _: module)
    return source, module, distribution, direct_url


def test_installed_identity_records_actual_metadata_and_source(installed):
    source, _, _, direct_url = installed
    result = provenance.package_identity("example-dist", "example")
    assert result["version"] == "1.2.3"
    assert result["direct_url"] == direct_url
    assert result["source_kind"] == "installed"
    assert result["imported_init"] == str(source)
    assert result["python_source_sha256"] == {
        "__init__.py": hashlib.sha256(source.read_bytes()).hexdigest()
    }


def test_changed_installed_source_cannot_claim_the_git_pin(installed):
    source, _, _, _ = installed
    source.write_text("VALUE = 2\n")
    with pytest.raises(RuntimeError, match="RECORD hash"):
        provenance.package_identity("example-dist", "example")


def test_added_source_cannot_claim_the_git_pin(installed):
    source, _, _, _ = installed
    source.with_name("unrecorded.py").write_text("VALUE = 3\n")
    with pytest.raises(RuntimeError, match="source files differ"):
        provenance.package_identity("example-dist", "example")


def test_pythonpath_override_is_rejected_unless_explicit(installed, tmp_path):
    _, module, _, _ = installed
    override = tmp_path / "baseline" / "example" / "__init__.py"
    override.parent.mkdir(parents=True)
    override.write_text("BASELINE = True\n")
    module.__file__ = str(override)
    with pytest.raises(RuntimeError, match="PYTHONPATH override"):
        provenance.package_identity("example-dist", "example")
    result = provenance.package_identity("example-dist", "example", allow_source_override=True)
    assert result["source_kind"] == "source_override"
    assert result["imported_init"] == str(override)
    assert (
        result["python_source_sha256"]["__init__.py"]
        == hashlib.sha256(override.read_bytes()).hexdigest()
    )


def test_editable_source_is_identified_as_mutable_source(installed):
    source, _, distribution, direct_url = installed
    distribution.files = []
    direct_url.clear()
    direct_url.update({"url": source.parent.parent.as_uri(), "dir_info": {"editable": True}})
    assert provenance.package_identity("example-dist", "example")["source_kind"] == "editable"


def test_date_uses_phoenix_even_when_utc_has_crossed_midnight(monkeypatch):
    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 10, 5, 2, tzinfo=UTC).astimezone(tz)

    monkeypatch.setattr(provenance, "datetime", FixedDateTime)
    assert provenance.evidence_date() == "2026-10-04"


def test_editable_baseline_inside_same_repository_is_still_an_override(installed):
    source, module, distribution, direct_url = installed
    root = source.parent.parent
    distribution.files = []
    direct_url.clear()
    direct_url.update({"url": root.as_uri(), "dir_info": {"editable": True}})
    baseline = root / ".cache" / "baseline" / "example" / "__init__.py"
    baseline.parent.mkdir(parents=True)
    baseline.write_text("BASELINE = True\n")
    module.__file__ = str(baseline)
    with pytest.raises(RuntimeError, match="PYTHONPATH override"):
        provenance.package_identity("example-dist", "example")
    assert (
        provenance.package_identity("example-dist", "example", allow_source_override=True)[
            "source_kind"
        ]
        == "source_override"
    )
