"""Lightweight public-surface probes for the Standard ASR ecosystem audit.

This script does not load Core ML or acquire model artifacts. It checks static
plugin discovery/server projections, the current provider-params wire boundary,
subtitle rendering, and reference-server engine lifetime with a fake engine.
"""

from __future__ import annotations

import base64

from fastapi.testclient import TestClient
from standard_asr import TranscriptionResult, discover_models, to_srt, to_vtt
from standard_asr.toolchain.server import create_app

MODEL = "std-qwen3asr-ane/1.7b"


def probe_installed_plugin_surfaces() -> None:
    registry = discover_models(strict=True)
    assert registry.names() == [
        "std-qwen3asr-ane/1.7b",
        "std-qwen3asr-ane/1.7b-short-dictation",
    ]

    with TestClient(create_app(registry=registry)) as client:
        assert client.get("/v1/health").json() == {"status": "ok"}
        models = client.get("/v1/models")
        assert models.status_code == 200
        assert {item["key"] for item in models.json()} == set(registry.names())

        for surface in ("metadata", "capabilities", "config-schema", "params-schema"):
            response = client.get(f"/v1/{surface}/{MODEL}")
            assert response.status_code == 200, response.text

        params_schema = client.get(f"/v1/params-schema/{MODEL}").json()
        assert "max_new_tokens" in params_schema["properties"]

        response = client.post(
            "/v1/transcribe:json",
            json={
                "model": MODEL,
                "audio": base64.b64encode(b"not-decoded-before-options-validation").decode(),
                "options": {"provider_params": {"max_new_tokens": 32}},
            },
        )
        assert response.status_code == 422, response.text
        assert response.json()["detail"][0]["type"] == "extra_forbidden"


def probe_renderers() -> None:
    result = TranscriptionResult(text="hello", duration=1.25)
    assert "00:00:00,000 --> 00:00:01,250" in to_srt(result)
    assert "00:00:00.000 --> 00:00:01.250" in to_vtt(result)


class _Engine:
    def __init__(self, owner: _Registry) -> None:
        self.owner = owner

    def transcribe(self, audio: object, params: object = None) -> TranscriptionResult:
        return TranscriptionResult(text="ok", duration=0.1)

    def close(self) -> None:
        self.owner.closes += 1


class _Registry:
    """The subset of ModelRegistry used by the REST transcription route."""

    def __init__(self) -> None:
        self.creations = 0
        self.closes = 0

    def create(self, model: str) -> _Engine:
        assert model == MODEL
        self.creations += 1
        return _Engine(self)


def probe_reference_server_engine_lifetime() -> None:
    registry = _Registry()
    with TestClient(create_app(registry=registry)) as client:
        payload = {
            "model": MODEL,
            "audio": base64.b64encode(b"fake-audio").decode(),
        }
        assert client.post("/v1/transcribe:json", json=payload).status_code == 200
        assert client.post("/v1/transcribe:json", json=payload).status_code == 200

    # Current reference-server behavior: one new engine per REST request and no
    # close() call, including when the app lifespan exits.
    assert registry.creations == 2
    assert registry.closes == 0


if __name__ == "__main__":
    probe_installed_plugin_surfaces()
    probe_renderers()
    probe_reference_server_engine_lifetime()
    print("ecosystem probes passed")
