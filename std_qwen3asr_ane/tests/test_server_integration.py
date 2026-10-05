"""Reference-server integration through real discovery and a fake native boundary."""

from __future__ import annotations

import base64
import io
import wave
from types import SimpleNamespace
from typing import ClassVar

import pytest
import test_plugin as plugin_fixtures
from standard_asr import discover_models

from std_qwen3asr_ane import runtime as runtime_module


@pytest.fixture
def fake_server_runtime(monkeypatch: pytest.MonkeyPatch):
    class Runtime:
        instances: ClassVar[list[Runtime]] = []
        max_audio_seconds = 30.0
        head_output: ClassVar[dict[str, str]] = {"kind": "logits"}

        def __init__(self, model_dir):
            self.model_dir = model_dir
            self.calls = []
            self.closed = False
            self.instances.append(self)

        def close(self, *, timeout=5):
            self.closed = True

        @staticmethod
        def _context():
            return SimpleNamespace(reset=lambda: None)

        def new_decoder_context(self):
            return self._context()

        def new_audio_context(self):
            return self._context()

        def transcribe(
            self,
            samples,
            *,
            language,
            max_new_tokens,
            context="",
            prefix_text="",
            **kwargs,
        ):
            self.calls.append(
                {
                    "sample_count": len(samples),
                    "language": language,
                    "max_new_tokens": max_new_tokens,
                    "context": context,
                    "prefix_text": prefix_text,
                    **kwargs,
                }
            )
            raw = "language English<asr_text>hello"
            return SimpleNamespace(
                text="hello",
                language="en",
                raw_model_language="English",
                raw_text=raw,
                token_ids=(1, 2),
                audio_tokens=3,
                timings={"total_seconds": 0.01},
            )

    monkeypatch.setattr(runtime_module, "CoreMLRuntime", Runtime)
    return Runtime


def _wav() -> bytes:
    audio = io.BytesIO()
    with wave.open(audio, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(b"\0\0" * 1600)
    return audio.getvalue()


@pytest.fixture
def server_client(tmp_path, monkeypatch: pytest.MonkeyPatch, fake_server_runtime):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from standard_asr.toolchain.server import create_app

    bundle = plugin_fixtures.make_bundle(tmp_path / "bundle")
    monkeypatch.setenv("STANDARD_ASR_ALLOW_DOWNLOAD", "0")
    monkeypatch.setenv("STANDARD_ASR_STD_QWEN3ASR_ANE__MODEL_DIR", str(bundle))
    monkeypatch.setenv("STANDARD_ASR_STD_QWEN3ASR_ANE__MAX_NEW_TOKENS", "32")
    with TestClient(create_app(registry=discover_models(strict=True))) as client:
        yield client


def test_server_discovers_plugin_without_loading_native_models(
    server_client, fake_server_runtime
) -> None:
    model = "std-qwen3asr-ane/1.7b"
    models = server_client.get("/v1/models")
    assert models.status_code == 200
    assert {item["key"] for item in models.json()} >= {
        model,
        "std-qwen3asr-ane/1.7b-short-dictation",
    }
    schema = server_client.get(f"/v1/params-schema/{model}")
    assert schema.status_code == 200
    assert set(schema.json()["properties"]) >= {
        "max_new_tokens",
        "disable_draft",
        "include_metrics",
    }
    config = server_client.get(f"/v1/config-schema/{model}")
    assert config.status_code == 200
    assert "model_dir" in config.json()["properties"]
    caps = server_client.get(f"/v1/capabilities/{model}")
    assert caps.status_code == 200
    assert caps.json()["streaming_input"]["supported"] is True
    assert caps.json()["streaming"]["finality_level"]["mode"] == "closed"
    assert fake_server_runtime.instances == []


def test_server_transcribes_with_portable_options_and_environment_config(
    server_client, fake_server_runtime
) -> None:
    response = server_client.post(
        "/v1/transcribe:json",
        json={
            "model": "std-qwen3asr-ane/1.7b",
            "audio": base64.b64encode(_wav()).decode("ascii"),
            "options": {"language": "en", "prompt": "Names: Qwen"},
        },
    )
    assert response.status_code == 200, response.text
    result = response.json()["result"]
    assert result["text"] == "hello"
    assert result["detected_language"] is None  # Forced language is not detection.
    call = fake_server_runtime.instances[0].calls[0]
    assert call["max_new_tokens"] == 32
    assert call["language"] == "en"
    assert call["context"] == "Names: Qwen"


def test_server_streams_closed_text_and_audio_progress(server_client, fake_server_runtime) -> None:
    with server_client.websocket_connect("/v1/stream/std-qwen3asr-ane/1.7b") as socket:
        socket.send_json(
            {
                "audio_format": {"encoding": "pcm_s16le", "sample_rate": 16000},
                "options": {"language": "en", "prompt": "Names: Qwen"},
            }
        )
        socket.send_bytes(b"\0\0" * 1600)
        socket.send_text("end")
        events = []
        while True:
            event = socket.receive_json()
            events.append(event)
            if event["type"] in {"done", "error"}:
                break

    assert events[-1]["type"] == "done", events
    closed = [event for event in events if event["type"] == "final"]
    assert len(closed) == 1
    assert closed[0]["text"] == "hello"
    assert closed[0]["stable_text"] == "hello"
    assert "stable_until" not in closed[0]
    assert closed[0]["finality"] == "closed"
    assert closed[0]["segment_id"] == "utterance-0"
    assert closed[0]["audio_processed_until"] is None
    assert closed[0]["extra"]["input_end_seconds"] == pytest.approx(0.1)
    calls = [call for runtime in fake_server_runtime.instances for call in runtime.calls]
    assert calls
    assert all(call["max_new_tokens"] == 32 for call in calls)
    assert all(call["language"] == "en" for call in calls)
    assert all(call["context"] == "Names: Qwen" for call in calls)


def test_server_rejects_discover_only_provider_params(server_client, fake_server_runtime) -> None:
    options = {"provider_params": {"max_new_tokens": 17}}
    response = server_client.post(
        "/v1/transcribe:json",
        json={
            "model": "std-qwen3asr-ane/1.7b",
            "audio": base64.b64encode(_wav()).decode("ascii"),
            "options": options,
        },
    )
    assert response.status_code == 422
    assert "provider_params" in response.text
    with server_client.websocket_connect("/v1/stream/std-qwen3asr-ane/1.7b") as socket:
        socket.send_json(
            {
                "audio_format": {"encoding": "pcm_s16le", "sample_rate": 16000},
                "options": options,
            }
        )
        error = socket.receive_json()
    assert error["type"] == "error"
    assert error["code"] == "bad_request"
    assert "provider_params" in error["message"]
    assert fake_server_runtime.instances == []
