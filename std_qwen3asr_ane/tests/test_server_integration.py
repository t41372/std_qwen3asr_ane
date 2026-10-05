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


def test_discovered_qwen_engine_serves_http_and_websocket(
    tmp_path, monkeypatch: pytest.MonkeyPatch, fake_server_runtime
) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from standard_asr.toolchain.server import create_app

    bundle = plugin_fixtures.make_bundle(tmp_path / "bundle")
    monkeypatch.setenv("STANDARD_ASR_ALLOW_DOWNLOAD", "0")
    monkeypatch.setenv("STANDARD_ASR_STD_QWEN3ASR_ANE__MODEL_DIR", str(bundle))
    model = "std-qwen3asr-ane/1.7b"

    with TestClient(create_app(registry=discover_models(strict=True))) as client:
        assert client.get(f"/v1/readiness/{model}").status_code == 200
        schema = client.get(f"/v1/params-schema/{model}")
        assert schema.status_code == 200
        assert set(schema.json()["properties"]) >= {
            "max_new_tokens",
            "disable_draft",
            "include_metrics",
        }

        response = client.post(
            "/v1/transcribe:json",
            json={
                "model": model,
                "audio": base64.b64encode(_wav()).decode("ascii"),
                "options": {
                    "provider_params": {
                        "max_new_tokens": 32,
                        "disable_draft": True,
                        "include_metrics": True,
                    }
                },
            },
        )
        assert response.status_code == 200, response.text
        result = response.json()["result"]
        assert result["text"] == "hello"
        assert result["extra"]["native"]["token_ids"] == [1, 2]

        with client.websocket_connect(f"/v1/stream/{model}") as socket:
            socket.send_json(
                {
                    "audio_format": {"encoding": "pcm_s16le", "sample_rate": 16000},
                    "options": {"provider_params": {"max_new_tokens": 17}},
                }
            )
            socket.send_bytes(b"\0\0" * 1600)
            socket.send_text("end")
            events = []
            while True:
                event = socket.receive_json()
                events.append(event)
                if event["type"] == "done":
                    break

        closed = [event for event in events if event["type"] == "final"]
        assert len(closed) == 1
        assert closed[0]["text"] == "hello"
        assert closed[0]["stable_text"] == "hello"
        assert "stable_until" not in closed[0]
        assert closed[0]["segment_id"] == "utterance-0"
        assert closed[0]["audio_processed_until"] == pytest.approx(0.1)

    assert len(fake_server_runtime.instances) == 1
    runtime = fake_server_runtime.instances[0]
    assert [call["max_new_tokens"] for call in runtime.calls] == [32, 17]
    assert runtime.closed
