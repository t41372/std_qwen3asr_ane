"""Verify the installed wheel through the real Standard ASR HTTP/WS server.

The default mode builds a uniquely located wheel, creates an isolated virtual
environment, installs the wheel with its server and diarization extras, and
re-executes this file with that environment's Python. The installed mode runs
Uvicorn on loopback and uses real Core ML, forced alignment, and diarization.

Lifecycle wrappers only count construction, cancellation, and close calls.
They delegate to the original implementation and never replace inference.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import http.client
import importlib.metadata
import json
import os
import platform
import socket
import subprocess
import threading
import time
import traceback
import urllib.parse
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DATE = "2026-10-02"
MODEL_GENERAL = "std-qwen3asr-ane/1.7b"
MODEL_SHORT = "std-qwen3asr-ane/1.7b-short-dictation"
EXPECTED_EN = (
    "Uh huh. Oh yeah, yeah. He wasn't even that big when I started listening to him, "
    "but and his solo music didn't do overly well, but he did very well when he started "
    "writing for other people."
)
EXPECTED_ZH = "甚至出现交易几乎停滞的情况。"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def _run_logged(command: list[str], *, cwd: Path, log: Path, env: dict[str, str]) -> None:
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log.write_text(completed.stdout)
    if completed.returncode:
        raise RuntimeError(
            f"Command failed with exit code {completed.returncode} after "
            f"{time.perf_counter() - started:.1f}s; see {log}"
        )


def _orchestrate(args: argparse.Namespace) -> int:
    root = args.root.resolve()
    output = args.output.resolve()
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + f"-{os.getpid()}"
    run_dir = root / "artifacts/release-readiness" / f"server-native-{run_id}"
    dist_dir = run_dir / "dist"
    environment = run_dir / "venv"
    dist_dir.mkdir(parents=True)

    command_env = os.environ.copy()
    command_env["UV_CACHE_DIR"] = str(root / ".cache/uv")
    command_env["STANDARD_ASR_ALLOW_DOWNLOAD"] = "0"

    _run_logged(
        ["uv", "build", "--wheel", "--out-dir", str(dist_dir)],
        cwd=root,
        log=run_dir / "build.log",
        env=command_env,
    )
    wheels = sorted(dist_dir.glob("std_qwen3asr_ane-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"Expected one wheel in {dist_dir}, found {len(wheels)}")
    wheel = wheels[0]

    _run_logged(
        ["uv", "venv", "--python", args.python, str(environment)],
        cwd=root,
        log=run_dir / "venv.log",
        env=command_env,
    )
    interpreter = environment / "bin/python"
    _run_logged(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(interpreter),
            f"{wheel}[server,diarization]",
        ],
        cwd=root,
        log=run_dir / "install.log",
        env=command_env,
    )

    installed_command = [
        str(interpreter),
        str(Path(__file__).resolve()),
        "--run-installed",
        "--root",
        str(root),
        "--output",
        str(output),
        "--run-dir",
        str(run_dir),
        "--wheel",
        str(wheel),
        "--wheel-sha256",
        _sha256(wheel),
    ]
    _run_logged(
        installed_command,
        cwd=run_dir,
        log=run_dir / "verify.log",
        env=command_env,
    )
    print(output)
    return 0


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _wait_for_server(port: int, server: Any, thread: threading.Thread) -> None:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if getattr(server, "started", False):
            return
        if not thread.is_alive():
            raise RuntimeError("Uvicorn exited before accepting connections")
        time.sleep(0.02)
    raise TimeoutError(f"Uvicorn did not start on port {port}")


def _request_json(
    port: int,
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
    *,
    timeout: float = 180,
) -> tuple[int, dict[str, Any] | list[Any]]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    body = None if payload is None else json.dumps(payload).encode()
    headers = {} if body is None else {"Content-Type": "application/json"}
    try:
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        raw = response.read()
        decoded = json.loads(raw)
        return response.status, decoded
    finally:
        connection.close()


def _audio_duration(path: Path) -> float:
    import soundfile as sf

    info = sf.info(path)
    assert info.channels == 1, f"Expected a mono fixture: {path}"
    return info.frames / info.samplerate


def _pcm16_16khz(path: Path) -> bytes:
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    samples, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    assert samples.ndim == 1, f"Expected a mono fixture: {path}"
    if sample_rate != 16_000:
        divisor = int(np.gcd(sample_rate, 16_000))
        samples = resample_poly(
            samples, 16_000 // divisor, sample_rate // divisor
        )
    pcm = np.clip(np.rint(samples * 32768.0), -32768, 32767).astype("<i2")
    return pcm.tobytes()


def _assert_bounded(items: list[dict[str, Any]], duration: float) -> dict[str, Any]:
    previous_start = 0.0
    for item in items:
        start = item.get("start")
        end = item.get("end")
        assert isinstance(start, (int, float)) and isinstance(end, (int, float)), item
        assert 0 <= start <= end <= duration + 1e-3, item
        assert start >= previous_start, item
        previous_start = start
    return {
        "count": len(items),
        "first_start": items[0]["start"] if items else None,
        "last_end": items[-1]["end"] if items else None,
        "bounded_and_ordered": True,
    }


def _stream(
    port: int,
    model: str,
    pcm: bytes,
    options: dict[str, Any],
) -> list[dict[str, Any]]:
    from websockets.sync.client import connect

    events: list[dict[str, Any]] = []
    with connect(f"ws://127.0.0.1:{port}/v1/stream/{model}", open_timeout=30) as websocket:
        websocket.send(
            json.dumps(
                {
                    "audio_format": {"encoding": "pcm_s16le", "sample_rate": 16_000},
                    "options": options,
                }
            )
        )
        websocket.send(pcm)
        websocket.send("end")
        while True:
            event = json.loads(websocket.recv(timeout=180))
            events.append(event)
            if event["type"] in {"done", "error"}:
                break
    return events


def _record_package(name: str) -> dict[str, Any]:
    distribution = importlib.metadata.distribution(name)
    direct_url = distribution.read_text("direct_url.json")
    provenance = json.loads(direct_url) if direct_url else None
    if provenance is not None and provenance.get("url", "").startswith("file:"):
        provenance = {
            "source": "local_wheel",
            "filename": Path(urllib.parse.urlsplit(provenance["url"]).path).name,
        }
    return {
        "version": distribution.version,
        "direct_url": provenance,
    }


class _Instrumentation:
    def __init__(self) -> None:
        self.engines: list[Any] = []
        self.runtimes: list[Any] = []
        self.aligners: list[Any] = []
        self.auxiliary: list[Any] = []
        self.sessions: list[Any] = []
        self.engine_close_calls: Counter[int] = Counter()
        self.runtime_close_calls: Counter[int] = Counter()
        self.aligner_close_calls: Counter[int] = Counter()
        self.auxiliary_close_calls: Counter[int] = Counter()
        self.session_cancel_calls: Counter[int] = Counter()
        self.session_close_calls: Counter[int] = Counter()
        self.cancellation_armed = threading.Event()
        self.cancellation_native_started = threading.Event()

    def install(self) -> None:
        from std_qwen3asr_ane.alignment import ForcedAligner
        from std_qwen3asr_ane.auxiliary import AuxiliaryModels
        from std_qwen3asr_ane.plugin import Qwen3ASREngine
        from std_qwen3asr_ane.runtime import CoreMLRuntime
        from std_qwen3asr_ane.streaming import Qwen3ASRSession

        instrumentation = self

        engine_init = Qwen3ASREngine.__init__
        engine_close = Qwen3ASREngine.close
        runtime_init = CoreMLRuntime.__init__
        runtime_close = CoreMLRuntime.close
        aligner_init = ForcedAligner.__init__
        aligner_close = ForcedAligner.close
        auxiliary_init = AuxiliaryModels.__init__
        auxiliary_close = AuxiliaryModels.close
        session_init = Qwen3ASRSession.__init__
        session_cancel = Qwen3ASRSession.cancel
        session_close = Qwen3ASRSession._close
        recognize = Qwen3ASRSession._recognize_via_engine

        def counted_engine_init(instance: Any, *args: Any, **kwargs: Any) -> None:
            engine_init(instance, *args, **kwargs)
            instrumentation.engines.append(instance)

        def counted_engine_close(instance: Any, *args: Any, **kwargs: Any) -> None:
            instrumentation.engine_close_calls[id(instance)] += 1
            engine_close(instance, *args, **kwargs)

        def counted_runtime_init(instance: Any, *args: Any, **kwargs: Any) -> None:
            runtime_init(instance, *args, **kwargs)
            instrumentation.runtimes.append(instance)

        def counted_runtime_close(instance: Any, *args: Any, **kwargs: Any) -> None:
            instrumentation.runtime_close_calls[id(instance)] += 1
            runtime_close(instance, *args, **kwargs)

        def counted_aligner_init(instance: Any, *args: Any, **kwargs: Any) -> None:
            aligner_init(instance, *args, **kwargs)
            instrumentation.aligners.append(instance)

        def counted_aligner_close(instance: Any, *args: Any, **kwargs: Any) -> None:
            instrumentation.aligner_close_calls[id(instance)] += 1
            aligner_close(instance, *args, **kwargs)

        def counted_auxiliary_init(instance: Any, *args: Any, **kwargs: Any) -> None:
            auxiliary_init(instance, *args, **kwargs)
            instrumentation.auxiliary.append(instance)

        def counted_auxiliary_close(instance: Any, *args: Any, **kwargs: Any) -> None:
            instrumentation.auxiliary_close_calls[id(instance)] += 1
            auxiliary_close(instance, *args, **kwargs)

        def counted_session_init(instance: Any, *args: Any, **kwargs: Any) -> None:
            session_init(instance, *args, **kwargs)
            instrumentation.sessions.append(instance)

        async def counted_session_cancel(instance: Any) -> None:
            instrumentation.session_cancel_calls[id(instance)] += 1
            await session_cancel(instance)

        async def counted_session_close(instance: Any) -> None:
            instrumentation.session_close_calls[id(instance)] += 1
            await session_close(instance)

        def observed_recognize(instance: Any, *args: Any, **kwargs: Any) -> Any:
            if instrumentation.cancellation_armed.is_set():
                instrumentation.cancellation_native_started.set()
            return recognize(instance, *args, **kwargs)

        Qwen3ASREngine.__init__ = counted_engine_init
        Qwen3ASREngine.close = counted_engine_close
        CoreMLRuntime.__init__ = counted_runtime_init
        CoreMLRuntime.close = counted_runtime_close
        ForcedAligner.__init__ = counted_aligner_init
        ForcedAligner.close = counted_aligner_close
        AuxiliaryModels.__init__ = counted_auxiliary_init
        AuxiliaryModels.close = counted_auxiliary_close
        Qwen3ASRSession.__init__ = counted_session_init
        Qwen3ASRSession.cancel = counted_session_cancel
        Qwen3ASRSession._close = counted_session_close
        Qwen3ASRSession._recognize_via_engine = observed_recognize


def _cancel_stream(
    port: int,
    model: str,
    pcm: bytes,
    instrumentation: _Instrumentation,
) -> dict[str, Any]:
    from websockets.sync.client import connect

    prior_sessions = len(instrumentation.sessions)
    instrumentation.cancellation_native_started.clear()
    instrumentation.cancellation_armed.set()
    connection = connect(f"ws://127.0.0.1:{port}/v1/stream/{model}", open_timeout=30)
    websocket = connection.__enter__()
    try:
        websocket.send(
            json.dumps(
                {
                    "audio_format": {"encoding": "pcm_s16le", "sample_rate": 16_000},
                    "options": {},
                }
            )
        )
        websocket.send(pcm)
        assert instrumentation.cancellation_native_started.wait(60), (
            "Cancellation probe never reached genuine native recognition"
        )
    finally:
        connection.__exit__(None, None, None)
        instrumentation.cancellation_armed.clear()

    deadline = time.monotonic() + 60
    session = None
    while time.monotonic() < deadline:
        if len(instrumentation.sessions) > prior_sessions:
            session = instrumentation.sessions[-1]
            if (
                session._native_cancelled.is_set()
                and instrumentation.session_close_calls[id(session)] >= 1
            ):
                break
        time.sleep(0.02)
    assert session is not None
    assert session._native_cancelled.is_set()
    assert instrumentation.session_close_calls[id(session)] >= 1
    return {
        "client_disconnected_during_native_recognition": True,
        "session_teardown_called": True,
        "public_cancel_calls": instrumentation.session_cancel_calls[id(session)],
        "native_cancellation_token_set": True,
        "scope": (
            "Cooperative cancellation after the current Core ML prediction boundary; "
            "this does not claim mid-prediction preemption."
        ),
    }


def _installed(args: argparse.Namespace) -> int:
    import uvicorn
    from standard_asr import discover_models
    from standard_asr.toolchain.server import create_app

    import std_qwen3asr_ane

    root = args.root.resolve()
    output = args.output.resolve()
    wheel = args.wheel.resolve()
    run_dir = args.run_dir.resolve()
    installed_module = Path(std_qwen3asr_ane.__file__).resolve()
    assert installed_module.is_relative_to(run_dir / "venv"), installed_module
    general_dir = root / "artifacts/qwen3-asr-1.7b"
    short_dir = root / "artifacts/qwen3-asr-1.7b-short-dictation"
    alignment_dir = root / "artifacts/auxiliary/alignment"
    diarization_dir = root / "artifacts/auxiliary/diarization"
    en_audio = root / "artifacts/evaluation/smoke/qwen_official_en.wav"
    zh_audio = root / "artifacts/evaluation/smoke/qwen_official_zh.wav"

    instrumentation = _Instrumentation()
    instrumentation.install()
    registry = discover_models(strict=True)
    assert set(registry.names()) == {MODEL_GENERAL, MODEL_SHORT}, registry.names()
    configs = {
        MODEL_GENERAL: {
            "model_dir": str(general_dir),
            "use_alignment": True,
            "alignment_dir": str(alignment_dir),
            "use_diarization": True,
            "diarization_dir": str(diarization_dir),
            "stream_chunk_seconds": 2.0,
        },
        MODEL_SHORT: {
            "model_dir": str(short_dir),
            "stream_chunk_seconds": 12.0,
        },
    }
    app = create_app(registry=registry, engine_configs=configs)
    port = _free_port()
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host="127.0.0.1",
            port=port,
            log_level="warning",
            ws_max_size=16 * 1024 * 1024,
        )
    )
    thread = threading.Thread(target=server.run, name="server-native-uvicorn", daemon=True)
    thread.start()
    _wait_for_server(port, server, thread)

    evidence: dict[str, Any] = {
        "schema_version": 1,
        "recorded_date": DATE,
        "recorded_at": datetime.now(UTC).isoformat(),
        "status": "running",
        "scope": (
            "Installed-wheel loopback Uvicorn deployment with genuine Core ML inference, "
            "CPU forced alignment, and sherpa-onnx diarization."
        ),
        "reproduction": {
            "command": ".venv/bin/python research/release-readiness/verify_server_native.py",
            "build_output": (
                "A unique artifacts/release-readiness/server-native-<UTC>-<PID>/dist directory."
            ),
            "installed_module": "<isolated-env>/site-packages/std_qwen3asr_ane/__init__.py",
        },
        "limitations": [
            "This is a fixed official-fixture release test, not a corpus-quality benchmark.",
            "No concurrency throughput or performance claim is made.",
            "Cancellation is cooperative at native prediction boundaries.",
        ],
        "resolved_during_verification": [
            {
                "defect": (
                    "The forced-alignment worker inherited the server wheel's site-packages "
                    "through PYTHONPATH, allowing parent dependencies to shadow its pinned runtime."
                ),
                "observed_failure": (
                    "Parent tokenizers 0.23.2 shadowed worker tokenizers 0.22.2 and was rejected "
                    "by pinned Transformers 4.57.6."
                ),
                "resolution": (
                    "Launch Python in isolated mode with a package-only importlib bootstrap and "
                    "remove inherited PYTHONPATH."
                ),
            }
        ],
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "wheel": {
            "filename": wheel.name,
            "sha256": args.wheel_sha256,
            "bytes": wheel.stat().st_size,
            "isolated_environment": True,
            "extras": ["server", "diarization"],
        },
        "packages": {
            name: _record_package(name)
            for name in (
                "std-qwen3asr-ane",
                "standard-asr",
                "coremltools",
                "sherpa-onnx",
                "sherpa-onnx-core",
                "fastapi",
                "uvicorn",
                "websockets",
            )
        },
        "inputs": {
            "en": {
                "path": "artifacts/evaluation/smoke/qwen_official_en.wav",
                "sha256": _sha256(en_audio),
                "expected_text": EXPECTED_EN,
            },
            "zh": {
                "path": "artifacts/evaluation/smoke/qwen_official_zh.wav",
                "sha256": _sha256(zh_audio),
                "expected_text": EXPECTED_ZH,
            },
        },
        "configs": {
            MODEL_GENERAL: {
                "model_dir": "artifacts/qwen3-asr-1.7b",
                "manifest_sha256": _sha256(general_dir / "manifest.json"),
                "alignment_dir": "artifacts/auxiliary/alignment",
                "diarization_dir": "artifacts/auxiliary/diarization",
                "use_alignment": True,
                "use_diarization": True,
                "stream_chunk_seconds": 2.0,
            },
            MODEL_SHORT: {
                "model_dir": "artifacts/qwen3-asr-1.7b-short-dictation",
                "manifest_sha256": _sha256(short_dir / "manifest.json"),
            },
        },
    }

    try:
        health_status, health = _request_json(port, "GET", "/v1/health")
        assert health_status == 200 and health == {"status": "ok"}

        readiness: dict[str, Any] = {}
        for model in (MODEL_GENERAL, MODEL_SHORT):
            status, body = _request_json(port, "GET", f"/v1/readiness/{model}")
            assert status == 200, body
            assert isinstance(body, dict) and body["ready"] is True, body
            readiness[model] = body
        evidence["readiness"] = readiness

        en_wav = en_audio.read_bytes()
        en_duration = _audio_duration(en_audio)
        status, response = _request_json(
            port,
            "POST",
            "/v1/transcribe:json",
            {
                "model": MODEL_GENERAL,
                "audio": base64.b64encode(en_wav).decode("ascii"),
                "options": {
                    "word_timestamps": "word",
                    "diarization": {},
                    "provider_params": {"include_metrics": True},
                },
            },
        )
        assert status == 200, response
        assert isinstance(response, dict)
        en_result = response["result"]
        assert en_result["text"] == EXPECTED_EN, en_result["text"]
        en_words = en_result["words"]
        assert en_words
        en_speakers = sorted(
            {word["speaker"] for word in en_words if word.get("speaker") is not None}
        )
        assert en_speakers
        evidence["http_general_en"] = {
            "status": status,
            "text": en_result["text"],
            "detected_language": en_result["detected_language"],
            "timestamps": _assert_bounded(en_words, en_duration),
            "speaker_labels": en_speakers,
            "native_metrics_present": "native" in en_result["extra"],
        }

        status, response = _request_json(
            port,
            "POST",
            "/v1/transcribe:json",
            {
                "model": MODEL_SHORT,
                "audio": base64.b64encode(zh_audio.read_bytes()).decode("ascii"),
            },
        )
        assert status == 200, response
        assert isinstance(response, dict)
        short_result = response["result"]
        assert short_result["text"] == EXPECTED_ZH, short_result["text"]
        evidence["http_short_zh"] = {
            "status": status,
            "text": short_result["text"],
            "detected_language": short_result["detected_language"],
        }

        zh_pcm = _pcm16_16khz(zh_audio)
        zh_duration = _audio_duration(zh_audio)
        ws_events = _stream(
            port,
            MODEL_GENERAL,
            zh_pcm,
            {"word_timestamps": "char", "diarization": {}},
        )
        assert ws_events[-1]["type"] == "done", ws_events[-1]
        closed = [
            event
            for event in ws_events
            if event["type"] == "final" and event.get("finality") == "closed"
        ]
        assert closed, ws_events
        ws_text = "".join(event["text"] for event in closed)
        assert ws_text == EXPECTED_ZH, ws_text
        ws_words = [word for event in closed for word in (event.get("words") or [])]
        assert ws_words
        evidence["websocket_general_zh"] = {
            "terminal_event": ws_events[-1]["type"],
            "event_types": [event["type"] for event in ws_events],
            "closed_text": ws_text,
            "timestamps": _assert_bounded(ws_words, zh_duration),
            "audio_processed_until": closed[-1]["audio_processed_until"],
            "speaker_labels": sorted(
                {word["speaker"] for word in ws_words if word.get("speaker") is not None}
            ),
        }
        assert abs(closed[-1]["audio_processed_until"] - zh_duration) <= 1e-3

        evidence["websocket_disconnect_cancellation"] = _cancel_stream(
            port,
            MODEL_GENERAL,
            _pcm16_16khz(en_audio),
            instrumentation,
        )

        profiles = Counter(engine.config.profile for engine in instrumentation.engines)
        assert profiles == Counter({"general": 1, "short-dictation": 1}), profiles
        assert len(instrumentation.runtimes) == 2
        assert len(instrumentation.aligners) == 1
        assert len(instrumentation.auxiliary) == 1
        assert instrumentation.auxiliary[0]._diarizer is not None
        evidence["pool_before_shutdown"] = {
            "engine_constructions": dict(sorted(profiles.items())),
            "native_runtime_constructions": len(instrumentation.runtimes),
            "forced_aligner_constructions": len(instrumentation.aligners),
            "diarization_backend_loaded": True,
            "one_engine_per_configured_model": True,
            "lease_reuse": (
                "Readiness, repeated HTTP requests, completed WebSocket, and disconnected "
                "WebSocket used the same general-profile engine construction."
            ),
        }
    except BaseException as error:
        evidence["status"] = "failed"
        evidence["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        raise
    finally:
        server.should_exit = True
        thread.join(timeout=180)
        evidence["shutdown"] = {"uvicorn_thread_stopped": not thread.is_alive()}
        if not thread.is_alive():
            engine_closed = [
                instrumentation.engine_close_calls[id(engine)] for engine in instrumentation.engines
            ]
            runtime_closed = [
                instrumentation.runtime_close_calls[id(runtime)]
                for runtime in instrumentation.runtimes
            ]
            aligner_closed = [
                instrumentation.aligner_close_calls[id(aligner)]
                for aligner in instrumentation.aligners
            ]
            auxiliary_closed = [
                instrumentation.auxiliary_close_calls[id(auxiliary)]
                for auxiliary in instrumentation.auxiliary
            ]
            native_handles_closed = all(
                all(model._resources["model"] is None for model in runtime._prediction_models())
                for runtime in instrumentation.runtimes
            )
            native_buffers_released = all(
                all(not model._resources["buffers"] for model in runtime._prediction_models())
                for runtime in instrumentation.runtimes
            )
            auxiliary_resources_released = all(
                auxiliary._aligner is None and auxiliary._diarizer is None
                for auxiliary in instrumentation.auxiliary
            )
            evidence["shutdown"].update(
                {
                    "engine_close_calls": engine_closed,
                    "native_runtime_close_calls": runtime_closed,
                    "forced_aligner_close_calls": aligner_closed,
                    "auxiliary_close_calls": auxiliary_closed,
                    "native_model_handles_closed": native_handles_closed,
                    "native_input_buffers_released": native_buffers_released,
                    "auxiliary_resources_released": auxiliary_resources_released,
                }
            )
            if evidence["status"] != "failed":
                assert engine_closed == [1, 1], engine_closed
                assert runtime_closed == [1, 1], runtime_closed
                assert aligner_closed == [1], aligner_closed
                assert auxiliary_closed == [1], auxiliary_closed
                assert native_handles_closed
                assert native_buffers_released
                assert auxiliary_resources_released
                evidence["status"] = "passed"
        _write_json(output, evidence)

    print(json.dumps({"status": evidence["status"], "evidence": str(output)}))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name(f"server-native-{DATE}.json"),
        help="Portable JSON evidence path.",
    )
    parser.add_argument("--python", default="3.12", help="Python version for the isolated env.")
    parser.add_argument("--run-installed", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--wheel", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--wheel-sha256", help=argparse.SUPPRESS)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.run_installed:
        if args.run_dir is None or args.wheel is None or args.wheel_sha256 is None:
            raise SystemExit("Installed mode needs --run-dir, --wheel, and --wheel-sha256")
        return _installed(args)
    return _orchestrate(args)


if __name__ == "__main__":
    raise SystemExit(main())
