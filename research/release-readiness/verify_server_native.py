"""Verify the installed wheel through the real Standard ASR HTTP/WS server.

The default mode builds a uniquely located wheel, creates an isolated virtual
environment, installs the wheel with its server and diarization extras, and
re-executes this file with that environment's Python. The installed mode runs
Uvicorn on loopback and uses real Core ML, forced alignment, and diarization.

Each preset runs in a separate process with official environment configuration.
Observation wrappers only record cancellation; they never replace inference.
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
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evidence_provenance import evidence_date, runtime_provenance

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
    with log.open("w") as stream:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            text=True,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
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
    _write_json(
        output,
        {
            "status": "running",
            "stage": "build_and_install",
            "run_directory": str(run_dir),
            "recorded_date": evidence_date(),
        },
    )

    command_env = os.environ.copy()
    command_env["UV_OFFLINE"] = "1"
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
        samples = resample_poly(samples, 16_000 // divisor, sample_rate // divisor)
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


class _CancellationProbe:
    """Observe real cancellation without retaining engines or native resources."""

    def __init__(self) -> None:
        self.armed = threading.Event()
        self.native_started = threading.Event()
        self.session_closed = threading.Event()
        self.native_cancelled = False

    def install(self) -> None:
        from std_qwen3asr_ane.runtime import CoreMLRuntime
        from std_qwen3asr_ane.streaming import Qwen3ASRSession

        transcribe = CoreMLRuntime.transcribe
        close = Qwen3ASRSession._close
        probe = self

        def observed_transcribe(instance: Any, *args: Any, **kwargs: Any) -> Any:
            if probe.armed.is_set():
                probe.native_started.set()
            return transcribe(instance, *args, **kwargs)

        async def observed_close(instance: Any) -> None:
            await close(instance)
            if probe.armed.is_set():
                probe.native_cancelled = instance._native_cancelled.is_set()
                probe.session_closed.set()

        CoreMLRuntime.transcribe = observed_transcribe
        Qwen3ASRSession._close = observed_close


def _cancel_stream(port: int, model: str, pcm: bytes, probe: _CancellationProbe) -> dict[str, Any]:
    from websockets.sync.client import connect

    probe.armed.set()
    try:
        with connect(f"ws://127.0.0.1:{port}/v1/stream/{model}", open_timeout=30) as websocket:
            websocket.send(
                json.dumps(
                    {
                        "audio_format": {"encoding": "pcm_s16le", "sample_rate": 16_000},
                        "options": {},
                    }
                )
            )
            websocket.send(pcm)
            assert probe.native_started.wait(60), "No native transcription started"
        assert probe.session_closed.wait(60), "Disconnected session did not close"
        assert probe.native_cancelled, "Native cancellation token was not set"
    finally:
        probe.armed.clear()
    return {
        "client_disconnected_after_native_transcription_started": True,
        "session_teardown_observed": True,
        "native_cancellation_token_set": True,
        "scope": "Cooperative cancellation at prediction boundaries, not preemption.",
    }


def _failure(error: BaseException) -> dict[str, str]:
    return {
        "type": type(error).__name__,
        "message": str(error),
        "traceback": traceback.format_exc(),
    }


def _configured_environment(root: Path, preset: str) -> dict[str, str]:
    """Use the official engine-scoped init-config environment convention."""
    short = preset == "short"
    bundle = "qwen3-asr-1.7b-short-dictation" if short else "qwen3-asr-1.7b"
    values = {
        "MODEL_DIR": str(root / "artifacts" / bundle),
        "USE_ALIGNMENT": "false" if short else "true",
        "ALIGNMENT_DIR": str(root / "artifacts/auxiliary/alignment"),
        "USE_DIARIZATION": "false" if short else "true",
        "DIARIZATION_DIR": str(root / "artifacts/auxiliary/diarization"),
        "STREAM_CHUNK_SECONDS": "2.0",
    }
    return {f"STANDARD_ASR_STD_QWEN3ASR_ANE__{key}": value for key, value in values.items()}


def _check_stream(events: list[dict[str, Any]], expected: str) -> list[dict[str, Any]]:
    assert events[-1]["type"] == "done", events
    closed = [event for event in events if event["type"] == "final"]
    assert closed, events
    text = "".join(event["text"] for event in closed)
    assert text == expected, text
    assert all(event["finality"] == "closed" for event in closed), closed
    assert all(event["stable_text"] == event["text"] for event in closed), closed
    assert all("stable_until" not in event for event in closed), closed
    return closed


def _probe_installed(args: argparse.Namespace) -> int:
    import uvicorn
    from standard_asr import discover_models
    from standard_asr.toolchain.server import create_app

    root = args.root.resolve()
    output = args.output.resolve()
    evidence: dict[str, Any] = {"status": "running", "preset": args.preset}
    server = None
    thread = None
    try:
        evidence["runtime_provenance"] = runtime_provenance()
        configured = _configured_environment(root, args.preset)
        os.environ.update(configured)
        os.environ["STANDARD_ASR_ALLOW_DOWNLOAD"] = "0"
        os.environ["UV_OFFLINE"] = "1"
        evidence["environment"] = {
            key: value.replace(str(root), "<repository>") for key, value in configured.items()
        }
        model = MODEL_SHORT if args.preset == "short" else MODEL_GENERAL
        bundle = Path(configured["STANDARD_ASR_STD_QWEN3ASR_ANE__MODEL_DIR"])
        evidence["manifest_sha256"] = _sha256(bundle / "manifest.json")
        registry = discover_models(strict=True)
        assert set(registry.names()) == {MODEL_GENERAL, MODEL_SHORT}, registry.names()
        probe = _CancellationProbe()
        probe.install()
        port = _free_port()
        server = uvicorn.Server(
            uvicorn.Config(
                create_app(registry=registry),
                host="127.0.0.1",
                port=port,
                log_level="warning",
                ws_max_size=16 * 1024 * 1024,
            )
        )
        thread = threading.Thread(target=server.run, name="server-native-uvicorn", daemon=True)
        thread.start()
        _wait_for_server(port, server, thread)
        status, health = _request_json(port, "GET", "/v1/health")
        assert status == 200 and health == {"status": "ok"}, health
        status, models = _request_json(port, "GET", "/v1/models")
        assert status == 200 and {item["key"] for item in models} == {MODEL_GENERAL, MODEL_SHORT}, (
            models
        )
        evidence["discovered_models"] = models
        for endpoint in ("capabilities", "metadata", "params-schema", "config-schema"):
            status, body = _request_json(port, "GET", f"/v1/{endpoint}/{model}")
            assert status == 200, body
            evidence[endpoint] = body

        audio = (
            root
            / "artifacts/evaluation/smoke"
            / ("qwen_official_zh.wav" if args.preset == "short" else "qwen_official_en.wav")
        )
        expected = EXPECTED_ZH if args.preset == "short" else EXPECTED_EN
        encoded = base64.b64encode(audio.read_bytes()).decode("ascii")
        evidence["http_input"] = {"sha256": _sha256(audio), "expected_text": expected}
        options = {} if args.preset == "short" else {"word_timestamps": "word", "diarization": {}}
        status, body = _request_json(
            port,
            "POST",
            "/v1/transcribe:json",
            {
                "model": model,
                "audio": encoded,
                "options": options,
            },
        )
        evidence["http"] = {"status": status, "response": body}
        assert status == 200, body
        result = body["result"]
        assert result["text"] == expected, result["text"]
        if args.preset == "general":
            assert result["words"], result
            evidence["http"]["timestamps"] = _assert_bounded(
                result["words"], _audio_duration(audio)
            )
            assert any(word.get("speaker") is not None for word in result["words"])

        # Provider params remain Python-only; schema discovery does not permit wire submission.
        status, body = _request_json(
            port,
            "POST",
            "/v1/transcribe:json",
            {
                "model": model,
                "audio": encoded,
                "options": {"provider_params": {"include_metrics": True}},
            },
        )
        evidence["provider_params_rejected"] = {"status": status, "response": body}
        assert status == 422, body

        zh_audio = root / "artifacts/evaluation/smoke/qwen_official_zh.wav"
        options = {} if args.preset == "short" else {"word_timestamps": "char", "diarization": {}}
        events = _stream(port, model, _pcm16_16khz(zh_audio), options)
        evidence["websocket"] = {"input_sha256": _sha256(zh_audio), "events": events}
        closed = _check_stream(events, EXPECTED_ZH)
        if args.preset == "general":
            words = [word for event in closed for word in event.get("words") or []]
            assert words, closed
            evidence["websocket"]["timestamps"] = _assert_bounded(words, _audio_duration(zh_audio))
            assert any(word.get("speaker") is not None for word in words)
            assert abs(closed[-1]["audio_processed_until"] - _audio_duration(zh_audio)) <= 1e-3
            evidence["cancellation"] = _cancel_stream(port, model, _pcm16_16khz(audio), probe)
        else:
            # Main gates all timing fields when the negotiated timestamp mode is none.
            assert all(event["audio_processed_until"] is None for event in closed)
        evidence["status"] = "passed"
    except BaseException as error:
        evidence["status"] = "failed"
        evidence["failure"] = _failure(error)
        raise
    finally:
        if server is not None:
            server.should_exit = True
        if thread is not None:
            thread.join(timeout=60)
            evidence["uvicorn_thread_stopped"] = not thread.is_alive()
            if thread.is_alive():
                evidence["status"] = "failed"
                evidence["shutdown_failure"] = "Uvicorn did not stop within 60 seconds"
        _write_json(output, evidence)
    return 0 if evidence["status"] == "passed" else 1


def _installed(args: argparse.Namespace) -> int:
    import sys

    import std_qwen3asr_ane

    root = args.root.resolve()
    run_dir = args.run_dir.resolve()
    evidence: dict[str, Any] = {
        "schema_version": 2,
        "recorded_date": evidence_date(),
        "recorded_at": datetime.now(UTC).isoformat(),
        "status": "running",
        "scope": "Installed-wheel official main server; genuine Core ML, alignment and diarization.",
        "limitations": [
            "Fixed official fixtures, not a corpus-quality or throughput benchmark.",
            "Each preset uses its own process and engine-scoped environment configuration.",
            "No server pool, readiness endpoint, or engine-close contract is assumed.",
            "Cancellation is cooperative at prediction boundaries.",
        ],
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "wheel": {
            "filename": args.wheel.name,
            "sha256": args.wheel_sha256,
            "bytes": args.wheel.stat().st_size,
            "extras": ["server", "diarization"],
        },
        "presets": {},
    }
    try:
        installed_module = Path(std_qwen3asr_ane.__file__).resolve()
        assert installed_module.is_relative_to(run_dir / "venv"), installed_module
        assert _sha256(args.wheel) == args.wheel_sha256
        evidence["runtime_provenance"] = runtime_provenance()
        evidence["packages"] = {
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
        }
        for preset in ("general", "short"):
            probe_output = run_dir / f"{preset}.json"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--run-installed",
                "--preset",
                preset,
                "--root",
                str(root),
                "--output",
                str(probe_output),
            ]
            try:
                _run_logged(
                    command, cwd=run_dir, log=run_dir / f"{preset}.log", env=os.environ.copy()
                )
            finally:
                if probe_output.exists():
                    evidence["presets"][preset] = json.loads(probe_output.read_text())
        evidence["status"] = "passed"
    except BaseException as error:
        evidence["status"] = "failed"
        evidence["failure"] = _failure(error)
        raise
    finally:
        _write_json(args.output.resolve(), evidence)
    print(json.dumps({"status": evidence["status"], "evidence": str(args.output)}))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, required=True, help="JSON evidence path.")
    parser.add_argument("--python", default="3.13", help="Isolated Python version.")
    parser.add_argument("--run-installed", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--preset", choices=("general", "short"), help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--wheel", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--wheel-sha256", dest="wheel_sha256", help=argparse.SUPPRESS)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.run_installed:
        if args.preset:
            return _probe_installed(args)
        if args.run_dir is None or args.wheel is None or args.wheel_sha256 is None:
            raise SystemExit("Installed mode needs --run-dir, --wheel, and --wheel-sha256")
        return _installed(args)
    try:
        return _orchestrate(args)
    except BaseException as error:
        evidence = json.loads(args.output.read_text()) if args.output.exists() else {}
        if evidence.get("status") != "failed":
            evidence.update(status="failed", failure=_failure(error))
            _write_json(args.output, evidence)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
