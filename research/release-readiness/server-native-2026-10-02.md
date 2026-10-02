# Installed-wheel native server verification — 2026-10-02

The release-candidate source at `a8ae9140de59b666b1f0bd9c9db10471e0bad669` passed a real loopback Uvicorn deployment test on Apple Silicon. The verifier built a unique `std_qwen3asr_ane-0.2.0a1-py3-none-any.whl`, installed it with `[server,diarization]` into an isolated Python 3.12 environment, and confirmed that the environment resolved Standard ASR commit `5f6eef25e35e5e66e9010474e6dee531021e61f1`. The tested wheel SHA-256 is `3902eb40c7750f9ceb5225157abdea281343f475fb07341422263375708442fd`.

Both configured models reported ready. A real HTTP request through the general model reproduced the fixed native English transcript exactly and returned 37 ordered, finite word spans between 0.40 and 14.64 seconds. The request exercised the pinned CPU forced aligner and sherpa-onnx diarization models. A second HTTP request through the short-dictation model reproduced the fixed Chinese transcript exactly.

A real WebSocket session through the reused general engine emitted two partials, one closed final, and `done`. Its final text was exactly `甚至出现交易几乎停滞的情况。`; all 13 character spans were ordered and bounded by the 4.2039375-second input, and `audio_processed_until` equalled that duration. A separate WebSocket client disconnected after genuine native recognition began and before ending audio. Server teardown set the engine's cooperative native cancellation token. This verifies cancellation at completed Core ML prediction boundaries; it does not claim mid-prediction preemption.

Lifecycle instrumentation delegated every call to the real implementation. It observed one engine construction per configured model across readiness, HTTP, completed WebSocket, and disconnected WebSocket leases. Shutdown stopped Uvicorn, called both engine and native-runtime close paths once, closed the one forced-aligner worker once, released all native model handles and persistent input buffers, and cleared the auxiliary aligner and diarizer owners.

The clean install exposed and fixed one packaging-only defect that the development environment had hidden. The forced-alignment worker previously placed the server wheel's whole site-packages directory on `PYTHONPATH`, so the parent environment's `tokenizers==0.23.2` shadowed the worker's pinned `0.22.2` and failed under Transformers 4.57.6. The worker now starts Python in isolated mode, exposes only the plugin package through an importlib bootstrap, and removes inherited `PYTHONPATH`. A subprocess regression test covers this boundary.

Reproduce from the repository root with:

```sh
.venv/bin/python research/release-readiness/verify_server_native.py
```

The command always creates a new ignored build and environment directory under `artifacts/release-readiness/server-native-<UTC>-<PID>/`. The complete portable record is [`server-native-2026-10-02.json`](server-native-2026-10-02.json). This fixed-fixture verification is not a corpus-quality or concurrency-performance benchmark, and no throughput claim is made.
