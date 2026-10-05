# Standard ASR ecosystem, packaging, CLI, and server audit

Date: 2026-09-22

## Scope and evidence level

This audit covers Standard ASR discovery, end-user installation, both installed CLIs, config and provider schemas, artifact commands, doctor/compliance, result renderers, and the reference HTTP/WebSocket server. Conclusions use the fresh upstream checkout at `references/standard-asr-audit-2026-09-22` (`1b2cf3fa5860c075e5160eb60b26b708a7c8bfea`) and the plugin baseline described in `brief.md`. The installed/pinned Standard ASR Python implementation is byte-identical to the fresh checkout, so code findings apply to the dependency the plugin actually installs.

Evidence labels used below:

- **Static confirmation**: established from the source and package metadata.
- **Mock reproduction**: exercised through public CLI/server/library surfaces without loading Core ML or downloading weights.
- **Native-unverified**: would require the real converted bundle and ANE execution; no such claim is made here.

The lightweight probe is [`probe_ecosystem.py`](./probe_ecosystem.py). It uses the public `discover_models`, `create_app`, `TranscriptionResult`, `to_srt`, and `to_vtt` surfaces. It does not acquire artifacts or load Core ML.

## Overall assessment

The plugin is a strong Standard ASR adapter at the in-process and command-line boundaries. Its two presets are discoverable, class metadata is available without construction, config and provider params are typed, artifacts are explicit, ordinary inference never downloads, portable parameter gating is inherited correctly, result/event models are standard, and the installed Standard ASR CLI gives end users one canonical path for `list`, `show`, `status`, `pull`, `prepare`, `transcribe`, `doctor`, and `compliance`.

The largest practical blocker is the reference server's engine ownership model. It creates a new engine for every REST request and WebSocket connection, does not retain a prepared instance, and does not invoke the plugin's explicit `close()` method. That behavior is tolerable for lightweight remote-client adapters, but it prevents this multi-gigabyte local Core ML engine from receiving the warm, process-long lifetime a useful inference server needs. The mock probe proves construction/close calls; it does **not** prove a retained native-memory leak, because no real model was loaded and the runtime also has finalizers.

The next issues are installation and coverage gaps around the optional server extra, the intentionally deferred provider-params wire path, artifact-blind health, and several documentation/secondary-CLI inconsistencies. These should be fixed in the shared ecosystem rather than by adding a Qwen-specific HTTP API.

## What works well

### Discovery and declarations

- The package registers exactly two `standard_asr.models` entry points, `std-qwen3asr-ane/1.7b` and `std-qwen3asr-ane/1.7b-short-dictation` (`pyproject.toml:24-26`). The entry points target concrete engine classes, so Standard ASR can resolve class metadata without instantiating or loading native code.
- The plugin publishes a typed `Qwen3ASRConfig` and a closed `Qwen3ASRParams` with request-specific `max_new_tokens` (`std_qwen3asr_ane/src/std_qwen3asr_ane/plugin.py:95-169`), binds them as `config_type` and `provider_params_type` (`plugin.py:175`, `plugin.py:231`), and declares typed properties, capabilities, and artifact metadata (`plugin.py:176-230`).
- The declared model boundary is honest: array input at 16 kHz, 30-second general and 12-second short batch limits, PCM wire encodings, selectable/detectable languages, batch and streaming language override, prompt support, revisable partials, and closed finals (`plugin.py:176-211`, `plugin.py:612-620`). Unsupported timestamps, diarization, candidate-language restriction, reconnect, and rollover remain unsupported rather than fabricated.
- Construction is lazy: native runtime state starts as `None` and the constructor only parses config (`plugin.py:233-238`). Existing tests explicitly guard against filesystem/device/runtime access during construction (`std_qwen3asr_ane/tests/test_plugin.py:155-174`). That makes `list`, `show`, schema endpoints, and compliance discovery cheap.

### Installation and artifact lifecycle

- The primary end-user path is one VCS tool install followed by the shared CLI (`README.md:62-80`). The plugin exposes the upstream `standard-asr` entry point inside the isolated tool environment (`pyproject.toml:28-32`), so a separate core installation is not required for normal CLI use.
- Standard ASR is pinned to an immutable revision (`pyproject.toml:9-10`), and the conversion source checkpoint is pinned too (`std_qwen3asr_ane/src/std_qwen3asr_ane/conversion/build.py:13-15`).
- Inference dependencies and conversion dependencies are separated. `standard-asr pull` can launch the packaged conversion worker through `uv`, so the normal environment does not need Torch/Transformers and the optional MLX draft cannot silently replace the converter stack (`std_qwen3asr_ane/src/std_qwen3asr_ane/acquisition.py:91-99`; `conversion/toolchain.py:1-25`; `plugin.py:320-365`).
- Artifact status is context-sensitive: streaming excludes the unused draft, while batch includes it when configured (`plugin.py:287-318`). Inference never acquires (`declared_metadata.may_acquire_during_inference=False`, `plugin.py:212-219`), and unavailable artifacts retain Standard ASR exception types.
- `uv pip check --python .venv/bin/python` passed after directing uv's cache into the workspace: 106 installed packages were compatible.

### CLI behavior

Actual lightweight commands, all with no model acquisition:

| Command | Observed result |
|---|---|
| `standard-asr list` | Exit 0; both presets listed. |
| `standard-asr show std-qwen3asr-ane/1.7b` | Exit 0; canonical capability/metadata JSON and config schema rendered. |
| `STANDARD_ASR_ALLOW_DOWNLOAD=0 standard-asr status ... --json` | Exit 0; one missing required artifact, `downloads_disabled`, readiness `unavailable`, no acquisition. This exit behavior is upstream-defined; scripts use `--require-ready` (`docs/content/specification/cli.md:35-48`). |
| `standard-asr doctor` | Exit 0; plugin and core NumPy requirements were shown and no NumPy conflict was found. |
| `STANDARD_ASR_ALLOW_DOWNLOAD=0 standard-asr compliance run` | Exit 0. It explicitly noted that recorded event-sequence and result checks are library-only, as required by the CLI spec (`docs/content/specification/cli.md:82-95`). |
| `standard-asr --help` | Shows the complete shared command set, including `serve`. |

The CLI preserves clean stdout: text transcription goes to stdout and diagnostics go to stderr, while `--json` carries the full result (`references/standard-asr-audit-2026-09-22/src/standard_asr/toolchain/cli.py:2097-2142`; authority `docs/content/specification/cli.md:97-107`). Init config and runtime options remain distinct. This plugin's `max_new_tokens` is also available as init config, which is convenient for one-shot CLI use.

### Result, renderer, and streaming integration

- Batch returns a standard `TranscriptionResult` with text, detected language, actual duration, and diagnostics (`plugin.py:542-587`). It does not invent segments/words.
- The upstream SRT/VTT renderers still work: when `segments is None`, they create a single whole-transcript cue over the known duration (`references/standard-asr-audit-2026-09-22/src/standard_asr/renderers.py:279-313`, `316-432`). The probe confirmed both SRT and VTT output for a Qwen-shaped result. These are whole-utterance subtitles, not alignment.
- Streaming uses `asyncio.to_thread` for native recognition (`streaming.py:152-165`), emits revisable partials with `stable_until=0`, emits a closed final, and lets the base emit `done` and reduce the result (`streaming.py:198-231`). It distinguishes cancellation, configured/bundle audio limits, decoder capacity, and invalid audio/context with structured codes (`streaming.py:232-263`).
- The reference server correctly passes encoded input into Standard ASR audio negotiation rather than assuming a sample rate (`server.py:603-610`, `646-652`), executes blocking batch work in a worker thread (`server.py:1645-1657`), sanitizes validation and internal errors, and bounds HTTP bodies plus WebSocket frames/sessions. The WebSocket bridge forwards diagnostics and scrubs engine error-event `extra` before sending it (`docs/content/specification/server-api.md:171-201`).

## Findings

### ECO-1 — P1 practical blocker: the reference server has no reusable engine lifetime

**Classification:** Confirmed upstream limitation exposed critically by this plugin. Ownership: Standard ASR server/protocol, with the plugin already providing the cleanup operation it would need.

**Trigger and consequence:** Any two REST requests or two WebSocket connections. REST calls `_create_engine_or_http_error`, which invokes `registry.create(model)` for each request (`references/standard-asr-audit-2026-09-22/src/standard_asr/toolchain/server.py:1526-1575`), then `_run_transcription` uses the new instance (`server.py:1613-1648`). WebSocket construction repeats the same pattern per connection (`server.py:815-817`). There is no engine cache or application-lifespan owner. The bridge closes the **session** (`server.py:1515-1523`), not the engine.

For Qwen, a new engine lazily creates a separate Core ML runtime on first inference (`std_qwen3asr_ane/src/std_qwen3asr_ane/plugin.py:233-238`, `plugin.py:462-472`). Its lock is per instance (`plugin.py:238`), so separate requests do not share the plugin's serialization lane. The plugin has deterministic `close()` (`plugin.py:519-533`), but the structural Standard ASR interface has no engine-lifetime method (`references/standard-asr-audit-2026-09-22/src/standard_asr/runtime/interface.py:140-259`) and the server never calls this plugin method.

**Mock reproduction:** `probe_reference_server_engine_lifetime()` sends two requests through public `create_app`. It observes `creations == 2` and `closes == 0`, including after `TestClient` lifespan exit. This proves server ownership behavior only. It does not establish that native memory remains permanently allocated: the real runtime uses `weakref.finalize` on model wrappers (`std_qwen3asr_ane/src/std_qwen3asr_ane/runtime.py:89-101`) and was not loaded in this probe. The demonstrated practical harms are lost warm reuse, repeated native load/specialization opportunity, lack of deterministic close, and multiple independent engines under concurrency.

**Recommended fix:** Add an upstream application-lifespan engine manager keyed by model and effective server-side config. Construct lazily or during an explicit warmup, share the instance across REST and WebSocket, and retire it once on application shutdown. Add a standard engine lifecycle (`close`, preferably with sync/async boundary rules) or a server-owned context/factory contract. The Qwen plugin's existing lock can serialize shared use, and its existing `close()` can be adopted without a vendor server.

**Acceptance tests:**

1. Two sequential REST requests for the same model create one engine.
2. REST and WebSocket use the same configured engine where the lifecycle policy permits it.
3. Concurrent calls do not construct duplicate instances; Qwen execution remains serialized by its shared lock.
4. Application shutdown invokes close exactly once after active work ends, including cancellation and failed-request paths.
5. A deliberately failing close is safe-logged and does not disappear; a later shutdown step does not claim successful disposal.
6. A real-bundle validation measures first versus second request load/latency and process/system memory. Until run, do not claim a leak or a quantified improvement.

### ECO-2 — P2 installation/CI gap: `serve` is exposed by the ordinary tool install, but its extra is not installable through a documented plugin recipe

**Classification:** Confirmed packaging and documentation gap, not a core correctness violation. Server support is intentionally optional upstream.

**Trigger and consequence:** Follow the advertised `uv tool install git+...` command (`README.md:64-72`) and then use the installed `standard-asr serve` shown by help. The plugin dependency is bare pinned `standard-asr`, and the only plugin extra is `gpu-draft` (`pyproject.toml:9-22`), while the upstream server dependencies live exclusively in `standard-asr[server]` (`references/standard-asr-audit-2026-09-22/pyproject.toml:110-140`). The docs say only “Install Standard ASR's `server` extra in that same environment” (`docs/usage.md:117-120`). In an isolated `uv tool` environment, a host `pip install` is not an actionable recipe for modifying that tool. The upstream error itself recommends `pip install 'standard-asr[server]'` (`references/standard-asr-audit-2026-09-22/src/standard_asr/toolchain/cli.py:2145-2174`), which is appropriate in an application venv but incomplete for this README's primary tool-install workflow.

This audit did not create a clean networked tool environment, so it does not claim an observed `serve` failure. The package metadata proves that the primary install does not request the optional extra.

**Coverage blind spot:** The plugin's server test imports FastAPI and tests schemas plus one HTTP request (`std_qwen3asr_ane/tests/test_artifact_lifecycle.py:228-259`), but CI runs all tests after installing the `convert` group (`.github/workflows/ci.yml:39-44`). That group includes `qwen-asr` (`pyproject.toml:37`), whose Gradio dependency supplies FastAPI in the current lock. The separate ordinary-wheel lane runs list/compliance/status only and never imports or launches the server (`.github/workflows/ci.yml:46-54`). Thus the test does not prove the documented ordinary installation can serve.

**Recommended fix:** Add a plugin `server` extra that depends on the exact pinned upstream URL **with** its `server` extra, keeping upstream as the dependency authority; document one tool command such as installing `std-qwen3asr-ane[server]` or using `uv tool install --with` against the same pinned upstream revision. Do not copy FastAPI/Starlette/WebSocket dependency lists into the plugin and do not add a Qwen-specific server.

**Acceptance tests:**

1. Build the wheel, install `std-qwen3asr-ane[server]` into an empty tool environment without dev/convert groups, and run `standard-asr serve --help` plus an app startup/health request.
2. Install the base wheel into another empty tool environment and confirm `serve` fails with a clear recipe that actually modifies that same tool environment.
3. Assert `websockets`, `python-multipart`, FastAPI, Starlette, and uvicorn come from the upstream extra.
4. Run the server integration test in its own clean CI job, independent of `qwen-asr`/Gradio.

### ECO-3 — P2 upstream opportunity: provider params are published but deliberately unusable through CLI/HTTP/WebSocket

**Classification:** Deliberate Standard ASR 0.2 limitation, not adapter misuse. Ownership: upstream protocol/toolchain.

**Trigger and consequence:** Discover `/v1/params-schema/std-qwen3asr-ane/1.7b`, which publishes `Qwen3ASRParams.max_new_tokens`, then try to send `{"provider_params":{"max_new_tokens":32}}`. The server schema route intentionally exposes the type (`server.py:685-714`), but `_build_params` validates only `WireRuntimeParams` and rejects the field (`server.py:1745-1767`). CLI `--options` does the same (`references/standard-asr-audit-2026-09-22/src/standard_asr/toolchain/cli.py:2178-2220`). The authority explicitly calls provider params “discover-only, not sendable” and defers the JSON-schema-over-wire path (`docs/content/specification/server-api.md:38-45`, `102-108`). The probe confirmed an HTTP 422 `extra_forbidden` while the immediately adjacent schema advertises the setting.

For this plugin, the lost feature is per-request output budget (`plugin.py:166-169`, `589-593`). A server operator can set the engine-wide default through environment config, but clients cannot vary it per request. `standard-asr transcribe` can use the init-config twin `--set max_new_tokens=...` because it constructs a one-shot engine, but that is not equivalent for a long-running server.

**Recommended fix:** Complete the upstream typed provider-params wire path. After resolving the selected engine class, validate `options.provider_params` with that class's closed `provider_params_type` and place the typed instance in `RuntimeParams`. Apply the same operation in CLI, REST, and WebSocket. Keep this inside Standard ASR; a redundant Qwen API would defeat the ecosystem goal.

**Acceptance tests:**

1. REST, WS, and CLI accept valid Qwen `max_new_tokens` and deliver a `Qwen3ASRParams` instance to both batch and streaming hooks.
2. Unknown fields, out-of-range values, and params for a different model fail before inference with sanitized 422/`bad_request`/CLI usage errors.
3. Switching model keys revalidates against the new type; no raw dict reaches an engine.
4. Models with no `provider_params_type` reject a non-empty provider object and continue to accept portable-only options.

### ECO-4 — P2 upstream operational gap: server health does not report artifact readiness

**Classification:** Confirmed upstream server scope gap, amplified by this plugin's explicit 4.3 GB acquisition lifecycle.

**Trigger and consequence:** Start the server before `standard-asr pull`, or point it at a corrupt/missing bundle. `/v1/health` always returns `{"status":"ok"}` (`references/standard-asr-audit-2026-09-22/src/standard_asr/toolchain/server.py:491-504`). The server has no artifact-status/readiness route; per-model static metadata reports only lifecycle upper bounds (`server.py:525-558`). The plugin can already report exact configured readiness without loading Core ML (`plugin.py:255-318`), and the CLI exposes it, but deployment probes cannot distinguish a live server from one that will return a scrubbed 503 on every transcription.

**Recommended fix:** Preserve `/v1/health` as liveness and add an operator-facing readiness surface backed by `artifact_status`, preferably integrated with the pooled engine/config from ECO-1. Do not expose local paths/actions on an unauthenticated public response; return a stable readiness summary and safe-log the detailed report, or make detailed status an explicitly protected operator endpoint.

**Acceptance tests:**

1. Missing bundle: liveness 200, readiness non-200/unready, no acquisition or directory creation.
2. Ready bundle: readiness 200 without running inference.
3. Corrupt, downloads-disabled, required draft, and streaming-with-unused-draft contexts produce distinct internal reports and the documented safe public state.
4. A status-hook failure is logged and does not become a false ready verdict.

### ECO-5 — P2 test gap: the plugin claims HTTP/WebSocket integration but tests only HTTP

**Classification:** Confirmed plugin test/claim gap; no WebSocket defect reproduced.

**Evidence:** Usage says the Standard server exposes the plugin through HTTP/WebSocket (`docs/usage.md:117-120`), and the historical audit's integration row makes the same claim (`docs/standard-asr-audit.md:51-53`). The sole plugin server test checks `/v1/models`, four metadata/schema routes, and JSON batch transcription (`std_qwen3asr_ane/tests/test_artifact_lifecycle.py:228-259`). No plugin test uses `/v1/stream` or `websocket_connect`. Direct session tests are valuable but do not cross config-frame parsing, binary PCM transport, diagnostic frames, event scrubbing, bridge teardown, or actual WebSocket dependency installation.

**Recommended fix:** Add a fake-runtime WebSocket integration test in the clean server CI environment proposed in ECO-2. Use the real discovered Qwen entry point and standard server; do not create a vendor socket implementation.

**Acceptance tests:**

1. Connect to both presets, send the JSON config frame, split s16le/f32le samples across awkward frame boundaries, signal end with text, and observe partial, closed, done in compliant order.
2. Verify language/prompt and a degraded phrase hint diagnostics frame reach the client.
3. Verify incomplete PCM, configured/bundle audio limit, model capacity, cancellation/disconnect, and native error become the documented safe codes with error `extra` stripped.
4. Verify session teardown runs and no worker remains active after disconnect.

### ECO-6 — P2 plugin CLI drift: the installed vendor CLI duplicates transcription and advertises the wrong default bundle path

**Classification:** Confirmed plugin-owned CLI/documentation defect.

**Trigger and consequence:** Run `qwen3-asr-ane --help`. It advertises `transcribe` with “default `artifacts/qwen3-asr-1.7b`” (`std_qwen3asr_ane/src/std_qwen3asr_ane/cli.py:92-100`). The handler omits `model_dir` when the flag is absent (`cli.py:168-173`), so `Qwen3ASRConfig` actually selects the Standard ASR cache (`plugin.py:103-105`, `141-163`). The root README correctly says the old repository-relative `artifacts/` location is no longer implicit (`README.md:135-143`).

The command is also a second user-facing transcription interface with its own flags and JSON-only output (`cli.py:162-188`), despite the package intentionally exposing the upstream Standard ASR CLI as the canonical interface (`pyproject.toml:28-32`). It lacks the standard CLI's artifact advisory, `--json`/clean-text modes, prompt and portable options, diagnostics rendering, and discovery semantics. The README wisely does not teach this command, but installing it keeps the ambiguity.

**Recommended fix:** In this pre-alpha project, remove the vendor `transcribe` subcommand and describe `qwen3-asr-ane` as conversion/inspection tooling, or make it an explicit thin delegation to the canonical Standard ASR path without maintaining a second flag vocabulary. At minimum fix the default text immediately.

**Acceptance tests:**

1. Help contains no stale `artifacts/` default.
2. Every end-user transcription example uses `standard-asr transcribe`.
3. If a vendor alias remains, a parity test covers cache/config, language, prompt/options, text/JSON output, diagnostics, error exit codes, and close behavior against the Standard CLI.

### ECO-7 — P3 documentation gap: built-in subtitle renderers are usable but undiscoverable in plugin docs

**Classification:** Confirmed documentation opportunity, not an implementation gap.

**Evidence and consequence:** Standard ASR exports SRT/VTT renderers, and they deliberately render a `segments=None` result as one whole-text cue over `duration` (`renderers.py:279-313`, `316-432`). Qwen returns duration with no segments (`plugin.py:573-579`). The probe confirmed output, but neither README nor `docs/usage.md` mentions `to_srt`/`to_vtt`. Users may assume subtitles are wholly unavailable because the model lacks timestamps, or invent an unnecessary renderer.

**Recommended fix:** Add a small Python example and label the result accurately: one utterance-wide cue, useful as a container format, not word/segment alignment. Keep forced alignment listed as unsupported.

**Acceptance test:** A fake batch result from the plugin renders valid SRT/VTT with cue start 0 and cue end equal to input duration; docs state that this is not aligned timing.

### ECO-8 — P2 portability/discovery gap: unsupported hosts are not represented in package or Standard ASR readiness metadata

**Classification:** Static, cross-platform risk; native-unverified on a wrong host. Ownership is shared: plugin early diagnosis plus an upstream deployment-requirements schema would give the best result.

**Trigger and consequence:** Install/discover on non-Darwin, Intel macOS, or macOS below 15. The human README clearly requires Apple Silicon/macOS 15+ (`README.md:62-65`), but package dependencies have no platform marker (`pyproject.toml:8-16`), Standard ASR properties have no platform/hardware requirement field (`references/standard-asr-audit-2026-09-22/src/standard_asr/contract/properties.py:186-211`), and plugin artifact status checks files/download policy only (`plugin.py:255-318`). The only platform reporting is the separate diagnostic helper (`std_qwen3asr_ane/src/std_qwen3asr_ane/diagnostics.py:22-37`). An application can therefore discover the model and be offered acquisition before learning the host cannot run the ANE bundle.

**Recommended fix:** Make plugin artifact/status preflight return a typed unavailable/operator-action state for unsupported OS/version/architecture before download. Upstream should add structured deployment requirements or an availability projection so UIs can distinguish “installed” from “runnable here.” Avoid hiding the plugin completely: explicit status with a remedy is more useful than silent discovery filtering.

**Acceptance tests:** Matrix fake platform facts for Darwin arm64 supported/old macOS, Darwin x86_64, and non-Darwin; unsupported cases must not download/convert and must give an actionable status/CLI message. A supported host keeps the current behavior.

### ECO-9 — P3 doctor scope can be mistaken for a general dependency verdict

**Classification:** Deliberate upstream scope with misleadingly broad final wording.

**Evidence:** The doctor implementation analyzes NumPy constraints only (`references/standard-asr-audit-2026-09-22/src/standard_asr/toolchain/doctor.py:685-722`), and CLI help correctly calls it “plugin dependency (numpy) conflicts” (`toolchain/cli.py:522-537`). Its successful report ends “No dependency conflicts detected,” which is broader than the analysis. It does not evaluate possible `transformers`, `tokenizers`, `coremltools`, or optional-extra conflicts. Conversion isolation substantially reduces this plugin's risk, but the optional `gpu-draft` and co-installed plugins can still have non-NumPy constraints.

**Recommended fix:** Either generalize doctor to installed-distribution constraint satisfiability or make the headline explicitly “No NumPy dependency conflicts detected.” Keep the three-state exact/unknown logic; do not turn heuristic absence into a clean verdict.

**Acceptance tests:** A test environment with compatible NumPy but an intentionally incompatible non-NumPy shared dependency must either be diagnosed or receive wording that clearly limits the clean verdict to NumPy.

## Recommended sequencing

1. Upstream engine pooling and lifecycle (ECO-1). This determines whether the reference server is practical for a large local engine.
2. Plugin `server` installation recipe plus isolated server CI and WebSocket coverage (ECO-2, ECO-5).
3. Upstream typed provider params over every wire/CLI path (ECO-3), preserving one shared API.
4. Server readiness (ECO-4) and host support in artifact/status reporting (ECO-8).
5. Remove or align the vendor transcription CLI, document renderers, and narrow doctor wording (ECO-6, ECO-7, ECO-9).

## Probe results

Executed:

```text
.venv/bin/python research/standard-asr-audit-2026-09-22/probe_ecosystem.py
ecosystem probes passed
```

The run also emitted a Starlette deprecation warning for the installed `httpx` TestClient integration. The declared conversion dependency graph and CI setup, rather than this warning alone, establish that these tests are not an isolated validation of the intended server installation recipe.

Other executed probes are listed in “CLI behavior.” `uv pip check` passed. No pull, model download, conversion, native inference, actual socket listener, real ANE memory measurement, or real-bundle HTTP/WebSocket request was run.

## Inspected files

Plugin/product:

- `AGENTS.md`, `pyproject.toml`, `uv.lock`, `README.md`, `CONTRIBUTING.md`, `.github/workflows/ci.yml`
- `docs/usage.md`, `docs/standard-asr-audit.md`, `docs/installation-verification.json`
- `std_qwen3asr_ane/src/std_qwen3asr_ane/plugin.py`, `streaming.py`, `cli.py`, `runtime.py`, `diagnostics.py`, `acquisition.py`, `conversion_worker.py`, `conversion/build.py`, `conversion/toolchain.py`
- `std_qwen3asr_ane/tests/test_plugin.py`, `test_streaming.py`, `test_artifact_lifecycle.py`

Fresh upstream:

- `AGENTS.md`, `README.md`, `pyproject.toml`
- `docs/content/mission.md`, `engine-authors/adapt-an-asr-system.md`, `specification/protocol.md`, `specification/cli.md`, `specification/server-api.md`
- `src/standard_asr/contract/capabilities.py`, `params.py`, `properties.py`, `metadata.py`, `results.py`
- `src/standard_asr/plugins/discovery.py`, `runtime/interface.py`, `runtime/streaming.py`
- `src/standard_asr/toolchain/cli.py`, `doctor.py`, `server.py`, `renderers.py`, `compliance.py`

## Unverified boundaries

- Native artifact acquisition/conversion and its progress/error output.
- Real Qwen result bodies through REST and real Qwen events through WebSocket.
- Actual Core ML load retention, shutdown memory release, and warm-server performance.
- Clean `std-qwen3asr-ane[server]` installation, because that extra does not yet exist and no networked scratch tool was created.
- Unsupported-host behavior on Intel macOS, older macOS, and non-Darwin.
- Authentication/rate limiting, which upstream explicitly assigns to deployment rather than the reference server.
