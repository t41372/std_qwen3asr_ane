# Final native regression — 2026-10-02

## Outcome

All final real-model regression gates passed against product commit
`a8ae9140de59b666b1f0bd9c9db10471e0bad669` and the root `.venv` installation
of Standard ASR commit `5f6eef25e35e5e66e9010474e6dee531021e61f1`.
The current package was exercised without a `PYTHONPATH` override. The archived
`884c22e` source override was used only for the older side of the explicit
baseline comparison.

The machine-readable evidence is:

- `native-baseline-884c22e-2026-10-02.json`
- `longform-validation-2026-10-02.json`
- `packed-parity-2026-10-02.json`

All three records have `status: passed`, empty failure lists, exact verifier
hashes, and no performance claim. The dated 2026-09-22 evidence was not
overwritten.

## Exact native baseline

The two official smoke fixtures and the first two manifest rows from each fixed
English and Chinese held-out set produced six comparisons. Every current result
matched the archived `884c22e` result for normalized text, raw text, token IDs,
and EOS ID.

## Long-form and streaming quality

The original preregistered cases and thresholds were retained.

| Case | Duration | Reference error | Joined individual-baseline error | Result |
|---|---:|---:|---:|---|
| General English | 35.105 s | WER 1.22% | WER 1.22% | pass |
| General Chinese | 33.880 s | CER 0.00% | CER 0.00% | pass |
| Short English | 21.780 s | WER 0.00% | WER 0.00% | pass |
| Short Chinese | 25.110 s | CER 0.00% | CER 1.41% | pass |
| Short mixed English/Chinese | 16.204 s | CER 0.00% | CER 0.00% | pass |

Whole-input Chinese streaming, the default two-second partial regression, and
233-frame irregular PCM streaming all matched their public batch transcript
exactly. Their terminal order, monotonic bounded cursors, result durations,
and unique closed-window sample spans passed. The verifier composes closed text
from each event's exact `text_separator` and requires it to equal
`result.text`.

The 46.505-second diarization fixture retained 1.90% WER against both the label
and joined individual baseline. It returned 105 ordered, bounded words, two
speaker labels, six unique measured turns, and a 97.14% speaker-labeled word
fraction. The new projection emitted 11 segment finals across two native input
windows (nine plus two). Coverage is therefore checked by unique adjacent
window spans rather than incorrectly counting each segment as another input
window. Every exposed global source range was bounded and sliced its exact
event text.

## Packed parity and telemetry

Three fixed real-audio inputs matched their serial results for text, raw text,
language, token IDs, and EOS ID. Every outcome reported `execution="packed"`
and shared one group measurement. Per-item results expose
`packed_item_preparation_seconds`; shared measurements use the
`packed_group_*` names. Packed results omit the unverifiable per-item
`prefill_calls`, `generation_seconds`, and `total_seconds` fields.

The evidence retains the observed group lane, cache-slot, decoder-call,
head-call, and elapsed fields because they are runtime facts. Concurrent host
work was uncontrolled, so no latency, throughput, speedup, energy, or hardware
placement conclusion is drawn from them.

## Source identity

The long-form evidence records exact SHA-256 hashes for the plugin, streaming,
long-form, diarization, audio frontend and cache, postprocessing, auxiliary,
result-text, runtime, bulk, and verifier sources. Key hashes are:

| Source | SHA-256 |
|---|---|
| `streaming.py` | `bfc04328636f65624d69f51d66ed3f7c82f605eb08871b59dd3baf95f10a2834` |
| `audio.py` | `a4739c9ff8783c60d770329aa5c53c3d0c5dd2700e3e2cb543e9b4043227e46d` |
| `runtime.py` | `da7977b1692d6eb0861f73c85027f3a6b8cde423fc98b6e01a428532087d023f` |
| `result_text.py` | `ffd7f8c510f69d0eed49b91d3e9c88066091b30a627cb8c914760dc776fb7b48` |
| target manifest | `6fb52962b00b709994915b857551c92a13074b2223e31de94ac3b73b284fa43f` |
| standalone batch-head manifest | `8900f471afe71c67bea755cf69ce69f6e90549ac99acc654d6d46f2d598a408e` |

The JSON records are authoritative for the complete hash set and exact public
events/results.

## Reproduction

Run from the repository root with the already-synchronized root `.venv` and
the existing local model artifacts:

```console
./.venv/bin/python research/release-readiness/compare_native_baseline.py compare
./.venv/bin/python research/release-readiness/verify_longform.py
./.venv/bin/python research/release-readiness/verify_packed_parity.py
```

Core ML execution may require the normal unsandboxed local-machine test
environment. The scripts do not download, convert, or rebuild model artifacts.

Historical benchmark tables and the September evidence retain their published
latency, speedup, throughput, and energy numbers unchanged. This final run
validates correctness and contract behavior only.
