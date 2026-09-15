# Round 2 evidence

Interpret the results through [the result report](../../results-round2.md) and
[the preregistered gates](../../optimization-round2.md). A report's `complete`
means its run finished; it does not mean the candidate passed every gate. In
particular, LUT6 passed the EN/ZH tolerance checks but regressed on additional
languages and is not recommended generally.

`decision.json` records the adopted short-profile checks and their scope. The
general default is unchanged. Both actual 1/10/60-minute idle runs are complete;
the samples in `power/idle-*.jsonl.gz` preserve their observed timing coverage.

- `measurements.json` collects small reports with hashes of their original local
  files. Missing reports remain explicit. Verbose per-layer KV hashes and mixed
  diagnostic transcripts remain in the referenced originals under `artifacts/`.
- The two candidate freeze files predate their respective held-out evaluations.
  The short profile's eligible held-out subset has 137 clips. Its later EOS
  diagnostic repeats the same frozen candidate without tuning it.
- `payloads.json` fingerprints actual inference files, including weight blobs,
  embeddings and tokenizer assets. It also records source hashes at the start
  of the final LUT6 validation stage. Later diagnostic scripts have their own
  hashes in their reports; no inference code was changed during that stage.
- `toolchain.json` records the observed Mac/OS/Swift/SDK details. Instruments
  versions and attribution limits are in the trace summaries.
- `idle-environments.json` records both actual Python environments. They differ
  because the conversion and GPU-draft dependency groups are incompatible;
  compare each wake with its own warmed reference.
- `power/` contains deterministic gzip copies of the original ABBA and idle power
  samples. `sources.json` hashes compressed and original bytes. These are
  whole-machine PSTR estimates with varying desktop activity, not calibrated
  or per-device energy measurements.

Repository paths in these JSON copies are relative where possible; their source
hashes refer to the originals before path conversion. Use the checked-in
`experiments/workflows/round2_*.sh` scripts from the repository root. They supply
the absolute executable paths required by Instruments. Use fresh output paths
and inspect any failed stage before resuming; do not overwrite earlier evidence.

Models, audio, the large native replay fixture and raw Instruments host metadata
are not included here. The native fixture and executable hashes identify the
local files used in that bounded decoder-only diagnostic.
