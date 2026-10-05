# 2026-09-22 Standard ASR audit evidence

Start with the [consolidated report](../../docs/standard-asr-comprehensive-audit-2026-09-22.md) and [upstream feedback](../../standard-asr-feedback-2026-09-22.md). The six domain reports retain detailed authority references, recommendations, and unverified boundaries.

The audit changes documentation and adds model-free probes only. It does not repair the observed product defects. A probe that exits successfully may be asserting the current defect, not the desired fixed behavior.

## Reproduce the observations

From the repository root with its existing locked development environment:

```sh
STANDARD_ASR_ALLOW_DOWNLOAD=0 .venv/bin/python research/standard-asr-audit-2026-09-22/probe_contracts.py
STANDARD_ASR_ALLOW_DOWNLOAD=0 .venv/bin/python research/standard-asr-audit-2026-09-22/probe_artifacts.py
STANDARD_ASR_ALLOW_DOWNLOAD=0 .venv/bin/python research/standard-asr-audit-2026-09-22/probe_streaming.py
STANDARD_ASR_ALLOW_DOWNLOAD=0 .venv/bin/python research/standard-asr-audit-2026-09-22/probe_audio_results.py
STANDARD_ASR_ALLOW_DOWNLOAD=0 .venv/bin/python research/standard-asr-audit-2026-09-22/probe_ecosystem.py
```

The ecosystem probe needs the already installed FastAPI/TestClient dependencies. Their presence in this development environment does not establish that an ordinary base-only installation can serve HTTP/WebSocket. The artifact probe uses temporary placeholder files and never loads their contents as native models.

Saved observations are `contracts-probe.log`, `artifacts-probe.json`, `streaming-probe.json`, `audio-results-probe.json`, and `ecosystem-probe.log`. The parent independently reran all five probes after the domain reviews; all exited zero. `verification.json` distinguishes the baseline suite, the two sandbox-related native rechecks, and audit-specific probes.

## Source identity

`snapshot.json` records both model declarations, every schema, and SHA-256 hashes for the plugin and fresh upstream Python sources. Regenerate with:

```sh
.venv/bin/python research/standard-asr-audit-2026-09-22/snapshot.py
```

This requires the fresh reference clone at `references/standard-asr-audit-2026-09-22`, checked out at `1b2cf3fa5860c075e5160eb60b26b708a7c8bfea`. The reference clone is intentionally ignored by Git. Its upstream is `https://github.com/standard-voice/standard_asr.git`; the reports also record every authoritative relative path and line.

The installed framework's 39 Python files matched that clone byte for byte. The only change since the project's pinned upstream revision is in two CI workflows. The official Qwen capability comparison uses the explicitly identified older local Qwen checkout, not an unverified claim about its latest remote release.
