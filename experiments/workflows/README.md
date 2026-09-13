# Optimization loop scripts

Shell scripts that run the measurements in the order the results file reports
them. Terms (smoke, selection set, held-out, ABBA, trace) are explained in
`research/glossary.md`. In short: "smoke" is two official test recordings
(English 15 s, Chinese 4 s); "selection" is 200 sentences used to choose
between candidates; "held-out" is another 200 used once for the final check;
"ABBA" is alternating run order for energy measurements; a "trace" is an
Instruments recording of Neural Engine and GPU activity.

These are the serial, hardware-bound stages of the candidate loop used in this
workspace. They are shell wrappers over the model-agnostic tools
(`experiments/evaluate.py`, `experiments/benchmark_energy.py`,
`experiments/trace_ane.py`, `experiments/measure_system_memory.py`) and the
package CLI. Run them one at a time from the workspace root with no other model
work on the machine; each waits for nothing and writes into `artifacts/`.

1. Preregister gates (see `research/preregistration-2026-09-13.md`) before any
   candidate result exists, and commit that file.
2. `candidate_gate.sh <bundle-name>`: smoke warm latency (1 warmup, 5 repeats)
   and the selection-set quality run with paired comparisons against the FP16
   ANE and official CPU FP32 baselines. Use it to choose among candidates.
3. `final_gate.sh <bundle-name>`: for the single chosen candidate only. Held-out
   quality (once), placement of every graph, equal-work ABBA energy against the
   FP16 bundle and the MLX references, peak process memory, an isolated
   Instruments ANE trace, real-time streaming, silence, plugin verification and
   the Standard ASR compliance CLI pointed at the bundle.
4. `p14_and_draft_evidence.sh`: the evidence run for the two-model p14 bundle
   (placement, streaming, silence, compliance, isolated trace + binding) and
   the GPU-draft experiment (200-utterance token parity, equal-work ABBA
   energy against serial p14, system memory). Needs the separate
   `experiments/mlx_draft/` environment.
5. `closing_checks.sh`: after the above, the speculative trace, the rebuild
   of p14 through the CLI path with `weight.bin` hash comparison, and a
   fresh-clone `uv sync --frozen` + pytest check (`SCRATCH` sets the clone
   directory).
6. `draft_review_followup.sh` and `idle_power.sh`: review follow-ups (the
   speculative-only trace, the LUT8 compact-head probe) and the 30-second idle
   power sample taken after the chains finish.
7. Record everything, including failures, in `research/technical-blog.md`.

`<bundle-name>` is the suffix of `artifacts/qwen3-asr-1.7b-<bundle-name>`;
the scripts use the `-compiled` sibling produced by `qwen3-asr-ane compile`.

Added 2026-09-13 (afternoon), after review of the comparison set:

8. `quality_matrix.sh`: every GPU reference (MLX bf16 / 8-bit / 4-bit, official
   PyTorch on MPS) on the selection and held-out sets, paired against the
   official CPU FP32 model, the FP16 ANE bundle and the 8-bit ANE bundle.
9. `draft_plugin_parity.sh`: the packaged draft path through the Standard ASR
   engine (`draft_dir` set) on both 200-sentence sets; hypotheses must equal
   the serial default bundle's. Needs the `.venv-draft` environment (see the
   root README) and the draft bundle from `qwen3-asr-ane build-draft`.
10. `energy_matrix.sh [repeats]`: one-session equal-work energy for every path
    in the results table (FP16, 8-bit 7-file, 8-bit 2-file, MLX bf16/8-bit/4-bit,
    packaged draft), forward then reversed order, with a 30 s idle sample.
11. `memory_matrix.sh`: wired and process memory for MLX 8-bit and the packaged
    draft path.
