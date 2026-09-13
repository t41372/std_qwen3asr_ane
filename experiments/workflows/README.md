# Optimization loop scripts

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
4. Record everything, including failures, in `research/technical-blog.md`.

`<bundle-name>` is the suffix of `artifacts/qwen3-asr-1.7b-<bundle-name>`;
the scripts use the `-compiled` sibling produced by `qwen3-asr-ane compile`.
