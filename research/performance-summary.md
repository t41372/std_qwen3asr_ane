# ANE and MLX performance


Measured on a MacBook Pro M5 Max (64 GB, macOS 27.0). Latency is the median of 5 warm runs on two official test recordings (English 15 s, Chinese 4 s). Energy is the whole-machine power estimate from the Mac's power controller (`PSTR`), integrated over equal work and divided by seconds of audio, all seven paths measured in one session (mean of two blocks, forward and reversed order, idle 6.1 W); it is not a wall meter. Memory is the system-wide wired memory added by loading the model and transcribing. Quality is paired word error rate (English, LibriSpeech) and character error rate (Chinese, FLEURS) on 200 selection and 200 held-out sentences, reported as the difference from the official FP32 model in percentage points. Only "identical" rows produce byte-identical text; every other path differs from the official model on some sentences (punctuation, sentence breaks, number formatting, the occasional word) and the error-rate difference is what is compared. Methods, limits and raw numbers are in [research/results-2026-09-13.md](results-2026-09-13.md) (Chinese); terms are in [research/glossary.md](glossary.md).

| Path | EN 15 s | ZH 4 s | J per audio second | Wired memory | Error rate vs official (pp) |
|---|---:|---:|---:|---:|---|
| Neural Engine, FP16 (uncompressed) | 2.06 s | 0.50 s | 3.29 | 4.2 GB | −0.24 to +0.05 |
| **Neural Engine, 8-bit (default)** | **1.43 s** | **0.35 s** | **2.30** | **2.5 GB** | −0.46 to −0.08 |
| Neural Engine verify + GPU draft (optional) | 0.61 s | 0.20 s | 1.41 | 6.5 GB | identical text to default, 400/400 |
| GPU, MLX 8-bit | 0.30 s | 0.11 s | 1.54 | 2.5 GB | −0.41 to 0.00 |
| GPU, MLX bf16 | 0.46 s | 0.14 s | 1.98 | 4.1 GB | −0.30 to +0.05 |
| GPU, MLX 4-bit | 0.21 s | 0.09 s | 1.11 | 2.1 GB | −0.25 to +0.55 (Chinese worse) |
| GPU, official PyTorch (MPS, bf16) | 0.78 s | 0.20 s | not measured | not measured | −0.22 to 0.00 |
| CPU, official PyTorch (FP32) | 2.74 s | 0.80 s | not measured | not measured | reference |

The error-rate ranges combine separate English WER and Chinese CER point
estimates from selection and held-out sets; they are not confidence intervals.
No regression was detected for the default LUT8 bundle under the stated gate.
Negative point estimates are not evidence that quantization improves the model.

What the numbers say:

- On this machine every GPU path is faster than the Neural Engine path, and at equal quality the GPU also uses less total energy: MLX 8-bit needs 1.54 J per second of audio, the Neural Engine 8-bit bundle 2.30 J. The Neural Engine draws about a third of the GPU paths' average power (25 W against 70 to 79 W), keeps the calling Python process's own CPU time low (about 0.05 cores against 0.6; that figure counts only this process, not Core ML's separate service processes or other system work, so it is not whole-machine CPU use), adds little to the Python process's memory, and leaves the GPU free; it does not win on energy per utterance.
- Relative to the FP16 Neural Engine bundle, the 8-bit bundle's gains come from weight compression (27% faster, 27% less energy) and from splitting the decoder into 2 Core ML files instead of 7 (about 5%). Among the serial default-path candidates measured in that study, these produced repeatable end-to-end gains. Similar LUT4/LUT8 probe times are consistent with a bit-width-independent compute/decompression floor; they do not uniquely identify its cause. Further graph-shape, cache and activation-quantization experiments are tracked in [optimization round 2](optimization-round2.md).
- The optional draft path doubles speed and cuts energy by 39% relative to the serial ANE default without changing output: a 0.6B model on the GPU proposes 15 tokens, the 1.7B model on the Neural Engine verifies them in one call and keeps only the prefix it would have produced itself. Every emitted token is the 1.7B model's own choice; on 400 evaluation sentences the text was identical to the serial path. Its 1.41 J estimate differs from MLX 8-bit by about 8%, which this measurement method does not establish as an energy advantage. Cost: a second model in memory and a busy GPU.
- The closest practical quality-matched comparison is Neural Engine 8-bit against MLX 8-bit. The ANE bundle uses palettized weights (a lookup table per 32 output channels) for the decoder and output layer; the MLX 8-bit reference uses affine quantization of the decoder only (group 64). Their quantizers, kernels and prefill/generation shapes differ. MLX 4-bit is the fastest path but loses 0.5 to 1.0 percentage points of Chinese character accuracy, the same trade this project rejected for its own 4-bit bundle.

Not supported: word timestamps, speaker diarization, restricting candidate languages, and audio longer than 30 seconds per utterance. Streaming (partial results while audio arrives) is supported through the serial path.
