# Apple runtime options for Qwen3-ASR 1.7B

Research date: 2026-09-12. Scope: a macOS 27.0 (26A428) M5 Max with 64 GB of unified memory, and this repository's existing FP16 Core ML bundle: a fixed 1,024-token cache, T=16 prompt batches, seven stateful decoder partitions, and `CPU_AND_NE` execution. This is research only: no model weights were downloaded, converted, compiled, or run for this note.

## Decision

**Keep Core ML as the production route and treat Core AI as a separately-qualified research branch.** Apple has introduced Core AI in the 27 SDK, but it is a new `.aimodel` runtime and toolchain rather than an upgrade to Core ML 9 or a drop-in compiler for this `.mlpackage` bundle. Neither Apple nor the current public Qwen3-ASR 1.7B ports provide evidence that it will beat the project's 0.435-second 15-second English MLX baseline while retaining quality and making most work run on ANE.

Core ML remains the only option here that can explicitly exclude the GPU: [`MLComputeUnits.cpuAndNeuralEngine`](https://developer.apple.com/documentation/coreml/mlcomputeunits/cpuandneuralengine) allows CPU and ANE but not GPU. This project already uses that policy and records both compute-plan estimates and hardware traces. Core AI can prefer ANE, but it cannot request ANE-only execution; unsupported operations may run on GPU or CPU. That distinction matters more than an API name when the requirement is lower energy through mostly-ANE execution.

The current build should therefore make Core ML faster first. A Core AI migration is worth a bounded prototype only after its fixed-shape decoder has passed source parity, placement, latency, and energy gates on this M5—not after a successful export.

## What is new in macOS 27

### Core AI is a new framework, not Core ML 10

[Core AI](https://developer.apple.com/documentation/coreai/) is Apple’s new on-device neural-network deployment stack. It converts models with `coreai-torch` into `.aimodel` assets, specializes them for the current hardware, and exposes Swift APIs such as `AIModel`, `InferenceFunction`, `NDArray`, and `ComputeStream`. Apple’s [WWDC26 introduction](https://developer.apple.com/videos/play/wwdc2026/324/) describes conversion, specialization, AOT compilation, profiling, and a debugger as one Core AI workflow. The separate, official [apple/coreai-models](https://github.com/apple/coreai-models) repository contains export recipes, PyTorch primitives, and a Swift runtime package.

Core AI is available from macOS/iOS 27.0 and Xcode 27.0 onward according to the [official repository](https://github.com/apple/coreai-models). The SDK on this machine contains `CoreAI.framework`; its Swift interface marks the APIs `@available(macOS 27.0, ...)`. The active developer directory normally points to Command Line Tools, but `/Applications/Xcode-beta.app/Contents/Developer` is Xcode 27.0 build `27A5209h` and contains the framework.

There is an important local tooling limit: with that `DEVELOPER_DIR`, `xcrun` still cannot find the documented `coreai-build` executable. No `coreai`, `coreai_torch`, or `coreai_opt` Python package is installed in the system interpreter either. The framework can be developed against, but this particular Xcode installation cannot yet perform the documented AOT recipe. This is a host-toolchain fact, not a claim that `coreai-build` does not exist in newer Xcode 27 builds.

### Core AI’s ANE control is a preference, not a restriction

`SpecializationOptions(preferredComputeUnitKind: .neuralEngine)` is documented as a preference. [Apple’s specialization guide](https://developer.apple.com/documentation/coreai/managing-model-specialization-and-caching) says the default chooses CPU, GPU, and ANE combinations to minimize latency. An Apple engineer further clarifies in the [Core AI forum answer](https://developer.apple.com/forums/thread/831967) that a neural-engine preference retains all compute units, schedules ANE-capable operations there, and falls back for other operations; only `.cpuOnly` excludes alternatives. There is no Core AI runtime API reporting the device for individual operations. Apple directs developers to the Core AI Instruments trace for that evidence.

For a model with a large projection on GPU, a preferred-ANE flag does not satisfy the project goal. The acceptance record must include a trace showing ANE activity and the absence or bounded duration of GPU activity, alongside overall wall time and measured energy.

### AOT, specialization, and profiling are useful, but not an acceleration claim

Core AI automatically specializes an `.aimodel`; the result is specific to the hardware and OS. [AOT compilation](https://developer.apple.com/documentation/coreai/compiling-core-ai-models-ahead-of-time) moves much of this startup work to a development machine, while [`AIModelCache`](https://developer.apple.com/documentation/coreai/managing-model-specialization-and-caching) retains the resulting specialization until source change, OS update, or cache eviction. This can reduce load latency, but it does not reduce per-token decoder math.

The [Core AI Instruments guide](https://developer.apple.com/documentation/coreai/analyzing-model-runtime-performance-with-instruments) combines specialization, load, setup, inference, ANE, GPU, and CPU timelines. It is a better placement/e2e timing tool for a Core AI candidate than Core ML’s static compute plan. It is still necessary to capture a real trace; a preferred device or a successful load is not hardware evidence.

The macOS 27 release notes label Core AI as beta and list material caveats: AOT can fail for some models; dynamic-shape state plus dynamic outputs can fail unless outputs are preallocated; `coreai-torch` 0.4.0 assets have a specialization bug fixed by converting with 0.4.1+; custom Metal-kernel assets may fail to load; and several compression forms may miss ANE. See [the Core AI section of the macOS 27 release notes](https://developer.apple.com/documentation/macos-release-notes/macos-27-release-notes). The same note’s “large model loading >1 GB” ANE improvement is explicitly scoped to iOS 27, so it is not evidence of an M5 Mac speedup.

## Core ML 9 and macOS 27: what is and is not usable here

[coremltools 9.0](https://github.com/apple/coremltools/releases/tag/9.0), already pinned by this project, adds Python 3.13 support, int8 model I/O, model-state read/write, macOS 26 deployment targets, and `AllowLowPrecisionAccumulationOnGPU`. The latter is explicitly a GPU optimization hint, not an ANE transformer feature. The release notes do not identify a new Core ML 9 ANE decoder compiler, stateful-KV facility, or macOS 27 transformer path that makes the present graph obsolete.

The new int8 I/O type could reduce conversion overhead only if an input or output is semantically safe to quantize and the surrounding host contract changes with it. It cannot be applied casually to the decoder’s hidden states, masks, RoPE inputs, or logits without a new quality proof. State read/write is not a replacement for the project’s existing `MLState` lifecycle and does not remove the current host dispatches.

The current Core ML approach retains two advantages over a Core AI migration:

* It supports a controlled `CPU_AND_NE` execution policy, so GPU work is disallowed rather than merely discouraged.
* It already has the correct audio-prompt semantics, stable FP16 RMSNorm, T=16 prefill, partitioned persistent KV cache, quality evaluation, `MLComputePlan` inspection, and real ANE trace evidence.

Core ML compute-plan output remains *anticipated* device placement. Apple documents `preferredComputeDevice` and operation costs as compiler estimates, not runtime telemetry. Continue to use the existing hardware trace as the placement gate; do not use an all-ANE plan as a performance or energy result.

## Official Core AI model guidance applied to this graph

Apple has no Qwen3-ASR recipe in the current [Core AI Models catalog](https://github.com/apple/coreai-models/tree/main/models). Its audio recipes cover CLAP, Wav2Vec 2.0, and Whisper; its Qwen recipes cover text Qwen3/Qwen3 MoE and Qwen3-VL. The official [Whisper export](https://github.com/apple/coreai-models/blob/main/models/whisper/export.py) is a useful API example but has a dynamic decoder-input sequence and is not a prevalidated low-latency, stateful Qwen3-ASR design.

Apple’s own [Core AI model-authoring guidance](https://github.com/apple/coreai-models/blob/main/skills/skills/model-authoring/SKILL.md) is specific enough to form a prototype contract:

| Area | Applicable Core AI ANE recipe | Consequence for this project |
| --- | --- | --- |
| Layout and projections | Fully static BC1S tensors `(B, H*D, 1, S)` with `Conv2d(1x1)` projections. | This matches the present channel-first Core ML decoder. Preserve it; do not port the decoder as ordinary `nn.Linear` tensors if ANE is the goal. |
| Attention | Per-head sequential attention with explicit masks rather than GPU-oriented fused SDPA. | Keep the present non-materialized GQA/head grouping and static causal masking. |
| Precision | FP16 only on ANE; the guidance says no FP32 literals anywhere in the ANE graph. | The stable RMSNorm/SiLU/GELU representation needs a new Core AI parity proof. A FP32 fallback would undermine the ANE objective. |
| Shapes | Static inputs, outputs, intermediates, and cache capacity. Dynamic shapes are the GPU-oriented path. | Use fixed prompt chunks and an S=1 decoder; retain a fixed 1,024 cache ceiling. Pad and mask tails rather than growing decoder shapes. |
| KV cache | The ANE pattern is read-only functional I/O: cache supplied as input and updated K/V returned as outputs. Apple warns against stateful transforms for token generation because state can reset between inferences. | Do **not** assume the current Core ML `MLState` graph transfers unchanged. A Core AI ANE candidate requires explicit K/V input/output plumbing, ownership, and tests across prompt prefill and generated tokens. |
| Compression | Static, palettized weights are the energy-oriented recommendation; FP8, non-FP16-value palettization, and sparse weights may not execute on ANE in the current beta. | Start with FP16 for semantic and placement validation. Only then evaluate a single compression form against the frozen multilingual quality corpus. No 4-bit/int8 speed or energy claim is valid before that gate. |

The guidance also draws the intended split clearly: macOS dynamic KV/linear-INT4 models are optimized for scale and typically GPU; static, palettized models are the energy/ANE representation. Therefore a Core AI macOS port should not use the stock dynamic Qwen export and infer that `preferredComputeUnitKind(.neuralEngine)` will move it to ANE.

## Concrete options

### 1. Continue the present Core ML design — recommended

Use the existing seven-partition, stateful, T=16 FP16 bundle as the base. The next performance work should change one graph/runtime parameter at a time—partition depth, prompt batch length, cache size, head grouping, or host-buffer reuse—while preserving the project’s conversion and evaluation gates. This is the lowest integration risk path because it maintains the Python Standard ASR plugin and the real `CPU_AND_NE` restriction.

For each candidate record these values under the same frozen corpus and decode protocol:

1. source/graph/weight hash, Xcode, OS, coremltools, compute-unit policy, and model load result;
2. `MLComputePlan` only as anticipated placement/cost;
3. a hardware trace containing Core ML, ANE, GPU, CPU, and clock lanes;
4. end-to-end latency split into audio frontend, encoder, prompt prefill, decode, and postprocessing;
5. English WER, Chinese CER, multilingual edge cases, token parity/teacher-forced logits, and the matched MLX result; and
6. measured wall-power samples with the existing idle-subtraction protocol, never latency-derived joules.

Recompiling the unchanged Core ML asset under the current macOS/Xcode is a worthwhile *measurement control*, but not a claimed compiler upgrade. If it improves a trace, compare that change to the old environment before changing graph topology.

### 2. Core AI static-ANE prototype — research only

This is a re-authoring effort, not an artifact conversion:

1. Keep the current 100-frame convolution chunks and 104-token encoder windows. Do not replace them with a 30-second global-attention encoder merely because a public port pads to 30 chunks; the original masking/window semantics must remain identical.
2. Express each encoder/decoder partition in the official static FP16 BC1S contract. Keep the present stable norm and exact activations, and prove it against the authoritative FP32 model before exporting.
3. Export distinct static prefill (`S=16`) and decode (`S=1`) inference functions or assets. Each accepts fixed-size cache tensors and returns updated cache tensors. Continue to pad/mask invalid prompt rows exactly as today.
4. Start with the current seven decoder partitions. Consolidating partitions is an independent compiler-capacity experiment; it must not be bundled with a runtime migration.
5. Drive Core AI with a neural-engine preference, explicitly preallocate outputs for the static contract, and AOT-compile only after the installed `coreai-build` tool is available. Re-specialize after an OS/Xcode update because artifacts are platform-specific.
6. Accept it only if it has real Core AI/ANE/GPU Instruments evidence, a materially ANE-dominant trace, no significant paired quality loss, lower measured energy per audio-second, and end-to-end latency below both the present Core ML result and the 0.435-second MLX reference on the same 15-second English clip.

This branch also needs a Swift/Core AI integration boundary or a supported Python runtime integration. The production package is presently Python/Core ML; switching model formats cannot be treated as a converter-only implementation detail.

### 3. Core AI dynamic-GPU port — useful control, not a solution

Core AI’s dynamic-Qwen path can be useful to compare current Apple GPU code generation with MLX and to validate an audio-embedding bridge. It does not satisfy the ANE/energy target. Apple’s guidance positions dynamic shapes, stateful caches, standard layouts, and linear quantization as the GPU route. Do not let a strong GPU-only result displace the Core ML ANE path unless the requirement itself changes.

## Public Qwen3-ASR work checked since January 2026

No public implementation found meets all of: original Qwen3-ASR 1.7B, full ASR pipeline, mostly verified ANE execution, quality parity, and a paired speed/energy win over MLX. This is a bounded search result, not a claim of global nonexistence.

| Project | What the project itself reports | Assessment against this project |
| --- | --- | --- |
| [aoiandroid / weiren119 Qwen3-ASR-1.7B-CoreML](https://huggingface.co/aoiandroid/Qwen3-ASR-1.7B-CoreML) | FP16/INT8 encoder, but an FP32 mixed-INT8 decoder. Its card explicitly says “Decoder runs on GPU only.” The shown Taiwanese result is 20 samples. | A useful diagnosis of decoder RMSNorm overflow; not an ANE decoder, energy result, or quality-equivalent replacement. |
| [Reza2kn Mega-ASR CoreML](https://huggingface.co/Reza2kn/mega-asr-coreml) | A Qwen3-ASR-derived fine-tune; ONNX encoder plus FP32 Core ML decoder. The card reports FP16 NaNs and an ANE compiler/load failure, then uses GPU. | Has an audio-embedding bridge and conversion code, but it is a different checkpoint and not a 1.7B ANE success. |
| [soniqo/speech-swift](https://github.com/soniqo/speech-swift) / its 0.6B Core ML artifacts | Fixed-shape split decoder/MLState patterns target CPU+ANE, but the readily documented Qwen Core ML model is 0.6B and its encoder uses `.all` in the runtime. | The T-batched prompt/cache runtime ideas remain relevant; dimensions, compile behavior, and numerical range do not transfer to 1.7B. |
| [john-rocky Core AI Model Zoo Qwen3-ASR port](https://github.com/john-rocky/coreai-model-zoo/blob/main/models/qwen3-asr/README.md) | Full 1.7B pipeline: fixed 30-chunk audio encoder, audio-embedding bridge, and a Core AI GPU decoder. It reports one Japanese 17-token exact greedy gate, M4 Max GPU prefill around 0.4 s and 45–57 ms/token decode (18–24 tok/s), and about 2.9 GB on disk. | The best public engineering reference for a Core AI audio-embedding bridge, but explicitly GPU-measured, not M5/ANE-measured, with no energy result and only a one-clip end-to-end quality gate. It cannot establish an MLX win or meet the mostly-ANE requirement. |

The last port is particularly valuable as source material, not as an implementation to substitute unchanged. Its design is a GPU-oriented Core AI engine with a fixed `[1,1]` decoder call; its document says prefill runs as pipelined S=1 calls. The project’s existing T=16 prefill is specifically designed to avoid that host-dispatch cost. Reuse its audio-embedding/input-ID bridge only after proving that it preserves this project’s prompt semantics and does not force a GPU fallback.

## Evidence and links

Primary Apple sources used above:

* [Core AI overview](https://developer.apple.com/documentation/coreai/), [WWDC26 introduction](https://developer.apple.com/videos/play/wwdc2026/324/), and [Core AI Instruments](https://developer.apple.com/documentation/coreai/analyzing-model-runtime-performance-with-instruments).
* [Core AI specialization and cache behavior](https://developer.apple.com/documentation/coreai/managing-model-specialization-and-caching), [compute-unit kinds](https://developer.apple.com/documentation/coreai/computeunitkind), and Apple staff’s [preference/fallback clarification](https://developer.apple.com/forums/thread/831967).
* [macOS 27 Core AI release notes](https://developer.apple.com/documentation/macos-release-notes/macos-27-release-notes), including beta limitations and deployment bugs.
* [apple/coreai-models](https://github.com/apple/coreai-models), its [model catalog](https://github.com/apple/coreai-models/tree/main/models), [Qwen export guidance](https://github.com/apple/coreai-models/blob/main/models/README.md), [model-authoring guidance](https://github.com/apple/coreai-models/blob/main/skills/skills/model-authoring/SKILL.md), and [Whisper export](https://github.com/apple/coreai-models/blob/main/models/whisper/export.py).
* [coremltools 9.0 release notes](https://github.com/apple/coremltools/releases/tag/9.0), [Core ML compute-unit policy](https://developer.apple.com/documentation/coreml/mlcomputeunits), and [Core ML compute-plan documentation](https://developer.apple.com/documentation/coreml/mlcomputeplan).

The repository’s earlier, more detailed third-party Core ML survey remains in [prior-art.md](prior-art.md). This note updates the decision for the new Core AI runtime and does not replace that evidence.
