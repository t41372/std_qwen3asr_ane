# Bounded native decoder diagnostic

`DecoderBench.swift` replays 100 real generation steps exported by the Python
runtime. Each segment starts with the original KV state. Both harnesses use the
same compiled decoder partitions, inputs and teacher-forced trajectory, and
compare every output hidden element with the exported values. No generation is
continued beyond EOS to manufacture a 100-token sentence.

The timed region includes decoder prediction and host array chaining only.
Audio, tokenization, prefill, LM head, fixture loading, initial KV restoration and
parity checks are excluded. Thus this estimates the benefit available from
replacing the Python bridge between decoder partitions; it cannot establish an
end-to-end speedup or justify a whole-plugin rewrite by itself.

Run from the repository root, on a quiet Mac with Core ML / ANE access. Use new
output paths for each invocation. Build with Swift 6 and a macOS 15+ deployment
target:

```sh
.venv/bin/python experiments/export_native_fixture.py \
  --model-dir artifacts/qwen3-asr-1.7b-p14-lut8-g32-compiled \
  --manifest artifacts/evaluation/smoke/manifest.jsonl \
  --steps 100 --output artifacts/evaluation/round2/native-fixture

xcrun swiftc -O -swift-version 6 -parse-as-library \
  -target arm64-apple-macosx15.0 \
  -module-cache-path artifacts/evaluation/round2/swift-module-cache \
  experiments/native/DecoderBench.swift \
  -o artifacts/evaluation/round2/native-decoder-bench

.venv/bin/python experiments/benchmark_native_fixture.py \
  --model-dir artifacts/qwen3-asr-1.7b-p14-lut8-g32-compiled \
  --fixture artifacts/evaluation/round2/native-fixture \
  --output artifacts/evaluation/round2/native-python.json

artifacts/evaluation/round2/native-decoder-bench \
  artifacts/qwen3-asr-1.7b-p14-lut8-g32-compiled \
  artifacts/evaluation/round2/native-fixture \
  artifacts/evaluation/round2/native-swift.json
```

Add `-sanitize=address` to the Swift build for the separate safety run, use a
different executable/output name, and exclude its timing from performance
comparisons. Both implementations perform one warmup and five measured passes.
Repeat the two processes in reversed order when examining small differences.

Swift owns its `MLMultiArray` inputs and directly passes model outputs to the
next partition. Unsafe buffer access stays inside Core ML's scoped closures;
pointers never escape. State writes respect the strides supplied during mutable
access. The Python public `write_state` bridge requires FP32 input on this SDK;
casting the saved FP16 values to FP32 is exact and happens outside timing.

Fixture tensor sizes, types, paths and SHA256 hashes are validated. The fixture
is bound to the bundle manifest hash; retain the separately fingerprinted model
payloads with the run. Artifacts remain local and are not committed.
