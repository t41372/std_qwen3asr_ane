# Standard ASR 音频、结果与错误边界审计

审计日期：2026-09-22。插件基线为 `884c22e`；Standard ASR 权威代码为 `references/standard-asr-audit-2026-09-22` 的 `1b2cf3fa5860c075e5160eb60b26b708a7c8bfea`。共同 brief 已确认该版本与项目锁定版本的 Python 协议代码一致。

本报告只检查 Standard ASR 的音频类型、协商、重采样、限制、结果模型、renderer、diagnostic、exception 和 protocol boundary 与 ANE adapter/runtime 的衔接。没有下载权重、转换 bundle 或运行真实 Core ML 推理。`probe_audio_results.py` 使用 fake runtime，经 public API 复现协议行为；它只能证明控制流、类型和数据投递，不能证明识别质量或 ANE 数值一致性。

## 结论

主要成功项是：本插件诚实地只声明 native `ARRAY`/16 kHz 输入，但继承的 Standard ASR 管线仍让应用可用 path、bytes、base64 和 array；strict/best-effort 的缺失采样率行为、重采样 diagnostic、30/12 秒静态 batch 边界、portable batch exception、stream error event、KV/output budget 的无截断拒绝都能工作。

需要处理或反馈的边界如下。

| 优先级 | 分类 | 结论 |
|---|---|---|
| P1 | 上游协议安全歧义 | terminal stream error 只存在于事件；`session.result()` 没有 `error` 字段，也不抛错。只读取 collapsed result 的应用会把失败误读成空/部分成功。 |
| P1 | 已确认 adapter/native 缺陷，跨报告 | Standard ASR 接受 `en-US` 等 BCP-47 refinement，插件却原样传给只接受基础 tag 的 runtime；batch 变 `TranscriptionError`，stream 变 `invalid_audio_or_context`。 |
| P2 | 已确认 cross-layer 音频边界缺口 | direct `AudioArray` 是 passthrough；Standard 不 downmix、不限制维度/幅度，且只诊断不消毒 non-finite。ANE runtime 只接受 nonempty finite mono，导致 stereo/non-finite/empty 被误分类成引擎故障，而超出 `[-1,1]` 的有限值静默进入特征提取。 |
| P2 | 已确认 metadata-loss 缺陷 | native parser 先把未知或 compound model language 归一化为 `None`，adapter 承诺的 `detected_language_unmapped` diagnostic 在真实路径不可达。 |
| P2 | 已确认 mode-limit 不一致 | whole-input streaming 先经过全局 `Properties.max_audio_duration`，所以 general 的 `audio=31s` 在 session 建立前被 30 秒静态边界拒绝，即使配置和 loaded bundle 都允许 180 秒；incremental wire 路径却可继续。 |
| P2 | 上游 capability/result 表达缺口 | 插件知道 input frontier 和最终输入时长，但只能把 `audio_prefix_seconds` 放进 private `extra`；标准 `audio_processed_until` 被 `streaming.timestamps` 绑定为 speech timestamp。collapsed streaming result 也丢失已知 duration。 |
| P3 | 可选 observability 改进 | native `raw_text`、token ids、audio token count 和 timings 不进入 Standard result。协议允许放 `result.extra`，但这些属于内部调试数据，当前不暴露不是合规缺陷。 |

## 权威路径与输入矩阵

Standard ASR 的 `AudioInput` 有六个显式 variant，bare coercion 只接受 `str/PathLike`、`bytes`、`ndarray` 与 `(ndarray, sample_rate)`；bare string 永远是本地 path（上游 `audio/input.py:34-57,59-105,151-244,257-333`）。插件只声明 `accepted_input={ARRAY}`，native/accepted/required sample rate 均为 16 kHz（`plugin.py:176-188`）。因此实际可用性来自 inherited negotiation，而不是插件自己重写 decoder。

| 应用输入 | 实际路径 | fake-runtime 结果 | 审计结论 |
|---|---|---|---|
| `AudioArray`、`(ndarray, sr)` | array passthrough，必要时 resample | 成功 | 正确；adapter 收到 float32/16 kHz。 |
| bare `ndarray` | `sample_rate=None` | strict 抛 `AudioProcessingError`；best-effort 假定 16 kHz并发 `assumed_sample_rate` | 正确。 |
| `AudioPath`、bare path | decode to mono array，再 resample | 8 kHz WAV 成功并有 decode/resample diagnostics | 正确；依赖可用 decoder。 |
| `AudioBytes` | decode to mono array，再 resample | 成功 | 正确。 |
| `AudioBase64` | bounded b64 decode，再 audio decode/resample | 成功 | 正确。 |
| `AudioUrl` | 无 fetch path | `IncompatibleAudioInputError`，提示 caller 提供本地文件 | 符合当前代协议；Standard 不代为抓 URL。 |
| `AudioStorageUri` | 插件不接受 `storage_uri` | `IncompatibleAudioInputError` | 符合当前代协议；Standard 不是 cloud upload/fetch broker。 |

协商权威是上游 `audio/negotiation.py:326-414,417-512,535-594`；转换执行、rate、duration 和 diagnostic 合并在 `audio/conversion.py:201-292,577-738` 与 `runtime/interface.py:1455-1617`。probe 还确认 8 kHz 输入重采样到 16 kHz并报告 `resampled_with=scipy`；如果 scipy 不可用，上游会使用 anti-aliasing fallback 并把 warning 写入 diagnostic（`audio/resampling.py:35-82`，`audio/conversion.py:687-738`）。

incremental streaming 建立时由 Standard ASR fail-closed 检查 encoding、mono 和 sample rate；它不重采样 wire frame（`runtime/interface.py:513-592`）。插件声明并实现 `pcm_s16le` 与 `pcm_f32le`（`plugin.py:185`，`streaming.py:174-196`）。现有测试覆盖默认 s16、f32、odd byte tail、NaN 与超长输入；这些行为与声明一致。

## Finding 1 — terminal error 被 collapsed result 丢弃

**严重度：P1。分类：上游协议/文档歧义，不是 ANE plugin 独有缺陷。**

触发条件是任意非 recoverable stream error。ANE 插件能正确发出具体事件：native 未分类异常由 base producer 投影为 `engine_error`；`ModelLimitError` 由插件投影为 `bundle_capacity_exceeded`；invalid PCM/context 为 `invalid_audio_or_context`；session/bundle duration 为 `audio_limit_exceeded`（插件 `streaming.py:232-264`；上游 producer `runtime/streaming.py:2920-3039`）。

问题在 collapse 边界。`TranscriptionResult` 没有 `error` 字段（上游 `contract/results.py:550-620`），`StreamReducer.result()` 只构造 text、segments、detected language 和 diagnostics，完全不记录 error event（`runtime/streaming.py:1099-1168`）。`TranscriptionSession.result()` 原样返回这个 reducer 结果（`:3345-3351`）。fake runtime 复现：

- `RuntimeError("native detail")` → terminal event `error(code="engine_error")`，随后 `session.result()` 返回 `text=""`、`segments=[]`，无错误标记；
- `ModelLimitError("decoder KV cache exhausted")` → terminal `bundle_capacity_exceeded`，collapsed result 仍是同一无错误形状。

应用若完整消费并检查每个 event，错误不会丢；但 app guide 紧接事件介绍后又把 `session.result()` 描述为与 batch 相同的 constant-shape result（`docs/content/app-developers/streaming.md:53-64,109-119`），没有说明 collapsed result 不能表达终态失败。只使用这个 convenience API 的应用会得到 fake success；若错误前已有 final，甚至会得到貌似完整的 partial result。

**建议所有权与修复：Standard ASR protocol。** 在 pre-alpha 阶段明确选择一种可移植语义：`TranscriptionResult.error: StreamError | None`，或让 `session.result()` 在 terminal non-recoverable error 后抛一个携 code/retry 信息的 typed exception。不要把 error 塞进 `extra`，也不要要求每个插件发自定义 diagnostic；这是公共 collapse contract。recoverable error 与已有 partial/final 的语义也应一并钉死。

**验收：** 对 `engine_error`、artifact errors、backpressure、deadline、plugin-defined capacity error 和 recoverable `content_lost` 建表测试。delivered terminal event 与 `session.result()`/exception 必须表达同一 terminal outcome；只调用 collapse API 不得得到无错误的成功形状。wire SDK 采用相同规则。

## Finding 2 — Standard 允许的 BCP-47 refinement 未归约到 Qwen tag

**严重度：P1。分类：已确认 adapter/native 缺陷；同时见 `contracts.md` 与 `native-features.md`。**

Standard ASR 的 RFC 4647 lookup 接受 `en-US` 对 declared `en`，并保留 refinement 交给 engine 归约。插件 batch 与 streaming 把 `params.language` 原样传入 runtime（`plugin.py:542-572`；`streaming.py:138-145`）；runtime 只允许 `LANGUAGE_NAMES` 的基础 key（`runtime.py:575-598`）。所以 forced refinement 在 batch 被 generic `TranscriptionError` 包装，在 streaming 被错误归为 `invalid_audio_or_context`。

forced base tag 的结果语义本身正确：native 可使用 requested language，但 Standard result 的 `detected_language` 保持 `None`，没有把 caller 的选择伪报成检测结果（`languages.py:54-68`，`plugin.py:624-645`）。缺陷是 handoff tag 没归约。

**建议所有权与修复：adapter/runtime boundary。** 用一个共享 helper 把 Standard 已接受的 BCP-47 refinement 解析为 Qwen primary tag，再供 batch/stream prompt 使用；保留 Standard 的 refinement diagnostic。不要缩小 declared language list 来绕过 Standard 的 lookup 语义。

**验收：** batch 与两种 streaming input 各测试 `en-US`、script/region refinement、base tag 和 `auto`。native handoff 必须使用基础 Qwen tag；forced result/event 的 `detected_language` 仍为 `None`；不得产生 `invalid_audio_or_context`。

## Finding 3 — direct AudioArray 的 canonical shape/range 与 ANE runtime 不相容

**严重度：P2。分类：已确认 cross-layer contract gap；包含 upstream 语义矛盾和 adapter 错误归类。**

`AudioArray` 明确允许 mono 1-D 或 `(n_samples, n_channels)` multi-channel，canonical 是 float32 mono `[-1,1]`（上游 `audio/input.py:87-105`）。但 `AudioArray -> ARRAY` 选择零成本 passthrough（`audio/negotiation.py:417-436`），执行时只做 `np.asarray(float32)`，不会 downmix、clip 或验证维度（`audio/conversion.py:577-636`）。non-finite 也被有意原样转发，只添加 warning（`:452-479`）。这与规范把 canonical array 写成 mono `[-1,1]`、并声称 batch 可 downmix 的文本存在张力（`docs/content/specification/protocol.md:39,117-140,979`）。Properties 又没有 array channel/range/finite 约束，插件无法通过 declaration 说“ARRAY，但只接受 finite mono”。

ANE runtime 则明确要求 `samples.ndim == 1`、nonempty、finite（`runtime.py:575-594`）。插件 `_transcribe` 的 catch-all 把这些 `ValueError` 全包成 `TranscriptionError("Qwen3-ASR Core ML inference failed.")`（`plugin.py:580-587`）。public fake-runtime 复现结果：

- stereo `AudioArray([n,2], 16000)` 直达 runtime，随后成为 `TranscriptionError`，而非 downmix + diagnostic 或 caller-owned audio error；
- NaN array 在 Standard conversion 产生的 warning 无机会随 result 返回，最终也是 generic `TranscriptionError`；
- empty same-rate array绕过 resampler 的 empty check，最终同样是 generic `TranscriptionError`；
- `[2.0,-2.0]` 有限 array 无 diagnostic、无 clip，原值进入 runtime。实际 runtime 会据此计算 mel，可能静默改变识别质量。

decode path 没有同一问题：`decode_audio(..., target_channels=1)` 会输出 mono normalized array（上游 `audio/conversion.py:605-636`）。因此同一 stereo 波形用 WAV/path 输入与 direct array 输入有不同边界行为。

**建议所有权与修复：** Standard ASR 先决定 direct array 的 normative domain。若 canonical 是交付保证，所有 ARRAY delivery 应统一 downmix/finite/range policy并发准确 diagnostic；若 direct array 真的是 caller-owned raw passthrough，则 Properties 必须能声明 channel count、finite/range requirement，让 negotiation 在 plugin hook 前 fail-loud。插件可先做明确 preflight，将 mono/finite/empty 归为 `AudioProcessingError`，但不应自行静默 clip/downmix而与 Standard 的其他 engine 分叉。

**验收：** 同一 stereo/NaN/out-of-range/empty waveform 经 `AudioArray`、WAV bytes 和 path 三条路径，得到规范指定且一致的 adapter input或一致的 typed caller error。任何 downmix/clip/sanitize 都有 machine-readable diagnostic；任何拒绝都不得映射成 engine 5xx。

## Finding 4 — unknown/compound model language 在 diagnostic 前消失

**严重度：P2。分类：已确认 plugin runtime/parser defect。**

adapter 已写好诚实降级：auto mode 如果 model 返回已知名称就映射为 BCP-47；未知名称应使 `detected_language=None` 并带 `detected_language_unmapped` diagnostic / stream `unmapped_model_language` extra（`languages.py:44-68`，`plugin.py:624-645`，`streaming.py:167-172`）。但真实 runtime 的 `parse_output()` 在提取 `language ...` 行时立即调用 `normalize_model_language()`，未知值直接变 `None`（`runtime.py:250-263`）。后层再也看不到原值。

probe 把 `language Klingon<asr_text>hello` 串过真实 parser 再进 adapter helper，得到 `text="hello"`、parser language `None`、零 diagnostic。`Chinese,English` 也被现有测试明确归为 `None`；官方 long-audio helper 可产生 compound language string，这一点在 `native-features.md` 有完整上游证据。当前 fake plugin tests 直接让 fake runtime 返回 `"Klingon"`（`tests/test_plugin.py:516-530`；streaming 对应测试），因此绕过了正是会丢值的 parser，形成测试盲点。

**建议所有权与修复：runtime parser + adapter。** `RuntimeResult` 保留 raw model-language token；只有 adapter 决定它是否能放进 standardized `detected_language`。unknown/compound 值进入 diagnostic 或明确命名的 engine-specific field，不得伪造 BCP-47，也不得静默丢弃。

**验收：** 从 raw decoder text 开始测试 known single language、`language None`、缺少 language 行、unknown 与 `Chinese,English`。known single 映射到 BCP-47；forced request不报告 detection；其余原始 metadata 可见但不进入 `detected_language`。

## Finding 5 — whole-input streaming 被 batch/global 静态 duration 提前截断

**严重度：P2。分类：已确认 cross-layer mode-limit mismatch。**

`EngineBase.start_transcription(audio=...)` 复用 `_prepare_audio()`，其中无条件传 `properties.max_audio_duration`（上游 `runtime/interface.py:2239-2248,1585-1617`）。转换层在 decoded array 上提前按 sample count 检查 duration（`audio/conversion.py:482-508`）。插件 general/short Properties 分别是 30/12 秒（`plugin.py:176-188,612-621`）。另一方面 streaming config 默认 180 秒，session 再以 `min(config.stream_max_audio_seconds, runtime.max_audio_seconds)` 判定真实 limit（`plugin.py:135-139`；`streaming.py:113-125`）。

fake runtime 宣告 180 秒、config 也设 180 秒时：incremental wire 可以进入 session，而 `start_transcription(audio=31s)` 在 hook 前直接抛 `AudioProcessingError`，永远到不了 bundle-specific check。插件 metadata 所写的 “minimum of configuration and loaded bundle duration” 对 whole-input mode 不完整；还有隐藏的 static Properties limit。

默认 shipped general bundle 的 30 秒仍与 Properties 一致，所以这不是默认安装的 31 秒 regression；它影响更大 context bundle、自建 bundle以及“streaming output”两种 transport 的一致性。一个 `max_audio_duration` 字段同时承担 batch preflight 与 whole-input streaming，上游当前无法按 mode 表达。

**建议所有权与修复：Standard ASR Properties/capability model + plugin declaration。** 增加 mode-specific duration bound，或明确 global property 必须涵盖两种 mode并把 loaded-bundle动态限制留给 adapter。随后把 `x_qwen3asr_streaming` 的 effective limit描述补全。不要仅调大 30 秒而让默认 batch 的超限从 caller-owned `AudioProcessingError` 退化成 late engine 5xx。

**验收：** 对 general、short 和一个 fake 180 秒 bundle，把 12/30/31/180/180+ε 秒同时跑 batch、whole-input streaming 和 incremental streaming。每个 mode 的公开声明、preflight、terminal code 和 loaded bundle限制必须一致。

## Finding 6 — streaming progress、duration 与 renderer 可用性

**严重度：P2。分类：可行功能缺口 + 上游表达限制；当前没有伪造 timestamp。**

插件每次 cumulative decode 精确知道已处理 sample 数，却把它写为 `event.extra["audio_prefix_seconds"]`（`streaming.py:152-165,221-230`）。标准字段 `audio_processed_until` 留空，因为 capability 的 `streaming.timestamps` 只有 `native_frame_aligned|post_align|none`，而 compliance 将任何 cursor 都视为 ASR timestamp；当前声明是 `none`，填写会成为 `stream_exceeds_timestamps`（上游 `contract/capabilities.py:487-503`，`compliance.py:2423-2491`）。输入 frontier 不是模型测得的 speech timestamp，所以插件没有冒充该能力是正确的。

结果是 portable consumer 看不到 progress cursor；`session.result().duration` 也始终是 `None`，即使成功 session 的最终输入长度已知。probe 的 0.5 秒成功 session得到 partial/final private extra 0.5，所有标准 cursor为 `None`，collapsed duration也是 `None`。

renderer 随之出现有意但需要文档化的差异：同一段 0.5 秒 fake audio 的 batch result 是 `segments=None` + known duration，SRT renderer产生 `00:00:00,000 --> 00:00:00,500` 的全文 cue；stream reducer把 closed event归为一个 `start=None,end=None` segment，默认 renderer按协议抛 `SubtitleRenderingError`。调用方显式选择 `on_unrenderable="collapse"` 后虽保留全文，却因为 collapsed duration也是 `None` 而使用 Standard 的 3 秒 fallback cue，不是已知的 0.5 秒输入长度。该行为来自上游 null/render policy（`contract/results.py:11-16,550-620`；`renderers.py:280-316,316-420`），不是 plugin renderer bug。当前模型没有 forced aligner，不能拿 chunk update time伪造 speech span。

**建议所有权与修复：** Standard ASR 把“已消费 input frontier”与“识别出的 speech timestamp”拆开，或为 `audio_processed_until` 定义不会暗示 alignment 的 capability；插件随后迁移 private extra。无论 timestamp 方案如何，session 可在成功终态把已知 submitted/processed duration写入 collapsed result。文档明确 no-timestamp stream 的 renderer 需要 explicit collapse。不要用 `audio_prefix_seconds` 合成 word/segment时戳。

**验收：** portable app 只读标准字段即可看到 monotonic consumed-audio cursor和最终 duration；同一字段不承诺词/段 alignment。无 aligner时 `start/end/words` 仍为空，default renderer继续 fail-loud，显式 collapse保留全文。

## Native limits、exception 与 diagnostics：已确认成功

- **静态 duration：** decoded batch/whole-input audio超过 30/12 秒，在 inference 前由 Standard 抛 `AudioProcessingError`（上游 `audio/conversion.py:482-508`；plugin Properties `plugin.py:176-188,612-621`）。边界用 `>`，恰好 limit 可通过。
- **loaded bundle audio limit：** native `prepare_prompt()` 再按 manifest `max_audio_seconds` 精确检查，抛 `ModelLimitError`（`runtime.py:378-388,575-594`）。batch adapter 保留 message 与 `__cause__`，包装成 portable `TranscriptionError`（`plugin.py:580-587`）；stream 返回 `audio_limit_exceeded` 或 `bundle_capacity_exceeded`，并区分 limit source（`streaming.py:113-125,236-259`）。
- **prompt/context capacity：** Standard 先按 declared `PROMPT_MAX_TOKENS=128`做 strict reject或 best-effort truncation diagnostic；native 再按真实 tokenizer 的 `len(prompt)+max_new_tokens-1 <= cache_length`检查（`plugin.py:77-83,190-207`；`runtime.py:605-616`）。后者防止 word estimate低估 BPE token时越界。
- **output budget：** serial generation到 `max_new_tokens`仍无 EOS时抛 `ModelLimitError`，拒绝 truncated transcript（`runtime.py:759-784`）；speculative decoder有同一 guard（`speculative.py:70-75`）。这是正确的 no-silent-truncation 行为。
- **portable batch error：** native未分类 exception 变 `TranscriptionError`并保留 cause，符合上游 batch runtime contract（上游 `contract/exceptions.py:190-230`；协议 `protocol.md:571-573`）。
- **stream error：** base producer保证任何逃逸 exception成为 drop-proof terminal `engine_error`；artifact exception有专用 code（上游 `runtime/streaming.py:2920-3039`）。插件把已知 caller/session/bundle condition分成自己的稳定 code，没有让 exception直接炸掉 iterator。
- **diagnostic merge：** Standard gating、language、conversion 和 plugin diagnostics按顺序合并进 batch result（上游 `runtime/interface.py:1523-1583`），stream建立期 diagnostics附到 `session.diagnostics()`（`:2228-2288`）。probe确认 resample、decode、assumed rate均可见。

## 输出模型与被省略字段

batch adapter只交付 `text`、合法时的 `detected_language`、精确输入 `duration` 与 diagnostics（`plugin.py:542-579`）。以下 Standard fields为空，与声明相符：`language_confidence=None`，`segments=None`，`words=None`，`channels=None`，`extra={}`。插件没有声明 timestamps/diarization/channel separation，所以不应为了填表而伪造这些值。

native `RuntimeResult` 还有 `raw_text`、`token_ids`、`audio_tokens` 与 `timings`（`runtime.py:189-196,789-806`），adapter全部丢弃。它们不是 Standard 一等字段；若未来要公开，应在稳定 schema、隐私和体积策略确定后进入明确命名的 `extra`。特别是 `raw_text` 含模型控制 metadata，不宜无条件进入普通 end-user result。当前省略属于合理的内部边界。

renderer 本身没有发现 plugin-specific defect。batch 结果无 segments 时可用 known duration产生单 cue；streaming untimed segments默认 fail-loud，显式 `collapse` 可保全文。没有运行实际播放器，只核对了 renderer authority 与模型形状。

## 验证

运行 audit probe：

```bash
.venv/bin/python research/standard-asr-audit-2026-09-22/probe_audio_results.py
```

运行插件 fake-native regression：

```bash
.venv/bin/pytest -q \
  std_qwen3asr_ane/tests/test_plugin.py \
  std_qwen3asr_ane/tests/test_streaming.py \
  std_qwen3asr_ane/tests/test_runtime.py
```

结果：`104 passed in 4.78s`。

运行 fresh upstream audio/result tests（项目 venv没有 pytest-cov，所以清空 upstream addopts）：

```bash
PYTHONPATH=references/standard-asr-audit-2026-09-22/src \
  .venv/bin/pytest -q -o addopts='' \
  references/standard-asr-audit-2026-09-22/tests/test_audio_input.py \
  references/standard-asr-audit-2026-09-22/tests/test_audio_negotiation.py \
  references/standard-asr-audit-2026-09-22/tests/test_audio_conversion.py \
  references/standard-asr-audit-2026-09-22/tests/test_resampling.py \
  references/standard-asr-audit-2026-09-22/tests/test_results.py
```

结果：`426 passed in 1.03s`。这证明当前实现与自己的测试一致；它不会否定以上缺少测试或由设计本身造成的边界缺口。

## 已检查文件与未验证边界

上游已检查：`AGENTS.md`、mission、protocol、engine-author guide、streaming app guide；`audio/input.py`、`format.py`、`negotiation.py`、`conversion.py`、`resampling.py`、`loader.py`、`wav.py`、`wire.py`；`contract/properties.py`、`capabilities.py`、`params.py`、`language.py`、`results.py`、`exceptions.py`；`runtime/interface.py`、`gating.py`、`protocol_boundary.py`、`streaming.py`；`renderers.py` 及相关 upstream tests。

插件已检查：`plugin.py`、`streaming.py`、`runtime.py`、`audio.py`、`audio_context.py`、`streaming_context.py`、`speculative.py`、`languages.py`、`profiles.py`、`errors.py` 和相应 plugin/runtime/streaming/audio tests。

未验证：真实 bundle 的识别质量、所有 30 个语言的真实 auto metadata、真实 compound/code-switch输出频率、Core ML 对超范围振幅的数值影响、真实 decoder 对各种压缩容器的行为、fallback resampler在缺 scipy环境的听感、真实 180 秒 bundle、renderer在播放器中的端到端显示，以及 prompt 128-token margin对真实 tokenizer所有对抗字符串的充分性。这些不能从 fake-runtime pass推断。
