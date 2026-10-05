# Standard ASR contract / capability audit

审计日期：2026-09-22。产品基线为 `884c22e`；Standard ASR 的结论以 `references/standard-asr-audit-2026-09-22` 的 `1b2cf3f` 为准。brief 已确认该版本与项目锁定版本的 Python 协议代码相同，差异仅在 CI 工作流。

本报告只检查协议契约、声明、参数与本地适配器/运行时的衔接。没有下载权重、运行真实 Core ML 推理或修改产品源代码。`probe_contracts.py` 中的结论明确是 model-free mock/reproduction；不能替代真实 bundle 的质量、延迟或 tokenizer 容量验证。

## 结论

两个入口点都能被发现，类级 metadata、显式 artifact 生命周期、纯构造、闭合的 `ProviderParams`、标准参数门控、wire format 和流式生命周期均通过当前 Standard ASR 合规检查。不能把大量正确委托给 `EngineBase` 的行为误报成插件缺失：它确实负责 provider 参数 swap 安全、strict/best_effort、候选语言的诊断性忽略、phrase hints 到 prompt 的 opt-in 降级、音频协商和流式 session 建立。

确认有三个需要处理的产品问题：

| 优先级 | 分类 | 问题 |
|---|---|---|
| P1 | 确认缺陷 | Standard ASR 接受的 BCP-47 细化语言（如 `en-US`）被原样交给只接受基础语言 tag 的 Qwen 运行时；batch 变成 `TranscriptionError`，streaming 变成错误事件。 |
| P2 | 确认契约/发现性缺陷 | `std-qwen3asr-ane/1.7b` 的 init config 可选择 `profile="short-dictation"`，实际选择短听写 bundle，却仍以 general preset 的类级身份和 Properties 被发现。 |
| P2 | 确认缺陷，条件触发 | 实际 CoreML 路径在解析阶段丢弃未知模型语言名，令后续承诺的 `detected_language_unmapped` diagnostic 不可达。 |

另有一个低优先级 config UX 问题：本插件选用的 `LanguageConfigMixin` 会公开 `default_candidate_languages`，但插件正确地声明 candidates 不支持；用户可保存一个每次 auto 请求都会被诊断为忽略的默认值。这不是本插件漏接 native 能力；插件可通过只声明契约所需的 `default_language` 避免该字段。将两个字段拆成上游 helper mixin 仍会改善作者体验。

## 已读的权威与范围

- 共同 brief：`research/standard-asr-audit-2026-09-22/brief.md:1-27`。
- 上游使命、协议与作者指南：`references/standard-asr-audit-2026-09-22/AGENTS.md:1-44`、`docs/content/mission.md:1-64`、`docs/content/specification/protocol.md:209-300,308-445,485-573,775-825,948-1025,1362-1388`、`docs/content/engine-authors/adapt-an-asr-system.md:1-246`、`docs/content/engine-authors/plugin-entry-points.md:1-246`。
- 上游 executable authority：`src/standard_asr/contract/capabilities.py`、`params.py`、`language.py`、`properties.py`、`metadata.py`、`runtime/config.py`、`runtime/gating.py`、`runtime/interface.py`、`runtime/streaming.py`。
- 产品端：`pyproject.toml`、`plugin.py`、`streaming.py`、`runtime.py`、`languages.py`、`profiles.py` 及其协议测试。
- native 家族的可行性参考：`references/Qwen3-ASR/README.md:59-67,155-207,289-305` 与 `qwen_asr/inference/qwen3_asr.py:300-364`、`utils.py:37-106`。这不是对当前 ANE bundle 已实现能力的证明。

未验证的边界：真实 1.7B/short Core ML bundle、真实 tokenizer 下的 prompt 余量、真正的自动语言输出分布、任何 ForcedAligner 的 ANE 转换和性能，以及本地 bundle 路径被手动指向非该入口点 artifact 时的行为。

## 两个 preset 的静态面

入口点是正确且彼此独立的：`pyproject.toml:24-26` 将 general 绑定到 `Qwen3ASREngine`，short 绑定到 `ShortDictationEngine`。这符合 Standard ASR 的入口点模型；entrypoint 直接是具体 class，所以不用 factory return annotation。

| 静态项目 | `std-qwen3asr-ane/1.7b` | `std-qwen3asr-ane/1.7b-short-dictation` | 证据 |
|---|---|---|---|
| concrete class / config schema | `Qwen3ASREngine` / `Qwen3ASRConfig` | `ShortDictationEngine` / `ShortDictationConfig` | `plugin.py:172-175,607-621` |
| `model_name` / `max_audio_duration` | `1.7b` / 30 s | `1.7b-short-dictation` / 12 s | `plugin.py:176-188,615-620` |
| 输入与采样率 | array；native/accepted/required 都是 16 kHz | 相同 | `plugin.py:180-184` |
| incremental wire | mono 16 kHz；`pcm_s16le`、`pcm_f32le` | 相同 | `plugin.py:185`；上游 session guard `runtime/interface.py:1962-2007` |
| 可选/可检测语言 | 30 个 BCP-47 基础 tag + `auto`；可检测集合为该 30 tag | 相同 | `plugin.py:186-187`，`languages.py:8-39` |
| init 默认 generation budget | 256 | 128 | `plugin.py:114,607-609` |
| capabilities / metadata / provider params | 完全相同的 class-level tree、artifact declaration、`Qwen3ASRParams` | 继承相同值 | `plugin.py:190-231,612-621` |

`BaseProperties.max_audio_duration` 是批量/whole-input audio 的静态输入边界，不是对每一种 prompt、generation budget 与已载入 bundle 组合的成功保证。native `prepare_prompt()` 仍会以真实 cache 检查 `len(prompt) + max_new_tokens - 1`（`runtime.py:605-616`），streaming 也会用配置和载入 bundle 的较小音频限制（`streaming.py:113-125`）。因此本报告不会把“Properties=30 秒”说成“所有 30 秒以内请求必然成功”。

`declared_metadata` 的 artifact 区段是完整且合适的：适用、支持显式 acquisition、且推理中不会隐式 acquisition（`plugin.py:212-219`）。其 `x_qwen3asr_streaming` 描述了 cumulative-audio、context 和 rollover 限制（`plugin.py:220-229`）；这是静态运营说明，不是可通过 `supports()` 查询的标准 capability，归属正确。`EngineBase` 的 artifact status/acquisition 模板再调用产品的 `_artifact_requirements` / `_acquire_artifacts`，所以不是缺失 artifact 生命周期。

## 完整 capability tree 与有效值

Standard ASR 规定 `DeclaredCapabilities` 是免实例化可读的完整静态树，`effective_capabilities` 只可收窄（`protocol.md:318-335`；实现 `capabilities.py:654-810`）。两个类都没有覆写 `effective_capabilities`，因而继承 `EngineBase` 的“effective = declared”实现（`runtime/interface.py:809-841`）。`use_draft` 只改变精确 speculative decode 的实现路径而不改变输出契约，故不需要收窄。

下表列出 canonical JSON 的每个标准节点；未列出的标准节点均为 fail-closed `false` / enum 的 off 值，而不是“未知”。直接 probe 得到的 canonical JSON 与下列静态代码一致。

| 路径 | 两个 preset 的值 | 结论 / native 落点 |
|---|---|---|
| `batch` | present | 支持完整音频转写。 |
| `batch.language.runtime_override` | true | `params.language` 经 base resolution 后给 `_transcribe`，再变成 native tag / `None`。 |
| `batch.language.candidate_languages` | false | native CoreML prompt 只有一个可选强制语言参数，没有 candidates。见“语言”小节。 |
| `batch.word_timestamps` | false，`granularities=[]` | 当前结果没有 words/segments/timestamps。 |
| `batch.guidance.prompt` | true，`max_tokens=128` | `params.prompt` 传给 native `context`。 |
| `batch.guidance.phrase_hints` | false | 没有原生 keyterms；可由上游 gate opt-in 降级为 prompt。 |
| `batch.diarization` / `.always_on` | false / false | 当前 native result 没有 speaker；没有假造标签。 |
| `streaming` | present | 支持 incremental audio 和 whole-input streaming output。 |
| `streaming.language.*` | override=true，candidates=false | 与 batch 同样解析，params 在 session 创建时冻结。 |
| `streaming.word_timestamps` | false | 当前没有 timestamps。 |
| `streaming.guidance.prompt` | true，`max_tokens=128` | session 把 frozen prompt 交给每次 cumulative decode。 |
| `streaming.guidance.phrase_hints` / `.mutable_mid_stream` | false / false | 可降级一次；会话中不更新 guidance。 |
| `streaming.diarization` / `.always_on` | false / false | 不产生 speaker。 |
| `streaming.emits_partials` | true | 每个 chunk decode 产生 partial。 |
| `streaming.re_segments` | false | 同一 `utterance-0` 累积更新，不产生 supersede。 |
| `streaming.word_stability` | false | partial 一律 `stable_until=0`。 |
| `streaming.reconnect` | `unsupported` | 没有 reconnect 协议。 |
| `streaming.finality_level` | `closed` | 结束时产生 closed event；现有 test 记录验证。 |
| `streaming.timestamps` | `none` | event 不输出 start/end/words。 |
| `streaming_input` / `streaming_output` | true / true | 对应 PCM 喂入和流式事件输出。 |
| `self_resamples` | false | 不自称内部重采样；Properties 的 16 kHz 要求由标准层协商。 |

流式实现与以上声明一致：`Qwen3ASRSession._decode()` 构造 `stable_until=0` partial（`streaming.py:152-165`），结束时产生 closed event（`streaming.py:221-231`），并没有 timestamp / word / speaker / supersede 分支。`test_incremental_pcm_tail_prefix_and_closed_event` 也显式断言 partial stability、无 timestamps、closed final（`tests/test_streaming.py:121-151`）。这是一项已验证的成功，而不是缺失 `re_segments` 或 `word_stability` 的实现。

## EngineBase 已提供的路径，和插件真正负责的部分

`EngineBase.transcribe()` 已顺序执行协议线检查、语言 config 总性检查、`provider_params` 与 standard params gate、语言/候选语言解析、audio preparation、作者 `_transcribe` hook、结果 speaker 合成和 diagnostics 合并（`runtime/interface.py:1455-1583`）。`transcribe_async()` 是其异步桥。产品只需要 `_transcribe`，它在锁内惰性加载 runtime，再传 language、generation budget 和 prompt context，最后以 `TranscriptionResult` 返回（`plugin.py:542-587`）。

同理，`EngineBase.start_transcription()` 负责 input/output axis、wire format、参数 gate、语言解析、whole-input audio preparation、session diagnostics 和 deadline 覆盖；产品 `_start_transcription()` 仅构造 `Qwen3ASRSession`（`runtime/interface.py:2079-2290`，`plugin.py:595-604`）。`recommended_wire_format()` 也是基类由 Properties 派生（`runtime/interface.py:2009-2061`），不是插件漏实现。

插件另行正确覆写 `prepare()`，以锁保护的 `_ensure_model_loaded()` 作显式预热（`plugin.py:454-518`），符合 init 纯、后续惰性加载的分界。构造只解析 config 与建立空引用/锁（`plugin.py:233-238`）。已有 `test_construction_is_pure_and_runtime_import_is_lazy` 覆盖此点（`tests/test_plugin.py:155-174`）。

## Init config 与每请求参数

| 面 | 字段与职责 | 审计结果 |
|---|---|---|
| 标准 init | `strict`、`allow_private_urls` | 都来自 `BaseConfig`；`strict` 正确驱动 base gate。插件只接受 ARRAY，URL 放宽当前不影响 native 输入。`config.py:829-888`。 |
| 标准 init | `default_language="auto"`、`default_candidate_languages`、`download_root` | 语言轴 totality 正确；候选 default 的 UI 问题另列为上游反馈。`plugin.py:95-105`，`config.py:1982-2009`。 |
| 插件 init | profile、bundle/source/draft paths、draft 开关/量化、默认 `max_new_tokens`、stream cadence/queue/limit | 这些固定于 engine lifetime，除了 profile 问题外归 init 合理。`plugin.py:102-139`。 |
| 标准 runtime | `language`、`candidate_languages`、timestamps、diarization、prompt、phrase hints、degrade policy | `RuntimeParams` 是封闭模型，均由 base gate；产品没有额外顶层 runtime 字段。`protocol.md:485-505`，`gating.py:163-253`。 |
| provider runtime | `Qwen3ASRParams.max_new_tokens` | 正确使用 terminal、closed `ProviderParams` 子类；每请求值经 `_generation_budget()` 给 batch 和 streaming，未改变 config 默认。`plugin.py:166-169,589-593`。 |

`max_new_tokens` 的可接受性仍依赖当前 bundle 的真实 cache 和 prompt；runtime 在进入 decoder 前显式检查并拒绝超过 cache 的组合（`runtime.py:575-616`）。没有证据表明 `max_tokens=128` 的静态 guidance 声明已经不够保守：Standard ASR 规定它只是 script-aware 近似而非该 tokenizer 的精确 BPE 计数，作者应留 headroom（`protocol.md:518-524`）。这是一个要做真实 bundle matrix 的验证项，不是本次静态审计可确认的 defect。

`provider_params` swap safety 是正确的 inherited 行为。gate 在任何 audio/native 工作前以精确 `type(provided) is expected` 检查（`gating.py:163-188,685-723`）；两 entrypoint 的相同 Qwen provider schema 只含语义相同的 generation budget，未观察到跨 preset 被悄悄丢弃的字段。`check_provider_params_swap_safety` 和本报告 probe 都验证外来 terminal type 一律抛 `InvalidProviderParamError`。不要为“插件没有手写 isinstance 检查”报缺陷。

## 语言、检测和 candidates

### 已正确实现的部分

`LANGUAGE_NAMES` 以 30 个 Qwen 显式强制语言控制映射为 BCP-47，Properties 把它们同时用于 selectable/detectable，再加入 `auto`（`languages.py:1-39`，`plugin.py:186-187`）。上游 Qwen 原生库也列出同一 30 个 canonical names 并要求 forced language 是其中之一（`references/Qwen3-ASR/qwen_asr/inference/utils.py:37-106`）。因此不把 22 个中文方言识别宣传成额外可选择或可检测 BCP-47 tag 是谨慎且正确的；是否能提供稳定的方言语言元数据尚未做真实模型验证。

forced language 时插件传 `None` 到 `detected_language`，所以不会把强制选择伪报为 auto detection；auto 时可将已知模型语言名映回 BCP-47（`plugin.py:542-579,624-644`，`languages.py:44-68`）。candidate list 没有 native 参数：上游 native `transcribe` 只接收每个音频一个 optional forced language（`qwen3_asr.py:300-364`），当前 ANE `build_prompt` 也只插入一个 language control（`runtime.py:210-233`）。故 `candidate_languages=false` 是有依据的诚信声明，不应为了“功能覆盖率”把多候选硬限制伪装成可支持。

当 caller 在 auto 模式传 candidates，Standard ASR 特意不论 strict/best_effort 都产出 `candidate_languages_ignored` 而不把它传给 native（`language.py:278-309`）。model-free public probe 确认了该 diagnostic。这是协议规定的 carve-out，不是静默丢参。

### P1：RFC 4647 细化语言没有归约到 Qwen 基础 tag

**触发与后果。** `RuntimeParams(language="en-US")` 是合法 tag。Standard ASR 用 `en` 命中 declared selectable set 后会保留完整 `en-US`，并明确要求引擎自行归约（`protocol.md:282-284`；`runtime/interface.py:1769-1831`）。产品 batch 将 `params.language` 原样传入 `runtime.transcribe`（`plugin.py:545,567-572`）；streaming 也是如此（`streaming.py:138-145`）。而 ANE prompt builder 只允许 `LANGUAGE_NAMES` 的基础 key（`runtime.py:220-233`），所以 `en-US`、`zh-Hant` 等会抛 `ValueError`。batch 被产品包装成 `TranscriptionError`（`plugin.py:580-587`）；streaming 会投影为 `invalid_audio_or_context` terminal event（`streaming.py:251-264`），把语言兼容性错误误归到 audio/context。

**证据。** `probe_contracts.py` 通过 public `engine.transcribe` 和 recording runtime 证明 base gate 接受 `en-US`，并且 adapter 的 native handoff 保持 `en-US`；不加载模型。上游规范的归约义务和 runtime 的基础-key 检查共同构成完整静态证明。

**所有权与修复。** adapter/runtime，P1。加一个单一 helper，将已经被 Standard ASR 认可的 BCP-47 tag 归约为 Qwen 的 primary language code，然后所有 native 调用共享它；`auto` 仍转 `None`。不能仅缩小 `selectable_languages`，因为 Standard 的 RFC 4647 行为就是允许 refinement。不要对 `default_language` 采用此放宽：该字段按规范仍必须精确属于 selectable set。

**验收测试。** 对两个 preset，batch 和 incremental streaming 各用 `en-US`、`zh-Hant`；recording/native spy 必须分别得到 `en`、`zh`，并完成而不产生 engine error。强制语言下 `detected_language is None`。不属于任一 primary tag 的合法 BCP-47 仍遵守 strict/best_effort 的 Standard gate。

### P2：未知检测语言的 disclosure 在真实路径丢失

**触发与后果。** `detected_language()` 的注释承诺 model 返回未发布语言名时输出 `None` 加 `detected_language_unmapped` diagnostic（`plugin.py:624-644`）。但是 `CoreMLRuntime.parse_output()` 已对 metadata language 调 `normalize_model_language()`（`runtime.py:250-263`）；未知 name 直接变 `None`。`RuntimeResult` 只有这个规范化后的 `language` 与 raw transcript（`runtime.py:189-196,786-794`），调用端于是把 `None` 传给 disclosure helper，正常返回空 diagnostics。现有测试通过 monkeypatch 让 fake runtime 直接返回 `language="Klingon"`（`tests/test_plugin.py:516-532`），没有覆盖真实 `parse_output`。

**边界。** 未证实真实 Qwen bundle 会实际输出未知 name；缺陷是“若发生，当前宣称的诚实 disclosure 不可能发生”，不是“模型必然会输出 Klingon”。

**所有权与修复。** runtime/adapter，P2。使 `parse_output` 同时保留 raw language name，或让 `RuntimeResult` 有 `unmapped_model_language`；batch `detected_language()` 和 streaming `_language_fields()` 均使用该值。仍不可把未知 name 写入 `detected_language`，因为该字段必须是具体 BCP-47。

**验收测试。** 对真实 parser 输入 `language Klingon<asr_text>hello`，batch result 与 streaming content events 都有 `detected_language=None`，并有一次 `detected_language_unmapped`（或等价的现有 structured disclosure）且 `provided="Klingon"`；known `English` 仍映射 `en`。

## P2：general entrypoint 的 `profile` 是不应存在的 preset 选择器

**触发与证据。** Standard ASR 明定模型选择属于 entrypoint preset，不属于 init `model` field（`protocol.md:808-810`；作者入口点指南也要求每个 preset 用各自 class/entrypoint）。这里已有独立 short entrypoint，但 general `Qwen3ASRConfig.profile: ProfileName` 仍允许 `"short-dictation"`（`plugin.py:102`）。validator 会选择 `qwen3-asr-1.7b-short-dictation` 路径并把默认 generation budget 变成 128（`plugin.py:141-163`），实例类型和 class-level `properties` 却仍为 `Qwen3ASREngine` / `1.7b` / 30 seconds（`plugin.py:172-188`）。现有测试本身固定了该行为（`tests/test_plugin.py:183-203`）。本报告的 public probe 同时打印 general class 及其 short bundle selection。

**精确影响。** 这不是“任何 ≤30-second request 必须成功”的断言：真实 native context 可以因 prompt、generation budget 或 bundle 限制更早拒绝。问题在于 discoverable general model 的静态身份、UI/调度可读 Properties、config schema 与实际选中的 short preset 不再是同一个 preset；应用无法仅靠入口点元数据看到自己配置选择了 short 的 12-second 设计边界。它也把已有的 short entrypoint 变成可绕过的平行选择机制。

**所有权与修复。** adapter/config，P2。general config 不应接受 profile 来更换 preset；short only 应由 `ShortDictationEngine` / `ShortDictationConfig` 的 entrypoint 选择。保留 `model_dir` 作为 artifact location 是正常 init config，但在加载时应把可检查的 manifest identity/duration 与该 class 的 preset expectations 作早期一致性检查，避免把明显错误的 artifact 归为不透明 native failure。后半点需要定义对自建 general bundles 的兼容政策，故本审计不把手工 `model_dir` 指向不同 duration 的情况单独计作确认 defect。

**验收测试。** `registry.create("std-qwen3asr-ane/1.7b", profile="short-dictation")` 必须被 config schema/validation 拒绝；short entrypoint 仍以 12-second Properties 和 128 default 创建。两类 class 以 manifest 作预热时，对明确属于另一 preset 的 bundle 在 native inference 前给 actionable `ConfigError`。`standard-asr show` / config schema 不再向 general model 显示会更换 preset 的 profile field。

## 可实现但当前未实现的能力

| 能力 | 判断 | 证据和建议 |
|---|---|---|
| batch word/segment/char timestamps | 可行的新特性，当前诚实不支持 | Qwen 参考实现可用独立 `Qwen3-ForcedAligner-0.6B` 产生 timestamps（`references/Qwen3-ASR/README.md:65,176-207,295-305`）；当前 ANE runtime `RuntimeResult` 没有 words/timestamps，Properties/capabilities 正确为 false。实现需要新 artifact、转换/runtime、结果映射、配置和 `effective_capabilities`（未配 aligner 时收窄），绝不能只翻 capability。 |
| streaming post-aligned timestamps | 可行但更大 | 上游 Qwen streaming 明示不返回 timestamps（`README.md:289-291`）。可在 closed final 后 post-align，但需要明确 latency/内存/错误契约和 `streaming.timestamps=post_align`，不是已有 native frame-aligned 功能。 |
| phrase hints | 已有安全 fallback；原生 feature 未实现 | native 有 free-text `context`，当前 Standard ASR 的 opt-in phrase-hints-to-prompt degradation 是完整、带 diagnostic 的语义，不应错误地把 `phrase_hints=true`。若实现真正 boost term，应声明 limits、保留 fallback 意义并做可量化质量验证。 |
| candidate language hard restriction | 当前有正当理由不支持 | 当前 native 只有一个 forced language；候选 allowlist/preference 未见入口。除非 native prompt/decoder 有可验证的限制语义，否则不能把列表拼进 context 后宣称 hard candidates。 |
| diarization / reconnect / word stability / segment rollover | 当前有正当理由不支持 | 当前 result/event 没有 speaker、reconnect 和 supersede；partial 全可改写且 `stable_until=0`。需要独立模型/会话设计后才可加。 |

## 可避免的 config UX，及可选的 Standard ASR helper 改进

`LanguageConfigMixin` 同时暴露 `default_language` 和 `default_candidate_languages`（`runtime/config.py:1982-1996`）。作者指南说 config 必须有可用的 `default_language`，并建议“inherit `LanguageConfigMixin` to get the field”（`docs/content/engine-authors/adapt-an-asr-system.md:20-25`）；它没有规定必须继承该 mixin。IC.5 把“字段出现在 config model”定义为 applicable（`protocol.md:802-806`）。因此本插件目前选择该 mixin 后，config schema 显示一个不被 native/capability 支持的 default candidates 字段；用户设置它后，base 每个 auto request 都产生 `candidate_languages_ignored`（`runtime/interface.py:1730-1845`，`language.py:278-309`）。

插件可直接声明 `default_language`，保留 IC.6 的 totality，同时不展示无效 default candidates；这是可独立落地的 P3 UX 改进。上游若把 `default_language` 与 `default_candidate_languages` 拆成独立 mixin，作者无需重复字段定义，属于可选 helper 改进，而不是当前协议阻塞或 native feature gap。

## 验证记录

已运行，均成功：

```text
.venv/bin/python research/standard-asr-audit-2026-09-22/probe_contracts.py
.venv/bin/pytest -q \
  std_qwen3asr_ane/tests/test_plugin.py::test_discovery_and_official_compliance \
  std_qwen3asr_ane/tests/test_plugin.py::test_short_profile_defaults_preserve_explicit_and_environment_values \
  std_qwen3asr_ane/tests/test_streaming.py::test_prompt_and_phrase_hint_degradation_are_real_capabilities \
  std_qwen3asr_ane/tests/test_artifact_lifecycle.py::test_short_preset_is_discoverable_and_has_its_own_limits
.venv/bin/standard-asr compliance entrypoints
.venv/bin/standard-asr compliance run std-qwen3asr-ane/1.7b
.venv/bin/standard-asr compliance run std-qwen3asr-ane/1.7b-short-dictation
```

两次 `compliance run` 明确没有合成 `check_event_sequence` 与 `check_transcription_result`；产品已有 recorded event/result tests 覆盖它们的正常样本，但合规 green 不会发现本报告的 profile identity、RFC refinement mapper 或 parser data-loss 问题。这正是应补上上文验收测试的原因。
