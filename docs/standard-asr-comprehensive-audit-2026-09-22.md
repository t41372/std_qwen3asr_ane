# Standard ASR 全面整合調查 — 2026-09-22

> 本文是修復前 `884c22e` 的調查快照。後續實作、已通過的驗證及尚待完成的發佈工作，請以 [handoff](handoff-2026-09-22.md) 與[工作紀錄](release-readiness-2026-09-22.md) 為準。

**目前尚不能稱為完整、高度利用 Standard ASR。** 基本整合已經相當多：公開模板、音訊協商、請求門控、兩種串流輸入方式、背壓、模型取得與標準 CLI 都有實際接入。但仍有合法請求在原生邊界失敗、工件就緒判定不足、診斷在跨層傳遞時消失，以及可合理實作但尚未提供的長錄音與時間戳功能。現有 compliance 與測試通過，沒有排除這些問題。

這一輪是調查與修復規劃，沒有修改推理引擎、升級依賴、轉換或下載模型。報告評估可觀察的程式行為與測試缺口，不推斷過去作者的動機。

## 調查基線、方式與證據強度

| 對象 | 查證版本／範圍 |
|---|---|
| 本專案 | `884c22e`；調查開始時工作目錄乾淨 |
| Standard ASR 最新來源 | 直接 clone 到 `references/standard-asr-audit-2026-09-22`；`1b2cf3fa5860c075e5160eb60b26b708a7c8bfea` |
| 專案鎖定來源 | `8b124e8c8fbcb6b0382792262bee05595b895440` |
| 最新與鎖定版本的差異 | 僅兩個 CI workflow 的 dependency 更新；Python 與協定相同 |
| 已安裝 Standard ASR | 39 個 Python 檔案逐檔比對，與最新 clone 完全相同 |
| 官方 Qwen 參考實作 | 本地 `references/Qwen3-ASR`，`7c6daf77a2421100f5fb066495372c00129d39ff`；未宣稱是其最新遠端版本 |
| 執行驗證 | 現有 496 項測試分兩次全部通過：初跑 494 通過、2 項受 Core ML 編譯沙箱限制；只重跑該兩項後通過。標準 compliance 通過 |
| 不在本輪證據內 | 真實完整模型轉寫、全語言 WER/CER、長錄音品質、ForcedAligner 的 ANE 轉換可行性、耗電與效能重測、乾淨環境重新取得大型模型 |

使用三個 Terra xhigh 與三個 GPT-5.6 Sol high 子任務。每組先讀[共同背景](../research/standard-asr-audit-2026-09-22/brief.md)、上游使命與作者指南，再分別閱讀其契約、框架實作、外掛及原生路徑。主調查者交叉複核所有結論，合併重複根因，將「符合協定的限制」與「接錯介面」分開。

例如，沒有輸出 `audio_processed_until` 原本被提議當作外掛漏接，複核後確認目前 timestamp capability 會禁止它，改列上游表達能力問題；`profile` 的問題也限定為 preset 身份與 IC.7，而不是聲稱 30 秒以內必須在任意 prompt 下成功。

可重現資料：[版本與驗證紀錄](../research/standard-asr-audit-2026-09-22/verification.json)、[實際兩個 preset 的完整宣告與 schema](../research/standard-asr-audit-2026-09-22/snapshot.json)、[快照產生程式](../research/standard-asr-audit-2026-09-22/snapshot.py)。快照逐檔記錄 39 個上游與 30 個產品 Python 檔案的 SHA-256；它證明來源身份，並不宣稱每個檔案都做過完整形式驗證。

## 已確認的問題

P1 表示優先處理的使用者功能／契約問題；P2 表示應排入近期修復。以下將共用根因合併，避免不同子任務對同一問題重複計數。

| ID | 優先級／責任 | 觸發與後果 | 建議修復與驗收 |
|---|---|---|---|
| C1 | P1，adapter／原生邊界 | `language="en-US"`、`"zh-Hant"` 經 Standard ASR RFC 4647 門控合法通過，但原樣送入只接受 `en`、`zh` 等 key 的 runtime。Batch 失敗；streaming 報 `invalid_audio_or_context`。 | 以共用 helper 將已協商的 runtime tag 轉成 Qwen 控制；保留標準 refinement diagnostic。兩個 preset × batch／whole-input／incremental streaming 驗證，強制語言的 `detected_language` 仍為空。 |
| C2 | P1，工件與 manifest 契約 | 缺少 runtime 必需欄位的 manifest、short preset 指向 general bundle、draft 與 target 不相容，仍可能被 `artifact_status()` 回報 `ready`。使用者看到可用，載入才失敗。 | 抽出 status 與 runtime 共用的廉價 manifest／依賴相容性檢查；驗證 decoder partitions、cache/duration、short 身份、draft model/revision/target binding。可靠知道不相容就回報 `corrupt`／action；昂貴而未驗證的事項不能假稱通過。測試逐一刪改 manifest 欄位，且 status 不載入權重、不初始化裝置。 |
| C3 | P2，取得前置條件 | `_acquisition_gate()` 只以 `source.json` 存在判定來源可用；錯誤 provenance／不完整來源可能顯示 `can_acquire_now=true`，實際 pull 隨即失敗。 | 共用來源 metadata／檔案完整性驗證；在可廉價判定時回報精確 blocker 與修復 action。分別驗證來源缺失、錯誤 revision、損壞 JSON、殘缺 checkpoint、離線有／無可用來源。 |
| C4 | P2，streaming 錯誤分類 | 包住整個 producer 的 `except ValueError` 同時攔截 PCM 錯誤與原生載入／設定／執行錯誤，一律標成 `invalid_audio_or_context`。 | 將 PCM 驗證例外限定在輸入處；native fault 交給標準 `engine_error` 或精確轉譯。以相同合法音訊、不同 native 例外驗證分類；保持 artifact 和容量錯誤的專用資訊。 |
| C5 | P2，parser／adapter 診斷 | 真實 parser 先把未知語言名稱映射成 `None`，後續 `detected_language_unmapped` 無法觸發；即使 fake runtime 直接給未知名稱，streaming 也只放在 event `extra`，沒有使用標準 session diagnostic。 | 保留 raw language 至 adapter，單一已知語言才寫 BCP-47；未知／複合名稱以有界 diagnostic 揭露。從真實 parser 串到 batch 與 streaming 的測試必須覆蓋，不能只讓 fake 越過 parser。 |
| C6 | P2，config／preset 身份 | general entry point 仍接受 `profile="short-dictation"`，改選短句 bundle 與預設值，卻以 general 的 class-level 身份與 30 秒 Properties 被發現。 | 用既有 short entry point 選 preset；general schema 拒絕跨 preset 的 profile 切換。依 IC.7 整理相容政策，保留合法 artifact location 配置。這不是宣稱靜態 30 秒等同任意請求的成功保證。 |
| C7 | P2，文件與驗證可信度 | 舊能力 JSON 仍記錄 `supports_explicit_acquisition=false`、未填 duration、舊 cache 路徑；舊 audit 宣稱已完成 BCP-47 映射；vendor CLI help 仍宣稱預設使用 `artifacts/`。 | 將舊文件明確標示歷史快照，讓新的來源綁定矩陣成為調查入口。後續驗證從公開 API 經真實 parser／prompt builder 到 native 邊界；整理重複的 vendor transcription 介面與過時 help。 |
| C8 | P2，上游音訊交付／adapter 分類 | direct `AudioArray` 可帶 stereo，框架 passthrough 不 downmix；stereo、NaN、空陣列抵達 ANE 後被包成 generic `TranscriptionError`。超出 `[-1,1]` 的有限樣本則無 diagnostic 進入特徵處理。 | 先釐清標準 canonical array 是交付保證還是 caller 義務，統一各 carrier 的交付／拒絕策略。外掛可先把明確的輸入錯誤轉成 `AudioProcessingError`，不能靜默 clip/downmix。用同波形的 file／bytes／array 驗證一致性。 |

C1、C5、C6 的精確來源與重現見[契約報告](../research/standard-asr-audit-2026-09-22/contracts.md)；C2、C3 見[工件報告](../research/standard-asr-audit-2026-09-22/artifacts.md)；C4、C5 的 session 行為見[串流報告](../research/standard-asr-audit-2026-09-22/streaming.md)；C8 見[音訊／結果報告](../research/standard-asr-audit-2026-09-22/audio-results.md)。

关键原始碼位置：`plugin.py:102,150-156,253-285,287-318,454-469,542-587,624-644,725-815`；`runtime.py:222-223,250-263,331-390,575-616`；`streaming.py:113-172,251-264`。均位於 `std_qwen3asr_ane/src/std_qwen3asr_ane/`。上游規範的語言歸約義務在 `docs/content/specification/protocol.md:282`，preset 選擇在 `:808-810`，artifact status 在 `:882`。

## 能力對照：現在用了什麼，還能補什麼

下表涵蓋完整標準 capability tree 以及不在 capability tree 內的重要公共能力。兩個 preset 的 capability tree 相同，差異是身份、預設 decoding budget 與靜態 duration。完整欄位值見 snapshot，不以 supported 比例充當成熟度分數。

| Standard ASR 能力 | 現況與判斷 | 接下來的行動 |
|---|---|---|
| `batch` | 已實作單次完整音訊轉寫；標準的 batch 不表示多錄音陣列批次推理 | 補長錄音協調器；多請求 batching 是另項 throughput 工作 |
| `streaming_input`／`streaming_output` | 都已接入；增量 PCM 與完整輸入後串流輸出 | 保留兩條公共路徑並做相同語言／錯誤修復 |
| batch／streaming language override | 已接，細化語言出錯（C1） | 修原生控制映射 |
| 自動語言偵測與 Properties 語言列表 | 30 個官方控制，forced 不冒充 detection；異常 metadata 丟失（C5） | 補 raw metadata 與診斷鏈 |
| batch／streaming candidate languages | 未支援原生硬限制；標準明確以 diagnostic 忽略是目前協定的特殊規則 | 若要補，需驗證 LID／受限解碼的選擇規則；不能把列表拼進 prompt 就宣稱支援 |
| batch／streaming prompt 與 constraints | 已映射至原生 context；標準約略 128-token gate，加原生精確 cache 預檢 | 保留分層檢查；未證明任意 128 個約略 token 均可容納 |
| batch／streaming phrase hints | 原生不支援；已支援顯式 `degrade_to_prompt` 與標準 diagnostic | 合理利用了框架；真正詞彙加權需另外實作與品質驗證 |
| `streaming.guidance.mutable_mid_stream` | false；參數在建立 session 時固定 | 上游目前沒有公共 guidance 更新方法；先定義語意及更新 API |
| batch／streaming word timestamps：word／segment／char | 目前 false；沒有時間對齊來源 | 可接官方獨立 ForcedAligner，詳見下一節 |
| `streaming.timestamps` | `none`；沒有偽造 chunk-boundary speech timestamps | 有真實 alignment 後再用 `post_align`；processing cursor 問題見 upstream feedback |
| batch／streaming diarization 及 always-on | 全部 false；没有 speaker model | 若產品需要，加入獨立 diarization／alignment pipeline；Qwen ASR 權重本身不提供此承諾 |
| `streaming.emits_partials` | true；同一 utterance 的全文替換 partial | 已正確實作；不能把 partial 次數當品質指標 |
| `streaming.word_stability` | false；partial `stable_until=0` | rollback token prefix 不等於不可變 transcript prefix；需新的穩定文字演算法與驗證 |
| `streaming.finality_level` | `closed`；尾音訊 flush 後定稿 | 已接；長錄音需要跨片段的定稿策略 |
| `streaming.re_segments` | false；沒有 supersede | 一般串接多個 closed segments 不必開這個 flag；只有真的替換 segment 才實作其契約 |
| `streaming.reconnect` | `unsupported`；本地 bounded session 不自動恢復 | 不能以重新建立 KV state 冒充 seamless reconnect；本地 rollover 是不同問題 |
| `self_resamples` | false，正確由標準重採樣 | 不應為提高支援比例而設 true |
| AudioPath／Bytes／Array／Base64 | 檔案型 carrier 經標準 decode／downmix／resampling；正常 mono array 可直送 | 已接；direct array 的 canonical 邊界仍有 C8，不能籠統宣稱所有形狀都會正規化 |
| AudioUrl／StorageUri | 對 array 引擎沒有轉換路徑，明確拒絕 | 目前上游禁止替這類引擎 fetch；不屬於外掛漏接 |
| 增量音訊格式 | mono 16 kHz `pcm_s16le`／`pcm_f32le`；處理任意 byte split 與末尾半樣本 | 現有上游不做增量 resampling／downmix，來源需先轉換 |
| typed init config／env／ProviderParams | 已用標準 config，provider `max_new_tokens`、exact-type swap safety 正確 | 修 preset 選擇（C6）；釐清不適用的 config 欄位 |
| discovery／static metadata／effective capabilities | 兩個具體 class entry points，可免模型載入查 schema；目前 effective=declared | 新增選配 aligner 時才按配置收窄；不能用 instance properties 改寫 class 身份 |
| status／pull／refresh／cache／offline | 已接標準模板、cache root、原子發布及鎖；pinned immutable refresh no-op 正確 | 修 C2／C3；改善取得前置條件與可觀察進度 |
| prepare／close／資源生命週期 | 惰性載入、鎖定序列化、target-only prepare、close 失敗保留 ownership | 補 cooperative native cancellation；上游 server 的 ownership 另見 feedback |
| async batch／SyncSession／feed／manual input | 已繼承標準橋接與 session 管理 | 不需要再建一套 vendor async API |
| input/output backpressure／deadlines／terminal reduction | 已採用；deadline 是牆鐘時間，不是音訊容量 | 真正停止 native 工作與保存 terminal failure 狀態仍有缺口；whole-input streaming 額外受全局 Properties duration 限制 |
| TranscriptionResult／Diagnostic／錯誤邊界 | 標準 schema 已使用；缺少的 confidence／speaker 不造數值 | 修 C4／C5；可觀察性與 optional result fields 詳見專項報告 |
| SRT／VTT／JSON | batch 的無 segments＋已知 duration 可渲染一個全文 cue；streaming 產生 untimed segment，預設渲染會拒絕 | streaming 還丟失已知 duration；0.5 秒 probe 在顯式 collapse 時落到 3 秒 fallback。需補時長傳遞；alignment 才能提供真正語音片段時間 |
| CLI／doctor／HTTP／WebSocket | 經標準工具鏈可達，無需 vendor server；本專案尚無 WS transport 測試 | 補實際 serve 安裝方法／隔離測試；上游 reference server 生命週期、artifact readiness 與 typed wire params 仍有限制 |
| compliance | 類宣告、門控、事件／結果形狀有測試 | 加入 feature adoption matrix、跨層 probe 與 opt-in real-model 層，不能只看 exit 0 |

## 原生引擎應補的功能

| 次序 | 功能 | 建議設計與驗收邊界 |
|---|---|---|
| 1 | cooperative native cancellation | 目前取消輸出很快，但 worker 可跑完整次 autoregressive decode、繼續佔鎖。傳入 cancellation token，在兩次安全 native 呼叫之間檢查；不釋放仍被 Core ML 使用的 buffers。驗證取消後不再開始下一步 decode、下一請求可進入、session 狀態無污染。 |
| 2 | 長錄音 batch 與 streaming 接續 | 官方 toolkit 已有低能量處切片再轉寫的參考路徑，本引擎尚未移植。先以現有可驗證 bundle 做 segmentation、必要重疊與文字去重；跨邊界 WER/CER、重複／漏詞、靜音與 code-switch 必須實測。大 cache 可作另項 profile／效能選項，不能代替接續演算法。 |
| 3 | optional ForcedAligner | 官方是另一個 0.6B 模型，支持語言範圍小於 ASR 的 30 種；需獨立 acquisition、runtime、按請求的 artifact closure、capability narrowing 與 Word／Segment 映射。先驗證支援語言的 batch alignment，再考慮 streaming final 的 post-align。ANE 移植尚未證實，不宣稱現有 ASR 圖直接能做。 |
| 4 | 可觀察性與部署完整度 | 保留正確的 request timing／容量與診斷；改善來源與 worker 前置條件、phase/count 進度、server 安裝說明。只輸出有真實量測與明確語意的欄位。 |
| 5 | 獨立音訊請求 batching | Qwen wrapper 有多輸入介面，本引擎鎖內一次一段；現有 frontend B4 是同一錄音內的小塊 batching。需要新 scheduler／graph 支援並量測 throughput，不能把標準 `batch=true` 當成已完成。 |

硬語言候選限制、原生 phrase boosting、diarization、可變 guidance、word stability 並非簡單 adapter glue。它們需要演算法、模型或上游 API 的實作與驗證；現況宣告 false 是誠實的，但應有明確理由與再評估條件。詳見[原生功能報告](../research/standard-asr-audit-2026-09-22/native-features.md)。

## 建議拆成可驗收的後續變更

1. `fix: normalize negotiated language controls`：C1；兩個 preset、兩種模式與真實 prompt builder 共同驗證。
2. `fix: validate artifact compatibility before reporting readiness`：C2／C3；共用 validator、真實形狀的 metadata fixtures、offline 與 draft closure。
3. `fix: preserve inference diagnostics across result boundaries`：C4／C5；parser 到 batch/session 的完整鏈，保留 Standard ASR exception ownership；另以獨立 audio normalization／validation 變更處理 C8。
4. `fix: keep preset selection in model entry points`：C6；schema、CLI、server discovery 一致，說明舊 profile 用法如何改選既有 preset。
5. `feat: cancel native decoding between predictions`：資源取消與 buffers ownership 驗證。
6. `feat: transcribe long recordings with validated segmentation`：獨立品質工作，附預先定義的 WER/CER 與邊界完整性門檻。
7. `feat: add optional forced alignment`：先證實模型路徑，再接 artifact、capability、results 與 renderer。
8. `docs: publish verified feature coverage and server setup`：文件、schema snapshot、標準流程的安裝／wire 測試一起維護。

以上是工作切分建議，並未建立分支、commit、PR 或聲稱功能已修好。Standard ASR 自身的完整案例、現有協定限制與建議放在 [standard-asr-feedback-2026-09-22.md](../standard-asr-feedback-2026-09-22.md)。

其他不應遺漏的近期事項：普通安裝缺少 server extra 的明確安裝命令與隔離 CI；GPU draft 的 MLX prerequisite 應在昂貴 target warm-up 前發現；acquisition 可使用真實 bytes/files counts 與更準確的 phase；unsupported host 的 OS/arch/version 應有早期診斷。這些並不全是已重現的協定缺陷：離線全新 worker cache、非 Apple Silicon 主機與 real-model server memory 行為仍只有靜態分析／設計評估，詳見各專項的未驗證邊界。

## 專項報告與重現入口

| 面向 | 報告 | model-free probe |
|---|---|---|
| capability／config／language／guidance | [contracts.md](../research/standard-asr-audit-2026-09-22/contracts.md) | [probe_contracts.py](../research/standard-asr-audit-2026-09-22/probe_contracts.py) |
| session／events／cancel／deadlines | [streaming.md](../research/standard-asr-audit-2026-09-22/streaming.md) | [probe_streaming.py](../research/standard-asr-audit-2026-09-22/probe_streaming.py) |
| artifacts／conversion worker／readiness | [artifacts.md](../research/standard-asr-audit-2026-09-22/artifacts.md) | [probe_artifacts.py](../research/standard-asr-audit-2026-09-22/probe_artifacts.py) |
| native／official Qwen 能力 | [native-features.md](../research/standard-asr-audit-2026-09-22/native-features.md) | 來源鏈與既有 focused tests，未跑新模型 |
| discovery／install／CLI／HTTP／WS | [ecosystem.md](../research/standard-asr-audit-2026-09-22/ecosystem.md) | [probe_ecosystem.py](../research/standard-asr-audit-2026-09-22/probe_ecosystem.py) |
| audio／result／diagnostic／error | [audio-results.md](../research/standard-asr-audit-2026-09-22/audio-results.md) | [probe_audio_results.py](../research/standard-asr-audit-2026-09-22/probe_audio_results.py) |

Probe 是對本次缺陷的重現與已支援路徑的觀察；它們成功執行不表示被重現的缺陷已修好。正式修復時應將相應驗收條件放入產品測試，並把預期由「重現目前錯誤」改成「符合契約」。
