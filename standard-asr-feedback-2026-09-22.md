# Standard ASR 回饋：原生 ANE 外掛的完整能力調查

> 以下編號問題保留修復前的證據。2026-10-02 的處理狀態見下表；上游實作已提交於 `5f6eef25e35e5e66e9010474e6dee531021e61f1`，見 [Standard ASR PR #106](https://github.com/standard-voice/standard_asr/pull/106)。目前引擎的支援與限制請見[能力對照](docs/standard-asr-capability-coverage.md)，驗收見 [release ledger](docs/release-readiness-2026-09-22.md)。

## 2026-10-02 處理狀態

| 原問題 | 已完成的對應處理 | 原提案的邊界 |
|---|---|---|
| 1：server lifecycle／readiness | 標準同步 close 契約、依固定 model/config 的 singleflight engine pool、REST/WS lease、取消與 shutdown cleanup、read-only readiness endpoint。 | 每個已註冊 model 至多一個固定配置；不是可由請求任意新增配置的無界快取。沒有以此宣稱並行速度提升。 |
| 2：session completion | async／sync 都提供 status、嚴格 final result、明確 partial_result；失敗不再冒充成功空結果。 | Partial snapshot 是使用者明確選擇保留的資料，不是成功證明。 |
| 3：cursor／alignment／duration | 獨立 audio_progress capability、單調 cursor、measured input duration。 | Speech timing 仍需要真實 alignment；無此能力時不填造語音 spans。 |
| 4：有效容量 | Mode-aware non-widening duration hook；外掛 bounded windows 與 total recording guards；原生精確 token/cache 檢查。 | 原提案中的跨引擎、逐請求、逐 session 的一般化 remaining-capacity 查詢仍是未標準化設計，沒有宣稱本次新增了這個公共 API。 |
| 5：worker／SDK 前置條件 | 外掛共用 source validation、確切 dependency receipt/Python ABI fingerprint、offline feasibility；先檢查選配 SDK。Conversion／alignment worker 不繼承父 site-packages。Doctor 明示只分析 NumPy。 | 上游尚無一般化的 known/unknown SDK prerequisites 資料模型；外掛用明確 artifact report、deployment metadata 與 ConfigError 表達可知部分。 |
| 6：wire ProviderParams | CLI／REST／WS 都依選定引擎的 concrete type 驗證；schema、錯誤與 swap safety 同步。 | Engine-specific params 保持非可攜；沒有任意 dict 繞過驗證。 |
| 7：跨層驗證 | 已增加公開 API regression、真模型證據、乾淨 wheel 矩陣、來源綁定能力快照、完整結果與生命週期驗證。 | 沒有把目前 compliance 工具不存在的通用 runtime/acquisition flags 宣稱為已實作；這項通用作者工具提案保留給上游後續設計。 |
| 8：無效 candidate default | 外掛已實作真正 candidate restriction，因此此配置對 full-logits target 有效；compact target 的 narrowing 與標準 diagnostic 已揭露。 | 通用 mixin 拆分屬上游設計建議，不再是此外掛的功能阻塞。 |
| 9：canonical arrays | 共用 mono／finite／range 政策；ARRAY、encoded 與 resampling 邊界一致，輸入錯誤在 native hook 前歸類。 | 有損 normalization 會保留診斷；不把它當作聲學品質保證。 |

本輪另修正了上游結果組合缺陷：舊 reducer 會 strip／space-join，破壞 CJK 與精確分段空白，且丟棄 Word channel 和 Segment extra。新契約提供 `text_separator`、唯一 composition helper、speaker segment／word／extra 保存，以及 separator-aware frozen-prefix／supersede 檢查。真實 speaker split 可以沿標準 event → result → wire 傳遞，無需把標準資訊藏進 vendor payload。

最終 review 另重現 ProviderParams 自訂 `PydanticCustomError` 被重建成內部 KeyError。已保留 code、原樣 rendered message 與 `provider_params` 路徑，避免二次展開任意 context；原始 ValidationError 仍在 cause。CLI 返回 usage error，REST 返回經清理的 typed 422。關於 shutdown timeout 的建議，保留「先等待 active work，再關閉 engine」的 ownership 契約，並明確文件化沒有隱含 drain deadline；硬性 process deadline 由部署層處理，不以略過未完成 native work 假裝成功清理。

這些一般化上游設計建議與引擎已宣告功能的缺陷分開記錄；能力 false 不會為了完成矩陣而改成不實 true。舊紀錄中的「沒有修改上游來源」描述的是 2026-09-22 調查當時，並非現在狀態。

日期：2026-09-22。Standard ASR：`1b2cf3fa5860c075e5160eb60b26b708a7c8bfea`；外掛：`std_qwen3asr_ane@884c22e`。最新 Standard ASR 已直接 clone 到 `references/standard-asr-audit-2026-09-22`，與外掛 pin `8b124e8c` 的 Python／規範內容相同；以下不是套用舊版文件後推測的新問題。

完整外掛調查見 [總報告](docs/standard-asr-comprehensive-audit-2026-09-22.md)。本文件只記錄上游問題、刻意但影響產品的限制，以及改善外掛作者工作流程的建議。沒有向 GitHub 發送 issue 或修改上游來源。

## 1. Reference server 缺少適合重量級本地引擎的生命週期管理

**分類：已確認的上游 runtime／工具鏈限制，優先處理。**

本外掛的 engine constructor 是純配置；第一次推理才建立 Core ML 模型，單一 engine 以鎖序列化其 stateful native inference，`close()` 等待／釋放模型並保留失敗時的 ownership。Reference server 的 REST 與 WS 路徑卻為每個請求／連線呼叫 `registry.create(...)`，沒有共用 engine pool，也沒有與應用 lifespan 對應的顯式 engine close。

證據：Standard ASR `src/standard_asr/toolchain/server.py:815-817,1573-1575,1645-1648`；外掛 `std_qwen3asr_ane/src/std_qwen3asr_ane/plugin.py:233-238,454-472,519-557`。[ecosystem probe](research/standard-asr-audit-2026-09-22/probe_ecosystem.py) 用真實 FastAPI 公開路徑與 fake engine 觀察兩次 REST：兩次建構、app shutdown 後沒有顯式 `close()`。

**影響。** Standard server 可以正確回傳一次辨識，不代表它已提供合適的持續服務行為。原生模型不能跨請求保留 warm state；不同實例的鎖也無法限制同一裝置的總載入量。這次沒有量測 Core ML 的實際殘留記憶體，不能把「未顯式 close」直接等同確定的永久 memory leak。

**建議。** 明確定義 server 如何擁有引擎與 session 資源：依 model key 和有效 init config 建立可重用實例；限制同時建構數，避免首次請求競態重複載入；區分可共用 engine 與 session-private state；在 shutdown 等待 active work 並執行明確釋放。先決定標準 engine cleanup 的公共協定，再讓 server 使用它，避免靠任意 `getattr(close)` 或 GC 猜測。快取也必須有容量／逐出策略，不應無上限保留每種配置。

**驗收。** 重複同 model/config 的 REST 共用指定數量的 engine；同時首次請求不重複初始化；不同 config 不誤共用；WS session state 隔離；shutdown／例外／取消均完成可觀察 cleanup。另以真實本地引擎量測 warm latency、峰值記憶體與並行 admission；不能用 fake 建構次數宣稱效能改善已達成。

**相關缺口：liveness 與 readiness。** `/v1/health` 在 `server.py:491-504` 固定回報 `ok`，它可合理作 liveness，卻不能作 model readiness；目前也沒有 artifact-status HTTP route。建議保留 liveness，另外提供使用 `artifact_status()` 的 operator readiness 摘要，按 batch／streaming context 區分所需 draft。status 可能讀慢速檔案系統，應放在可管理的 worker／期限內，不阻塞 event loop。服務未取得模型時應為 live-but-unready；此檢查不做 acquisition、不載入模型，也不承諾完整 native health。

## 2. Session 的結果快照無法表達本次操作如何終止

**分類：已確認的公共 API 限制，現況不是外掛違約。**

Standard ASR 正確把 artifact 問題與一般 native 問題轉成不同 terminal error events。但是 `session.result()` 主要歸約已接受的內容，`session.diagnostics()` 則保存 startup／lifecycle／engine diagnostics；terminal error 不成為其中的穩定 failure state。一次失敗 session 可能留下空結果，或只留下先前已完成段落。`TranscriptionResult` 沒有 `error` 欄位；外掛不能自行塞入一個不存在的標準欄位。

證據：`src/standard_asr/runtime/streaming.py:986-1168,2976-3003,3274-3363`；`contract/results.py` 的結果 schema。[串流 probe](research/standard-asr-audit-2026-09-22/probe_streaming.py) 的 `native_value_error` 與 cancellation 記錄重現 event terminal 與結果快照的差別。

**影響。** 只拿最終 result 做下游處理的應用，需要另外保存事件來分辨「成功但無語音」與「出錯、取消、期限到達」。目前文件範例迭代事件後取得 result，不會自動將 result 升格成成功保證。

**建議。** 設計 session-owned completion snapshot（例如終止種類及最後 terminal event 的查詢），或在文件明確提供一個完整消費範例，要求检查 terminal event 後才將結果視為成功。是否擴充 TranscriptionResult 應另作協定設計，不能讓每個外掛發明不相容的私有 `error`。

**驗收。** `done`、cancel、deadline、native fault、artifact fault、先完成部分片段後失敗，都能在 async 與 SyncSession 以相同方法查明；保留部分內容不代表隱藏失敗。

## 3. Processing cursor 與語音 alignment 被同一 capability 綁住

**分類：已確認的表達能力／文件語意問題。**

規範的 `audio_processed_until` 是已處理音訊的單調游標，與 segment／word 語音邊界不同（`docs/content/specification/protocol.md:1107-1118`）。本地 cumulative engine 每次推理完成後能精確知道已處理多少 samples，但没有 alignment 資訊。

目前 `StreamTimestampsCap.mode` 僅有 `native_frame_aligned`、`post_align`、`none`（`contract/capabilities.py:487-503`），compliance 將任何 `audio_processed_until` 視為必須有 timestamp capability（`compliance.py:2423-2491`）。本外掛因而只在 `extra.audio_prefix_seconds` 提供其真實輸入前綴長度；不能為了讓游標可用，謊稱有 native alignment。

**建議。** 將處理進度與語音時間對齊的來源分開建模，或先明確說明現有 mode 在 processing-only 情境的合法含義。不要要求無 alignment 的 engine 使用 `post_align`／`native_frame_aligned`，也不要因為 segment start/end 不可得，就抹掉可量測的處理進度。

**驗收。** 無 word／segment timestamps 的引擎可以誠實聲明與輸出 processing cursor；capability、event guard、compliance 與 wire client 判斷一致；不能將 processing cursor 解讀成已凍結文字或語音邊界。

**結果時長也需有獨立傳遞方式。** 成功 session 已知 0.5 秒輸入，reducer 仍返回 `duration=None`。同一 probe 的 batch 可渲染 0.5 秒全文 SRT cue，streaming 因 untimed segment 預設拒絕；呼叫者明示 `collapse` 後又落入 3 秒 fallback。這個 fallback 是使用者選擇的合成呈現，不是 alignment 缺陷，但已知 input duration 沒有傳遞十分可惜。應讓原生 session 交付已量測的輸入時長，與 processing cursor、word/segment speech times 各自區分。證據見 [audio-results-probe.json](research/standard-asr-audit-2026-09-22/audio-results-probe.json)。

## 4. Artifact／設定／請求共同決定的有效容量仍無公共查詢

**分類：先前回饋仍成立；這次補上目前兩個 preset 與已存在機制的界線。**

已有 `BaseProperties.max_audio_duration`，本外掛也已宣告 general 30 秒、short 12 秒。已有 `StreamDeadlines.max_session_seconds`，那是應用選擇的牆鐘期限。已有 `effective_capabilities`，只可收窄宣告。這些都不能完整表達「某個已配置 bundle、特定 prompt／生成預算、目前 session 已累積前綴，還能接多少音訊」。

本外掛的有效串流上限取 `min(config.stream_max_audio_seconds, runtime.max_audio_seconds)`；原生還檢查 `len(prompt) + max_new_tokens - 1 <= cache_length`。同一段音訊的可接受性可能隨 request context 改變。設定預設的 180 秒僅是應用上限，不代表 shipped 30 秒 bundle 變成 180 秒能力。

證據：外掛 `streaming.py:113-145,198-264`、`runtime.py:575-616`、`plugin.py:139,176-188,607-621` 與本次[串流報告](research/standard-asr-audit-2026-09-22/streaming.md)。本地歷史檔 `standard-asr-feedback-2026-09-13.md` 第 8 節也有較早案例，但該檔由 `.git/info/exclude` 排除，不是本報告必需的可攜證據。

**建議。** 在不載入模型／不跑推理的前提下定義可用的容量查詢與預檢層次：靜態 preset、artifact metadata、request budget、session remaining capacity。區分硬上限、估計、未知；區分單次 input、model window、累積 audio、wall time；不要把未知當無限。此處不預設欄位名稱或要求標準自動切音訊。

**驗收。** App 能在正確生命週期階段得到已知限制與未知部分；Python／CLI／wire 語意一致；快速 replay 和慢速錄音不能混淆；容量不足時說明可否開新段、已有文字是否 final，而不是只給 generic error。

**兩種 streaming input 的差異已重現。** `start_transcription(audio=31s)` 會先經 `_prepare_audio()` 套用 general 的全局 30 秒 Properties，即使 fake runtime 與 session config 都允許 180 秒；incremental PCM 則只到 session／bundle 判斷。這不是預設 30 秒 bundle 的 regression，但對自建長 context bundle／未來 long-form profile 是實際 mode 差異。外掛 metadata 的 `minimum_of_configuration_and_loaded_bundle_duration` 對 whole-input 少寫了全局上限。請將 mode-specific bound 一起納入設計，不要單純調大 Properties 掩蓋問題。

## 5. Acquisition 工具環境與推理 SDK 的前置條件缺少清楚位置

**分類：部分外掛可改善，部分需要上游決定公開語意。**

本外掛的普通推理環境不安裝 Torch；`pull` 透過含固定依賴的 worker 轉換模型。即使來源 checkpoint 已在本地，離線且 worker cache 為空時，`uv run --offline` 仍無法建立工具環境。`can_acquire_now` 不能只由 model source 是否存在判斷。現況沒有明確的 acquisition feasibility「未知」值；runtime packages 又不等同模型工件。

另一例是選配 GPU draft：模型檔案可全部存在，但沒有 MLX 套件。外掛目前在 ANE target 載入後才回報有修復提示的 ConfigError。這是可提前修正的外掛部署體驗問題，不能直接說 artifact 文件存在就必須替 SDK 健康狀態背書。

證據：外掛 `plugin.py:253-285,360-373,493-557`、`acquisition.py:85-105`、`conversion_worker.py:1-9`；`pyproject.toml:9-22`。完整情境在[工件報告 A4／A5](research/standard-asr-audit-2026-09-22/artifacts.md)。Standard ASR 的 doctor 目前明確只精確分析 numpy 衝突，不能把 doctor exit 0 當成 conversion／MLX／serve 均可運作的證據。

**建議。** 釐清 acquisition prerequisites、runtime dependencies、persistent artifacts 的責任與查詢方式。短期外掛可在昂貴載入前檢查選配依賴、保留 ConfigError；對離線 worker 只在有可用環境證據時宣稱可立即取得。上游可設計能表達 known／unknown prerequisite 的機制，不應迫使外掛造一個「SDK 權重工件」。

**驗收。** 本地 source + 無 worker cache + offline、本地 source + 已準備工具環境、缺少 optional runtime extra，分別得到準確且一致的前置資訊；status 不為了檢查而安裝套件、聯網或建立檔案。

## 6. Typed ProviderParams 在 wire／CLI 上是 discover-only

**分類：目前刻意的 D5 協定限制，不是 adapter 參數漏接。**

Python API 能用 `Qwen3ASRParams(max_new_tokens=...)` 逐次調整輸出預算。標準 `/v1/params-schema/{model}` 也可展示其 schema。但 `WireRuntimeParams` 刻意不接受 `provider_params`，CLI portable options 與 REST／WS 因而不能使用它；CLI 可以改 init default，這與 remote per-request 能力不同。

證據：`contract/params.py` 的 `WireRuntimeParams` 定義與 D5 drift assertion；外掛 `plugin.py:166-170,589-593`；[ecosystem probe](research/standard-asr-audit-2026-09-22/probe_ecosystem.py)。

**建議。** 若完整遠端引擎參數是需求，先設計可由選定 model 的 concrete ProviderParams type 驗證的 wire extension，明確處理身份、schema version、swap safety、錯誤 redaction 與可攜性。不能把 arbitrary JSON dict 直接塞進 RuntimeParams，也不要要求每個外掛另建 vendor HTTP API。若維持 discover-only，UI 應標示不可在該 transport 提交。

**驗收。** 發布 schema 的 UI 不再呈現無法提交的假控制；若新增 typed wire 支援，正確 engine params 接受、錯誤／未知／跨 engine params 一律 fail-loud，所有 transport 使用同一驗證規則。

## 7. Compliance 與作者範例缺少 feature adoption／跨層驗證

**分類：工具／工作流程改善；不要求合規檢查替模型功能背書。**

這次 default compliance 全通過，既有 496 項測試也通過，仍發現：合法語言 refinement 只在真實 prompt builder 被拒絕；未知語言只在真實 parser 被丟棄；artifact fixture 自述不是可用模型、缺少 runtime 必需 metadata，卻用來斷言 ready。它們共同反映「模擬層的輸入比真正下一層宽鬆」。

上游已明確說 default compliance 不做模型完整性證明；AR.10 也記錄 opt-in runtime／acquisition flags 尚未實作（`docs/content/specification/protocol.md:898`，以該段目前文字為準）。不能把未提供的 flags 當成已可用工具，也不應讓預設檢查自動下載模型或觸發付費推理。

**建議。**

- 提供分層報告：靜態 declaration、synthetic gating、artifact metadata、錄製 event/result、使用者指定 fixture 的真實 inference，各自顯示跑了與未跑什麼。
- 由能力樹生成完整 adoption matrix；unsupported 項保留理由、原生模型是否有實作路徑、再評估條件。不要機械地因 false 發警告，否則會鼓勵不實 true。
- 作者測試範本包含 RFC 4647 refinement 到 native control、真實 parser 到 diagnostics、同一 manifest validator 到 status／runtime、strict/best-effort／wrong-engine params，以及 canceled session 與 native worker 的區別。
- 用 fixture 驅動 opt-in real-model checks；記錄音訊、模型／artifact revision、request params、事件與結果。形狀正確不等於聲學品質正確，合規通過不等於高 feature coverage。

**驗收。** 本次已知反例能在合適層被發現，報告清楚說明 fake／實體模型證據；「已讀完整指南」不能代替可執行的跨層檢查。

## 8. Config mixin 把不適用的候選語言預設帶入 schema

**分類：可由外掛自行改善的低優先級 UI 問題；上游 helper 改善建議。**

`LanguageConfigMixin` 同時提供 `default_language` 與 `default_candidate_languages`（`runtime/config.py:1982-1996`）。本外掛沒有 candidate-language 能力，卻因繼承該 mixin 向 settings UI 公開此欄位；設定後 auto 請求只得到 `candidate_languages_ignored`。這是有診斷的合法 fallback，沒有暗中 enforced candidates。

作者指南要求「有語言軸的 config 必須有 usable default_language」，**沒有要求必須繼承 mixin**。外掛可以自行宣告 default_language，因此這不是必須等上游修好才可行動的 blocker。

**建議。** 上游拆開兩個 mixin 或提供 schema applicability 指引；外掛先不展示無效的 default candidate 控制。驗收只需 schema 與已宣告能力相符，不能反過來為讓欄位成立而偽造 candidate 能力。

## 9. Direct AudioArray 的交付規則與 encoded audio 不一致

**分類：已重現的跨層行為差異；上游需釐清 canonical 的保證，外掛可先改善輸入錯誤分類。**

`AudioArray` 文件允許一維 mono 或 `(samples, channels)`；規範的 canonical audio 又寫 mono float32 `[-1,1]`。但 array-to-array 的 negotiation 走 passthrough，conversion 只轉 float32，不 downmix／限制幅度，也刻意保留 non-finite 值加 warning。相對地，encoded audio decode 會請求 mono。

證據：`audio/input.py:87-105`、`audio/negotiation.py:417-436`、`audio/conversion.py:452-479,577-636`；規範 `protocol.md:39,979`。外掛原生 `runtime.py:588-594` 需要非空、有限、一維輸入，而 `plugin.py:580-587` 把不符合條件的 ValueError 包成 generic `TranscriptionError`。

[公開 API probe](research/standard-asr-audit-2026-09-22/probe_audio_results.py) 的[保存輸出](research/standard-asr-audit-2026-09-22/audio-results-probe.json) 顯示：stereo array、NaN array、空 array 都變成 generic native fault；有限的 `[2,-2]` 則原值進入 runtime、沒有 diagnostic。普通 mono、檔案／bytes／base64 和重採樣的正常路徑能成功。此證據只到資料交付與 validation，未量測幅度改變造成的辨識品質。

**建議。** 先選定可維護的公共規則：canonical 是否為標準交付給 engine 的保證？若是，讓每個 ARRAY delivery 共用 downmix／shape／finite／range 政策並揭露有損變換。若只是 caller 輸入慣例，就讓引擎能宣告其輸入約束並在 hook 前得到一致的 caller-owned rejection。不要把 codec 解碼、raw ndarray 和 resampling 三條路各自留下不同規則。外掛可先明確拒絕非空／finite／mono 以外的輸入並轉成 `AudioProcessingError`；不能一面寫標準已做 normalization，一面等 generic inference exception。

**驗收。** 同一波形經 array／file／bytes 交付时，確認數據形狀、取值、diagnostics 與錯誤 ownership；覆蓋 mono／stereo、空、非有限、超範圍、重採樣與 strict/best-effort。每次有損處理都可見，輸入錯誤不能落成一般 engine 5xx。

## 先前回饋的狀態

本地 `standard-asr-feedback-2026-09-13.md` 是歷史紀錄。此輪不能重複聲稱「無標準 cache helper」、「沒有 explicit acquisition」、「沒有 ProviderParams」、「沒有任何 HTTP 測試」：這些在目前外掛已接入／修正。`resolve_download_root`、DownloadConfigMixin、獨立 short entry point、worker isolation 都已存在。

仍應繼續追蹤的是有效容量、取得工具環境、更多真實 end-user server 驗證，以及合規／feature adoption 的差別。新增的 runtime 發現則是語言原生映射、已知不相容 manifest 的 ready 誤判、跨層診斷資料流與 server engine ownership；它們各有自己的證據與責任，不能全部歸咎於「文件不夠」。
