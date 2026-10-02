# Standard ASR 能力採用與限制

這份文件描述 release-readiness 分支的實作；[2026-09-22 調查](standard-asr-comprehensive-audit-2026-09-22.md)保留修復前的證據。能力宣告、請求可用性與聲學品質是三件事：schema 能表達一項功能，不表示已取得其模型；測試符合契約，也不表示所有語言都有相同辨識品質。

## 標準能力

| 能力 | 實作與可用條件 |
|---|---|
| Batch、async batch | 使用 Standard ASR 的音訊準備、參數協商、診斷與結果處理；長錄音以有限 native windows 辨識。 |
| Streaming input／output | 支援增量 PCM 與完整音訊輸入後的串流輸出；使用標準 session、背壓、期限、SyncSession 與終止狀態。 |
| 語言選擇、自動偵測 | 官方 30 種控制語言；合法 BCP-47 refinement 映射到原生 base control；強制語言不冒充偵測，未知 raw language 有診斷。 |
| Candidate languages | Batch／streaming 都有真正的 language-header grammar 限制，最多 8 種；需要 full-logits target。 |
| Prompt | Batch／streaming 都傳入原生 context；標準約略 128-token 限制之外，原生仍檢查真正 token/cache 容量。 |
| Phrase hints | 最多 16 詞、每詞 128 字元的 soft token-score bias；需要 full logits，不保證詞彙出現。 |
| Word timestamps | 明確啟用並取得 CPU ForcedAligner 後，batch／streaming final 支援 word、segment、char；不是由錄音長度分配時間。 |
| Diarization | 明確啟用、安裝 extra 並取得模型後，保留 measured speaker turns，跨窗口追蹤身份；歧義與 overlap 不亂填 speaker。Diarization-only 結果有說者分段，不偷偷增加未請求的 Word channel。 |
| Partials、finality | Revisable partial；closed window 從完整音訊独立重辨識；一個 window 可以輸出多個說者分段。 |
| Audio progress、duration | 單調輸入處理游標與實際總輸入時長，和語音對齊分開；不能視為文字已固定。 |
| Exact segment composition | 分段保留原文與明確 separator；中文、空格與標點不由 reducer 猜測。Word／Segment source offsets 指向結果全文。 |
| Result、diagnostics、errors | 保留語言、optional output、raw evidence 與錯誤 ownership；成功 result、明確 partial snapshot 與 terminal failure 分開。 |
| Audio carriers | Path／bytes／base64／array 由 Standard ASR 準備成 finite mono float32 16 kHz；downmix、non-finite repair、clipping 與 resampling 使用標準診斷。 |
| Discovery、config、provider params | General／short 是獨立 entry points；typed init config、env defaults、每次請求 params 與 schema 使用標準介面；CLI／REST／WS 都驗證 concrete provider type。 |
| Artifact lifecycle | 標準 status／pull／refresh 與 cache policy；共用 source／manifest／binding validation、原子發布、進度、offline reuse、隔離 worker。推理不暗中下載。 |
| Lifecycle、server | Lazy load、明確 prepare／close、安全 prediction 邊界取消；標準 server 按固定配置共用 engine，管理 active leases、readiness 與 shutdown。 |
| Renderers、toolchain | JSON／SRT／VTT、CLI、doctor、compliance 使用 Standard ASR；沒有另一套 vendor transcription server 或 CLI。 |

`effective_capabilities` 只會收窄宣告：預設沒有 auxiliary output；alignment／diarization 隨配置啟用；compact serial target 無法取得全詞彙 scores，所以不宣稱 candidate restriction／phrase bias。依標準規則，無此能力的候選語言會以 diagnostic 告知忽略；phrase hints 依 strict／best-effort 與顯式 prompt fallback policy 處理。

## 有意保留的限制

| 項目 | 理由與使用方式 |
|---|---|
| `word_stability=false` | Token rollback 與已凍結字元不是同一保證。Partial 可以修改；依賴不可變文字的 app 應等 closed final。 |
| `re_segments=false` | 多個 closed segments 的接續不需要 supersede。引擎不事後替換已 closed 的段落。 |
| `reconnect=unsupported` | 本機 session 不提供 checkpoint/resume 協定。重建模型或開新 session 不等於無縫接續。 |
| `mutable_mid_stream=false` | 請求 guidance 在 session 建立時固定；Standard ASR 沒有此引擎可採用的標準更新操作。 |
| `self_resamples=false` | Whole-input 重採樣交給框架；增量 PCM 的來源需提供 mono 16 kHz。 |
| Diarization 非 always-on | 它是可選且有額外 CPU／模型成本的能力；不替普通辨識自動啟動。 |
| Alignment 語言範圍 | `zh/en/yue/fr/de/it/ja/ko/pt/ru/es`。Diarization 的文字映射也需要 aligner，因此有相同範圍；不支援時明確報錯。 |
| AudioUrl／StorageUri | ARRAY engine 不私自下載 URL 或解讀 storage URI；使用上游允許的輸入 carrier。 |
| Confidence／額外概率 | 沒有校準或量測依據時留空，不把 token scores 冒充校準 confidence。 |
| 音訊與 token 容量 | 30／12 秒是 native windows；可設總錄音 guard。過大的 prompt／generation budget 會拒絕，未產生 EOS 會失敗，不回傳偽裝完整的截斷全文。 |

## 引擎擴充

`transcribe_many()` 是明確標記的 Python 擴充，最多每組 16 個獨立输入；它不是標準 `batch` 一詞的另一種解讀。可用的 target-bound compact head、分離 KV lanes、各 lane 的 RoPE 與 block mask 共同實作 packed decoding。需要 score guidance、缺少或損壞選配 head、容量不足或長錄音等情形會走有說明的 serial target fallback。

每筆 outcome 包含結果或錯誤與 execution。`not_run` 表示尚未 dispatch；`unknown` 表示已 dispatch 但沒有可信的逐項 execution 回報。Packed metrics 把每筆準備成本和共享 group calls／elapsed time 分開，不把 token 數冒充 prediction 次數。

GPU draft 是另一項選配：batch 可由 GPU 提案、ANE target 驗證；streaming 與 bulk 走 target。Full-logits guidance 會明確 bypass draft。

## 可重現性

完整 schema、兩個 preset 的 declared tree，以及 full-logits／compact × auxiliary 配置的有效 query surface，保存在 [2026-10-02 快照](../research/release-readiness/capability-snapshot-2026-10-02.json)，由 [`capability_snapshot.py`](../research/release-readiness/capability_snapshot.py)產生。工具比對實際 import 的 Standard ASR 與指定 clone 的逐檔 SHA-256，拒絕版本不一致；synthetic head descriptor 只用來檢查 narrowing，沒有把它當作 ready model。

[Release ledger](release-readiness-2026-09-22.md)記錄測試、真模型與安裝證據。[Standard ASR feedback](../standard-asr-feedback-2026-09-22.md)保留上游問題的背景與處理狀態。歷史 WER/CER、速度與能源表格保持原樣；不能把舊 benchmark 套到新功能或其他硬體。
