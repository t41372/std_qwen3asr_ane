# Standard ASR 能力採用與限制

本插件以官方 main `97bfdb2`（#107）為契約。#106／#108 的 API 和設計均不採用；舊調查與驗收報告不代表目前主線相容性。

| 能力 | 插件行為 |
| --- | --- |
| Batch、async batch | 使用官方參數協商、語言解析、音訊準備和結果處理。長錄音由插件分成有限 native windows。 |
| Streaming input／output | 增量 PCM 與完整音訊後串流輸出；沿用標準 session、背壓、期限、SyncSession 和 reducer。 |
| 語言 | 30 種原生控制語言、BCP-47 refinement 與自動偵測；強制語言不冒充偵測。 |
| Candidate languages | 最多 8 種，限制原生 language-header grammar；需要 full logits。 |
| Prompt／phrase hints | Prompt 經標準 gate 和實際 token 容量檢查；最多 16 個 phrase hints，提供 soft token-score bias。重疊詞組不取消彼此的有效後綴。 |
| Word timestamps | 明確啟用並取得 CPU ForcedAligner 後，支援 word、segment、char 實測對齊。 |
| Diarization | 明確啟用並取得模型；保留實測 speaker segments、跨窗口 identity 和未確定歸屬。Diarization-only 不增加未請求的 word channel。 |
| Partials／finality | Partial 的 `stable_text` 為空；完整窗口獨立重辨識後產生 closed finals。 |
| Streaming 結果 | `result()` 是 finalized snapshot。以 terminal `done`／`error` 判斷完成。全文遵循標準的 trim＋空格拼接，包含 CJK 分段邊界。 |
| 時間與原始證據 | 有 timestamp capability 才送標準 audio cursor。輸入窗口位置保留在 event.extra；done.extra 有插件自有 input duration。標準 snapshot 不填 duration 或頂層 words；words 在各 segment 中。 |
| Source offsets | Batch 參照完整 result.text；streaming words 參照其 event／segment.text，並標記 `source_coordinate_space="segment_text"`。 |
| Audio carriers | Standard ASR 處理 path、bytes、base64、array 的協商／解碼／重採樣。插件將多聲道 array 平均為 mono 並提供 diagnostic；保留 finite amplitude，拒絕 non-finite／空音訊。 |
| Discovery／configuration | General 和 short 為獨立 entry points；使用標準 typed init config、環境預設、schema 和 Python provider params。 |
| Artifact lifecycle | 標準 status／pull／refresh 和 cache policy；插件提供來源／binding validation、原子發布、隔離 worker 和離線重用。推理不下載模型。 |
| Resource lifetime | 插件提供 lazy load、prepare／close 和安全 prediction 邊界取消。Core ML 資源有 native buffer ownership 與 finalizer。 |
| CLI／server | 使用官方 CLI、REST／WS、renderers、doctor 和 compliance。Wire options 是 portable params，不接受 typed Python provider params。Server 每次請求建立 engine，以環境變數設定預設。 |

`effective_capabilities` 隨設定收窄：未啟用 auxiliary 時不宣稱 alignment／diarization；compact target 不宣稱需要 full logits 的候選限制／phrase bias。

保留的原生限制：

- `partial_stability=false`，partial 可以修改；需要不可变文字的應用等 closed final。
- 不宣稱 reconnect、re_segments 或 mutable mid-stream guidance。
- Whole-input 重採樣使用框架；增量輸入須為 mono 16 kHz PCM。
- Alignment 與 diarized text attribution 支援 `zh/en/yue/fr/de/it/ja/ko/pt/ru/es`。
- 不產生未校準的 confidence，不用輸入窗口長度捏造 speech timestamps。
- 原生窗口為 30／12 秒；總錄音限制可設定。超出 prompt／generation 容量或未產生 EOS 時報錯，不冒充完整輸出。

`transcribe_many()` 是插件 Python 擴充。每组最多 16 筆；有限 workers 讓每筆輸入走官方完整 pipeline，單一 coordinator 執行 packed native decoding 與 auxiliary 處理。無適用 head、長錄音、guidance 或容量不合適時提供明確 serial fallback。它不使用 GPU draft；GPU draft 是普通 batch 的另一選配。

[本輪驗收紀錄](release-readiness-2026-10-04.md)與[整合修正紀錄](../standard-asr-feedback-2026-10-04.md)說明實際驗證範圍。歷史速度、能源及 WER/CER 表格不代表本輪新測量。
