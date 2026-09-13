# 預先登記：優化驗收門檻（2026-09-13）

本文件在任何新候選結果出現之前寫定並提交。之後只能追加「結果」，不能修改門檻；若必須改變測量方法，須在 technical log 記錄原因、保留舊結果並重跑受影響的 baseline。

## 硬體與環境

- MacBook Pro，Apple M5 Max（18 核 CPU），64 GB，macOS 27.0 build 26A428；接 AC 電源。
- Python 3.12.13、coremltools 9.0、Standard ASR main `b63bb73b`（protocol 0.2.0）。
- 所有硬體測量序列執行，期間不做轉換、不跑其他模型；量測前先讓轉換造成的發熱恢復（≥ 60 秒閒置）。

## 參照與 baseline

- 品質參照：官方 `Qwen3ASRModel` CPU FP32（既有結果：selection set 的 `official.jsonl`、held-out set 的 `heldout-baseline.jsonl`）。
- ANE 起點：`artifacts/qwen3-asr-1.7b-compiled`（FP16、T16、cache 1024、30 秒）。這是本輪所有「改善」的主要比較對象。
- 速度／能耗 baseline，全部都要報告，不可只挑有利者：
  1. ANE FP16（本專案起點）。
  2. 官方 PyTorch 在本機：CPU FP32（既有）；若 MPS bf16 可執行則加入。
  3. MLX-Audio 0.5.3 BF16（既有）；若候選使用量化權重，另加 MLX 相同位元數（4-bit／8-bit）版本。
- 明確預期：M5 Max GPU 的記憶體頻寬遠高於 ANE 有效頻寬，ANE 在單句 latency 上預期**不會**勝過 MLX。受測的主張是「相對於明確指名的 baseline，在 latency／能耗／記憶體上的改善」，不是「全面勝過 MLX」。

## 語料

- Selection set（用來挑候選）：`librispeech-balanced-100` + `fleurs-zh-balanced-100`（各語 1–100 列）。
- Held-out gate（只對最終選定的一個候選跑一次）：`*-balanced-200/heldout-100.jsonl`（各語 101–200 列）。此集合曾用於 FP16 ANE 的一次評測，從未用於候選選擇。看到 held-out 結果後不得重新選擇候選；失敗就保留 FP16 預設並如實報告。
- Smoke（latency／能耗）：`qwen-official-en`（15.051 s）、`qwen-official-zh`（4.204 s）。
- 長音訊：`long-stream-180-en`（180 秒合成串接，僅作串流診斷）。

## 品質門檻（候選 − ANE FP16；配對、同 normalizer、language auto、max_new_tokens 256）

- 點估計：EN WER Δ ≤ +0.25 pp；ZH CER Δ ≤ +0.40 pp。
- 95% paired bootstrap CI 上界 ≤ +1.0 pp（各語）。
- 對官方 FP32：EN WER Δ ≤ +0.35 pp；ZH CER Δ ≤ +0.50 pp（點估計）。
- 覆蓋率 100%：無失敗、無 token 上限錯誤、無非有限值。
- 以上先在 selection set 上用於挑候選；選定的候選必須在 held-out set 也全部通過。

## Latency

- Smoke：warm process、每筆 1 次 warmup + 5 次量測，取中位數；報告 EN、ZH 與分段（encoder／prefill／generation）。
- Corpus：selection set 100+100 的 corpus RTF，協定與既有 `coreml-t16-quality` 相同（warmups 0、repeats 1），以便直接比較。
- 「有實際意義的改善」= 兩個 smoke 的 warm 中位數都比 ANE FP16 降低 ≥ 15%。

## 能耗

- 方法：`experiments/benchmark_energy.py`（SMC `PSTR` 整機估計，約 1 Hz），等工作量 = N 輪 × 2 個 smoke；每個 backend 至少 2 個 block，ABBA 交錯、各自獨立 process；前後 12 秒閒置括號；AC 供電，記錄電量與充電狀態。
- 指標：gross J／音訊秒；另報 above-idle J／音訊秒（idle 取同一場次 30 秒閒置平均）。
- 「有實際意義的改善」= gross J／音訊秒比 ANE FP16 低 ≥ 15%，且所有 A block 都優於所有 B block。
- 已知限制：不是校準的 wall-plug 能量，也不是逐裝置歸屬。`sudo powermetrics` 需要管理員密碼，本輪未取得；若之後取得，補做逐 rail 測量，不改門檻。

## 記憶體

- `/usr/bin/time -l`（max RSS 與 peak memory footprint）包住「載入 bundle + 轉錄兩個 smoke 一次」的 process；另報 bundle 磁碟大小。
- 已知限制：ANE 端／IOSurface 配置可能未被完整計入。

## ANE placement 證據

- `qwen3-asr-ane inspect`（MLComputePlan）對候選每個 graph：所有具成本估計的算子 preferred ANE；量化權重的 LUT 解壓不得落到 CPU。
- 最終候選以 `experiments/trace_ane.py`（Instruments，`DEVELOPER_DIR` 指向 Xcode）錄一次孤立 trace：每個元件都有 ANE Prediction；報告 ANE 忙碌時間占 predict 牆鐘的比例。

## Floor 分解（診斷，非門檻）

- 4 層 T1 partition（真實第 0–3 層權重）：cache {256, 1024, 4096}、不寫 KV、只有 MLP、slice_update 寫入、LUT4／LUT8。每個變體 10 次 warmup 後取 30 次中位數，與既有 probe 協定相同。

## 預設產物切換規則

只有同時滿足：品質門檻（selection 與 held-out 都通過）、smoke latency 改善 ≥ 15%、能耗不劣於 ANE FP16、ANE placement 驗證、串流與 compliance 測試通過，預設才切到新候選。否則新候選以可選建置選項交付，並如實報告結果。

## 明確延後的方向

- 草稿模型／speculative decoding（0.6B 任何裝置、SenseVoice）：在核心 ANE 逐 token 成本到達已量測的 floor 之前不重啟。若之後重啟，必須先定義「主要使用 ANE」的量化方式（各裝置搬移 bytes 或 FLOPs），不能事後定義。
