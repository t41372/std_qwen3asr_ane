# Handoff — 2026-09-13

前一輪（GPT Codex）的交接文件保留在 [research/handoff-2026-09-12.md](research/handoff-2026-09-12.md)；本文件描述接手後的狀態。數字與證據以 [research/results-2026-09-13.md](research/results-2026-09-13.md) 為準，決策過程在 [research/technical-blog.md](research/technical-blog.md) 第 23 節之後。

## 現況

- 正式 entry point `std-qwen3asr-ane/1.7b`，Standard ASR main `b63bb73b`／protocol 0.2.0，compliance CLI exit 0，300 tests 通過。
- 預設產物 `artifacts/qwen3-asr-1.7b` → `qwen3-asr-1.7b-p14-lut8-g32-compiled`：8-bit palettized（LUT8 g32）decoder 與 LM head 分成 2 個 14 層 Core ML model、FP16 audio graphs、1024-position cache、30 秒。與 FP16 起點相比：smoke warm latency −30%、磁碟 4.4→2.8 GB；整機 J／音訊秒 −27% 是 7-partition LUT8 對 FP16 的同場次量測，p14 對 7-partition 差 1–4%。held-out 品質門檻在 7-partition LUT8 上跑，p14 的 400 句輸出逐 token 相同、繼承結論。ANE 硬體忙碌 90.1% 轉錄牆鐘（trace 綁定檔在 `artifacts/telemetry/p14-lut8-g32-attribution.json`）。
- 已量測但未併入 plugin 的 ANE+GPU 路線：MLX 上的 0.6B 草稿 + ANE T16 驗證，200 句逐 token 與 serial 相同，selection 2.0–2.25×、整機能耗 −38%，但 wired 記憶體 6.4 GB、GPU 不再空閒、依賴 transformers 5（與 `convert` 群組衝突，獨立環境 `experiments/mlx_draft/`）。是否作為選配功能是產品決定，見 results 的 ANE+GPU 節。
- 已否定的路線（都有量測）：4-bit palette（速度與 8-bit 相同、品質超門檻）、linear int4 per-block（慢一倍）、動態 `slice_update`／`scatter` KV 寫入（此 macOS 27 beta 無法載入）、grouped／SDPA／fused 投影、28 層單一 partition（無法建立 execution plan）、SenseVoice 草稿。
- 相對 GPU：M5 Max 上 MLX bf16／8-bit／4-bit 與官方 MPS 的 latency 都低於 ANE；ANE 的優勢是平均功率約三分之一、process 記憶體極小、GPU 空閒。整機能耗 LUT8 與 MLX bf16 相近，高於 MLX 4-bit。這是誠實的結論，不要再嘗試用 latency 主張勝過 GPU。

## 可以繼續的方向（依預期收益排序）

1. **T64 prefill + 共用 state**（已證明 FP16 下 E2E −13～15%）：草稿模式下 prefill 已是最大單項（200 句 82 s 中 ANE 端 frontend／encoder／prefill 佔 43 s），serial 模式下也是 EN 的固定成本；要先量兩組 decoder 同時載入的系統記憶體，因為 ANE 端權重可能被具體化兩次。
2. **把 GPU 草稿做成選配功能**：需要 `[tool.uv] conflicts` 把 `gpu-draft` extra 與 `convert` 群組分開（transformers 5 對 <5）、把 `MLXDraft` 與 T16 驗證 head 搬進套件並列入 artifact 需求、只支援 batch 路徑、以假草稿測試 `TokenDecoder` 協定、再做一次 code review。量測已完成，缺的是產品決定與整合工作。
3. 更好的 4-bit（校準式或更小 group）只換記憶體，不換速度。
4. `sudo powermetrics` 逐 rail 能耗（需要管理員密碼）。

## 重現

```sh
export UV_CACHE_DIR="$PWD/.cache/uv" HF_HOME="$PWD/.cache/huggingface"
uv sync --project std_qwen3asr_ane --frozen --group convert
std_qwen3asr_ane/.venv/bin/pytest -q std_qwen3asr_ane/tests
std_qwen3asr_ane/.venv/bin/standard-asr compliance run std-qwen3asr-ane/1.7b   # 預設 bundle（p14），不需環境變數
experiments/workflows/candidate_gate.sh lut8-g32     # smoke latency + selection quality（門檻所用的 7-partition bundle）
experiments/workflows/final_gate.sh lut8-g32         # held-out、能耗、記憶體、trace、串流、compliance
experiments/workflows/p14_and_draft_evidence.sh      # p14 證據 + GPU 草稿實驗（需 experiments/mlx_draft/ 環境）
experiments/workflows/closing_checks.sh              # 草稿 trace、CLI 重建 hash 比對、fresh clone 安裝
```

其他 bundle 用 `STANDARD_ASR_STD_QWEN3ASR_ANE__MODEL_DIR=<bundle>` 指定。

硬體量測必須序列、期間不做轉換。`artifacts/` 不在版本控制內；bundle 之間以 hash 證明相同的 weight.bin 為 hard link，不要原地修改任何 weight.bin。
