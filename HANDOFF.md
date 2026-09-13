# Handoff — 2026-09-13

前一輪（GPT Codex）的交接文件保留在 [research/handoff-2026-09-12.md](research/handoff-2026-09-12.md)；本文件描述接手後的狀態。數字與證據以 [research/results-2026-09-13.md](research/results-2026-09-13.md) 為準，決策過程在 [research/technical-blog.md](research/technical-blog.md) 第 23 節之後。

## 現況

- 正式 entry point `std-qwen3asr-ane/1.7b`，Standard ASR main `b63bb73b`／protocol 0.2.0，compliance CLI exit 0，300 tests 通過。
- 預設產物：8-bit palettized（LUT8 g32）decoder 與 LM head、FP16 audio graphs、1024-position cache、30 秒。與 FP16 起點相比：warm latency −27%、整機 J／音訊秒 −27%、磁碟 4.4→2.8 GB；held-out 品質在預先登記門檻內（兩語都略優）。ANE 硬體忙碌約 86% 轉錄牆鐘。
- 已否定的路線（都有量測）：4-bit palette（速度與 8-bit 相同、品質超門檻）、linear int4 per-block（慢一倍）、動態 `slice_update`／`scatter` KV 寫入（此 macOS 27 beta 無法載入）、grouped／SDPA／fused 投影、28 層單一 partition（無法建立 execution plan）、SenseVoice 草稿。
- 相對 GPU：M5 Max 上 MLX bf16／8-bit／4-bit 與官方 MPS 的 latency 都低於 ANE；ANE 的優勢是平均功率約三分之一、process 記憶體極小、GPU 空閒。整機能耗 LUT8 與 MLX bf16 相近，高於 MLX 4-bit。這是誠實的結論，不要再嘗試用 latency 主張勝過 GPU。

## 可以繼續的方向（依預期收益排序）

1. **T64 prefill + 共用 state**（已證明 FP16 下 E2E −13～15%，LUT8 下比例較小）：要先量兩組 decoder 同時載入的系統記憶體，因為 ANE 端權重可能被具體化兩次。
2. **GPU 上的 0.6B 草稿 + ANE 1.7B 驗證**（speculative greedy，輸出與 serial 完全相同）：估計 E2E 可再快 2×，但會加入 MLX 依賴與 GPU 功耗，且必須先定義「主要使用 ANE」的量化方式（FLOPs 或 bytes）。這是使用者明確允許測試的 ANE+GPU 假說，尚未做。
3. 更好的 4-bit（校準式或更小 group）只換記憶體，不換速度。
4. `sudo powermetrics` 逐 rail 能耗（需要管理員密碼）。

## 重現

```sh
export UV_CACHE_DIR="$PWD/.cache/uv" HF_HOME="$PWD/.cache/huggingface"
uv sync --project std_qwen3asr_ane --frozen --group convert
std_qwen3asr_ane/.venv/bin/pytest -q std_qwen3asr_ane/tests
STANDARD_ASR_STD_QWEN3ASR_ANE__MODEL_DIR=artifacts/qwen3-asr-1.7b-lut8-g32-compiled \
  std_qwen3asr_ane/.venv/bin/standard-asr compliance run std-qwen3asr-ane/1.7b
experiments/workflows/candidate_gate.sh lut8-g32     # smoke latency + selection quality
experiments/workflows/final_gate.sh lut8-g32         # held-out、能耗、記憶體、trace、串流、compliance
```

硬體量測必須序列、期間不做轉換。`artifacts/` 不在版本控制內；bundle 之間以 hash 證明相同的 weight.bin 為 hard link，不要原地修改任何 weight.bin。
