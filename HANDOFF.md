# 交接（2026-09-13）

給下一個接手的人，或一週後忘光的自己。先讀 [README](README.md) 知道專案是什麼，再讀這份。名詞在 [名詞解釋](research/glossary.md)。前一輪（2026-09-12，另一個 AI 助理）的交接保留在 [research/handoff-2026-09-12.md](research/handoff-2026-09-12.md)。它的耗電數字是在充電時量的，不能和本輪比；其他內容只當線索。

一句話說專案：把 Qwen3-ASR 1.7B 語音辨識模型轉成 Core ML，跑在 Mac 的 Neural Engine（Apple 晶片裡專門跑 AI 模型的省電單元）上，包成 Standard ASR（一套語音辨識引擎的共同介面規範）的外掛。

## 現在在哪裡

做完了，程式碼全部提交，沒有東西在背景跑。

- **可以用**：套件 `std_qwen3asr_ane/` 裝得起來，300 個測試全部通過，Standard ASR 的介面檢查通過，`qwen3-asr-ane transcribe` 對真實錄音會出正確文字。
- **有三個模型版本要分清楚**。本輪做了兩個 8-bit 壓縮版，權重完全相同，差別只在解碼器的 28 層切成幾個 Core ML 檔案：
  - 「7 個檔案版」（每個 4 層）：所有品質門檻是在它身上跑的。目錄 `artifacts/qwen3-asr-1.7b-lut8-g32-compiled`。
  - 「2 個檔案版」（每個 14 層）：現在的預設。目錄 `artifacts/qwen3-asr-1.7b-p14-lut8-g32-compiled`，`artifacts/qwen3-asr-1.7b` 是指向它的符號連結。少幾次系統呼叫，快約 5%。沒有另外跑品質門檻，因為權重相同、400 句輸出逐字相同，品質結果直接沿用。
  - 「未壓縮版」：接手時的起點，`artifacts/qwen3-asr-1.7b-compiled`，所有「快 30%」「省 27%」都是和它比。
  - 目錄名裡的 `lut8 g32` 是壓縮方法：權重用查表法壓成 8 位元，每 32 個輸出通道共用一張表。`p14` 是每個檔案 14 層。編碼器（把聲音變特徵的部分）沒壓縮，因為它只佔辨識時間約 2%，壓了沒好處。
- **品質**：8-bit 版在 200 句挑選集和 200 句驗證集上，錯誤率和未壓縮版、和官方 PyTorch 版都量不出差別。門檻是事先寫在 [research/preregistration-2026-09-13.md](research/preregistration-2026-09-13.md) 的：英文詞錯誤率最多差 +0.25 個百分點、中文字錯誤率最多差 +0.40 個百分點。挑選集實測差 0.00 和 −0.03，驗證集 −0.04 和 −0.19。
- **速度**（2 個檔案版）：兩段測試錄音（英文 15 秒、中文 4 秒）分別 1.43 秒和 0.35 秒，比未壓縮版快 30%。
- **耗電**（7 個檔案版）：整機估計，每秒音訊 2.3 焦耳，比未壓縮版少 27%。2 個檔案版在另一場次量到 2.15 到 2.24 焦耳，和 7 個檔案版差 1 到 4%。
- **真的在 Neural Engine 上**：用 Instruments（Xcode 的效能分析工具）錄到辨識期間 Neural Engine 忙 90%，GPU 沒動，每次 Core ML 呼叫都對得上一段 Neural Engine 工作。證據檔在 `artifacts/telemetry/p14-lut8-g32-attribution.json`。
- **和 GPU 比**：這台 M5 Max 上 GPU 路線都比 Neural Engine 快。MLX（Apple 的 GPU 機器學習框架）快 2.5 到 7 倍，官方 PyTorch 走 GPU（PyTorch 的 Apple GPU 後端，叫 MPS）快 1.8 倍。Neural Engine 的優勢是功率三分之一、Python 程序記憶體小、GPU 空閒。整機耗電上 8-bit 版和 MLX bf16（未壓縮的 16 位元格式）差不多，輸給 MLX 4-bit。

## 試過但沒採用的路線

每一條都有量測，細節在 [實驗紀錄](research/technical-blog.md) 第 23 到 27 節。

失敗或不值得的：

- **4-bit 查表壓縮**：速度和 8-bit 一樣（Neural Engine 解壓縮的成本看元素數，不看位元數），品質變差超過門檻。
- **另一種 4-bit 格式**（按區塊線性量化，Core ML 叫 linear int4 per-block）：慢一倍。Neural Engine 對這種格式沒有硬體加速的解壓縮。
- **解碼器切成 1 個檔案（28 層）**：轉得出來，但載入時 Core ML 回報「無法建立執行計畫」，就是系統排不出這個模型怎麼跑。
- **只更新 KV 快取的一格**（KV 快取是解碼器記住前文的記憶體；試了 Core ML 的 `slice_update` 和 `scatter` 兩種寫入運算）：這版 macOS 27 beta 載不起來。現在是整塊快取一起寫，成本約 4%。
- **推測解碼用 SenseVoice 當小模型**（前一輪試的）：拿到的公開 Core ML 版本在這台機器上編碼器輸出全是無效數值，修不好，放棄。

成功但沒放進套件的：

- **推測解碼，Neural Engine 加 GPU**：小模型（0.6B）在 GPU 上猜 15 個字，大模型（1.7B）在 Neural Engine 上一次驗證，只保留猜對的。輸出和純 Neural Engine 逐字相同，200 句快 2.0 到 2.25 倍，耗電少 38%。沒放進套件的原因：要多載一個模型，常駐記憶體從 2.6 GB 變 6.4 GB；GPU 不再空閒；它用的 `mlx-audio` 要 transformers 5，本套件的轉換工具要 transformers 4，只能放在獨立環境 `experiments/mlx_draft/`。要不要做成選配功能需要你決定。

## 接下來可以做什麼（按預期收益排）

1. **提示階段一次處理 64 個 token**。現在解碼器一次處理 16 個 token（token 是模型處理文字的最小單位）。辨識一開始要把音訊特徵餵進解碼器，叫提示階段。推測解碼模式下這已經是最花時間的單項：200 句總共 82 秒，其中 43 秒在這裡。之前在未壓縮版上量過，改成一次 64 個可以讓整體快 13 到 15%。做之前要先量兩組解碼器同時載入的常駐記憶體，因為 Neural Engine 端的權重可能被複製兩份。
2. **把推測解碼做成選配功能**。量測都做完了，缺的是整合：用 uv 的 `[tool.uv] conflicts` 設定（宣告兩組依賴互斥，不能同時裝）把兩個 transformers 版本分開、把小模型和大模型驗證時用的精簡版輸出層搬進套件並列入模型檔需求、只支援批次不支援串流、用假的小模型寫測試、再做一次程式碼審查。
3. **更好的 4-bit**（用真實資料校正壓縮誤差，或更小的分組）。只省記憶體，不會更快。
4. **逐零件耗電**（`sudo powermetrics`）。需要管理員密碼，本輪沒有。

## 怎麼重跑

最小檢查，確認東西還活著：

```sh
export UV_CACHE_DIR="$PWD/.cache/uv" HF_HOME="$PWD/.cache/huggingface"
uv sync --project std_qwen3asr_ane --frozen --group convert --python 3.12
std_qwen3asr_ane/.venv/bin/pytest -q std_qwen3asr_ane/tests
std_qwen3asr_ane/.venv/bin/standard-asr compliance run std-qwen3asr-ane/1.7b     # 介面檢查，不載入模型
std_qwen3asr_ane/.venv/bin/qwen3-asr-ane transcribe artifacts/evaluation/smoke/qwen_official_en.wav
```

完整量測，腳本在 `experiments/workflows/`，說明在那裡的 README。順序：

1. `candidate_gate.sh lut8-g32`：速度加挑選集品質。
2. `final_gate.sh lut8-g32`：驗證集、耗電、記憶體、Instruments、串流、介面檢查。
3. `p14_and_draft_evidence.sh`：2 個檔案版的證據，加推測解碼實驗。
4. `closing_checks.sh`：推測解碼的 Instruments、用正式指令重建模型並確認權重檔內容和出貨版完全相同、乾淨目錄重新安裝。

要換別的模型檔，設環境變數 `STANDARD_ASR_STD_QWEN3ASR_ANE__MODEL_DIR=<路徑>`。

這些腳本需要幾個不在版本控制裡的東西：測試錄音 `artifacts/evaluation/smoke/`、評估語料 `artifacts/evaluation/silu-validation/`（目錄名是歷史原因，前一輪為了查模型裡一個數學函數 SiLU 在 Neural Engine 上的精度問題建的，後來一直沿用）、GPU 對照組用的 `artifacts/source/Qwen3-ASR-1.7B-MLX-4bit`。怎麼準備寫在 [research/evaluation-plan.md](research/evaluation-plan.md)。推測解碼另外需要 `experiments/mlx_draft/` 環境和 `artifacts/source/Qwen3-ASR-0.6B`，見 [experiments/mlx_draft/README.md](experiments/mlx_draft/README.md)。

注意事項：

- 硬體量測要一個一個跑，期間不要做模型轉換。
- `artifacts/` 裡不同模型包之間，內容相同的 `weight.bin` 是硬連結。不要就地修改任何 `weight.bin`。
- `artifacts/model-aliases.json` 記錄預設模型從哪個換到哪個，以及各自 manifest（模型包的檔案清單）的內容指紋。
