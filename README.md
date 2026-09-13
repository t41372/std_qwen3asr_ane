# Qwen3-ASR 1.7B 跑在 Apple Neural Engine 上

讀者設定：會用 Python 和 macOS、聽過語音辨識，但完全不認識這個專案。專有名詞第一次出現時會解釋，完整說明在 [名詞解釋](research/glossary.md)。

## 這是什麼

Qwen3-ASR 1.7B 是阿里巴巴開源的語音辨識模型，17 億參數。它分兩部分：編碼器把聲音變成特徵，解碼器根據特徵一個字一個字產生文字。解碼器有 28 層，辨識的時間大部分花在這裡。這個專案把整個模型轉成 Core ML 格式，讓它跑在 Mac 晶片裡的 Neural Engine 上。

- Core ML 是 Apple 的模型執行框架。模型要先轉成它的格式，才能交給 macOS 排到硬體上跑。
- Neural Engine 是 Apple 晶片裡專門跑 AI 模型的單元，和 CPU、GPU 是三個不同的東西。特點是很省電，但記憶體頻寬比 GPU 小，所以跑大模型不一定比 GPU 快。

成品是一個 Python 套件 `std_qwen3asr_ane/`。它是 [Standard ASR](https://github.com/standard-voice/standard_asr) 的外掛。Standard ASR 是一套語音辨識引擎的共同介面規範（目前 0.2 版），任何引擎只要照它的介面做，就能用同一組指令和 Python API 呼叫。這個外掛的模型名稱是 `std-qwen3asr-ane/1.7b`。

這個專案是接手前一輪（2026-09-12，另一個 AI 助理）的成果繼續做的。前一輪做到「未壓縮版能在 Neural Engine 上跑」，本輪從那裡開始。

## 現在的狀態

可以裝、可以用。限制在這一節最後。

- 預設模型是 8-bit 壓縮版，跑在 Neural Engine 上。用 Instruments（Xcode 附的效能分析工具，能錄下每個硬體單元什麼時候在忙）錄下來，辨識期間 Neural Engine 有 90% 的時間在忙，GPU 完全沒動。
- 品質和原版一樣。在 200 句從未拿來調參的英文加中文測試上，錯誤率和未壓縮版、和官方 PyTorch 版都量不出差別。
- 比接手時的未壓縮版快 30%，耗電少 27%，常駐記憶體少 1.6 GB。
- 這台 Mac 的 GPU 還是比較快。MLX（Apple 自己的 GPU 機器學習框架）快 2.5 到 7 倍，官方 PyTorch 程式碼走 GPU 也快 1.8 倍。Neural Engine 的好處是平均功率只有 GPU 路線的三分之一、Python 程序的記憶體很小、GPU 可以留給別的事。
- 另外做了一個「推測解碼」實驗，混用 Neural Engine 和 GPU：小模型（0.6B）在 GPU 上先猜接下來 15 個字，大模型（1.7B）在 Neural Engine 上一次驗證，只保留猜對的部分。因為每個字最後都經過大模型認可，結果和大模型自己逐字算完全一樣，只是快。實測快 2 倍、耗電少 38%。它沒有放進正式套件，原因有三：
  - 要多載一個模型，常駐記憶體從 2.6 GB 變 6.4 GB。
  - GPU 不再空閒。
  - 它用的 `mlx-audio` 套件要 transformers 5，本套件的轉換工具要 transformers 4，裝不進同一個環境。
- 300 個測試全部通過。

不支援：逐字時間戳、說話者分離、限定候選語言（例如「只在中文和日文裡選」）、超過 30 秒的長音訊自動接續。這些請求會在辨識開始前被拒絕，或是忽略那個選項照常辨識。所有量測只在一台 M5 Max 上做過。

## 主要數字

測試錄音是 Qwen 官方的兩段：英文 15 秒、中文 4 秒。速度是熱機後 5 次的中位數。品質欄的「配對比較」是指同一句話讓兩個版本各辨識一次再比錯誤率。bf16 和 fp16 都是未壓縮的 16 位元浮點數格式。MPS 是 PyTorch 的 Apple GPU 後端。耗電是 Mac 電源管理晶片回報的整機功率，換算成每秒音訊耗多少焦耳；它不是插座上的電表，也分不出哪個零件耗了多少。詳細方法和限制在 [量測結果](research/results-2026-09-13.md)。

| 方案 | 英文 15 秒 | 中文 4 秒 | 每秒音訊耗電 | 常駐記憶體 | 品質 |
|---|---:|---:|---:|---:|---|
| Neural Engine，未壓縮（接手時的起點） | 2.06 s | 0.50 s | 3.19 J | 4.2 GB | 比較基準 |
| **Neural Engine，8-bit 壓縮（預設）** | **1.43 s** | **0.35 s** | **2.34 J** | **2.6 GB** | 與基準量不出差別 |
| Neural Engine 驗證 + GPU 猜字（實驗） | 0.60 s | 0.20 s | 比預設少 38% | 6.4 GB | 與預設逐字相同 |
| GPU，MLX 4-bit | 0.21 s | 0.09 s | 1.15 J | 2.1 GB | 沒有做配對比較 |
| GPU，MLX bf16 | 0.46 s | 0.14 s | 1.99 J | 4.1 GB | 沒有做配對比較 |
| 官方 PyTorch，GPU（MPS 後端，bf16） | 0.78 s | 0.20 s | 沒有量 | 沒有量 | 沒有做配對比較 |
| 官方 PyTorch，CPU | 2.74 s | 0.80 s | 沒有量 | 沒有量 | 品質參照，其他版本都和它比 |

三個備註：

- 耗電每個方案量兩次，表上是平均。不同場次量的數字不能直接相除，所以實驗版只寫「少 38%」，那是和預設版在同一場次量的結果。
- 預設版的速度是現在這個版本量的。耗電是用「解碼器切 7 個檔案」的版本量的；現在預設把解碼器切成 2 個檔案，權重相同、輸出逐字相同，另一場次量到的耗電差 1 到 4%。「切幾個檔案」的意思見下面的安裝說明。
- 實驗版的速度是這兩段錄音的數字。200 句的平均是 2.0 到 2.25 倍。

## 怎麼裝

需要 Apple Silicon Mac、macOS 15 以上、Python 3.12、[uv](https://docs.astral.sh/uv/)（Python 套件管理工具）。所有量測在 M5 Max、64 GB、macOS 27.0 上做的，其他機器沒驗證過。

從這個目錄執行。四個步驟：下載原始模型、轉成 Core ML、壓縮成 8-bit、編譯成可直接載入的格式。

```sh
export UV_CACHE_DIR="$PWD/.cache/uv"
export HF_HOME="$PWD/.cache/huggingface"
uv sync --project std_qwen3asr_ane --python 3.12 --group convert
uv run --project std_qwen3asr_ane --group convert qwen3-asr-ane download
uv run --project std_qwen3asr_ane --group convert qwen3-asr-ane build \
  --token-batch-size 16 --layers-per-partition 14 --output artifacts/qwen3-asr-1.7b-fp16
uv run --project std_qwen3asr_ane --group convert qwen3-asr-ane compress \
  --source artifacts/qwen3-asr-1.7b-fp16 --output artifacts/qwen3-asr-1.7b-lut8 --bits 8
uv run --project std_qwen3asr_ane qwen3-asr-ane compile \
  --source artifacts/qwen3-asr-1.7b-lut8 --output artifacts/qwen3-asr-1.7b
```

- `download` 抓約 4.3 GB，之後全部離線。
- `build` 的兩個參數：`--token-batch-size 16` 是解碼器一次處理 16 個 token（token 是模型處理文字的最小單位，大約一個中文字或半個英文詞）；`--layers-per-partition 14` 是把解碼器的 28 層切成 2 個 Core ML 檔案、每個 14 層。切的檔案越少，每產生一個字要呼叫系統的次數越少。之前的版本切成 7 個檔案、每個 4 層。
- `compress` 把解碼器的權重從 16 位元壓成 8 位元。
- `compile` 之後第一次載入約 35 秒（系統在做裝置最佳化），之後每次不到 2 秒。

用法：

```sh
uv run --project std_qwen3asr_ane qwen3-asr-ane transcribe recording.wav
uv run --project std_qwen3asr_ane standard-asr list
```

```python
from standard_asr import discover_models

engine = discover_models().create("std-qwen3asr-ane/1.7b", model_dir="artifacts/qwen3-asr-1.7b")
print(engine.transcribe("recording.wav").text)
```

跑測試和介面檢查：

```sh
uv run --project std_qwen3asr_ane --group convert pytest std_qwen3asr_ane/tests -q
uv run --project std_qwen3asr_ane standard-asr compliance run std-qwen3asr-ane/1.7b
```

測試共 300 個，全部通過。沒下載原始模型時有 2 個會跳過。第二個指令是 Standard ASR 附的介面檢查，確認這個外掛的介面符合規範；它不載入模型。

串流（邊錄邊辨識）也支援：先給暫定結果、錄完給定稿，可以取消、可以指定語言、可以給最多 128 個 token 的上下文提示。詳見 [research/standard-asr-features.md](research/standard-asr-features.md)。

## 文件地圖

- [量測結果](research/results-2026-09-13.md)：所有數字、量法、限制。
- [驗收門檻](research/preregistration-2026-09-13.md)：在看到結果之前就寫死的通過標準。
- [實驗紀錄](research/technical-blog.md)：按時間順序的完整過程，包括失敗的路線。
- [交接](HANDOFF.md)：接下來可以做什麼、怎麼重跑。
- [名詞解釋](research/glossary.md)。

## 重跑量測

`experiments/` 底下是量測工具，`experiments/workflows/` 是把它們串起來的腳本。模型檔和量測結果放在 `artifacts/`，不進版本控制。

- `evaluate.py`：品質評估。固定語料、統一的文字正規化、同一句話讓兩個版本各辨識一次再比（配對比較）。
- `benchmark_energy.py` 加 `power_v2/`：等工作量的整機耗電。
- `trace_ane.py` 加 `bind_trace_evidence.py`：用 Instruments 錄 Neural Engine 和 GPU 的工作時段，並把證據和模型檔的 hash 綁在一起。
- `benchmark_mlx.py`：GPU 對照組（MLX-Audio），獨立環境。
- `benchmark_mlx_draft.py` 加 `mlx_draft/`：推測解碼實驗，獨立環境。
- `probe_decoder_floor.py`：拆解解碼器每一步的時間花在哪裡。這個分析決定了用 8-bit 壓縮和減少檔案數。
