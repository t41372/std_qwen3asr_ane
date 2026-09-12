# ANE placement、品質及效率的自動驗證方案

研究日期：2026-09-12。範圍：本機 macOS 27、Core ML / coremltools 公開 API；不修改 runtime、不下載模型、不使用 sudo。本文件區分已查證事實、當前環境限制及**本專案建議門檻**。以下方法尚未對完整 Qwen3-ASR 模型執行；不能據此宣稱模型已移植成功。

## 可以證明什麼

`CPU_AND_NE` 的意義是允許 CPU 與 Neural Engine、排除 GPU。CPU fallback 仍然合法，故成功載入及 predict 並不證明 ANE 執行。`ALL` 讓系統自行選擇，也不是 ANE 保證。這是 Apple 的配置語義。[Apple MLComputeUnits](https://developer.apple.com/documentation/coreml/mlcomputeunits)

建議將驗證結果分成獨立欄位，而不是一個容易誤導的 `ane_enabled: true`：

| 層級 | 自動產出 | 能支持的結論 | 不能支持的結論 |
| --- | --- | --- | --- |
| 配置 | compute units、模型 SHA、軟硬體版本 | 模型允許 ANE | 實際曾使用 ANE |
| 計畫 | 每個 operation 的 supported / preferred / cost | 編譯器預期把哪些工作放 ANE | 真實執行時間、能耗、佔用率 |
| 動態 | Core ML + Neural Engine trace、CPU_ONLY 對照 | 在對應推理區間確有 ANE 活動 | 每個 operation 的實際 FLOPs 比例 |
| 品質 | 固定 corpus 的 WER / CER 及配對差值 | 此測試分布下未超過退化容忍值 | 所有語言、場景都零退化 |
| 效率 | 端到端 latency / RTF、CPU time、RSS | 真實應用成本和速度 | 僅由 CPU 變少推斷電力下降 |
| 能耗 | 同機受控 power samples 積分 | 此工作負載的估計 joules 差异 | ANE 全程 process-exclusive、跨機絕對能效 |

## 本機實際探測紀錄

按執行順序記錄，供之後 technical blog 引用：

1. `sw_vers` 回報 macOS `27.0`、build `26A428`；`uname -m` 回報 `arm64`。M5 Max / 64 GB 是使用者提供的設備資料，此子任務沒有重新核實硬體欄位；sandbox 內 `sysctl -n machdep.cpu.brand_string hw.memsize` 被拒。
2. `xcode-select -p` 是 `/Library/Developer/CommandLineTools`；直接 `xcrun xctrace list templates` 找不到 utility。這不表示未安装 Xcode。
3. `/Applications` 存在 `Xcode.app` 和 `Xcode-beta.app`。以單次命令的 `DEVELOPER_DIR=/Applications/Xcode-beta.app/Contents/Developer` 呼叫 xctrace，不修改全局 developer directory。
4. 該 xctrace 執行在 sandbox 內退出，原因是無法在 `~/Library/Caches/com.apple.dt.InstrumentsCLI/path_manager` 建立 cache。**已找到工具，但尚未取得可用 templates 或 trace**。此為 OS sandbox 拒絕，未提交權限升級，亦非 auto-review rejection。
5. `/usr/bin/powermetrics --help` 成功，列出 `cpu_power`、`gpu_power`、`ane_power`、`thermal` sampler 及 NUL-separated plist 輸出。
6. 執行一次 `powermetrics --samplers cpu_power,gpu_power,ane_power --sample-count 1 --sample-rate 100 --format plist`，退出碼 1，訊息是需要 superuser。沒有使用 sudo、修改 sudoers 或啟動 daemon。
7. 讀取本機 Xcode-beta SDK 的 `MLComputePlan*.h` 及 xctrace man page，與 Apple Python source 交叉核對；發現 Python 文檔範例及 experimental benchmark 命名容易被誤讀，詳見下面。

## MLComputePlan：正確的 Python 用法

公開 wrapper 在 macOS 14.4 以上提供 compute plan。輸入是 compiled `.mlmodelc`，不是原始 `.mlpackage`。`MLModel.get_compiled_model_path()` 傳回的暫存路徑只在 Python model 物件存活期間有效；需要持久化就複製到本專案 artifact 目錄。[Apple bridge source](https://github.com/apple/coremltools/blob/main/coremlpython/CoreMLPython.mm)、[MLModel API](https://apple.github.io/coremltools/source/coremltools.models.html)

以下是 collector 設計範例，**未在本子任務實跑**。正式版本應寫入 JSON、記錄 exception，並對每個 shape bucket / model shard 分別執行：

```python
import coremltools as ct
from coremltools.models.compute_device import MLNeuralEngineComputeDevice
from coremltools.models.compute_plan import MLComputePlan


def visit_operations(block, path):
    for index, operation in enumerate(block.operations):
        operation_path = f"{path}/op/{index}"
        yield operation_path, operation
        for block_index, child in enumerate(operation.blocks):
            yield from visit_operations(child, f"{operation_path}/block/{block_index}")


model = ct.models.MLModel("model.mlpackage", compute_units=ct.ComputeUnit.CPU_AND_NE)
plan = MLComputePlan.load_from_path(
    path=model.get_compiled_model_path(),
    compute_units=ct.ComputeUnit.CPU_AND_NE,
)
program = plan.model_structure.program
if program is None:
    raise ValueError("Expected an ML Program; inspect other model types separately")

rows = []
for function_name, function in program.functions.items():
    for path, operation in visit_operations(function.block, function_name):
        usage = plan.get_compute_device_usage_for_mlprogram_operation(operation)
        cost = plan.get_estimated_cost_for_mlprogram_operation(operation)
        rows.append({
            "path": path,
            "operator": operation.operator_name,
            "outputs": [value.name for value in operation.outputs],
            "preferred": type(usage.preferred_compute_device).__name__ if usage else None,
            "supported": [type(device).__name__ for device in usage.supported_compute_devices]
            if usage else None,
            "ane_preferred": isinstance(usage.preferred_compute_device, MLNeuralEngineComputeDevice)
            if usage else None,
            "estimated_weight": cost.weight if cost is not None else None,
        })
```

API 要點：`program.functions` 才是 function map；`ComputeUnit` 是單數；`get_compiled_model_path()` 是 MLModel 的方法。網頁中可見 `program["main"]`、`ComputeUnits`、`get_compiled_path()` 等不一致範例，不應直接照抄。`usage` 或 `cost` 可以是 `None`；保留 unknown，不能當成 CPU 或零成本。[Apple compute_plan Python source](https://apple.github.io/coremltools/_modules/coremltools/models/compute_plan.html)

`supported_compute_devices` 表示可執行的裝置，`preferred_compute_device` 表示 framework 偏好的裝置。Python docstring 對 supported 的描述有 copy/paste 問題，本機 SDK header 的語義明確。這些都是 anticipated device usage。`cost.weight` 是 operation 佔整個模型執行的估計工作量，範圍 `[0, 1]`，不是毫秒或瓦特。[Apple supportedComputeDevices](https://developer.apple.com/documentation/coreml/mlcomputeplandeviceusage/supportedcomputedevices)、[Apple preferredComputeDevice](https://developer.apple.com/documentation/coreml/mlcomputeplandeviceusage/preferredcomputedevice)、[Apple MLComputePlan.Cost](https://developer.apple.com/documentation/coreml/mlcomputeplan-1w21n/cost)

### 如何彙整而不灌水

以下為本專案建議：

- 每個 function / shard 分開保留 operation count、已知 weight 總和、unknown count、ANE preferred weight 總和及 CPU preferred weight 總和。不要只報 ANE operation 百分比：幾個大型 matmul 比許多 shape ops 重。
- 對沒有分支的 function，可另列 `sum(ane_preferred_weight) / sum(known_weight)`，明確命名為 **ANE 計畫成本比例（已知成本部分）**。同時列 unknown 數量，避免分母刪除未知重算子而虛增比例。
- 嵌套 block 遞迴收集是為了完整性，不代表所有分支都會執行。未知 loop 次數及不同 function 的 normalized weights 不直接相加。無法可靠彙整就保留 function 層結果。
- 多 shard 的整體估計應按實際呼叫次數與各 shard 的同步 predict wall time 加權：`sum(call_count × predict_time × shard_plan_share) / sum(call_count × predict_time)`；結果仍是混合估計，不能命名為 hardware utilization。
- audio encoder、decoder prefill、decoder token step、LM head 各自報告。完整 ASR 若大量 decoder 仍在 GPU/CPU，只能宣傳對應子模組的 ANE 移植。
- 對所有部署 shape bucket 和精度版本重新產生 plan。不能用短序列 FP16 的結果替長序列或量化版本背書；重大 macOS / coremltools 更新後重跑。

## 無 root 的自動 benchmark

穩定核心應由自己維護的小型 runner 量測 wall clock，搭配 `MLModel.load_duration_in_nano_seconds`、`last_predict_duration_in_nano_seconds`。後兩者量的是 Core ML API 呼叫區間；沒有 Python audio preprocessing、tokenization 及解碼迴圈的全部成本。[Apple MLModel source](https://apple.github.io/coremltools/_modules/coremltools/models/model.html)

**不要把 `MLModelBenchmarker.benchmark_operation_execution()` 當成硬體 profiler。** 當前 Apple source 使用整體 prediction sample 乘 `estimated_cost.weight` 計算 operation 時間。它適合排查計畫中的重點算子，不能獨立證明 ANE 時間。此外其 prediction helper 每次 iteration 建立 state；對 KV cache 的真實 autoregressive decode，需自訂持續更新 state 的 runner。[Apple experimental perf_utils source](https://raw.githubusercontent.com/apple/coremltools/main/coremltools/models/ml_program/experimental/perf_utils.py)

建議每個測試 case 執行以下流程：

1. 鎖定 source model revision、converter SHA、模型 package hash、uv.lock、OS build、compute units、shape、precision、decoder options、audio hash、隨機 seed。warmup、compile、load、首筆、穩態分開。
2. 各模式使用獨立 process，依隨機或 ABBA 次序測 `CPU_ONLY`、`CPU_AND_NE`、`CPU_AND_GPU`、`ALL`，再測相同 Qwen revision 的 MLX 基線。一次只跑一個 benchmark；品質資料集下載與其他模型編譯不得並行。
3. 每個 shape 至少 5 次 warmup、30 次有效 microbenchmark；端到端至少 5 個重複 batch。若變異係數 > 5%，以 capped retry 重新測量並保存被排除批次與理由；不要挑最快結果。
4. decode 微測試重放固定 tokens、相同初始 KV 狀態及上下文長度，避免输出少了反而看似更快；端到端則使用真實生成直到 EOS，記錄 token count。
5. 使用 monotonic clock 記錄 preprocessing、encoder、prefill、每個 decode step、postprocessing、首個 token / partial、final，並記錄 `resource.getrusage` 的 user/sys CPU time 與 macOS RSS 單位。單 process RSS 不包含所有 framework / daemon 成本。
6. 報告 p50 / p95、RTF `wall_seconds / audio_seconds`、tokens/s、peak RSS 以及原始逐筆 JSONL。輸入讀取與模型下載不能混入穩態推理。
7. 記錄 power mode、AC/battery、已知 thermal 狀態；無法讀取的欄位標 unknown。不中斷使用者 app、不自行調整全機設定。

CPU time 下降、GPU 沒有工作、CPU_AND_NE 比 CPU_ONLY 快，都是有用交叉檢查，單獨使用都不是 ANE 動態證據；Core ML 可能存在不同 CPU specialization。

## 無 GUI 的 Instruments 路徑，以及目前缺口

Apple 的 Core ML Instruments 工作流可將 model prediction 區間與 Neural Engine / GPU hardware activity 對齊；Core ML request 是非同步提交，因此應查硬體 lane，而不是只看 request 的標籤。[Apple WWDC22 Core ML profiling](https://developer.apple.com/videos/play/wwdc2022/10027/)

本機 Apple 提供的 CLI 說明在 [xctrace man page](/Applications/Xcode-beta.app/Contents/Developer/usr/share/man/man1/xctrace.1)。可設計完全由 CLI 驅動的 pipeline：

```sh
DEVELOPER_DIR=/Applications/Xcode-beta.app/Contents/Developer xcrun xctrace list templates
DEVELOPER_DIR=/Applications/Xcode-beta.app/Contents/Developer xcrun xctrace list instruments
# 以下為模板；runner 路徑需換成實際 benchmark entry point。
DEVELOPER_DIR=/Applications/Xcode-beta.app/Contents/Developer xcrun xctrace record \
  --template 'Core ML' --time-limit 30s --no-prompt \
  --output artifacts/validation/run.trace \
  --launch -- /absolute/path/to/benchmark-runner
DEVELOPER_DIR=/Applications/Xcode-beta.app/Contents/Developer xcrun xctrace export \
  --input artifacts/validation/run.trace --toc --output artifacts/validation/toc.xml
```

`Core ML` 是預期模板名稱，必須由 list 的實際結果確認；可用 `--instrument` 增加已列出的 Neural Engine / GPU instrument。先讀 TOC 再選可輸出的 table XPath，禁止憑印象寫死 table schema。解析器保留 exporter / Xcode 版本、trace SHA、原始 XML。若該版本無法 export 所需 hardware table，結果標 `runtime_placement: unavailable`，不能把沒有 rows 當成零 ANE。

透過 runner 的 signpost 或精確時間標記辨識 encoder / prefill / decode 各區間，按 process / model 關聯 hardware events；若只有系統級事件，就利用隔離工作負載、重複執行和 CPU_ONLY 負對照降低背景 ANE 污染。這仍應註明 attribution 限制。profile run 有 overhead，性能主結論來自另外的 unprofiled run。

`--no-prompt` 使流程不等待互動，但不授予隱私、developer 或 sandbox 權限。目前 cache 寫入受限使連 templates 枚舉都尚未成功；未驗證 TCC、developer tracing permission、硬體 table 可匯出性。**所以當前 session 可以完成 plan / latency / quality 自動迴圈，不能誠實宣称完整動態 tracing 已經無人介入可用。** 後續若需在現有 sandbox 外執行，需先告知會觸及 Instruments 的使用者 cache；不能更改 HOME 或全局權限來繞過限制。

macOS 27 文件另有 Core AI Instruments，但 `.aimodel` / Core AI 是不同 framework 工作流，不能把 Core AI 的工具文檔直接視為 Core ML `.mlpackage` 的能力保證。[Apple Core AI profiling](https://developer.apple.com/documentation/coreai/inspecting-debugging-and-profiling-core-ai-models)

## 功耗：不能在無權限時編造數字

本機 powermetrics help 明確表示 subsystem power 為估計值，適合同設備內調優，不適合作跨設備比較。`ane_power` 在此 OS 版本列為 dedicated rail power / frequency sampler；某些舊版的欄位或 sampler 組合可能不同，所以 collector 必須先探測本機 help。

當前普通使用者無法取得樣本；應輸出 `energy.status = permission_required`、`energy.joules = null`。不要將 Activity Monitor energy impact、CPU time、ANE core count 或計畫成本換算為 joules。`MLComputeDevice.get_all_compute_devices()` 與 `total_core_count` 只提供設備資訊。[Apple compute_device source](https://apple.github.io/coremltools/_modules/coremltools/models/compute_device.html)

未來在已具適當權限的 measurement 環境中，本專案建議：

- 使用固定權限、固定 sampler、有限時長的 collector，結果只寫專案 artifact 目錄。此文件不建議自動安裝 root helper 或修改 sudoers。
- 以 100–500 ms 採樣、至少 30–60 s 的重複工作負載攤平瞬時採樣偏差；短語音需批量重放。以 monotonic timing 對齊暖機、idle、有效測試區間。
- plist 依 NUL 分隔逐筆解析；保存 raw samples、單位、實際 sample duration、invalid flag。未知 key 或缺樣本必須 fail / unavailable，不能默認 0。
- 逐 rail 計算 `E = sum(P_watts × dt_seconds)`，同報 CPU / GPU / ANE 及可取得的總和；這個和若未含 DRAM / 全機其他負載，不叫整機功耗。
- 同報未扣 idle 的 gross energy、扣除鄰近 idle baseline 的 incremental energy。負 incremental estimate 保留為 measurement noise，不能截成零後當最佳結果。
- 主要比較 `joules / audio_second`、同語料總 joules、p95 final latency。ANE 的瞬時瓦數高低不能獨立代表端到端節能。
- idle-before / workload / idle-after 配對至少 5 輪，交錯順序，比較 bootstrap CI。背景 ANE activity 明顯的批次標污染，保存而不偷偷刪除。

## 品質測試與可宣傳的建議門檻

以下均為本專案初始政策建議，不是 Apple 或 Qwen 官方標準。允許在量測前調整政策，不能看完結果才改容忍度。

| Gate | 建議條件 |
| --- | --- |
| 數值 smoke | 每個 shape 有有限輸出、正確 shape / mask / state 更新；encoder cosine ≥ 0.999 作診斷起點，另保留 max / relative error。不能以此替代 WER |
| Token 邊界 | 固定 reference prompt + token replay，記錄每步 logits top-1、top-k、margin 與首個 divergence；不因小 margin 的 token 翻轉就直接判整體模型失敗 |
| 品質 release | 在封存的代表性語料，paired bootstrap 的 WER 差值 95% 上界 ≤ +0.3 個百分點；在基線 WER ≥ 1% 的切片另要求相對差上界 ≤ +5%；中文另測預先固定 normalization 的 CER |
| 品質覆蓋 | 英文、中文、混合語言、噪音、短指令、長音訊、靜音各有切片；建議先 ≥ 1,000 utterances / 10 h，再依 CI 寬度增量。沒有統計力就判 inconclusive |
| 可靠性 | 全部 corpus 無 crash / timeout / runaway decode；silence hallucination、切段丢字、跨 session KV 污染、cancel/reset 有自動用例 |
| 計畫 placement | encoder / prefill / decode / LM head 每個重大 shard 有計畫；主要線性代數的計畫成本 ≥ 90% preferred ANE，且任何 unknown 重算子均阻止通過；其他占比單列 |
| 實際 placement | 各主要階段有可關聯的 ANE hardware 活動，CPU_ONLY 對照沒有相同模式，沒有主要階段完全 CPU/GPU fallback；需保留 raw trace |
| 應用 latency | 固定本機 workload p95 RTF < 1 是最初可用目標；相對 MLX 的速度分開報，並不作 ANE 移植成功的必要条件 |
| 節能宣傳 | 同機受控 energy / audio_second 至少下降 10%，且 95% CI 不跨零改善；品質 gate 通過，並披露 latency 的變化 |

標點、大小寫、數字、中文分詞與空白處理固定在評分版本中，基線與候選共用同一 scoring pipeline。除總 WER 外保存 `(S, D, I, N)`、utterance 級結果與 paired bootstrap seed；multilingual WER 不能任意平均成單一總分掩盖 regression。silence 單獨評估，不用空 reference 除以零。

「絕大部分工作在 ANE」應限定為**主要 neural model 計算**，以全部主要階段的計畫與 trace 交叉支持，另外揭露 CPU preprocessing / token loop / data movement。不能把 ANE active duration / wall duration 說成 FLOPs 佔比；高佔用率也不保證省電。只有 encoder 完成時，清楚稱為 encoder ANE port。

## 建議 artifact 與 agent 決策流程

每次 run 寫一個不可覆蓋的 `artifacts/validation/<run-id>/`：`manifest.json`、`capabilities.json`、`placement-*.json`、`timings.jsonl`、`quality.jsonl`、`summary.json`，有能力時追加 `.trace`、export XML、`power.plist`。manifest 中的 phase / bucket / dtype / hash 讓之後 agent 能精確重放失敗。

agent 按序執行：capability probe → 小模型 API smoke → 單 block parity → 完整 shard parity → 計畫檢查 → 真實 token replay → 小 corpus → 封存 corpus → 性能 → trace → energy。每步結果用 `pass` / `fail` / `inconclusive` / `unavailable` 區分；未量測不等於通過。不可因 profiler 權限缺失停止其餘無權限需求的品質、轉換和效能迴圈。

這個分層使 agent 可自動找到第一個數值分歧、CPU fallback 的高成本 operation、decode state 問題與性能回歸，並在每次轉換後重跑同一測試矩陣。當前缺少的是可用 tracing 權限與 power samples；其餘 pipeline 可以先建立並運作，宣傳文案則由已通過的 gate 生成。

## 同日後續：collector 實作與真模型 plan 驗證

研究完成後，依主 agent 指派新增 `std_qwen3asr_ane.diagnostics`。它以 lazy import 實作 `inspect_compute_plan(model_path, compute_units="cpu_and_ne")`，支援 ML Program 的全部 function / nested block、NeuralNetwork layer 和 pipeline 子模型。JSON 保留所有 operation、未知 device、缺失與非法 cost，另按 function / model scope 和 device 彙整原始 cost。沒有推導實際硬體比例。`python -m std_qwen3asr_ane.diagnostics --environment` 只輸出 OS、架構及套件版本，避免 hostname、序號、UUID；本機確認 coremltools 9.0、Python 3.12.13、Standard ASR 0.2.0.dev0。

第一次在 sandbox 中 inspect 已有 frontend package，Core ML 無法在 macOS 的 `/var/folders/.../T/` 建立 compiler working directory；Python `get_compiled_model_path()` 隨後拋出 bare `Exception`。因此 CLI boundary 特別保留該異常捕獲並輸出 `unavailable`，不偽裝成功。新增回歸測試，連同遞迴結構、未知成本、NaN 成本與敏感路徑隱藏等測試共 5 項通過，Ruff check / format 通過。

先向主 agent 說明 Core ML temporary/cache 寫入範圍，取得工具 escalation 後，以正常使用者環境重試成功；沒有改 HOME、sudoers 或系統設定，没有執行 prediction。frontend plan 有 51 operations，其中 15 個 preferred ANE 且有 cost，cost 加總約 1；其餘 36 未知。接著 inspect decoder layer 0 probe：559 operations 中 249 個有 cost、全部 preferred ANE，cost 加總約 1；其餘 310 個 unknown 均為 `const`。完整 JSON 保存於 `artifacts/probes/decoder-layer-0-plan.json`。這驗證了本機 Python collector 可用及該 probe 的編譯器配置預期，**仍未證明完整 decoder 或完整 ASR 的動態 ANE placement**。

## 同日後續：實際 hardware trace 已打通

接著主 agent 指派建立動態驗證，預先說明 Xcode / Core ML 正常 temporary/cache 寫入會超出 workspace。工具 escalation 允許後，局部指定 `DEVELOPER_DIR=/Applications/Xcode-beta.app/Contents/Developer`，成功列出 `Core ML` template 及 `Neural Engine`、`GPU` instruments。沒有使用 sudo、輸入密碼、改 macOS 設定或出現權限互動。這解決了前文「當前 sandbox 無法取得 trace」的環境缺口；powermetrics 的 superuser 限制仍未解決。

依時間順序（America/Phoenix，UTC−07:00）：

1. **10:41:18–10:41:31**，以20秒上限啟動 layer0 真實權重 probe。新增 `experiments/repeat_microbench.py`，使用真 token embedding，重複 stateful predict 8秒，context 邊界重建 state。8,687 次輸出全部 finite，runner 正常退出。
2. 匯出 TOC 確認存在 `ane-hw-intervals`；再 export ANE / GPU / Core ML / clock tables。ANE表有 **8,687** 筆對應 `decoder-layer-0-scale1...Prediction`，與 predict 次數完全相等，hardware duration 總和 **5.721002375秒**。這是實際硬體事件，不是 MLComputePlan 推估。
3. **10:42:51–10:42:54**，嘗試相同模型的 CPU_ONLY 負對照，Core ML load 失敗：`E5MinimalCpu` pattern error，繼而 MIL→EIR `std::bad_cast` / -14。保存 exit1、原始log與trace；0 ANE events **不能當作成功負對照**，因為根本未推理。
4. 主 agent 回報完整 ASR smoke 已成功，因而將動態驗證升級到全流程。**10:43:23–10:44:09**，以90秒上限 launch `experiments/evaluate.py --backend coreml ... --warmups 0 --repeats 1`。兩筆中英文音訊均成功，target exit0，trace長 **46.081563秒**，沒有到上限。
5. Full-ASR ANE hardware table 記錄 **2,433** 筆 Prediction：frontend21、encoder3、7個 decoder partitions 各336、LM head57，涵蓋全部主要匯出元件。Prediction duration 總和 **8.616689秒**。另外10筆 Compile標籤總和22.053862秒，已獨立列出，不混入推理時間。
6. GPU table 全系統有111,833 rows，但 target PID 的0 rows，另1 row無可歸屬 process。機器不是GPU idle；可說此 trace 未觀察到屬於目標進程的GPU執行。ANE schema沒有PID，依compiled model label與有界工作負載歸屬，並保留背景事件。
7. `coreml-os-signpost` 及raw Core ML signposts在此工具組合都是0 rows。因此沒有取得逐CoreML operation的CPU/accelerator timing；不能把空表當成Core ML未執行。硬體 ANE / GPU tables 則有真實資料。
8. 新增 `experiments/trace_ane.py` 支援 record / export / summarize，解析XML引用ID、逐label duration、interval union、target GPU PID attribution，保存command/log。以三個實際錄製檔完成export+summary驗證，兩個scripts的Ruff check/format通過。記錄模型與script SHA-256到 `artifacts/telemetry/provenance.json`。

完整證據、限制與重放命令見 `artifacts/telemetry/README.md`；主要机器可讀結果為 `artifacts/telemetry/full-asr-summary.json`，成功ASR結果為 `full-asr.jsonl`。原始trace/TOC含Instruments自帶的host/device/process metadata，保留本地；summary不散布這些host識別資訊。

我們現在可以說：**在這台Mac的兩筆成功ASR smoke中，實際觀察到frontend、encoder、全部decoder partitions與LM head對應的ANE硬體Prediction活動。** 這支持完整推理路徑的ANE移植，並與先前計畫資料交叉印證。仍不能說量到了各operation的FLOPs占比、零CPU工作、全部場景品質不退化或功耗下降；該trace有profiling overhead，且設備同時有日常活動，不能作正式速度比較。CPU_ONLY負對照的compiler相容性和energy採樣仍是後續缺口。

## 同日後續：將證據重新綁定到實際 T16 候選

主 agent 完成 token batch size 16 的候選後，指派重新收集全部主要graph與完整ASR trace，不能讓舊T1證據替新候選背書。新增 `experiments/inspect_bundle_placement.py`，依候選manifest逐個載入frontend、encoder、7個decoder partitions及LM head，保存完整operation資料與所有package檔SHA-256。此步驟不與自己的trace並行，避免compiler活動混入測量。

十個graph全部完成，保存在 `artifacts/validation/t16-placement/`。11,720個有cost算子全部preferred ANE；14,292個unknown全部是const，沒有已知CPU/GPU preferred，也沒有invalid cost。frontend為15個known ANE ops，encoder4,640，每個decoder partition1,002，LM head51。各graph已知cost之和各自約1，沒有跨graph加總成比例。

接著錄製 `t16-full-asr`，**10:52:56–10:53:56** 兩筆smoke成功、hypothesis與T1一致，但硬體Prediction總數7,436遠超預期。檢查label後發現：帶compiler UUID的10組候選label總599次，而其他plain/generic labels有6,837次。立即向主agent查證，得知T16英文100筆品質評測正在同時推理。這個差異說明硬體table是系統級，不能把總ANE活動直接歸屬launched process。

保留污染trace並寫 `t16-full-asr-attribution.json`，將candidate exact labels、compiled UUID、計數、manifest SHA、placement summary SHA、trace summary SHA分開記錄，明確標 `inferred_with_background`、`isolated_trace_verified=false`。候選label時間總和2.445041130秒也只標profiled觀察，不作benchmark。將sidecar規約交给workflow agent，gate須檢查hashbinding與逐label計數，背景已確認的trace必須inconclusive。

主agent等英文品質測試結束，確認停下其他ANE推理後，才重錄一次 `t16-full-asr-isolated`；為避免額外磁碟壓力，沿用既有graph計畫與bundle，不複製或多重inspect模型，也未清除任何cache。**10:57:51.590–10:58:33.014**，trace長41.424082秒、target exit0、兩筆smoke都成功且仍同T1。錄完立即通知主agent可啟動中文候選評測，再進行純export與hash檢查。

隔離trace有599筆實際ANE Prediction，全部映射到預期10graph：frontend21、encoder3、7decoder各74、LM head57。沒有額外generic或background Prediction；候選hardware interval總和2.206172204秒。target GPU hardware rows為0，unknown GPU process rows也為0；全系統仍有8,544筆GPU活動。compile interval另計，不混入Prediction。

此輪label沒有compiler UUID suffix，因此sidecar記 `compiled_label_uuid=null`，不捏造識別碼。`t16-full-asr-isolated-attribution.json` 以協調隔離、exact model labels、launched候選參數、成功輸出與manifest/package hashes綁定證據；trace後重算全部模型檔hash與計畫時一致。manifest SHA為 `09e8cc648a8e6269c77bad0b5e16a741fdfed062f5131093bf0a0640e0d18f02`。詳細角色計數與限制見 `artifacts/telemetry/t16-evidence.md`。

嚴格區分三類結論：**可見ANE Prediction** 覆蓋實際T16候選全部graph；**CPU工作量未知**，因Core ML signpost表仍空且CPU前後處理不在graph計畫內；**CPU_ONLY失敗** 僅指先前T1 layer編譯失敗，T16沒有做CPU_ONLY相容性驗證。此次不是WER gate，也不是速度或能耗benchmark。ANE table仍沒有PID，隔離與label/hashbinding提升歸屬可靠性，但不等於硬體PID attestation。

## 同日最終候選：precise graph、固定buffer生命週期與run hash

主agent將最終候選改為 `artifacts/qwen3-asr-1.7b-precise`：encoder使用unfused erf GELU，decoder使用stable-exp SiLU。較早T16證據不再能直接替這個實際graph版本背書，因此依指派僅做一次完整計畫收集與一次隔離batch trace；沒有轉模型、下載或複製權重。

先與plugin agent協調，確認evaluate與CoreML路徑的當前修改已落定。此時runtime使用固定FP32 borrowed-input buffers、複製outputs及explicit close；evaluate在finally關閉backend，另保存cleanup sidecar。Plugin agent確認檔案已寫完且Ruff通過，並凍結這些路徑直到trace後hash核對完成。將10份source/run-input保存快照與SHA，避免trace日後只剩一個無法還原的腳本名字。

使用既有 `inspect_bundle_placement.py` 完成10graph計畫，保存於 `artifacts/validation/precise-placement/`，包含逐graph JSON、完整模型檔hash及collection.log。12,028個有cost算子全部preferred ANE，14,432個unknown全部是const。Frontend27、encoder4,740、每decoder1,030、LM head51個known-cost ANE ops；沒有已知CPU/GPU preferred。與舊T16計畫保存的weight hash比較，10個weight.bin內容全部一致，包括9個改變activation表達式的graph和原本不變的LM head；這是內容一致驗證，並未把它誤稱為inode共享驗證。

主agent確認所有managed ANE/CPU大推理已停止，只做host工作後，啟動唯一一次 `precise-full-asr-isolated` trace。**12:31:24.636–12:32:11.333 America/Phoenix**，90秒上限下實際錄製46.696507秒，兩筆中英文batch smoke成功，target exit0。Explicit-close sidecar回報 `explicit_close_v1 / succeeded`，無cleanup error。

這次實際ANE hardware Prediction有 **623** 筆：frontend21、encoder3、7decoder各77、LM head60，全部具候選對應的compiler-UUID label，沒有generic/unmatched/background Prediction。Prediction interval總和 **2.368766724秒**；10個Compile intervals的26.574534083秒另列，不混成推理時長。此計數與舊T16的599不同，不能硬套舊輸出或舊trace的調用量。

GPU hardware table有183,540筆系統級活動，target PID對應0筆，但4筆process未知；所以只能說沒有觀察到target-attributed GPU活動，不能聲稱絕對GPU零工作。ANE表仍無PID，CoreML summary/raw signpost仍空，因此逐operation的實際CPU時間仍未知。沒有替precise重測CPU_ONLY，也沒有將舊T1編譯失敗当作本候選的負對照。

錄完後重新核對全部source、source快照及graph檔hash，一致；再將原始trace每個檔的相對路徑、大小和SHA排序，生成canonical JSON tree hash。Attribution sidecar同時綁定manifest、placement summary、trace summary/tree、source manifest、evaluation輸出、cleanup報告和command list，生成run fingerprint。完成這些大檔hash讀取後才通知主agent可開始正式latency/silence，並通知plugin解除source凍結。

此候選manifest SHA為 `7ae9d6e0af119b3e1c3f58a13919e1bb42ff2ba39291fe83ac3ef5df3fa8606d`，run fingerprint為 `ca9d4dbdddc7753ce54505601cb923bb77189701aeeb521961fb2a9254a963a0`。Raw trace邏輯檔案總量659,571,728 bytes，比初始200MB預估大，主要伴隨大量系統GPU事件；開始時只讀preflight顯示71GiB可用，沒有清除任何raw evidence或他人cache。

完整說明與hash規約見 `artifacts/telemetry/precise-evidence.md`；gate應使用 `precise-full-asr-isolated-attribution.json` 與對應summary，不能再沿用T16 sidecar。這證明最終precise **batch** 候選的主要graph有實際ANE活動，且此次資源關閉成功；並非streaming驗證、普遍WER/CER非劣性、正式latency比較或節能證明。主agent另有400筆品質結果，其樣本數與統計力限制不因這兩筆smoke或硬體trace而消失。

## 最後metadata/tokenizer修正：避免重編譯，仍綁定新final身份

主agent最後將precise以APFS clone建立 `artifacts/qwen3-asr-1.7b-final`，只按官方 `from_pretrained(fix_mistral_regex=True)` 修正tokenizer pretokenizer與相關manifest metadata。既有equivalence報告記錄390種audio-token counts乘31種language settings（含auto），共12,090組default prompt input IDs完全相同，非pretokenizer部分也相同。此範圍是無任意context的batch；不能延伸成所有stream-prefix等價。

依指派不再inspect十個graph。逐package驗證完整檔案集合與每檔SHA，全部與precise-placement相同，並核對相同macOS build26A428、architecture及coremltools版本。將父計畫JSON原樣作為重用證據，產出 `artifacts/validation/final-placement/summary.json`，明列parent summary/manifest hashes、reuse理由及 `coreml_reinspection_performed=false`。另hash實際final tokenizer、embedding、mel filters和equivalence報告；保存12份source/run-input快照。這樣既避免無意義的CoreML重inspect，也不拿舊manifest identity替新bundle背書。

在主agent確認source/runtime/evaluate凍結且無其他managed推理下，只再錄一次 `final-full-asr-isolated`。**12:59:09.601–12:59:52.574 America/Phoenix**，90秒上限內實際42.972844秒、target exit0、explicit close succeeded。兩筆hypotheses與precise smoke逐字相同。

Final硬體trace有623筆ANE Prediction，全部對應新run的10個compiled UUID labels：frontend21、encoder3、7decoder各77、LM head60，沒有generic/unmatched/background Prediction。Prediction intervals累計2.311256917秒，compile/load另列。GPU有41,500筆系統事件，target-attributed為0，unknown-process為4；仍不聲稱絕對GPU零活動。ANE無PID及CoreML signposts空表的限制維持不變。

錄製後再次核對所有graph、auxiliary runtime assets、source及快照hash，全部相同；保存原始trace tree hash與新run fingerprint，綁定equivalence報告、輸出、cleanup及command list。原始trace邏輯檔案大小219,399,510 bytes；preflight47GiB可用，沒有清cache或刪舊證據。完成export和大檔hash後立即通知主agent可跑最後stream/silence，且本task不再執行推理、不改source或model artifacts。

Final manifest SHA為 `9c705eba4ee069b04b13dcb08ab8abdcfcc1ba889aa925fc5f0c2ce26b28f659`，run fingerprint為 `e43cdb6578ac6ae793fc2d5113c3f4385755162f6c64b85bcc12c800721acf92`。說明見 `artifacts/telemetry/final-evidence.md`，strict gate使用 `final-full-asr-isolated-attribution.json`；後續只promote canonical symlink，保持實際final artifact不變即可保留hashbinding。這仍只是有界batch硬體驗證，普遍品質非劣性、streaming/silence、CPU_ONLY相容性與節能由其各自證據決定。
