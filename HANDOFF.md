# Handoff — 2026-09-12

## 先讀這段

主目標**未完成**。使用者要求 Qwen3-ASR 1.7B 的 Standard ASR main/v0.2 插件，絕大部分神經網路工作在 ANE，品質不明顯退步，且速度及能耗勝過 MLX；後來把要求提高到所有約定 benchmark 都勝出。不能再把研究版或部分通過當成完成。

最新方向：使用者正在整理新的工作方式，允許先把目前工作收束、寫 handoff，再在乾淨上下文繼續。他也問 SenseVoiceSmall 為何出現。已說明它只是一條額外的草稿加速研究，最終 token 仍由 1.7B 驗證；目前沒有成功、沒有接入正式插件。**先等使用者的新工作方式，不要因為這份文件又自動擴大 SenseVoice 支線。**

這輪的問題是追逐了太多小型優化和輔助模型故障，沒有及早限制探索成本。接續工作應先重新決定一個有界假說及驗收方法，再動手。

使用者偏好：Traditional Chinese；Conventional Commits；可讀、可維護的程式碼；主代理自己完成大部分實作。研究可委派 Terra xhigh / Luna max；已完成的研究代理全部結束。工作限制在本 workspace，uv/HF cache 也放這裡。

## 可用的程式與驗證狀態

- 正式 entry point：`std-qwen3asr-ane/1.7b`。預設仍為 serial greedy Qwen3-ASR 1.7B，沒有 SenseVoice、0.6B draft、量化或 speculative decoding 自動切換。
- package：`std_qwen3asr_ane/`，Python 3.12.13，Core ML Tools 9.0。Standard ASR main 由 uv.lock 固定為 `b63bb73bdef9be436fbae182d630452fe3a88f0b`。
- 已加入穩定 `.mlmodelc` 載入、`qwen3-asr-ane compile` 指令及 compiled artifact 存在性檢查。
- streaming session 保留只對完全相同 prompt embeddings 有效的 KV prefix；改變的後綴會覆寫。失敗後重建狀態，避免 NaN 污染。只有 closed final 不可再改，partial 的 `stable_until=0`。
- `stream_max_audio_seconds` 設定預設 180，仍受載入 artifact 的容量及實際 token 預算限制。設定變大不會神奇地擴大舊模型。
- `CoreMLRuntime.prepare_prompt()` 和 `_decode_step(..., all_rows=True)` 是已測試的策略接點。`speculative.py`、`transcript_draft.py` 是測試過的實驗用算法元件，**未被 plugin 呼叫**。
- `PersistentInputModel` 保留固定 FP32 浮點輸入 owners；整數控制輸入保留原 dtype。`close_many()` 可先釋放全部共用 model handles，再等待 input owners 退場。
- 最後收尾：**289 tests passed**；修改檔案 Ruff 通過；Standard ASR CLI compliance exit 0；正式 API 實際辨識成功，文字「甚至出现交易几乎停滞的情况。」。
- 收尾證據：`artifacts/validation/handoff-compliance.log`、`artifacts/evaluation/smoke/handoff-production-smoke.json`。
- 所有本輪啟動的 benchmark / conversion / probe 都已返回完成或失敗，沒有刻意留下推理工作或監控。

最後已存在的主要 checkpoint 是 `8703143 feat: reuse streaming prefixes and persist compiled ANE models`；本 handoff 與後續實驗會另存一個 conventional commit。用 `git log -3 --oneline` 查看最終收尾 commit。

## 模型產物：不要混淆

| 路徑（相對 workspace） | 用途與狀態 |
|---|---|
| `artifacts/qwen3-asr-1.7b` | 仍指向 `qwen3-asr-1.7b-final`；歷史預設，FP16 / T16 / cache1024 / 30秒 |
| `artifacts/qwen3-asr-1.7b-final` | 完成 activation/tokenizer 修正並有既有品質與 ANE trace 的基準 |
| `artifacts/qwen3-asr-1.7b-compiled` | 上述相同權重/圖的穩定 compiled 路徑；目前最適合重現正式插件的快速載入 |
| `artifacts/qwen3-asr-1.7b-context4096[-compiled]` | cache4096 / T16 / 180秒；已完成真實三分鐘串流診斷，尚未廣泛評測或重新做完整硬體 trace |
| `artifacts/qwen3-asr-1.7b-prefill64[-compiled]` | cache1024 / T64；較寬 prefill／verifier 實驗，有兩個 smoke 的 exact-token parity |
| `artifacts/qwen3-asr-0.6b-draft[-compiled]` | 官方 0.6B 的 ANE 草稿模型；可用，但整體加速不足，不是正式插件的辨識模型 |
| `artifacts/probes/` | 單 partition、量化、attention、LM head、state compatibility 探測；不是可交付完整引擎 |
| `artifacts/drafts/sensevoice-small/` | 第三方 SenseVoice 產物與失敗實驗；**不能作為可用草稿 backend** |

`[-compiled]` 表示 raw 和 compiled 兩個目錄均存在。Source 1.7B revision：`7278e1e70fe206f11671096ffdd38061171dd6e5`；0.6B revision：`5eb144179a02acc5e5ba31e748d22b0cf3e303b0`。

產物之間使用 APFS clones 或 hash 證明相同後的 hard links。**不要原地修改任何 weight.bin**，否則可能改到父產物。所有新實驗使用新目錄；既有 build/compile 指令通常拒絕覆寫完整產物。

## 實際進展與尚未達標的數據

### 載入與正式基準

相同 FP16 bundle 首次載入約 **36.335 秒**，下一個獨立 process 經穩定 compiled 路徑載入 **1.539 秒**。這是既有裝置特化快取的重用，**不是首次安裝／首次編譯只需1.5秒**。warm 生成沒有因此變快。

原正式 Standard ASR benchmark（各一輪 warmup，三次重複）:

| 音訊 | ANE | MLX-Audio 0.5.3，同官方 BF16 權重 |
|---|---:|---:|
| 英文 15.051秒 | 2.105秒 | 0.435秒 |
| 中文 4.204秒 | 0.513秒 | 0.137秒 |

新的 prototype 有些降低了 ANE 自己的時間，但尚未在真實完整流程勝過 MLX。尚未加入 MLX 4-bit 等更強 baseline 的比較。

### 能耗已能測到方向，結果目前不利

IOReport 的 CPU／ANE 等 358 個 mJ channels 原始 payload 凍結；CPU residency 仍正常前進，GPU Energy 也會動。替代 channel、不同 subscription dictionary 都沒有解決。無法確定是 macOS27 beta 哪個內部 driver 問題；**零 counter 不是零功耗**。

找到普通 UID 可讀的 SMC `PSTR` 整機功率。30秒 idle / 四CPU負載 / recovery 的平均是 **12.18 / 53.95 / 15.86W**，可反映負載，有過渡延遲。

固定60輪兩音訊，共120次完整辨識、1155.31125秒有用音訊：

| | ANE baseline | MLX BF16 |
|---|---:|---:|
| 總耗時 | 155.152秒 | 34.375秒 |
| 平均 PSTR | 37.88W | 77.99W |
| 整機估计能量 | 5876.40J | 2681.05J |
| J／音訊秒 | 5.086 | 2.321 |

ANE 功率較低但更久，總能耗約 **2.19倍**。這是一次 A/B 的 SMC 整機估計，兩次均 AC 且充電；不是校準過的 wall-plug 能量、不是逐裝置 ANE attribution，也還不是 ABBA 統計驗收。尚未對後續 prototypes 重做能耗比較。

資料：`artifacts/power-v2/asr-baseline-ane-a1/`、`asr-baseline-mlx-b1/`；方法：`research/power-next-steps.md`。

### 三分鐘串流

`artifacts/evaluation/long-stream-180-en/` 串接19個完整 LibriSpeech 句子，加入短間隔，恰好180秒。是已標示的 synthetic diagnostic，不是新自然長音訊語料。

- ANE stream 每10秒更新一次，完整到 closed→done，event/result compliance 通過。
- stream WER **6/436**；ANE batch **7/436**；MLX batch **7/436**。
- stream 快速 feed replay 總計76.767秒，包含18次累積辨識；不能拿它與一次 MLX batch 4.737秒直接作速度比較。
- stream 與 batch 文字不完全相同，但本例 stream 較少一個替換錯誤。
- 無 indefinite rollover、沒有移植好的長期 encoder cache；預設 artifact 仍是30秒，未偷偷切換到較慢的大 cache。

### 品質覆蓋仍不足

上一階段新 held-out 各100筆：EN official 34/2382 vs ANE 33/2382；ZH official 244/3737 vs ANE 246/3737。更嚴格的品質 gate 仍 inconclusive。不要用同一批診斷資料反覆選候選後稱它為新的 held-out，也不要為了宣布完成而放寬既有 gate。

## 已做的優化：避免再重跑沒有收益的項目

| 實驗 | 結果 |
|---|---|
| 四層 decoder T1 | 3.838ms；T16約4.2ms。去掉 padded token 只小幅改善 |
| 四層 T1 palette8 / 6 / 4 | 2.747 / 2.877 / 2.712ms。僅單 partition，沒有完整量化品質評測 |
| 14層 T1 / palette8 | 12.845 / 9.059ms。邊界減少只帶來有限改善 |
| grouped attention / native SDPA / fused QKV+gate-up | 約3.984 / 3.832 / 3.981ms；沒有實質收益，預設都未啟用 |
| compact T1 vocabulary head | 4.789 vs4.838ms，60真實hidden的token一致，提速很小 |
| compact T16 vocabulary head | 每有效token約0.344 vs4.567ms；60個hidden逐token一致，有用但尚不是完整pipeline勝出 |
| 0.6B ANE draft，K7，T16 batch head | EN約1.923秒，ZH0.520秒；仍慢於MLX |
| 再加T64 prefill＋public state copy | EN約1.716秒，ZH0.512秒；copy本身約63–67ms |
| 真實相容state直接分享 | 四層T16→T1的5個後續位置：output max_abs=0、每個KV array相同。後續補完7-partition T64→T16完整模型測試，詳見下段；仍未整合正式插件 |
| ground-truth transcript oracle＋T64 target＋T16 head | EN0.400秒、ZH0.127秒。**排除了取得草稿的成本，是明確標示的理想下界，不是實際引擎性能、不可宣傳勝過MLX** |

speculative greedy 邏輯使用 held target token：T16一次驗證最多15個draft tokens。只接受target同意的token；mismatch後以target correction覆寫位置；完整接受時補上draft尚未消耗的最後token。79個算法測試及兩個真實smoke的exact-token parity通過。`TranscriptDraft` 另用suffix anchors做有界resync，插入、刪除、重複、不相關和空草稿都不會改變target輸出。

證據主要在 `artifacts/evaluation/smoke/*speculative*.jsonl`、`oracle-transcript-t64-k15.jsonl` 及 `artifacts/probes/*benchmark.json`。小模型 multifunction / enumerated-state 載入失敗也保留在 probes；不要把這些失敗誤推論成所有 Core ML state sharing 都不支援。

### 收束後補完的既有實驗：完整 shared-state 對照

自動目標續行時，只補完已準備好的 state-sharing 驗證，沒有繼續 SenseVoice 或改動推理程式。使用同一組既有T64 prefill、T16 generation、0.6B ANE draft及T16 vocabulary head；先跑share、再跑相同copy控制，各一輪warmup＋三次測量、兩音訊，**合計16次完整推理全部與serial 1.7B逐token一致**，兩程序exit0、明確close完成。

| 完整流程中位数 | Shared state | Explicit copy | 減少 |
|---|---:|---:|---:|
| 英文 | 1.6903秒 | 1.7565秒 | 66.2ms（約3.8%） |
| 中文 | 0.4528秒 | 0.5221秒 | 69.3ms（約13.3%） |

Copy本身約62.7–64.4ms；兩輪同時量到的serial基準接近（EN約2.10秒、ZH約0.512秒）。這是兩段smoke、一次順序A/B的有界證據，**仍遠慢於MLX，沒有新的能耗或廣泛品質結論**。也不是對所有Core ML版本／shape／artifact宣告state互通。

原始檔：`artifacts/evaluation/smoke/shared-state-full-k7.jsonl`、`copied-state-full-k7-control.jsonl`；含來源hash的摘要：`shared-state-full-comparison.json`。腳本是已存在的`experiments/benchmark_speculative.py`，只切換`--state-transfer share|copy`。這項已備妥工作現在已完成驗證，接續仍依使用者的新workflow決定下一步。

## SenseVoiceSmall 支線：現況與問題

起因：0.6B Qwen draft 仍有28層，相同16Q/8KV heads，在ANE上逐token成本過高。嘗試改用非自回歸 CTC 模型一次提出整段文字，由1.7B逐token驗證。它是可能的加速輔助，並不完成或替代1.7B移植本身。

已下載：`FluidInference/sensevoice-small-coreml`，revision `0e0bf30bfc6836f182ccd1d89984df919c949e26`，INT8 encoder、FP32 CPU frontend、vocab。只支援 **zh/yue/en/ja/ko**，不是30或50+語言。upstream source metadata revision `3847d57b6bdf2dd8875cb1508d2af43d80a16bf7`。模型遵循自訂 FunASR model license；原始名稱、來源及條款保存在產物目錄，不能當成本專案 Apache-2 權重。

有用的診斷：

- CPU `kaldi-native-fbank`＋LFR＋CMVN 和公開 Core ML frontend 共同範圍 max_abs **4.53e-5**、relative L2 **2.11e-6**。
- 公開 frontend 固定右補7個frame後stride6，沒有裁到upstream的ceil(T/6)，部分長度多一個LFR尾列。已從MIL確認、明確記錄，沒有假裝完整feature shapes一致。
- INT8 encoder實際輸出全NaN（EN bucket256，shape `[1,260,25055]`，6,514,300個nonfinite）。
- 預期compute plan：2207個op preferred CPU，2928個unknown，全部cost缺失；沒有證明此host實際在ANE跑了它。`CPU_AND_NE`本身不能保證ANE。
- infrequent-reshape hint、finite mask、較大的LayerNorm epsilon、repeat padding均未修好。
- 只改MIL成static256卻保留不相容metadata的實驗，在獨立process以NSException abort：`There is no function in the program library for the default shapes.`
- 另用公開compiler產生一致fixed I/O metadata，再帶入原始MIL/weights的fixed128/256/512實驗能load，仍全NaN。

**根因未解。** 不要宣稱已定位為某個LayerNorm、mask或driver bug。`experiments/sensevoice_draft.py`、`validate_sensevoice_draft.py`、`patch_sensevoice_mask.py`、`specialize_sensevoice.py` 都是未成功的研究程式，不在正式plugin路徑。失敗產物不能被artifact availability誤當成numerical validation。不要直接再投入大段时间修這個第三方模型；先依使用者新workflow決定值不值得。

`research/fast-draft-options.md` 記錄其他候選（含Omnilingual CTC）但尚未下載／整合它們。沒有使用GPU草稿；這輪所有真實神經網路draft實驗也限制CPU_AND_NE。

## 最短重現方式

```sh
export UV_CACHE_DIR="$PWD/.cache/uv"
export HF_HOME="$PWD/.cache/huggingface"
uv sync --project std_qwen3asr_ane --frozen --group convert --group draft
std_qwen3asr_ane/.venv/bin/pytest -q std_qwen3asr_ane/tests
std_qwen3asr_ane/.venv/bin/standard-asr compliance run std-qwen3asr-ane/1.7b
std_qwen3asr_ane/.venv/bin/python -m std_qwen3asr_ane.cli transcribe artifacts/evaluation/smoke/qwen_official_zh.wav --model-dir artifacts/qwen3-asr-1.7b-compiled
```

`draft` group只是SenseVoice前處理實驗需要的 `kaldi-native-fbank==1.22.3`，不是基本推理依賴。不要無意用不帶groups的sync移除正在使用的conversion tools。

三分鐘重現（用新的output檔名）：

```sh
std_qwen3asr_ane/.venv/bin/python experiments/verify_streaming_runtime.py \
  --model-dir artifacts/qwen3-asr-1.7b-context4096-compiled \
  --audio artifacts/evaluation/long-stream-180-en/audio.wav \
  --output artifacts/evaluation/long-stream-180-en/recheck.json \
  --chunk-seconds 10 --max-new-tokens 768
```

Core ML/IOKit在Codex sandbox內可能不能連到系統service；既有工具都以普通使用者、`require_escalated`離開sandbox執行，沒有sudo、改系統設定或要求密碼。`.git`在sandbox是唯讀，git add/commit也需要同樣的正常工具核准。

硬體benchmark必須序列、沒有其他模型或conversion負載並行。使用`experiments/benchmark_energy.py`做等工作量PSTR比較；不要拿有instrumentation、oracle、cold compile或多次stream replay的數據代替普通batch latency。

磁碟曾多次大幅變動；收尾約163GiB可用。沒有刪除workspace外快取、snapshot或其他應用資料。不要推論為何空間恢復。

## 接續前的必要判斷

1. 先讀使用者即將提供的新workflow；本handoff不是要求繼續目前研究策略。
2. 確認比較範圍與具體benchmark，尤其MLX BF16／4-bit、短句／長音訊／streaming及能耗條件。不要偷偷挑較弱baseline或排除不利case。
3. 選一條有界路線。相容state、小型phase優化、完整8-bit候選、CTC verifier都是不同問題，避免一次同時改全部。
4. 候選必須通過真實ASR品質、完整latency、能耗及實際ANE證據，才能改預設產物。單partition更快、oracle更快或flag寫CPU_AND_NE都不夠。
5. 主目標仍active／unfinished；本輪handoff完成不代表主目標完成。

完整時間順序與更早的activation修正：`research/technical-blog.md`。較早的品質/硬體結果：`research/results.md`、`artifacts/validation/final-evidence.json`。
