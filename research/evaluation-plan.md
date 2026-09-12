# 可重現的 ASR 品質與 latency 評估

日期：2026-09-12。這份文件描述已建立的 runner、資料來源與下一階段驗收方法；任何沒有實際採集的功耗數據均保持空值。

## 已完成與限制

- `experiments/prepare_eval.py` 已下載 Qwen 官方 English / Chinese WAV，共約 2.31 MB；`artifacts/evaluation/smoke/manifest.jsonl` 可直接作為 runner 輸入，`provenance.json` 保存來源、版本與 hash。
- `experiments/evaluate.py` 支援本地 CoreML runtime（預設 `cpu_and_ne`）與官方 `Qwen3ASRModel` CPU FP32 eager baseline。模型必須預先存在於 `--model-dir`；runner 不負責模型下載。
- 同一份 manifest 可依序餵給兩個 backend。它將每筆 warmup／measured attempt 寫入 JSONL，保留成功、失敗、空字串、hypothesis、reference、language、耗時、RTF、edit counts。模型載入失敗也會為每筆樣本留下 error，並以非零 exit code 結束。
- 本次已驗證 normalization、edit counts、fixture 解碼及重採樣、fake backend 完整 runner、repeat 聚合、load-error 保存、paired bootstrap 和失敗拒絕邏輯；fake backend 測試不作為模型效能證據。
- 首次真實 CoreML run 已由主實驗執行，結果在 `artifacts/evaluation/smoke/coreml-first.jsonl`。英語刪掉開頭 `Mm`，WER=1/38≈2.63%；中文 CER=0/13。這是兩筆 smoke，不是 WER 達標結論。
- 官方 CPU FP32 smoke 亦已完成；`artifacts/evaluation/smoke/paired-first.json` 是兩份結果的配對報告，單語只有一筆故沒有可用單語 CI。
- 後續已完成 **100 English + 100 Chinese** 固定測試 subset 的準備與完整檔案 hash／解碼驗證；見下節。資料準備完成不代表這 200 筆模型品質已通過 gate。
- MLX baseline 尚未加入或安裝依賴。之後可新增 lazy adapter，但必須保留相同音訊、prompt、max tokens、normalizer 和結果 schema，並鎖定 MLX 模型量化版本；不能拿網路上的 MLX WER 直接作配對 baseline。

## 官方 smoke references 的來源

使用 [Qwen 官方 forced-aligner example](https://github.com/QwenLM/Qwen3-ASR/blob/7c6daf77a2421100f5fb066495372c00129d39ff/examples/example_qwen3_forced_aligner.py) 的 `URL_EN`／`URL_ZH` 和 `TEXT_EN`／`TEXT_ZH`。工具以 AST 讀取字串 literal，**不執行下載的 Python code**；reference 是作者提供的範例逐字稿，沒有宣稱另做人工 ground-truth 校對。

| Fixture | 長度 | 原始音訊 SHA-256 | 用途 |
|---|---:|---|---|
| [English WAV](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav) | 15.05125 s | `f9b4440ac8393e47c14a6240e9739dea09b645bb1592b8f2dd48feb9666cea7f` | colloquial speech、較長 audio prefill |
| [Chinese WAV](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_zh.wav) | 4.2039375 s | `46dbc998c9d1d48111267c40741dd3200f2e5bcf4075f8c4c97f4451160dce50` | 中文 CER、語言識別 |

程式碼 repo 為 Apache-2.0，但 example 未另列音訊授權；provenance 清楚保留此差異。音訊 URL 可變，實驗以下載到的 bytes hash 識別；runner 會驗 manifest 的 `audio_sha256`，避免後續被替換。

## 命令

以下從 workspace 根目錄執行。使用 project 的 convert group 取得官方 baseline 與 jiwer；沒有變更 pyproject。若 uv cache 權限受限，可設 `UV_CACHE_DIR="$PWD/.cache/uv"`。

```bash
# 已有 smoke 目錄時不重複下載；工具拒絕覆寫既有 manifest/provenance。
uv run --project std_qwen3asr_ane --group convert python experiments/prepare_eval.py \
  --dataset smoke --output artifacts/evaluation/smoke

# 同一 manifest，分開跑、避免同時占用 memory/compute。
uv run --project std_qwen3asr_ane --group convert python experiments/evaluate.py \
  --backend coreml --manifest artifacts/evaluation/smoke/manifest.jsonl \
  --model-dir artifacts/qwen3-asr-1.7b \
  --output artifacts/evaluation/coreml-warm.jsonl --warmups 1 --repeats 3

uv run --project std_qwen3asr_ane --group convert python experiments/evaluate.py \
  --backend official --manifest artifacts/evaluation/smoke/manifest.jsonl \
  --model-dir /absolute/path/to/local/Qwen3-ASR-1.7B \
  --output artifacts/evaluation/official-fp32.jsonl --warmups 0 --repeats 1

uv run --project std_qwen3asr_ane --group convert python experiments/evaluate.py \
  --compare artifacts/evaluation/official-fp32.jsonl artifacts/evaluation/coreml-warm.jsonl \
  --output artifacts/evaluation/paired-smoke.json --bootstrap-samples 2000 --seed 20260912
```

`--language-mode auto` 是預設。`--language-mode manifest` 才以 manifest 的 BCP-47 language code 強制解碼語言；兩個 backend 的 mode 必須相同。`--max-new-tokens` 預設 256；長音訊應預先選合理上限並對兩個 backend 一致，不應在看到輸出後只改 candidate。`--torch-threads` 可固定 CPU baseline threads。

## Manifest 與結果契約

一行一個 JSON object，必要欄位如下；相對 audio path 以 manifest 所在目錄為基準，不以當前 shell 目錄為基準。

```json
{"id":"sample-001","audio_path":"audio/sample-001.wav","reference":"Published reference text","language":"en","split":"test"}
```

選填 `audio_sha256`、`source_url`、`speaker_id`、`dataset_revision`。空 reference 是合法的靜音案例；空 audio、非有限 samples、缺檔與 hash 不符都要留下 error。重複／空 sample ID 和非字串 reference 屬 manifest 格式錯誤，整個 run 在開始前失敗。

音訊用 libsndfile 解碼成 float32，聲道算術平均成 mono，必要時以 SciPy polyphase resampling 到 16 kHz。原始檔案 hash 被保存；兩個 backend 共用相同 preprocessing。

`<output>.summary.json` 保存 package versions、OS／architecture、model load seconds、manifest hash、model manifest/config（若存在）及其 hash、coverage、failed IDs、非決定性 repeats、latency 與品質摘要。它沒有重新 hash 全部數 GB 權重；嚴格發布 benchmark 應另以 build manifest 的完整 artifact hashes 驗證權重。`cpu_and_ne` 只是允許 CPU+ANE 的配置，不是 ANE placement 證據。

## WER／CER 定義

固定 normalizer ID：`nfkc_casefold_unicode_punctuation_to_space_v1`。

1. Unicode NFKC、casefold。
2. Unicode category `P*` 標點替換成空白；合併多餘空白。
3. WER 使用空白分詞；CER 另外移除所有空白再按 Unicode code point 比較。
4. 不做數字展開、簡繁轉換、同義詞替換、英文 contraction 特例或中文斷詞。因此 `wasn't` 變成兩個 units，這套 WER 不保證與官方 leaderboard normalizer 一樣。

以 jiwer edit alignment 累加 `(S+D+I) / (H+S+D)`，不是平均每句 error rate。只用 measured repeat 0 做一次品質統計，重複跑不會人為擴大樣本數。所有 measured repeats 用於 latency，也檢查 hypothesis 是否改變。

中文、粵語、日語、泰語從 aggregate WER 排除，按 CER 評估；每筆仍保留 diagnostic WER counts，不能把中文一整句算成一詞的 WER 拿來宣傳。摘要同時按 language 分組。無 language 的文字仍算 WER；所以正式 corpus 必須提供 language。

空 reference 產生的 hallucination 保留 insertion counts；單筆 denominator 為 0 時 rate=null。corpus 中它的 insertion 仍會加入總 errors；若整個 corpus 都是靜音，rate=null，應另報 hallucinated characters／samples。失敗樣本不偽造空 hypothesis，summary 必须以 coverage 和 failed IDs 顯示，不能只看成功子集得出品質結論。

## 配對統計

`--compare baseline.jsonl candidate.jsonl` 檢查：sample IDs 完全相同，reference、language、audio hash、forced-language mode、max tokens、normalizer 相同，所有 measured attempts 成功且 repeats 決定性一致。若不滿足，輸出帶 `problems` 的 invalid report，exit 1，不產生品質 confidence interval。

通過時以 utterance 為單位，有放回地抽相同的 paired indices，重算 corpus micro-average delta。報告方向固定 **candidate minus baseline**，正值是退化；提供 percentile 95% CI、seed、resample 數與每語言結果。單語只有一筆時 CI=null；少於 30 筆標為 exploratory。全 corpus WER 仍排除上述 character languages。抽到全空 reference 的 resample 不計 rate，並記錄被排除的 resample 數。

這不是 speaker-cluster bootstrap。準備好的 English subset 已涵蓋 40 speakers，但各 speaker 有 2–3 筆相關 utterances，正式宣稱前仍應擴充 cluster bootstrap。FLEURS metadata 沒有 speaker ID，無法聲稱已控制 speaker 群聚。兩個相同結果檔都可能只是中斷後的相同前綴，所以正式比較還須確認兩份 summary 的 expected count 對上原始 manifest；pair checker 本身只能檢驗輸入 rows。

「不明顯影響 WER」的 non-inferiority margin 應在較大實驗前先決定，並同時報 absolute percentage-point 和 relative delta；本工具不替使用者暗設門檻。两筆 smoke 沒有能力估計這個 margin。

## 已準備的 100 English + 100 Chinese corpus

`experiments/prepare_corpus.py` 使用臨時 **PyArrow 21.0.0** 直接讀官方版本化 parquet；不需要 datasets／torchcodec，也沒有修改 engine pyproject。完整來源存在 `artifacts/evaluation/sources`，下載量 EN 350,452,636 bytes、ZH 695,674,033 bytes；每份都核對 HF LFS 公布的 SHA-256。每次下載有 60 秒 socket timeout 和 10 分鐘整體上限，不會無限 retry。

| Subset manifest | 已選音訊 | 分散情況 | 時長 | 選取前排除 |
|---|---:|---|---|---|
| `artifacts/evaluation/librispeech-balanced-100/manifest.jsonl` | 100 | 40 speakers，每人 2–3 筆 | 合计 808.015125 s；最長 28.41 s | 2620 test-clean 中 9 筆超過 30 s |
| `artifacts/evaluation/fleurs-zh-balanced-100/manifest.jsonl` | 100 | 兩個 gender labels 各 50；**無公開 speaker ID，未驗證 speaker 分散** | 合計 1090.56 s；最長 21.08 s | 945 Mandarin test 中 1 筆超過 30 s |

選樣先排除非正數或超過 30 秒、空 reference、過大音訊，解碼／欄位異常則直接失敗並保留詳細 row error。每個 group 內按 `SHA256(seed:source_row:source_id)` 排序，再依 stable-hash 排好的 groups 輪流取樣，seed 固定 20260912。**不同 count 的 manifests 是 nested prefixes**，不受模型輸出影響；英文以 speaker 分組，中文僅以可用 gender 欄位分組。這是有 duration／group balancing 偏差的工程評估 subset，不是全 corpus 的無偏 WER 估計，也不應用 test subset 選量化超參數。

```bash
# 已執行成功。若重建，選新的 output；已核對的 sources parquet 可重用。
uv run --project std_qwen3asr_ane --group convert --with 'pyarrow>=20,<22' \
  python experiments/prepare_corpus.py --dataset librispeech --count 100 \
  --output artifacts/evaluation/librispeech-balanced-100

uv run --project std_qwen3asr_ane --group convert --with 'pyarrow>=20,<22' \
  python experiments/prepare_corpus.py --dataset fleurs --count 100 \
  --output artifacts/evaluation/fleurs-zh-balanced-100
```

固定版本與 whole-parquet hash：

- [LibriSpeech test-clean parquet](https://huggingface.co/datasets/openslr/librispeech_asr/blob/71cacbfb7e2354c4226d01e70d77d5fca3d04ba1/clean/test/0000.parquet)：revision `71cacbfb7e2354c4226d01e70d77d5fca3d04ba1`，SHA-256 `7113aa4c3cf963fb54697145719a7725f984c8836d1c494a554cbb9f1a017df0`。
- [FLEURS Mandarin test parquet](https://huggingface.co/datasets/google/fleurs/blob/70bb2e84b976b7e960aa89f1c648e09c59f894dd/parquet-data/cmn_hans_cn/test-00000-of-00001.parquet)：revision `70bb2e84b976b7e960aa89f1c648e09c59f894dd`，SHA-256 `87c0aebbe183f3a36ac87b5c3421b6ab57036824744ff695029a3f858e7622fd`。

[OpenSLR LibriSpeech HF mirror](https://huggingface.co/datasets/openslr/librispeech_asr) 和 [Google FLEURS](https://huggingface.co/datasets/google/fleurs) 均標示 CC-BY-4.0。English reference 使用 `text`，Chinese 使用 `raw_transcription`，再由 runner 統一 normalize。HF viewer 的 FLEURS rows API 本次回 500，因此最終流程不依賴 signed viewer audio URLs。

每筆保存 source row／ID、audio hash、dataset revision、speaker ID（若有）、gender 和 sampling group。兩份 `provenance.json` 均 `complete=true`，包括全 source row 數、所有 exclusions、selected rows／group counts、reference field、授權和來源。失敗時保存 `complete=false`、error 並 exit 1；部分資料不算準備成功。

`prepare_eval.py` 仍保留原始 optional datasets streaming/prefix 模式供其他 config 研究；本次採集未使用該模式。正式 200 筆以本節的 balanced manifests 為準。

## Latency、ANE 與 energy 分別驗證

Runner 的 wall clock 包住同步 `transcribe`，包含 mel／encoder／prefill／decode，排除 model load、檔案解碼和 resampling。`--warmups 1` 為**每筆**一輪，不只第一筆；warmup 記錄保留但不併入 latency。`--warmups 0` 包含第一次 predict 的可能編譯／lazy initialization，因此第一輪結果不是 warm latency。run 一律 serial；相同 OS、thermal／power mode、其他 workload、max tokens 下比較。

RTF=transcribe seconds/audio seconds，低越好。corpus RTF 以總 inference time/總 audio time 計算；median/p95 是每 attempt 的秒數，不是混合不同長度後的速度公平比較。模型內分段 timings 另外保存，可辨認 prefill 是否主導。

Device placement 應由 build/diagnostic 工具另存 MLComputePlan 和 runtime profiling；功耗需另外使用實機可讀 telemetry、明確 idle subtraction 與時間窗。此 runner `energy.measured=false`、`joules=null`，不把 latency 或 CPU_AND_NE 設定換算成省電百分比。

## Chronological record

1. 讀目前 runtime／plugin contract 與官方 Qwen inference API，確認 CPU FP32 baseline 及 lazy CoreML 呼叫。
2. 找到官方 forced-aligner example 同時提供音訊與 reference；固定 revision，避免用模型 hypothesis 冒充 GT。
3. 建立 runner 的結果契約、edit-count 聚合與 paired bootstrap；建立 bounded fixture/corpus 準備器。
4. 下載 1 EN + 1 ZH 並驗 WAV headers、bytes hashes；主實驗立即用 manifest 跑完整 CoreML，兩筆均成功。
5. 用無模型測試檢查 error persistence、repeat／language scoring 和配對拒絕條件；lint/format 通過。
6. 查核大 corpus revision／parquet LFS hashes，先固定 metadata-balanced 選樣規則；臨時加入 PyArrow，完整下載並核對兩份來源。
7. 準備 100 English（40 speakers）+100 Mandarin（speaker metadata 缺失，僅 gender balancing）；核驗全部選中音訊 bytes hash、解碼、duration 與 reference，兩份 provenance 完整。模型大 corpus WER gate 和 energy 仍須由實際推理／量測完成。
