# Qwen3-ASR CoreML／ANE prior art 查核

查核日期：2026-09-12。範圍：作者模型卡、公開 repository metadata 與原始碼；沒有下載模型權重、沒有在本機重跑第三方 benchmark。下列效能與品質數字均是作者報告，不能當成本專案量測。`past_chat` 沒有用作證據。

## 結論

**Qwen3-ASR-1.7B 的 CoreML 移植已存在；本次未找到已公開、可重現、以 ANE 承擔絕大部分推理且維持品質的 1.7B 完整 pipeline。** 因此不能宣稱第一個 Qwen3-ASR CoreML port；可以把研究問題定義為「可驗證的 1.7B ANE 主導推理、品質回歸與 Standard ASR 整合」。這是有限範圍搜尋的結論，不是不存在性證明。

0.6B 已有拆分 decoder、固定 token batch、MLState 的 ANE runtime 可參考；1.7B 已公開版本明言 decoder 使用 FP32／GPU。將 0.6B runtime 換上 1.7B 權重不是直接替換：hidden size、MLP 寬度、encoder 深度、編譯資源和激活範圍都要重驗。

| 工作 | 實際模型 | 已查證執行路徑 | 與本專案關係 |
|---|---|---|---|
| aoiandroid / weiren119 HF 套件 | 原版 Qwen3-ASR **1.7B** | CoreML encoder；FP32 decoder **GPU-only** | 已有可下載 1.7B CoreML artifact，非 ANE 成功案例 |
| aufklarer HF + soniqo/speech-swift | CoreML 預設 **0.6B**；MLX 可選 1.7B | 分成兩段的 decoder 預設 CPU+ANE；encoder 預設 `.all` | 最接近可借鑑的 ANE decoder runtime |
| Reza2kn/mega-asr-coreml | **1.7B 衍生 fine-tune** Mega-ASR | ONNX encoder + CoreML FP32 decoder；作者報 ANE 拒編譯 | 可讀 ANEMLL conversion bridge，但不是原版 1.7B 的 ANE 實績 |
| FluidAudio / mobius | **0.6B** | 作者報 encoder 100% ANE，decoder GPU | encoder layout 改寫與失敗紀錄；FluidAudio 已移除 backend |

證據分別見以下章節。CoreML、Apple Silicon、MLState、`.cpuAndNeuralEngine` 均不能單獨證明全圖在 ANE；最後一項仍容許 CPU fallback。

## 1. aoiandroid 與 weiren119：相同模型卡，1.7B GPU decoder

固定版本：[aoiandroid](https://huggingface.co/aoiandroid/Qwen3-ASR-1.7B-CoreML/blob/3fa66508c26b2668136e3a3d2b1d6ad789e16467/README.md)、[weiren119](https://huggingface.co/weiren119/Qwen3-ASR-1.7B-CoreML/blob/4125879a682d11922678927636eef85d38b27a82/README.md)。兩份 README bytes 的 SHA-256 相同：`a976f202b1a5d35d1d9f047b975bd1336ceae3bcb2de9cf3084e7575c71d34ff`。這只證明說明文件相同，未比較數 GB 權重，不能推論兩者獨立完成或判定原作者。

模型卡列出 FP16 encoder（INT8 weight）、FP32 decoder（mixed INT8）、獨立 FP16 embeddings。關鍵短引文：**“Decoder runs on GPU only”**。作者稱 decoder 第 25 層 hidden magnitude 可達 10,876，直接平方約 118 million，超過 FP16 65,504；使用 `[x,-x]` LayerNorm 式 RMSNorm 與預先算好的 RoPE sin/cos，仍保留 FP32 compute。20 筆臺灣華語樣本的 CER 為 PyTorch 0.2009、CoreML 0.2064；規模不足以支持廣泛 WER 無退化聲明。[模型卡](https://huggingface.co/aoiandroid/Qwen3-ASR-1.7B-CoreML/blob/3fa66508c26b2668136e3a3d2b1d6ad789e16467/README.md)

兩個 HF tree 均未列 `.py`。README 宣稱 source repository 有轉換碼，卻沒有提供實際連結；本次未定位到可驗證的對應 conversion repo。授權 metadata：Apache-2.0；這不等於已找到所有 conversion dependency 的授權。[aoiandroid tree](https://huggingface.co/aoiandroid/Qwen3-ASR-1.7B-CoreML/tree/3fa66508c26b2668136e3a3d2b1d6ad789e16467)、[weiren119 tree](https://huggingface.co/weiren119/Qwen3-ASR-1.7B-CoreML/tree/4125879a682d11922678927636eef85d38b27a82)

## 2. aufklarer 與 qwen3-asr-swift：查 current source，別只看 README

目前可用 runtime repo 是 [soniqo/speech-swift](https://github.com/soniqo/speech-swift)，本次固定 commit `d655076badd143f99c9ce19642fbea8b643ccc0b`。搜尋中的 `ivan-digital/qwen3-asr-swift` 舊名和各 fork 不能當不同的 ANE 成果。

HF [config.json](https://huggingface.co/aufklarer/Qwen3-ASR-CoreML/blob/8c6bc87b87856930b435550e94ce47de710ce4ed/config.json) 明列 `source_model=Qwen/Qwen3-ASR-0.6B`、hidden 1024、MLP 3072、28 layers、KV context 1024、兩段各 14 layers、固定 `enumerated_t=[128]`。其 HF 模型卡仍描述單一 decoder，但 tree 已是 `decoder_part1` / `decoder_part2`；引用新架構應以 config 與 runtime 為準。

[CoreMLTextDecoder.swift](https://github.com/soniqo/speech-swift/blob/d655076badd143f99c9ce19642fbea8b643ccc0b/Sources/Qwen3ASR/CoreMLTextDecoder.swift#L12) 的可重用設計：

- 每段 decoder 各自持有 `MLState`，中間只傳 hidden states。
- 固定 T=128，prefill 一次處理一批；單 token decode 用保留 cache slots 做 scratch padding，attention mask 排除 scratch。
- runtime 預設 `.cpuAndNeuralEngine`；註解稱整體層數下 EnumeratedShapes 無法 ANE 編譯，因此採固定 T。
- cache 1024 並非全可用：127 slots 保留 scratch，所以真正可寫位置上限是 897 個 slots。
- MLMultiArray 輸出必須按 dtype 與 strides 取值；FP16／ANE padding 不能當 contiguous Float32。

[CoreMLASRModel.swift L28–42](https://github.com/soniqo/speech-swift/blob/d655076badd143f99c9ce19642fbea8b643ccc0b/Sources/Qwen3ASR/CoreMLASRModel.swift#L28) 的 encoder 預設 `.all`，註解說 30 秒固定圖在多數 Mac 上 GPU 效果佳；decoder 才預設 CPU+ANE。故 HF 的 full Neural Engine 宣稱不能直接當目前整套預設的 device placement 實測。`transcribeWithoutMLX` 才是避免 MLXArray 計算的路徑，另一條 CoreML API 還有 Metal round-trip。[推理文件](https://github.com/soniqo/speech-swift/blob/d655076badd143f99c9ce19642fbea8b643ccc0b/docs/inference/qwen3-asr-inference.md)

品質方面，作者發現舊 encoder 對 padded mel 做 unmasked global attention，test-clean n=200 WER 24.88%；重建 chunk/block mask 後 3.02%，同組 0.6B MLX INT8 為 1.82%，1.7B MLX INT8 為 1.52%。其新 encoder 描述為 100 mel frames 做 convolution chunk、800 frames 做 transformer attention window；長度 input/output 明確遮掉 padding。這提示「能轉錄」遠不足以驗收。[同一推理文件](https://github.com/soniqo/speech-swift/blob/d655076badd143f99c9ce19642fbea8b643ccc0b/docs/inference/qwen3-asr-inference.md)

本次在 speech-swift recursive tree 與 HF tree 未找到對應 `.py` conversion scripts。可重用 runtime／設計，但**尚非完整可重跑的轉換 pipeline**。runtime [LICENSE](https://github.com/soniqo/speech-swift/blob/d655076badd143f99c9ce19642fbea8b643ccc0b/LICENSE) 與 HF metadata 均 Apache-2.0。

## 3. Mega-ASR CoreML：有轉換碼，但作者明確承認 ANE 失敗

[固定 HF revision](https://huggingface.co/Reza2kn/mega-asr-coreml/tree/f67c7dda01d7e10412d75a733548f8a97e9b24da)。這是 `zhifeixie/Mega-ASR`，模型樹為 Qwen3-ASR-1.7B 的 fine-tune，不能把其品質等同原版 Qwen checkpoint。模型卡記錄 `.CPU_AND_NE` 無法 load、ANE compiler error，`.ALL` 使用 GPU；encoder 尚是 ONNX FP32。FP16 產生 NaN，conversion 選 FLOAT32。[模型卡](https://huggingface.co/Reza2kn/mega-asr-coreml/blob/f67c7dda01d7e10412d75a733548f8a97e9b24da/README.md)

可讀來源：

- [convert_embeds.py](https://huggingface.co/Reza2kn/mega-asr-coreml/blob/f67c7dda01d7e10412d75a733548f8a97e9b24da/convert_embeds.py)：embedding-input ANEMLL conversion。
- [convert_embeds_mixed.py](https://huggingface.co/Reza2kn/mega-asr-coreml/blob/f67c7dda01d7e10412d75a733548f8a97e9b24da/convert_embeds_mixed.py)：把 ANEMLL Qwen forward 擴成接受 audio/text embeddings；attention LUT8、MLP LUT4；`ct.convert` 明設 FLOAT32；16 個 logits shards。
- [inference_asr.py](https://huggingface.co/Reza2kn/mega-asr-coreml/blob/f67c7dda01d7e10412d75a733548f8a97e9b24da/inference_asr.py)：mel、ONNX encoder、prompt embedding scatter、stateful CoreML decode 的串接參考。

腳本硬編 `/tmp/Anemll` 到 Python import path，還依賴 ANEMLL monkey-patch；不是乾淨 uv package，也未在本次定位被依賴 ANEMLL 的精確 commit。可借鑑 embedding bridge 和 sharded LM head，採用前要 pin dependency 並移除環境假設。HF metadata 為 Apache-2.0；ANEMLL code 應另查 LICENSE 與 attribution。

## 4. FluidAudio／mobius：0.6B encoder 的 ANE 證據及歷史陷阱

[FluidAudio PR #410](https://github.com/FluidInference/FluidAudio/pull/410)（2026-03-22 merged）作者報告把 18 層 encoder 的 linear 改 1×1 Conv2D、採 `(B,C,1,S)` 和 per-head einsum；M4 Max encoder median 11.61→7.60 ms、100% ANE scheduling。其 18 layers／896 hidden／14 heads 對應 0.6B；同一 PR 明確指出 decoder 留在 GPU。10 筆 test-clean 中 9 筆逐字相同是小樣本結果，不足以證明廣域品質等價。

[FluidAudio PR #676](https://github.com/FluidInference/FluidAudio/pull/676)（2026-06-10 merged）移除了 Qwen3 library backend、CLI、tests、registry、smoke workflow；作者 6 月 13 日回答移除原因是 **“Lack of popularity”**。不能把移除推論為 ANE 技術上不可能，也不能使用過時 API 文件宣稱目前 main 支援 Qwen3。

可重用 conversion repo 是 [FluidInference/mobius](https://github.com/FluidInference/mobius/tree/4040a39f760290bb1d43a72dc82e5894f27b0f5c/models/stt/qwen3-asr-0.6b/coreml)，固定 commit `4040a39f760290bb1d43a72dc82e5894f27b0f5c`：

| 檔案 | 已讀證據／限制 |
|---|---|
| [convert-qwen3-asr.py](https://github.com/FluidInference/mobius/blob/4040a39f760290bb1d43a72dc82e5894f27b0f5c/models/stt/qwen3-asr-0.6b/coreml/convert-qwen3-asr.py#L247) | encoder trace 固定 100 mel frames；另輸出 embedding、LM head、decoder stack/prefill；後三者選擇 FP32 |
| [individual_components.py](https://github.com/FluidInference/mobius/blob/4040a39f760290bb1d43a72dc82e5894f27b0f5c/models/stt/qwen3-asr-0.6b/coreml/individual_components.py#L157) | FullWrapper 把單一 chunk 當全 attention；沒有因此實現跨 8 個 conv chunks 的 inference attention window |
| [convert_stateful_decoder.py](https://github.com/FluidInference/mobius/blob/4040a39f760290bb1d43a72dc82e5894f27b0f5c/models/stt/qwen3-asr-0.6b/coreml/convert_stateful_decoder.py) | 28 層、56 個 FP16 KV states、hidden size 1024 寫死；目前 conversion 選 FLOAT16，與早期報告「FP32 必要」並不完全一致 |
| [convert_decoder_fused.py](https://github.com/FluidInference/mobius/blob/4040a39f760290bb1d43a72dc82e5894f27b0f5c/models/stt/qwen3-asr-0.6b/coreml/convert_decoder_fused.py) | stateful decoder 合併 final RMSNorm／LM head；仍寫死 0.6B dimensions |
| [QWEN3_ASR_COREML.md](https://github.com/FluidInference/mobius/blob/4040a39f760290bb1d43a72dc82e5894f27b0f5c/models/stt/qwen3-asr-0.6b/coreml/QWEN3_ASR_COREML.md) | 歷史 bug diary：norm overflow、mel frontend、RoPE halves、prefill overhead、cache length 敏感區；可變成回歸案例，但作者對 compiler root cause 的斷言未在此獨立驗證 |

上述公開 tree 有 uv pyproject／lock；卻沒有以 ANE encoder 命名的新 1×1 Conv2D rewrite 腳本，不能保證 main 上基本 converter 就等於 #410 的 benchmark graph。Mobius 和 FluidAudio 均 [Apache-2.0](https://github.com/FluidInference/mobius/blob/4040a39f760290bb1d43a72dc82e5894f27b0f5c/LICENSE)。

## 5. 對本專案的可執行建議（本研究推論）

1. **先保留官方語義。** 從官方 model/config 建立 FP32 reference；逐一比對 mel、100-frame convolution chunk、800-frame attention、padding mask、output length、RoPE 與完整 logits。將 99/100/101、799/800/801 mel frames 及短尾巴列為邊界案例。不要直接採用第三方 single-chunk encoder 當完整 reference。
2. **讓模型分析決定 FP16 改寫。** 在真實 multilingual audio + teacher-forced decode 收集逐層 norm／attention／MLP 激活範圍與第一個非有限值。FP32 是現有 graph 的 workaround，不能推論所有數學等價改写都必須 FP32。測試可逆 scale／穩定 norm 需要同時驗 logits 和 WER；不要靠大幅 clip 掩蓋錯誤。
3. **將 ANE placement 變成 gate。** 每個模型記錄 OS、晶片、coremltools、graph/weights hash、compute units、compile 結果、per-operation preferred device 和 estimated cost。Apple 的 [MLComputePlanDeviceUsage](https://developer.apple.com/documentation/coreml/mlcomputeplandeviceusage) 定義的是 anticipated/preferred devices，仍應配 runtime profiling；以時間／cost 權重計算主導工作，避免用大量廉價 ops 稀釋 GPU 上的大矩陣乘法。
4. **decoder 同時試圖切分與固定形狀。** 0.6B 的 14+14 layers、T=128、scratch slots 是起始實驗點，不是 1.7B 最佳值。讓 agent 自動掃描 layers-per-model、prefill T、context、head shards，記錄 compile time、placement、tokens/s、energy/audio-second、WER；失敗配置留證據。
5. **品質比較配對且明確。** 原版 FP32／MLX／每個 ANE candidate 用同一音訊清單、normalization、language/context、decode limit；至少包含多語短句、長句、靜音、噪音、數字、專有名詞。分開衡量 token parity 和 WER；第三方 20 筆 CER 或 n=200 English WER 不足以支撐本專案跨語種承諾。
6. **優先建立可重跑 artifact contract。** 固定 source revision、uv lock、model revision/hash、conversion config、unit placement、quality/energy report；Standard ASR wrapper 只依賴這個已驗證的 engine artifact。研究失敗也輸出 machine-readable report，讓 coding agent 能自己選下一個實驗。

## 本次查核的 chronological log

1. 搜尋 1.7B CoreML 名稱，找到 aoiandroid／weiren119；模型卡明言 decoder GPU，修正「CoreML=ANE」的初步可能誤讀。
2. 搜尋 aufklarer 與 qwen3-asr-swift，沿連結找到 current soniqo/speech-swift；HF 卡為 0.6B。
3. 讀 current repository tree、runtime 和 HF config，發現 split decoder、固定 T，以及 encoder `.all`；比 model card 更具體。
4. 查到 Mega-ASR 的完整 Python bridge；確認是 fine-tune、FP32 GPU 與 ONNX encoder，未把它誤算成原版 ANE 成果。
5. 追 FluidAudio encoder PR 和移除 PR，確認 0.6B 的 ANE encoder 先例、GPU decoder 和目前 main 的支援狀態。
6. 讀 mobius converter，辨識其 single-window 限制、寫死 dimensions，以及報告和 current code 的年代差異。
7. Pin 可讀來源 revision、比對兩份 1.7B README hash，整理為這份筆記。沒有修改 reference repositories、下載大模型或代替本機 benchmark。
