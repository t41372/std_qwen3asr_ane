# Qwen3-ASR 1.7B ANE：首版實測結果

目前是研究預覽版。完整模型的主要神經網路計算已有 ANE 硬體證據，Standard ASR 0.2 batch／streaming 接口可用。沒有宣稱比 MLX 更快、節能或普遍品質非劣性。

## 固定環境與來源

- M5 Max、64 GB，macOS 27.0 build 26A428；Python 3.12.13、coremltools 9.0。
- 官方 checkpoint：`Qwen/Qwen3-ASR-1.7B`，revision `7278e1e70fe206f11671096ffdd38061171dd6e5`。
- Corpus測量使用Standard ASR main `3383126e165358e63be1971a671d2a52ffceace6`，protocol 0.2.0。收尾乾淨wheel安裝亦驗證較新的main `b63bb73bdef9be436fbae182d630452fe3a88f0b`；其runtime原始碼與先前版本逐檔相同。
- 最終 bundle：`artifacts/qwen3-asr-1.7b-final`，manifest SHA256 `9c705eba4ee069b04b13dcb08ab8abdcfcc1ba889aa925fc5f0c2ce26b28f659`。
- Converter 禁止不準確的 native GELU／SiLU fusion，使用精確 erf GELU 與穩定 exp SiLU。Ablation 的所有 learned-weight payload 均保持位元一致，沒有 fine-tuning。

## 品質：新樣本與診斷樣本分開

選樣使用固定 revision、parquet／音訊 SHA256 與 seed，並在候選推理前封存。英文涵蓋40名說話者；中文 FLEURS 沒有 speaker ID，只平衡 gender，不宣稱 speaker 平衡。評分使用固定 NFKC、大小寫及標點處理；沒有在看到數字表記退步後更換 normalizer。

下表只列未參與診斷的200筆新樣本。差值為 ANE 減去官方 CPU FP32；單位是百分點。

| 語言／指標 | 樣本 | 官方 FP32 | ANE | 差值 | paired bootstrap 95% 區間 |
|---|---:|---:|---:|---:|---:|
| 英文 WER | 100 | 34/2382＝1.427% | 33/2382＝1.385% | −0.042 | [−0.313, +0.205] |
| 中文 CER | 100 | 244/3737＝6.529% | 246/3737＝6.583% | +0.054 | [−0.207, +0.351] |

新200筆結果接近參照，但不能以「差異不顯著」當作已證明非劣性。嚴格 release gate 還要求更窄區間及更大的語料覆蓋，目前仍是 **inconclusive**。Bootstrap 是 utterance-level，尚非 speaker-cluster bootstrap。

另外200筆舊診斷樣本：英文 WER 2.244%→2.006%，中文 CER 6.743%→6.334%。修正前的 ANE 中文 CER 為7.562%；這個明顯回歸被 gate 抓到，經 hybrid 交叉測試定位到 encoder GELU。舊樣本參與了診斷，因此不把其改善包裝成獨立泛化證據。

400/400 音訊均完成推理，沒有 crash、timeout 或 token-budget failure。最終 tokenizer 對齊官方 `fix_mistral_regex=True`；12,090個音訊 token 長度×語言組合的無 context prompt 逐一驗證 input IDs 完全一致，其他 tokenizer／decoder／詞表組件及所有神經網路圖亦一致，因此上述 batch 結果可沿用。Context與stream prefix有另外的tokenizer／串流測試。

原始結果與區間：`artifacts/evaluation/silu-validation/{diagnostic,heldout,all}-paired-precise.json`，逐筆 JSONL 同目錄。Corpus與方法見 [evaluation-plan.md](evaluation-plan.md)。

## ANE 執行證據

10個主要 graph 的計畫中，12,028個具有成本估計的算子全部 preferred ANE；14,432個 unknown 全是 const。計畫只是預期配置，不是實際硬體使用率。

隔離的最終 Instruments trace 記錄623筆 ANE Prediction：frontend 21、encoder 3、7個 decoder 各77、LM head 60。全部主要階段都有對應事件，沒有未歸屬或背景 ANE Prediction。Graph、權重、manifest、tokenizer、source與trace tree的hash均綁定並於前後核對。Target PID的GPU事件為0，但有4筆未知process的GPU事件；不宣稱整部電腦沒有GPU活動。

CPU仍處理音訊特徵、tokenizer、embedding lookup、argmax、控制與資料搬移。現有trace不提供可靠的逐算子CPU時間，也不能把ANE時間占比說成FLOPs比例。詳見 `artifacts/telemetry/final-evidence.md` 與 `final-full-asr-isolated-attribution.json`。

## 速度與功耗

本專案沒有其他推理同時執行時，對同樣两個官方smoke音訊各暖機一次、測三次，中位數如下。ANE經真正Standard ASR插件路徑；MLX使用相同官方BF16權重、明確GPU與前後同步。

| 音訊 | 長度 | ANE | MLX BF16 |
|---|---:|---:|---:|
| 英文 | 15.051 s | 2.105 s | 0.435 s |
| 中文 | 4.204 s | 0.513 s | 0.137 s |

ANE已快於即時，但目前仍慢於MLX。這是兩個短樣本的warm latency，不能當成全面performance保證。冷模型載入／編譯另計，初次prepare實測曾約43秒；應在開始收音前prepare。T16分塊prefill將早期英文7.277秒降至約2秒。

`powermetrics`需要管理員憑證，`sudo -n`探測立即失敗；未索取密碼、未修改sudoers。因此能耗是 **unavailable**，沒有joules或節能宣稱。

另已實測不需sudo的IOReport途徑。API可存取，但CPU負載以及10,721次ANE小模型推理的前、中、後，CPU／ANE／DRAM能耗計數均為零；GPU會計數，卻不能代替總耗能。電池整機功率可以讀到變化，但曾約36秒不刷新，不適合短音訊能耗gate。詳見 [nonroot-power.md](nonroot-power.md)。這個結論不是單純因為沒有管理員權限，而是目前沒有通過有效性驗證的CPU＋ANE能耗來源。

## 接口與可靠性

支援batch、增量PCM16／float32串流、whole-input streaming output、可修訂partial、closed final、tail flush、取消、backpressure、sync bridge、語言控制與context prompt。Phrase hints使用標準的顯式degrade_to_prompt。單次session上限30秒；timestamps、diarization、硬候選語言限制及長串流rollover未宣告支持。

真機串流曾揭露native input lifetime crash。固定每模型FP32 input buffers、複製output ownership與顯式close後，完整模型兩輪realtime中文、idle、close均通過。這是實測mitigation，並非證明CoreML沒有其他native failures；細節見 [coreml-input-lifetime.md](coreml-input-lifetime.md)。功能矩陣與用法見 [standard-asr-features.md](standard-asr-features.md)。

最終bundle另通過一次realtime streaming與batch逐字一致驗證；0.1、0.5、5、29.99秒全零PCM均回傳空文字，顯式close成功。結果見`artifacts/evaluation/smoke/final-streaming-silence.json`。預設路徑已指向final bundle；舊T1搬到`artifacts/qwen3-asr-1.7b-native-t1`，歷史報告仍保留當時路徑，對照關係記於`artifacts/model-aliases.json`。

本工作未發布到PyPI、未上傳模型或建立外部issue。所有程式、原始證據與按時間排序的決策留在工作區；完整過程見 [technical-blog.md](technical-blog.md)。
