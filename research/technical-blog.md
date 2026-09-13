# 將 Qwen3-ASR 1.7B 移植到 Apple Neural Engine：實驗紀錄

這份紀錄按時間順序更新。成功條件是 Standard ASR main 協議可用、主要模型計算有 ANE 執行證據、辨識品質有對照、效率與耗能可重現；Core ML 格式本身不是 ANE 成功證據。

## 2026-09-12 — 01：建立工作區與查證起點

工作區最初只有空的 `std_qwen3asr_ane/` 與一份過往研究對話。實機為 M5 Max、64 GB、macOS 27.0 build 26A428，可用磁碟約 85 GiB。所有模型、依賴快取與實驗資料安排在工作區內，避免污染其他專案。

取得 Standard ASR main，commit `3383126e165358e63be1971a671d2a52ffceace6`，套件版本 `0.2.0.dev0`。公開 v0.1.1 不作為開發目標；依賴指向 main，uv lock 將記錄實際 commit。

第一輪搜尋找到既有 1.7B Core ML 模型；其 GPU/FP32 decoder 限制仍須逐個查證，因此目前不能宣稱這是第一個 Core ML 移植，也不能把既有模型的 Core ML 標籤當成 ANE 證據。過往 ChatGPT 對話只作研究線索。

遇到的第一個環境問題是 sandbox shell 無法解析 GitHub；以受審核的網路存取重新 clone 成功。沒有修改全域網路設定。`aufklarer/qwen3-asr-swift` 已不存在，改讀實際可取得的 fork 並尋找維護中的上游。

發現參考 encoder 省略原始 chunk/block attention，這種簡化不能在未評測下繼承。將以官方模型和逐段數值比較確保語義，而不是只追求轉換成功。

## 2026-09-12 — 02：接口先閉環，模型分開驗證

建立 uv 套件 `std-qwen3asr-ane`、model key `std-qwen3asr-ane/1.7b`。建構與探索不載入權重，artifact status 只檢查本機 bundle，不偷偷下載或用一次假轉錄觸發安裝。尚無發布的轉換成品，因此 acquisition 回報結構化 `action_required`，指向本機 build 命令。17 個 adapter 測試及官方 compliance run 通過；這些測試使用假 runtime，只證明接口契約。

取得官方 1.7B checkpoint，共約 4.3 GiB，存於 `artifacts/source/Qwen3-ASR-1.7B`。Python 3.12 venv 與 uv/Hugging Face cache 均在工作區。第一個安裝錯誤是 hatchling 預設拒絕 direct Git dependency，設定 allow-direct-references 後成功。當次 resolver 選到 coremltools 9.0、torch 2.14.0；後者超出 coremltools 宣告已測試的 2.7.0，保留警告並以真實轉換測試確認，不把安裝成功當相容性證据。

encoder 採 100-frame conv chunk 與 104-token attention window，保留尾部與位置重設語義。短於100-frame的單獨音訊需要清掉多餘 convolution activations；這和把所有輸入補零後直接裁輸出不同。20項 PyTorch parity 測試通過。又發現官方 eager 路徑未傳入 block mask，因此與 FA2 的 cu_seqlens 路徑不同；我們把語義選擇明確記錄為獨立 FA2 windows，後續辨識品質比較須揭露此差異。

## 2026-09-12 — 03：第一個 ANE decoder 圖與數值反例

以真實 layer 0 權重建立 Conv2d projections、穩定 RMSNorm、固定128長度 MLState cache 的 decoder probe。Core ML 圖成功轉換。MLComputePlan 報告249個具有 cost 的運算全部 preferred ANE，另310個無 cost 的項目全部是 const。這只是編譯器預計分配，尚非硬體 telemetry。

第一次用64倍 residual縮放，PyTorch 重寫與官方 layer 的最大誤差約1e-6，但 Core ML CPU+ANE 連續8個token的相對L2誤差約0.19–0.25，雖然全部有限值且每步約1–3 ms，仍完全不能接受。這是一個必要的失敗測試：速度、圖分配、沒有NaN三者都不足以判定模型正確。正在用不同縮放倍數與 CPU_ONLY 對照定位；權重被縮得過小、FP16 subnormal行為列為待驗假設，尚未認定根因。

CPU_ONLY 的第一個圖在載入時出現 Core ML execution plan error -14。調整無上限 clamp 的圖表達（改成 maximum），分離編譯和數值問題。所有探測的log與JSON保留在 `artifacts/probes/`。

系統的 Core ML compiler 在sandbox內不能建立工作目錄，即使 TMPDIR 指向 /private/tmp 仍然失敗。事先向使用者說明後，對限定模型的 compile/predict/plan 使用受審核執行；只允許框架正常管理 temporary/cache，未修改系統設定、未使用 sudo。

## 2026-09-12 — 04：完整模型第一次說出正確文字

撤掉64倍全域縮放，保留以amax穩定化RMSNorm。epsilon改以sqrt(epsilon)先除尺度再平方表示，避免把重要常數直接存成FP16 subnormal；這項改寫的layer0實測仍約1%相對L2誤差，不能說它已完全解決ANE數值差異。

逐層掃描28層真實prompt：FP16 residual最大10880、gate最大145、up最大137.75、product最大14904，全程有限；最終8/8 argmax符合官方FP32。最後一層將約10840的residual降到496，觀察到相消誤差。這輪證據反對不必要的全域縮放，也支持先測真實音訊。

完成bundle：frontend、24-layer encoder、7個4-layer decoder partitions、19塊詞彙projection、CPU embedding、tokenizer與mel filters。官方source revision為`7278e1e70fe206f11671096ffdd38061171dd6e5`；從所有Hugging Face download metadata交叉檢查commit一致，寫入source.json及bundle manifest。

首次端到端在完全禁止GPU的CPU_AND_NE設定成功：官方英文15.051秒音訊→7.277秒，中文4.204秒→2.324秒。英文輸出與作者提供reference只差開頭Mm；中文「甚至出现交易几乎停滞的情况。」完全一致。與官方PyTorch FP32/eager對照，英文原版開頭是「Uh huh」，其餘一致；中文相同。官方CPU4threads本次約3.166秒/0.807秒。這些是小樣本smoke且可能有並行實驗干擾，不能宣稱WER穩定或效能勝出。

首輪分階段timing定位瓶頸：英文features0.002秒、encoder0.032秒、prefill5.800秒、generation1.443秒；中文encoder0.010秒、prefill1.982秒。應先優化prefill，而非繼續微調已很快的encoder。完整adapter/mel/decoder測試當時77項通過。

## 2026-09-12 — 05：建立動態證據與分塊prefill實驗

Xcode Instruments CLI成功錄到Neural Engine hardware table；layer0 probe的8687次predict與8687筆同名ANE Prediction事件一一吻合。這比compute plan更強，但目前只覆蓋單層probe；完整ASR trace正在收集。

CPU_ONLY負對照在CoreML compiler發生E5MinimalCpu / error -14，不能宣稱已得到有效負對照。保留失敗trace與log，仍可分別查實際ANE事件、模型標籤與CPU_ONLY支援缺陷。

開始T=16固定寬度decoder實驗：prefill一次16token，generation仍只讓一個有效token更新cache，其餘padding的update matrix全部為0。整個utterance使用同一個模型及MLState，避免不同prefill/decode模型之間不可假定的state共享。PyTorch測試驗證分塊、尾部2token、最後單token與逐token的hidden及KV cache一致。新bundle另存`artifacts/qwen3-asr-1.7b-t16`，保留已成功的T=1模型作對照。

## 2026-09-12 — 06：速度改善、完整硬體證據與第一個 corpus gate 失敗

T16維持兩個smoke逐字輸出不變，英文從7.277秒降至2.024秒，prefill從5.800秒降至0.452秒；中文從2.324秒降至0.507秒。這是首輪時序證據，不把profiled或並行量測當正式benchmark。

針對T16全部10個graph收集plan：11,720個有cost算子全部preferred ANE，14,292個unknown全為const，各graph分別有完整成本，不跨graph加總成虛構FLOPs比例。第一次完整T16 trace與100筆品質評測碰在一起，留下大量背景ANE事件；自動gate正確判為inconclusive。停止其他ANE工作後重錄：599筆Prediction，7個decoder各74次、frontend21、encoder3、LM head57，沒有额外背景Prediction，target GPU events為0。Sidecar綁定實際bundle/trace/模型hash；ANE schema沒有PID，因此只以隔離、模型標籤和預期调用量歸屬，沒有捏造逐operation CPU timing。

固定200筆評測在看模型結果之前完成選樣：LibriSpeech test-clean100筆、40位speaker、808秒；FLEURS Mandarin test100筆、1091秒，無speaker欄位，僅按gender各50筆，不能宣稱speaker平衡。來源revision、parquet hash、每條音訊hash與reference都保存。

英文100筆：official FP32 WER47/2094=2.2445%，ANE43/2094=2.0535%；差值-0.191pp，utterance paired bootstrap95%區間[-0.475,+0.050]pp。中文100筆：official CER247/3663=6.7431%，ANE277/3663=7.5621%；差值+0.819pp，95%區間[+0.283,+1.487]pp。所有200條均正常結束，但**中文品質gate失敗**，不能用英文的改善抵消中文退步，也不能宣布模型已品質等價。

MLX參照使用獨立uv環境（mlx-audio0.5.3、mlx0.32.2、transformers5.17.0），直接strict-load同一份官方BF16權重並在記憶體轉置conv，不另下載或修改source。MLX中文同100筆CER236/3663=6.4428%。暖機後smoke英文約0.439秒、中文約0.140秒，明顯快於目前ANE候選。這是誠實的限制：ANE移植已成立，速度尚未追上MLX，功耗又沒有可用權限，不能宣傳節能。

## 2026-09-12 — 07：否證最初假說，改查精度

中文主要回歸中的部分樣本涉及阿拉伯數字與中文數字表記。仍保留事先固定的raw CER評分，不能在看完測試結果後改normalizer把回歸洗掉。以10個事後選出的diagnostic樣本做官方FP32 global/window成對比較：global重跑10/10一致；改成window反而少2個errors，候選仍多26個errors。MLX也確實使用window attention。因此「分窗語義導致主要退步」被否證，不能為了讓baseline看起來更近而盲目改encoder。

下一步是4個代表樣本的官方CPU FP16對照，以及單layer混合精度CoreML probe：主要conv/matmul/softmax保留FP16，其餘scalar/residual math保留FP32，分辨普通FP16量化、ANE subnormal/fusion與最後layer大值相消。先只建96MB單層，避免在尚未證實方向時再產生完整4.4GB模型。

另外，工作區約18GB、系統temporary約5.6GB，但可用磁碟從最初85GiB降到14GiB。尚未確定其餘下降由何處造成；不假定全是我們的cache，也不清理其他應用或Time Machine快照。已停止新增大模型變體；MLX重用原weights就是為了避免再耗4.4GB。權限探測`sudo -n powermetrics`立即回報需要密碼，未取得或索取密碼，能耗繼續標為unavailable。

## 2026-09-12 — 08：隔離 ANE SiLU 誤差

四個代表樣本的官方整模型CPU FP16全部與FP32 normalized文字一致，CER totals均為2，而原ANE候選為13。普通FP16本身不能解釋回歸。Encoder LayerNorm微模型一開始全被分配到CPU；加入identity 1×1 projection才取得真正ANE計畫。直接寫F.layer_norm與手寫版幾乎相同，因為MIL早已融合成layer_norm+batch_norm，沒有證據支持盲目替換norm。

主線建立無KV state的first-token decoder，逐一輸出norm、v、attention residual、gate、up、product及final，排除cache和attention歧義。真實token151644上，ANE gate/up誤差約0.12–0.15%，但SiLU product突然升到1.34%。把**ANE實際輸出的gate/up**帶回CPU重算精確product，仍與ANE product差1.32%，於是排除了上游誤差累積。

微圖比較發現，F.silu與x*sigmoid(x)都被MIL融合成原生silu，兩者均不準。x*(tanh(x/2)+1)/2降低誤差，但負端有相消。最後選用`x * exp(min(x,0)) / (1 + exp(-abs(x)))`：exp輸入永遠非正、分母介於1與2、無clipping、MIL不會再融合回silu。四個真token的product誤差降至0.046–0.054%，約26–39倍改善，全部算子仍preferred ANE。完整單layer誤差也下降，但這仍不能替代整模型品質驗證。

另外嘗試FP32 scalar/residual + FP16重運算的mixed圖，仍遇CoreML execution plan -14。改單次cache寫入不能解決；直接copy_又觸發coremltools的No matching select or slice，完整slice assignment雖可export但CPU_ONLY仍拒絕。這個方向沒有證明有效，正式SiLU ablation回復已驗證的原cache表達，僅改SiLU。

新候選用APFS clone共享未變模型，重新轉decoder後比對binary SHA256，7個weight.bin全與原來相同，因此再以hardlink共享已驗證不變的權重。新版本只新增graph描述，避免多佔2.7GB權重。這也提供很直接的ablation證據：權重一個bit都沒變，改的是graph math。新候選`artifacts/qwen3-asr-1.7b-stable-silu`仍標unvalidated。

將原deterministic corpus selection延伸到每語言200筆，確認前100筆與原版完全一致，新增的後100筆與任何診斷樣本不重疊。在候選推理前封存兩種用途：200筆舊樣本作diagnostic retest，200筆新樣本作held-out。已啟動完整400筆候選和新200筆官方參照。早期回報顯示單算子精度改善並未立刻消除所有中文數字表記差異，因此繼續做encoder/decoder hybrid交叉定位，沒有提前宣布成功。

## 2026-09-12 — 09：加入 streaming，真機測試揭露生命週期問題

使用者追加streaming與盡可能多Standard ASR能力的要求。核對官方：streaming公開入口目前限vLLM，算法是累積音訊重新编码、前幾chunk不固定prefix、之後回退K個raw output tokens，再接續解碼。不是已有可直接搬用的持久causal encoder state。

實作Standard ASR v0.2 session：PCM16/float32增量輸入、whole-input streaming output、batch/stream context prompt、語言override、partial→closed→done、tail flush、cancel與有界queue backpressure。Raw metadata與可見文字分開，rollback避免UTF-8截斷；partial的stable_until保守為0。Phrase hints仍透過標準的degrade_to_prompt，不假裝native boost；候選語言hard restriction、timestamps、diarization等未有真正引擎支持者仍不宣告。當前session明確限30秒，超限structured error，不以未驗證的rollover偷偷切詞。

首輪mock/契約測試通過，真實realtime中文卻出現SIGSEGV。讀取本次python crash報告，故障在背景dispatch thread的_PyObject_Free→libcoremlpython→MLFeatureValue dealloc→MLE5InputPortBinder resetAfterLingering。小模型thread/main-thread/idle對照未穩定重現，因此不能把假說當完整根因證明。

源碼檢查發現coremltools9的predict會把提供的FP16 NumPy input原地替換為新的FP32 array。只保留原FP16 arrays無法保住實際被native借用的owner。加入每模型一組固定FP32 buffers，以copyto填資料，輸出也copy成host-owned arrays；沒有逐call無限保留。close先dropmodel，依Python refcount確認native借用退場再放buffers，timeout明確失敗並保留資源。

同一完整模型連做兩輪4.2039秒realtime中文，每輪正常closed→done、event compliance通過、final均「甚至出现交易几乎停滞的情况。」；每輪idle5秒、explicit close（約41ms）與close後idle5秒全部正常、exit0。這是已在真機驗證的ownership mitigation，尚非小模型乾淨A/B的底層bug證明。全部原始記錄保存在`artifacts/evaluation/smoke/streaming-lifetime-persistent.jsonl`。

## 2026-09-12 — 10：以 hybrid 交叉測試定位真正的品質瓶頸

穩定SiLU候選在舊中文前79筆只從233降到232 errors，許多數字表記仍未恢復。因此把「算子誤差已降低」與「端到端CER已修好」明確分開。

選四個既有diagnostic樣本861/331/192/554，每個在獨立有timeout的process交叉兩條路徑：ANE encoder＋官方FP32 decoder，以及官方FP32 window encoder＋ANE stable decoder。結果4/4轉錄完全隨encoder走：前者重現候選，後者恢復官方；CER totals分別13與2。這包含不只是數字格式的192「这巧克力」對「热巧克力」。

同一份mel輸入下，ANE與官方window encoder的embedding相對L2高達16.6–22.2%，cosine約0.975–0.987。Host mel與官方mel相對誤差僅5–9e-7，官方encoder改吃host mel只變約3e-6，排除了host特徵處理。最初synthetic mel的好結果沒有覆蓋真語音分布；這是本次驗證流程要記住的教訓。新的優先方向是encoder原生activation／融合行為，先作GELU與分階段數值探測，沒有盲目改大模型。

公開engine.close及context manager也已加入：與推理共用Lock，成功才清runtime，可重新prepare。Retirement以指數退避避免閒置高頻polling，interpreter shutdown不再嘗試創建清理thread。這改善正常explicit close路徑；不將Python interpreter teardown當作已被完整證明安全的情境。

## 2026-09-12 — 11：找到 encoder GELU 誤差與精確替代

GELU probe在861真實conv1 activation量到原生ANE相對L2誤差2.735%，而未融合的精確erf公式只有0.0433%，約63倍改善。FFN第0/4/23層亦有同方向差異。前端尚未進入Transformer就已有4.45–5.48%的embedding誤差，後續24層再將它放大。

單純改成`0.5*x*(1+erf(x/sqrt(2)))`仍會被CoreML的`common::fuse_gelu_exact`融合回原生GELU。移除該pass（也移除tanh approximation fusion）後，erf/mul/add圖維持全部preferred ANE，且微模型實測保持精度。這次可以保留數學上精確的GELU，不必接受tanh近似。

正式converter新增pass policy及結構guard：若輸出MIL重新出現native gelu/silu即讓conversion失敗，避免未來coremltools升級把修正悄悄融合掉。Encoder的所有conv、FFN及最後projector使用erf公式。29項encoder/decoder PyTorch parity測試通過。

候選`artifacts/qwen3-asr-1.7b-precise`只替換frontend與encoder圖，保留stable-SiLU decoder；兩個新weight.bin的SHA256又與舊版完全相同，得以共享。這次在bundle內保存human-readable conversion source snapshot及hash，並在build結束檢查source沒有於途中更動；補上早期實驗只存fingerprint卻未保留source內容的可重現性缺口。再次啟動相同400筆評測，仍未用新200筆的錯誤樣本來選擇修正。

## 2026-09-12 — 12：最終驗證、封裝與交付邊界

400/400推理完成、零失敗。舊診斷中文100筆从277錯降到232錯，六個代表的數字與「热巧克力」均恢復官方形式。未參與診斷的新100英文WER為官方34/2382、ANE33/2382；新100中文CER為官方244/3737、ANE246/3737。這些觀察接近參照，但置信區間仍不足以通過更嚴格的普遍非劣性release gate。完整數值與區間另外集中於results.md，不用混合診斷樣本的總平均掩蓋統計限制。

最終tokenizer採官方wrapper使用的fix_mistral_regex=True；該選項會改pre-tokenizer的大小寫／Unicode切分。未直接假定它無影響：保留父bundle，另建final bundle，枚舉390個audio-token長度×31個語言／auto選項，共12090個default prompt，input IDs全部一致；其他tokenizer組件也完全相同。因此既有無context batch品質證據可沿用，context及stream prefix另做測試。replay工具為experiments/finalize_tokenizer.py。

Final bundle的10個graph與precise逐檔hash相同，明確重用父計畫，不浪費重複compile。新的隔離trace仍實際執行兩語音訊：623筆ANE Prediction涵蓋全部元件、無額外background/unmatched ANE，source／graph／tokenizer／trace全部hash綁定。Target GPU為0，但4筆GPU事件無process歸屬，限制如實保留。

在沒有本專案其他推理並行時，正式Standard ASR插件warm測量中位數為英文2.105秒、中文0.513秒；同官方BF16權重的MLX為0.435／0.137秒。修正沒有讓ANE贏過MLX，不能宣稱更快；功耗仍無權限，不宣稱省電。可用磁碟曾由12GiB恢復至71GiB，沒有人工刪除任何外部快取、snapshot或其他應用資料，不能把恢復原因當作已被驗證。

最後以final bundle跑realtime中文、batch一致性與四種長度靜音，全部通過，explicit close正常。預設artifact路徑切到final，舊T1保留為native-t1並保存歷史alias映射。Artifact status補上package內部模型／weights存在性與symlink containment檢查；通用evaluator新增Standard ASR backend以及finally close／cleanup sidecar，後續其他硬體plugin可沿用同一corpus與評分，不需再寫ASR呼叫膠水。

uv已建立wheel與sdist；乾淨runtime環境不安裝Torch即可探索模型與檢查artifact。Git main直接依賴無法單靠offline解析，因此乾淨安裝使用已授權網路取main，得到b63bb73bdef9be436fbae182d630452fe3a88f0b；比對runtime source與原3383126版本完全相同。交付保持research preview：30秒session上限、沒有時間戳／diarization／長串流rollover、沒有能耗數據，且品質release coverage仍不足。

## 2026-09-12 — 13：補查無管理員功耗來源

為避免把「powermetrics需要密碼」誤當成所有能耗途徑都不可用，另用Astra medium子代理做有界研究。依macmon固定commit的primary source，在工作區編譯幾KB的Objective-C IOReport讀取器；沒有安裝global工具或更改系統設定。離開Codex sandbox後以普通UID501即可訂閱Energy Model，證明此路徑不需要sudo。

但在這個macOS27 build，四個CPU busy workers期間CPU Energy全為0。進一步以已知ANE小模型跑10.000692秒、10,721次有限輸出predict，前中後25個窗口的CPU／ANE0／DRAM0仍全0。GPU在24/25窗口會計數，但不能替代CPU＋ANE總耗能。這是無效／停滯counter，不是零功耗。

IORegistry的電池整機功率也可無root讀取，約與Voltage×InstantAmperage一致；45秒觀測中有一次約35.9秒後才刷新的變化，積分語義與cadence未校準，不能拿來評估短utterance。原始資料與時鐘偏移校正保存在artifacts/power-probe，結論見nonroot-power.md。能耗gate仍unavailable，現在有比單純權限不足更完整的原因，未為了交付而編造省電數字。

最終程式通過156個測試、Ruff與Standard ASR compliance（含sync bridge）。自動workflow的protocol與actual-device gates通過；品質的嚴格release gate仍inconclusive。乾淨wheel環境無Torch、使用tokenizers0.23.2及最新main，實際中文轉錄與result compliance均成功；wheel source與工作區package逐檔一致，LICENSE與NOTICE亦已打包。

## 2026-09-12 — 14：撤回過早完成判定，繼續改善串流、速度與能耗驗證

使用者指出上述研究版沒有達成原始目標。這個判斷正確：ANE placement、協議相容和有限樣本的品質接近，只完成部分工作；不能把剩餘目標寫成限制就宣告完成。建立新的 active goal，要求長串流、速度優化、可信能耗對照與充分品質驗證完成後才關閉。

30秒是目前bundle metadata、session buffer和1024-token decoder cache共同設下的產品限制，不是Qwen3-ASR的streaming上限。官方公開streaming依然累積音訊並回退文字prefix，沒有現成的無限causal encoder state可直接使用。下一版先建立精確prompt-prefix重用：比較完整embedding的實際值，只重用完全相同的因果前綴，向下對齊prefill block，再覆寫改變的suffix；未來位置的stale KV由mask隔離。長session另外擴展cache與設計有證據的分段，不能只拿掉檢查。

載入路徑檢查發現每個runtime都重新開啟.mlpackage。Apple文件指出CompiledMLModel配合穩定.mlmodelc路徑才能重用裝置特化快取，因此新增明確的immutable compiled-bundle實驗，保存來源檔案hash與host版本，在相同權重下量測冷啟動與warm推理。這時尚未把預期改善當成量測結果。

第二輪功耗probe直接檢查364個Energy Model channel的原始payload：CPU SRAM等替代channel與ANE同樣凍結，但CPU residency正常前進。這縮小到能量資料路徑，仍不能斷言是某個特定driver bug。另找到可由普通使用者讀取的SMC PSTR整機功率，每秒有更新；先做受控負載與cadence驗證，再規劃長時間等工作量能耗對照。

## 2026-09-12 — 15：載入改善、真實串流重用與第一組有效整機能耗

相同final FP16模型轉成穩定.mlmodelc路徑。第一次runtime載入36.335秒，第二個獨立process只有1.539秒，兩個smoke的全部文字一致。Warm英文仍約2.1秒、中文約0.50秒；這是一個明確的重複啟動改善，沒有把它混同於token生成加速。功能成為`qwen3-asr-ane compile --source ... --output ...`明確準備指令，來源保持不變，compiled bundle保留來源檔案hash，Standard ASR artifact檢查亦支持這種格式。

Exact streaming prefix context已整合。真實15秒英文串流的closed文字與batch相同，event/result compliance通過，explicit close正常。這個real-run是功能驗證，當時另有CPU轉換工作，不採其latency作性能證據。193個測試通過，包含context失敗後的NaN cache恢復、不同session隔離及compiled artifact存在性檢查。三分鐘測試目前只是fake runtime契約驗證；真正更長artifact與長音訊驗證仍待完成。

SMC active control的30秒閒置／四CPU負載／恢復平均12.18／53.95／15.86W，確認正常負載響應與過渡延遲。接著在沒有其他本專案模型工作時，ANE與MLX各完成相同60輪兩音訊，共120次辨識、1155.31125秒音訊。ANE用155.152秒、估計5876.40J（5.086J／音訊秒）；MLX用34.375秒、2681.05J（2.321J／音訊秒）。ANE平均37.88W低於MLX的77.99W，但因為更慢，總能耗約2.19倍。數據推翻了「低瓦數就比較省電」的直覺，優化需要降低每項有用工作的能耗。

這一組是SMC整機估計、單次A/B診斷，沒有外部功率計校準或ANE逐裝置歸屬。兩次都在AC且充電；power trace保存這些條件，PSTR遠低於含充電的SystemPowerIn，不能把它稱為wall-plug energy。邊界平移±2秒的估計範圍ANE5838.69–5904.56J，MLX2551.67–2794.81J；這個差距遠大於邊界延遲，但更小的未來改善需要ABBA重複和條件檢查。原始證據在artifacts/power-v2/asr-baseline-{ane-a1,mlx-b1}。

## 2026-09-12 — 16：從 token 寬度轉向權重頻寬與解碼算法

真實四層partition的固定T1約3.838ms，而既有T16约4.2ms，消除15個padding位置只提供小幅改善。大部分成本是權重供應，不能靠Python微調補足與MLX的差距。小型多function模型載入報functionName/modeltype錯誤；小型與真實四層EnumeratedShapes模型雖能轉換，卻在此host執行計畫建構報-14。固定真實T1可正常執行。這些是具體實驗的失敗，尚未證明所有CoreML flexible/stateful組合都不支援；若需雙寬度，公開MLState read/write提供明確狀態轉移的替代路徑。

開始投影權重palettization，保留FP16 activation、穩定SiLU、相同KV布局。為避免對數億重複BF16來源值跑昂貴的隨機k-means，利用FP16最多65536個bit patterns建立完整histogram，以元素出現次數作Lloyd更新權重，再按實際FP16 LUT精度重算nearest assignment。這仍優化全部權重的MSE，沒有抽樣，也不等於端到端品質保證。四層weight.bin由384MiB降到8-bit的193MiB；實際T1中位数2.747ms，比FP16約快28%。6-bit約2.877ms，顯示bit更少不自動更快。4-bit仍在測量。

使用者進一步要求各benchmark都勝過MLX，實作改由主代理為主，研究子代理只處理有界資料查核。量化還有品質與頻寬上限，因此開始研究0.6B作draft、1.7B作最終verifier的精確greedy speculative decoding：一次搬入1.7B權重可驗證多個token，只有被1.7B驗證的前綴才能輸出。這是待實作與測試的方向，絕不把draft結果直接冒充1.7B。

## 2026-09-12 — 17：三分鐘真實串流與 speculative decoding 首輪結果

轉出4096-position、T16的完整target decoder，encoder及七個decoder的相同weight.bin仍可按hash共享。新artifact明確宣告180秒，沒有改寫原始30秒artifact的歷史。Stable compiled bundle也只共享完全同hash的immutable weights，避免為每個context bucket浪費數GiB磁碟。

長音訊diagnostic串接19個完整LibriSpeech句子並加入短間隔，保留每段來源hash／已知起迄，補尾靜音至恰好180秒。這不是獨立的自然長音訊語料。10秒更新一次的真實ANE串流完整處理180秒並送出closed→done，event/result compliance通過，沒有詞級deletion或insertion。串流為6/436 WER，batch為7/436；文字不完全相同，不能把batch equality当作所有streaming策略的必要語義。MLX同音訊也是7/436，warm約4.737秒。ANE stream整個快速feed replay耗76.767秒，這包含18次累積辨識，不與一次batch的4.737秒直接相比。較大cache第一次模型載入32.939秒，explicit close22.7ms。資料在artifacts/evaluation/long-stream-180-en。

更小位元數沒有持續改善：4-bit四層T1為2.712ms，與8-bit的2.747ms接近，尚未作量化品質評測。Grouped attention透過Q head與O projection column重排，把16個獨立head改成2個batch；Torch因果輸出與KV parity通過，但真機3.984ms不如原始3.838ms。原生CoreML SDPA亦約3.832ms，沒有實質改善。14層partition約12.845ms，換算28層25.69ms，只比七個四層26.87ms略快；其8-bit palette版14層9.059ms，仍不足以獨自解決差距。Compact T1 LM head保持60個真實hidden states的所有greedy選擇，4.789ms對原始4.838ms，改善很小；下一步改成T16一次求多個位置的最大值，這才可能攤薄詞彙權重讀取。

已在ANE轉出官方0.6B draft（revision 5eb144179a02acc5e5ba31e748d22b0cf3e303b0），兩者tokenizer與prompt IDs實際一致。新增純算法greedy verifier及79個拒絕位置／EOS／預算／stale suffix測試。採held-target-token策略，所以T16最多搭配15個draft proposals，不犯off-by-one。拒絕後只輸出target correction，之後覆寫相同絕對位置；full acceptance則補上draft尚未消耗的最後proposal，避免KV中留下缺口。

第一個K7真機版本兩語全部與serial target逐token完全相同。英文47個proposals接受42個、7次target verification、52次draft step，但總時間約2.11–2.13秒，比serial2.05–2.08秒略慢；中文也是約0.552秒對0.500秒。0.6B本身仍有28層與相同16Q/8KV heads，ANE上其逐token代價沒有隨parameter數等比例下降。高acceptance並不保證端到端加速，必須改善draft與prefill、批次LM head，再量完整流程。

## 2026-09-12 — 18：批次 LM head、狀態轉移與理想下界

T16 compact vocabulary head在60個真實hidden states維持全部greedy選擇，扣掉各自第一call後，每個有效token成本由4.567ms降至0.344ms。將它接入K7 verifier後，英文完整流程約1.923秒、中文0.520秒，仍沒有勝过MLX。再加入T64 prefill並用公開MLState read/write轉移112MiB級的cache，英文約1.716秒、中文0.512秒；copy本身約63–67ms，吃掉短句的prefill收益。

接著補做更具體的state compatibility小實驗。早先的toy multifunction載入失敗並不能回答兩個真實static模型能否接受相同MLState。使用相同四層decoder的T16與T1模型，在8-token prefill後，將原state直接交給T1；另一條對照則先public read/write copy到T1新state。後續5個位置的output max_abs均0、所有key/value arrays逐元素完全一致。這只證明此host上的這對模型；尚未整合成正式runtime策略或廣泛驗證全部shape／partition。

為判斷值得找更快draft與否，另作明確標示的oracle診斷：ground-truth文字經TranscriptDraft提出候選，T64 target搭配T16 head驗證，與原T16 serial target逐token一致。英文target驗證下界約0.400秒、中文0.127秒。這**不包含取得草稿的成本**，不是可交付引擎性能，也不能宣稱已勝過MLX。它只表明把逐token draft換成更便宜的整段文字提案可能值得研究。

Q/K/V與gate/up projection合併通過Torch因果輸出及KV parity，但真機四層T1約3.981ms，沒有改善，因此仍未啟用。

## 2026-09-12 — 19：SenseVoice 草稿支線未成功

研究找到非自回歸SenseVoiceSmall公開CoreML產物，嘗試以其CTC全文當草稿，由1.7B保留最終token決定權。它只涵蓋中、英、日、韓、粵五語，不是1.7B移植必需的部分，也尚未進入正式插件。

固定取得FluidInference conversion revision 0e0bf30bfc6836f182ccd1d89984df919c949e26，保留upstream model attribution與自訂FunASR model license。原始frontend配置／CMVN來自FunAudioLLM/SenseVoiceSmall revision 3847d57b6bdf2dd8875cb1508d2af43d80a16bf7。用kaldi-native-fbank1.22.3＋NumPy重建CPU frontend，共同feature範圍max_abs4.53e-5、relative L2 2.11e-6。公開frontend在某些長度多一個尾端LFR row，原因是固定右補7個frame後stride6，沒有裁至upstream的ceil(T/6)。這個差異被記錄，沒有隱藏。

真正阻塞在encoder：英文T256輸出shape正確，但6,514,300個logits全部nonfinite。Integer input控制值與dtype已確認保留。預期compute plan的2207個已知device ops全為CPU，2928個unknown，所有estimated costs缺失；因此不能因為指定CPU_AND_NE或外部模型卡就斷言它在此host跑ANE。

infrequent-reshape hint、將-inf mask改為-10000、提高LayerNorm epsilon及repeat-padding均未修復。只改MIL固定輸入形狀卻沿用原enum metadata的獨立process因default-shape不一致NSException abort；之後用public compiler產生一致fixed I/O metadata，再保留原MIL arithmetic及weight bytes的128/256/512版本能load，但仍全NaN。這些是假說被否定／尚未解決的結果，沒有證明某個特定算子或driver是根因。

這條支線花了太多時間。應更早設定停止條件，避免未先確認關鍵device/shape路徑就持續修第三方模型。所有失敗資產與程式留在實驗區，不作為可用backend。

## 2026-09-12 — 20：依使用者的新方向收束並交接

使用者詢問進度及SenseVoice與目標的關係，表示正在整理更好的工作方式，允許先收束、寫handoff，之後使用乾淨上下文繼續。停止擴大研究，整理原始資料、預設引擎和未啟用的實驗邊界。

收尾通過289個tests、修改檔案的Ruff、Standard ASR main CLI compliance及正式引擎的真實中文辨識。SenseVoice、oracle、quantization和speculative prototypes均未進入default plugin；預設model alias也沒有指向未驗證的新候選。工作狀態與重現命令寫入HANDOFF.md，原始速度／能耗／廣泛品質目標仍未完成，沒有再次標記完成。

## 2026-09-12 — 21：補完已準備的完整模型 state-sharing 驗證

自動目標續行後，範圍限制在收尾前已寫好、尚未跑完整模型的shared-state選項。沒有修改推理程式、沒有擴大SenseVoice研究。先核對乾淨的1a83e63工作樹，再用既有T64 prefill／T16 generation／0.6B ANE draft／T16 vocabulary head，跑share與copy兩個獨立程序，每個兩音訊、各一warmup和三次測量。所有16次完整推理均與原本serial 1.7B逐token一致，兩程序exit0、explicit close正常。

Shared state的英文中位1.6903秒，copy控制1.7565秒，相差66.2ms；中文0.4528對0.5221秒，相差69.3ms。Copy本身62.7–64.4ms，而兩程序同時量到的serial target基準都約英文2.10秒、中文0.512秒。結果支持在這個host與這組相容models中避免state拷貝能省掉這部分開銷；不將兩個smoke推廣成所有shape、CoreML版本、廣泛品質或能耗證明，且完整流程仍慢於MLX。

原始資料與帶hash摘要保存在artifacts/evaluation/smoke/shared-state-full-k7.jsonl、copied-state-full-k7-control.jsonl、shared-state-full-comparison.json，結果補入HANDOFF.md。沒有改預設引擎或產物。

## 2026-09-12 — 22：驗證純1.7B greedy的寬prefill收益

自動續行時，將範圍限制為已驗證的共用state對核心1.7B路徑是否有益。新增獨立benchmark子類，只覆寫prepare_prompt的路由；兩種模式沿用相同正式greedy loop與generation handles。沒有0.6B draft或SenseVoice，也沒有oracle草稿。各次配對交替執行順序，兩段smoke各一warmup及三次測量，8組配對的全部token相同、explicit close成功，記錄complete與verified_pairs，失敗不會冒充完整驗證。

英文完整推理由2.0954秒降到1.8178秒（約13.3%），中文0.5100秒降到0.4333秒（約15.0%）。英文prefill由0.4193秒降至0.1415秒，中文0.1522秒降至0.0733秒；生成時間基本不變。這是可以歸因到prefill的核心改善，但還沒有勝過MLX，也不是廣泛品質或節能證明。

原型額外完整載入prefill runtime，因此base載入1.612秒之外另需約0.394秒。沒有隱藏這項冷啟動取捨，也沒有將原型整合為預設插件。資料與來源hash在artifacts/evaluation/smoke/dualwidth-greedy.jsonl及dualwidth-greedy.summary.json，更新handoff供新workflow評估。

## 2026-09-13 — 23：接手、預先登記門檻、分解 ANE 逐 token 成本的 floor

新工作方式下接手。先確認起點：289 tests 通過、Standard ASR main 仍是 uv.lock 鎖定的 `b63bb73b`、M5 Max／macOS 27.0。以既有 profile 與孤立 Instruments trace 重新判讀瓶頸：兩段 smoke 的 623 筆 ANE Prediction 合計 2.311 秒，對應 predict 牆鐘約 2.53 秒，ANE 硬體忙碌約 91%。瓶頸主要是 ANE 每個 token 的成本本身（7 個 partition 各約 4.15 ms＋LM head 4.5 ms ≈ 33 ms／token）；predict 之外與之內的 host 開銷約 9%，之後 ANE 變快時這個比例會相對上升。在任何新候選結果出現前，先把品質／latency／能耗／記憶體門檻與比較對象寫進 `research/preregistration-2026-09-13.md` 並提交。

之前的 probe 顯示 LUT4 與 LUT8 幾乎同速，代表存在與權重 bytes 無關的 floor。用真實第 0–3 層建 T1 變體逐項移除工作（`experiments/probe_decoder_floor.py`，10 warmup／30 次中位數）：

| 4 層 T1 變體 | 中位數 | 權重 bytes |
|---|---:|---:|
| FP16 baseline，cache 1024 | 3.83 ms | 403 MB |
| 只有 MLP | 2.08 ms | 302 MB |
| 只有 attention | 1.93 ms | 101 MB |
| 不寫 KV cache | 3.05 ms | 369 MB |
| cache 256／4096 | 3.16／7.37 ms | 403 MB |
| LUT8 g32／LUT4 g16／LUT4 g32／LUT4 g64 | 2.78／2.73／2.71／2.73 ms | 203／101 MB |
| linear int8 per-channel | 2.73 ms | 202 MB |
| linear int4 per-block 32／64 | 7.51／7.52 ms | 113／107 MB |

判讀：（1）MLP-only 的權重串流約 145 GB/s，是 FP16 權重的有效頻寬；（2）attention 與 cache 成本隨 cache 長度線性成長，約 0.22 ms／層／1024 位置，180 秒用的 4096 cache 會讓每層 attention 成本超過權重成本；（3）全 cache 的 mul/add 寫入約 0.14 ms／層；（4）所有可用的壓縮格式（LUT4／LUT8／int8）都收斂到同一個時間，代表 ANE 解壓是按權重元素數計費（約 120 G 元素／秒），位元數只影響記憶體與 DRAM 流量；（5）linear int4 per-block 慢一倍，不走快速路徑，排除。

另外確認：以 `cache[..., pos:pos+1] = k` 追蹤出的 MIL `slice_update`（動態 begin）在本機 macOS 27 beta 用 CPU_ONLY 也載入失敗（execution plan error -14），與先前 enumerated shapes 的失敗碼相同；`index_copy_` 沒有 coremltools 轉換。動態部分寫入目前在此 host 不可用。

決策：先把 palettization 做成套件功能（`qwen3-asr-ane compress`，decoder partitions 與 LM head，encoder 保持 FP16），建 LUT8 g32 與 LUT4 g16 兩個完整 bundle，用 selection set 依預先登記的門檻挑選；cache 長度與寫入方式的結構改善另行量測。三個靜態研究 subagent 回報：mlx-community 有 4-bit／8-bit（decoder-only）轉換可作同位元 baseline；官方 PyTorch 可在 MPS bf16 執行，`evaluate.py` 加入 `--device/--dtype`；協議審查指出 prompt 上限未誠實宣告、限制錯誤訊息被扁平化等待修。

## 2026-09-13 — 24：完整壓縮 bundle 的 selection-set 結果與候選決定

以 `qwen3-asr-ane compress` 從 `qwen3-asr-1.7b-final` 產生兩個完整 bundle 並 compile：LUT8 g32（decoder 7×403→203 MB，head 622→314 MB）與 LUT4 g16（decoder 7×101 MB，head 157 MB）。MLComputePlan（此時只查了 LUT4 g16 的 `decoder_00` 與 `lm_head`）：partition 1030 個有成本算子全部 preferred ANE，head 51 個亦全部 ANE；另有 1267／142 個沒有成本估計的算子（常數）裝置為 unknown。選定候選 LUT8 的全部 8 個 graph 在第 25 節另行檢查。

同一 process 內的 smoke warm latency（各 1 warmup、5 次，中位數；GPU baseline 在同一場次序列量測，輸出文字與 ANE 相同）。預先登記時已知 partition 級 LUT8 約快 28%，但沒有任何完整 bundle 的品質結果；15% 的 latency 門檻是在那個先驗下寫的。如預期，M5 Max 的 GPU 路徑在 latency 上全部快於 ANE：

| bundle | 英文 15.05 s | 中文 4.20 s | 首次載入 |
|---|---:|---:|---:|
| FP16（起點） | 2.060 s | 0.502 s | 1.86 s（已快取） |
| LUT8 g32 | 1.502 s（−27%） | 0.368 s（−27%） | 34.1 s（首次特化） |
| LUT4 g16 | 1.543 s | 0.365 s | 34.2 s |
| MLX-Audio 0.5.3 BF16（GPU） | 0.455 s | 0.139 s | 1.31 s |
| MLX-Audio 8-bit（GPU，decoder-only 量化） | 0.296 s | 0.110 s | 0.62 s |
| MLX-Audio 4-bit（GPU，decoder-only 量化） | 0.214 s | 0.088 s | 0.57 s |
| 官方 PyTorch MPS bf16 sdpa（GPU；3 次） | 0.784 s | 0.198 s | 4.34 s |

Selection set（100 EN LibriSpeech + 100 ZH FLEURS，warmups 0、repeats 1，與既有 FP16 結果配對）：

| 候選 | EN WER | ZH CER | Δ vs FP16（pp，95% CI） | corpus RTF |
|---|---:|---:|---|---:|
| FP16（既有） | 42/2094 = 2.006% | 232/3663 = 6.333% | — | 0.135 |
| LUT8 g32 | 42/2094 = 2.006% | 231/3663 = 6.306% | EN 0.000 [0.000, 0.000]；ZH −0.027 [−0.086, 0.000] | 0.097 |
| LUT4 g16 | 49/2094 = 2.340% | 259/3663 = 7.071% | EN +0.334 [−0.146, +0.826]；ZH +0.737 [−0.077, +1.554] | 0.096 |

LUT8 的英文 normalized 錯誤數與 FP16 逐句相同（Δ=0、CI [0, 0] 是配對錯誤數相同的結果，不代表位元相同）：原始字串有 7/100 英文、7/100 中文不同，例如一個專名替換與句界移動，中文淨錯誤數少 1 個字元；對官方 FP32 仍是 −0.239／−0.437 pp。另要說明：品質 baseline 是 `qwen3-asr-1.7b-precise`（9/12 的既有結果，tokenizer 與 final 不同但 `tokenizer-equivalence.json` 證明無 context 的 batch input IDs 全部相同），latency baseline 是由 `final` 編譯的 `-compiled`，也是 LUT8 的直接父產物；corpus RTF 的 FP16 0.135 來自 9/12 的 precise run，同日的 compiled FP16 smoke RTF 為 0.133。LUT4 g16 雖然速度相同，但兩語的點估計都超過預先登記的門檻（EN ≤ +0.25、ZH ≤ +0.40），依規則淘汰，不因為記憶體較小而放寬。**候選確定為 LUT8 g32**，接著只對它跑一次 held-out gate、ABBA 能耗、記憶體、Instruments trace、串流與 compliance。速度改善來自 ANE 按元素解壓的快速路徑，與位元數無關，因此 4-bit 只有記憶體優勢；若之後要用 4-bit，需要更好的量化方法（例如逐群更小、或校準式），不是換 group size 就能解決。

## 2026-09-13 — 25：LUT8 g32 的最終 gate：held-out、能耗、記憶體、ANE trace、串流與審查修正

Held-out（各語 101–200 列，只跑一次）：LUT8 EN WER 32/2382 = 1.343%（FP16 33 = 1.385%；Δ −0.042 pp，CI [−0.136, 0.000]），ZH CER 239/3737 = 6.396%（FP16 246 = 6.583%；Δ −0.187 pp，CI [−0.590, 0.000]）；對官方 FP32 為 −0.084／−0.134 pp，CI 都跨零。200 筆全部完成、無 token 上限錯誤。原始字串有 1 EN、6 ZH 與 FP16 不同。預先登記的品質門檻全部通過。Corpus RTF 0.099（FP16 held-out 為 0.135）。

整機 SMC `PSTR` 能耗（ABBA 先導組，AC 供電、電池 80% 未充電，各 backend 兩個獨立 process block；ANE 40 輪、MLX 100 輪，因此各 block 的音訊秒數不相等，這點違反預先登記的「等工作量」，等工作量的重跑記在第 26 節）：

| block | 音訊秒 | 牆鐘 | 平均 W | J／音訊秒（邊界 ±2 s 範圍） |
|---|---:|---:|---:|---|
| ANE FP16 #1／#2 | 770 | 102.4／102.7 s | 23.8／24.5 | 3.163／3.269（3.11–3.32） |
| ANE LUT8 #1／#2 | 770 | 74.8／74.9 s | 23.4／24.7 | 2.275／2.401（2.22–2.45） |
| MLX bf16 #1／#2 | 1926 | 58.1／57.8 s | 69.9／73.9 | 2.108／2.216（2.03–2.26） |
| MLX 4-bit #1／#2 | 1926 | 30.6／30.6 s | 83.6／83.2 | 1.331／1.324（1.23–1.39） |

兩個 LUT8 block 都低於兩個 FP16 block（−27%／−28%），符合門檻；LUT8 的整機 J／音訊秒與 MLX bf16 相差約 +8%，仍高於 MLX 4-bit。ANE 的平均功率只有 GPU 路徑的三分之一，但耗時較長；整機數字包含閒置底功率，above-idle 的比較要等同場次的閒置量測（第 26 節）。前一輪 handoff 的 37.9 W／5.09 J 是在充電時量的，不能與本輪直接比較。

記憶體（`/usr/bin/time -l`，載入 + 兩段 smoke 一次）：ANE FP16 peak footprint 663 MB、ANE LUT8 661 MB、MLX bf16 5.81 GB、MLX 4-bit 3.32 GB、官方 MPS bf16 6.61 GB。ANE process 的 footprint 幾乎不隨權重大小變動，代表 ANE 端的權重配置不在 process 統計內；磁碟上 LUT8 compiled bundle 2.8 GB（FP16 4.4 GB）。系統層級的 wired memory 差異另行量測。

Instruments 孤立 trace（Xcode 26 的表名為 `ane-hw-intervals-internal`，欄位與舊版相同，`trace_ane.py` 已接受別名）：兩段 smoke 共 623 筆 ANE Prediction，與 FP16 trace 相同數量；7 個 decoder partition 各 77 次共 1.477 s、LM head 60 次 0.121 s、encoder 3 次 0.016 s、frontend 21 次 0.015 s，Instruments 的 ANE-active 區間合計 1.628 s，佔轉錄牆鐘 1.903 s 的 85.6%（FP16 trace 為約 90%，ANE 變快後 host 側的 mask 建構、輸入輸出複製與 151936 維 argmax 等固定成本相對上升）；目標 PID 的 GPU 硬體列為 0。每個預期的 Core ML 呼叫都恰好對應一筆 ANE 區間（EN 16 frontend／2 encoder／62 decoder／49 head，ZH 5／1／15／11），這個閉合是歸屬的主要依據。每次 partition 預測 2.74 ms（FP16 為 4.1 ms）。

串流：15 秒英文即時餵入，7 個 partial、closed final 在 15.63 s，event／result compliance 通過，closed 文字與 batch 相同；0.1／0.5／5／29.99 秒靜音全回傳空字串，explicit close 18 ms。`verify_plugin_runtime` 與 `standard-asr compliance run`（以環境變數指向 LUT8 bundle）exit 0。

獨立審查（唯讀）發現並已修正：（1）`compress` 可能在 group size 不整除通道數時靜默保留 FP16 權重而 manifest 仍宣稱壓縮——現在逐個 conv 驗證權重來源為 `constexpr_lut_to_dense`／blockwise 解量化，否則拒絕；（2）串流把 `ModelLimitError` 當成 `invalid_audio_or_context`——改為 `bundle_capacity_exceeded` 結構化錯誤；（3）未映射語言名只有 batch 有揭露——串流的 partial／closed 也在 `extra` 揭露；（4）`--group-size` 在兩種 scheme 語意不同——manifest 記錄 granularity 與軸；（5）壓縮 provenance 補上 coremltools／numpy 版本與逐檔 hash；CLI 拒絕 `linear` 6-bit；prompt 預算由 256 降到 128 並說明串流 prefix 的最壞情況。沒有採用 Lloyd 提前終止，因為會改變已出貨 LUT 的位元可重現性。Benchmark 審查的 LUT4 placement 混淆與英文「逐字相同」錯誤已在第 24 節更正；LUT8 全部 graph 的 placement、同日 FP16 corpus RTF 與等工作量能耗排在第 26 節補做。

## 2026-09-13 — 26：審查補做的量測、等工作量能耗、更少 partition

Benchmark 審查要求的補做：（1）LUT8 bundle 全部 10 個 graph 的 MLComputePlan——frontend 27、encoder 4740、每個 decoder partition 1030、LM head 51 個具成本算子全部 preferred ANE，沒有 CPU-only 算子；unknown 裝置的 14432 個算子是 `const`（14217）與 `constexpr_lut_to_dense`（215，LUT 解壓，計畫不給裝置與成本，但其時間包含在量到的 ANE 區間內）。（2）同日 compiled FP16 的 selection set：品質與 9/12 的 precise run 完全相同（EN 42/2094、ZH 232/3663），corpus RTF 0.133，LUT8 0.097（−27%），配對 Δ 不變。（3）floor probe 改用高斯 hidden state、cache 中段位置與真實 RoPE 重跑：baseline 3.82、attention-only 1.94、MLP-only 2.10、不寫 KV 3.06、cache 256／4096 3.14／7.36、LUT4 2.72、int8 2.73、LUT8 2.74 ms，與全零輸入的結果差在 0.1 ms 內，第 23 節的分解成立；`.floor.json` 現在記錄真實的 mode 與輸入協定。

等工作量能耗（每 block 40 輪 × 2 smoke = 770.2 音訊秒，ABBA 反向重複；同場次 30 秒閒置 4.93 W，先前 handoff 的 12.18 W 閒置是充電時量的）：ANE FP16 3.172／3.202、ANE LUT8 2.307／2.382、MLX bf16 1.970／2.011、MLX 4-bit 1.135／1.165 J 每音訊秒；above-idle 分別為 2.52／2.55、1.83／1.90、1.82／1.86、1.06／1.09。LUT8 對 FP16 −27%，兩個 block 都優於 FP16 的兩個 block，通過預先登記的能耗門檻；對 MLX bf16 gross 高 17%、above-idle 持平；MLX 4-bit 只用一半。ANE 路徑高於閒置約 19 W，不能用「低瓦數」推論省電，也不能把整機功率當成 ANE 歸屬證據；process 端 CPU 使用率已加入 `benchmark_energy.py`（rusage），第 27 節的 p14 block 會有數字。

更少 partition：用 `build_decoder_variant.py --layers-per-partition 14` 建 2 個 14 層 partition 的 T16 bundle 再 LUT8 壓縮（p14），smoke EN 1.426 s、ZH 0.351 s（7 partition LUT8 為 1.499／0.367，再快約 5%，文字相同）；28 層單一 partition 轉換與 compile 都成功，但載入時 `Failed to build the model execution plan`，放棄。`build` 指令新增 `--layers-per-partition {4,7,14}`。p14 的逐句一致性、held-out 與能耗 ABBA 排在第 27 節。

同場次其他 baseline 的 warm smoke（EN／ZH 中位數）：官方 PyTorch CPU FP32 2.743／0.800 s（1 warmup、3 次）、官方 MPS bf16 sdpa 0.784／0.198 s、MLX bf16 0.455／0.139 s、MLX 8-bit 0.296／0.110 s、MLX 4-bit 0.214／0.088 s；ANE LUT8 1.502／0.368 s 比官方 CPU 路徑快 1.8／2.2 倍，但慢於所有 GPU 路徑。KV 寫入的替代寫法：`scatter_along_axis` 模型無法載入，`torch.where` 寫回 state 無法轉換（`No matching select or slice`），全 cache 寫入成本約 4%，不再追。

## 2026-09-13 — 27：p14 成為預設、ANE+GPU 草稿假說的端到端量測

p14（2 個 14 層 partition、LUT8 g32）補齊證據後切成預設：5 個 graph 的 MLComputePlan 具成本算子全部 preferred ANE；串流 9 個事件、靜音四種長度皆空、`compliance run` exit 0；孤立 Instruments trace 238 筆 ANE Prediction（decoder 2×77、head 60、encoder 3、frontend 21，呼叫次數封閉），ANE 忙碌 1.624 s 佔轉錄牆鐘 1.802 s 的 90.1%（7 partition 為 1.628／1.903 s，85.6%）——ANE 忙碌時間幾乎不變，省下的是 Core ML 呼叫之間的 CPU 側開銷，process 平均忙碌核心數 0.081→0.050。等工作量 ABBA 對 7 partition：2.235／2.148 對 2.250／2.244 J／音訊秒，差在 1–4%，不宣稱更省電。p14 沒有單獨過門檻：權重、LUT、head 與 LUT8 g32 相同，selection＋held-out 400 句 greedy 輸出逐 token 相同，品質結論繼承。`artifacts/qwen3-asr-1.7b` 改指 `qwen3-asr-1.7b-p14-lut8-g32-compiled`，`model-aliases.json` 記錄前後 manifest sha；`build` 的 `--layers-per-partition` 預設改為 14。附帶修正：`bind_trace_evidence.py` 原本從 time-info XML 找目標 PID 而得到 null，改讀 trace toc 的 `<process>`（PID 與 exit status 現在寫進綁定檔）。

ANE+GPU 假說（使用者允許測試的第三種配置）：GPU 上以 MLX 跑 Qwen3-ASR-0.6B（bf16 載入後記憶體內 4-bit 量化 decoder，audio tower 不量化）作為草稿，每輪提出 15 個 token，ANE 上用既有的 T16 decoder graph 一次驗證，再用一個 T16 的 LUT8 compact head 取 argmax，只接受與 1.7B greedy 相同的前綴並補一個 bonus token（`speculative.py`，草稿 KV cache 用 `trim` 回退）。輸出定義上應與 serial greedy 相同，但 T16 批次驗證與逐 token 推論是不同的 Core ML 執行（且驗證 head 是另一個 LUT8 artifact），所以一致性是量出來的：selection 200 句逐 token 全部相同（`specdraft-q4-k15-selection.jsonl`），smoke 兩段文字相同。速度：selection EN 84.5→37.6 s（2.25×）、ZH 88.7→44.5 s（1.99×），接受率 78.9%／72.3%，每句平均 2.8 次驗證、36 次草稿步（每步約 2.3 ms）；smoke（p14，能耗 block 內 40 輪中位數）EN 1.421→0.602 s、ZH 0.350→0.197 s。基準是出貨 bundle 的 serial 路徑（生成時每 token 走 16 列的 T16 graph）；T1 generation 的 serial bundle 每步較便宜（第 23 節），沒有量這個對照，所以倍率是「對出貨 bundle」而不是「對最好的 serial ANE」。smoke 的倍率是 4 種草稿設定裡最好的一組（bf16／4-bit × k=7／15：EN 1.92／2.14／2.04／2.36），k=15 是 T16 block 的結構上限、4-bit 是最便宜的草稿，所以調參空間小，但正式數字用 selection 200 句。時間分解（200 句合計 82.1 s）：ANE 端 frontend／encoder／prefill 42.8 s、生成 31.2 s（其中草稿 16.8 s）、草稿 encoder＋prefill 8.1 s——prefill 現在是最大的一塊。等工作量 ABBA（p14 serial→spec→spec→serial，各 40 輪 × 2 smoke）：serial 2.225／2.207、spec 1.379／1.344 J／音訊秒（−38%；同場次閒置 5.47 W，above-idle 1.72／1.70 對 1.15／1.12），平均功率 24.0 對 32.7 W，process 忙碌核心數 0.05 對 0.28；系統 wired 記憶體推理後 6.42 GB（serial 2.57 GB），因為草稿先以 bf16 載入再量化且 MLX 的緩衝常駐。

「主要在 ANE」在草稿模式下的定義：預先登記要求先定義再量，我是看過 smoke 結果後才定義的，這點如實記錄。三個指標都報：（1）ANE 忙碌牆鐘占比，用同一 PID 的 Instruments ANE 與 GPU 區間：只跑草稿路徑的孤立 trace（`benchmark_mlx_draft.py --skip-serial`，兩段 smoke 各 1 warmup + 1 次）以 ANE Prediction 首尾為工作窗 1.703 s，ANE 區間聯集 1.041 s（61.1%）、窗內 GPU 區間聯集 0.337 s（19.8%，窗前 0.015 s 載入階段排除），其餘約 19% 是 CPU 側；（2）名目 FLOPs 占比：目標 1.7B 處理 7018 個提議＋559 個錨點 token、草稿 0.6B 跑 7200 步，ANE 約 75%；（3）權重位元組流量占比：ANE 每次驗證串流一次 1.7 GB，草稿每步約 0.3 GB，ANE 約 31%——按流量算，草稿模式的多數位元組在 GPU。這條路的代價：依賴 MLX 與 transformers 5（`mlx-audio` 0.5.3 要求 `transformers>=5.14`，與 `convert` 群組的 `<5` 衝突，只能在獨立環境 `experiments/mlx_draft/` 跑）、GPU 不再空閒、記憶體加倍。它沒有併入 plugin：預設路徑仍是純 ANE，草稿路徑以可重現的實驗與 lock file 保留，是否作為選配功能是產品決定。

獨立審查（benchmark 完整性）的結論與修正：parity 檢查是完整 token 序列相等且不符即中止，時間窗兩邊都含草稿 encoder／prefill 與驗證 head，能耗等工作量成立；但 (1) 記錄沒寫驗證 head 的身分——補了 provenance sidecar，並讓 `benchmark_mlx_draft.py` 之後每筆記錄都帶 head／manifest／草稿 revision 的 sha256 與 mlx 版本；(2) LUT8 compact head 沒直接和 LUT8 `lm_head` 對照過——`benchmark_compact_head.py` 加 `--bundle` 後對 p14 bundle 的 60 個真實狀態重跑，0 個不一致；(3) 倍率的基準要寫清楚是出貨 T16 bundle 的 serial 路徑；(4) 能耗腳本的 spec 路徑改成和 serial 一樣回傳 parse 後的文字；(5) DecoderCursor 的詞彙 chunk 大小原本寫死 8192，改為由 bundle head 推導並檢查 chunk 數一致；(6) speculative 的預算用盡改拋 `ModelLimitError`，與 serial runtime 一致。
