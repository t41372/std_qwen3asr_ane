# 第二輪優化：結果與驗證狀態

本輪在 M5 Max、64 GB、macOS 27.0 上執行，基準是 p14／LUT8 g32／T16／
cache 1024。主要可用改善目前來自縮小短語音所需的 KV cache；並沒有找到
能在保持既定品質下，大幅加速所有長度語音的通用 decoder 改法。

本輪已完成。候選不因單項速度或記憶體結果而取代 general 預設；
未通過的路線保留診斷結果。預先設定的範圍與門檻見
[optimization-round2.md](optimization-round2.md)。

## 短語音 profile

`short-dictation` 明確限定每段最多 12 秒、cache 512，預設輸出 budget 128。
`general` 維持 30 秒、cache 1024、budget 256。超過音訊上限或實際
prompt/output 容量會報錯；沒有自動切段、靜默截斷或自動切換 profile。

以下比較兩邊都使用 128-token budget。從既有 400 筆回歸集按音訊長度篩出
284 筆（153 英文、131 中文），每筆各測三輪，baseline/candidate 交錯執行。
先取每筆的三輪中位數，再計算整體分位數，避免把重複次數當成更多獨立樣本。

| 指標 | 基準 | short-dictation | 延遲降低 |
|---|---:|---:|---:|
| p50 | 0.747078 s | 0.605828 s | 18.9% |
| p90 | 1.098984 s | 0.890738 s | 18.9% |
| p95 | 1.199201 s | 0.968368 s | 19.3% |

852 組 measured attempts 的輸出 token 全部一致。候選在打開新 held-out 前
已[凍結](evidence/round2/short-candidate-freeze.json)；額外 137 筆符合 12 秒
限制的 held-out（75 英文、62 中文）也全部一致。這是 421 個不同 utterance，
不能寫成完整 400 筆 general 工作範圍都得到同樣加速。

初始記錄確認兩邊在同一步正常遇到 EOS，但沒有保存終止 EOS 的具體 ID。
同一凍結候選的補充診斷已完成：421 筆的輸出 token 與 EOS ID 全部一致。
沒有據此重選或調整候選。

新增粵語、日語、韓語、德語、法語、西班牙語各 50 筆，共 300 筆，也全部
通過 token 與 EOS 配對。另列的 12 個增益／噪音／非語音案例同樣一致。
乾淨語音共 721 筆；情境變體不視為額外獨立的自然語音樣本。

另外檢查兩筆真實英／中音訊與十二個情境案例的實際 MLState：全部 KV
值有限，已使用位置的 key/value 數值與 baseline 完全一致。

獨立程序量測的 peak process footprint 約由 671 MiB 降至 579 MiB；
系統 wired memory 的推論後增量約為 2382 與 2369 MiB，大致持平。
程序記憶體不包含所有 ANE 配置，不能把前者的下降宣稱為整台機器同幅節省。
五個模型的 compute plan 中所有已知運算均指向 ANE。Instruments 另記錄到
五個模型共 266 次 ANE prediction，與工作量呼叫數相符。沒有可歸屬於目標
PID 的 GPU interval；另有 17 個 GPU row 無法識別 process。ANE 表本身沒有
PID，因此以隔離工作量、精確模型 label 與呼叫數作歸屬，不宣稱追蹤能排除
所有未知事件。使用修正過 cache 長度說明的 `short-trace-attribution-v2.json`。

相同兩筆短語音、128-token budget、每區塊 60 輪的 ABBA 能耗結果：

| 區塊 | baseline（J/audio-s）| short（J/audio-s）|
|---|---:|---:|
| 第一組 | 4.44997 | 3.67248 |
| 第二組 | 4.33837 | 3.57714 |
| 區塊中位數 | 4.39417 | 3.62481 |

全機 PSTR 積分估計低約 17.5%；即使將起訖邊界各平移 ±2 秒，所有 short
區塊仍低於所有 baseline 區塊。不過 idle brackets 約在 18–29 W 間變動，
桌面活動持續存在，這不是低背景的系統 idle 測量，更不是獨立 ANE 能耗或
校準電表結果。不要與舊報告中不同音訊、不同桌面條件的 J/audio-s 混比。

凍結產物已複製到明確 profile 的預設目錄，並以真實引擎驗證可直接使用
`profile="short-dictation"`。general 的預設產物維持原樣。

## Streaming 與 draft 載入

Streaming session 現在保存可重用的 frontend chunk、encoder window 和穩定
STFT／原始 log-mel prefix。只有 padded input、mask 和 dtype 完全相等時
才重用圖輸出。整段的 log-mel 動態範圍裁切仍會重新計算，因此後來出現
較大訊號時，先前 chunk 可能正確失效。FFT prefix 保留在 100-frame
邊界，避免不同 FFT batch 對齊造成浮點尾數差異。

加入 8 秒靜音前綴的英／中串流對照，在每次 partial 都維持 token 與 raw
text 一致。這是 cumulative replay 與 cache 的直接比較，不能用 batch
與 streaming 文字相同取代這項驗證，因為 streaming 本身還有 prefix rollback。

`prepare()` 只載入 ANE target；即使設定了 draft，streaming 也不會載入 GPU
模型。第一次 batch 才載入 draft。Draft 缺失或載入失敗會清楚回報，target
仍可供 streaming 使用。長時間 loaded idle 與首次喚醒測試已完成，見下文；
本輪不加入自動 eviction。

英／中真實 paced streaming 的 general 與 short-dictation 事件內容、最終文字
完全一致，且都通過事件與結果合規。中文例子在兩個 profile 中都有既有的
數字格式差異：streaming 寫「八零二点一一a」，batch 寫「802.11a」。
因此不能把「streaming 與 batch 相同」當成這項優化的普遍保證。

## 其他候選的實際結果

| 候選 | 觀察 | 決策 |
|---|---|---|
| T64 prefill + T1 generation | selection 200 的 p50 約快 16.6%，但兩筆英文 token 不同 | 未通過精確 token 門檻 |
| T64 prefill + T16 generation + compact head | 200 筆 token 一致，p50 約快 3%；argmax 配置到 CPU | 不採用 |
| T1 compact head | 60 個真實 hidden row 的 greedy 選擇一致；head 2.623 → 2.819 ms | 沒有速度優勢，且未通過配置門檻 |
| 最後 p14 decoder + compact head 融合 | 117 個大卷積全部 LUT8；19 個 argmax 與 index stack 仍在 CPU | 在配置門檻停止 |
| 校準 W8A8 head | compute plan 已知運算全指向 ANE；同 decoder 換 head 的整體 smoke 慢約 6–7% | 不擴大 |
| 選擇性四層 W8A8 decoder | 真實 KV 下 3.274 → 3.248 ms，改善不足 1%，數值已有變化 | 不擴大 |
| Swift decoder bridge | 100 個真實 generation step，完整 hidden parity；隔離 ABBA 約快 1.4% | 局部診斷，不能支持重寫插件 |
| LUT6 g32 | EN／ZH 門檻通過、wired memory 中位數減少約 17.2%；新增語言出現退步 | 不推薦為通用方案 |

T1 的兩筆失敗為 `librispeech-clean-test-2169` 與 `librispeech-clean-test-158`，
分別涉及標點／大小寫和逗號。用 T16 prefill + T1 generation 可重現，改回
T16 generation 則兩筆都與基準一致。因此不能因 WER 正規化忽略了標點，
就聲稱 T1 路線保持精確輸出。

W8A8 校準使用獨立的 200 筆英／中資料：head 保存首／中／尾的 600 個
hidden row；decoder 收集第一個分區前四層、16 個 projection-input tap
在真實 prefill/generation 中的全域極值。手動聚合避免 SDK 9 實驗性
collector 在合併前覆寫 min/max 的問題。W8A8 探測保留 FP16 normalization、
softmax、RoPE 和 KV；沒有把未校準的隨機 hidden 當成品質證據。

Tiny state／enumerated shape 實驗連 fixed-width control 都載入失敗，因此
只能記錄這些 probe 未能建立可用執行計畫，不能推論 Core ML 普遍不支援
state + enumerated shapes。完整現有 decoder 的失敗另有保留。

LUT6 selection 200 相對 LUT8 的英文 WER delta 為 +0.048 pp
（95% CI 約 −0.151 至 +0.262 pp），中文 CER 為 +0.082 pp
（約 −0.209 至 +0.425 pp）。這些是 paired utterance bootstrap 的工程門檻，
不是統計上證明 non-inferiority。候選與新的 100 EN／100 ZH held-out 已
[另行凍結](evidence/round2/lut6-candidate-freeze.json)，不重用短語音已看過的資料。

完整 400 筆的三輪交錯回歸也已完成：英文 WER +0.045 pp（CI 上界
+0.201 pp），中文 CER +0.095 pp（CI 上界 +0.453 pp），符合原定工程門檻。
每筆三輪中位數的整體 p50 為 0.837873 → 0.827465 s（約快 1.2%），
p95 為 1.643183 → 1.618333 s。英文 p95 約快 1.6%，中文 p95 約慢 1.0%，
均符合記憶體候選的延遲門檻。這仍不是通用的大幅加速方案。

新的 200 筆 held-out 也符合既定門檻：英文 WER +0.135 pp（95% CI
−0.086 至 +0.393 pp），中文 CER 差異 0（−0.268 至 +0.240 pp）。
因此可以說未超過預先接受的品質退步幅度，不能說 LUT6 的輸出完全不變。

新增語言各 50 筆的結果揭露了限制。以下均為 candidate 減 baseline：

| 語言 | 指標 | 差異（pp）| 95% CI（pp）|
|---|---|---:|---:|
| 粵語 | CER | +1.380 | −0.002 至 +3.233 |
| 日語 | CER | +0.676 | −0.112 至 +1.560 |
| 韓語 | WER | 0 | −1.171 至 +1.183 |
| 德語 | WER | −0.105 | −1.332 至 +0.933 |
| 法語 | WER | +0.661 | +0.084 至 +1.303 |
| 西班牙語 | WER | 0 | −0.404 至 +0.491 |

這些額外樣本較少，但不能忽略已觀察到的退步。因此不把 LUT6 當成通用
替代品或新增正式 profile。保留產物與結果，並以九個已知退步案例做有界的
component 混合精度定位；那組刻意挑出的案例不能用來估算整體品質。

有界定位已完成：九個已知退步案例中，只把 head 恢復 LUT8 仍有九個
退步；把前 14 層恢復 LUT8 可令六個案例 token 完全回到 baseline，但
仍有三個退步；只恢復後 14 層則仍有七個退步。恢復半個 decoder 的方案
只降低約 10.95% 的模型 weight payload，還不是系統記憶體的實測降幅。
因此不擴大這些混合方案，也不把 LUT6 加入正式 profile。

## Loaded idle 與喚醒

每個 age 都從上一個請求結束後重新計時；不是從啟動時算累積時間。
下表是一筆固定英語音訊、每個 age 一次首次請求的觀察，不能當作 p95。

| 條件 | 暖機參考 | 閒置 1 分鐘後 | 10 分鐘後 | 60 分鐘後 |
|---|---:|---:|---:|---:|
| target-only | 1.417 s | 1.502 s | 1.505 s | 1.521 s |
| draft-loaded | 0.612 s | 0.723 s | 0.735 s | 0.722 s |

六次喚醒文字都與暖機時一致，兩個引擎都成功關閉。兩個條件實際都確認 `prepare()` 不載入
draft，configured draft 只在第一次 batch 載入。這次 draft 的第一次 batch
為 1.692 s，之後的暖機參考為 0.612 s；首次 batch 的載入成本不能混入
穩態速度比較。

桌面活動持續變化。target-only 的全機平均 PSTR 從 unloaded bracket 的
16.56 W、60 分鐘 loaded 窗口的 11.16 W，到 released bracket 的 9.16 W，
不能據此說載入模型會降低功率，也不能把窗口差異直接當成模型 idle 成本。
該 60 分鐘內的呼叫程序 CPU 累積 1.40 秒，包含量測器自身工作；不包含
Core ML daemon 與 kernel。`vm_stat` 也是全域計數，長時間漂移不能全算給模型。
draft-loaded 的 60 分鐘平均 PSTR 為 7.51 W，呼叫程序 CPU 累積 1.56 秒。
這不能解讀成 draft 比 target-only 更省 idle power：兩個時段的桌面活動不同。
兩個 60 分鐘窗口各有 3600 筆功率樣本，最大取樣間隔約 1.010 秒；實際
閒置 age 皆不少於要求值。六次首次請求在這筆音訊上，較各自暖機參考多
約 0.085–0.123 秒，沒有據此宣稱一般化的延遲分位數。

關閉前至釋放後的 wired 計數下降約 2432 MiB（target）及 5181 MiB（draft）；
這些是全域 snapshot 差，不是精確的單模型記憶體歸屬。兩個環境的 Python／
套件版本保留於 `evidence/round2/idle-environments.json`。

## 完成狀態

短語音 profile、精確 audio cache、延後 draft 載入已完成實作與上述驗證。
T1、compact／fused head、W8A8、LUT6 與本輪粗粒度混合配置保留研究結果，
沒有取代 general 預設。Native 只完成有界診斷，沒有重寫插件。

精簡可追溯數據收錄於 [measurements.json](evidence/round2/measurements.json)；
未完成或不存在的報表保留為 missing，不視為通過。完整逐筆輸出、power
samples、trace 和模型 payload 留在 `artifacts/evaluation/round2/`。
