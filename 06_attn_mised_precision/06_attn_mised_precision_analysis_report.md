# Attention Mixed-Precision & Sparse Attention 分析報告
**供顧問討論用 — 2026-04-16**

---

## 一、任務背景與目標

本資料夾針對 BCCTN 模型的 **Multi-Head Attention (MHA)** 模組進行混合精度與稀疏化分析，目標是在不損失語音品質的前提下，找到最省面積、最省功耗的硬體精度配置。

### 模型架構重點

- **模型**：Binaural Complex Convolutional Transformer Network (BCCTN)，用於雙耳語音增強
- **MHA 配置**：embed_dim=256，num_heads=16，head_dim=16，sliding window W=31
- **複數運算**：MHA 執行 4 次實數 pass（re×re、re×im、im×re、im×im），分析時對四路結果取平均
- **FP8 QAT**：模型已預先套用 FP8 Fake Quantization（e4m3 格式），分析在此基礎上進行

### 分析的五個子任務

| # | 任務 | 方法 |
|---|---|---|
| 1 | 各子操作動態範圍分析 | Hook 截取 Q/K/V/QKT/attn_weights/SV 中間值 |
| 2 | 精度靈敏度模擬 | 對 QKV、累加器、SV 各別注入 fake quantization |
| 3 | Attention Entropy / Effective Span | Shannon entropy + top-k L1 近似誤差 |
| 4 | Top-k 稀疏注意力全模型評估 | 替換 softmax 輸出，計算 SI-SNR / ILD / IPD |
| 5 | 混合精度硬體建議 | 整合上述結果，輸出 per-stage 精度配置 |

---

## 二、分析結果

### 2.1 各子操作動態範圍

| 子操作 | min | max | std | 動態範圍 | log₂(|max|) |
|---|---|---|---|---|---|
| Q | -33.86 | +35.38 | 3.62 | 69.2 | 5.14 |
| K | -54.17 | +46.02 | 4.85 | 100.2 | 5.76 |
| V | -65.49 | +50.93 | 5.73 | 116.4 | 6.03 |
| **QK^T** | **-480.9** | **+911.5** | 43.81 | **1392.4** | **9.83** |
| attn_weights | 0.0 | 1.0 | 0.039 | 1.0 | 0.0 |
| Score×V | -65.49 | +50.93 | 5.94 | 116.4 | 6.03 |

**關鍵發現**：
- **QK^T 動態範圍高達 1392**（log₂ ≈ 9.83），是所有子操作中最大的。這是因為 QK^T 為矩陣乘法的累加結果，值域不受 softmax 壓縮。
- **attn_weights（softmax 後）恆在 [0, 1]**，std 只有 0.039，是最好量化的階段。
- **V 的 std（5.73）略高於 Q（3.62）**，但 V 乘以的是 [0,1] 的注意力權重，整體輸出 Score×V 與 V 同量級。

---

### 2.2 精度靈敏度分析

**實驗設計**：對 Q/K/V 投影、QK^T 累加器、Score×V 三個階段分別注入不同精度的 fake quantization，量測 MHA 輸出 MSE 相對於 fp32 基準的變化。

**FP32 基準 MSE**：0.7046（由 FP8 QAT wrapper 與直接 fp32 計算的誤差決定）

**最低可用精度（MSE 在閾值 1.057 以內）**：

| 階段 | 最低精度 | 說明 |
|---|---|---|
| Q/K/V 投影 | **fp16** | fp8 已超出可接受誤差 |
| QK^T 累加器 | **int16** | fp8/int8 因動態範圍 1392 >> fp8 max 448 而失敗 |
| Score×V 輸出 | **fp16** | softmax 後值域小，但 fp8 仍超限 |

**為何 QK^T 不能用 fp8 / int8？**
- fp8_e4m3 max = 448，而 QK^T max = 911.5 → **直接溢位**
- fp8_e5m2 max = 57344 雖不溢位，但指數位數不足導致精度差
- int8 qmax = 127，scale = 911.5/127 ≈ 7.2，LSB = 7.2 → 量化誤差過大

---

### 2.3 Attention Entropy 與有效注意力跨度

**逐 head 分析結果**：

| Head | Entropy (bits) | Effective Span | 類型 |
|---|---|---|---|
| 0 | 1.03 | 0.0058 | sharp |
| 1 | 0.81 | 0.0050 | sharp |
| 2 | **2.61** | 0.0109 | **flat** |
| 3 | **2.84** | 0.0115 | **flat** |
| 4 | 0.65 | 0.0046 | sharp |
| 5 | **2.13** | 0.0091 | **flat** |
| 6 | 0.50 | 0.0041 | sharp |
| 7 | 0.62 | 0.0045 | sharp |
| 8 | 0.70 | 0.0047 | sharp |
| 9 | 1.61 | 0.0074 | sharp |
| 10 | 0.62 | 0.0045 | sharp |
| 11 | **0.39** | 0.0037 | sharp（最 sharp） |
| 12 | 1.14 | 0.0061 | sharp |
| 13 | 1.12 | 0.0060 | sharp |
| 14 | **2.91** | 0.0113 | **flat（最 flat）** |
| 15 | 1.13 | 0.0062 | sharp |

- **Mean entropy：1.30 bits**（理論最大值 4.95 bits，只用了 26%）
- **Sharp heads（entropy < 2 bits）：12/16**（75%）
- **Flat heads（entropy ≥ 2 bits）：4/16**（heads 2, 3, 5, 14）

**Top-k 近似誤差**：

| k | L1 近似誤差 |
|---|---|
| 4 | 0.00041 |
| 8 | 0.00017 |
| 16 | 0.00004 |

即使只保留每個 head 在 31 個 window 位置中最大的 4 個，近似誤差僅 0.00041。

---

### 2.4 Top-k 稀疏注意力全模型評估

**實驗**：將 softmax 輸出中非 top-k 的權重歸零並重新歸一化，然後跑完整模型 inference 計算語音品質指標。

| k | MHA MSE | SI-SNR L (dB) | SI-SNR R (dB) | ILD err (dB) | IPD err (rad) |
|---|---|---|---|---|---|
| **4** | 5.58 | **9.50** | **8.98** | **0.275** | **1.455** |
| 8 | 2.24 | 9.52 | 8.98 | 0.246 | 1.456 |
| 12 | 1.30 | 9.52 | 8.98 | 0.247 | 1.455 |
| 16 | 0.94 | 9.50 | 8.98 | 0.280 | 1.453 |
| 24 | 0.73 | 9.51 | 8.96 | 0.251 | 1.455 |
| 31 | 0.70 | 9.49 | 8.97 | 0.279 | 1.453 |
| **full** | **0.70** | **9.51** | **8.98** | **0.284** | **1.453** |

**關鍵發現**：
- **k=4 的 SI-SNR、ILD、IPD 與 full attention 幾乎相同**（SI-SNR 差距 < 0.02 dB，ILD 差距 < 0.01 dB）
- MHA 輸出的 MSE 在 k=4 時為 5.58（比 full 高），但此 MSE 並未轉化成可聽見的品質差距
- 這意味著 **31 個 window 位置中平均只有 4 個位置是真正有效的注意力目標**

---

### 2.5 混合精度配置建議

| Pipeline 階段 | 建議精度 | 理由 |
|---|---|---|
| Wq/Wk/Wv/Wo 權重 | **fp16** | 動態範圍中等，fp8 會造成品質下降 |
| Q/K/V 激活值 | **fp16** | 同上 |
| **QK^T 累加器** | **int16** | 動態範圍 1392，fp8/int8 均溢位或精度不足 |
| Score×V 輸出 | **fp16** | 稍保守，未來可嘗試 int8 |
| proj_in / proj_out | **fp16** | 邊界層，保持 fp16 |

---

## 三、硬體效益估算

| 指標 | 數值 | 說明 |
|---|---|---|
| MAC 面積 vs FP32 | **0.25×（省 75%）** | fp16 MAC 面積約為 fp32 的 1/4 |
| 權重記憶體壓縮 | **2.0×** | fp32 → fp16，SRAM/DRAM 使用量減半 |
| Attention 計算量節省 | **87.1%** | top-k=4 只算 4/31 個位置 |

---

## 四、待討論問題

以下問題建議與顧問討論：

### Q1：QK^T int16 累加器的硬體實現方式？
目前建議用 **int16 固定點累加器**來承接 QK^T。但具體的 scale factor 策略尚未確定：
- 選項 A：per-layer static scale（最簡單，但需要離線校準）
- 選項 B：per-head dynamic scale（需要額外 pass 或 online max-tracking）
- 是否有 int16 → fp16 的轉換硬體可以復用？

### Q2：4 個 flat heads（2, 3, 5, 14）是否需要特殊處理？
這 4 個 flat heads entropy > 2 bits，使用 top-k=4 時的近似誤差較其他 heads 大。
- 硬體是否可以支援 per-head 的 k 值（sharp heads 用 k=4，flat heads 用 k=8）？
- 或者統一用 k=8 並接受稍多的計算量？

### Q3：V 是否可以比 Q/K 更積極壓縮？
分析顯示 V 的 std（5.73）高於 Q（3.62），但 V 乘的是 [0,1] 的注意力權重，最終輸出 Score×V 的量級與 V 相同。
- 若 Q/K 用 fp16，V 嘗試 int8 + per-group scale，是否划算？
- 這對硬體設計有何影響（QKV 投影是否可以共用同一個 MAC array）？

### Q4：Top-k 稀疏注意力的硬體實現複雜度？
Top-k masking 需要對 31 個 attention weights 做排序，找出前 k 大的位置。
- 排序電路（sorting network）vs. 逐步比較的選取方式，面積差距如何？
- 另一個選項：**threshold-based sparse attention**（只保留 > threshold 的權重），硬體上比 top-k 更簡單，但 threshold 需要動態調整。

### Q5：這個分析是基於 2 個測試樣本，結論是否夠可信？
由於計算資源限制，目前的精度靈敏度分析只跑了少量樣本。正式的 20-sample 全量分析還未完成。
- 建議在決定最終硬體精度配置前，先跑完整 20-sample 版本確認數字穩定。

---

## 五、下一步行動建議

| 優先序 | 行動 | 負責 |
|---|---|---|
| **P1** | 跑 20-sample 完整分析，確認數字穩定性 | 研究生 |
| **P2** | 討論 QK^T int16 累加器的 scale 策略 | 顧問 + 研究生 |
| **P3** | 評估 per-head k 值的硬體可行性（4 flat heads） | 顧問 |
| **P4** | 嘗試 V int8 + per-group scale，re-training 驗證 | 研究生 |
| **P5** | 研究 threshold-based sparse attention 電路面積 | 顧問 |

---

*分析腳本：`attn_precision_analyzer.py`*
*原始數據：`results/` 資料夾*
*分析日期：2026-04-16*
