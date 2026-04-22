# 01 — MAC Utilization & Pipeline Bottleneck Analysis
**分析日期**：2026-04-16  
**模型**：BCCTN（Binaural Complex Convolutional Transformer Network）  
**目標硬體**：256-MAC ASIC，weight-stationary dataflow，64-bit DMA bus

---

## 任務說明

本次分析的目標是：在不重新訓練的前提下，以 **per-layer 粒度** 量化 BCCTN 模型在 256-MAC PE array 上的實際利用率，並找出 MAC idle cycles 的成因與位置。

### 硬體假設

| 參數 | 數值 |
|---|---|
| PE array size | 256 MACs |
| DMA bus width | 64-bit（8 bytes/cycle）|
| Weight precision | FP8（1 byte/weight）|
| Activation precision | FP8（1 byte/element）|
| On-chip WBUF | 64 KB |
| On-chip ABUF | 32 KB（估計）|
| Clock | 200 MHz |

### 模型設定

| 參數 | 數值 |
|---|---|
| fft_len | 128（F = 65 bins）|
| win_len / win_inc | 128 / 80 |
| kernel_num | [16, 32, 64, 128, 256, 256]（6 stages）|
| kernel_size | 5×1（depthwise separable）|
| embed_dim | 512 |
| num_heads | 32 |
| attn_win | 31（sliding window）|
| binaural | True（L+R 雙通道）|
| 輸入長度 | 1 秒 stereo @ 16kHz → T = 200 frames |

---

## 分析方法

針對每一層計算：

- **Theoretical MAC cycles** = total_MACs / 256
- **Weight DMA cycles** = weight_bytes / 8
- **Activation DMA cycles** = (input_act_bytes + output_act_bytes) / 8
- **MAC utilization** = compute_cycles / (compute_cycles + max(weight_dma, act_dma))
- **W4A8 utilization**：同上，但 weight bytes 減半（4-bit weight）

Attention 模組進一步拆解為：input_proj、QKV projection、QK^T、Softmax（LUT）、Score×V、out_proj、output_proj 七個子操作。

---

## 整體數字

| 指標 | 數值 |
|---|---|
| 總 MACs | 1,687,427,200（1.69 G）|
| 計算週期（compute） | 6,591,514 |
| Weight DMA 週期 | 435,372 |
| Activation DMA 週期 | 1,169,500 |
| DMA : Compute 比例 | 0.24 : 1 |
| 整體瓶頸類型 | **Compute-bound**（80% compute / 20% DMA）|

### MAC 分布

| 模組 | MACs | 佔比 |
|---|---|---|
| Attention | 1,309,081,600 | **77.6%** |
| Decoder | 253,558,400 | 15.0% |
| Encoder | 124,787,200 | 7.4% |

---

## 核心發現

### 發現 1：Attention 主導算力，但內部瓶頸不均勻

Attention 佔全模型 77.6% 的 MACs。但內部不同子操作的利用率差異顯著：

| 子操作 | 利用率 | 瓶頸類型 |
|---|---|---|
| input_proj（flatten → embed）| 92.6% | compute-bound |
| QKV projection（Wq/Wk/Wv）| 92.6% | compute-bound |
| output projection | 92.6% | compute-bound |
| QK^T 矩陣乘 | 49.6% | activation DMA |
| Score × V 矩陣乘 | **33.0%** | activation DMA |
| Softmax | 0 MACs | LUT（非 MAC）|

**結論**：權重投影（linear matmul）接近飽和；QK^T 和 Score×V 是 activation DMA 瓶頸——sliding window 縮短了 sequence，但 T=200 的 activation 搬運量仍然壓著 DMA bus。

---

### 發現 2：Depthwise Conv 有結構性的 PE 利用率問題

所有 encoder/decoder 的 depthwise 子層利用率僅 **13–17%**，且是 **activation DMA bottleneck**。

根本原因：
- kernel = 5×1（frequency axis），每個 spatial tile 只有 **5 個 MAC 有效**
- PE array 有 256 個 MAC → **單 tile 的空置率為 98%**
- Depthwise 不做跨 channel reduction，無法靠增加 channel 數來填滿 PE

這是 depthwise separable conv 映射到 weight-stationary 陣列的**結構性限制**，不是模型設計缺陷。

---

### 發現 3：Pointwise Conv 利用率隨深度快速成長

Pointwise（1×1）conv 的利用率取決於 C_in × C_out 的大小：

| Layer | C_in → C_out | 利用率 | 瓶頸 |
|---|---|---|---|
| Enc1 pointwise | 1 → 8 | **10.0%** | activation DMA |
| Enc2 pointwise | 8 → 16 | 40.0% | activation DMA |
| Enc3 pointwise | 16 → 32 | 57.1% | compute |
| Enc4 pointwise | 32 → 64 | 72.7% | compute |
| Enc5 pointwise | 64 → 128 | 84.2% | compute |
| Enc6 pointwise | 128 → 128 | **88.9%** | compute |

越深的 pointwise 層越接近理論上限，但最早的幾層（channel 數少、頻率解析度高）受 activation DMA 嚴重拖累。

---

### 發現 4：整體 compute-bound 掩蓋了層級間的嚴重不均

整體 DMA:Compute = 0.24:1，看起來是 compute-bound，但這是加總後的平均值。實際上：

- **Early conv layers**（低 channel、高解析度）：activation DMA bottleneck，利用率 10–17%
- **Deep pointwise layers**（高 channel）：compute-bound，利用率 73–89%
- **Attention weights（QKV/out_proj）**：compute-bound，利用率 92.6%
- **Attention score（QK^T, Score×V）**：activation DMA bottleneck，利用率 33–50%

系統的整體效率被高利用率的 attention weight 層拉高，但存在幾個明顯的**效率低谷**。

---

### 發現 5：W4A8 對 MAC 利用率提升有限，但對 SRAM 有效

| 精度 | 平均利用率（非零層）|
|---|---|
| FP8 | 45.3% |
| W4A8 | 45.4% |
| 差值 | +0.1 pp |

W4A8 幾乎沒有改善利用率，原因：

- 利用率低的層（early conv）是 **activation DMA bottleneck**，W4 減少的是 weight bytes，對 activation 搬運無效
- 利用率高的層（QKV proj）是 **compute-bound**，weight DMA 已經不是瓶頸

**W4A8 的真正收益在於 SRAM 佔用**：MHA 四個矩陣（Wq/Wk/Wv/Wo）FP8 下共 512KB（雙路複數），W4 壓縮到 256KB，可讓 weight 更容易留在 WBUF 內，減少 DRAM round-trip，降低實測 latency。

---

## 三大優先改進方向

### 方向 A：解決 Depthwise Conv 的 PE 空置問題（高優先）

現況：每個 depthwise tile 只用到 5/256 個 MAC（2%）。

可行策略：
- **im2col + batched matmul**：把多個 channel tile 打包成矩陣，讓 PE array 一次處理多個 channel，提升並行度
- 或在排程上將多個 depthwise 層**流水線化**，讓 PE array 在一個 cycle 同時處理不同 channel 的 tiles

---

### 方向 B：解決 QK^T / Score×V 的 Activation DMA 問題（高優先）

現況：QK^T 利用率 49.6%，Score×V 利用率 33%，均因 activation 搬運過慢。

可行策略：
- **On-chip tiling**：確保 Q/K/V 的 sliding window 片段完整放入 ABUF（32KB），避免每個 window step 都要 DRAM round-trip
- 當前 ABUF=32KB，T=200、embed_dim=512、FP8 → Q 或 K 單一 sliding window 片段 = 31×512 = 15.9KB，**剛好可以放入 ABUF**；應確認硬體 tiling 策略確實利用此特性

---

### 方向 C：Early/Late Conv 層的 Activation DMA 優化（中優先）

現況：Enc1/Dec6 利用率僅 10%，activation tensor 大（F×T 高解析度）但 channel 少。

可行策略：
- 考慮在 Enc1 前增加 **spatial downsampling**（在 frequency 或 time axis），在 channel 還少的時候先縮小 activation 大小
- 評估將 Enc1 的 pointwise 與 depthwise 合併為單一 regular conv，讓 C_in × kH 的 reduction 可以填滿更多 MAC

---

## 待進一步驗證的問題

1. **W4A8 re-training 後的語音品質**：ILD/IPD 保持度是否下降？（尤其 QKV projection 從 FP8 → W4）
2. **實測 DRAM 搬運次數**：分析假設 activation 每次都從 DRAM 搬；若 ABUF tiling 策略良好，實際 DMA 週期可能更少
3. **Softmax 的 LUT 硬體成本**：估計 ~8 LUT ops/element，T=200、32 heads、window=31 → 總 LUT ops ≈ 1.6 M；需確認 LUT pipeline 是否成為獨立瓶頸
4. **Depthwise 的 im2col 可行性**：需確認 WBUF 是否能容納重排後的 weight tile，以及 reshape 的控制邏輯開銷

---

## 產出檔案

| 檔案 | 說明 |
|---|---|
| `utilization_profiler.py` | 分析腳本（可重跑，不需 GPU）|
| `results/utilization_table.csv` | 55 層的完整 per-layer 數據 |
| `results/utilization_summary.txt` | 自動產生的關鍵數字摘要 |
| `results/utilization_plot.png` | Per-layer 利用率 bar chart（FP8 vs W4A8）+ cycle breakdown |
