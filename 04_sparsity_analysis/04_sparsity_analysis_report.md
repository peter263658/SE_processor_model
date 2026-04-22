# 04_sparsity_analysis — 任務說明與分析結論

**日期**：2026-04-16  
**分析資料夾**：`04_sparsity_analysis/`  
**對應模型**：BCCTN_sep_conv_causal_slide_win_quant_256dim（epoch=134）  
**模型規格**：embed_dim=256, num_heads=16, attn_win=31, fft_len=128, depthwise=True, FP8 QAT

---

## 一、任務目標

分析模型的權重（weight）與激活值（activation）的稀疏性（sparsity），判斷是否存在**結構化稀疏（structured sparsity）**，以及在哪個 threshold 下成立。

**核心區分**：
- **結構化稀疏**（整個 channel / head / block 都為零）→ 硬體可以直接跳過整塊計算，等於「免費加速」
- **隨機/非結構化稀疏**（個別 weight 為零，但排列散亂）→ 需要 bitmap + 壓縮格式，硬體開銷高，通常不划算

**分析對象涵蓋**：
1. 每層 weight sparsity（多個 threshold）
2. Block-level sparsity（block size = 4, 8, 16, 32）
3. Channel-level sparsity（整個 output channel 是否全為零）
4. Activation sparsity（對 20 個測試樣本做 forward，記錄 PReLU 輸出）
5. Attention score sparsity（W=31 滑動窗口內的分數分佈）
6. 結構分類與 pruning candidate 清單

---

## 二、分析腳本

**主腳本**：`04_sparsity_analysis/sparsity_analyzer.py`

執行指令：
```bash
mamba activate bcctn
cd /raid/R12K41024/04_sparsity_analysis

python sparsity_analyzer.py \
    --checkpoint /raid/R12K41024/BCCTN_sep_conv_causal_slide_win_quant_256dim/outputs/2026-03-30/15-18-17/logs/lightning_logs/version_0/checkpoints/epoch=134-step=130680.ckpt \
    --test_dir /raid/R12K41024/BCCTN/Dataset/noisy_testset/snr_0dB \
    --num_samples 20 \
    --results_dir ./results
```

**輸出檔案**（位於 `results/`）：
| 檔案 | 內容 |
|------|------|
| `weight_sparsity.csv` | 每層 weight sparsity（5 個 threshold） |
| `block_sparsity.csv` | Block-level sparsity（block size 4/8/16/32） |
| `activation_sparsity.csv` | 每層 activation sparsity（20 samples） |
| `attention_sparsity.csv` | W=31 窗口內 attention score 近零比例 |
| `attention_heatmap.png` | 平均 attention pattern 熱力圖 |
| `sparsity_summary.json` | 每層結構化/隨機分類結果 |
| `pruning_candidates.txt` | 可移除的 channel / head 清單 |
| `weight_sparsity_plot.png` | 每層 sparsity 長條圖（threshold=5e-2） |
| `summary.md` | 詳細文字結論 |

---

## 三、分析結論

### 3.1 Weight Sparsity

| Threshold | 平均稀疏率 | 最大稀疏率 | 有 channel 稀疏的層數 |
|-----------|-----------|-----------|----------------------|
| 1e-4      | 極低       | —         | 0                    |
| 1e-3      | 極低       | —         | 0                    |
| 1e-2      | **6.9%**   | 80%       | 0                    |
| 5e-2      | **25.5%**  | 100%      | 1                    |
| 1e-1      | **43.1%**  | 100%      | 4                    |

**結論**：在實用 threshold（1e-2）下，平均 weight sparsity 僅 7%，整體偏低。雖然在寬鬆 threshold（1e-1）下能達到 43%，但這些零散分佈在各個位置，**不具備結構性**。

---

### 3.2 結構化 vs. 隨機稀疏分類

共分析 52 層：

| 分類 | 層數 | 說明 |
|------|------|------|
| Dense（無可利用稀疏） | 6 | Attention 投影層（Q/K/V/O），稀疏率趨近 0 |
| **Random sparse（隨機稀疏）** | **43** | 數值上稀疏，但無 channel / block 結構 |
| Block-sparse（block=4, thr=5e-2） | 2 | `decoder.5.0.pointwise_r/i`，比例約 50% |
| Channel-prunable | 1 | `decoder.5.0.depthwise_i`，只有 1 個 filter |

**結論**：**稀疏性幾乎全為隨機非結構化**。主要的 encoder/decoder 卷積層（depthwise + pointwise）在合理 threshold 下均無可跳過的完整 channel 或 block。唯一的 block-sparse 層和 channel-prunable 層都在最後一個 decoder 輸出 stage，參數量極小，實際加速效益可以忽略。

---

### 3.3 Attention Head 分析

- Q/K/V/O 投影矩陣（256×256 複數）在所有 threshold 下稀疏率趨近 **0**（dense）。
- 無任何 head 的 output projection column group 全為近零。
- **全部 16 個 head 都有貢獻，無法移除任何 head**。

---

### 3.4 Activation Sparsity（PReLU 輸出）

PReLU 激活值在 threshold=0.1 時，各層稀疏率：

| 位置 | 稀疏率範圍 |
|------|-----------|
| Encoder PReLU（real/imag） | 33% – 59% |
| Decoder PReLU（real/imag） | 19% – 50% |
| `decoder.2.2.i_prelu`（最高） | 45%（甚至在 1e-3 threshold 下） |

關鍵觀察：所有層的 `consistent_dead_channels`（跨 sample 始終為零的 channel）均接近 **0**。

**結論**：PReLU 激活值數值上確實有 20–60% 在 threshold=0.1 以下，但這是**動態稀疏**（依輸入而異），並非固定的 dead channel。硬體若要利用這個特性，需要 per-sample 零值偵測與動態跳過電路，ASIC 實作成本高，通常不值得。

---

### 3.5 Attention Window 分析

- 對 20 個測試樣本取平均 attention pattern，已儲存為 `results/attention_heatmap.png`。
- 在 W=31 滑動窗口**內部**，分數 < 0.01 的比例 = **0.0**（零）。

**結論**：窗口內所有 31 個位置的 attention score 在 softmax 後均不接近零，**所有位置都有貢獻**。目前 W=31 已是一種設計上的結構化稀疏（attention 矩陣 S×S 中大部分被 mask 為 -inf）。若要將窗口從 W=31 縮減到 W=16 或 W=8，需要實際評估語音品質（ILD/IPD/STOI），風險較高。

---

## 四、硬體影響總結

| 發現 | 對 ASIC 的影響 |
|------|---------------|
| Weight 稀疏率 7–43%，但隨機 | 需要 bitmap + 壓縮儲存，硬體開銷高，**不建議實作** |
| 無 channel 級結構稀疏 | Encoder/decoder 卷積無法做 channel-skip 加速 |
| Block-sparse 只在最後輸出層 | 只有 2 個小層（1×8, 1×8 weights），MAC 節省可忽略 |
| PReLU activation 稀疏 20–60% | 動態稀疏，需要 zero-skip 邏輯，**ASIC 成本不划算** |
| Attention W=31 窗口內全部有效 | 無法安全縮小窗口，需重新 training 驗證 |
| 16 個 attention head 全部有用 | 無 head pruning 機會 |

**總體結論（供顧問討論）**：

> 在目前訓練狀態下，BCCTN 模型的稀疏性**無法被硬體有效利用**。模型整體為密集（dense）分佈，稀疏是訓練收斂後的隨機現象，非設計刻意引入。若要獲得可加速的結構化稀疏，需要：
>
> 1. **Structured pruning 訓練**：加入 L1 regularization 或 group sparsity penalty，強制整個 channel 的 weight 趨向零，再 fine-tune 恢復品質。
> 2. **Weight sharing / low-rank 分解**：以矩陣分解替代稀疏，對 ASIC 更友善。
> 3. **Window 縮減評估**：若 W=31 → W=16 的語音品質損失可接受，attention 矩陣的結構稀疏度可提升（從 ~80% → ~90%），這是目前**唯一有潛力的結構化稀疏方向**，但需重新 training 確認。

---

## 五、與現有優先序的對應

根據 CLAUDE.md 的分析方向優先序：

| 優先序 | 方向 | 本次結論 |
|--------|------|---------|
| T3 | `04_sparsity_analysis/` | **稀疏結構不存在，硬體投入無效益。建議 T3 降低優先序，聚焦 T1/T2。** |

相比之下，`02_weights_bitwidth/`（W4A8 混合精度）和 `01_utilization_analysis/`（MAC 瓶頸）的硬體投入回報率更高。
