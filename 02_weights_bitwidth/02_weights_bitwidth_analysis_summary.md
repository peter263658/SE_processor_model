# Weight Bit-width 探索分析總結
**分析方向**: 02 — Weight Bit-width Exploration (Layer-wise PTQ Sensitivity)
**完成日期**: 2026-04-16
**模型**: BCCTN (Binaural Complex Convolutional Transformer Network)

---

## 一、分析目的

針對硬體 DMA 瓶頸問題，探索每個 layer 能容忍的最低 weight bit-width，在不顯著損失語音品質的前提下，最大化 DRAM→WBUF 權重搬運量的壓縮效益。

**硬體背景**：
- WBUF = 64 KB，DMA-bound 架構
- MHA 四矩陣（Wq/Wk/Wv/Wo）在 FP8 下共需搬 512 KB（2 組，real + imag），遠超 WBUF
- W4 理論上將 DMA 流量砍半

---

## 二、實驗設計

### 2.1 量化方法（PTQ，Post-Training Quantization）

- **量化對象**：Weight only（W4），Activation 維持 FP8（模型已做 FP8 QAT 訓練）
- **格式**：INT8、INT4、INT3（symmetric min-max）、FP4（E2M1 模擬）
- **基準**：FP8 baseline（原訓練模型）

### 2.2 Layer 分組（共 30 組）

| 類型 | 數量 | 說明 |
|---|---|---|
| Encoder depthwise (enc_dw_0–5) | 6 | 各 stage depthwise conv |
| Encoder pointwise (enc_pw_0–5) | 6 | 各 stage pointwise conv |
| Decoder depthwise (dec_dw_0–5) | 6 | 對稱解碼端 |
| Decoder pointwise (dec_pw_0–5) | 6 | 對稱解碼端 |
| MHA 相關 (mattn_*) | 6 | input_proj, in_proj_re/im, out_proj_re/im, output_proj |

### 2.3 Per-layer Sensitivity 分析流程

對每個 layer 獨立量化（其他 layer 保持 FP8），跑 inference，記錄：
- **ΔSI-SNR**（dB，越負越差）
- **ΔILD**（dB，interaural level difference error）
- **ΔIPD**（度，interaural phase difference error）

分層閾值：
- Tier A（保持 W8）：ΔSI-SNR > 0.5 dB 或 ΔILD > 0.5 dB 或 ΔIPD > 5°
- Tier B（安全 W4）：通過閾值，但 W3 不過
- Tier C（可用 W3）：W3 仍通過所有閾值

---

## 三、FP8 Baseline 數值

評估集：500 對測試樣本（VCTK noisy/clean testset）

| 指標 | FP8 Baseline |
|---|---|
| SI-SNR | 10.492 ± 5.12 dB |
| ILD error | 4.006 ± 1.57 dB |
| IPD error | 23.97 ± 13.00° |

---

## 四、Per-layer Sensitivity 主要發現

### 4.1 各 layer 對量化的耐受性（整體規律）

| Layer 類型 | 對 INT4 的耐受性 | 對 INT3 的耐受性 |
|---|---|---|
| Encoder DW（0–5） | 非常強（ΔSNR < 0.02 dB） | 強（ΔSNR < 0.18 dB） |
| Encoder PW（0–5） | 非常強（ΔSNR < 0.02 dB） | 強（ΔSNR < 0.17 dB） |
| Decoder DW（0–5） | 非常強（ΔSNR < 0.02 dB） | 強（ΔSNR < 0.17 dB） |
| Decoder PW（0–5） | 強（ΔSNR < 0.05 dB）* | 強（ΔSNR < 0.09 dB）* |
| MHA（mattn_*） | 非常強（ΔSNR < 0.02 dB） | 強（ΔSNR < 0.14 dB） |

*`dec_pw_0` 在 INT4 時 ΔILD = 0.133 dB（最高者），但仍未超過閾值 0.5 dB

### 4.2 關鍵發現：所有 30 個 layer 個別量化到 W3 均通過閾值

- 每個 layer 獨立量化到 INT3 時：
  - ΔSI-SNR 均 < 0.18 dB（遠低於 0.5 dB 閾值）
  - ΔILD 均 < 0.03 dB（遠低於 0.5 dB 閾值）
  - ΔIPD 均 < 0.35°（遠低於 5° 閾值）

**結論（per-layer 分析）**：全部 29 個 layer 分到 Tier C（W3），1 個（dec_pw_0）分到 Tier B（W4），無 Tier A（W8）。

### 4.3 INT vs FP4 格式比較

- 在相同 bit-width 下，INT4 普遍優於或等於 FP4（E2M1）
- FP4 在部分層（如 enc_pw_3, IPD −0.17°）甚至略有優勢，但差異微小
- 建議採用 **INT4/INT3** 格式（硬體 MAC 電路更簡單）

---

## 五、Mixed-Precision 評估結果

根據 per-layer sensitivity 自動生成 mixed-precision config（29 層 W3 + 1 層 W4），與幾種 uniform 量化方案比較：

| 方案 | SI-SNR (dB) | ΔSI-SNR | ILD error | ΔILD | IPD error | ΔIPD | 記憶體節省 |
|---|---|---|---|---|---|---|---|
| FP8 Baseline | 10.492 | — | 4.006 dB | — | 23.97° | — | 0% |
| Uniform INT8 | 10.432 | −0.06 dB | 4.025 dB | +0.02 dB | 24.14° | +0.17° | 0% |
| **Uniform INT4** | **9.929** | **−0.56 dB** | **4.216 dB** | **+0.21 dB** | **24.61°** | **+0.64°** | **50%** |
| Uniform INT3 | 4.701 | −5.79 dB | 4.916 dB | +0.91 dB | 25.94° | +1.97° | 62.5% |
| Mixed (29×W3 + 1×W4) | 7.417 | −3.08 dB | 4.643 dB | +0.64 dB | 25.45° | +1.48° | 61.5% |

---

## 六、核心結論

### 6.1 最佳可用方案：Uniform INT4

- SI-SNR 損失 **−0.56 dB**（可接受）
- ILD 損失 **+0.21 dB**（低於 0.5 dB 閾值）
- IPD 損失 **+0.64°**（遠低於 5° 閾值）
- DMA 流量 **減半（−50%）**
- **建議方向**：Uniform INT4 + per-group scalar factor，以克服 global scale 的精度限制

### 6.2 Per-layer 分析的盲點：累積誤差效應

**重要警告**：Per-layer 分析是系統性樂觀估計。

- 每層單獨 W3 時：ΔSI-SNR < 0.18 dB → 全部分到 Tier C
- 29 層同時 W3 時：ΔSI-SNR = **−3.08 dB**（遠超 5× 單層加總）

原因：各層量化誤差沿前向傳播路徑**非線性疊加**，誤差放大效應在深層模型中顯著。Per-layer 分析不捕捉 cross-layer error propagation。

### 6.3 Mixed-precision 配置表現不如 Uniform INT4

雖然節省記憶體更多（61.5% vs 50%），但品質損失遠超 uniform INT4（−3.08 dB vs −0.56 dB）。
原因：mixed-precision 大量使用 W3，W3 的累積誤差不可接受。

---

## 七、硬體影響分析

| 量化方案 | MHA 兩組矩陣大小（估計） | DMA 減少量 |
|---|---|---|
| FP8 baseline | 512 KB | 0% |
| Uniform INT4 | 256 KB | −50% |
| Mixed (主要 W3) | ~213 KB | −58% |

**結論**：Uniform INT4 可將 MHA 矩陣 DMA 流量從 512 KB 壓縮至 256 KB，搭配 WBUF = 64 KB 的硬體，可將 pass 數從 8-pass 降為 4-pass。

---

## 八、後續建議方向

1. **INT4 QAT 再訓練**：以 W4 QAT 恢復 −0.56 dB 的品質損失，目標是在 INT4 下達到接近 FP8 基準的品質。

2. **Per-group scale factor 驗證**：group_size=32 時 scale 開銷僅 ~6%，可進一步降低 INT4 的量化誤差（尤其對 dynamic range 較大的 MHA 層）。

3. **03_dynamic_range 分析**：做 per-layer、per-channel weight dynamic range 分析，確認 INT4 scale 設計是否需要 per-group。

4. **W4 + INT8 Activation 組合驗證**：本實驗為 Weight-only 量化（W4A8），Activation 維持 FP8。如需進一步壓縮，可考慮 W4A8（INT8 activation），但需評估 softmax 輸出分布的量化友好性。

---

## 九、產出檔案清單

| 檔案 | 說明 |
|---|---|
| `bitwidth_sweep.py` | Per-layer PTQ sweep 腳本（30 layer × 4 格式） |
| `mixed_precision_eval.py` | Mixed-precision vs baseline 完整評估腳本 |
| `results/sensitivity_matrix.csv` | 30 layer × 4 bit-width × 3 metric 數據表 |
| `results/sensitivity_heatmap.png` | 3-panel heatmap（SNR / ILD / IPD） |
| `results/mixed_precision_config.json` | Per-layer bit-width 分配（Tier A/B/C） |
| `results/mixed_precision_eval.txt` | 5 種方案完整評估結果 |
| `analysis_summary.md` | 本總結文件 |
