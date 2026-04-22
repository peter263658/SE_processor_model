# 03 — Weight Dynamic Range Analysis：任務說明與分析結論

> **用途**：供顧問討論使用。分析日期：2026-04-16。  
> **對應資料夾**：`03_dynamic_range/`  
> **checkpoint**：`model.ckpt`（BCCTN causal sliding-window variant，embed_dim=256，32 heads）

---

## 一、任務目標

分析 BCCTN 模型每一層 weight 的動態範圍（dynamic range），以決定：

1. 各層適合使用 **FP** 還是 **INT4** 量化格式
2. Scalar factor 的粒度：per-tensor / per-channel / per-group
3. W4A8 硬體設計中 **group_size**（scalar factor 分組大小）的最佳選擇
4. 複數層（real / imaginary）能否共用同一組 scale factor
5. MHA 的 32 個 head 是否需要 per-head scaling

分析對象為 **6 階段 Complex Depthwise Separable Conv Encoder-Decoder + MHA**，weight 共 104 個 tensor（不含 BN/PReLU）。

---

## 二、模型結構與 Weight 分類

| 類別 | 層名規律 | 代表 shape | 數量 |
|------|---------|-----------|------|
| Depthwise Conv (real/imag) | `depthwise_r/i` | `(Cout, 1, 5, 1)` | 24 |
| Pointwise Conv (real/imag) | `pointwise_r/i` | `(Cout, Cin, 1, 1)` | 24 |
| MHA in_proj (Q/K/V × re/im) | `in_proj_weight` | `(256, 256)` × 6 | 6 |
| MHA out_proj (re/im) | `out_proj` | `(256, 256)` × 2 | 2 |
| MHA proj_in/out (complex) | `input_proj`, `output_proj` | `(256, 256)` × 4 | 4 |
| BN / PReLU（不納入量化分析）| `bn_r/i.weight`, `r/i_prelu` | scalar/small | — |

---

## 三、五個關鍵問題與結論

### Q1：各層的 channel / head 分布是否均勻？（決定 per-channel vs per-tensor scaling）

#### Conv 層（per-output-channel 分析）

- **42/42 個** conv weight tensor 的 cross-channel variance（`var_max`）均 > 1e-4 → **全部需要 per-channel 或 per-group scaling**，不能用 per-tensor
- 差異最大的層集中在 encoder 前幾層（channel 數少、個別 channel 影響大）與 decoder pointwise conv：

  | 層 | var_max (cross-channel) |
  |----|------------------------|
  | encoder.0.pointwise_i | **5.26e-01** |
  | encoder.0.pointwise_r | 4.94e-01 |
  | encoder.2.depthwise_i | 6.74e-02 |
  | encoder.1.depthwise_r | 5.50e-02 |

- Depthwise conv 由於 `Cin=1`，每個 output channel 只有 5 個 weight → 各 channel 差異天然較大，**必須 per-channel scale**

#### MHA 層（per-head 分析，32 heads）

- **9/12** head-group tensor 的 cross-head `var_max` > 1e-3 → **非均勻**，部分 head 的 weight 幅度明顯大於其他
- 差異最大的 head-group：

  | 層 | var_max |
  |----|---------|
  | output_proj.imag | **6.00e-03** |
  | MultiheadAttention_im.in_proj_weight.V | 3.73e-03 |
  | MultiheadAttention_im.out_proj | 2.53e-03 |

- 只有 `input_proj.real/.imag`（複數 projection 層）和 `in_proj_weight.V (re)` 三組接近均勻（var_max ≈ 7.5e-4 ~ 9.9e-4）

**→ 結論**：MHA 需要 per-head scaling；直接用 per-tensor scale 會因部分 head outlier 而損失精度。

---

### Q2：W4 Scalar Factor 的最佳 group_size？

量化方式：INT4 對稱量化，scale = max(|group|) / 7。

| Group Size | 平均 MSE | 相對 per-tensor | 相對 gsz=16 |
|-----------|---------|----------------|-------------|
| per-tensor（整層 1 個 scale） | 1.314e-03 | 基準 (1.00×) | — |
| gsz = 16 | **9.572e-04** | 0.73× | 最精確 |
| gsz = 32 | 1.048e-03 | 0.80× | 1.09× |
| gsz = 64 | 1.113e-03 | 0.85× | 1.16× |
| gsz = 128 | 1.170e-03 | 0.89× | 1.22× |
| gsz = 256 | 1.202e-03 | 0.91× | 1.26× |

> 說明：group_size 越小 → scale 數越多 → MSE 越低 → 量化精度越高，但硬體 overhead 越高（需搬 scale 資料）。

**→ 結論**：所有 group_size 的 MSE 均在 gsz=16（最細粒度）的 1.26× 以內，遠低於 2× 門檻。  
**建議使用 `group_size = 256`**：硬體 overhead 最低，量化精度損失僅 ~9%（相較 per-tensor 節省 8.5%），符合 <10% overhead 設計目標。

若需更高精度，`gsz = 32` 是次優選擇（MSE 僅 1.09× gsz=16，overhead 適中）。

---

### Q3：哪些層有 outlier，需要 FP 格式？

判斷標準：kurtosis > 5（重尾分布）→ 建議 FP 格式。

**5 個層建議使用 FP 格式，全部集中在 decoder 的 pointwise conv：**

| 層 | Kurtosis | 形狀 |
|----|---------|------|
| decoder.3.pointwise_i | **12.33** | (16, 64, 1, 1) |
| decoder.4.pointwise_i | 9.36 | (8, 32, 1, 1) |
| decoder.3.pointwise_r | 8.46 | (16, 64, 1, 1) |
| decoder.0.pointwise_i | 7.33 | (128, 256, 1, 1) |
| decoder.0.pointwise_r | 6.38 | (128, 256, 1, 1) |

- **共同特點**：全部為 decoder 的 pointwise conv，實數（real）和虛數（imag）均有問題，且集中在 decoder stage 0、3、4（skip connection 入口處）
- Encoder 所有層 kurtosis < 3，MHA 所有層 kurtosis < 2 → 均屬正常分布，INT4 可處理

**→ 結論**：99/104 個 weight tensor 適合 INT4；5 個 decoder pointwise conv 需考慮 FP8 或保留 FP16，或在 QAT 時特別處理。

---

### Q4：Real 和 Imaginary 的分布是否相似？可否共用 scale？

只有兩層為複數型態（`complex64`）的 weight，其餘均已在模型內分離為 real/imag 參數：

| 層 | std_ratio (real/imag) | kurtosis_diff | 分布相似？ |
|----|----------------------|---------------|----------|
| `mattn.input_proj` | 0.988 | 0.010 | ✅ 相似 |
| `mattn.output_proj` | 1.003 | 0.089 | ✅ 相似 |

**→ 結論**：Real 和 Imaginary 分布高度相似（std 差異 < 2%，kurtosis 差 < 0.1）。  
**Real 與 Imaginary 可共用同一組 scale factor**，無需各自獨立 scaling。

---

### Q5：MHA 的 32 個 Head 的 weight 幅度是否均勻？

| Tensor | var_max | 是否均勻 |
|--------|---------|---------|
| input_proj.real / .imag | 9.4e-04 / 9.9e-04 | ✅ 均勻 |
| in_proj_weight_re.V | 7.5e-04 | ✅ 均勻 |
| in_proj_weight_re.Q | 1.0e-03 | ⚠ 邊界 |
| in_proj_weight_re.K | 1.6e-03 | ❌ 非均勻 |
| in_proj_weight_im.Q/K/V | 1.5e-03 ~ 3.7e-03 | ❌ 非均勻 |
| out_proj_re / out_proj_im | 1.2e-03 / 2.5e-03 | ❌ 非均勻 |
| output_proj.real / .imag | 1.9e-03 / 6.0e-03 | ❌ 非均勻 |

**→ 結論**：多數 MHA tensor 的 head 間 weight 幅度不均勻（9/12 非均勻），尤其 imaginary 路徑差異更大。  
**建議實作 per-head scaling**。若硬體不支援，per-group（gsz=32~64 沿 head 維度）是退而求其次的選項。

---

## 四、綜合建議（Hardware-Aware Quantization 策略）

| 層類型 | 建議格式 | Scale 粒度 | Group Size | 備註 |
|--------|---------|-----------|-----------|------|
| Encoder depthwise conv (×12) | INT4 | per-channel | N/A | Cout 少，per-channel 必要 |
| Encoder pointwise conv (×12) | INT4 | per-group | 256 | 大 tensor，gsz=256 夠用 |
| Decoder depthwise conv (×12) | INT4 | per-channel | N/A | 同 encoder |
| Decoder pointwise conv stage 0,3,4 (×5) | **FP8** | per-channel | N/A | kurtosis > 5，有 outlier |
| Decoder pointwise conv 其他 (×7) | INT4 | per-group | 256 | 正常分布 |
| MHA Wq/Wk/Wv/Wo (re+im × 4) | INT4 | per-head | N/A | 9/12 head-group 非均勻 |
| MHA proj_in/proj_out (complex) | INT4 | per-head | N/A | 均勻但保守起見建議 per-head |
| BN scale | FP16（保留） | — | — | 參數量極少，不值得量化 |

### 關鍵設計數字

- **W4 group_size = 256**：scale overhead ≈ 1/256 ≈ 0.4%（遠低於 10% 目標）
- **5 個 FP 層**佔總 conv weight 參數量較小比例，FP8 的 SRAM 開銷可接受
- **MHA 共需 12 組 per-head scale**（每組 32 個 scale values）→ 額外 384 Bytes，可接受

---

## 五、後續行動項目（供討論）

1. **FP decoder pointwise conv 的處理方式**：
   - 選項 A：保留 FP8（e4m3），完全不量化這 5 層
   - 選項 B：INT4 + 較細 per-channel scale（kurtosis 高但 per-channel 可能緩解）
   - 選項 C：QAT 中針對這 5 層加 L2 regularization 壓低 kurtosis

2. **MHA per-head scale 的硬體實作**：
   - 是否能在 weight-stationary pipeline 中，在每個 head 切換時更換 scale？
   - 或改用 per-group（gsz=32 沿 row 方向）作為硬體友好的近似？

3. **Dynamic range ratio 很高**（部分層達 10^5）：
   - 這表示有極小的非零 weight，但這些極小值在 INT4 量化後會被 round 到 0 → 實際上是「自然稀疏化」
   - 需評估這些接近 0 的 weight 是否對輸出有意義

4. **接下來的 re-training 驗證（方向 T2）**：
   - 依此分析結果設定 mixed-precision QAT：FP8 for 5 decoder layers，INT4+gsz=256 for others
   - 觀察 ILD/IPD 保持度是否受 decoder outlier 層影響

---

## 六、輸出檔案索引

| 檔案 | 說明 |
|------|------|
| `results/dynamic_range_table.csv` | 104 個 tensor 的全域統計（min/max/std/kurtosis/dynamic_range_ratio） |
| `results/per_channel_variance.csv` | 42 個 conv 層的 cross-channel variance |
| `results/per_head_variance.csv` | 12 個 MHA head-group 的 cross-head variance |
| `results/real_imag_comparison.csv` | 2 個複數層的 real/imag 相似度對比 |
| `results/group_size_sensitivity.csv` | 每層在各 group_size 下的 W4-INT4 MSE |
| `results/group_size_plot.png` | MSE vs group_size 趨勢圖 |
| `results/format_recommendation.json` | 每層 FP/INT4 + scale 粒度建議（機器可讀） |
| `results/histograms/` | 104 張 per-layer weight magnitude histogram |
| `results/summary.md` | 分析腳本自動生成的摘要（英文，對應 5 個 key questions） |
| `dynamic_range_analyzer.py` | 分析腳本（可重跑：`python dynamic_range_analyzer.py --checkpoint model.ckpt`） |
