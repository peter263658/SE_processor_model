# BCCTN 模型 FP8 量化交接報告

**專案路徑**: `/raid/R12K41024/BCCTN_sep_conv_causal_slide_win_quant`
**日期**: 2026-03-23

---

## 1. 量化策略總覽

| 項目 | 規格 |
|------|------|
| **量化格式** | IEEE Float8 **E4M3** (`float8_e4m3fn`) |
| **可表示範圍** | ±448.0（最小正規數 ≈ 2⁻⁶） |
| **量化方式** | Quantization-Aware Training (QAT) |
| **量化目標** | **W8A8**（權重 8-bit + 激活值 8-bit） |
| **Scaling 策略** | Per-tensor dynamic scaling |
| **反向傳播** | Straight-Through Estimator (STE) |

### E4M3 位元編排
```
[S][EEEE][MMM]    S:符號(1) E:指數(4,bias=7) M:尾數(3) = 8 bits
```

### Scaling 公式
```
scale = 448.0 / max(|tensor|)
fp8_value = cast_to_e4m3(tensor × scale)
dequantized = fp8_value / scale
```

每個 tensor 獨立計算 scale（per-tensor），部署時需為每層儲存對應的 scale factor。

---

## 2. 模型架構與量化層對照表

架構：**Encoder(×6) → MultiAttnBlock → Decoder(×6)**

### 2.1 Encoder（每層結構 × 6 層）

kernel_num = [2, 16, 32, 64, 128, 256, 256]，kernel_size = 5

| 子模組 | PyTorch 類型 | 量化 | 說明 |
|--------|-------------|------|------|
| `depthwise_r` | `Conv2d` | 🔴 W8A8 | Depthwise conv（real），groups=in_ch |
| `depthwise_i` | `Conv2d` | 🔴 W8A8 | Depthwise conv（imag） |
| `pointwise_r` | `Conv2d` | 🔴 W8A8 | 1×1 pointwise（real） |
| `pointwise_i` | `Conv2d` | 🔴 W8A8 | 1×1 pointwise（imag） |
| `bn_r` / `bn_i` | `BatchNorm2d` | ⚪ FP32 | BN 不量化 |
| `r_prelu` / `i_prelu` | `PReLU` | ⚪ FP32 | 激活不量化 |

> 每層 4 個量化 Conv2d × 6 層 = **24 個量化層**

### 2.2 MultiAttnBlock

| 子模組 | 類型 | 資料型態 | 量化 | 說明 |
|--------|------|---------|------|------|
| `input_proj` | `Linear` | **complex64** | 🔴 W8A8 | real/imag 分別量化 |
| `MHA_re.in_proj` | `Linear` | float32 | 🔴 W8A8 | Q/K/V 投影（real 路徑, embed=512, heads=32） |
| `MHA_re.out_proj` | `Linear` | float32 | 🔴 W8A8 | 輸出投影（real） |
| `MHA_im.in_proj` | `Linear` | float32 | 🔴 W8A8 | Q/K/V 投影（imag 路徑） |
| `MHA_im.out_proj` | `Linear` | float32 | 🔴 W8A8 | 輸出投影（imag） |
| `output_proj` | `Linear` | **complex64** | 🔴 W8A8 | real/imag 分別量化 |

### 2.3 Decoder（每層結構 × 6 層）

| 子模組 | 類型 | 量化 | 說明 |
|--------|------|------|------|
| `pointwise_r` / `pointwise_i` | `Conv2d` | 🔴 W8A8 | 1×1 轉換通道數 |
| `depthwise_r` / `depthwise_i` | `ConvTranspose2d` | 🔴 W8A8 | Transposed depthwise |
| `bn_r` / `bn_i` | `BatchNorm2d` | ⚪ FP32 | 前 5 層有 BN |
| `r_prelu` / `i_prelu` | `PReLU` | ⚪ FP32 | 前 5 層有 |

> 每層 4 個量化層 × 6 層 = **24 個量化層**

### 2.4 量化統計

| 類別 | 量化層數 |
|------|---------|
| Encoder Conv2d | 24 |
| MultiAttnBlock Linear + MHA | ~6 |
| Decoder Conv2d / ConvTranspose2d | 24 |
| **合計** | **~54** |
| 未量化（BN, PReLU, STFT/ISTFT） | ~30+ |

---

## 3. Complex64 處理方式（硬體關鍵）

### 3.1 Complex 卷積 = 4 次 Real 卷積

```
output_real = conv_r(input.real) − conv_i(input.imag)
output_imag = conv_r(input.imag) + conv_i(input.real)
```

**硬體意義**：
- `conv_r` 和 `conv_i` 權重皆為 **real FP8**
- 輸入拆成 real/imag 兩個 real FP8 tensor
- 硬體**不需要 native complex 支援**，只需 real FP8 MAC
- 每個 complex conv = **4 次** real conv + 加/減法

### 3.2 Complex Linear（input_proj / output_proj）

權重的 real 和 imag 部分**各自獨立量化**，各有獨立 scale：

```
W = W_real + j·W_imag    （各有各的 scale）
Y_real = W_real @ X_real − W_imag @ X_imag
Y_imag = W_real @ X_imag + W_imag @ X_real
```

匯出的權重檔案中 complex 層拆成 `_real.txt` 和 `_imag.txt`。

### 3.3 Complex MultiheadAttention

內部 2 個獨立 `nn.MultiheadAttention`（MHA_re、MHA_im），同樣展開為 4 次 real MHA。

---

## 4. 權重匯出格式

使用 `export_fp8_weights.py` 匯出。每層產生 `.txt` + `.csv`：

```
# Layer : encoder.model.0.0.depthwise_r.weight
# Shape : [1, 1, 5, 1]
# Scale : 89.600000          ← 硬體推論需要此值
# Format: FP8 E4M3
#
# Index      Hex        Decimal
  0          0x3C        1.00000000e+00    ← Hex 可直接寫入硬體記憶體
  1          0xBC       -1.00000000e+00    ← Decimal 用於驗證
```

---

## 5. 檔案清單

### 新增

| 檔案 | 功能 |
|------|------|
| `DCNN/utils/fp8_qat.py` | QAT 核心：STE、Parametrization、Complex 處理、enable_fp8_qat() |
| `export_fp8_weights.py` | QAT checkpoint → per-layer FP8 hex+decimal 權重檔 |
| `run_export_fp8_weights.sh` | 匯出 shell wrapper |

### 修改

| 檔案 | 內容 |
|------|------|
| `DCNN/trainer.py` | 第 73-77 行呼叫 `enable_fp8_qat()` |
| `config/config.yaml` | `fp8_qat: true`, `fp8_dtype: "e4m3"` |
| `config/training.yaml` | 同上 |

---

## 6. 硬體設計注意事項

1. **MAC 精度**：FP8 × FP8 → 建議累加器用 FP16/FP32 防溢位
2. **Per-tensor Scale**：每層有獨立 scale factor（FP32），推論前需載入
3. **Complex 運算**：只需 real FP8 MAC，每個 complex op = 4× real op
4. **BatchNorm**：推論時可 fuse 進 Conv（合併為 affine），建議預處理
5. **Causal Attention**：sliding window mask（window_size=31），硬體需支援
6. **STFT/ISTFT**：使用 FFT，建議獨立模組處理，不做 FP8 量化
7. **DesignWare overflow**：E4M3 max=448.0，QAT 後溢位機率極低
