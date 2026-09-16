# Fusions Applied by `ckpt2hwdat.py`

> **ERRATUM (2026-07-07)**：本文 F 項寫的 STFT/iSTFT window「Hann」有誤——訓練與 deploy 實際使用 **Hamming**（07 handoff 起即為 Hamming；RTL 亦以 Hamming 實作並 bit-exact 對過 deploy，見 19_algo_final T4 spec 511/512）。引用本文件時以 Hamming 為準；原文保留不改（append-only 慣例）。

本文件列出匯出硬體 bin 時，`ckpt2hwdat.py` 幫忙折掉的常數（fuse）以及**沒有 fuse、要由硬體 pipeline 處理**的運算，好讓硬體端清楚知道匯出資料的語意。

---

## Fusion 1 — BatchNorm + Conv bias → post-MAC per-channel FMA（兩組）

### 來源
Encoder 每個 stage：`Depthwise → Pointwise → NaiveComplexBatchNorm2d → ComplexPReLU`
Decoder 每個 stage（除最後一 stage）：`Pointwise → Depthwise → NaiveComplexBatchNorm2d → ComplexPReLU`

`NaiveComplexBatchNorm2d` 對 real 跟 imag **各有獨立** BN（`bn_r`, `bn_i`），每個 BN 有 `weight (γ), bias (β), running_mean (μ), running_var (σ²)`。

### 為何不直接 fold 進 conv weight
Complex conv 的 `fc_r.weight` 同時參與 real 輸出 (`fc_r(x.real) - fc_i(x.imag)`) 和 imag 輸出 (`fc_r(x.imag) + fc_i(x.real)`)。BN_r 只縮放 real 輸出、BN_i 只縮放 imag 輸出，兩邊的縮放因子不同，**不能同時折進同一組 `fc_r.weight`**，否則破壞掉另一條路徑。

### Fusion 做法
把 BN 跟兩個 complex conv bias 折進 **post-MAC per-channel FMA（兩組：real 輸出用一組、imag 輸出用一組）**。

設 `b_r`, `b_i` 是 `pointwise_r.bias`、`pointwise_i.bias`（encoder 時用 pw，decoder 時用 dw）：

```
s_r[c] = γ_r[c] / sqrt(σ_r²[c] + eps)
s_i[c] = γ_i[c] / sqrt(σ_i²[c] + eps)

# 複數乘法展開後，conv bias 對 real/imag 輸出的貢獻不同：
#   real 輸出： (b_r - b_i)
#   imag 輸出： (b_r + b_i)
bn_bias_r[c] = s_r[c] * (b_r[c] - b_i[c] - μ_r[c]) + β_r[c]
bn_bias_i[c] = s_i[c] * (b_r[c] + b_i[c] - μ_i[c]) + β_i[c]
```

### 硬體執行順序（post-MAC）
每個 complex MAC output 完成後：
```
out_real[c] = bn_scale_r[c] * mac_real[c] + bn_bias_r[c]
out_imag[c] = bn_scale_i[c] * mac_imag[c] + bn_bias_i[c]
```

### 儲存位置
```
blocks/{encoder|decoder}/stage_{idx}/
  bn_scale_r.fp16.bin    # shape: (C_out,)
  bn_bias_r.fp16.bin     # shape: (C_out,)
  bn_scale_i.fp16.bin
  bn_bias_i.fp16.bin
```

### 沒有 BN 的 stage（decoder 最後一 stage）
`has_bn=false`、`has_prelu=false`。這時：
- `bn_scale_r = bn_scale_i = 1` (所有 channel)
- `bn_bias_r[c] = b_r[c] - b_i[c]`（純 conv bias 的 real 部分）
- `bn_bias_i[c] = b_r[c] + b_i[c]`

硬體 pipeline 可以**永遠走同一條 post-MAC FMA path**，用這個統一介面讀出即可。

---

## Fusion 2 — 1/√(head_dim) → Q 部分 pre-scale

### 來源
Multi-Head Attention 內 `attn = softmax(Q·K^T / sqrt(D) + mask)`，其中 `D = embed_dim / num_heads = 256/16 = 16`，故 `1/sqrt(D) = 0.25`。

### Fusion 做法
`in_proj_weight` shape = `(3*E, E)`，前 `E` 列是 Q、中間 `E` 列是 K、後 `E` 列是 V。per-head INT4 scale shape = `(3*H, 1, 1)`，前 `H`=16 是 Q 的 head scale。

兩件事：
1. 把 scale 的前 H 個（Q heads）乘上 `1/sqrt(D)`：
   ```
   scale_fused[0:H] = scale[0:H] / sqrt(D)
   ```
2. `in_proj_bias` 的前 E 個（Q rows）乘上 `1/sqrt(D)`：
   ```
   bias_fused[0:E] = bias[0:E] / sqrt(D)
   ```

**INT4 整數值不變**。因為 `round(W/scale)` 在 `(W, scale) → (k·W, k·scale)` 下不變。我們只是把放大倍率從 runtime 除法搬到了儲存的 scale 上。

### 硬體執行
Hardware compute `attn_raw = Q_dequant · K_dequant^T`（**不用再除 sqrt(D)**）之後直接進 softmax。

### 儲存位置
```
weights/mattn.mattn.MultiheadAttention_{re,im}/in_proj_weight/
  w_int4.packed.bin
  scale.fp16.bin        # 已 pre-scaled，前 16 個 head 除過 sqrt(16)
  bias.fp16.bin         # 已 pre-scaled，前 256 entries 除過 sqrt(16)
  meta.json             # q_rows_prescaled_by_inv_sqrt_D: true
```

---

## Fusion 3 — INT4 nibble packing（格式 fuse）

### 做法
所有 INT4 weight 以 signed 4-bit 存放，範圍 `[-7, 7]`（qmax=7，非 8，因為對稱量化）。
兩個 nibble 打包成 1 byte：
- **Low nibble**（bit[3:0]）= 第一個元素（flatten 後 index 偶數）
- **High nibble**（bit[7:4]）= 第二個元素（flatten 後 index 奇數）

Byte 中的 4-bit 值用 **two's-complement signed 4-bit** 表示：
`0x0..0x7` 代表 `0..7`，`0x8..0xF` 代表 `-8..-1`。注意我們的 clamp 上限是 7 不是 8，所以 `0x8` 實際上不會出現，但硬體 decoder 還是得正確解碼整個 4-bit signed range。

### Debug bin
同一目錄下還有 `w_int4.raw_s8.bin`，是未打包的 `int8` per-element 值（每元素 1 byte，僅為硬體驗證用，正式 RTL loader 用 packed 版本）。

---

## 沒有 Fuse 的部分（硬體 pipeline 要原生處理）

| # | 運算 | 為何不 fuse | 硬體要做什麼 |
|---|---|---|---|
| A | **PReLU `α`** | PReLU 是非線性，無法跟前後線性項合併 | 每個 block 讀一個 scalar `α_r` 與 `α_i`，對 BN+bias 輸出做 `y = x if x ≥ 0 else α·x` |
| B | **Softmax** | 非線性 | 標準 softmax unit |
| C | **Top-k + renormalize** | 動態（每個 row 的 mask 不同）、不可預計算 | softmax 後跑 k=4 tournament → scatter binary mask → 沿 last dim 求 sum → 除法 |
| D | **Complex 4-pass 加減** | dataflow 而非常數 | re·re 減 im·im、re·im 加 im·re 的累加順序 |
| E | **`apply_mask` mode 'E'** | 多項 magnitude/angle 運算 | `|mask|·|spec|·exp(j·(∠mask + ∠spec))`，並 pad DC bin = 0 |
| F | **STFT/iSTFT window** | 不是常數 fuse，是固定 ROM | 把 `misc/stft_window.fp16.bin`（Hann, length=128）存進 ROM |
| G | **Attention causal + sliding-window mask** | 隨 sequence length 動態產生（`dist > window_size=31` 則 -∞） | 硬體可在 softmax 前加 bias table，或直接把這類 position 的 score clamp 到 -INF |

---

## 匯出產物 Layout Cheat-sheet

```
export_out/
├─ manifest.json              # 全域 summary，列出所有 block / layer 的 meta
├─ layer_config.json          # dump_hw_config.py 的等效輸出（per-layer precision）
├─ blocks/
│  ├─ encoder/stage_0/
│  │   ├─ bn_scale_r.fp16.bin  bn_bias_r.fp16.bin
│  │   ├─ bn_scale_i.fp16.bin  bn_bias_i.fp16.bin
│  │   ├─ prelu_alpha_r.fp32.bin  prelu_alpha_i.fp32.bin
│  │   └─ meta.json
│  │  (encoder/stage_1..5, decoder/stage_0..4 同結構)
├─ weights/
│  ├─ encoder.model.0.0.depthwise_r/
│  │   ├─ w_int4.packed.bin     # nibble-packed INT4
│  │   ├─ w_int4.raw_s8.bin     # debug: 未打包 int8
│  │   ├─ scale.fp16.bin        # per-output-channel scale
│  │   └─ meta.json
│  ├─ decoder.model.0.0.pointwise_r/     # outlier FP8
│  │   ├─ w_fp8.e4m3.bin
│  │   ├─ bias.fp16.bin
│  │   └─ meta.json
│  └─ mattn.mattn.MultiheadAttention_re/
│      ├─ in_proj_weight/
│      │   ├─ w_int4.packed.bin
│      │   ├─ scale.fp16.bin    # 已 fuse 1/√D on Q
│      │   ├─ bias.fp16.bin     # 已 fuse 1/√D on Q
│      │   └─ meta.json
│      └─ out_proj/
│          ├─ w_int4.packed.bin  scale.fp16.bin  bias.fp16.bin  meta.json
└─ misc/
   ├─ stft_window.fp16.bin
   ├─ istft_window.fp16.bin
   └─ meta.json          # fft_len, hop, masking_mode
```

---

## Hardware pseudo-code（一個 block 完整跑完）

```
# Encoder stage c (depthwise → pointwise → BN → PReLU)
# 1. Depthwise (INT4 per-channel, bias=False by code)
dw_r_out = complex_mac_DW(x.real, x.imag, W_int4=dw_r, scale=dw_r_scale)
dw_i_out = complex_mac_DW(x.real, x.imag, W_int4=dw_i, scale=dw_i_scale)

# 2. Pointwise (INT4 per-channel, bias=True but already folded into bn_bias)
pw_real = complex_mac_PW_real(dw_r_out, dw_i_out,
                              W_int4_r=pw_r, W_int4_i=pw_i, scales=..)
pw_imag = complex_mac_PW_imag(dw_r_out, dw_i_out, ...)

# 3. Post-MAC FMA (fused BN + conv bias)
y_real = bn_scale_r * pw_real + bn_bias_r       # per-channel
y_imag = bn_scale_i * pw_imag + bn_bias_i

# 4. PReLU
out_real = PReLU(y_real, alpha_r_scalar)
out_imag = PReLU(y_imag, alpha_i_scalar)
```

對應 MHA：
```
# 1. in_proj: Q/K/V 三段 matmul，一次完成
qkv = int4_mac(x, W_int4=in_proj_weight, scale=in_proj_scale_fused)
                                          # scale 已經把 Q 部分除過 sqrt(D)
qkv += bias_fused                          # bias 的 Q 部分也已除 sqrt(D)
Q, K, V = split(qkv, 3, dim=last)
# 2. attention：Q·K^T 不再除 sqrt(D)
attn = Q @ K^T + mask                      # (sliding window mask from config)
attn = softmax(attn)
# 3. top-k + renorm（硬體 TOP_K_UNIT）
attn = topk_mask_and_renorm(attn, k=4)
# 4. attn @ V
out = attn @ V
# 5. out_proj
out = int4_mac(out, W_int4=out_proj, scale=out_proj_scale) + out_proj_bias
```
