# Mixed-Precision W4A8 QAT — 硬體交接資料

**對應軟體路徑**：
- `07_mixed_precision_qat/` — FP8 QAT baseline ckpt 產地（`epoch=134-step=130680.ckpt`）
- `08_mixed_precision_qat_from_scratch/` — W4A8 PTQ 驗證 + 量化模組
**交付對象**：硬體顧問 / Accelerator RTL team

---

## 0. TL;DR

| 硬體區塊 | 功能 | 對應軟體規格 |
|---|---|---|
| **MAC Array** | INT4 × FP8 乘法；FP16 accumulator | W4A8 + FP8 act |
| **Dequant Unit** | INT4 → FP8/FP16；3 種 granularity（per-channel / per-group along in_ch / per-head） | 對應 DW conv / PW conv / MHA |
| **WBUF / Weight SRAM** | INT4 2:1 nibble packed；scale table 同步載入；保留 6 層 FP8 outlier | decoder PW stage {0,3,4} 維持 FP8 |
| **Per-layer Precision CSR/ROM** | 每層 `(format, granularity, group_size)` 可程式化 | 全網 54 層精度配置表 |
| **Complex MAC** | 4-pass (re·re, im·im, re·im, im·re) | ComplexConv / ComplexLinear / ComplexMHA |
| **Attention Engine** | QK^T / Softmax / Score×V 走 FP16；softmax 後 top-k=4 + renormalize | top-k attention + fp16 accumulator |
| **Activation Quantizer** | 每個 layer input 做 FP8 (E4M3) cast，per-tensor dynamic | 激活全 FP8 |
| **Scale SRAM** | 隨 weight tile 同步載入；broadcast 到對應 MAC lane | weight-stationary dataflow |
| **Post-MAC per-channel FMA** | 兩組 FMA（real / imag）fuse BN + bias | NaiveComplexBatchNorm2d 不能 fold 進 conv weight |

**最新 PTQ 數字（FP8 ckpt 直 quant，無 retrain，VCTK paper-style 8-SNR avg / 750 pairs）**：

| Config | MBSTOI | SegSNR Impr (dB) | ILD (dB) | IPD (°) |
|---|---|---|---|---|
| MHA-only baseline (conv FP8, top-K=4) | 0.8397 | 2.84 | 3.25 | 15.51 |
| Full W4A8 (per-channel conv, per-head MHA, top-K=4) | 0.8325 | 2.65 | 3.28 | 15.75 |
| Full W4A8 + fp16 accumulator | 0.8326 | 2.65 | 3.28 | 15.75 |
| Full W4A8 (no top-K, per-channel) | 0.8333 | 2.67 | 3.28 | 15.75 |
| **Full W4A8 + per-group(gsz=32 along in_ch) + top-K=4 (硬體推薦)** | **0.8340** | 2.59 | 3.27 | 15.74 |
| Full W4A8 + per-group(gsz=32 along in_ch) (no top-K) | 0.8347 | 2.60 | 3.26 | 15.74 |
| Full W4A8 + per-group(gsz=16 along in_ch) + top-K=4 | 0.8335 | 2.62 | 3.27 | 15.74 |

**硬體可下決定的點**：
1. **fp16 accumulator 對品質零影響**（差異 < 0.0002 MBSTOI）→ MAC datapath 固定 FP16 accum，免費。
2. **top-K=4 attention sparsity 對品質零影響**（在所有 gsz 下都 < 0.001 MBSTOI 差異）→ 保留以省 Score×V 的 ~7.7× compute。
3. **PW conv 用 per-group along in_channels (gsz=32)** 是甜蜜點：比 per-channel 多賺 +0.0014 MBSTOI；scale 儲存從 ~0.5 KB 漲到 ~6 KB（仍 < 10 KB total）。**gsz=16 沒有更好（−0.0005 MBSTOI）**——本模型 outlier spread ~2~3x，gsz=32 已夠隔離。
4. **Decoder PW stages {0, 3, 4}（re/im 共 6 層）必須保 FP8**；其他配置都救不回這幾層的 outlier。
5. **MHA Q/K/V/O 用 per-head scale (head_dim=16)** 已驗證 W4 可行。

---

## 1. 量化精度與粒度（軟硬體共通定義）

### 1.1 全網精度配置表

模型：`BinauralAttentionDCNN`，6-stage encoder + 6-stage decoder + 1× transformer block。
Config：`embed_dim=256`, `num_heads=16`, `head_dim=16`, `kernel_num=[16,32,64,128,256,256]`, `kernel_size=5`, `attn_win=31`。

| 層類別 | 層名（模組路徑） | 權重格式 | Scale 粒度 | Group size |
|---|---|---|---|---|
| Encoder DW | `encoder.model.{0..5}.0.depthwise_{r,i}` | INT4 | per-channel | — |
| Encoder PW | `encoder.model.{0..5}.0.pointwise_{r,i}` | INT4 | per-group along **in_channels** | 32 |
| Decoder DW | `decoder.model.{0..5}.0.depthwise_{r,i}` | INT4 | per-channel | — |
| Decoder PW (normal) | `decoder.model.{1,2,5}.0.pointwise_{r,i}` | INT4 | per-group along **in_channels** | 32 |
| **Decoder PW (outlier)** | **`decoder.model.{0,3,4}.0.pointwise_{r,i}`** | **FP8 (E4M3)** | **per-tensor** | — |
| MHA `in_proj_weight` (Q/K/V fused) | `mattn.mattn.MultiheadAttention_{re,im}.in_proj_weight` | INT4 | per-head（3·H = 48 groups） | — |
| MHA `out_proj.weight` | `mattn.mattn.MultiheadAttention_{re,im}.out_proj.weight` | INT4 | per-head (16 groups) | — |
| ComplexLinear `input_proj` / `output_proj` | `mattn.input_proj`, `mattn.output_proj` (complex) | INT4 | per-head (real + imag 各 16 groups) | — |
| **All activations** | — | **FP8 (E4M3)** | per-tensor (dynamic) | — |
| **All MAC accumulators** | — | **FP16** | — | — |
| **Attention QK^T / Softmax / Score×V** | transformer block | **FP16** | — | — |

**注意**：per-group 是沿 **in_channels (Conv2d weight 的 dim=1)** 切，不是沿 kernel 的 last dim 切——對 PW conv (kW=1) 來說，沿 last dim 切等於沒效果。硬體 indexing 規則參見 §2.2。

**保留 FP8 的 6 個 layer（必須 hard-code）**：
```
decoder.model.0.0.pointwise_r / .pointwise_i
decoder.model.3.0.pointwise_r / .pointwise_i
decoder.model.4.0.pointwise_r / .pointwise_i
```

### 1.2 量化公式

**INT4 對稱量化（weight）**：
```
q_max  = 7
scale  = amax(|W_group|) / q_max          (≥ 1e-12, FP16 stored on hw)
W_q    = clamp(round(W / scale), -7, 7)
W_hat  = W_q * scale                       (dequant)
```

**FP8 E4M3（activation 與 outlier weight）**：
- `torch.float8_e4m3fn`（±448 range；1-bit sign / 4-bit exp / 3-bit mantissa）
- 標準 E4M3 converter 即可

### 1.3 Attention datapath（含 fp16 accumulator 與 top-k）

```
# Linear 進 MHA：W = INT4, act = FP8, MAC accum FP16, output FP8
q = Wq(x_fp8); k = Wk(x_fp8); v = Wv(x_fp8)

# Cast 到 FP16 進入 attention 內部 datapath
q, k, v = to_fp16(q, k, v)
attn   = (q @ k.T) / sqrt(d)               # QK^T 在 FP16
attn  += causal_sliding_mask_fp16          # mask（streaming 下天然由 KV cache 長度限制）
attn   = softmax(attn, dim=-1)             # FP16 softmax

# Top-K = 4 sparsification + renormalize
idx    = topk(attn, k=4, dim=-1).indices
mask   = zeros_like(attn); mask.scatter_(-1, idx, 1)
attn   = attn * mask
attn   = attn / (attn.sum(-1, keepdim=True) + 1e-8)

out    = attn @ v                          # Score × V 在 FP16
out    = to_fp8(out)                       # cast 回 FP8 再進 out_proj
out    = Wo(out_fp8)                       # Linear: INT4 × FP8 → FP16 accum → FP8
```

---

## 2. 硬體區塊規格

### 2.1 MAC Datapath

**需要功能**：
1. **INT4 weight lane**：每個 weight word 打包 2 個 INT4（8-bit byte = 兩個 nibble）。
2. **Dequant 前置級**：進 MAC 前做 `W_dequant = W_int4_signed × scale`。三種設計：

   | 方案 | Dequant | MAC 乘法 | 面積 | 延遲 | 備註 |
   |---|---|---|---|---|---|
   | A. 展開成 FP8 | `W_fp8 = round_to_fp8(INT4 × scale_fp16)` | FP8 × FP8 → FP16 | 小 | +1 stage | 最接近既有 FP8 MAC |
   | B. 展開成 FP16 | `W_fp16 = INT4 × scale_fp16` | FP16 × FP8 → FP16 | 中 | +1 stage | MAC 乘法器寬度升級 |
   | **C. Microscaling（推薦）** | 不 dequant，INT4 × FP8 直接乘 | INT4 × FP8 → INT accum，group 邊界乘 `scale_fp16` 後進 FP16 global accum | **最小** | 0 | per-group / per-head 都能對齊 MAC tile |

   **建議方案 C**：per-group(32) 與 per-head(16) 都能在 MAC tile 邊界觸發 scale 乘法，不必每個 lane 都掛 dequant unit。

3. **Accumulator 固定 FP16**（已驗證品質不掉）。
4. **Complex MAC**：4-pass 時，re·re 與 −im·im 累加進輸出 real，re·im 與 im·re 累加進輸出 imag。若要省延遲，可用 Gauss 3-multiply trick（多 2 個 adder + 1 buffer）。

**Scale dtype**：
- 軟體端 forward 時動態計算 (`amax(|W|)/7`)，PyTorch 預設 FP32，無梯度。
- 硬體端建議存 **FP16**（~10⁻⁵~10⁵ 範圍 + 11-bit 精度，足以覆蓋所有 per-group scale）。FP8 mantissa 不夠精度，不可接受。
- `ckpt2hwdat.py` 在 export 時將 FP32 scale cast FP16 寫入 bin；此 cast 誤差 << INT4 量化誤差，可忽略。

### 2.2 Scale Broadcast Unit

三種 granularity 的 scale 取用 pattern 與 index 計算：

| 粒度 | Scale shape | Addressing 規則 | 建議實作 |
|---|---|---|---|
| **per-channel**（DW conv） | `[C_out]` | `scale[co]` 沿 spatial 廣播 | 小型 scale ROM，每 row 一個 scale |
| **per-group(32) along in_ch**（PW conv, `[C_out, C_in, 1, 1]`） | `[C_out, C_in/32]` | `addr = co × num_groups + (ci >> 5)`（32 = 2^5） | scale 隨 weight tile 一起搬；WBUF 相鄰 side-SRAM |
| **per-head**（MHA `[r·E, E]` 或 `[E, E]`，head_dim=16） | `[r·H]`（in_proj: r=3, H=16 → 48）或 `[H]`（out_proj: 16） | row 軸每 16 行共用一個 scale，`addr = row // 16` | 小型 scale register file |

**統一 indexing 公式**：硬體只要能表達 `scale_addr = co × ceil(C_in / group_size) + (ci // group_size)`（group_size 從 per-layer CSR/ROM 讀），就能通吃三種粒度——per-head = group_size=head_dim=16；per-channel = group_size = full row size。

**全網 scale 總量（dim=256, H=16, gsz=32, FP16）**：

| 區塊 | Scale 數 | 位元組 |
|---|---|---|
| MHA in/out_proj (re+im, per-head) | ~160 | ~320 B |
| MHA in_proj_weight (Q/K/V fused, per-head ×2 for re/im) | ~96 | ~192 B |
| Encoder/Decoder DW (per-channel, 12 stage × C_out × re+im) | ~1500 | ~3 KB |
| Encoder/Decoder PW (per-group(32) along in_ch, deep stages dominate) | ~3000 | ~6 KB |
| **Total** | — | **< 10 KB** |

相對於 WBUF 64 KB 完全是小角色。

**排版建議**：WBUF 內或旁邊開一塊 scale side-SRAM；DMA descriptor 帶 `(weight_base, scale_base)` 兩個 pointer，FSM 同步推進。weight tile 載入時 scale 只需讀一次（per-group / per-head broadcast 到對應 lane）。

### 2.3 WBUF / Weight SRAM Layout

原 WBUF：16 Banks × 64-bit × 256 Depth = 64 KB（無法一次容納整個模型，需分 tile 搬運）。

**全模型 weight byte（dim=256 / H=16）**：

| 模組 | tensor | FP8 bytes | W4A8 bytes |
|---|---|---|---|
| Encoder 全部 | real conv | 59 KB | **30 KB** |
| Decoder 全部（含 6 層 FP8 outlier） | real conv | 113 KB | **62 KB** |
| `mattn.input_proj` (complex) | 65.8 K complex elem | 132 KB | **66 KB** |
| MHA `in_proj_weight` + `out_proj` (re+im) | real | 526 KB | **263 KB** |
| `mattn.output_proj` (complex) | 65.8 K complex elem | 132 KB | **66 KB** |
| **Total** | — | **~962 KB** | **~487 KB** |

**Layout 建議**：
- **Weight 區（INT4）**：2-nibble per byte 打包；一次載入一個 tile（例：256×256 INT4 = 32 KB 剛好半個 WBUF）。
- **Weight 區（FP8 outlier）**：decoder stage {0,3,4} 的 `pointwise_{r,i}`（6 個 tensor）固定分配小區塊，或與 INT4 區共用但以 layer id 決定讀取寬度。
- **Scale side-SRAM**：獨立 < 10 KB，與 weight tile 同步載入（weight-stationary pipeline）。

**DMA 縮減**：
- dim=512 FP8 baseline → dim=256 W4A8：**2.80 MB → 487 KB（~5.75×）**
- dim=256 FP8 → dim=256 W4A8：**962 KB → 487 KB（~2×）**

embed_dim 由 512 → 256 已貢獻 MHA 4× 縮減（MHA 矩陣是 E²），再搭 W4A8 再砍半。

### 2.4 Per-layer Precision Config

每層 control unit 要知道精度 + 粒度，sequencer 進入該層時讀出此配置以設定 dequant unit、WBUF 讀寬、MAC 切換。

**Config struct（每層 ≈ 4 bytes）**：
```c
struct LayerCfg {
    uint2  weight_format;      // 00=INT4, 01=FP8_E4M3, 10=FP8_E5M2
    uint2  scale_granularity;  // 00=per-tensor, 01=per-channel, 10=per-group, 11=per-head
    uint16 group_size;         // per-group → 32；per-head → head_dim=16
    uint8  num_heads;          // per-head → 16
    uint2  act_format;         // 01=FP8_E4M3 (固定)
    bool   is_complex;         // 複數層 → 4-pass
    bool   is_transposed;      // decoder ConvTranspose2d
};
// 全網 54 層 × ~4 bytes ≈ 216 bytes
```

**實作選項**：

| 情境 | 建議 |
|---|---|
| 只跑這顆 ckpt，配置不變 | **小 ROM + state machine**（~216 B），面積近乎零 |
| 想支援多顆 ckpt / fallback 試驗 | **CSR register file** + host AXI-lite 寫入 |
| 折中 | ROM 主路徑，少數 override switch（format bypass、top-k 開關）留 CSR |

軟體交付：`dump_hw_config.py` 從 parametrization 狀態讀出所有層的 `(format, granularity, group_size)` 吐 JSON/CSV 給 RTL 燒 ROM 或寫入 CSR。

### 2.5 Attention Engine

**精度流程**（已驗證）：

```
輸入:        x_fp8
  ↓  Wq/Wk/Wv: INT4 weight × FP8 act → FP16 accum → FP8
Q_fp8, K_fp8, V_fp8
  ↓  cast FP16 (FP8→FP16 converter)
Q_fp16, K_fp16, V_fp16
  ↓
score = Q_fp16 @ K_fp16.T / sqrt(d_fp16)   (FP16 GEMM)
score += causal_sliding_mask_fp16
  ↓  FP16 Softmax (max-subtract + exp LUT + reciprocal sum)
attn_fp16 (B, H, T, T)
  ↓  TOP_K_UNIT (k=4, T_eff=31 in streaming)
  ├─ tournament 31 個 score 找最大 4 個
  ├─ 建立 binary mask
  ├─ element-wise mask
  └─ 沿 T 軸求和 → 每 row 除 (+1e-8 stability)
attn_fp16 (稀疏 4-of-31)
  ↓
out_fp16 = attn_fp16 @ V_fp16   (FP16 GEMM)
  ↓  cast FP8
out_fp8
  ↓  out_proj: INT4 × FP8 → FP16 accum → FP8
output_fp8
```

**硬體細節**：
- 兩次 matmul（QK^T 與 Score×V）固定 FP16 輸入輸出；可與 conv MAC array 共用或獨立 FP16 GEMM 單元。
- Softmax 在 FP16 下足夠（LUT 寬度、exp 輸出寬度、reciprocal accuracy 均覆蓋）。
- **Sequence length（streaming 視角）**：軟體驗證時 T ≈ 157~160 是 dataset **2 秒剪輯** artifact，**不是硬體常數**。Streaming 硬體每 frame 推一次，**只看 `attn_win=31` 與 `top_k=4`**；KV cache 31 slot 的 ring buffer 即可。TOP_K_UNIT 的 tournament 只對 31 個 score 找最大 4 個。
- Mask window：causal sliding window=31；streaming 下 KV cache 長度 ≤ 31 天然等效 mask，硬體不用建 T×T mask tensor。
- **Top-K=4 省下什麼**：
  - QK^T 不省（要先算完才能排 top-K）
  - Softmax 不省（仍 31 個 exp）
  - **Score × V 省 ~7.7×**：每 head 每 frame 由 31×16 MAC 降到 4×16 MAC
  - Score 暫存：31 → 4（+ 4 個 index）

### 2.6 Activation FP8 Quantizer

每個 Conv / Linear / MHA 的輸入先 cast 成 FP8 (E4M3)：
- 硬體已有 FP8 converter 可直接複用。
- 激活採 **dynamic per-tensor** scale。
- 軟體端透過 `forward_pre_hook` 對每個 Conv / Linear / MHA 輸入做 quantize；硬體端對應「input fetch 後立即 FP8 cast」。

### 2.7 Post-MAC per-channel FMA（fuse BN + bias）

#### 為什麼不能把 BN 折進 complex conv weight

`ComplexConv2d` 是兩組實 conv `fc_r`、`fc_i` 組合：
```
y.real = fc_r(x.real) − fc_i(x.imag)
y.imag = fc_r(x.imag) + fc_i(x.real)     # ← fc_r 同時出現在 real 與 imag 兩條路
```
`NaiveComplexBatchNorm2d` 對 real / imag 各有獨立 BN（`s_r = γ_r/√(σ_r²+ε)`, `s_i = γ_i/√(σ_i²+ε)`）。把 `s_r` 折進 `fc_r.weight`，`y.imag` 那條路卻要用 `s_i`——一組 weight 沒辦法同時套兩個不同 scale。BN **不能 fuse 進 conv weight**。

#### 解法：fuse 到 MAC 輸出後的兩組 per-channel FMA

```
s_r[c] = γ_r[c] / √(σ_r²[c] + ε)
s_i[c] = γ_i[c] / √(σ_i²[c] + ε)

bn_scale_r[c] = s_r[c]
bn_bias_r [c] = s_r[c] · (b_r[c] − b_i[c] − μ_r[c]) + β_r[c]
bn_scale_i[c] = s_i[c]
bn_bias_i [c] = s_i[c] · (b_r[c] + b_i[c] − μ_i[c]) + β_i[c]
```

硬體在每個複數 MAC 後擺兩組 per-channel FMA：
```
out_real[c] = bn_scale_r[c] · mac_real[c] + bn_bias_r[c]
out_imag[c] = bn_scale_i[c] · mac_imag[c] + bn_bias_i[c]
```

#### 實作細節
- 兩組 FMA 各需一塊 per-channel SRAM 存 `{bn_scale, bn_bias}`，寬度 **FP16**。
- 放在 MAC accumulator FP16 輸出之後、PReLU 之前。
- **沒有 BN 的 stage**（decoder 最後一 stage）：軟體 export 時填 `bn_scale_{r,i} = 1`，`bn_bias_r[c] = b_r[c] − b_i[c]`，`bn_bias_i[c] = b_r[c] + b_i[c]`，硬體 pipeline **永遠走同一條 post-MAC FMA path**，不需特例 bypass。
- Storage：每 stage `4 × C_out × FP16`，12 stages 合計幾 KB 量級。
- 詳見 `08_software_to_hardware/FUSIONS.md` §Fusion 1，bin 路徑 `blocks/{encoder|decoder}/stage_{idx}/bn_{scale,bias}_{r,i}.fp16.bin`。

---

## 3. 對應軟體模組

### 3.1 量化核心
| 檔案 | 角色 |
|---|---|
| `08_mixed_precision_qat_from_scratch/DCNN/utils/mixed_precision_qat.py` | INT4 STE + 三種 granularity parametrization + FP8 outlier 保留 + top-k 掛載 + fp16 accumulator propagation；提供 `enable_mixed_precision_qat(model, config, top_k, fp16_accumulator)` |
| `08_mixed_precision_qat_from_scratch/DCNN/utils/complexPyTorch/complexLayers.py` | `_mha_attention_forward` 帶 `fp16_accumulator` 與 `top_k` 參數的自訂 attention 路徑；`ComplexMultiheadAttention` 透過 `use_custom_forward` flag 切換 |
| `08_mixed_precision_qat_from_scratch/DCNN/trainer.py` | 從 config 讀 `qat_scope` (`mha_only`/`full`)、`top_k_attention`、`int4_group_size`、`fp16_accumulator` 並傳進 enable function |
| `08_mixed_precision_qat_from_scratch/eval.py` | `--hydra_overrides` CLI，允許組合測試（例：`training.qat_scope=full training.top_k_attention=4 training.int4_group_size=32 +training.fp16_accumulator=true`）|

### 3.2 Config 對應

```yaml
training:
  mixed_precision_qat: true
  qat_scope: full              # mha_only | full
  top_k_attention: 4           # null = dense；4 = 硬體 default
  int4_group_size: 32          # PW conv 沿 in_ch 切的 group size
  fp16_accumulator: true       # 對應硬體 FP16 accum + FP16 attention datapath
  fp8_dtype: e4m3              # 激活 + outlier weight 格式
```

### 3.3 Verification scripts

| 檔案 | 驗證項目 |
|---|---|
| `verify_custom_attention.py` | 自訂 attention forward vs `nn.MultiheadAttention` FP32 下 bit-exact |
| `verify_mixed_precision_enable.py` | 整個模型接上 mixed-precision 後 forward + backward 不破 |
| `verify_w4a8.py` | W4A8 真的有套用：parametrized weight 數量 + 與 FP32 forward output diff |

---

## 4. Dataflow / Pipeline

### 4.1 Weight-stationary（推薦）

Weight tile 一次載入 WBUF，scale tile 同步載入 scale SRAM，activation 串流進來。每 tile 內只做一次 dequant-scale 廣播。

### 4.2 Complex 4-pass

| Pass | 運算 | 寫入 |
|---|---|---|
| 1 | `W_re · X_re` → acc_re | acc_re += |
| 2 | `W_im · X_im` → acc_re (減) | acc_re −= |
| 3 | `W_re · X_im` → acc_im | acc_im += |
| 4 | `W_im · X_re` → acc_im (加) | acc_im += |

對 per-group/per-head scale：pass 1 與 pass 2 的 scale **不同**（`W_re` 與 `W_im` 各有自己的 scale table），control unit **不能重用** `W_re` 的 scale 到 `W_im`。

### 4.3 Decoder outlier 的 pipeline 切換

Decoder stage {0, 3, 4} 的 pointwise 是 FP8 不是 INT4。Control unit 進入這三層時要：
1. 切換 dequant pipeline：bypass INT4 dequant，直接 FP8 × FP8。
2. WBUF 讀寬度切換（FP8 = 8-bit/word，INT4 = 4-bit/word = 2:1 packed）。

### 4.4 Attention FP16 pipeline

MHA 內部兩次 matmul（QK^T、Score×V）是 FP16 × FP16 → FP16，與 conv/linear 的 INT4 × FP8 → FP16 → FP8 不同：
- 可獨立配置一個 FP16 GEMM 單元（attention 專用），或在 MAC array 動態切換 multiplicand 格式。
- Softmax / top-k / renormalize 這些 scalar pipeline stage 都在 FP16。

---

## 5. 估算效益

**全模型 DMA byte 對照**：

| 組合 | Encoder | Decoder | input_proj | MHA | output_proj | **Total** |
|---|---|---|---|---|---|---|
| dim=512 FP8 | 59 KB | 113 KB | 263 KB | 2,101 KB | 263 KB | **~2,800 KB** |
| dim=256 FP8 | 59 KB | 113 KB | 132 KB | 526 KB | 132 KB | **~962 KB** |
| **dim=256 W4A8（目標）** | **30 KB** | **62 KB** | **66 KB** | **263 KB** | **66 KB** | **~487 KB** |

**其他指標**：

| 指標 | FP8 baseline | W4A8 mixed |
|---|---|---|
| Encoder/Decoder PW weight | 全 FP8 | ~90% INT4 + ~10% FP8（6 層 outlier） |
| Encoder/Decoder DW weight | 全 FP8 | 全 INT4 per-channel（−50%） |
| Attention 計算量 (per frame) | 31×16 Score×V MAC | **4×16**（−7.7×） |
| Attention KV cache | 31 frame | 31 frame |
| DMA bandwidth（per inference） | baseline | **−82%** vs dim=512 FP8；**−49%** vs dim=256 FP8 |
| **MBSTOI (PTQ 推薦 config: gsz=32 + top-K=4)** | — | **0.8340**（vs MHA-only 0.8397，Δ −0.0057）|
| **MBSTOI (PTQ best: gsz=32, no top-K)** | — | **0.8347**（vs MHA-only 0.8397，Δ −0.0050）|
| **SegSNR Impr (PTQ 推薦 config, dB)** | — | **2.59**（vs MHA-only 2.84，Δ −0.25）|
| Model accuracy after QAT retrain | — | 預期 ≥ PTQ 數字，可接近或超過 0.8397 |

---

## 6. 給硬體顧問的可提問清單

1. 現有 MAC 是否已支援 **INT4 input + FP16 accumulator**？若否，新增路徑的面積/功耗成本？
2. WBUF 的 bank/depth 是否能切出獨立 scale SRAM？或要共用同一 memory（帶 tag）？
3. TOP_K_UNIT 的面積估算：k=4 tournament vs heap，對 T=31 的 latency？
4. 複數 4-pass pipeline 是否已存在？若無，要做 2× MAC（真假分開）還是維持 4-pass？
5. Per-layer precision 切換的延遲（避免層間 bubble）？
6. FP8 E4M3 converter 是否為現成 IP？FP8 ↔ FP16 converter？
7. Scale 精度 FP16 足夠，還是要 BF16？
8. Attention 的 FP16 GEMM 是獨立單元還是與 conv MAC array 共用？

---

## 附錄：常見疑問

**Q1. INT4 × scale 要變 FP8 再乘，scale 是 FP8 嗎？**
不是。軟體端 scale 是 FP32（動態 `amax/7`），硬體端建議存 FP16。FP8 mantissa 不夠精度。詳見 §2.1。

**Q2. 三種 granularity 在硬體怎麼統一？**
統一成 `scale_addr = co × ceil(C_in / group_size) + (ci // group_size)`，group_size 從 per-layer config 讀（per-head=16，per-group=32，per-channel=full row）。詳見 §2.2。

**Q3. top-K=4 少算什麼？**
只省 Score × V（~7.7×），QK^T 與 softmax 不省。對品質零影響，可放心保留。詳見 §2.5。

**Q4. dim=256 W4A8 vs dim=512 FP8 的 DMA 縮減？**
2.80 MB → 487 KB，約 **5.75×**。其中 4× 來自 embed_dim 減半（MHA 是 E²），再 ~2× 來自 W4A8。詳見 §2.3、§5。

**Q5. Scale 要存多少？怎麼排？**
全網 < 10 KB（MHA ~500 B、Conv PW ~6 KB、DW ~3 KB），獨立 side-SRAM。詳見 §2.2。

**Q6. Per-layer precision CSR 一定要做嗎？**
不一定。只跑這顆 ckpt 的話，54 層配置燒成 ~216 B ROM 即可。多 ckpt / fallback 才需 CSR register file。詳見 §2.4。

**Q7. T ≈ 160 在 streaming 硬體不用管？**
對。T 是 dataset 2 秒剪輯 artifact；硬體只看 `attn_win=31`（KV cache 31 slot）與 `top_k=4`（tournament 4 entry）。詳見 §2.5。

**Q8. Per-group 為什麼沿 in_channels 切？**
PW conv 的 weight shape 是 `(out, in, 1, 1)`，沿 last dim (kW=1) 切 group 等於沒效果。沿 in_channels (dim=1) 切，每個 group 涵蓋 32 個 in_channel × 1 × 1 = 32 個 weight，能有效隔離 outlier。詳見 §2.2 與 §1.1。
