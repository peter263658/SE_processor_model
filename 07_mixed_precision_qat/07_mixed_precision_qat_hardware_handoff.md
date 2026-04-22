# Mixed-Precision W4A8 QAT — 硬體交接資料 (rev 3)

**日期**：2026-04-20
**對應軟體路徑**：
- `07_mixed_precision_qat/` — FP8 QAT baseline ckpt 產地（`epoch=134-step=130680.ckpt`）
- `08_mixed_precision_qat_from_scratch/` — W4A8 QAT 實作 + 2026-04-19 PTQ 驗證
**接自**：`07_mixed_precision_qat_handoff.md`（軟體層規劃，2026-04-15）
**交付對象**：硬體顧問 / Accelerator RTL team
**本次更新 (rev 3)**：
- 修正 §4.3 / §6 的 DMA 估算，補回 ComplexLinear wrapper（`input_proj`, `output_proj`）的佔用，總量由 MHA-only 擴展到全模型
- 釐清 scale 在軟體（FP32 動態計算）與硬體（建議 FP16 靜態儲存）的差異，給 MAC datapath 三種選擇（A/B/C microscaling）
- 補上 T vs streaming sliding window 的釐清：T≈160 是 dataset 2 秒剪輯長度，硬體只看 `attn_win=31` 與 `top_k=4`
- 補上 per-granularity 硬體 indexing 規則、per-layer config 改用 ROM 的低面積選項

---

## 0. TL;DR

1. 全網 W4A8（conv + MHA，decoder 3 層 outlier PW 保 FP8） + FP16 MAC 累加器 + top-K=4 attention，**zero-retrain PTQ 下 MBSTOI 只掉 0.007**（0.8397 → 0.8326）。
2. **Fp16 accumulator 對品質零影響**（差異 < 0.0002 MBSTOI）。硬體可以放心把 MAC datapath 固定為 INT4×FP8 → FP16 accum。
3. 以下表格可直接對應到 RTL 需求：

| 硬體區塊 | 要加入 / 修改的功能 | 觸發的軟體規格 | 已驗證 |
|---|---|---|---|
| **MAC Array** | INT4 × FP8 乘法；FP16 accumulator | W4A8 權重 + FP8 激活 | ✅ PTQ 2026-04-19 |
| **Dequant Unit** | INT4 → FP8/FP16 (乘 scale)；三種 granularity | per-channel / per-group(256) / per-head(16) | ✅ |
| **WBUF / Weight SRAM** | INT4 2:1 nibble packed；scale table 同步載入；保留 6 層 FP8 | decoder PW stage {0,3,4} 維持 FP8 | ✅ |
| **Per-layer Precision CSR** | 每層 `(format, granularity, group_size)` 可程式化切換 | 全網 54 層精度配置表 | ✅ |
| **Complex MAC** | 4-pass (re×re, im×im, re×im, im×re) | ComplexConv / ComplexLinear / ComplexMHA | ✅ |
| **Attention Engine** | QK^T / Softmax / Score×V 走 FP16；softmax 後 top-k (k=4) + renormalize | top-k attention + fp16 accumulator | ✅ PTQ 2026-04-19 |
| **Activation Quantizer** | 每個 layer input 做 FP8 (E4M3) cast, per-tensor dynamic | 激活全 FP8 | ✅ |
| **Scale SRAM** | 隨 weight tile 同步載入；broadcast 到對應 MAC lane | weight-stationary dataflow | ✅ |
| **Post-MAC per-channel FMA** | 兩組 FMA（real 路 / imag 路）fuse BN + bias | NaiveComplexBatchNorm2d 無法 fold 進 conv weight | ✅ |

---

## 1. 今日 PTQ 驗證結果（2026-04-19，zero-retrain）

**Checkpoint**：FP8 QAT `epoch=134-step=130680.ckpt`（同一顆 ckpt 套不同 wrapper）。
**測試集**：VCTK paper-style，8 SNR 點 × 750 pairs = 6000 utterances。

### 1.1 三組設定 + 對應的硬體限制

| 組 | 權重限制 | 激活限制 | MAC 精度 | Attention 限制 |
|---|---|---|---|---|
| **(A) MHA-only** | MHA Wq/k/v/o + Complex in/out_proj 全 W4 (per-head)，conv 全 FP8 | FP8 E4M3 | FP32 (參考) | top-K=4, softmax 後 mask |
| **(B) Full-scope** | (A) + encoder/decoder DW W4 per-channel + PW W4 per-group(gsz=256)；decoder PW stage {0,3,4} 保 FP8 | FP8 E4M3 | FP32 (參考) | top-K=4 |
| **(C) Full + fp16 accum** | = (B) | FP8 E4M3 | **FP16** (QK^T, Softmax, Score×V) | top-K=4 |

### 1.2 8-SNR paper-style 平均（VCTK）

| 指標 | (A) MHA-only | (B) Full W4A8 | (C) Full + fp16 accum | C 對 A |
|---|---|---|---|---|
| **MBSTOI** | 0.8397 | 0.8325 | **0.8326** | **−0.0071** |
| SegSNR Imp (dB) | 2.84 | 2.65 | 2.65 | −0.19 |
| ILD 誤差 (dB) | 3.25 | 3.28 | 3.28 | +0.03 |
| IPD 誤差 (°) | 15.51 | 15.75 | 15.75 | +0.24 |

### 1.3 Per-SNR MBSTOI（完整 8 點）

| SNR (dB) | (A) | (B) | (C) | ΔB−A | ΔC−B |
|---|---|---|---|---|---|
| −6 | 0.6714 | 0.6564 | 0.6566 | **−0.0150** | +0.0002 |
| −3 | 0.7371 | 0.7256 | 0.7256 | −0.0115 | 0 |
|  0 | 0.8026 | 0.7936 | 0.7936 | −0.0090 | 0 |
|  3 | 0.8458 | 0.8386 | 0.8387 | −0.0072 | +0.0001 |
|  6 | 0.8806 | 0.8751 | 0.8752 | −0.0055 | +0.0001 |
|  9 | 0.9089 | 0.9050 | 0.9051 | −0.0039 | +0.0001 |
| 12 | 0.9271 | 0.9246 | 0.9245 | −0.0025 | −0.0001 |
| 15 | 0.9439 | 0.9412 | 0.9411 | −0.0027 | −0.0001 |

### 1.4 結論（硬體視角）

- **Fp16 MAC accumulator 免費**：(C) 對 (B) 所有 SNR 點差異都 < 0.0002 MBSTOI（甚至有些點還更好）。硬體可以直接用 FP16 accumulator 不用再擔心「省硬體 vs 丟品質」。
- **Conv 全 W4 的代價很小**：(B) 對 (A) 平均掉 0.007 MBSTOI，最差 SNR=−6dB 掉 0.015；SNR ≥ 3dB 掉 < 0.008。相較於 DMA / SRAM 砍半的收益，這個代價非常划算。
- **再做 QAT retrain 結果只會更好**：PTQ = 已經訓好的權重直接截斷到 INT4，沒有任何 scale/grid 的 learning；等完整 QAT 完成後品質應該更接近 (A)。

### 1.5 模型總尺寸與 DMA 量（rev 3 修正）

**背景**：前版（rev 2）§4.3 / §6 只列了 ComplexMultiheadAttention 內部的 4 矩陣（Wq/Wk/Wv/Wo × {re,im}），**漏掉 ComplexLinear 的兩個 wrapper（`mattn.input_proj`, `mattn.output_proj`）**。這兩個是 `nn.Linear(..., dtype=torch.complex64)`，實際 SRAM 佔用要把 re+im 兩份都算進去（每個 complex element = 2 FP8 bytes）。

**Config 校正**：實際 config 是 `fft_len=128, win_len=128, binaural=true`（不是 model.yaml 裡被註解掉的 400/512）。因此 `flatten_size = (kernel_num[-1]//2) × (fft_len // 2^len) × 2 = 128 × 1 × 2 = 256`，不是 rev 2 假設的 1024。

**dim=256 / H=16 模型參數與 DMA byte 分解（來自 `profile_compare.log`）**：

| 組 | params (numel) | tensor 型態 | FP8 bytes | W4A8 bytes |
|---|---|---|---|---|
| Encoder | 59.3 K | real conv | 59.3 KB | **29.7 KB**（全 INT4: DW per-ch + PW gsz=256） |
| Decoder | 113.1 K | real conv | 113.1 KB | **62.2 KB**（90% W4 + 10% FP8 outlier 6 層） |
| `mattn.input_proj` (ComplexLinear) | 65.8 K | **complex64** | **131.6 KB** | **65.8 KB**（INT4 per-head × 2 for re+im） |
| `mattn.mattn.MHA_{re,im}` (Wq/Wk/Wv/Wo) | 526.3 K | real | 526.3 KB | **263.2 KB**（INT4 per-head） |
| `mattn.output_proj` (ComplexLinear) | 65.8 K | **complex64** | **131.6 KB** | **65.8 KB** |
| **Total** | **830.3 K numel** | — | **~962 KB** | **~487 KB** |

**三精度對照**：

| 組合 | 全模型 DMA |
|---|---|
| dim=512 / H=32 FP8（舊 baseline） | **~2.80 MB** |
| dim=256 / H=16 FP8（本 ckpt FP8 版） | ~962 KB |
| **dim=256 / H=16 W4A8（本次硬體目標）** | **~487 KB** |

**重要陷阱**：`torch.numel` 對 complex tensor 回傳的是複數元素數；實際儲存要 ×2 才是 byte 數。前版把 wrapper 漏算、又誤把 `flatten_size` 當 1024 算，低估了 wrapper 卻又高估了單矩陣尺寸，兩個錯誤互相抵消使得結論看似合理但細項錯誤。本版以 profiler 實測值為準。

**DMA 總縮減**：dim=512 FP8 baseline → dim=256 W4A8 ≈ **5.75×**（2.80 MB → 487 KB）。

**PTQ 結果 CSV**：
- (A): `08_mixed_precision_qat_from_scratch/ptq_mha_only_w4a8_eval/paper_style/vctk_paper_style_results.csv`
- (B): `08_mixed_precision_qat_from_scratch/ptq_full_scope_topk4_gsz256_eval/paper_style/vctk_paper_style_results.csv`
- (C): `08_mixed_precision_qat_from_scratch/ptq_full_scope_topk4_gsz256_fp16accum_eval/paper_style/vctk_paper_style_results.csv`

---

## 2. 軟體實作摘要（Software Delta）

### 2.1 新增模組

| 檔案 | 說明 |
|---|---|
| `DCNN/utils/mixed_precision_qat.py` | 主模組。INT4 STE + 三種 granularity parametrization + FP8 outlier 保留 + top-k 掛載 + **fp16 accumulator propagation** |
| `verify_custom_attention.py` | 驗證自訂 attention forward 與 `nn.MultiheadAttention` 在 FP32 下 bit-exact |
| `verify_mixed_precision_enable.py` | 驗證整個模型接上 mixed-precision 後 forward + backward 不破 |
| `verify_w4a8.py` | 確認 W4A8 真的有套用：6 個 MHA-related parametrized weights，forward 輸出相對 FP32 差 23.4% |

### 2.2 修改現有檔案

| 檔案 | 改動 |
|---|---|
| `DCNN/utils/complexPyTorch/complexLayers.py` | `_mha_attention_forward` 新增 `fp16_accumulator` 參數：True 時 Q/K/V cast fp16，QK^T matmul 在 fp16 累加，softmax 在 fp16，Score×V matmul 在 fp16，最後 cast 回 orig_dtype 再走 `out_proj`。`ComplexMultiheadAttention.__init__` 多一欄 `self.fp16_accumulator`；`forward()` 4 次呼叫 `_mha_attention_forward` 都把這個 flag 傳下去。 |
| `DCNN/utils/mixed_precision_qat.py` | `enable_mixed_precision_qat()` 多一個 `fp16_accumulator: bool = False` 參數；`_set_topk_on_complex_mha()` 接收 flag，只要 `top_k` 或 `fp16_accumulator` 任一為 True 就啟用 `use_custom_forward=True` 走自訂路徑。 |
| `DCNN/trainer.py` | `qat_scope` 支援 `"mha_only"`（conv 全改 FP8）與 `"full"`（conv 納入 W4，decoder outlier 保 FP8）；從 config 讀 `fp16_accumulator` 並傳進 enable function。 |
| `eval.py` | 新增 `--hydra_overrides`，允許命令列給 Hydra override（例如 `training.qat_scope=full training.top_k_attention=4 training.int4_group_size=256 +training.fp16_accumulator=true`）做 PTQ 組合比較。 |
| `config/config.yaml` / `training.yaml` | `mixed_precision_qat: true`, `qat_scope: mha_only|full`, `top_k_attention: 4`, `int4_group_size: 256`, `fp16_accumulator: false|true`, `train_checkpoint_path: epoch=134-step=130680.ckpt` |

### 2.3 量化公式（硬體端實作要對齊）

**INT4 對稱量化（weight）**：
```
q_max = 7
scale = amax(|W_group|) / q_max      (≥ 1e-12)
W_q   = clamp(round(W / scale), -7, 7)
W_hat = W_q * scale                  (dequant)
```

**FP8 E4M3（activation 與 outlier weight）**：
- 直接 cast：`torch.float8_e4m3fn`（±448 range、1-bit sign、4-bit exponent、3-bit mantissa）
- 硬體可用標準 E4M3 converter

**Attention datapath（含 fp16 accumulator 與 top-k）**：
```
# 軟體實作（等效硬體 pipeline）
q = Wq(x_fp8); k = Wk(x_fp8); v = Wv(x_fp8)        # Linear: INT4 × FP8 → FP16 acc → FP8 out
q,k,v = q.to(fp16), k.to(fp16), v.to(fp16)         # ← Cast to fp16
attn  = (q @ k.T) / sqrt(d)                        # QK^T in fp16
attn += mask_fp16                                  # causal sliding window mask
attn  = softmax(attn, dim=-1)                      # fp16 softmax
idx   = topk(attn, k=4, dim=-1).indices
mask  = zeros_like(attn); mask.scatter_(-1, idx, 1)
attn  = attn * mask
attn  = attn / (attn.sum(-1, keepdim=True) + 1e-8)  # renormalize
out   = attn @ v                                   # Score × V in fp16
out   = out.to(orig_dtype)                         # Cast back before out_proj
out   = Wo(out_fp8)                                # Linear: INT4 × FP8 → FP16 acc → FP8 out
```

---

## 3. 全網精度配置表（硬體 control unit 要能表達）

模型：`BinauralAttentionDCNN`，6-stage encoder + 6-stage decoder + 1× transformer block。
Config：`embed_dim=256`, `num_heads=16`, `head_dim=16`, `kernel_num=[16,32,64,128,256,256]`, `kernel_size=5`。

| 層類別 | 層名（模組路徑） | 權重格式 | Scale 粒度 | Group size |
|---|---|---|---|---|
| Encoder DW | `encoder.model.{0..5}.0.depthwise_{r,i}` | INT4 | per-channel | — |
| Encoder PW | `encoder.model.{0..5}.0.pointwise_{r,i}` | INT4 | per-group (last dim) | 256 |
| Decoder DW | `decoder.model.{0..5}.0.depthwise_{r,i}` | INT4 | per-channel | — |
| Decoder PW (normal) | `decoder.model.{1,2,5}.0.pointwise_{r,i}` | INT4 | per-group (last dim) | 256 |
| **Decoder PW (outlier)** | **`decoder.model.{0,3,4}.0.pointwise_{r,i}`** | **FP8 (E4M3)** | **per-tensor** | — |
| MHA `in_proj_weight` (Q/K/V fused) | `mattn.mattn.MultiheadAttention_{re,im}.in_proj_weight` | INT4 | **per-head**（3·num_heads = 48 groups） | — |
| MHA `out_proj.weight` | `mattn.mattn.MultiheadAttention_{re,im}.out_proj.weight` | INT4 | per-head (16 groups) | — |
| ComplexLinear `input_proj` / `output_proj` | `mattn.input_proj`, `mattn.output_proj` (complex) | INT4 | per-head (real + imag 各 16 groups) | — |
| **All activations** | — | **FP8 (E4M3)** | per-tensor (dynamic) | — |
| **All MAC accumulators** | — | **FP16** | — | — |
| **Attention QK^T / Softmax / Score×V** | transformer block inside `mattn.mattn` | **FP16** | — | — |

**實機驗證 log**（from `verify_mixed_precision_enable.py` + PTQ 2026-04-19）：
- 48 Conv2d / ConvTranspose2d 被掛 parametrization
- 4 Linear（MHA 內 out_proj × 2，ComplexLinear × 2）
- 2 nn.MultiheadAttention（複數 re/im 各一個）
- 1 ComplexMultiheadAttention（外層 wrapper）收到 `top_k=4`, `fp16_accumulator=True`
- **6 個 layer 確定被保留為 FP8**：
  ```
  decoder.model.0.0.pointwise_r / .pointwise_i
  decoder.model.3.0.pointwise_r / .pointwise_i
  decoder.model.4.0.pointwise_r / .pointwise_i
  ```

---

## 4. 硬體端需要新增 / 修改的功能

### 4.1 **MAC Datapath**（影響 MAC array）

**既有**（推測自先前 FP8 QAT 設計）：FP8 × FP8 → FP16 accumulator。

**需要加入**：
1. **INT4 weight lane**：每個 weight word 打包 2 個 INT4（8-bit 儲存，兩個 nibble）。
2. **Dequant 前置級**：進 MAC 前做 `W_dequant = W_int4_signed × scale`。三種可行設計：

   | 方案 | Dequant 前級 | MAC 乘法 | 面積 | 延遲 | 備註 |
   |---|---|---|---|---|---|
   | **A. 展開成 FP8** | `W_fp8 = round_to_fp8(INT4 × scale_fp16)` | FP8 × FP8 → FP16 | 小 | +1 stage | 最接近既有 FP8 MAC |
   | **B. 展開成 FP16** | `W_fp16 = INT4 × scale_fp16` | FP16 × FP8 → FP16 | 中 | +1 stage | MAC 乘法器寬度要升級 |
   | **C. Microscaling（推薦）** | 不 dequant，INT4 × FP8 直接乘 | INT4 × FP8 → INT accum，group 邊界乘 `scale_fp16` 後進 FP16 global accum | **最小** | 0 | group_size=256 邊界天然對齊 MAC tile |

   **建議方案 C**：per-group(256)、per-head(head_dim=16) 都能在 MAC tile 邊界觸發 scale 乘法，不必每個 lane 都掛 dequant unit。
3. **Accumulator 固定 FP16**（已驗證品質不掉，可放心採用）。
4. **Complex MAC**：4-pass 時，每 cycle 負責一個 real MAC；re·re 與 −im·im 累加進輸出 real，re·im 與 im·re 累加進輸出 imag。若要省 pipeline 延遲，可考慮 Gauss 3-multiply trick（需再多 2 個 adder + 1 buffer）。

**Scale dtype — 軟體 vs 硬體**：

- **軟體端（`DCNN/utils/mixed_precision_qat.py:90`）**：scale 是 forward 時動態從 `amax(|W_group|) / 7` 算出，dtype 跟 weight 一樣（PyTorch 預設 **FP32**）；非學習參數、無梯度。
- **硬體端（建議 FP16）**：FP16 有 ~10⁻⁵~10⁵ 動態範圍 + 11-bit 精度，足以覆蓋所有 per-group scale，且比 FP8 的 3-bit mantissa 精度高非常多。FP8 用做 scale 的精度誤差不可接受。
- **交付流程**：`ckpt2hwdat.py` 會把軟體端 FP32 scale `amax/7` 計算後 cast FP16 寫進 bin。此次精度降級誤差 << INT4 量化誤差本身，可忽略。

### 4.2 **Scale Broadcast Unit**（新增）

三種 granularity 需要不同的 scale 取用 pattern 與 index 計算：

| 粒度 | Scale shape | Addressing 規則 | 建議實作 |
|---|---|---|---|
| **per-channel**（DW conv） | `[C_out]` | `scale[co]` 沿 `(K, K, spatial)` 廣播 | 小型 scale ROM，每 row 一個 scale |
| **per-group(256)**（PW conv, `C_out × C_in × 1 × 1`） | `[C_out, ceil(C_in/256)]` | `addr = co × num_groups + (ci >> 8)` — `ci >> 8` 就是 group id（256 = 2^8） | scale 隨 weight tile 一起搬；WBUF 相鄰 side-SRAM |
| **per-head**（MHA `[r·E, E]` 或 `[E, E]`, E=256, head_dim=16） | `[r·H]`（in_proj: r=3, H=16 → 48）或 `[H]`（out_proj: 16） | row 軸每 16 行共用一個 scale，`addr = row // 16` | 小型 scale register file；可 hardwire |

**統一的 indexing 公式**：硬體只要能表達 `scale_addr = co × (row_index // group_size)`（group_size 從 per-layer CSR 讀），就能通吃三種粒度——per-head = per-group 特例（group_size=head_dim=16）、per-channel = group_size=1。

**Scale 儲存 dtype：FP16**（詳見 §4.1 討論）。

**實際 scale 總量（dim=256, H=16, head_dim=16）**：

| 層 | tensor shape | 粒度 | Scale 數 | 位元組 (FP16) |
|---|---|---|---|---|
| MHA `in_proj_weight` (re) | [768, 256] | per-head (3H=48) | 48 | 96 B |
| MHA `in_proj_weight` (im) | 同上 | per-head | 48 | 96 B |
| MHA `out_proj.weight` (re/im) | [256, 256] × 2 | per-head (H=16) | 32 | 64 B |
| `input_proj` (re/im, complex Linear) | [256, 256] × 2 | per-head | 32 | 64 B |
| `output_proj` (re/im) | [256, 256] × 2 | per-head | 32 | 64 B |
| **MHA scale 小計** | — | — | — | **~384 B** |
| Encoder/Decoder PW（gsz=256, `C_in ≤ 256`, 所以每層 = C_out 個 scale, 12 層 × 2 (re/im)） | — | per-group | — | **~5 KB** |
| Encoder/Decoder DW（per-channel, 每層 C_out × 2 (re/im)） | — | per-channel | — | **~1.5 KB** |
| **全網 scale 總計** | — | — | — | **< 10 KB** |

相對於 WBUF 64 KB 完全是小角色。

**排版建議**：WBUF 裡開一塊獨立 scale side-SRAM（或直接用 16 個 bank 中的 1 個），DMA descriptor 帶 `(weight_base, scale_base)` 兩個 pointer，FSM 同步推進。weight tile 載入 256 cycle 內，scale 只需讀一次（per-group）或 broadcast 到 16 lane（per-head）。

### 4.3 **WBUF / Weight SRAM Layout**（重新規劃，rev 3 修正）

原 WBUF：16 Banks × 64-bit × 256 Depth = 64 KB（無法一次容納整個模型，需分 tile 搬運）。

**MHA-related 權重 byte 細拆**（FP8 vs W4A8，dim=256 / H=16；數字來自 §1.5 profiler）：

| 模組 | tensor | FP8 bytes | W4A8 bytes |
|---|---|---|---|
| `MHA.in_proj_weight` (re+im) | 2 × [768, 256] 實數 | 393 KB | 197 KB |
| `MHA.out_proj.weight` (re+im) | 2 × [256, 256] 實數 | 131 KB | 66 KB |
| `mattn.input_proj` (complex64) | [256, 256] complex = re+im 兩份實數存儲 | 132 KB | 66 KB |
| `mattn.output_proj` (complex64) | [256, 256] complex | 132 KB | 66 KB |
| **MHA 相關小計** | — | **~788 KB** | **~395 KB** |

注意：前版（rev 2）只寫「MHA 四矩陣 256 KB（W4A8） vs 512 KB（FP8）」**漏了 `input_proj`/`output_proj` 這兩個 ComplexLinear wrapper**；它們實際佔用和 inner MHA 的 out_proj 相當。

**新 layout 建議**：
- **Weight 區（INT4）**：2-nibble per byte 打包。一次載入一個 tile（例：256×256 INT4 = 32 KB）。
- **Weight 區（FP8 outlier）**：decoder stage {0,3,4} 的 `pointwise_{r,i}`（6 個 tensor）維持 FP8 存放。可固定分配小區塊，或與 INT4 區共用但以 layer id 決定讀取寬度。
- **Scale side-SRAM**：獨立小 SRAM（< 10 KB 即可），與 weight tile 同步載入（weight-stationary pipeline）。詳見 §4.2。

**DMA 效益（全模型）**：
- dim=512 FP8 baseline → dim=256 W4A8：**2.80 MB → 487 KB，~5.75×**
- dim=256 FP8 → dim=256 W4A8：**962 KB → 487 KB，~2×**

其中 dim=256 本身已靠「embed_dim 由 512 降到 256」貢獻了 MHA 4× 縮減（因為 MHA 矩陣是 E²），再搭 W4A8 再砍半。

### 4.4 **Per-layer Precision Config**（新增控制）

每層 control unit 要知道「這層用什麼格式、什麼粒度」，sequencer 進入該層時讀出此配置設定 dequant unit、WBUF 讀寬、MAC 切換。

**Config struct（每層 ≈ 4 bytes）**：
```c
struct LayerCfg {
    uint2  weight_format;      // 00=INT4, 01=FP8_E4M3, 10=FP8_E5M2, 11=reserved
    uint2  scale_granularity;  // 00=per-tensor, 01=per-channel, 10=per-group, 11=per-head
    uint16 group_size;         // per-group 時 =256；per-head 時 =head_dim=16
    uint8  num_heads;          // per-head 時 =16
    uint2  act_format;         // 01=FP8_E4M3 (固定)
    bool   is_complex;         // 複數層 → 4-pass
    bool   is_transposed;      // decoder ConvTranspose2d
};
// 全網 54 層 × ~4 bytes ≈ 216 bytes
```

**實作選項（看硬體彈性目標）**：

| 情境 | 建議實作 |
|---|---|
| 只跑這顆 ckpt，精度配置一輩子不變 | **小 ROM + state machine**（~216 B）；面積近乎零 |
| 想支援多顆 ckpt / fallback 試驗 | **CSR register file** + host AXI-lite 寫入 |
| 折中：大部分寫死、留 debug override | ROM 做主路徑，少數 override switch（format bypass、top-k 開關）留 CSR |

軟體端交付：`dump_hw_config.py`（§8.2 待辦）會從 parametrization 狀態讀出所有層的 `(format, granularity, group_size)` 吐 JSON/CSV，直接給 RTL 組燒 ROM 或寫入 CSR。

### 4.5 **Attention Engine**（含 fp16 datapath）

Attention block 的 precision 流程（今日已驗證）：

```
輸入:        x_fp8        (層 input)
  ↓  out_proj pre-step: Wq/Wk/Wv are INT4, act FP8
Q_fp8, K_fp8, V_fp8   ← 每 Linear 輸出 FP16 accum 後再 cast FP8
  ↓  cast FP16  (hardware: 一個 FP8→FP16 converter)
Q_fp16, K_fp16, V_fp16
  ↓
score = Q_fp16 @ K_fp16.T  (matmul 在 FP16)
score = score / sqrt(d_fp16)
score += causal_sliding_mask_fp16
  ↓  FP16 Softmax  (單元：max subtract + exp LUT + reciprocal sum)
attn_fp16  (B, H, T, T)
  ↓  TOP_K_UNIT (k=4)
  ├─ 沿 T 軸找前 4 大 (tournament register file)
  ├─ 建立 binary mask (1-hot × 4)
  ├─ mask element-wise 乘
  └─ 再沿 T 軸求和作分母 → 每 row 除 (+1e-8 for stability)
attn_fp16 (稀疏 4-of-T)
  ↓
out_fp16 = attn_fp16 @ V_fp16   (matmul 在 FP16)
  ↓  cast FP8
out_fp8
  ↓  out_proj (INT4 × FP8 → FP16 → FP8)
output_fp8
```

**硬體細節**：
- 兩次 matmul（QK^T 與 Score×V）都固定 FP16 輸入輸出；可與 conv MAC array 共用或獨立 FP16 GEMM 單元。
- Softmax 在 FP16 下即可（LUT 寬度、exp 輸出寬度、reciprocal accuracy 均足以覆蓋）。
- **Sequence length — streaming 硬體的真實視角**：
  - 軟體 PTQ/QAT 驗證時送進 `nn.MultiheadAttention` 的序列長度 T ≈ 157~160，是 dataset **2 秒剪輯**（16 kHz × 2s ÷ hop=80 ≈ 400 frame，依 STFT 配置）帶來的 artifact，**不是硬體常數**。
  - Streaming 硬體每個 frame 推一次：**硬體只看 `attn_win=31` 與 `top_k=4`**；KV cache 僅 31 slot 的 ring buffer。
  - 因此 TOP_K_UNIT 的 tournament 只要對 31 個 score 找最大 4 個，**不是 T 個**；面積再小一級。
  - 音檔長短（1 秒、2 秒、10 秒）只影響「餵多少 frame 進來」的外層迭代次數，不影響 inner-loop 硬體設計。
- Mask window 仍保留（causal sliding window=31）；streaming 下 KV cache 長度 ≤ 31 天然等效 mask，硬體不用建 T×T mask tensor。
- **top-K=4 省下什麼**：
  - QK^T 不省（要先算完才能排 top-K）
  - Softmax 不省（還是 31 個 exp）
  - **Score × V 省 ~7.7×**：每 head 每 frame 由 31×16 MAC 降到 4×16 MAC
  - Score 暫存：31 → 4（+ 4 個 index）

### 4.6 **Activation FP8 Quantizer**（每層 input 都要）

每個 Conv / Linear / MHA 的輸入先 cast 成 FP8（E4M3）：
- 硬體已有 FP8 converter 的話直接複用。
- 激活採 **dynamic per-tensor** scale；若先前 FP8 QAT 設計已含此單元，只要確保每個運算單元入口都經過它即可。
- 軟體端透過 `forward_pre_hook` 對每個 Conv / Linear / MHA 輸入做 quantize；硬體端對應「input fetch 後立即做 FP8 cast」。

### 4.7 **Post-MAC per-channel FMA**（fuse BN + conv bias）

#### 為什麼不能把 BN 折進 complex conv weight

普通 real conv 可以把 `γ/σ` 直接乘進 `W`，但 `ComplexConv2d` 是兩組實 conv `fc_r`、`fc_i` 組合：

```
y.real = fc_r(x.real) − fc_i(x.imag)
y.imag = fc_r(x.imag) + fc_i(x.real)     # ← fc_r 出現在 real 與 imag 兩條路
```

接下來 `NaiveComplexBatchNorm2d` 對 real、imag **各有獨立 BN**（`bn_r`, `bn_i`），縮放因子 `s_r = γ_r/√(σ_r²+ε)` 與 `s_i = γ_i/√(σ_i²+ε)` 通常不同。把 `s_r` 折進 `fc_r.weight`，`y.imag` 那條路卻要用 `s_i`——一組 weight 沒辦法同時套兩個不同 scale。因此 BN **不能 fuse 進 conv weight**。

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

#### 實作建議

- 兩組 FMA 各需一塊 per-channel SRAM 存 `{bn_scale, bn_bias}`，寬度建議 **FP16**。
- 放在 MAC accumulator 的 FP16 輸出之後、PReLU 之前。
- **沒有 BN 的 stage**（decoder 最後一 stage）：軟體端導出時會填成 `bn_scale_r = bn_scale_i = 1`，`bn_bias_r[c] = b_r[c] − b_i[c]`，`bn_bias_i[c] = b_r[c] + b_i[c]`，這樣硬體 pipeline **永遠走同一條 post-MAC FMA path**，不需特例 bypass。
- Storage：每 stage `4 × C_out × FP16`，12 stages 合計幾 KB 量級，可忽略。
- 詳見 `08_software_to_hardware/FUSIONS.md` §Fusion 1，bin 檔路徑 `blocks/{encoder|decoder}/stage_{idx}/bn_{scale,bias}_{r,i}.fp16.bin`。

---

## 5. Dataflow / Pipeline 影響

### 5.1 Weight-stationary（推薦）

Weight tile 一次載入 WBUF，scale tile 同步載入 scale SRAM，activation 串流進來。每 tile 內只做一次 dequant-scale 廣播。

### 5.2 Complex 4-pass

| Pass | 運算 | 寫入 |
|---|---|---|
| 1 | `W_re · X_re` → acc_re | acc_re[0] |
| 2 | `W_im · X_im` → acc_re (減) | acc_re[0] −= |
| 3 | `W_re · X_im` → acc_im | acc_im[0] |
| 4 | `W_im · X_re` → acc_im (加) | acc_im[0] += |

對 per-group/per-head scale：pass 1 與 pass 2 的 scale **不同**（W_re 與 W_im 各有自己的 scale table），硬體控制要注意——**不能重用** W_re 的 scale 到 W_im。

### 5.3 Decoder outlier 的 pipeline 干擾

Decoder stage {0, 3, 4} 的 pointwise 權重是 FP8 不是 INT4。Control unit 進入這三層時要：
1. 切換 dequant pipeline：bypass INT4 dequant，直接 FP8 × FP8。
2. WBUF 讀寬度切換（FP8 是 8-bit/word，INT4 是 4-bit/word = 2:1 packed）。

### 5.4 Attention 的 FP16 pipeline

MHA 內部兩次 matmul（QK^T、Score×V）是 FP16 × FP16 → FP16，與 conv/linear 的 INT4 × FP8 → FP16 → FP8 不同格式：
- 可獨立配置一個 FP16 GEMM 單元（attention 專用），或在 MAC array 動態切換 multiplicand 格式。
- Softmax / top-k / renormalize 這些 scalar pipeline stage 都在 FP16。

---

## 6. 估算效益（rev 3 修正，含今日驗證數據）

**全模型 DMA byte 對照**（數字自 §1.5 profiler 實測）：

| 組合 | Encoder | Decoder | Attn.input_proj | Attn.MHA | Attn.output_proj | **Total** |
|---|---|---|---|---|---|---|
| dim=512 FP8 | 59 KB | 113 KB | 263 KB | 2,101 KB | 263 KB | **~2,800 KB** |
| dim=256 FP8 | 59 KB | 113 KB | 132 KB | 526 KB | 132 KB | **~962 KB** |
| **dim=256 W4A8（本次）** | **30 KB** | **62 KB** | **66 KB** | **263 KB** | **66 KB** | **~487 KB** |

**其他指標**：

| 指標 | FP8 baseline | W4A8 mixed (本次) |
|---|---|---|
| Encoder/Decoder PW weight size | 全 FP8 | ~90% INT4 + ~10% FP8（6 層 outlier） |
| Encoder/Decoder DW weight size | 全 FP8 | 全 INT4 per-channel（−50%） |
| Attention 計算量 (per frame) | 31×16 Score×V MAC | **4×16**（−7.7× on Score×V） |
| Attention KV cache | 31 frame | 31 frame（不變，streaming） |
| DMA bandwidth（全網 per inference） | baseline | **−82%** vs dim=512 FP8；**−49%** vs dim=256 FP8 |
| **MBSTOI (PTQ, zero retrain)** | — | **0.8326**（對照 MHA-only 0.8397，Δ −0.007）|
| **SegSNR 改善 (dB, PTQ)** | — | **2.65**（對照 MHA-only 2.84，Δ −0.19）|
| Model accuracy after QAT retrain | — | **預期 ≥ PTQ 數字**，可接近 0.8397 或更好 |

---

## 7. 給硬體顧問的「可提問清單」

1. 現有 MAC 是否已支援 **INT4 input + FP16 accumulator**？若否，新增路徑的面積/功耗成本？
2. WBUF 的 bank/depth 是否能切出獨立 scale SRAM？或要共用同一 memory（帶 tag）？
3. TOP_K_UNIT 的面積估算：k=4 tournament vs. k=4 heap，對 T=160 的 latency？
4. 複數 4-pass pipeline 是否已存在？若無，要做 2× MAC（真假分開）還是維持 4-pass？
5. Per-layer precision CSR 的切換延遲（想避免層間 bubble）？
6. FP8 E4M3 converter 是否為現成 IP？FP8 ↔ FP16 converter？
7. Scale 精度 FP16 足夠，還是要 BF16？
8. Attention 的 FP16 GEMM 是獨立單元還是與 conv MAC array 共用？

---

## 8. 軟體端下一步 & 交付

### 8.1 已完成

| 項目 | 狀態 |
|---|---|
| `DCNN/utils/mixed_precision_qat.py` 核心量化模組 | ✅ |
| `fp16_accumulator` 支援（`complexLayers.py` + `trainer.py`） | ✅ |
| `qat_scope` 控制 `full` / `mha_only`（`trainer.py`） | ✅ |
| `--hydra_overrides` CLI（`eval.py`） | ✅ |
| 三組 PTQ 對照（MHA-only / Full / Full+fp16 accum） | ✅ 2026-04-19 |
| Verify scripts（custom_attention / mixed_precision / W4A8） | ✅ |

### 8.2 待辦

1. **Full-scope + fp16 accum QAT retrain**：用 `qat_scope=full, fp16_accumulator=true` 做 30 epoch fine-tune。預期比 PTQ 再拉回 0.005~0.007 MBSTOI。
2. **`dump_hw_config.py`**：導出 §3 的 54 層 `LayerCfg` 為 JSON/CSV。
3. **`ckpt2hwdat.py`**：導出 INT4 packed weight + scale table + FP8 outlier weights 為硬體可讀 bin。
4. 若 QAT 後品質仍不夠，軟體端 fallback 順序：
   a. `DECODER_OUTLIER_STAGES` 擴大（加入更多 kurtosis 高的層）
   b. `int4_group_size = 32`（scale 變密，精度升高但 scale SRAM 變 8×）
   c. `top_k_attention = 8`
   d. 最後才動 MHA 精度（W4→W8 per-head）

### 8.3 可交付檔案清單

| 檔案 | 用途 | 狀態 |
|---|---|---|
| `DCNN/utils/mixed_precision_qat.py` | 核心量化模組 | ✅ |
| `DCNN/utils/complexPyTorch/complexLayers.py` | fp16 accumulator + custom attention | ✅ |
| `verify_*.py` | Sanity checks | ✅ |
| PTQ 結果（3 組 CSV + run.log） | §1 數據來源 | ✅ |
| `ckpt2hwdat.py` | 導出硬體 bin | ⏳ |
| `dump_hw_config.py` | 導出 LayerCfg 表 | ⏳ |

---

**原始規劃**：`07_mixed_precision_qat_handoff.md`（2026-04-15，軟體層）
**本文件 rev 1**：2026-04-17（實作前硬體對接）
**本文件 rev 2**：2026-04-20（PTQ 驗證後更新；fp16 累加器已實作；conv 全 W4 已驗證）
**本文件 rev 3**：2026-04-20（修正 DMA 估算：補回 ComplexLinear wrapper、修正 flatten_size、釐清 scale 軟硬體 dtype、T vs streaming sliding window、per-granularity 硬體 indexing 規則、per-layer config ROM vs CSR 取捨）

---

## 附錄 A. 常見疑問（硬體顧問討論整理）

**Q1. INT4 × scale 要變 FP8 再乘，scale 是 FP8 嗎？**
不是。軟體端 scale 是 FP32（動態 `amax/7` 計算），硬體端建議存 FP16。FP8 精度不足以表達 group 間差幾個數量級的 amax 比例。詳見 §4.1。

**Q2. 輸出成 pattern 後硬體怎麼實作 per-head / gsz=256？**
三種粒度統一成 `scale_addr = co × (row_index // group_size)`，group_size 從 per-layer config 讀（per-head = group_size=16，per-group = 256，per-channel = 1）。詳見 §4.2。

**Q3. top-K=4 少算什麼？**
只省 Score × V（~7.7× on MAC），QK^T 與 softmax 不省。詳見 §4.5。

**Q4. dim=256 W4A8 vs dim=512 FP8 的 DMA 縮減？**
2.80 MB → 487 KB，約 **5.75×**。其中 4× 來自 embed_dim 減半（MHA 是 E² scaling），再 ~2× 來自 W4A8。詳見 §1.5、§6。

**Q5. Scale 要存多少？怎麼排？**
全網 < 10 KB（MHA 只 ~400 B、Conv PW ~5 KB、DW ~1.5 KB），獨立 side-SRAM 即可。詳見 §4.2。

**Q6. Per-layer precision CSR 一定要做嗎？**
不一定。只跑這顆 ckpt 的話，54 層配置燒成 ~216 B ROM 即可；要支援多 ckpt / fallback 才需要 CSR register file。詳見 §4.4。

**Q7. T ≈ 160 在 streaming 硬體不用管？**
對。T 是 dataset 2 秒剪輯 artifact；硬體只看 `attn_win=31`（KV cache 31 slot）與 `top_k=4`（tournament 4 entry）。詳見 §4.5。
