# RTL Handoff: Attention Softmax Replacement

日期：2026-05-14
Model lineage：`09_weight_tied_mha/`（strict W4A8 + top-k=4 + tied MHA）
Active implementation / validation reference：`11_weight_tied_mha_fast_mbstoi/`（parallel-eval, ~13× 加速版）
Python reference：`11_weight_tied_mha_fast_mbstoi/DCNN/utils/complexPyTorch/complexLayers.py`
Note：final `shift_topk_pow2_clamp8_nomantissa` mode 目前是在 `11_weight_tied_mha_fast_mbstoi/` 實作與驗證；若後續要以 `09_weight_tied_mha/` 為唯一 source tree，需先把該 mode sync 回 `09`。

## 0. TL;DR

| Decision | Mode | 為什麼 |
|---|---|---|
| **⭐⭐ Primary RTL target (minimal area)** | `shift_topk_pow2_clamp8_nomantissa` | 在 `shift_topk_pow2` 上再疊兩個無代價簡化：clamp 範圍 -16 → -8（max right shift 8，含 zero 共 9 個 shift amount）+ mantissa-free（移除 `(1+f)` 路徑）。MBSTOI 0.8360，與 nearest baseline 完全一致。**softmax probability block 可做到無 general-purpose multiplier、無 divider、無 exp LUT，只剩 top-k selector + adder/subtractor + barrel shifter + LZD；QK/AV 的 MAC 不在此 claim 範圍內** |
| **⭐ Conservative primary** | `shift_topk_pow2` | 帶 mantissa 與 clamp -16 的 fully-featured 版本；如要保留 (1+f) 精度做為 safety margin 可選此 |
| **Golden fallback** | `shift` | 不 fuse top-k 的版本，MBSTOI +0.0021 略勝 baseline。若 top-k 設計未來會調整（如改 K=8）保留此路徑當 reference |
| **Debug equivalence** | `shift_topk_exact` | 同 fuse pipeline 但 denom 用 exact sum 不 round to pow2；與 `shift` 數值等價（差 < 1e-6），純驗證 fusion 邏輯正確 |
| **Plan B (低面積但需 divider)** | `relu2` | MBSTOI 掉 0.0045，硬體簡（ReLU + 平方），但需 true reciprocal/divider，無需 finetune |
| **Deferred** | `taylor2` | 不 finetune 無法用（MBSTOI 掉 0.06，ILD/IPD 比 noisy 還差） |

所有 viable 候選都通過 full 750-pair × 8 SNR VCTK paper-style eval；TIMIT (unmatched) cross-domain 目前已針對 conservative `shift_topk_pow2` 跑完，數據在 §3.5。minimal-area `shift_topk_pow2_clamp8_nomantissa` 已完成 VCTK sweep，建議在 tape-out 前補跑 TIMIT。**Micro-architecture sweep（rounding policy / clamp / mantissa）皆已完成**，見 §3.4。

---

## 1. 設計脈絡與目標

### 1.1 來自模型側的限制
- BCCTN binaural enhancement，ICASSP 2024 baseline，已做 strict W4A8 QAT + per-group scalar + tied MHA（MHA_re ≡ MHA_im）
- Attention 是 MHA（dim=256, heads=16），sliding-window 因果遮罩 `attn_win=31`。在目前 code 中 `attn_win` 是最大 past distance，所以每 row 最多可看 **31 past + current = 32** entries；softmax 之後再做 hard top-k=4
- 既有 ASIC 規格：64 complex MAC 1D flat、WBUF 32 KB / UFBUF 4 KB、L+R 共享權重、features 全程 on-chip
- 詳見 `CLAUDE.md` §「BCCTN ASIC 硬體規格」

### 1.2 為什麼要替換 softmax
- 原始 softmax 需 exp + division，exp LUT / Padé / fixed-point exp 都是面積大戶
- Attention probability path 是 02 hardware-aware analysis 列出的三大瓶頸之一
- 目標：用 adder / shifter / 簡單算術元件取代 exp + division，且 model 端不 retrain（如果可能）

### 1.3 候選空間
參考文獻分類（其他三條線屬於不同 scope，本 handoff 不處理）：
1. ✅ Shift-based softmax approximation — ISCAS 2023 hardware-efficient softmax，本 handoff 主推
2. ✅ Polynomial / Taylor softmax — TaylorShift，本 handoff 已驗證 eval-time 不可用
3. ✅ ReLU² / power normalization — ReLA / ReLUFormer，本 handoff 列為 plan B
4. ❌ Sparse projection (sparsemax/entmax) — 已有 top-k=4 後處理，邊際效益不大
5. ❌ Softmax-free attention (SimA, ReLUFormer, linear attn) — 改架構，需 retrain
6. ❌ Softermax / LUT exp — 保留 exp，與「exp-free」目標衝突

---

## 2. RTL 規格

### 2.1 操作上下文

```
非 fused pipeline (shift / relu2 / taylor2):
  QKV proj -> QK^T / sqrt(D) -> sliding-window mask
           -> [SOFTMAX BLOCK]  <- 本 handoff 範圍 (最多 32 entries)
           -> top-k=4 hard mask -> renorm -> attn × V -> out_proj

Fused pipeline (shift_topk_pow2 / shift_topk_exact, ⭐ primary):
  QKV proj -> QK^T / sqrt(D) -> sliding-window mask
           -> [FUSED SOFTMAX+TOPK BLOCK]   <- primary 範圍
              {score-domain top-k(4) -> 4-entry shift weight -> pow2 denom -> scatter}
           -> attn × V -> out_proj         (top-k hard mask + renorm 已 fused)
```

- Input：attention scores / logits `attn[B, H, T, T]`（來自 `QK^T / sqrt(D)`，已套用 sliding-window / causal visibility；這是 **softmax 前** 的 score，不是 probability）
- Software mask 表現：PyTorch dense reference 會把不可見位置（future 或 sliding-window 外）加上 `-inf`，讓 softmax/top-k 永遠選不到那些位置。
- RTL streaming 表現：不需要真的產生或儲存 `-inf`。scheduler 若保證只送入 past/current visible keys，softmax block 只需處理 valid window entries；early rows 或固定 32-lane datapath 可用 `valid_mask`/`valid_count` 控制哪些 lane 有效。
- Sliding window：目前 code 使用 `dist = query_idx - key_idx` 且 `dist > attn_win` 才 mask。`attn_win=31` 因此代表最多 **32** valid entries（current + 31 past），而不是總長 31。在 streaming RTL 中可視為最多 32 筆 score 的 valid window，而不是完整 T×T row。
- Output：sparse probability `attn_probs[B, H, T, T]`
  - Primary（fused）：每 row 只 4 個非零 entry，其餘為 0；row-sum 因 pow2 denom 不一定恰為 1（誤差 ≤ 1.5×）
  - Fallback / Plan B：row-sum = 1.0，仍經外部 top-k=4 mask
- Top-k=4 邏輯：primary 路徑 **已包含**；fallback 路徑保留外部 top-k mask 不變

### 2.2 數值精度建議

primary = `shift_topk_pow2_clamp8_nomantissa`（minimal area），K=4，clamp -8，無 mantissa：

| Stage | 建議精度 | 註 |
|---|---|---|
| Score input | FP16 / BF16 | 已驗證 fp16_accumulator 路徑可用，見 `enable_mixed_precision_qat` |
| Top-K selector | int idx + FP16 score | K=4，從最多 32 個 valid 中選 |
| Row-max subtract | FP16 | 只算 4 個 entry |
| `1.5 × x` | Q4.4 或 FP16 | `1.5x = x + x>>1`，無 mul |
| Floor only | int4 | clamp `floor ∈ [-8, 0]`；**fraction path 已移除** |
| Mantissa | — | **已移除**（無 `(1+f)` 計算） |
| `2^k`（純 shift） | k ∈ [-8, 0] | max right shift 8，含 zero 共 9 個 shift amount（從原本 max right shift 16 縮小） |
| Denominator sum | FP16 accumulator | 只累加 4 個 pow-of-2 weight |
| `round(log2(denom))` | int4 | 用 leading-one detector + LSB peek |
| Output probability | FP16 / FP8 | 4-entry barrel shift；N 個位置中只 4 個非零 |

對照 §4.1.1 conservative primary（含 mantissa、clamp -16）會多 fraction path（Q1.12）與更寬的 shift range。建議第一版 prototype 用 FP16 全程，bit-width sweep 屬未來工作（§5.1）。

目前軟體 eval 是 PyTorch reference / fake-quant 行為驗證，通常仍以 fp32 arithmetic 執行 attention block（除非打開 `training.fp16_accumulator=true`）。因此 §3 的結果不是 bit-accurate FP16 或 fixed-point RTL 結果；RTL 前仍需做 §5.1 的 FP16/BF16/Q-format rerun。

---

## 3. Quality 驗證資料（all full 750 pairs × 8 SNR VCTK paper-style）

### 3.1 主表

| Metric | Baseline (softmax) | **shift_topk_pow2** ⭐ | shift | shift_topk_exact | relu2 | taylor2 ❌ |
|---|---|---|---|---|---|---|
| MBSTOI Noisy | 0.7861 | 0.7861 | 0.7861 | 0.7861 | 0.7861 | 0.7861 |
| **MBSTOI Enhanced** | **0.8373** | **0.8360** | 0.8394 | 0.8394 | 0.8328 | 0.7770 |
| Δ vs baseline | 0 | **-0.0013** | +0.0021 | +0.0021 | -0.0045 | -0.0603 |
| SegSNR Noisy | -0.83 | -0.83 | -0.83 | -0.83 | -0.83 | -0.83 |
| SegSNR Enhanced | 1.50 | 1.95 | 2.06 | 2.06 | 2.08 | 0.61 |
| SegSNR Improvement | 2.33 | 2.78 | 2.89 | 2.89 | 2.91 | 1.44 |
| ILD Error Noisy (dB) | 3.7141 | 3.7141 | 3.7141 | 3.7141 | 3.7141 | 3.7141 |
| ILD Error Enhanced (dB) | 3.2505 | 3.2304 | 3.2264 | 3.2271 | 3.2742 | **3.8963** ⚠️ |
| IPD Error Noisy (°) | 15.43 | 15.43 | 15.43 | 15.43 | 15.43 | 15.43 |
| IPD Error Enhanced (°) | 15.68 | 15.69 | 15.62 | 15.63 | 15.82 | **16.87** ⚠️ |

⚠️ taylor2 Enhanced ILD/IPD **比 Noisy 還差** → attention 分布 collapse，空間資訊被破壞。

### 3.2 結論
- `shift_topk_pow2` **⭐ 主推**：MBSTOI 距 baseline 僅 -0.0013（在跨 run 噪聲層級內），SegSNR/ILD 仍勝過 baseline，IPD 持平。算術量降 ~8×（4 entries vs 32），輸出天然 sparse，外部 top-k+renorm 可移除。
- `shift_topk_exact` 與 `shift` 數值幾乎完全相同（MBSTOI 都 0.8394，ILD/IPD 差 < 0.01）→ 確認 fusion 邏輯正確；只差在 denom 是否 round 到 pow2。
- `shift`（不 fused）在所有 metric 上 ≥ baseline，保留作為 reference / golden fallback。
- `relu2` 在 MBSTOI 上小幅落後（-0.0045），但 SegSNR 仍較佳。需要 true divider，硬體面積評估時要納入。
- `taylor2` eval-time 完全不可用，除非 retrain 否則 RTL 端不要實作。

### 3.3 原始輸出位置

```
09_weight_tied_mha/results/ptq_strict/                                                          # baseline
11_weight_tied_mha_fast_mbstoi/softmax_rtl_shift_based/results/shift_strict_w32/                # shift
11_weight_tied_mha_fast_mbstoi/softmax_rtl_shift_based/results/shift_topk_exact_strict_w64/     # debug
11_weight_tied_mha_fast_mbstoi/softmax_rtl_shift_based/results/shift_topk_pow2_strict_w64/      # ⭐ primary
11_weight_tied_mha_fast_mbstoi/softmax_rtl_shift_based/results/shift_topk_pow2_floor_w64/       # rounding sweep
11_weight_tied_mha_fast_mbstoi/softmax_rtl_shift_based/results/shift_topk_pow2_ceil_w64/        # rounding sweep
11_weight_tied_mha_fast_mbstoi/softmax_rtl_shift_based/results/shift_topk_pow2_clamp8_w64/      # micro-arch sweep
11_weight_tied_mha_fast_mbstoi/softmax_rtl_shift_based/results/shift_topk_pow2_nomantissa_w64/  # micro-arch sweep
11_weight_tied_mha_fast_mbstoi/softmax_rtl_shift_based/results/shift_topk_pow2_clamp8_nomantissa_w64/  # ⭐ minimal-area
11_weight_tied_mha_fast_mbstoi/softmax_rtl_shift_based/results/shift_topk_pow2_vctk_timit_w64/  # cross-domain
11_weight_tied_mha_fast_mbstoi/softmax_rtl_taylor_relu/results/relu2_strict_w64/                # plan B
11_weight_tied_mha_fast_mbstoi/softmax_rtl_taylor_relu/results/taylor2_strict_w64/              # deferred
```

每個資料夾含 `vctk_paper_style_results_averages.txt`、`detailed_results.csv`、per-SNR 子目錄；含 TIMIT 的那個還有 `timit_paper_style_results_averages.txt`。

### 3.4 Micro-architecture sweep（全 750-pair × 8 SNR VCTK）

把 §3.1 的 primary 進一步拆解，找出能省的硬體元件：

| Variant | MBSTOI | Δ vs nearest | ILD (dB) | IPD (°) | 結論 |
|---|---|---|---|---|---|
| `shift_topk_pow2` (nearest, baseline of sweep) | 0.8360 | 0 | 3.2304 | 15.69 | reference |
| `shift_topk_pow2_floor` | 0.8271 | -0.0089 | 3.3090 | 15.81 | ❌ row-sum 偏大（sharper），偏離訓練分布 |
| `shift_topk_pow2_ceil` | 0.8157 | -0.0203 | 3.4468 | 16.00 | ❌ row-sum 偏小（smoother），更糟 |
| `shift_topk_pow2_clamp8` | 0.8362 | **+0.0002** | 3.2294 | 15.69 | ✅ **無代價**，max right shift 16 → 8 |
| `shift_topk_pow2_nomantissa` | 0.8359 | **-0.0001** | 3.2367 | 15.69 | ✅ **無代價**，拿掉 mantissa / `1+f` 路徑 |
| **`shift_topk_pow2_clamp8_nomantissa`** ⭐⭐ | **0.8360** | **0** | **3.2356** | **15.69** | ✅ **最終 RTL 候選**：clamp8 + mantissa-free 合併，無交互成本，無代價 |

### 3.5 TIMIT cross-domain（unmatched noise）— 驗證 conservative `shift_topk_pow2` 在 domain shift 下穩定

| Dataset | Noisy MBSTOI | Enhanced MBSTOI | Δ improvement | ILD Noisy → Enh | IPD Noisy → Enh |
|---|---|---|---|---|---|
| VCTK (matched) | 0.7861 | 0.8360 | +0.0499 | 3.71 → 3.23 dB | 15.43 → 15.69° |
| **TIMIT (unmatched)** | **0.7311** | **0.7927** | **+0.0616** | **3.29 → 2.98 dB** | **21.10 → 20.02°** |

TIMIT enhancement 幅度 **比 VCTK 更大**（+0.0616 vs +0.0499），ILD/IPD 也都改善。`shift_topk_pow2` 在 unmatched domain 沒有 collapse，cross-domain robust。final minimal-area `shift_topk_pow2_clamp8_nomantissa` 目前已完成 VCTK sweep，TIMIT 建議補跑一次作為 RTL sign-off 前的 sanity check。

---

## 4. 演算法規格（hardware pseudo-code）

### 4.1 `shift_topk_pow2_clamp8_nomantissa` — ⭐⭐ Primary RTL target (minimal area)

Python reference: `11_weight_tied_mha_fast_mbstoi/DCNN/utils/complexPyTorch/complexLayers.py` (`_shift_topk_softmax`, with `clamp_min=-8`, `use_mantissa=False`)

動機：在 fused top-k shift 之上，再疊兩個經驗驗證**無代價**的硬體簡化（見 §3.4 sweep 結果）：
- `clamp_min = -8`（從 -16）：top-4 內部 score spread 通常小；目前 VCTK eval 顯示 clamp8 沒有可觀測品質成本，但尚未用 histogram 證明 clamp 永遠不 trigger
- `use_mantissa = False`：拿掉 `1 + f` 線性近似，`2^s ≈ 2^floor(s)`；softmax probability block 可避免 general-purpose multiplier

```text
INPUTS:
  attn[N]       : N up to valid window size; software dense reference may contain -inf for masked positions
  valid_mask[N] : bool; True = valid (≤ 32 valid per row in current config; streaming RTL may omit invalid lanes)
  K = 4         : top-k width (from training.top_k_attention)

STEP 1  Score-domain top-k (skip masked):
  top_idx[0..K-1], top_score[0..K-1] = topk(attn where valid_mask, K, descending)

STEP 2  Centered, scaled, saturated — for K entries only:
  row_max = top_score[0]
  for i in 0..K-1:
    x[i] = top_score[i] - row_max          // in [-Δmax, 0]
    s[i] = clamp(1.5 * x[i], -8, 0)        // fixed-point RTL: 1.5x = x + (x>>1)
                                            // clamp to -8 instead of -16

STEP 3  Floor only (no fraction needed):
  for i:
    k_i[i] = floor(s[i])                   // int, range [-8, 0]
                                            // mantissa path removed entirely
    w[i]   = 1 << (8 + k_i[i])             // 2^k as fixed-point Q0.8
                                            // (or w[i] = 2^k for FP datatype)

STEP 4  Top-K denominator:
  denom = sum(w[0..K-1])                   // 4-entry adder tree

STEP 5  Round to nearest power-of-two:
  e = round(log2(denom))                   // LZD + LSB peek
                                            // (nearest is strict winner; floor/ceil verified worse)
  denom_pow2 = 1 << e

STEP 6  Scatter sparse output:
  for j in 0..N-1: p[j] = 0
  for i in 0..K-1: p[top_idx[i]] = w[i] >> e   // pow2 / pow2 = barrel shift

STEP 7  Fallback (僅 RTL 安全網，正常 K ≥ 1 不會 trigger):
  if all top_score == -inf:
    valid_count = popcount(valid_mask)
    for j: p[j] = valid_mask[j] ? 1/valid_count : 0
```

**所需硬體元件**（minimal area）：
- top-K selector（K=4，sliding window 最多 32 entries → 4-stage compare-swap 或 partial-sort tree）
- 4-entry adder/subtractor tree（centering + denom sum）
- 4 × (`1.5x = x + x>>1`) — fixed-point 下可用 shift-add 實作
- 4 × barrel shifter（`2^k`，max right shift 8；含 zero 共 9 個 shift amount）
- LZD + 1-bit rounding（`round(log2)`）
- 4 × barrel shifter（output scatter / `w >> e`）

**softmax probability block 沒有 general-purpose multiplier、沒有 divider、沒有 exp LUT、沒有 mantissa (1+f) pipeline**。其中 `1.5x` 在 fixed-point RTL 中用 shift-add 實作；若採純 FP16 datapath，則會是 exponent adjust + FP add 或 constant-multiply 類成本，不能視為零成本。QK/AV 的 MAC 乘法器仍在 softmax block 外。算術量比 §4.2 的 `shift` 少 ~8×，且 element 種類比 §4.2.1 的 `shift_topk_pow2` (with mantissa) 又進一步簡化。

**驗證資料**：MBSTOI 0.8360（與 nearest baseline 完全相同），baseline Δ=-0.0013，SegSNR/ILD/IPD 不退化。See §3.4 sweep 表的最後一列。

### 4.1.1 `shift_topk_pow2` — Conservative primary（含 mantissa 與 clamp -16）

如果 RTL 端想保守一點保留 `(1+f)` 精度，或想留 clamp 安全 margin 應對未來 K 或 attn_win 改變，這是同一個 fused fast path 的「full-featured」版本。Pseudo-code 與 §4.1 相同，差別：
- STEP 2: `clamp(1.5 * x[i], -16, 0)`（原 -8 → -16）
- STEP 3 extended: `f[i] = s[i] - k_i[i]; m[i] = 1 + f[i]; w[i] = m[i] >> -k_i[i]`

成本：多一個 8-bit fraction adder / mantissa pipeline；若用 fixed-point，可用 add+shift 實作，不一定需要 general multiplier，但 datapath 與 shift range 都比 minimal-area 版本更重。MBSTOI 0.8360（與 minimal 相同）。

**驗證資料**：見 §3.4 — MBSTOI 0.8360，與 baseline 差 -0.0013。

### 4.2 `shift` — Golden fallback (不 fused)

Python reference: `complexLayers.py:706–737` (`_shift_based_softmax`)

當 top-k 設計可能調整（如未來改 K=8 或拿掉 top-k），保留此路徑作為 reference。Pseudo-code 與 §4.1 大致相同，差別：
- **無 Step 1 top-K selector**：對最多 32 valid entries 全跑
- **Step 6 denom**：累加最多 32 個 weight（少數可能因 mask 為 0）
- **Step 8 scatter**：對最多 32 valid 位置全部寫，masked 位置寫 0
- 配合外部 top-k hard mask + renorm 才能得到 sparse 4-entry 輸出

```text
INPUTS: attn[N], valid_mask[N]

STEP 1  Row-max (over valid only):
  row_max = max(attn[i] for i where valid_mask[i])

STEP 2  Center & saturate:
  for i in 0..N-1:
    x[i] = valid_mask[i] ? (attn[i] - row_max) : 0
    s[i] = clamp(1.5 * x[i], -16, 0)

STEP 3-5  (同 §4.1 Step 4-5):
  k_i[i] = floor(s[i]);  f[i] = s[i] - k_i[i]
  m[i] = 1 + f[i]
  w[i] = valid_mask[i] ? (m[i] >> -k_i[i]) : 0

STEP 6  Denom over all valid entries:
  denom = sum(w[0..N-1])
  e = round(log2(denom));  denom_pow2 = 1 << e

STEP 7  Normalize:
  p[i] = w[i] >> e
```

**硬體差異**：跟 §4.1 比，需要最多 32-entry max-reduce、32-entry adder tree、32 個 weight pipeline。

**參考文獻**：ISCAS 2023 *Hardware-efficient Softmax Approximation for Self-Attention Networks* — 該論文回報 FPGA 上 LUT -15×、register -55×、latency -23×、BERT/GLUE 小於 1% accuracy drop。本工作 §4.1 的 fused 版進一步把算術量再降 ~8×。

### 4.3 `shift_topk_exact` — Debug equivalence mode

Python reference: `complexLayers.py:753–786` (`_shift_topk_softmax`, `pow2_denom=False`)

跟 §4.1 完全相同，**只差 Step 7**：不 round denom 到 pow2，用 exact reciprocal。**RTL 端不需要實作此 mode**，僅供仿真比對：

```text
STEP 7'  Exact reciprocal:
  p[top_idx[i]] = w[i] / denom    // 真除法
```

驗證用途：對同一輸入，`shift_topk_exact` 與 `shift + post-softmax top-k renorm` 應 bit-equivalent（誤差 ≤ 2.4e-7，純數值噪聲），確認 fusion 邏輯正確。詳見 [verify_custom_attention.py](verify_custom_attention.py) 的 `[shift_topk_exact] max |post-topk − fused|` 檢查項。

### 4.4 `relu2` — Plan B（低面積）

Python reference：`complexLayers.py:762–763` + `_normalize_attention_weights` (`:697–704`)

```text
INPUTS: attn[N], valid_mask[N]

STEP 1  ReLU + zero out masked:
  for i:
    r[i] = valid_mask[i] ? max(attn[i], 0) : 0

STEP 2  Square:
  for i:
    w[i] = r[i] * r[i]

STEP 3  Sum:
  denom = sum(w[0..N-1])

STEP 4  Normalize (true division required):
  if denom > eps:
    for i: p[i] = w[i] / denom
  else:
    count = popcount(valid_mask)
    for i: p[i] = valid_mask[i] ? (1.0 / count) : 0
```

**所需硬體元件**：comparator (ReLU)、multiplier (square)、adder tree、**reciprocal / divider**、popcount

跟 shift 比，多了 multiplier 與 divider，但少了 barrel shifter、leading-one detector、與整個 floor/fraction 邏輯。實際面積取決於 divider 實作方式（Newton-Raphson reciprocal vs LUT-assisted）。

### 4.5 `taylor2` — 不推薦，僅供完整性

```text
STEP 1  Mask -> 0:
  for i: x[i] = valid_mask[i] ? attn[i] : 0
STEP 2  Polynomial:
  for i: w[i] = max(1 + x[i] + 0.5 * x[i]*x[i], 0)
STEP 3  Normalize: same as relu2
```

未 finetune 不可用，本工作不再投入。

---

## 5. 開放問題 / 未來工作

### 5.1 必做 before tape-out
- ✅ **TIMIT cross-domain eval（conservative `shift_topk_pow2`）**：已完成，見 §3.5。`shift_topk_pow2` 在 unmatched noise 下沒有 collapse，improvement 幅度甚至更大。
- ✅ **Denom rounding sweep（floor / ceil / nearest）**：已完成，見 §3.4。**鎖定 nearest**，floor / ceil 均顯著退化。
- ✅ **Clamp 縮窄與 mantissa-free 各別驗證**：已完成，見 §3.4。兩者**各別都無代價**。
- ✅ **Combined `shift_topk_pow2_clamp8_nomantissa` VCTK eval**：已完成，見 §3.4。MBSTOI 0.8360，與 nearest `shift_topk_pow2` 持平。
- **Combined `shift_topk_pow2_clamp8_nomantissa` TIMIT eval**：尚未作為獨立 sign-off 完成；建議沿用 §6.3 TIMIT command，mode 改為 `clamp8_nomantissa` 補跑。
- **Bit-width sweep**：目前驗證在 fp32 Python 上做完。需要在 FP16 / BF16 / Q-format fixed-point 下重跑 §3 的 750-pair eval，確認 MBSTOI 不掉 > 0.005。
- **Bit-accurate RTL reference**：把 §4.1 的 pseudo-code 用 fixed-point Python（torchao 或自寫 numpy）實作，跟 RTL 對 bit-accurate。

### 5.2 可選的演算法工作
- **Taylor2 finetune-rescue**：30-epoch finetune 看能否把 MBSTOI 從 0.7770 救到 ≥ 0.83。腳本已備：`softmax_rtl_taylor_relu/run_finetune_exp_free_strict.sh taylor2`
- **Mixed-mode**：早層用 shift_topk_pow2、後層用 relu2，搭配 attention 不同層的敏感度差異
- **Streaming / per-row pipeline**：current 設計假設 row-by-row，硬體可考慮 column-fused 平行化
- **slope sweep（不建議）**：把 1.5x 改成 1.25x / 1.75x。1.5 ≈ log2(e)=1.4427 已近最佳，偏離理論值通常更差，跳過。
- **TIMIT 跨候選比較**：本 handoff 只在 TIMIT 上驗證 conservative `shift_topk_pow2`，未覆蓋 relu2 / clamp8_nomantissa 等變體；若硬體決策時仍在搖擺，可補測。

---

## 6. 軟體 / 硬體 contract

### 6.1 Bit-accurate reference
- 主要參考：`11_weight_tied_mha_fast_mbstoi/DCNN/utils/complexPyTorch/complexLayers.py`
- 函式：`_canonical_softmax_mode`、`_normalize_attention_weights`、`_shift_based_softmax`、`_shift_topk_softmax`（⭐ primary）、`_approx_attention_probs`
- 整合點：`_mha_attention_forward` 把 `F.softmax(attn, dim=-1)` 改成 `_approx_attention_probs(attn, softmax_mode)`

### 6.2 Sanity test
```bash
cd /raid/R12K41024/11_weight_tied_mha_fast_mbstoi
conda run -n bcctn python verify_custom_attention.py
```
應該看到（以下所有 mode 的 shape/finite 檢查全部 OK）：
```
[real] OK
[complex] OK
[top-k] output shape ok
[shift] output shape/finite ok
[shift_topk_exact] output shape/finite ok
[shift_topk_pow2] output shape/finite ok
[shift_topk_pow2_floor] output shape/finite ok
[shift_topk_pow2_ceil] output shape/finite ok
[shift_topk_pow2_clamp8] output shape/finite ok
[shift_topk_pow2_nomantissa] output shape/finite ok
[shift_topk_pow2_clamp8_nomantissa] output shape/finite ok
[taylor2] output shape/finite ok
[relu2] output shape/finite ok
[shift_topk_exact] max |post-topk − fused| = 2.38e-07
All sanity checks passed.
```

### 6.3 Full eval reproduction
```bash
cd /raid/R12K41024/11_weight_tied_mha_fast_mbstoi

# Primary (⭐ shift_topk_pow2，nearest rounding，clamp -16，含 mantissa)
MODE=shift_topk_pow2 METRIC_WORKERS=64 bash softmax_rtl_shift_based/run_eval_shift_strict.sh

# Minimal-area (⭐ shift_topk_pow2_clamp8_nomantissa，最終 RTL 候選)
MODE=clamp8_nomantissa METRIC_WORKERS=64 bash softmax_rtl_shift_based/run_eval_shift_strict.sh

# Golden fallback (shift)
METRIC_WORKERS=64 bash softmax_rtl_shift_based/run_eval_shift_strict.sh

# Debug equivalence (shift_topk_exact)
MODE=shift_topk_exact METRIC_WORKERS=64 bash softmax_rtl_shift_based/run_eval_shift_strict.sh

# Micro-arch sweep (rounding / clamp / mantissa 個別變體)
MODE=shift_topk_pow2_floor       METRIC_WORKERS=64 bash softmax_rtl_shift_based/run_eval_shift_strict.sh
MODE=shift_topk_pow2_ceil        METRIC_WORKERS=64 bash softmax_rtl_shift_based/run_eval_shift_strict.sh
MODE=shift_topk_pow2_clamp8      METRIC_WORKERS=64 bash softmax_rtl_shift_based/run_eval_shift_strict.sh
MODE=shift_topk_pow2_nomantissa  METRIC_WORKERS=64 bash softmax_rtl_shift_based/run_eval_shift_strict.sh

# Cross-domain (TIMIT) — 主要 primary 用，含 VCTK 同跑
CKPT=outputs/finetune_strict/2026-05-10_02-40-16/logs/lightning_logs/version_0/checkpoints/epoch=45-step=44528.ckpt
METRIC_WORKERS=64 CUDA_VISIBLE_DEVICES=0 conda run -n bcctn python eval.py \
    --model_checkpoint "$CKPT" \
    --vctk_test_dir /raid/R12K41024/BCCTN/Dataset/noisy_testset \
    --vctk_clean_dir /raid/R12K41024/BCCTN/Dataset/clean_testset \
    --timit_test_dir /raid/R12K41024/BCCTN/Dataset/noisy_testset_timit \
    --timit_clean_dir /raid/R12K41024/BCCTN/Dataset/clean_testset_timit \
    --output_dir results/shift_topk_pow2_vctk_timit_w64 \
    --paper_style_eval --batch_size 8 --limit_pairs 750 --metric_workers 64 \
    --hydra_overrides training.qat_scope=full training.top_k_attention=4 \
                      model.attention_softmax.mode=shift_topk_pow2

# Plan B
METRIC_WORKERS=64 bash softmax_rtl_taylor_relu/run_eval_exp_free_strict.sh relu2
```
單 mode 約 6 分鐘（在 224-core / 1 H100 上），含 TIMIT 約 17 分鐘。

### 6.4 預期 hardware regression test
- RTL 端對 1000 個隨機 score row 跑 primary mode 與 fp32 reference 比對：要求 top-4 indices 必須完全一致（top-k selector 正確性），entry-wise probability 絕對差 < 0.01
- 在實際 BCCTN audio 上重跑 §3 的 paper-style eval：要求 MBSTOI 不退化 > 0.005 vs Python reference

---

## 7. 相關 commit / 檔案清單

```
11_weight_tied_mha_fast_mbstoi/DCNN/utils/complexPyTorch/complexLayers.py  # final combo 主實作與 eval reference
09_weight_tied_mha/DCNN/utils/complexPyTorch/complexLayers.py       # model lineage；若以 09 為 source tree，需 sync final combo mode
09_weight_tied_mha/DCNN/models/model.py                              # config 透傳
09_weight_tied_mha/DCNN/utils/mixed_precision_qat.py                 # custom forward 觸發
09_weight_tied_mha/config/config.yaml                                # attention_softmax.mode
09_weight_tied_mha/verify_custom_attention.py                        # unit sanity
09_weight_tied_mha/softmax_rtl_shift_based/                          # shift 腳本與結果
09_weight_tied_mha/softmax_rtl_taylor_relu/                          # taylor2/relu2 腳本與結果
09_weight_tied_mha/HANDOFF_SOFTMAX_RTL_2026-05-14.md                 # 軟體側 handoff
09_weight_tied_mha/SPEC_MBSTOI_PARALLEL_2026-05-14.md                # eval 平行化 spec
11_weight_tied_mha_fast_mbstoi/HANDOFF_MBSTOI_PARALLEL_2026-05-14.md # eval 平行化 handoff
11_weight_tied_mha_fast_mbstoi/softmax_rtl_*/results/*_strict_w*/    # full 750-pair 結果
```

Git ref：`master` @ 2026-05-14（包含 softmax RTL 三模式 + MBSTOI 平行化）

---

## 8. Decision log（給未來 reviewer）

1. **為什麼把 `shift_topk_pow2_clamp8_nomantissa` 升為 primary？** 從 §3.4 sweep，clamp 從 -16 縮到 -8 與拿掉 mantissa 各別都是 MBSTOI Δ 在 ±0.0002 噪聲層級內（與 nearest 持平），合併後 0.8360 與 nearest 完全相同。**沒有可觀測 quality 代價，且硬體大幅簡化**：max right shift 從 16 縮到 8、整個 mantissa pipeline 拿掉，softmax probability block 變成「top-k → pow2 shift → adder → LZD → pow2 shift」。在 fixed-point shift-add 實作下，該 block 不需要 general-purpose multiplier / divider / exp LUT / fraction path。
2. **為什麼選 `shift_topk_pow2` 而不是 `shift` 作為 primary 系列？** 兩者在 baseline 比較上都在噪聲層級內（shift_topk_pow2 -0.0013、shift +0.0021），但 fused 版**算術量降 ~8×**（4 entries vs 32）、輸出天然 sparse（外部 top-k mask + renorm 可省）、denom 也只看 top-4 sum。當 top-k=4 已是 strict pipeline 固定設定時，沒有理由保留 32-entry softmax。
3. **`shift` 保留作為 golden fallback 的理由？** 如果未來模型側決定改 K（如 K=8 或拿掉 top-k），primary 需要重新 eval；那時 `shift` 的 32-entry 路徑作為 reference 仍有用。也是 ISCAS 2023 那篇 baseline 的直接對應。
4. **為什麼選 `shift_topk_pow2*` 而不是 `relu2` 作為 primary？** Primary MBSTOI 距 baseline 僅 -0.0013，比 relu2 的 -0.0045 更好；且 relu2 需要 **true reciprocal/divider**（Newton-Raphson 或 LUT），primary softmax probability block 可用 compare / add / shift / LZD 組成，面積更小。
5. **為什麼 denom rounding 不能用 floor 或 ceil？** §3.4 sweep 顯示 floor (MBSTOI -0.0102) 與 ceil (-0.0216) 都顯著退化。Floor 讓 denom_pow2 < 真實 denom → row-sum > 1（distribution sharper），ceil 反之 (smoother)，兩者都偏離網路訓練時習慣的 softmax sum=1 分布。`nearest` 是唯一可用的 rounding policy。
6. **為什麼不投入 taylor2 finetune？** taylor2 eval-time 完全壞掉（MBSTOI -0.0603，ILD/IPD 比 noisy 還差），即使 30-epoch finetune 救起來頂多跟 shift 持平，硬體上又沒比 primary 更省，投入產出比低。
7. **為什麼不做 Softermax / SoftEx？** 它們保留 exp（只是 base-2 + low-precision LUT），跟「exp-free」目標衝突。如果未來決定接受 LUT，再回頭評估。
8. **fused top-k 為什麼可行？** shift weight transform `w = (1+f) × 2^k` 在 `s ∈ [-16, 0]` 區間 monotonically increasing，所以 score-domain 的 top-k 與 weight-domain 的 top-k **集合相同**。因此先 top-k 不損失 top-K 那 4 個位置的選擇，只損失了 denom 用最多 32 個 entries 還是 4 個 entries 的差別 — 而我們已用 pow2 round 把 denom 進一步離散化。`shift_topk_exact` 與 `shift + post-topk` 等價（誤差 2.4e-7）即是此性質的數值佐證。
9. **TIMIT cross-domain 為什麼是必做？** Primary 系列改變 attention distribution，最怕在 unmatched 雜訊或 speaker 條件下 binaural cue 失準。§3.5 已在 conservative `shift_topk_pow2` 上跑了 750-pair × 8 SNR，improvement +0.0616 甚至比 VCTK 的 +0.0499 大，ILD/IPD 也都改善。final minimal-area `shift_topk_pow2_clamp8_nomantissa` 還需要補一次 TIMIT sign-off 才能把同樣 claim 套到最終 RTL 候選。
