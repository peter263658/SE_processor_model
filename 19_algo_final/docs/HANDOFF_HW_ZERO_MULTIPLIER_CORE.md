# HANDOFF — Zero-Multiplier Network Inference Core (hardware mapping)

Target: BCCTN binaural speech-enhancement inference core (encoder convs +
bottleneck MHA + decoder convs + BN + activations). This document specifies, op
by op, the arithmetic the hardware must implement and how each maps to
shift / add / mux + small constant tables. **No general multiplier is required
anywhere in the neural core.**

Out of scope (DSP front/back-end, separate units, still use multipliers): see §6.

---

## 0. Claim & convention

- **Claim**: every multiply inside the neural inference core is replaced by
  shift(s) + add/sub + sign-mux, plus load-time-precomputed constants. No
  runtime general multiplier, no divider, no `exp` LUT.
- Weight-scale granularity is **layer-type specific** (not one global rule):
  Conv pointwise = per-group of 32 input channels; Conv depthwise = per-output
  channel; MHA Q/K/V/O projection = per-head. See §1 / §3.1.
- **Block-float activation boundary**: activations enter every Conv/Linear/MHA
  arithmetic block as **e4m3 raw code + one per-tensor pow2 shared exponent**
  (block floating-point). The shared exponent `k` is an offline calibration
  constant (`k = floor(log2(448/amax))`); **there is no runtime amax/scale/descale**.
  The activation's descale `2^-k` is fused into the relevant post-accumulation
  descale as a pure **exponent offset**. For ordinary real-valued blocks this is
  one `k` at the layer input. For complex conv/projection blocks, the same real
  sub-conv weight can be applied to both real and imag activation slices; when
  `k_real != k_imag`, hardware must select the per-branch fused descale
  (`k_real` or `k_imag`) for that MAC branch (§3.1a). Because `k` is a single
  power of two, `LMul(code·2^k, w) == 2^k·LMul(code, w)` exactly, so this fusion
  is bit-exact (eager==fused, verified relL2 = 0) — the software MBSTOI equals
  the deployed hardware number. BN and
  PReLU (§3.5–3.6) compute a folded affine in wider precision; their outputs are
  **re-quantized to e4m3 (block-float) at the next arithmetic-block input**.
  *Exception*: MHA internal `out_proj` consumes the fp16 Score×V accumulator
  directly (W4A16, INT4 shift-add) — no block-float exponent there (§2, §4).
- All per-channel / per-group constants (§5) are computed once from the trained
  checkpoint and loaded into the chip; runtime datapath never multiplies.

---

## 1. Number formats

| Tensor | Format | Notes |
|---|---|---|
| Conv/Linear/MHA-proj weights | INT4 symmetric, code `q ∈ [-7,7]` | 4-bit signed, qmax=7 |
| Activations into Conv/Linear/MHA blocks | **block-float**: e4m3 (1-4-3) raw code + per-tensor pow2 shared exponent `k` | static calibrated `k=floor(log2 448/amax)`; no runtime scale; descale `2^-k` fused into post-acc weight descale (§0, §5) |
| Decoder PW-outlier weights (12 layers) | FP8 e4m3, **per-tensor scaled** (`code=fp8(w·448/amax)`, descale `amax/448`) | kept FP8, not INT4; see §3.4 |
| Conv **pointwise** weight scale | APoT `2^a ± 2^b`, **per-group of 32**, `a,b ∈ [-16,-1]` | 2 shift + 1 add |
| Conv **depthwise** weight scale | APoT, **per-output-channel**, `a,b ∈ [-16,-1]` | 2 shift + 1 add |
| **MHA Q/K/V/O proj** weight scale | APoT, **per-head**, `a,b ∈ [-16,-1]` | 2 shift + 1 add |
| BN folded scale | APoT `2^a ± 2^b`, **per-channel**, `a,b ∈ [-9,4]` | 2 shift + 1 add |
| PReLU slope | pow2 `±2^k`, **per-tensor (1 scalar each for real & imag)**, `k ∈ [-8,0]` | 1 shift + sign |
| Conv MAC accumulator | wide FP (≥fp16) | rounding after group-partial + APoT scale must be specified (§3.1) |
| Attention datapath accumulator | fp16 | |

---

## 2. Op → hardware summary

| Core op | Arithmetic | Hardware |
|---|---|---|
| Conv / Linear (PW per-group 32, DW per-channel) | INT4 `q` × block-float e4m3 act + APoT scale | shift-add MAC + delayed fused APoT/pow2 scale |
| MHA in_proj / external complex projections | INT4 `q` × block-float e4m3 act + **per-head** APoT scale | shift-add MAC + delayed scale; complex projections select real/imag branch exponent |
| MHA internal out_proj | INT4 `q` × fp16/wide Score×V accumulator + **per-head** APoT scale | W4A16 shift-add MAC exception; still no multiplier |
| Attention QK^T | L-Mul (m=8) | exp-add + mantissa/sig-code add (logical 8b; implement as 10b/16b slice) + offset |
| Attention softmax | top-4, exp→pow2 (no mantissa), pow2 denom | shift + shift-divide, no exp LUT / divider |
| Attention Score×V | L-Mul (m=8) over top-4 packet | exp-add + mantissa/sig-code add + gather |
| Decoder PW outlier (12 FP8 layers) | FP8 × FP8 L-Mul (m=4) | exp-add + mantissa/sig-code add (logical 4b; 6b fits one 8b slice) |
| BatchNorm | folded per-channel affine `a_q·x + b` | 2-shift+add (APoT) + add |
| PReLU | `x≥0 ? x : (±2^k)·x` | sign mux + 1 shift |

---

## 3. Op detail

### 3.1 Conv / Linear / MHA projection — W4A8 block-float shift-add MAC + delayed APoT scale

**Conv-PW / grouped-Linear case** (scale group = 32 input channels):

```
partial = Σ_{i in group}  q_i · code_i
out     = Σ_groups  (APoT_scale_g · 2^-k_branch) · partial_g
```

Here `code_i` is the raw e4m3 activation value with its block-float exponent
removed. `k_branch = k` for ordinary real-valued blocks. For complex blocks,
`k_branch` is selected from the real/imag activation slice used by that real MAC
branch (§3.1a).

The scale-group boundary is **layer-type specific** (same datapath, different
constant granularity): **Conv-DW** = one scale per output channel (no input
grouping); **MHA Q/K/V/O projection** = one scale **per head** (`APoT_scale_head`
applied to that head's accumulation). See §1 / the bullets below.

- `q_i · code_i`: `q_i` is a 4-bit signed integer (|q|≤7 ⇒ ≤3 significant bits).
  The product is `q_i` shifted-adds of the raw e4m3 activation value `code_i`
  (CSD form ⇒ ≤2 add/sub). No multiplier.
- Accumulate the shift-add products in a wide FP accumulator.
- **Delayed scale**: after a scale group's accumulations, scale the partial sum
  by `APoT_scale · 2^-k_branch`. The pow2 activation factor is an exponent offset;
  the APoT part is `2^a ± 2^b` (2 barrel shifts + 1 add/sub), then accumulate
  across groups. The scale factors out of the sum, so scale-after-accumulate is
  **algebraically equivalent** to dequantize-weight-before — **bit-identical
  only under exact/wide accumulation**. The hardware MUST define the rounding
  points (after each group-partial, and after the APoT scale); FP16/narrow
  accumulation in a different order can differ.
- **Scale granularity is layer-type specific** (what the chip stores): Conv
  pointwise = one APoT code per group of 32 input channels; Conv depthwise =
  per-output-channel; **MHA Q/K/V/O projection = per-head** (one code per head).
  All use `a,b ∈ [-16,-1]`.
- **APoT canonical form**: the second term `±2^b` is *optional* — present only
  when `b < a` (else 0). The weight-scale magnitude is always < 1 so the overall
  scale is positive; only the **residual (second-term) sign** is stored. (BN,
  §3.5, can need an overall sign too.)
- L and R channels share the same weights (single WBUF copy).

Config trace: `int4_scale_format=apot_clip`, `int4_group_size=32`,
`BCCTN_APOT_{A,B}_{MIN,MAX}=[-16,-1]`, `qat_scope=full`.

### 3.1a Complex conv/projection branch scheduling — per-branch activation exponent

Complex conv/projection uses four real branch MACs. Source-level convention:

```text
apply_complex(fr, fi, x)
y_real = fr(x_real) - fi(x_imag)
y_imag = fr(x_imag) + fi(x_real)
```

Therefore the same real sub-conv weight (`fr` or `fi`) can be reused with both
activation branches. Hardware must not assume one fused activation exponent per
weight when `k_real != k_imag`. Schedule branch scales as:

```text
fr with x_real: APoT(fr) · 2^-k_real
fr with x_imag: APoT(fr) · 2^-k_imag
fi with x_imag: APoT(fi) · 2^-k_imag
fi with x_real: APoT(fi) · 2^-k_real
```

Decoder split-pointwise stages have separate `x` and `skip` inputs (`groups_x=4`,
`groups_skip=2`). Compute the x-path complex pointwise and skip-path complex
pointwise independently, add `point_r/point_i`, then feed the depthwise transpose
stage. If the real/imag activation exponents differ, select the matching fused
descale variant per branch. TB/export artifacts encode this explicitly as
`*_fused_act_kreal` and `*_fused_act_kimag`; inspect
[`fused_descale_summary.json`](../hw_patterns_final/tb_pattern/weights/fused_descale_summary.json).
`branches_collapse=true` means the two variants are numerically identical and HW
may share the constant; `branches_collapse=false` means they are distinct. Current
known non-collapsed case: decoder stage 3 (`k_real=5`, `k_imag=6`).

`mattn.input_proj` and `mattn.output_proj` are also complex projections. Keep
separate real/imag activation exponents there; current TB export marks these MHA
complex projection entries as not fused into a single per-weight descale.

### 3.2 Attention QK^T and Score×V — L-Mul (m=8)

L-Mul approximates a float multiply `x·y` without a multiplier:

```
sign(out) = sign(x) xor sign(y)
exp(out)  = exp(x) + exp(y) + offset(m)        # integer add + constant
mant(out) = mant(x) + mant(y)                  # m=8-bit mantissa add (+ carry into exp)
```

- m = 8 mantissa bits. Hardware = 1 exponent adder + 1 (8-bit) mantissa adder +
  constant offset; no multiplier.
- QK scale: `qk_scale_mode=pow2`; `1/sqrt(head_dim)=1/4=2^-2` for `head_dim=16`.
- Block-float exponent placement for QK must happen before softmax because it
  changes the effective temperature:

  ```text
  q_eff     = q_code · 2^-k_q
  k_eff     = k_code · 2^-k_k
  score_eff = LMulAcc(q_code, k_code) · 2^-(k_q + k_k) · 2^-2
  ```

  The `2^-(k_q+k_k+2)` factor is a post-QK-accumulator exponent/shift before the
  softmax. Do not defer it until after softmax.
- Score×V runs over the **top-4 sparse packet** (§3.3): gather the 4 selected V
  rows, L-Mul each by its prob, accumulate. With probability packet
  `p_eff = p_code · 2^-8` and block-float V `v_eff = v_code · 2^-k_v`:

  ```text
  attn_out_eff = LMulAcc(p_code, v_code) · 2^-(8 + k_v)
  ```

  Apply this before the internal MHA `out_proj`. That `out_proj` is the W4A16
  exception: it consumes the attention accumulator value, not a re-quantized A8
  block-float boundary.
- Accumulate QK and Score×V in fp16.

Config trace: `attention_datapath.lmul_enabled=true`, `lmul_targets=[qk]`,
`prob_packet.score_v_mode=lmul`, `lmul_mantissa_bits=8`, `lmul_bit_exact=true`,
`fp16_accumulator=true`.

### 3.2a L-Mul exact integer datapath (RTL reference)

This is the bit-exact arithmetic the deployed core uses (source:
`DCNN/utils/complexPyTorch/complexLayers.py::_lmul_multiply_bit_exact`, enabled by
`lmul_bit_exact=true`). It is the contract for the m=4 (decoder outlier, §3.4) and
m=8 (attention QK / Score×V, §3.2) L-Mul units.

**Per-operand decode.** For QK^T and the decoder outliers both operands are FP8
e4m3 (block-float raw code). For **Score×V the probability operand is the q0.8
packet code `c`** (8-bit unsigned, scale 256), *not* e4m3 — **do NOT convert it to
e4m3**. Unpack the code directly into the L-Mul operand fields (source feeds the
dequantized `c/256` into the same m8 L-Mul, `complexLayers.py::_q08_packet_lmul_matmul`):

```text
if c == 0: product = 0
n       = floor(log2 c)        # 8-bit CLZ / priority encoder, n ∈ 0..7
exp_p   = n - 8                # the /256 (q0.8 scale) folds into the exponent — no divider
frac8_p = (c - 2^n) << (8 - n) # normalize to the m8 fraction; 8-bit code ⇒ LOSSLESS
sign_p  = 0
```
This is exactly `frexp(c/256)`. V is the e4m3/block-float operand. Then the **same
m8 mantissa adder** runs: `sig_code = 256 + frac8_p + frac8_v + 16`, carry-renorm,
`exp_out = exp_p + exp_v + carry − k_v`. The only added hardware is the q0.8 unpack
frontend (CLZ + shift + zero-detect); there is **no separate scaler** — both the
`/256` and the V descale `2^-k_v` are exponent offsets in the L-Mul exponent path.

**General operand decode (e4m3 operands — QK, decoder outliers, and V above):**

```text
x = (-1)^sx · 2^ex · (1 + fx)        # ex = unbiased exponent, fx ∈ [0,1) is the fraction
fx_code = round_or_floor(fx · 2^m)   # m-bit fraction code, integer in [0, 2^m - 1]
```
For an e4m3 operand the stored mantissa is 3 bits, so `fx_code = mantissa << (m-3)`
(`m≥3` ⇒ the e4m3 mantissa is carried **losslessly**; the low `m-3` bits are 0).

**The L-Mul "multiply" = ONE small integer add** (this replaces the mantissa product):

```text
sig_code = (1<<m) + fx_code + fy_code + offset_code         # integer add
           # offset_code = round(2^m · 2^-l(m));  l(4)=3 ⇒ offset_code=2
           #                                       l(8)=4 ⇒ offset_code=16
carry    = sig_code >= (1<<(m+1))                            # 1-bit compare
sig_code = carry ? (sig_code >> 1) : sig_code                # renormalize (drop LSB)
exp_out  = ex + ey + carry + Δexp_fused                      # integer add
sign_out = sx XOR sy
product  = (-1)^sign_out · 2^exp_out · (sig_code / 2^m)      # significand = 1.f, m-bit
```

- `Δexp_fused` folds in the block-float activation descale `2^-k` (§0) and any
  pow2 part of the weight descale — pure additive **exponent offset**, no multiply.
- **Operand-add width** (the bit budget the slice must hold), since
  `(1<<m) + fx + fy + offset ≤ 3·2^m`:
  - **m=4**: `sig_code ∈ [18, 48]` ⇒ **6-bit unsigned add** (the "m4 6-bit" budget).
  - **m=8**: `sig_code ∈ [272, 782]` ⇒ **10-bit unsigned add**.
- **Hardware per L-Mul**: 1 sign XOR + 1 exponent adder + 1 `(m+2)`-bit mantissa
  adder + 1 carry compare + conditional 1-bit shift/increment. **No multiplier, no
  LUT.** The `product` then enters the fp16 accumulator (exponent-align + add =
  ordinary FP accumulate; that aligner/adder is the only wide arithmetic and it is
  an *adder*, not a multiplier).
- L-Mul is an **approximate** multiply: it drops the `fx·fy` cross-term and adds a
  fixed `offset_code` to re-center the bias. The deploy model was trained under
  exactly this arithmetic (`lmul_bit_exact=true`), so it is the contract, not an
  approximation to "fix up".

### 3.3 Softmax — `shift_topk_pow2_clamp8_nomantissa` (probability block)

Per attention row (sequence position):

1. **Top-4 select** the 4 largest scores; the rest are dropped (0).
2. **exp → pow2, no mantissa**: `w = 2^clamp(floor(score'), -8, 0)`
   (pure power-of-two; the mantissa `1+frac` term is dropped). = 1 shift.
3. **Denominator → pow2**: `denom = round_pow2(Σ_top4 w)`; division by a
   power-of-two = 1 shift.
4. Output a **4-entry sparse probability packet**, quantized q0.8 (8-bit,
   scale 256), with a 5-bit local index per entry (≈52 bits/row vs 512 bits for
   a dense FP16 row).

No `exp` LUT, no divider, no general multiplier in the probability block.

Config trace: `attention_softmax.mode=shift_topk_pow2_clamp8_nomantissa`,
`prob_packet.{top_k=4, local_idx_bits=5, prob_bits=8, prob_scale=256}`.

### 3.4 Decoder pointwise outliers — FP8 × FP8 L-Mul (m=4)

The 12 outlier pointwise layers keep FP8 (not INT4) weights:
`decoder.model.{0,3,4}.0.pointwise_{r,i}_{x,skip}`. Multiply via L-Mul with
m=4 mantissa bits — sufficient to carry the FP8 e4m3 mantissa (3 bits) with no
extra mantissa-bit truncation, but **L-Mul is still an approximate FP8 multiply**
(the mantissa add replaces `(1+m_x)(1+m_y)`, dropping the cross-term). Same
exp-add + mantissa-add hardware as §3.2, narrower (4-bit) adder.

**Weight format = PER-TENSOR SCALED FP8 (not raw e4m3).** The trained model
quantizes these weights with a per-tensor dynamic scale, so the chip must store
the *scaled* code, not a plain `w.to(e4m3)` cast:

```
s      = 448.0 / amax(|w|)                 # per-tensor scalar (448 = e4m3 max)
code   = fp8_e4m3(w · s)                    # the e4m3 bits stored on chip
weight = code · (amax/448) = code / s       # == the validated module.weight (bit-exact)
```

Storing a plain `fp8_e4m3(w)` instead is **wrong** (it wastes the e4m3 range on
small-magnitude weights; measured ~4% relative output L2 error end-to-end). The
export now writes `w_fp8.e4m3.bin = fp8(w·s)`, the per-tensor `fp8_scale`
(`448/amax`) and `fp8_descale` (`amax/448`), plus an APoT-encoded descale.

**Descale and the zero-multiplier claim — RESOLVED (delayed APoT descale,
recovery-finetuned).** The per-tensor descale (`amax/448`) is a *single constant
per layer*, and it factors out of the per-element L-Mul:
`Σ (code_i / s)·a_i = (1/s)·Σ code_i·a_i`. So it is applied **once per output on
the fp16 accumulator**, not per MAC — the **delayed (post-accumulation) scale**,
exactly like the INT4 conv delayed APoT scale (§3.1). The deployed core realizes
it as a **2-term APoT (`2^a ± 2^b`) exponent offset / 2-shift+add folded into the
L-Mul output** (no multiplier), exported as `fp8_descale_apot_*`.

This APoT descale is an approximation of the exact `amax/448` (2-term APoT ⇒ up to
~9% per-layer error; folding it into the *deploy* weight one-shot cost ~4% output
L2 and dropped VCTK to 0.8335). It was therefore **closed by a 5-epoch recovery
finetune trained UNDER the delayed APoT descale** (`BCCTN_OUTLIER_DELAYED_DESCALE=1`):
the model adapts to the deployed arithmetic and recovers to **VCTK 0.8365 / TIMIT
0.7912** — above both gates and above the exact-descale deploy (0.8355 / 0.7902).
So the strict zero-multiplier core (descale = shift-add) is the deploy point; no
per-output multiply is needed. **APoT exponent range = `[-16, -6]`** (data-driven:
the 12 descale exponents live in `[-14,-8]`; 4-bit field with margin, no clamp).

**Current deploy = pow2 block-float activations (subsumes the above).** The
delayed-APoT weight descale is unchanged, but activations are now block-float
(§0/§1/§5): the activation descale `2^-k` is folded into *this same* post-acc
descale as an exponent offset. For ordinary branches the APoT a/b exponents shift
by the layer input `k`; for decoder split-pointwise complex branches, TB/export
emits `kreal/kimag` fused variants and HW selects the variant matching the
activation slice used by that branch (§3.1a). Since `k` is a single pow2,
`eager==fused` is bit-exact (verified relL2 = 0 on all 48 outlier L-Mul calls), so
the validated score is the HW number. A pow2 recovery FT gives **VCTK 0.8357 /
TIMIT 0.7898** (both gates), with **all runtime activation amax/scaling removed**.

Deploy ckpt: `ckpt/strict_core_pow2_blockfloat_deploy_ep3_vctk0.8357_timit0.7898.ckpt`
(activation calib table `ckpt/activation_calibration_p999_b64.json`; block-float
exponents exported under `export_out_strict18/activation_blockfloat/`).
Prior dynamic-activation deploy: `strict_core_apot_descale_deploy_ep2_vctk0.8365_timit0.7912.ckpt`.

Config trace: `low_power.outlier_fp8_lmul.enabled=true, mantissa_bits=4`;
source quantizer = `DCNN/utils/fp8_qat.py::FP8QuantizeSTE` (per-tensor amax scale);
delayed descale runtime = `all_lmul_multiply.py` (`BCCTN_OUTLIER_DELAYED_DESCALE`),
APoT range `BCCTN_OUTLIER_DESCALE_APOT_{MIN,MAX}=[-16,-6]`.

### 3.5 BatchNorm — folded per-channel affine (APoT scale + bias)

BN must NOT be folded into the complex conv (real path needs the weight in both
real and imag outputs ⇒ would double the conv weights). It is a **standalone
per-channel affine** instead. Real and imaginary parts are independent.

Precompute per channel (load time, from frozen running stats):

```
a   = gamma / sqrt(running_var + eps)
a_q = APoT(a),  a,b exponents ∈ [-9,4]
b   = beta - a_q · running_mean            # constant, folded with a_q
```

Runtime: `y = a_q · x + b` = (2 shifts + 1 add)·x + add. No multiplier.

- **Sign storage**: unlike the conv weight scale (always positive), `gamma` can
  be negative ⇒ the BN scale `a` can be negative. Store an **overall sign** plus
  the residual (second-term) sign, i.e. `a_q = sign·(2^a ± 2^b)`. Second term is
  optional (only when `b < a`).
- **Critical**: quantize the *effective* scale `a = gamma/sqrt(var+eps)`, NOT
  `gamma` alone — the `1/sqrt(var+eps)` factor would otherwise re-introduce a
  real multiply at inference.
- Exponent range `[-9,4]` is data-driven from the trained checkpoint: folded
  `|a| ∈ [0.0037, 14.8]`, `log2|a| ∈ [-8.07, +3.89]`, and ~34% of channels have
  `|a| ≥ 1` (positive exponents are required — do not reuse the conv weight-scale
  range `[-16,-1]`).

Config trace: `low_power.bn_affine.scale_mode=fold_apot`,
`pow2_min_exp=-9, pow2_max_exp=4, tie_real_imag_scale=false`.

### 3.6 PReLU — pow2 slope (sign mux + shift)

Per channel (real & imag independent):

```
y = x            if x >= 0
y = (±2^k) · x   if x <  0          # k ∈ [-8,0], sign stored per channel
```

Runtime = sign mux + 1 shift. No multiplier.

- Slope `α` quantized to `±2^k`, `k ∈ [-8,0]`. Data-driven: `|α| ∈ [0.006, 0.40]`
  (all < 1). The **sign is preserved** (some learned slopes are negative).

Config trace: `low_power.prelu_pow2.enabled=true, min_exp=-8, max_exp=0`.

---

## 4. Dataflow / memory (structural)

- **PE array**: 64 complex MAC lanes, 1-D flat.
- **PE arithmetic slice — one shared adder covers all 4 modes.** The PE core is a
  single configurable add/shift unit; the 4 arithmetic modes differ only in how
  the operands feed it and how wide the integer add is. Build it from a **6-bit
  base slice** (the minimum that holds the L-Mul m=4 mantissa sum, §3.2a) that can
  **chain** (carry between slices) into a 12-bit (2 slices) or 18-bit (3 slices)
  add. Two roles share the adder:
  - **INT4 weight modes (W4A8, W4A16)** — *significand shift-accumulate*. The INT4
    code `q` (|q|≤7, CSD ⇒ ≤2 signed digits, shift 0..3) gates shifted copies of
    the **activation significand**; the adder sums them. Activation significand =
    4 bits (e4m3, hidden-1+3) for W4A8, 11 bits (fp16, hidden-1+10) for W4A16. The
    weight exponent / APoT-pow2 / activation `2^-k` are applied later as the
    delayed scale (exponent offset).
  - **L-Mul modes (m4, m8)** — *mantissa add* (§3.2a): `sig_code = (1<<m) + fx +
    fy + offset`, one add, then carry-renorm.

  | mode | what the adder adds | integer width needed | 6-bit base slices |
  |---|---|---|---|
  | **L-Mul m4** (decoder outlier §3.4) | `(1<<4)+fx+fy+2`, ≤48 | **6 b** | **1** |
  | **W4A8** (INT4×e4m3 §3.1) | 4-b significand `<<`0..3, ±2 terms | **≈8 b** (7-b mag + sign) | 2 (chain) |
  | **L-Mul m8** (QK / Score×V §3.2) | `(1<<8)+fx+fy+16`, ≤782 | **10 b** | 2 (chain) |
  | **W4A16** (out_proj §3.2/§4) | 11-b significand `<<`0..3, ±2 terms | **≈15 b** | 3 (chain) |

  So with a 6-bit base: m4 = 1 slice; W4A8 and m8 = 2 chained; W4A16 = 3 chained.
  (Design note: an **8-bit** base slice instead lets **both** m4 *and* W4A8 fit in
  one slice and m8 + W4A16 in two — fewer chains, slightly more area per slice.
  Either base works; pick per area/timing budget. The deploy never needs a true
  multiplier in any mode.)
  - **MHA internal out_proj** is the only W4A16 user (input = on-chip fp16/wide
    Score×V accumulator, not an A8 boundary). Reuse the same chained adder; do
    **not** instantiate a separate W4A16 PE array.
- **L-Mul exponent path**: L-Mul is not only a mantissa adder. It also needs
  sign XOR, exponent add, mantissa carry adjustment, and a scale/exponent
  offset path. Because activation pow2 exponents and delayed scales are fused
  into this path, do not size it as a raw 4-bit FP8 exponent adder only; use a
  signed exponent-offset datapath (recommend >=7 bits) or the existing fp16
  exponent datapath before accumulation.
- **Weight reuse**: L and R channels share weights — single weight copy in WBUF.
- **Buffers**: WBUF = 32 KB, UFBUF = 4 KB. Feature maps stay fully on-chip.
- **MHA Q/K/V projection**: block-diagonal grouping `qkv_groups=4` — store and
  compute only the diagonal 64×64 blocks (off-diagonal weights do not exist).
  Required to meet the 64-lane @ 5 MHz real-time duty (attention MAC ≈ 1.10 M/frame,
  duty ≈ 0.81).
- **Decoder pointwise**: grouped `groups_x=4`, `groups_skip=2`; the skip input
  uses only the top-50% channels by L1 magnitude (fixed index set, loaded once).
  Real `Conv2d(groups=…)` + channel slice — both stored and computed weights are
  reduced. For split complex pointwise, schedule each real MAC branch with the
  matching `kreal/kimag` fused descale variant when real/imag activation exponents
  differ.
- **Attention probability transport**: top-4 sparse packet, 5-bit local index,
  q0.8 probability ⇒ ≈52 bits/row (vs 512 bits dense FP16).
- **Per-group weight scale**: group_size = 32 (one APoT code per 32 input
  channels).

---

## 5. Load-time constants (computed once from the checkpoint; no runtime multiply)

| Constant | Per | Format |
|---|---|---|
| INT4 weight codes `q` | weight element | 4-bit signed |
| **Activation block-float exponent `k`** | **activation boundary (per-tensor)** | signed int (`k=floor(log2 448/amax)`); descale `2^-k` fused into post-acc descale. Complex conv/projection branches may require separate `kreal/kimag` fused variants; `branches_collapse=true` means they are equal. Tables: `export_out_strict18/activation_blockfloat/`, `hw_patterns_final/tb_pattern/weights/fused_descale_summary.json` |
| Conv-PW weight APoT scale | group of 32 in-ch | (a-exp, b-exp, residual-sign), a,b∈[-16,-1] |
| Conv-DW weight APoT scale | output channel | (a-exp, b-exp, residual-sign), a,b∈[-16,-1] |
| MHA Q/K/V/O proj APoT scale | **head** | (a-exp, b-exp, residual-sign), a,b∈[-16,-1] |
| Decoder PW-outlier FP8 descale `amax/448` | **layer (per-tensor)** | exact fp32 `fp8_descale` + APoT (a,b∈[-16,-6], data-driven 4-bit); applied once on fp16 accumulator as delayed shift-add (§3.4) |
| BN folded scale `a_q` | channel (r & i) | APoT (a,b∈[-9,4]) + **overall sign** + residual-sign |
| BN folded bias `b` | channel (r & i) | additive constant |
| PReLU slope | per-tensor (1 scalar each for r & i) | (k-exp, sign), k∈[-8,0] |
| Decoder skip top-50% index set | layer | fixed channel indices |

---

## 6. Out of scope — DSP front/back-end (NOT part of the zero-mult core claim)

These still use multipliers and are separate units (run on the on-chip DSP
front/back-end, not the neural core):

- STFT / iSTFT.
- Analysis / synthesis window multiply.
- Waveform-domain mask application (`apply_mask`).

If an "end-to-end-to-waveform zero-multiplier" claim is later required, these
must be addressed separately; they are explicitly excluded here.

---

## 7. Per-op arithmetic checklist (sign-off)

- [ ] Activations: **block-float** e4m3 raw code + per-tensor pow2 exponent `k`
      (static calib, no runtime amax/scale). Descale `2^-k` is fused into the
      relevant post-accumulator descale. Complex conv/projection branches must
      select the matching `kreal/kimag` fused variant; `branches_collapse` is only
      a sharing optimization, not a scheduling assumption. Exponent table loaded
      from `activation_blockfloat/`. QK descale applied on QK accumulator BEFORE
      softmax.
- [ ] Conv/Linear/proj: INT4×block-float e4m3 raw-code shift-add + delayed
      APoT/pow2 scale — granularity: Conv-PW/Linear per-group(32), Conv-DW
      per-output-channel, MHA proj per-head.
- [ ] MHA structure (§8.2/§9.1): outer complex `mattn.input_proj` → **tied
      ComplexMHA = 2 real passes** (`a=M(emb.real)`, `b=M(emb.imag)`, shared
      weights) → combine `(a−b)+i(a+b)` → outer complex `mattn.output_proj`. Each
      pass keeps its **own** K/V history ⇒ **2× K/V ring (≈32 KB)**, not 1×.
- [ ] MHA internal out_proj: only W4A16 exception; reuse the same INT4 shift-add
      PE in chained-slice mode, input from Score×V fp16/wide accumulator.
- [ ] QK^T, Score×V: L-Mul m=8, fp16 accumulate. QK applies `2^-(k_q+k_k+2)` before
      softmax; Score×V applies `2^-(8+k_v)` before the internal W4A16 `out_proj`
      (Score×V prob operand is **q0.8 packet code**, not e4m3 — §3.2a).
- [ ] PE adder slices (§4): one shared add/shift unit from a **6-bit base slice**,
      chainable. L-Mul m4 = 1 slice (6 b); W4A8 = 2 (≈8 b); L-Mul m8 = 2 (10 b);
      W4A16 = 3 (≈15 b). (8-bit base is an alternative: m4+W4A8 each 1 slice,
      m8+W4A16 each 2 — pick per area/timing. Do **not** hard-fix "2×8-bit".)
- [ ] L-Mul exponent path: sign XOR + exponent add + mantissa carry + signed
      scale/exponent offset; do not size this as only a 4-bit FP8 exponent add.
- [ ] Softmax: top-4, 2^⌊·⌋ (no mantissa, clamp exp), pow2 denominator. No exp LUT / divider.
- [ ] Decoder PW outliers (12): per-tensor scaled FP8 (`code=fp8(w·448/amax)`, descale
      `amax/448`), FP8×FP8 L-Mul m=4; descale folded into L-Mul exponent (pow2/APoT) — §3.4.
- [ ] BN: folded `a_q·x + b`, `a_q` = APoT of `gamma/sqrt(var+eps)`, exp∈[-9,4].
- [ ] PReLU: pow2 slope, sign mux + shift, exp∈[-8,0].
- [ ] No general multiplier, no divider, no exp LUT in the neural core.

---

## 8. Per-layer operation & dimension map (frame-by-frame)

**Frame-by-frame convention.** The core processes **one STFT frame per pass**
(hop = `win_inc=80` samples ⇒ a new frame every 5 ms @ 16 kHz). The conv kernel is
`(5,1)` = **5-tap in frequency, 1 in time**, so every conv mixes only frequency and
each frame is **independent** in the conv stages; the captured tensors confirm the
time axis `T` is unchanged through all convs. The **only** cross-frame dependency is
the MHA, which attends over a causal `attn_win=31` window ⇒ a 31-frame K/V history
(§9). All feature maps below are per-frame `[C, F]` (complex: a real and an imag
plane); `MAC/frame` counts real multiply-equivalents = `Σ Cout·Cin_per_group·F_out`
× **4** for a complex conv (`fr·xr, fi·xi, fr·xi, fi·xr`, §3.1a).

**Binaural.** STFT → spectrogram per ear (L, R). Encoder and decoder run **per ear
with shared weights**; the two ears are **concatenated only at the bottleneck**
(`flatten C·F=128 per ear, stack L‖R = 256 = embed_dim`), so the MHA jointly attends
over both ears, then splits back. `head_dim = 256/16 = 16`.

### 8.1 Encoder (per ear; freq-only conv `k=(5,1)`, stride `(2,1)`, BN+PReLU after each stage)

| stage | depthwise (per-out-ch INT4, **W4A8**) | pointwise 1×1 (per-group-32 INT4, **W4A8**) |
|---|---|---|
| enc0 | `[1,64]→[1,32]` | `[1,32]→[8,32]` |
| enc1 | `[8,32]→[8,16]` | `[8,16]→[16,16]` |
| enc2 | `[16,16]→[16,8]` | `[16,8]→[32,8]` |
| enc3 | `[32,8]→[32,4]` | `[32,4]→[64,4]` |
| enc4 | `[64,4]→[64,2]` | `[64,2]→[128,2]` |
| enc5 | `[128,2]→[128,1]` | `[128,1]→[128,1]` |

Channels are **per real/imag path** (the complex channel count `kernel_num` =
`[16,32,64,128,256,256]` is 2× these). Each stage's output (top-50% channels by L1)
is held for the mirror decoder skip (§9). Freq: `64→32→16→8→4→2→1`.

### 8.2 Bottleneck MHA (one frame = one query; joint L‖R; `embed=256, heads=16, head_dim=16`)

**Two nesting levels — do not conflate (source: `model.py::MultiAttnBlock.forward`,
`complexLayers.py::_forward_tied`):**

1. **Outer complex projections** `mattn.input_proj` then (after attention)
   `mattn.output_proj` — complex `GroupedLinear` (`fc_r/fc_i`), `256→256`, **W4A8**,
   per-head APoT. They wrap the attention and **each run once per frame** (but are
   complex ⇒ §3.1a per-branch real/imag exponent; exported **not fused**).
2. **Tied ComplexMHA** (`self.mattn`, `MHA_re ≡ MHA_im = M`) — runs the internal
   real MHA **TWICE with shared weights**: `a = M(emb.real)`, `b = M(emb.imag)`,
   then combine `out = (a−b) + i(a+b)`. So the internal MHA is **2 real passes**,
   each with its own Q/K/V and its **own K/V history** (§9.3 ⇒ 2× ring).

Per **internal pass** (`_mha_attention_forward`, causal window `attn_win=31`):

| op | dim (per frame, per pass) | grouping | MAC mode | weight scale |
|---|---|---|---|---|
| internal **in_proj** Q,K,V | `256→256` ×3 | **qkv_groups=4** block-diagonal: 4 indep `64→64` blocks per Q/K/V (off-diag weights do not exist; −75% vs dense) | W4A8 | per-head APoT |
| QK^T | per head `q[16]·K_win[31,16] → 31` scores | 16 heads; causal window 31 | **L-Mul m8** | scale `2^-(k_q+k_k+2)` pre-softmax (§3.2) |
| softmax | top-4 of 31 per head → q0.8 packet | per head | shift/pow2 (§3.3) | — |
| Score×V | per head `top4 · V[16] → out[16]` (prob = q0.8) | 16 heads, top-4 gather | **L-Mul m8** | scale `2^-(8+k_v)` |
| internal **out_proj** | `256→256` | dense (`out_proj_groups=1`) | **W4A16** (input = fp16 Score×V accumulator) | per-head APoT |

Full per-frame MHA block = `input_proj` (once) → [internal MHA ×2 passes, shared
weights] → combine `(a−b)+i(a+b)` → `output_proj` (once). With `qkv_groups=4` each
in_proj block (64 wide) spans 4 heads.

### 8.3 Decoder (per ear; pointwise then transpose-conv; freq up `×2` per stage)

Each stage: `point = pointwise_x(x, groups_x=4) + pointwise_skip(skip_subset, groups_skip=2)`,
then `depthwise_transpose` (INT4 **W4A8**, per-out-ch, upsamples freq ×2), then BN+PReLU.
Stored pointwise weight shape = `[Cout, Cin/groups]`. **Skip** = top-50%-by-L1 subset
of the mirror encoder stage (fixed index set, loaded once).

| stage | pointwise_x (`groups_x=4`) | pointwise_skip (`groups_skip=2`, top-50%) | dw-transpose | pointwise MAC mode |
|---|---|---|---|---|
| dec0 | `[128,1]→[128,1]` | skip`[64,1]→[128,1]` | `[128,1]→[128,2]` | **L-Mul m4** (outlier) |
| dec1 | `[128,2]→[64,2]` | skip`[64,2]→[64,2]` | `[64,2]→[64,4]` | W4A8 |
| dec2 | `[64,4]→[32,4]` | skip`[32,4]→[32,4]` | `[32,4]→[32,8]` | W4A8 |
| dec3 | `[32,8]→[16,8]` | skip`[16,8]→[16,8]` | `[16,8]→[16,16]` | **L-Mul m4** (outlier) |
| dec4 | `[16,16]→[8,16]` | skip`[8,16]→[8,16]` | `[8,16]→[8,32]` | **L-Mul m4** (outlier) |
| dec5 | `[8,32]→[1,32]` | skip`[4,32]→[1,32]` | `[1,32]→[1,64]` | W4A8 |

The 12 **outlier** pointwise convs are decoder stages **0, 3, 4** (`pointwise_{r,i}_{x,skip}`),
FP8×FP8 **L-Mul m4** with the per-tensor delayed APoT descale (§3.4); they are also the
per-branch `kreal/kimag` case (§3.1a, stage-3 `k_real=5,k_imag=6`). Decoder output =
`[1,64]` complex per ear = the complex ratio **mask**.

### 8.4 Front/back-end (DSP, §6 — not zero-mult core)

STFT (per ear, 1 frame → 64-bin complex), complex-mask multiply (`mask · spectrogram`),
iSTFT + overlap-add → enhanced waveform.

---

## 9. Controller scheduling & data loading (per frame)

The controller runs one **frame pipeline** per hop. Weights do not all fit in WBUF
(32 KB) at once — INT4 RTL weights ≈ 169 KB total — so **weights stream from DRAM
into WBUF layer-by-layer** (the W4 packing vs FP8 is what makes this DMA fit the
duty; see CLAUDE.md W4A8 analysis). Feature maps and the skip subset stay on-chip
(UFBUF); the attention K/V history is a dedicated ring buffer.

### 9.1 Per-frame execution order (one hop)

```text
for each new STFT frame t (per ear e ∈ {L,R} unless noted):
  S0  DSP front-end:   x_spec[e] = STFT(frame_t[e])             # [1,64] complex  (§6)
  S1  Encoder enc0..enc5 (shared weights, freq-only):
        for s in 0..5:
          load Wdw[s], Wpw[s], BN[s], PReLU[s] from DRAM→WBUF   # stream per stage
          h = PReLU( BN( pw( dw(h) ) ) )                        # [C_s, F_s]
          UFBUF.store_skip(s, top50%_index[s] applied to h)     # for decoder skip
  S2  Bottleneck pack:  v_e = flatten(h_e)            # 128 per ear
  --- once both ears done ---
  S3  emb = concat(v_L, v_R)                          # 256 = embed_dim
  S4  MHA block (model.py MultiAttnBlock + tied ComplexMHA):
        load W_input_proj (per-head APoT, complex fc_r/fc_i)
        emb = mattn.input_proj(emb)                   # OUTER complex proj 256→256, W4A8 (once)
        # tied ComplexMHA = 2 real-MHA passes, SHARED weights M, then combine:
        for plane p in {emb.real → a, emb.imag → b}:  # 2 passes, each own K/V history
          load W_in_proj (4 diag 64×64 blocks/Q,K,V), W_out_proj  # shared across both passes
          Q,K,V = in_proj(p)                          # W4A8, qkv_groups=4
          KVring[p].push(K_t, V_t); evict frame t-31  # 31-frame causal window (per plane!)
          for head h in 0..15:
            score = LMul_m8(Q_h, KVring[p].K[:,h]) · 2^-(k_q+k_k+2)   # 31 scores
            prob  = softmax_top4_pow2(score)                          # 4-entry q0.8 packet
            o_h   = LMul_m8(prob, KVring[p].V[top4,h]) · 2^-(8+k_v)    # prob=q0.8, [16]
          pass_out = out_proj(concat o_h)             # W4A16 (fp16 accumulator input)
        mha = (a - b) + i(a + b)                       # complex combine
        load W_output_proj (per-head APoT, complex)
        attn = mattn.output_proj(mha)                 # OUTER complex proj 256→256, W4A8 (once)
        v_L, v_R = split(attn)
  S5  Decoder dec0..dec5 (shared weights, per ear):
        for s in 0..5:
          load Wpw_x[s], Wpw_skip[s], Wdw_T[s], BN, PReLU, skip_index[s]
          skip = UFBUF.load_skip(5-s)                  # mirror encoder stage (same frame t)
          point = pw_x(x, g=4) + pw_skip(skip, g=2)    # complex; per-branch kreal/kimag
          x = PReLU( BN( dw_transpose(point) ) )       # freq ×2
        mask[e] = x                                    # [1,64] complex
  S6  DSP back-end:  y[e] = iSTFT( mask[e] · x_spec[e] )  + overlap-add   # (§6)
```

### 9.2 What the controller loads, per layer

| layer class | weight load (DRAM→WBUF) | scale/const load | activation source |
|---|---|---|---|
| encoder dw/pw, decoder dw-T, MHA **internal** in_proj | INT4 codes (4-bit packed) | APoT (a/b-exp, residual-sign) per group/ch/head + **fused** `−k` exponent (§5) | UFBUF (current frame); raw e4m3 code |
| MHA **outer** `mattn.input_proj` / `mattn.output_proj` (complex `fc_r/fc_i`) | INT4 codes | per-head APoT + per-branch `kreal/kimag` (§3.1a, not fused) | UFBUF; once each per frame |
| MHA internal in_proj / out_proj (**shared across both tied passes**) | load once, reuse for real & imag pass | per-head APoT (in_proj); out_proj W4A16 | UFBUF; K/V ring per plane |
| decoder pw stages 0/3/4 (outlier) | e4m3 code (`fp8(w·s)`) | `fp8_descale_apot_*` **per branch** `_fused_act_kreal/kimag` (§3.1a/§5) | UFBUF cat(x, skip-subset) |
| decoder pw stages 1/2/5 | INT4 codes | per-group APoT + fused `−k` | UFBUF cat(x, skip-subset) |
| MHA QK / Score×V | — (activations only) | QK `2^-(k_q+k_k+2)`, Score×V `2^-(8+k_v)`; softmax pow2 consts | Q/K/V regs + **K/V ring** |
| BN / PReLU | folded `a_q`+bias / pow2 slope+sign | §5 tables | inline after conv |

### 9.3 Buffers & sizing notes

- **WBUF (32 KB)**: holds the current layer's weights; double-buffer (load layer
  `s+1` while computing `s`) to hide DRAM latency. Largest single weight tensor:
  MHA out_proj 256×256 INT4 = 32 KB (fills WBUF) — schedule it without a co-resident
  tensor, or tile by head.
- **UFBUF (4 KB)**: current-frame activations (largest `[8,64]` cmplx ≈ 2 KB/ear)
  **plus** the held encoder skip subsets (top-50%); reused as the frame advances.
  Skip is the **same frame** (conv is freq-only) ⇒ no cross-frame skip buffer.
- **K/V ring (separate, ≈32 KB)**: the tied ComplexMHA runs **2 passes** (real
  plane, imag plane) with their **own** K/V histories, so the ring holds K and V
  for **both** planes: `31 × 256 × 2(K,V) × 2(real,imag) × 1 B (e4m3 code) + shared
  exponents ≈ 32 KB`. Circular per plane: push current frame's K,V, evict frame
  `t-31`. This is the only state that persists across frames; size it explicitly
  (not part of UFBUF). (A 16 KB ring only suffices if the imag-pass history is
  recomputed each frame — not the default.)
- **Skip index ROM**: the top-50% channel index set per decoder stage (fixed, §5),
  loaded once at boot.
- **L/R**: encoder/decoder run twice (shared weights resident across both ears, so
  weight DMA is amortized over 2 ears). The two ears are **concatenated once**
  into the 256-wide bottleneck; the tied ComplexMHA then still runs its **2 real
  passes** (real/imag plane, §8.2) on that 256 — "joint L‖R" refers to the concat,
  not to a single attention pass.

### 9.4 Real-time budget

64 complex MAC lanes; attention MAC ≈ 1.10 M/frame; at 5 MHz lane clock the duty is
≈ 0.81 (headroom for the conv stages + DSP). The `qkv_groups=4` block-diagonal
projection (−75% MHA-proj MAC) and decoder grouping/skip-subset are what bring the
per-frame MAC under the real-time budget — keep them in the schedule (do not expand
to dense).

### 9.5 Post-accumulation scaling / BN / PReLU — one shared PPU pool

All post-MAC fix-ups are the **same shift-add / APoT / pow2 primitive** and are
**sequential on one datum but pipelined across data**:

```text
PE reduce → delayed descale (per scale-group; fused activation 2^-k, §0/§3.1)
          → cross-group sum
          → BN APoT affine + bias (per output, §3.5)
          → PReLU pow2 (per output, §3.6)
          → requant to e4m3 block-float (next layer input)
```

They never contend for the same datum at once (element i in PReLU while i+1 in BN
while i+2 in descale); layer N's BN/PReLU hide behind layer N+1's MAC+descale.

**Share them in ONE PPU pool — do not split into dedicated descale/BN/PReLU lanes.**
MHA is ~78% of compute and has **no BN/PReLU**; dedicated BN/PReLU lanes would idle
through all of attention. A shared pool does projection / QK / Score×V descales
during MHA and descale+BN+PReLU during conv ⇒ near-full PPU utilization.

**Sizing** — `PPU_lanes ≈ PE × (PPU_ops/frame) / (MAC_ops/frame)`. Reviewed per-frame load:

| PPU op class | ops/frame | reduction / scale |
|---|---:|---|
| conv/decoder delayed descale + 4 MHA projection descales | ≈ 48 K | 16 – 256 |
| BN APoT affine (10,752 elems; `decoder.model.5.0` has none) | ≈ 21.5 K | per-output |
| PReLU pow2 (10,752 elems) | ≈ 10.8 K | per-output |
| **total shared PPU load** | **≈ 80 K** | |

> **QK `2^-(k_q+k_k+2)` and Score×V `2^-(8+k_v)` are NOT PPU ops.** They are
> exponent offsets folded into the **L-Mul exponent adder inside the PE** (§3.2/§3.2a;
> for Score×V the q0.8 `/256` also folds there). They cost nothing extra and do not
> consume PPU lanes — the shared pool is only conv/decoder descale + MHA-projection
> descale + BN + PReLU.

- **real multiply-equivalent MAC ≈ 1.40 M/frame** (complex = 3 real mults, 2 ears)
  ⇒ `128 × 80K/1.40M ≈ 7.3` ⇒ **8 PPU lanes**.
- If sized against the doc's **normalized 1.10 M MAC/frame**, or if small-layer
  throttling is not allowed ⇒ ≈ 9.4 ⇒ **10–12 PPU lanes**.

**Recommendation: 8 shared PPU lanes (baseline); 10–12 if conservative.** Each lane
is a tiny shift-add (2 shift + 1–2 add + sign-mux); 8–12 is negligible vs 128 PE.
(MAC basis note: the source `apply_complex` is a naïve 4-mult complex op; the MAC
numbers here use the 3-mult complex-equivalent to stay consistent with the §-Headline
1.10 M / "MAC-equivalent" convention. The PPU **ratio**, and hence ~8 lanes, is
robust to the ×3-vs-×4 choice.)

**Do NOT provision for small-layer worst cases** (throttle PE width or FIFO-drain
instead):

| small layer | reduction / scale | naïve lanes @128 PE | why not provision |
|---|---:|---:|---|
| enc0 pointwise | 1 (`Cin=1`) | 128 | `Cout=8` — can't fill the PE array anyway |
| depthwise 5-tap (per-ch) | 5 | ~26 | low channel count, DMA-bound |
| ConvTranspose / deconv | ≈ 2.5 (stride-2 input share) | ~51 | small decoder layer; not the bottleneck |
| Score×V scale `2^-(8+k_v)` | — | **n/a (in PE)** | exponent offset in the L-Mul adder, not a PPU lane (§3.2a) |
| QK scale `2^-(k_q+k_k+2)` | — | **n/a (in PE)** | exponent offset in the L-Mul adder, not a PPU lane (§3.2) |

The conv small layers (enc0-pw, depthwise, deconv) have bad scale:MAC ratios but
tiny absolute op counts and never saturate 128 PE; FIFO-drain them, or cap their PE
width to ≈ `PPU_lanes × reduction`. The QK / Score×V scales are **not** PPU work at
all — they ride the L-Mul exponent path inside the PE.
