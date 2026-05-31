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
| PReLU slope | pow2 `±2^k`, **per-channel**, `k ∈ [-8,0]` | 1 shift + sign |
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
- **PE arithmetic slice**: each lane should expose two 8-bit integer adder slices
  that can run either **split** (two independent <=8-bit ops) or **chained**
  (one <=16-bit add/sub with carry between slices). This covers all current
  mixed-precision modes without a second PE array:
  - W4A8 INT4×block-float e4m3 raw-code shift-add: e4m3 significand is hidden-1+3 mantissa bits
    (4 bits); INT4 CSD shift is 0..3, so the product front-end needs about
    8-bit signed add/sub. Use one 8-bit slice.
  - Decoder FP8×FP8 L-Mul m=4: logical 4-bit mantissa add; the bit-exact
    sig-code path includes the hidden one and offset/carry, so budget 6 bits.
    Use one 8-bit slice.
  - Attention QK / Score×V L-Mul m=8: logical 8-bit mantissa add; the sig-code
    path needs about 10 bits with offset/carry. Chain two 8-bit slices as a
    16-bit adder.
  - MHA internal out_proj W4A16 exception: input is the on-chip fp16/wide
    Score×V accumulator, not an FP8 activation boundary. INT4 CSD shift 0..3
    over an fp16 significand (hidden-1+10 mantissa bits) needs about 15-bit
    signed add/sub. Chain two 8-bit slices as a 16-bit adder. This mode is only
    for the internal MHA out_proj; do **not** duplicate a full W4A16 PE array.
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
| PReLU slope | channel (r & i) | (k-exp, sign), k∈[-8,0] |
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
- [ ] MHA internal out_proj: only W4A16 exception; reuse the same INT4 shift-add
      PE in chained 2×8-bit mode, input from Score×V fp16/wide accumulator.
- [ ] QK^T, Score×V: L-Mul m=8, fp16 accumulate. QK applies
      `2^-(k_q+k_k+2)` before softmax; Score×V applies `2^-(8+k_v)` before the
      internal W4A16 `out_proj`.
- [ ] PE adder slices: support split 2×8-bit mode and chained 16-bit mode; map
      W4A8=8b, FP8-outlier L-Mul m4≈6b, attention L-Mul m8≈10b, W4A16≈15b.
- [ ] L-Mul exponent path: sign XOR + exponent add + mantissa carry + signed
      scale/exponent offset; do not size this as only a 4-bit FP8 exponent add.
- [ ] Softmax: top-4, 2^⌊·⌋ (no mantissa, clamp exp), pow2 denominator. No exp LUT / divider.
- [ ] Decoder PW outliers (12): per-tensor scaled FP8 (`code=fp8(w·448/amax)`, descale
      `amax/448`), FP8×FP8 L-Mul m=4; descale folded into L-Mul exponent (pow2/APoT) — §3.4.
- [ ] BN: folded `a_q·x + b`, `a_q` = APoT of `gamma/sqrt(var+eps)`, exp∈[-9,4].
- [ ] PReLU: pow2 slope, sign mux + shift, exp∈[-8,0].
- [ ] No general multiplier, no divider, no exp LUT in the neural core.
