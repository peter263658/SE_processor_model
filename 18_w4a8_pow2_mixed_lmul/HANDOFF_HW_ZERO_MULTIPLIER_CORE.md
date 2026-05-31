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
- **FP8 activation boundary**: the FP8 e4m3 format applies at the *inputs* of the
  Conv/Linear/MHA arithmetic blocks. BN and PReLU (§3.5–3.6) compute a folded
  affine in wider precision; their outputs must be **re-quantized to FP8 at the
  next arithmetic-block input**.
- All per-channel / per-group constants (§5) are computed once from the trained
  checkpoint and loaded into the chip; runtime datapath never multiplies.

---

## 1. Number formats

| Tensor | Format | Notes |
|---|---|---|
| Conv/Linear/MHA-proj weights | INT4 symmetric, code `q ∈ [-7,7]` | 4-bit signed, qmax=7 |
| Activations into Conv/Linear/MHA blocks | FP8 e4m3 (1-4-3) | re-quantized at each FP8 op boundary (§0, §3.5–3.6) |
| Decoder PW-outlier weights (12 layers) | FP8 e4m3 | kept FP8, not INT4 |
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
| Conv / Linear (PW per-group 32, DW per-channel) | INT4 `q` × FP8 act + APoT scale | shift-add MAC + delayed 2-shift+add scale |
| MHA Q/K/V/O projection | INT4 `q` × FP8 act + **per-head** APoT scale | shift-add MAC + delayed 2-shift+add scale |
| Attention QK^T | L-Mul (m=8) | exp-add + mantissa-add(8b) + offset |
| Attention softmax | top-4, exp→pow2 (no mantissa), pow2 denom | shift + shift-divide, no exp LUT / divider |
| Attention Score×V | L-Mul (m=8) over top-4 packet | exp-add + mantissa-add + gather |
| Decoder PW outlier (12 FP8 layers) | FP8 × FP8 L-Mul (m=4) | exp-add + mantissa-add(4b) |
| BatchNorm | folded per-channel affine `a_q·x + b` | 2-shift+add (APoT) + add |
| PReLU | `x≥0 ? x : (±2^k)·x` | sign mux + 1 shift |

---

## 3. Op detail

### 3.1 Conv / Linear / MHA projection — W4A8 shift-add MAC + delayed APoT scale

**Conv-PW / grouped-Linear case** (scale group = 32 input channels):

```
partial = Σ_{i in group}  q_i · a_i           # q_i: INT4 code, a_i: FP8 activation
out     = Σ_groups  APoT_scale_g · partial_g   # APoT_scale_g per group of 32
```

The scale-group boundary is **layer-type specific** (same datapath, different
constant granularity): **Conv-DW** = one scale per output channel (no input
grouping); **MHA Q/K/V/O projection** = one scale **per head** (`APoT_scale_head`
applied to that head's accumulation). See §1 / the bullets below.

- `q_i · a_i`: `q_i` is a 4-bit signed integer (|q|≤7 ⇒ ≤3 significant bits).
  The product is `q_i` shifted-adds of the FP8 value `a_i` (CSD form ⇒ ≤2
  add/sub). No multiplier.
- Accumulate the shift-add products in a wide FP accumulator.
- **Delayed scale**: after a scale group's accumulations, scale the partial sum
  by `APoT_scale = 2^a ± 2^b` (2 barrel shifts + 1 add/sub), then accumulate
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

### 3.2 Attention QK^T and Score×V — L-Mul (m=8)

L-Mul approximates a float multiply `x·y` without a multiplier:

```
sign(out) = sign(x) xor sign(y)
exp(out)  = exp(x) + exp(y) + offset(m)        # integer add + constant
mant(out) = mant(x) + mant(y)                  # m=8-bit mantissa add (+ carry into exp)
```

- m = 8 mantissa bits. Hardware = 1 exponent adder + 1 (8-bit) mantissa adder +
  constant offset; no multiplier.
- QK scale: `qk_scale_mode=pow2` ⇒ 1 shift.
- Accumulate in fp16.
- Score×V runs over the **top-4 sparse packet** (§3.3): gather the 4 selected V
  rows, L-Mul each by its prob, accumulate.

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

Config trace: `low_power.outlier_fp8_lmul.enabled=true, mantissa_bits=4`.

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
- **Weight reuse**: L and R channels share weights — single weight copy in WBUF.
- **Buffers**: WBUF = 32 KB, UFBUF = 4 KB. Feature maps stay fully on-chip.
- **MHA Q/K/V projection**: block-diagonal grouping `qkv_groups=4` — store and
  compute only the diagonal 64×64 blocks (off-diagonal weights do not exist).
  Required to meet the 64-lane @ 5 MHz real-time duty (attention MAC ≈ 1.10 M/frame,
  duty ≈ 0.81).
- **Decoder pointwise**: grouped `groups_x=4`, `groups_skip=2`; the skip input
  uses only the top-50% channels by L1 magnitude (fixed index set, loaded once).
  Real `Conv2d(groups=…)` + channel slice — both stored and computed weights are
  reduced.
- **Attention probability transport**: top-4 sparse packet, 5-bit local index,
  q0.8 probability ⇒ ≈52 bits/row (vs 512 bits dense FP16).
- **Per-group weight scale**: group_size = 32 (one APoT code per 32 input
  channels).

---

## 5. Load-time constants (computed once from the checkpoint; no runtime multiply)

| Constant | Per | Format |
|---|---|---|
| INT4 weight codes `q` | weight element | 4-bit signed |
| Conv-PW weight APoT scale | group of 32 in-ch | (a-exp, b-exp, residual-sign), a,b∈[-16,-1] |
| Conv-DW weight APoT scale | output channel | (a-exp, b-exp, residual-sign), a,b∈[-16,-1] |
| MHA Q/K/V/O proj APoT scale | **head** | (a-exp, b-exp, residual-sign), a,b∈[-16,-1] |
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

- [ ] Conv/Linear/proj: INT4×FP8 shift-add + delayed APoT scale — granularity:
      Conv-PW/Linear per-group(32), Conv-DW per-output-channel, MHA proj per-head.
- [ ] QK^T, Score×V: L-Mul m=8, fp16 accumulate, QK scale = pow2 shift.
- [ ] Softmax: top-4, 2^⌊·⌋ (no mantissa, clamp exp), pow2 denominator. No exp LUT / divider.
- [ ] Decoder PW outliers (12): FP8×FP8 L-Mul m=4.
- [ ] BN: folded `a_q·x + b`, `a_q` = APoT of `gamma/sqrt(var+eps)`, exp∈[-9,4].
- [ ] PReLU: pow2 slope, sign mux + shift, exp∈[-8,0].
- [ ] No general multiplier, no divider, no exp LUT in the neural core.
