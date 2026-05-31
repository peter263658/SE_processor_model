# Handoff Review Notes

This file records issues found while reviewing the latest handoff against the
source code and the dump scripts.

## 1. PReLU Granularity

`HANDOFF_HW_ZERO_MULTIPLIER_CORE.md` describes PReLU slope as per-channel. The
current source code constructs `ComplexPReLU` with default `torch.nn.PReLU()`,
which has `num_parameters=1`.

Actual deployed behavior:

- one scalar `alpha_r` per real PReLU module;
- one scalar `alpha_i` per imag PReLU module;
- each scalar is quantized to signed pow2 `[-8, 0]`.

Export handling:

- `software_to_hardware_18/ckpt2hwdat.py` exports the actual scalar parameter
  count and pow2 exponent/sign files.
- Hardware may broadcast the scalar across all channels in that block.

Impact:

- The zero-multiplier claim is still valid.
- Do not allocate per-channel PReLU ROM based solely on the older wording unless
  the model source is changed and retrained with per-channel PReLU.

## 2. MHA QKV Group Mask Timing

The source applies the `qkv_groups=4` block-diagonal mask in
`_mha_attention_forward` after reading the parametrized INT4 fake-quantized
`in_proj_weight`.

Actual deployed behavior:

1. INT4 per-head scale is computed on the underlying dense `in_proj_weight`.
2. INT4 q codes are produced.
3. `qkv_groups=4` mask zeroes off-diagonal q codes at runtime.

Export handling:

- The dump script matches source behavior: it computes per-head q/scale first,
  then applies the block-diagonal mask.
- It also emits a compact block file containing only the diagonal 64x64 Q/K/V
  blocks for hardware storage.

Impact:

- Matching source behavior is more important than recomputing scale only over
  diagonal blocks, because recomputing diagonal-only scales would change q codes.

## 3. QK Scale

The old 09 dump script pre-fused `1/sqrt(head_dim)` into Q projection scales and
bias. The current 18 handoff says `qk_scale_mode=pow2`, so QK scale is a runtime
shift.

Export handling:

- The 18 dump script does not pre-fuse QK scale into Q.
- MHA metadata records `qk_scale_runtime_pow2_exp = -2` for `head_dim=16`.

## 4. Conv Pointwise Small-Channel Cases

The handoff says Conv pointwise scale is per group of 32 input channels. For
some small grouped branches, the effective source shape has fewer than 32 input
channels per grouped Conv2d slice.

Export handling:

- If the source input-channel dimension is divisible by 32, the dump exports
  true per-group-32 scales.
- If it is smaller than 32, the dump exports one effective scale for that output
  branch and records `effective_scale_granularity` in layer `meta.json`.

Hardware interpretation:

- Treat these as partial/single input groups for the branch.

## 6. FP8 Outlier Weights Are Per-Tensor Scaled FP8 (FIXED 2026-05-30)

The 12 decoder PW-outlier layers use the source `FP8Parametrization`
(`DCNN/utils/fp8_qat.py::FP8QuantizeSTE`), which is a **per-tensor amax-scaled
FP8**, not a plain `w.to(float8_e4m3fn)` cast:

```
s      = 448.0 / amax(|w|)
weight = fp8_e4m3(w · s) / s        # == module.weight
```

Earlier the export wrote a plain `fp8_e4m3(w)`, which diverged from the validated
model by up to 0.14 per weight (~4% relative output L2 end-to-end). Fixed: the
export now writes `w_fp8.e4m3.bin = fp8(w·s)`, `fp8_scale` (`448/amax`),
`fp8_descale` (`amax/448`), `w_fp8.effective.fp32.bin` (= module.weight), and an
APoT-encoded descale. `verify_export_equiv.py` reconstructs `code/scale` and now
matches `module.weight` at maxabs = 0 for all 12 layers.

Zero-multiplier note: the per-tensor descale factors out of the per-element L-Mul
to one fp16-accumulator scaling per output; realize it as a pow2/APoT exponent
offset folded into L-Mul (shift-add). See `HANDOFF_HW_ZERO_MULTIPLIER_CORE.md`
§3.4 — the HW team must record whether they use the APoT descale (re-validate) or
one exact per-output multiply for these 12 layers.

**RESOLVED (delayed APoT descale + recovery finetune).** The deploy point uses the
**delayed 2-term APoT descale** (shift-add, no multiply), range `[-16,-6]`
(data-driven: epoch-2 descale exponents live in `[-14,-8]`; 4-bit field with
margin). Evaluating that path on the prior ckpt cost ~4% output L2 (VCTK 0.8335),
so a **5-epoch recovery finetune trained under the delayed APoT descale**
(`BCCTN_OUTLIER_DELAYED_DESCALE=1`, runtime in `code/DCNN/utils/all_lmul_multiply.py`)
recovers to **VCTK 0.8365 / TIMIT 0.7912** — above both gates and above the
exact-descale point. Deploy ckpt
`ckpt/strict_core_apot_descale_deploy_ep2_vctk0.8365_timit0.7912.ckpt`; re-dump via
`CKPT=<that> software_to_hardware_18/run_output.sh`. The "exact per-output multiply"
fallback is therefore NOT needed — the core stays multiplier-free.

## 5. BN Bias Fusion

The source still has conv bias parameters in the Conv modules. Since BN is
exported as a standalone folded affine after the complex MAC, the export folds
the effective complex conv bias into the BN bias:

- real path bias contribution: `b_r - b_i`
- imag path bias contribution: `b_r + b_i`

Export handling:

- `bn_bias_{r,i}_fused.fp32.bin` already includes this conv-bias contribution.
- Conv weight directories mark `conv_bias_fused_elsewhere`.

