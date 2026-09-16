# Strict-18 Architecture Summary

This is a short reading guide for hardware review. The authoritative detailed
mapping is `HANDOFF_HW_ZERO_MULTIPLIER_CORE.md`.

## Neural-Core Claim

The neural inference core uses no general multiplier, no divider, and no exp
LUT. Runtime arithmetic is shift, add/sub, mux/sign, small constant decode, and
L-Mul add-based floating product approximation.

Out of scope: STFT/iSTFT, analysis/synthesis windows, and waveform-domain mask
application.

## Main Dataflow

1. Binaural STFT features enter the encoder.
2. Encoder: 6 complex depthwise-separable Conv blocks.
3. Bottleneck: tied complex MHA with input/output projection.
4. Decoder: 6 complex depthwise-separable transpose blocks with grouped split
   pointwise branches and static skip-channel subset.
5. Mask application and iSTFT are outside the zero-mult neural core.

## Structural Compression

| Part | Deploy setting |
|---|---|
| Tied complex MHA | one shared real MHA module; complex output from two real passes |
| MHA QKV projection | `qkv_groups=4`, only diagonal 64x64 Q/K/V blocks stored/computed |
| Decoder pointwise | split x/skip branches, `groups_x=4`, `groups_skip=2` |
| Decoder skip | static top-50% channel index set, copied in `ckpt/p2_skip_indices_g42_topk_l1_r050.pt` |
| Attention probability | top-4 sparse q0.8 packet, 5-bit local index |

## Numeric Formats

| Item | Format |
|---|---|
| Conv/Linear/MHA projection weights | signed INT4, qmax=7 |
| Arithmetic-block activations | **block-float**: e4m3 raw code + per-tensor pow2 shared exponent (static calib; descale `2^-k` fused into post-acc weight descale as exponent offset; no runtime scaling) |
| Conv pointwise scale | positive APoT per 32 input channels, exponent range `[-16, -1]` |
| Conv depthwise scale | positive APoT per output channel, exponent range `[-16, -1]` |
| MHA projection scale | positive APoT per head, exponent range `[-16, -1]` |
| BN folded scale | signed APoT per channel, exponent range `[-9, 4]` |
| PReLU slope | signed pow2 per PReLU parameter, exponent range `[-8, 0]` |
| Decoder pointwise outliers | per-tensor scaled FP8 e4m3 (`code=fp8(w·448/amax)`, descale `amax/448`), FP8xFP8 L-Mul m4 |
| QK and Score x V | L-Mul m8 |

## Dump Script Semantics

Use `software_to_hardware_18/ckpt2hwdat.py`.

Important differences from the old `09_software_to_hardware` scripts:

- Builds the copied 18 strict model, not the old 09 `BinauralAttentionDCNN`.
- Reads the packaged strict config and checkpoint by default.
- Rewrites decoder skip path to the copied package skip-index file.
- Exports Conv 4D per-group scale on the input-channel axis, not the trailing
  kernel-width axis.
- Exports APoT scale code files: `a_exp`, `b_exp`, `has_second`, residual sign,
  plus overall sign for signed BN scale.
- Exports BN as folded `alpha_q * x + beta_fold`, where `alpha_q` is APoT of
  `gamma / sqrt(running_var + eps)`.
- Fuses complex conv bias into exported BN bias.
- Exports PReLU as signed pow2 exponent/sign.
- Exports decoder split pointwise branches explicitly:
  `pointwise_{r,i}_{x,skip}`.
- Exports MHA QKV with the `qkv_groups=4` block-diagonal mask and compact
  diagonal block files.
- Keeps QK `1/sqrt(head_dim)` as runtime pow2 shift metadata instead of
  pre-fusing it into Q projection scale.

## Selected Deploy Checkpoint

`ckpt/strict_core_pow2_blockfloat_deploy_ep3_vctk0.8357_timit0.7898.ckpt`

This is the deploy checkpoint: starting from the delayed-APoT-descale deploy, a
5-epoch recovery finetune UNDER **pow2 block-float activations**
(`BCCTN_ACTIVATION_FP8_MODE=calibrated_pow2_e4m3`) + the delayed 2-term APoT
FP8-outlier weight descale. Activations are e4m3 raw code + a per-tensor pow2
shared exponent (static calib `floor(log2 448/amax)`); the activation descale
`2^-k` fuses into each layer's post-acc weight descale as an exponent offset, so
**eager==fused exactly** (verified relL2 = 0) and **no runtime activation scaling**
is needed. Post-hoc MBSTOI screening over the top-3 epochs picked epoch 3 (val
−13.5300, best val and best MBSTOI; ep4 0.8351, ep2 0.8339).

Average MBSTOI enhanced (pow2 block-float = the deployed HW path):

- VCTK: `0.8357`   (gate 0.835; prior dynamic-activation deploy 0.8365)
- TIMIT: `0.7898`  (gate 0.7880; prior dynamic-activation deploy 0.7912)

Full-eval result dir: `results/apot_descale_eval/ft_pow2_ep3_full/` (VCTK+TIMIT).
Prior dynamic-activation deploy (0.8365/0.7912): `results/apot_descale_eval/ft_ep2_full/`.
Prior exact-descale point (`strict_core_epoch4_step1210_val13.6105.ckpt`,
0.8355 / 0.7902) is kept for reference.

