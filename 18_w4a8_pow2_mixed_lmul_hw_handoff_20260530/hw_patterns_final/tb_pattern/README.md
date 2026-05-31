# Testbench Pattern (block-float deploy)

Compact hardware-facing files only:

- `weights/`: packed INT4/FP8 weight codes + APoT/pow2 weight-scale codes, plus the
  **fused** descale = weight APoT exponents − k (k = the activation block-float exponent).
  - Most layers: one `*_apot_*_fused.int16.bin` (single activation exponent).
  - **Decoder split-pointwise** (`pointwise_*`): TWO variants
    `*_apot_*_fused_act_kreal.int16.bin` and `*_apot_*_fused_act_kimag.int16.bin`,
    because `apply_complex(fr, fi, x)` multiplies each sub-conv weight by BOTH the
    real-slice (cat.real exp) and imag-slice (cat.imag exp) activations. They are
    identical where cat.real==cat.imag (most stages) but DIFFER at stage 3
    (real=2⁻⁵, imag=2⁻⁶). HW must pick the variant matching the activation slice —
    do NOT assume one fused descale per weight here.
  See `weights/fused_descale_summary.json` (`branches_collapse` flags the equal ones).
- `blocks/`: BN APoT scale codes, fused BN bias constants, PReLU pow2 codes.
- `skip_indices/`, `misc/`: deployment constants.
- `activations_fp8/`: per-layer **input** activations as **raw e4m3 uint8 code**
  (block-float). No per-activation scale/descale is stored — the descale 2^-k is
  already fused into the weight descale above. Outputs/non-arithmetic leaves are
  NOT here (debug_reference only). Stft/iSTFT are Q1.15 DSP, not block-float.

Activations are block-float e4m3 (mode `calibrated_pow2_e4m3`): recorded in
`activations_fp8/meta.json`. Exponent table also at
`../export_out_strict18/activation_blockfloat/`.
NOT fused (kept with separate exponent): MHA in/out projections (real/imag k may
differ; out_proj is W4A16) and QK/score×V (activation×activation, no weight).
