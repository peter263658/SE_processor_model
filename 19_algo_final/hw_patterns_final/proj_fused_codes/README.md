# Projection fused codes (act-k) + bias — input_proj / output_proj

Closes the two RTL gaps for the outer complex projections (see
`docs/MHA_GOLDEN_HANDOFF_2026-06-08.md` Part D). Generated + verified by
`scripts/gen_proj_fused_codes_and_verify.py`. The same files are also dropped
**additively** into `tb_pattern/weights/<layer>/` (new filenames only).

## What's here (per `<layer>/<half>`)

- `scale_apot_{a,b}_exp_fused_act_kreal.int16` — weight APoT exponent **fused with
  the real-activation k** (`a+k_real`, `b+k_real`).
- `scale_apot_{a,b}_exp_fused_act_kimag.int16` — fused with the imag-activation k.
- `scale_apot_code_fused_act_{kreal,kimag}.u16` — packed 12-bit APoT-6a code
  (`[11:7]=a+24, [6:2]=b+24, [1]=has_second, [0]=res_sign`; value = `2^a ± 2^b`).
- `scale_apot_has_second.u8` — shared with the raw scale (fusion doesn't change it;
  `residual_sign` likewise comes from the raw `scale_apot_residual_sign.int8`).
- `<layer>/bn_bias_{r,i}_fused.fp16` — PPU β bias, **direct** (β_re=b_r, β_im=b_i).

## Per-PEG dealer wiring

`head(oc)=oc//16`. Complex MAC `P0=fc_r·x_real, P1=fc_i·x_imag, P2=fc_r·x_imag,
P3=fc_i·x_real`; `out_re=P0−P1, out_im=P2+P3`:

| PEG | weight | act | fused code to load |
|---|---|---|---|
| P0 | fc_r | x_real | `fc_r/…fused_act_kreal` |
| P1 | fc_i | x_imag | `fc_i/…fused_act_kimag` |
| P2 | fc_r | x_imag | `fc_r/…fused_act_kimag` |
| P3 | fc_i | x_real | `fc_i/…fused_act_kreal` |

`k_real,k_imag`: input_proj = −7,−6 (split-k); output_proj = −3,−3 (collapses,
both variants equal). `preshift=0` for the proj (overflow-safe), so the dealer
carries the full fused descale.

## Verified

Reconstruct `out = fused_descale · Σ(int4 · e4m3) + bias` (complex) vs the fp32
golden output: **R² = 1.00000000** for both layers. Raw apot without act-k →
R² = −7500 (input) / −49.8 (output); dropping bias → R² = 0.987 / 0.999999.
