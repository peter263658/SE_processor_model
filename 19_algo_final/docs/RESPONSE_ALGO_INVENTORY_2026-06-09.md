# Response — algo-inventory + chain blockers (2026-06-09)

Answers `CONSULT_ALGO_INVENTORY_2026-06-09.md` + `RELAY_CHAIN_BLOCKERS_2026-06-09.md`.
All confirmations are measured against the deploy ckpt on the golden wav.

## §A — rubber-stamp (all CONFIRMED)

1. **k_Q / k_K / k_V = −5 / −5 / −5** ⚠️ CORRECTED (was −7/−7/−6). The deploy uses ONE
   shared block-float exponent for Q/K/V (`mattn.mattn.activation_quantizer`, calib amax 11.45
   → −5). Full-seq |K|max=7.5, |V|max=13 OVERFLOW e4m3 at −7/−6 (only fit at −5). The earlier
   −7/−7/−6 was a find_k artifact on frame-351 small values. **in_proj cast_k = +5/+5/+5.** pow2-invariant → QK/ScoreV
   golden unchanged. **Confirmed final.**
2. **mha_in_proj contract** ✓ — REAL block-diagonal INT4, `qkv_groups=4`, oc=768 =
   [Q256|K256|V256] head-major, per-head **48-code** APoT [Q_h0..15, K_h0..15, V_h0..15],
   **bias=True** (in_proj_bias len 768, |max| 0.43). **Confirmed.**
3. **Current frame = local_idx 0** ✓ (`local_idx = query − key`, 0 = newest/own). **Confirmed.**
4. **QK accumulator = fp16 saturating tree** ✓ — softmax/score_v golden target = the
   **fp16-tree** packet (not fp32-QK). The ~1-head denom-pow2 divergence is the intended,
   accepted target (Part C: benign, top-k 99.8% stable). **Confirmed intended.**

## §B

5. **output_proj cast_k = −1.** ✅ **PROVEN, not just chosen.** Casting the output_proj
   output at consumer-k=+1 (`i_cast_k = −1`) reproduces the **actual decoder stage-0
   consumer's e4m3 codes 100.00%**; k=+2 (cast_k=−2) → **0.00%**. The −2 (from deploy
   meta `descale_pow2_exp=+2`) is a different quantity (weight descale), never the
   activation cast. **Drop-in corrected golden:**
   `hw_patterns_final/proj_fused_codes/mattn.output_proj/e4m3_out_castk_-1/out_{re,im}_code.{hex,u8}`
   ([399,256] re/im; L-half real == decoder pw_x codes 100%). Use this as the RTL target;
   delete the stale −2 `cast_k.txt` / `out_*_code.hex`.

6. **mask-apply fp16-mask golden — DELIVERED.** `hw_patterns_final/mask_apply_golden/{L,R}/`,
   full 64-bin × 399-frame, bin-major. Per (bin, frame): `spec_{re,im}.q214` (int16),
   `mask_{re,im}.fp16` (uint16), `masked_{re,im}.q214` (int16). **Rounding = round-half-to-EVEN
   (RNE)**, a **single** round to Q2.14 after the full-precision products + subtract/add
   (meta: `mask_apply_golden_meta.json`). ⚠️ Accumulator must carry ≥ Q2.14+guard (fp32/wide);
   **fp16-accumulate would NOT match the golden bit-exactly.** DC bin 0 = 0; spec/mask bins
   0..63 → masked bins 1..64.

7. **Last decoder layer (dec5 pointwise_x, sub-ops 28/29) = fp16-PASS, NO e4m3 cast.** ✅
   Confirmed. e4m3 mask → 1.9% RMS / 811 Q15-LSB (~60× the iSTFT 2-LSB budget); fp16 → 13 LSB.
   Change those ROM entries from PPU_CONV to the fp16-PASS path.

8. **Bus = Q2.14, not Q1.15.** Data: |spec|max=**1.46**, |masked|=**1.38** → both clip in
   Q1.15 [−1,1). Needs **Q2.14** (range [−2,2), step 2⁻¹⁴) → mask helper `FXP_FRAC=14`.
   (STFT-owner's bus decision; the data unambiguously requires Q2.14. `Param_def FX_FRAC=15`
   and the `Hamming.sv "Q2.14"` / `istft_reorder "Q1.15"` comments are inconsistent — reconcile to Q2.14.)

## §C

9. **mha_out_proj bias + imag-pass.** The out_proj bias **exists** (`…/MultiheadAttention_re/
   out_proj/bias.fp16.bin`, len 256, |max| 0.35, mean 0.11 — significant), it's just not
   mapped in the ROM. **Map it to the PPU β path (plain add, α=identity), in BOTH passes**
   (same pattern as in_proj). The per-pass out_proj golden for both passes already exists in
   `hw_patterns_final/mha_module_io/{real,imag}_pass/05_out_proj/` (input ctx256 + output256).
10. **Recombine = out_re = a−b, out_im = a+b** ✅ verified: a = out_proj(M(x.real)),
    b = out_proj(M(x.imag)), both **post-bias FP16**; `(a−b), (a+b)` reproduce the outer
    output_proj input (to the e4m3 block-float quantization, k=−3, that output_proj applies
    to its input). **Bias note:** since the same bias `bo` is added in each pass, it
    **cancels in the real output (a−b)** and **doubles in the imag (a+b = …+2·bo)** — this is
    automatic if you add `bo` per-pass then recombine (no special handling). After recombine
    the (a−b),(a+b) FP16 vector is e4m3-cast at **i_cast_k=+3** (output_proj input k=−3)
    before the outer output_proj.
11. **ROM descriptors (sub-ops 30–35):** algo-relevant fields confirmed — 30 input_proj
    (W4A8, split-k per-head, β bias, cast_k per consumer), 31 mha_in_proj (§A-2, cast +5/+5/+5),
    32 qk (fp16-tree, ×2⁻²), 33 score_v (fp16-tree packet), 34 mha_out_proj (W4A16, **β bias
    add — see §C-9**), 35 output_proj (W4A8 split-k, β bias, **cast_k=−1 §B-5**). The two
    descriptor edits needed: sub-op 34 bias pointer, sub-op 35 cast_k=−1.

## #1 (out_proj W4A16 fp16 operand path)

The out_proj activation **is** fp16 (the Score×V context) — W4A16 is correct, it's the only
16-bit-activation op in the core. The fp16 operand path is genuinely required (casting ctx to
e4m3 would be W4A8 and lose Score×V precision; not the deploy). If you want to avoid the fp16
path, the only alternative is e4m3 ctx (W4A8) — I can quality-screen that if you ever want it,
but the deploy is W4A16, so the fp16 path is the faithful build.
