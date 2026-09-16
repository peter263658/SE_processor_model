# Algo-consultant inventory — open items for the MHA chain (2026-06-09)

The live attention compute chain (in_proj→QK→softmax→score_v) is RTL-verified bit-exact. Below is the full
inventory of **algorithm-side** items the RTL still needs you to **confirm** or **provide**, grouped by
urgency. Items in §A are values already used in the (passing) RTL — please rubber-stamp or correct. §B/§C
block the remaining links (out_proj→recombine→output_proj, mask-apply). Each line: *what RTL assumes* → *ask*.

---

## §A — Confirm values already in the verified RTL (rubber-stamp / correct)

1. **Block-float exponents k_Q / k_K / k_V = −7 / −7 / −6** (recovered by `recover_k`, in_proj cast_k = +7/+7/+6).
   RTL stores KVBUF block-float, reads K ×2⁻⁷ / V ×2⁻⁶; pow2-invariant so QK/ScoreV golden unchanged. *Confirm final.*
2. **mha_in_proj contract**: REAL block-diagonal INT4, `qkv_groups=4`, oc=768=[Q256|K256|V256] head-major,
   ic=64, per-head **48-code** APoT scale order **[Q_h0..15, K_h0..15, V_h0..15]**, **bias=True**. *Confirm final.*
3. **Current frame = local_idx 0** (query's own / newest key; `local_idx = query_frame − key_frame`). *Confirm.*
4. **QK accumulator = FP16 saturating tree** (user-chosen), so softmax/score_v golden = the **fp16-tree** packet
   (not fp32-QK); accepted ~1-head packet divergence at the denom-pow2 boundary. *Confirm this is the intended target.*

## §B — Confirm / provide (BLOCKING the back-half + output stage)

5. **output_proj cast_k = −1 or −2?** Three conflicting values in-repo: on-disk golden `cast_k.txt`=**−2** (06-08,
   stale), script `gen_golden_proj.py:201`=**−1** (06-09), deploy meta `descale_pow2_exp=+2`→−2 (uncross-checked).
   = −(decoder stage-0 input block-float exp). *Confirm the value; we regen `out_{re,im}_code.hex`.*
6. **mask-apply golden dump** (offered in `RESPONSE_MASK_APPLY_NUMERICS`, file not on disk yet): per (bin 0..63,
   frame) hex codes — `spec_re/im` (Q2.14), `mask_re/im` (fp16), `masked_re/im` (fp16-mask result, Q2.14) — plus
   the exact **fp16→Q2.14 rounding mode** (half-even vs half-up). *Provide the full 64-bin×N-frame dump + rounding.*
7. **Last decoder layer (dec5 `pointwise_x`, sub-ops 28/29) must emit fp16 (PASS, no e4m3 cast)** so the mask
   reaches mask-apply at fp16 (e4m3 → 1.9% RMS). ROM currently has these as PPU_CONV (e4m3-cast). *Confirm fp16-PASS.*
8. **Spec / iSTFT-input / mask-apply bus = Q2.14, not Q1.15** (|spec|max=1.46 clips Q1.15; you flagged it).
   *Confirm Q2.14 so we set the mask helper FXP_FRAC=14.* (STFT-owner side, but you raised it.)

## §C — Missing goldens / finalize descriptors

9. **mha_out_proj IMAG-pass golden** not emitted — only `mha_out_proj_real/` exists; `gen_golden_mha_out_proj.py`
   supports the `imag` arg but it hasn't been run. *Recombine needs both passes — please emit the imag golden.*
10. **recombine** `ComplexMHA(x) = (a−b) + i(a+b)`, a=M(x.real), b=M(x.imag) = the two passes' **out_proj** outputs
    (FP16), applied AFTER internal out_proj, BEFORE outer output_proj. No standalone golden exists. *Confirm the
    formula + operand order (out_re=a−b, out_im=a+b) and that a,b are the post-bias out_proj FP16 vectors.*
11. **PROVISIONAL ROM descriptors** (`ce_subop_rom.svh` sub-ops 30–35: input_proj, mha_in_proj, qk, score_v,
    mha_out_proj, output_proj) are marked PROVISIONAL. *Confirm the offsets / dims / modes are final, or provide
    the finalized descriptor fields (esp. out_proj sub-op 34 — see the bias-mapping request to the exporter).*

---

### What each unblocks
- §B-5 → output_proj e4m3 output golden match (#7). §B-6,7,8 → mask-apply bit-exact verify.
- §C-9,10 → recombine (#6). §C-11 → freezes the whole attention sub-op walk.
- §A → just confirmation; RTL already passes against these.
