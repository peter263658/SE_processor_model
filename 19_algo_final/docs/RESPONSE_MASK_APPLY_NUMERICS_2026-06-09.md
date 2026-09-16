# Response — mask-apply (mode C) numerics & multiplier precision (2026-06-09)

Answers `CONSULT_MASK_APPLY_NUMERICS_2026-06-09.md`. All numbers from the real
deploy forward on the golden wav (`scripts/analyze_mask_apply_precision.py`),
captured at the actual `apply_mask` call.

**FULL golden (delivered):** `hw_patterns_final/mask_apply_golden/{L,R}/*.hex` — 64 bins ×
399 frames each (`spec_{re,im}.q214`, `mask_{re,im}.fp16`, `masked_{re,im}.q214`), format/
rounding in `mask_apply_golden_meta.json`. (`mask_apply_golden.json` is ONLY the 8-bin
frame-38 worked example, not the full dump — use the `{L,R}/*.hex`.)

## TL;DR — minimum multiplier

**The mask precision is the binding constraint, and it is set by your last decoder
layer — not by the multiplier.** The mask must reach the multiplier at **fp16**
(≈11-bit significand) or **≥ Q1.11** fixed. **e4m3 is far too coarse** (1.9% output
RMS). So:

- Last decoder layer: output the mask in **fp16** (it already computes in the fp16
  PEG tree — just **PASS fp16, do NOT e4m3-cast** the final mask).
- mask-apply multiplier: **fp16(mask) × spec(Q2.14) → round to the iSTFT-input
  Q-format.** Equivalently a ~**12-bit-significand × 16-bit** fixed multiply. Wider
  than that is wasted (the decoder's fp16 caps the achievable accuracy).

## Q1 — mask operand format

The deployed mask (decoder final output `x[:,0]`, complex) is **fp32 in the eager
reference** (not on the e4m3 or fp16 grid — it's the full-precision golden). On
hardware your decoder accumulates in the **fp16 PEG tree**, so the *real* mask is
**fp16**. The precision sweep (mask quantized to each format, then iSTFT, vs the
deploy waveform):

| mask format | masked-spec err | waveform vs deploy fp32 |
|---|---|---|
| fp32 (baseline) | 9e-8 | 0.01 Q15-LSB |
| **e4m3** | 4.8e-2 | **811 LSB, 1.9% RMS  ❌ too coarse** |
| **fp16** | 5.9e-4 | **13 LSB, 0.03% RMS  ✓** |
| Q1.7 | 4.7e-3 | 158 LSB, 0.39% |
| Q1.9 | 1.4e-3 | 49 LSB, 0.10% |
| Q1.11 | 4.5e-4 | 16 LSB, 0.034% |
| Q1.13 / Q1.15 | 5–9e-5 | 15 LSB (output-Q floor) |

→ **Consume the mask as fp16, not e4m3.** e4m3's 3-bit mantissa (step 0.125 near
|mask|≈1) injects ~6% per-coefficient error → 1.9% output RMS, ~60× the iSTFT
2-LSB budget. fp16 (or Q1.11–Q1.13) sits at the output-Q floor.

## Q2 — multiply precision / output format

- Operands: **mask fp16** × **spec Q2.14** (see Q-format flag below).
- A single `fp16 × fp16` multiply works (masked range ±1.4 fits fp16); or a
  `~12-bit signed × 16-bit signed` fixed multiply. The two products per output
  (`sr·mr` and `si·mi`) should be kept at **≥ the output frac width** before the
  subtract/add; round the final `real`,`imag` to the **iSTFT-input Q-format**.
- The output floor is the iSTFT-input fixed-point itself (~15 Q15-LSB if Q1.15),
  not the multiplier — so don't over-build the multiplier past fp16-mask precision.

## Q3 — mask channel layout / DC

- Decoder final output is **[1 ch, 64 bins, 399 frames] complex, per L/R** (two
  separate decoder calls — NOT 128ch). `x[:,0]` drops the channel dim →
  `mask_re/mask_im = [64 bins, 399 frames]`.
- Spec `cspecs` = STFT out `[64 bins, 399 frames]` (DC already removed).
- mode-C is element-wise per (bin, frame): `masked[b]=sr[b]·mr[b]−si[b]·mi[b]`,
  `b=0..63`. Then **F.pad adds DC bin 0 = 0** → masked `[65 bins]`, so **spec bin b
  → masked bin b+1**, masked bin 0 = 0. iSTFT consumes the 65-bin spectrum.

## Q4 — golden alignment (verified)

`spec = stft__0/output`, `mask = decoder.model.5.0` complex output (captured),
`masked = istft__0/input`. fp64 `modeC(spec, mask)` == captured masked == `istft`
input to **9e-8** (fp32), masked bin0 = 0. Ordering = `[bin, frame]`, spec/mask
bins 0..63, masked bins 1..64 (+DC 0). Worked frame 38, bins 1..8 in the JSON
(spec_re/im, mask_re/im, masked_re/im) — like the softmax_stepwise dump.

## ⚠️ Spec Q-format flag

The STFT output exceeds ±1 (**|spec|max = 1.46**, masked 1.38). **Q1.15 (range
[−1,1)) would clip** — the spec/masked path needs **Q2.14** (or wider int part).
Confirm the STFT RTL output is Q2.14, not Q1.15.

## Note on bit-exactness (same story as QK/proj)

The HW mask-apply will not be bit-exact to the *fp32* deploy, because the mask is
**fp16** (decoder fp16 tree), not fp32 — a bounded ~13 Q15-LSB / 0.03% RMS gap,
decoder-limited, not multiplier-limited. The bit-exact target for the RTL is the
**fp16-mask** masked spectrum (`masked_{re,im}.q214` in `{L,R}/`), not the fp32
golden — DELIVERED (`scripts/gen_mask_apply_golden.py`, full 64×399, both sides).
