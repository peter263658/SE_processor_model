# Consult — mask-apply (mode C) numerics: operand formats / multiply precision (2026-06-09)

The output stage applies the decoder's complex mask to the noisy STFT spectrum (config `masking_mode='C'`),
then iSTFT. The cores (STFT, iSTFT, fp8-cast) are RTL-verified (iSTFT 2 LSB, cast 0 real-err). The
mask-apply itself is the one place in the otherwise zero-multiplier design where a real multiply is
needed — the user leans toward a **small dedicated multiplier**; we need the precision/format pinned.

## The op (apply_mask.py, mode C, confirmed)
```
real, imag = spec_re·mask_re − spec_im·mask_im ,  spec_re·mask_im + spec_im·mask_re
# then F.pad DC bin (bin 0 = 0); out = complex(real, imag) -> iSTFT
```
- `spec` (spec_re, spec_im) = STFT output (cspecs), **Q1.15** in the DSP/STFT domain (FFT128, 64 bins,
  DC removed). golden: `debug_reference/layer_io_fp32/layers/stft__0/output_0.{real,imag}` (399×64).
- `mask` (mask_re, mask_im) = the **decoder final output** (x[:,0]), the neural-core output — **e4m3 / fp16?**
  (decoder is W4A8; last layer `decoder.model.5.0.pointwise_x`). Channel layout = L/R × 64 bins, real/imag.
- golden masked spectrum (iSTFT input): `istft__0/input_0.{real,imag}` (399×65, DC-padded).

## Questions
1. **mask operand format** into the multiply: is the decoder mask consumed as **e4m3**, **fp16**, or
   re-quantized to **Q1.15** to match the spectrum? (i.e. spec_Q15 × mask_? — what does the multiplier see?)
2. **multiply precision / output format**: small fixed-point multiplier (Q1.15 × Q1.15 → Q1.15 w/ round?),
   or fp16 multiply, or Q1.15 × fp8(mask)? And the accumulate of the 2 products per output (sr·mr − si·mi):
   Q-format + rounding of the subtract/add? Output back to Q1.15 for iSTFT?
3. **mask channel layout / DC**: how the decoder output [128ch?] maps to (mask_re[64], mask_im[64]) per
   L/R, and the DC-bin handling (the F.pad adds bin 0 = 0 -> 65 bins to iSTFT).
4. **golden alignment**: confirm spec=stft__0/output, mask=decoder-out, masked=istft__0/input so we can
   verify the RTL mask-apply bit-(or LSB-)exact; and the per-bin/per-frame ordering.

A worked per-bin golden (spec_re/im, mask_re/im, masked_re/im for a few bins of one frame) like the
softmax_stepwise dump would let us pin the multiplier format + rounding exactly.
