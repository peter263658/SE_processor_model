# BCCTN_sep_conv_causal_slide_win — Causal + Sliding Window Attention

**Current main focus.** Adds a sliding window constraint to the transformer attention, bounding the attention complexity from O(S²) to O(S × W) where W = 31 frames.

See the [root README](../README.md) for full setup, training, and evaluation instructions.

## Variant-specific notes

- Causal separable convolution + sliding window attention (`model.attn_win = 31`)
- `model.win_len = 128`, `model.fft_len = 128`
- `training.fp8_qat = true` — FP8 quantization-aware training enabled
- Hardware-friendly: bounded attention and reduced FFT size
