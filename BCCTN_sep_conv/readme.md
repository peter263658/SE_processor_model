# BCCTN_sep_conv — Depthwise Separable Convolution

Replaces standard `ComplexConv2d` with depthwise separable convolutions to reduce FLOPs and parameter count.

See the [root README](../README.md) for full setup, training, and evaluation instructions.

## Variant-specific notes

- `model.use_depthwise = true`
- Non-causal (full-utterance) processing
- Baseline for the causal and sliding-window variants
