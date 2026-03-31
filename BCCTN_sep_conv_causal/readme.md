# BCCTN_sep_conv_causal — Causal Streaming

Causal variant: convolutions use causal padding so the model only attends to past and current frames, enabling real-time streaming inference.

See the [root README](../README.md) for full setup, training, and evaluation instructions.

## Variant-specific notes

- Causal padding in all convolutional layers (no future frames)
- Depthwise separable convolution (`model.use_depthwise = true`)
- Suitable for low-latency deployment; latency = one STFT frame (80 samples @ 16 kHz = 5 ms)
