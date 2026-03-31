# BCCTN_sep_conv_causal_slide_win_relu — ReLU Activation

Swaps PReLU activations for ReLU to simplify the activation function for hardware implementation.

See the [root README](../README.md) for full setup, training, and evaluation instructions.

## Variant-specific notes

- Based on `BCCTN_sep_conv_causal_slide_win`
- All PReLU replaced with ReLU — removes learnable slope parameters
- Hardware-friendly: ReLU is simpler to implement in fixed-point arithmetic
