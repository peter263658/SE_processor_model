# BCCTN — Baseline

Original paper architecture without modifications.

See the [root README](../README.md) for full setup, training, and evaluation instructions.

## Variant-specific notes

- Standard (non-causal) convolution — processes full utterance at once
- Reference implementation used to validate other variants
- `kernel_size = (5, 2)` in frequency × time
