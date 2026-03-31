# BCCTN_quant — Standalone Quantization Experiment

Early quantization experiment on the baseline BCCTN architecture (before the causal/sliding-window variants).

See the [root README](../README.md) for full setup, training, and evaluation instructions.

## Variant-specific notes

- Uses `bcctn_quant` conda environment (PyTorch 2.2.0, CUDA 12.4, Python 3.9)
- Exploratory; main quantization work has moved to `BCCTN_sep_conv_causal_slide_win_quant`
