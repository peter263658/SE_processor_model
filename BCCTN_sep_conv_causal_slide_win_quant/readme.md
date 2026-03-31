# BCCTN_sep_conv_causal_slide_win_quant — FP8 QAT

FP8 Quantization-Aware Training (QAT) applied to the causal sliding-window model.

See the [root README](../README.md) for full setup, training, and evaluation instructions.

## Variant-specific notes

- Based on `BCCTN_sep_conv_causal_slide_win`
- `training.fp8_qat = true` — FP8 QAT enabled during training
- `training.fp8_dtype = "e4m3"` — uses e4m3 format (can switch to e5m2)
- Uses `bcctn_quant` conda environment (PyTorch 2.2.0, CUDA 12.4, Python 3.9)
