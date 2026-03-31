# BCCTN_sep_conv_causal_slide_win_quant — Quantized (PTQ)

Post-training quantization applied to the causal sliding-window model.

See the [root README](../README.md) for full setup, training, and evaluation instructions.

## Variant-specific notes

- Based on `BCCTN_sep_conv_causal_slide_win`
- Uses `bcctn_quant` conda environment (PyTorch 2.2.0, CUDA 12.4, Python 3.9)
- Quantization config in `config/config.yaml` (`training.fp8_qat`)
