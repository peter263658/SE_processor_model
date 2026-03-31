# BCCTN_sep_conv_causal_slide_win_quant_256dim — FP8 QAT, 256-dim Embedding

Reduced embedding dimension (256 instead of 512) combined with FP8 QAT for further memory and compute savings.

See the [root README](../README.md) for full setup, training, and evaluation instructions.

## Variant-specific notes

- Based on `BCCTN_sep_conv_causal_slide_win_quant`
- `model.embed_dim = 256` (halved from the default 512)
- `training.fp8_qat = true`, `training.fp8_dtype = "e4m3"`
- Explores accuracy vs. model size trade-off under QAT
