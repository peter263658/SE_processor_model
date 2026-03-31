# BCCTN_sep_conv_causal_slide_win_quant_256dim — Quantized, 256-dim Embedding

Reduced embedding dimension (256 instead of 512) for further memory and compute savings, combined with PTQ.

See the [root README](../README.md) for full setup, training, and evaluation instructions.

## Variant-specific notes

- Based on `BCCTN_sep_conv_causal_slide_win_quant`
- `model.embed_dim = 256` (halved from the default 512)
- Explores accuracy vs. model size trade-off
