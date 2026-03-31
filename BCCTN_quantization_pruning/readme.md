# BCCTN_quantization_pruning — Pruning + Quantization

Combined weight pruning and quantization experiments to study sparsity and bit-width reduction jointly.

See the [root README](../README.md) for full setup, training, and evaluation instructions.

## Variant-specific notes

- Applies magnitude-based weight pruning before or alongside quantization
- Analyzes weight sparsity distribution across encoder/decoder layers
- Uses `bcctn_quant` conda environment
