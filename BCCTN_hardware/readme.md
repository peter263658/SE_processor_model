# BCCTN_hardware — Hardware Deployment

Scripts and utilities for exporting the quantized model to hardware-friendly formats (weight dumps, activation patterns, PTQ calibration).

See the [root README](../README.md) for full setup, training, and evaluation instructions.

## Variant-specific notes

- `ptq_export.py` — runs post-training quantization and dumps weights/activations to `artifacts/`
- `frame_runner.py` — frame-by-frame streaming inference for latency measurement
- `dump_utils.py` / `quant_utils.py` — helper utilities for calibration and export
- `artifacts/` directory is excluded from git (contains large binary dumps)

## Key bottlenecks (hardware target)

1. Complex 5×1 convolutions (>80% of FLOPs)
2. Attention softmax over S×W (S ≈ 157 frames/s, W = 31 window)
3. STFT/iSTFT FFT operations
