# Agent Instructions: 02 — Weight Bit-width Exploration (FP vs INT, Layer-wise)

## Goal

Perform layer-by-layer quantization sensitivity analysis to determine the optimal bit-width (W8/W4/W3) and number format (FP vs INT) for each layer, with the goal of minimizing DMA traffic while preserving speech quality (SNR, MBSTOI, ILD, IPD).

## Background

- BCCTN is a binaural speech enhancement model with 6-stage complex encoder-decoder + MHA.
- The hardware target has 256 MACs and is DMA-bound — reducing weight bit-width directly reduces DMA time.
- Depthwise conv layers have very few parameters per filter (kernel_size=5), making them more sensitive to quantization.
- MHA projection matrices (Wq/Wk/Wv/Wo) are 256×256 each, large enough to tolerate lower bit-width.
- The model code is under `DCNN/models/model.py`.
- Trained checkpoint: `model.ckpt`.
- Evaluation script: `eval.py` (computes SNR, MBSTOI, ILD error, IPD error).

## What to Produce

### Script: `bitwidth_sweep.py`

A Python script that:

1. **Loads the trained model** from `model.ckpt`.
2. **Implements PTQ (Post-Training Quantization)** for each layer independently:
   - **INT quantization**: symmetric min-max scaling → round → clamp to [-2^(b-1), 2^(b-1)-1]
   - **FP quantization**: simulate reduced FP formats:
     - FP8 (E4M3, E5M2) — already done in training
     - FP4 (E2M1) — similar to NVFP4
   - Bit-widths to test: **8, 4, 3** bits
3. **Layer-wise sensitivity analysis**:
   - For each layer, quantize ONLY that layer to target bit-width while keeping all others at fp8.
   - Run inference on a small eval set (e.g., 50 samples from test set).
   - Record: ΔSI-SNR, ΔMBSTOI, ΔILD, ΔIPD relative to fp8 baseline.
4. **Mixed-precision recommendation**:
   - Based on sensitivity results, assign each layer to a precision tier:
     - Tier A (must keep W8): layers where W4 degrades quality > threshold
     - Tier B (safe for W4): layers tolerant to 4-bit
     - Tier C (can try W3): very robust layers
   - Proposed thresholds: ΔSI-SNR < 0.5dB, ΔMBSTOI < 0.02, ΔILD < 0.5dB, ΔIPD < 5°

### Script: `mixed_precision_eval.py`

Apply the recommended mixed-precision config and run full evaluation:
- Conv depthwise layers → W8 (expected sensitive)
- Conv pointwise layers → W4
- MHA projections (Wq/Wk/Wv/Wo) → W4
- Compare against uniform W8 and uniform W4 baselines

### Expected Output Files
- `bitwidth_sweep.py` — PTQ sweep script
- `mixed_precision_eval.py` — full eval with mixed precision
- `results/sensitivity_matrix.csv` — layer × bitwidth sensitivity data
- `results/sensitivity_heatmap.png` — heatmap visualization (layer vs bit-width, color = quality degradation)
- `results/mixed_precision_config.json` — recommended per-layer bit-width assignment
- `results/mixed_precision_eval.txt` — full eval results with mixed precision

## Model Config

```python
win_len=128, win_inc=80, fft_len=128
kernel_size=5, kernel_num=[16, 32, 64, 128, 256, 256]
embed_dim=512, num_heads=32, attn_win=31
use_depthwise=True, binaural=True
```

## Layer Naming Convention in Checkpoint

The model has these parameter groups (inspect via `model.state_dict().keys()`):
- `encoder.model.{0-5}.0.depthwise_conv_{real,imag}.weight` — depthwise conv
- `encoder.model.{0-5}.0.pointwise_conv_{real,imag}.weight` — pointwise conv
- `encoder.model.{0-5}.1.*` — BatchNorm
- `decoder.model.{0-5}.0.*` — transposed conv (depthwise + pointwise)
- `mattn.proj_in.*` — linear projection in
- `mattn.mha.*` — multi-head attention (in_proj_weight, out_proj.weight)
- `mattn.proj_out.*` — linear projection out

## Quantization Implementation Notes

```python
# INT symmetric quantization
def quantize_int(tensor, bits):
    qmax = 2**(bits-1) - 1
    scale = tensor.abs().max() / qmax
    q = torch.round(tensor / scale).clamp(-qmax, qmax)
    return q * scale  # dequantized

# FP4 simulation (E2M1)
def quantize_fp4(tensor):
    # representable values: ±{0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}
    # scale tensor to fit, then snap to nearest representable
    ...
```

## Key Questions to Answer

1. Which layers are most sensitive to quantization? (expect depthwise conv)
2. Can MHA projection matrices safely go to W4 or even W3?
3. What is the total weight memory savings of the recommended mixed-precision config vs uniform W8?
4. How much does ILD/IPD degrade? (spatial cue preservation is critical for binaural)
5. Is INT4 or FP4 better for each layer type?

## Environment

```bash
mamba activate bcctn
# Eval dataset paths (already configured in config/):
# Test noisy: /raid/R12K41024/BCCTN/Dataset/noisy_testset/
# Test clean: /raid/R12K41024/BCCTN/Dataset/clean_testset/
python bitwidth_sweep.py --checkpoint model.ckpt --num_samples 50
python mixed_precision_eval.py --checkpoint model.ckpt
```

## Deliverable Checklist
- [ ] `bitwidth_sweep.py` runs per-layer PTQ sweep
- [ ] Sensitivity matrix CSV with all layers × bit-widths
- [ ] Heatmap visualization
- [ ] Mixed-precision config JSON
- [ ] Full eval with mixed-precision vs baselines
- [ ] Memory savings calculation (bytes)
