# Agent Instructions: 03 — Weight Dynamic Range Analysis

## Goal

Analyze the dynamic range of every layer's weights at **per-layer, per-channel, and per-head granularity** to determine the optimal number format (FP vs INT) and scalar factor group size for hardware quantization.

## Background

- BCCTN binaural speech enhancement model: 6-stage complex depthwise separable conv encoder-decoder + MHA.
- Hardware target: 256 MACs, weight-stationary, considering W4A8 with per-group scalar factor.
- If weights have concentrated distribution (low kurtosis) → INT quantization suffices.
- If weights have long tails or outliers → need FP format or fine-grained per-channel/per-group scaling.
- The scalar factor group_size (e.g., 32, 64, 128) directly affects hardware overhead. The dynamic range analysis tells us what group_size is needed.
- Model code: `DCNN/models/model.py`, checkpoint: `model.ckpt`.

## What to Produce

### Script: `dynamic_range_analyzer.py`

A Python script that:

1. **Loads the trained model** checkpoint.
2. **For each weight tensor**, compute:
   - Global statistics: min, max, mean, std, kurtosis, skewness
   - Dynamic range ratio: max(|w|) / min(|w| > threshold) where threshold = 1e-6
   - **Per-output-channel analysis** (for Conv layers):
     - Each output channel's {min, max, std, kurtosis}
     - Cross-channel variance of these statistics (how different are channels from each other?)
   - **Per-head analysis** (for MHA):
     - Wq/Wk/Wv split into num_heads=32 chunks
     - Each head's {min, max, std, kurtosis}
     - Cross-head variance
3. **Histogram visualization** for each layer:
   - Overall weight distribution histogram
   - Overlay per-channel distributions (use alpha transparency)
   - Mark outlier threshold at ±3σ
4. **Group-size sensitivity analysis**:
   - For group_size ∈ {16, 32, 64, 128, 256}:
     - Compute per-group scale = max(|group|)
     - Compute quantization error (MSE) when quantizing to 4-bit INT with that group scale
     - Compare against per-tensor scale baseline
   - This tells us the optimal group_size for the hardware scalar factor.
5. **FP vs INT recommendation per layer**:
   - If kurtosis > 5 (heavy-tailed): recommend FP
   - If kurtosis < 3 and cross-channel variance < threshold: recommend INT with per-tensor scale
   - Otherwise: recommend INT with per-channel or per-group scale

### Expected Output Files
- `dynamic_range_analyzer.py` — main analysis script
- `results/dynamic_range_table.csv` — per-layer statistics table
- `results/per_channel_variance.csv` — cross-channel variance for each Conv layer
- `results/per_head_variance.csv` — cross-head variance for MHA layers
- `results/histograms/` — directory with per-layer histogram PNGs
- `results/group_size_sensitivity.csv` — quantization MSE vs group_size per layer
- `results/group_size_plot.png` — plot of quantization error vs group_size
- `results/format_recommendation.json` — per-layer FP/INT + group_size recommendation
- `results/summary.md` — executive summary with key findings

## Model Config

```python
win_len=128, win_inc=80, fft_len=128
kernel_size=5, kernel_num=[16, 32, 64, 128, 256, 256]
embed_dim=512, num_heads=32, attn_win=31
use_depthwise=True, binaural=True
```

## Layer Structure

Complex layers store weights as separate real/imag:
- `encoder.model.{0-5}.0.depthwise_conv_real.weight` — shape: (Cout, 1, K, 1)
- `encoder.model.{0-5}.0.depthwise_conv_imag.weight`
- `encoder.model.{0-5}.0.pointwise_conv_real.weight` — shape: (Cout, Cin, 1, 1)
- `encoder.model.{0-5}.0.pointwise_conv_imag.weight`
- `mattn.mha.in_proj_weight` — shape: (3*embed_dim, embed_dim), complex64
- `mattn.mha.out_proj.weight` — shape: (embed_dim, embed_dim), complex64
- `mattn.proj_in.weight`, `mattn.proj_out.weight` — linear projections

## Key Questions to Answer

1. Are weight distributions similar across channels/heads, or do some channels have much larger dynamic range? (determines per-channel vs per-tensor scaling)
2. What is the optimal group_size for W4 scalar factor? (target: <10% overhead → group_size ≥ 16)
3. Which layers have outliers that would benefit from FP format?
4. Do real and imaginary weight components have similar distributions, or should they be scaled separately?
5. For MHA: do all 32 heads have similar weight magnitudes, or are some heads much larger? (if so, per-head scaling is needed)

## Environment

```bash
mamba activate bcctn
python dynamic_range_analyzer.py --checkpoint model.ckpt
```

## Deliverable Checklist
- [ ] Per-layer statistics CSV
- [ ] Per-channel and per-head variance analysis
- [ ] Histogram plots for all layers
- [ ] Group-size sensitivity analysis with plot
- [ ] FP/INT format recommendation JSON
- [ ] Summary document with actionable conclusions
