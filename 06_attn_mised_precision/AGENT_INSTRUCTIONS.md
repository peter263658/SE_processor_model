# Agent Instructions: 06 — Attention Mixed-Precision & Density Analysis

## Goal

Analyze the MHA (Multi-Head Attention) block to determine optimal per-sub-operation precision and whether sparse/top-k attention can replace dense attention without quality loss.

## Background

- BCCTN uses a single-layer MHA with embed_dim=512, num_heads=32, sliding window W=31.
- The MHA operates on complex-valued data: it runs 4 real passes (re×re, re×im, im×re, im×im).
- Sub-operations have different numerical requirements:
  - **QK^T (outer product)**: largest dynamic range, needs high-precision accumulation
  - **Softmax**: non-linear, uses LUT on hardware
  - **Score × V**: precision requirements are lower (scores are [0,1] after softmax)
  - **Projection weights (Wq/Wk/Wv/Wo)**: matrix multiply, potentially compressible to W4
- Model code: `DCNN/models/model.py` and `DCNN/models/binaural_attention_model.py`.
- Checkpoint: `model.ckpt`.

## What to Produce

### Script: `attn_precision_analyzer.py`

A Python script that:

1. **Dynamic range analysis per sub-operation**:
   - Hook into MHA to capture intermediate values during inference:
     - Q, K, V tensors (after projection)
     - QK^T scores (before softmax)
     - Attention weights (after softmax)
     - Weighted output (Score × V)
   - For each, compute: min, max, mean, std, dynamic range, histogram
   - Run on ~20 test samples, aggregate statistics.

2. **Precision sensitivity per sub-operation**:
   - Simulate reduced precision at each stage independently:
     - Q/K: try fp8, fp4, int8, int4
     - V: try fp8, fp4, int8, int4
     - QK^T accumulator: try fp32, fp16, int16 (accumulation precision)
     - Score×V: try fp8, fp4
   - For each combination, measure output quality (compare against fp32 reference):
     - MSE of MHA output
     - If possible, run full model eval for the most promising configs
   - Identify the **minimum precision** at each stage that keeps MSE < threshold.

3. **Attention score density analysis**:
   - With sliding window W=31, the attention matrix is already (31/S)-dense where S≈157.
   - Within the non-zero window of 31 positions:
     - What is the average entropy of the attention distribution? (low entropy = peaky = sparse)
     - What fraction of scores are > 0.05? (effective attention span)
     - If we keep only top-k scores (k={4, 8, 16}), what is the approximation error?
   - **Per-head analysis**: do all 32 heads have similar attention patterns?
     - Some heads may be "sharp" (attend to 1-2 positions) vs "flat" (uniform attention)
     - Sharp heads can use very small k; flat heads need larger k or full window

4. **Top-k sparse attention simulation**:
   - Replace softmax output with top-k masking (keep k largest, zero out rest, renormalize)
   - For k ∈ {4, 8, 12, 16, 24, 31}:
     - Compute output MSE vs full attention
     - Run full model eval on test set (SI-SNR, MBSTOI, ILD, IPD)
   - Find the minimum k that preserves quality within thresholds.

5. **Mixed-precision recommendation**:
   - Propose a concrete precision assignment:
     - Wq/Wk/Wv/Wo weight precision
     - Q/K/V activation precision
     - QK^T accumulator precision
     - Score×V precision
   - Estimate hardware savings: reduced multiplier bit-width → smaller PE area and power.

### Expected Output Files
- `attn_precision_analyzer.py` — main analysis script
- `results/dynamic_range_per_subop.csv` — statistics for Q, K, V, QK^T, scores, output
- `results/precision_sensitivity.csv` — MSE at each precision combo
- `results/attention_entropy.csv` — per-head entropy and effective span
- `results/attention_pattern_heatmap.png` — average attention pattern (all heads)
- `results/per_head_patterns.png` — individual head attention patterns (grid)
- `results/topk_eval.csv` — quality metrics vs k
- `results/topk_plot.png` — quality vs k curves
- `results/precision_recommendation.json` — recommended per-sub-op precision
- `results/summary.md` — executive summary

## Model Config

```python
win_len=128, win_inc=80, fft_len=128
kernel_size=5, kernel_num=[16, 32, 64, 128, 256, 256]
embed_dim=512, num_heads=32, attn_win=31
use_depthwise=True, binaural=True
```

## MHA Architecture Detail

The complex MHA in `binaural_attention_model.py`:
- Input: (B, T, flatten_size) where flatten_size = kernel_num[-1]//2 * (fft_len // 2^6) * 2 (binaural)
- proj_in: Linear(flatten_size → embed_dim), complex
- mha: nn.MultiheadAttention(embed_dim, num_heads), run as complex (4 real passes)
- proj_out: Linear(embed_dim → flatten_size), complex
- Window attention: mask out positions > W/2 = 15 from diagonal

## Test Data

- Noisy test: `/raid/R12K41024/BCCTN/Dataset/noisy_testset/`
- Clean test: `/raid/R12K41024/BCCTN/Dataset/clean_testset/`

## Key Questions to Answer

1. What is the minimum precision for QK^T accumulation without quality loss?
2. Can V be quantized more aggressively than Q/K?
3. What is the effective attention span within the W=31 window? (how many positions actually matter?)
4. Can we reduce to top-8 or top-16 attention without degradation?
5. Are there "unimportant" heads that could be pruned or use lower precision?
6. What is the total hardware area/power reduction from the recommended mixed-precision config?

## Environment

```bash
mamba activate bcctn
python attn_precision_analyzer.py --checkpoint model.ckpt --test_dir /raid/R12K41024/BCCTN/Dataset/noisy_testset/ --clean_dir /raid/R12K41024/BCCTN/Dataset/clean_testset/ --num_samples 20
```

## Deliverable Checklist
- [ ] Per-sub-operation dynamic range statistics
- [ ] Precision sensitivity matrix
- [ ] Attention entropy and effective span analysis
- [ ] Per-head attention pattern visualization
- [ ] Top-k sparse attention evaluation
- [ ] Mixed-precision recommendation JSON
- [ ] Executive summary with hardware implications
