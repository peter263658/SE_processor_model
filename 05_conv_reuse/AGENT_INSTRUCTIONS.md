# Agent Instructions: 05 — Conv Weight Reuse Optimization

## Goal

Analyze opportunities for weight reuse in the convolutional layers to reduce DMA traffic and on-chip memory requirements. Focus on **cross-layer weight similarity** and **temporal activation reuse**.

## Background

- BCCTN has 6-stage encoder + 6-stage decoder, each with depthwise separable complex conv.
- L/R channels already share the same encoder/decoder weights (binaural shared weights) — this is an existing reuse.
- Hardware is DMA-bound: any reduction in unique weight data or redundant computation helps.
- Conv stride=(2,1) in frequency, stride=1 in time → adjacent frames share input activations.
- Model code: `DCNN/models/model.py`, checkpoint: `model.ckpt`.

## What to Produce

### Script: `reuse_analyzer.py`

A Python script that:

1. **Cross-layer weight similarity**:
   - For each pair of adjacent encoder layers (e.g., Enc-3 vs Enc-4, Enc-5 vs Enc-6):
     - Compute cosine similarity of pointwise conv weights (flatten to 1D vectors)
     - Compute L2 distance normalized by tensor norm
     - Do SVD on each weight matrix, compare singular value spectra
   - For encoder-decoder mirror pairs (Enc-k vs Dec-k):
     - Same similarity metrics
     - Are they approximately transposes of each other? (for transposed conv)
   - If any pair has cosine similarity > 0.9: flag as weight-sharing candidate.

2. **Low-rank factorization potential**:
   - For each pointwise conv weight (Cout, Cin, 1, 1) and MHA projection:
     - Compute SVD, plot singular value decay
     - Calculate: what rank r captures 95%, 99% of the Frobenius norm?
     - If rank r << min(Cin, Cout): the layer can be factored into two smaller layers (Cin→r, r→Cout), reducing both params and DMA.
   - Estimate memory savings: original = Cin×Cout bytes, factored = (Cin×r + r×Cout) bytes.

3. **Temporal activation reuse analysis**:
   - The model processes frames with hop=80 samples, window=128 → 60% overlap between consecutive frames.
   - For each Conv layer with stride=(2,1) in freq:
     - How many input activation rows are reused from the previous frame?
     - What is the theoretical DMA savings if we cache and reuse these rows?
   - Compute: `reuse_ratio = reused_elements / total_input_elements` per layer.
   - Estimate: required cache size (in bytes) to hold reusable activations.

4. **Depthwise vs pointwise reuse**:
   - Depthwise filters are tiny (5 weights per channel) — they fit in registers, DMA overhead is per-invocation scheduling, not bandwidth.
   - Pointwise filters are the heavy part — quantify how much of total weight DMA is pointwise.

### Expected Output Files
- `reuse_analyzer.py` — main analysis script
- `results/cross_layer_similarity.csv` — pairwise similarity metrics
- `results/low_rank_analysis.csv` — per-layer SVD rank at 95%/99% energy
- `results/singular_values/` — singular value decay plots per layer
- `results/temporal_reuse.csv` — per-layer activation reuse ratios
- `results/cache_requirements.csv` — required cache size for temporal reuse
- `results/summary.md` — findings and recommendations

## Model Config

```python
win_len=128, win_inc=80, fft_len=128
kernel_size=5, kernel_num=[2, 16, 32, 64, 128, 256, 256]  # note: [2] is prepended as input channels
embed_dim=512, num_heads=32, attn_win=31
use_depthwise=True, binaural=True
```

## Layer Weight Shapes (depthwise separable)

For encoder stage k (0-indexed), channels go kernel_num[k]→kernel_num[k+1]:
- Depthwise real: (Cout/2, 1, 5, 1) — very small
- Pointwise real: (Cout/2, Cin/2, 1, 1) — this is the bulk of parameters
- Same for imag parts
- Decoder mirrors encoder in reverse with transposed conv

## Key Questions to Answer

1. Are any adjacent layer pairs similar enough for weight sharing? (cosine sim > 0.9)
2. Can encoder/decoder mirror pairs share weights (like U-Net tied weights)?
3. What is the effective rank of pointwise conv and MHA projections? Is low-rank factorization worth it?
4. How much activation data is reusable between consecutive frames?
5. What on-chip cache size is needed to exploit temporal reuse?
6. What fraction of total weight DMA is pointwise conv vs depthwise?

## Environment

```bash
mamba activate bcctn
python reuse_analyzer.py --checkpoint model.ckpt
```

## Deliverable Checklist
- [ ] Cross-layer similarity matrix
- [ ] Low-rank SVD analysis with plots
- [ ] Temporal activation reuse ratios
- [ ] Cache size requirements
- [ ] Actionable summary with ROI estimates
