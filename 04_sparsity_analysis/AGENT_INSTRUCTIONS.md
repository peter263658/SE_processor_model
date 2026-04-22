# Agent Instructions: 04 — Zero/Sparsity Structure Analysis

## Goal

Analyze the sparsity patterns of weights and activations to determine if **structured sparsity** (hardware-skippable) exists, and at what threshold. Distinguish structured vs random sparsity — only structured sparsity is cost-effective for hardware acceleration.

## Background

- BCCTN binaural speech enhancement model targeting ASIC with 256 MACs.
- **Structured sparsity** (entire channel/head/tile is zero) → hardware can skip whole tiles (free speedup).
- **Random/unstructured sparsity** → needs bitmap + compressed format, hardware overhead is high and usually not worth it.
- The sliding window attention (W=31) is already a form of structured sparsity in the attention matrix.
- Model code: `DCNN/models/model.py`, checkpoint: `model.ckpt`.
- There are existing sparsity scripts (`axis_sparsity_analyzer.py`, `sparsity_results/`, `sparsity_plots/`) — review them first, but produce a comprehensive new analysis.

## What to Produce

### Script: `sparsity_analyzer.py`

A Python script that:

1. **Weight sparsity analysis**:
   - For each layer, at multiple thresholds: |w| < {1e-4, 1e-3, 1e-2, 5e-2, 1e-1}
   - Compute:
     - Overall zero ratio (fraction of weights below threshold)
     - **Channel-level sparsity**: fraction of output channels where ALL weights < threshold (structured!)
     - **Row/column sparsity**: for MHA projection matrices, fraction of rows/columns that are all-zero
     - **Block sparsity**: for block_size ∈ {4, 8, 16, 32}, fraction of blocks where all elements < threshold
   - Classify each layer: "dense" (<10% sparse), "random sparse" (>10% sparse but no structure), "structured sparse" (whole channels/blocks are zero)

2. **Activation sparsity analysis** (requires inference):
   - Register forward hooks on each layer to capture activations.
   - Run inference on ~20 test samples.
   - For each layer's output activation:
     - Compute average sparsity at thresholds: |a| < {1e-3, 1e-2, 1e-1}
     - Check if sparsity is structured: are there channels that are consistently near-zero across samples?
     - ReLU/PReLU outputs naturally have zeros — quantify this.

3. **Attention score sparsity**:
   - Hook into the MHA layer to capture attention weights (after softmax).
   - With sliding window W=31, most of the S×S matrix is already zero (structured by design).
   - Within the non-zero window: what fraction of scores are < 0.01? (near-zero after softmax)
   - Visualize: average attention pattern heatmap across test samples.
   - Can we reduce window from W=31 to W=16 or W=8 without quality loss?

4. **Structured vs random classification**:
   - For each layer, produce a verdict:
     - "No exploitable sparsity" — dense, skip in hardware
     - "Block-sparse at block=N, threshold=T" — can use block-sparse compute
     - "Channel-prunable: N channels removable" — can structurally prune
     - "Head-prunable: N heads removable" — for MHA

### Expected Output Files
- `sparsity_analyzer.py` — main script
- `results/weight_sparsity.csv` — per-layer weight sparsity at each threshold
- `results/activation_sparsity.csv` — per-layer activation sparsity
- `results/block_sparsity.csv` — block-level sparsity for various block sizes
- `results/attention_sparsity.csv` — attention score statistics
- `results/attention_heatmap.png` — average attention pattern visualization
- `results/sparsity_summary.json` — per-layer structured/random classification
- `results/pruning_candidates.txt` — list of channels/heads that could be pruned
- `results/weight_sparsity_plot.png` — bar chart per layer
- `results/summary.md` — key findings

## Model Config

```python
win_len=128, win_inc=80, fft_len=128
kernel_size=5, kernel_num=[16, 32, 64, 128, 256, 256]
embed_dim=512, num_heads=32, attn_win=31
use_depthwise=True, binaural=True
```

## Important Notes

- Complex layers have separate real/imag weights — analyze BOTH and report separately.
- A channel is "prunable" only if BOTH real and imag parts are near-zero.
- For depthwise conv: "channel sparsity" means a single filter (5 weights) is all near-zero.
- For activation analysis, use test data from: `/raid/R12K41024/BCCTN/Dataset/noisy_testset/`
- Existing results in `sparsity_results/` and `sparsity_plots/` — compare your findings against them.

## Key Questions to Answer

1. What percentage of weights are "near-zero" at practical thresholds (1e-2, 1e-1)?
2. Is the sparsity structured (whole channels/heads) or random?
3. Are there attention heads that contribute negligibly? (pruning candidates)
4. How sparse are activations after PReLU? Is there channel-wise structure?
5. Can the attention window be reduced from W=31? What's the effective window?

## Environment

```bash
mamba activate bcctn
python sparsity_analyzer.py --checkpoint model.ckpt --test_dir /raid/R12K41024/BCCTN/Dataset/noisy_testset/ --num_samples 20
```

## Deliverable Checklist
- [ ] Weight sparsity at multiple thresholds
- [ ] Block-level sparsity analysis
- [ ] Activation sparsity with forward hooks
- [ ] Attention score heatmap and sparsity stats
- [ ] Structured vs random classification per layer
- [ ] Pruning candidate list (channels + heads)
