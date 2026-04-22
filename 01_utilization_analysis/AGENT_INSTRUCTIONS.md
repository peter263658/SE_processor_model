# Agent Instructions: 01 — MAC Utilization & Pipeline Bottleneck Analysis

## Goal

Quantify the actual MAC utilization of the BCCTN accelerator (256 MACs) at **per-layer granularity** for both Conv and Attention stages. Identify where MAC idle cycles occur and why.

## Background

- This is a binaural speech enhancement model (BCCTN) targeting ASIC deployment with 256 MAC units.
- Architecture: 6-stage complex encoder-decoder (depthwise separable conv) + MHA bottleneck.
- Known from prior profiling: Conv stage ~0.3ms, Attn stage ~2.9ms, DMA:compute ratio ≈ 5.8:1.
- The hardware uses weight-stationary dataflow. WBUF = 16 Banks × 64-bit × 256 Depth = 64KB.
- The model code is in this directory under `DCNN/models/model.py` and `DCNN/models/binaural_attention_model.py`.
- A trained checkpoint exists at `model.ckpt`.
- An existing profiler is at `profile_compare.py` — use it as reference but build a new analysis script.

## What to Produce

### Script: `utilization_profiler.py`

Write a Python script that:

1. **Loads the trained model** from `model.ckpt` (PyTorch Lightning checkpoint, extract `state_dict`).
2. **For each layer**, compute:
   - **Theoretical MAC cycles** = total_MACs / 256 (our PE array size)
   - **DMA cycles** = weight_bytes / bandwidth (assume 64-bit bus @ 1 cycle/transfer = 8 bytes/cycle)
   - **Activation DMA** = input_activation_bytes + output_activation_bytes (assume fp8 = 1 byte per element)
   - **MAC utilization** = compute_cycles / (compute_cycles + max(weight_dma_cycles, activation_dma_cycles))
   - For depthwise conv layers: note that utilization is inherently low because there's no cross-channel reduction — each of the 256 MACs can only work on 1 channel at a time, but depthwise kernel_size=5 means only 5 MACs are useful per channel tile.
   - For pointwise (1×1) conv layers: utilization should be high since Cin×Cout parallelism maps well to 256 MACs.
3. **For MHA layers**, break down into sub-operations:
   - Wq/Wk/Wv/Wo projections (matrix multiply): compute_cycles vs weight DMA
   - QK^T computation
   - Softmax (non-MAC, estimate LUT cycles)
   - Score × V multiply
   - Note: complex MHA runs 4 real passes (re×re, re×im, im×re, im×im)
4. **Output a summary table** (CSV + printed) with columns:
   `layer_name | type | params | MACs | compute_cycles | weight_dma_cycles | act_dma_cycles | utilization(%) | bottleneck(compute/dma)`

### Expected Output Files
- `utilization_profiler.py` — the analysis script
- `results/utilization_table.csv` — per-layer utilization data
- `results/utilization_summary.txt` — key findings and bottleneck identification
- `results/utilization_plot.png` — bar chart of per-layer utilization (compute vs DMA)

## Hardware Assumptions (use these constants)

```python
NUM_MACS = 256          # PE array size
BUS_WIDTH_BYTES = 8     # 64-bit bus
WEIGHT_PRECISION = 1    # fp8 = 1 byte per weight
ACT_PRECISION = 1       # fp8 = 1 byte per activation
CLOCK_FREQ_MHZ = 200    # for time conversion (optional)
WBUF_SIZE_KB = 64       # on-chip weight buffer
ABUF_SIZE_KB = 32       # on-chip activation buffer (estimate)
```

## Model Config

```python
win_len=128, win_inc=80, fft_len=128
kernel_size=5, kernel_num=[16, 32, 64, 128, 256, 256]
embed_dim=512, num_heads=32, attn_win=31
use_depthwise=True, binaural=True
```

Input: 1-second stereo audio @ 16kHz → T=16000 samples → ~157 frames after STFT.

## Key Questions to Answer

1. Which layers have the lowest MAC utilization and why?
2. What fraction of total cycles is spent on DMA vs compute?
3. For depthwise conv: what is the effective utilization given kernel_size=5?
4. Is the MHA bottleneck primarily DMA (weight loading) or compute (QK^T matmul)?
5. If we switch from fp8 to W4A8, how does utilization change? (add a W4 column)

## Environment

```bash
mamba activate bcctn
python utilization_profiler.py
```

## Deliverable Checklist
- [ ] `utilization_profiler.py` runs without error
- [ ] CSV output with all layers
- [ ] Bar chart visualization
- [ ] Summary text with top-3 bottleneck layers identified
- [ ] W4A8 comparison column included
