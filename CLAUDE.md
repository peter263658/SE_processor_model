# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Implementation of **Binaural Speech Enhancement using Deep Complex Convolutional Transformer Networks** (ICASSP 2024). The model processes stereo audio to suppress noise while preserving spatial cues (ILD and IPD).

## Repository Structure

There are multiple variant directories, each a self-contained experiment:

| Directory | Description |
|---|---|
| `BCCTN/` | Baseline implementation |
| `BCCTN_sep_conv/` | Separable convolution variant |
| `BCCTN_sep_conv_causal/` | Causal (streaming) variant |
| `BCCTN_sep_conv_causal_slide_win/` | Causal + sliding window — **current focus** |
| `BCCTN_sep_conv_causal_slide_win_quant/` | Quantized variant |
| `BCCTN_sep_conv_causal_slide_win_relu/` | ReLU activation variant |
| `BCCTN_quantization_pruning/` | Pruning + quantization experiments |
| `BCCTN_hardware/` | Hardware implementation notes |
| `Dataset/` | Shared dataset root |

Within each variant, the internal structure is consistent:
- `DCNN/models/model.py` — main encoder-decoder architecture
- `DCNN/loss.py` — combined SNR + STOI + ILD + IPD loss
- `DCNN/trainer.py` — PyTorch Lightning `LightningModule`
- `DCNN/feature_extractors.py` — STFT/iSTFT wrappers
- `config/config.yaml` — master Hydra config (merges `model.yaml`, `training.yaml`, `dataset/`)
- `train.py` — training entry point (Hydra)
- `eval.py` — evaluation entry point

## Environment Setup

Use `mamba` instead of `conda` for significantly faster dependency solving.

```bash
# Main environment (PyTorch 2.3.1, CUDA 12.1, Python 3.10)
mamba env create -f bcctn.yml
mamba activate bcctn

# Quantization-specific environment (PyTorch 2.2.0, CUDA 12.4, Python 3.9)
mamba env create -f bcctn_quant.yml
mamba activate bcctn_quant
```

## Common Commands

All commands run from inside the variant directory (e.g., `cd BCCTN_sep_conv_causal_slide_win`).

**Training:**
```bash
python train.py                          # uses config/ via Hydra
HYDRA_FULL_ERROR=1 python train.py       # verbose error output
CUDA_VISIBLE_DEVICES=1 python train.py
bash run_train.sh                        # wrapper script
```

**Evaluation:**
```bash
bash run_eval.sh                         # uses paths in the script

# Or directly:
CUDA_VISIBLE_DEVICES=0 python eval.py \
    --model_checkpoint "outputs/.../checkpoints/epoch=X-step=Y.ckpt" \
    --vctk_test_dir   "./Dataset/noisy_testset" \
    --vctk_clean_dir  "./Dataset/clean_testset" \
    --timit_test_dir  "./Dataset/noisy_testset_timit" \
    --timit_clean_dir "./Dataset/clean_testset_timit" \
    --output_dir      "./results" \
    --paper_style_eval \
    --batch_size 8
```

**Data Preparation:**
```bash
python prepare_data.py \
    --clean_dir /path/to/vctk \
    --noise_dir /path/to/noisex92 \
    --hrir_path /path/to/hrir \
    --output_dir ./Dataset \
    --dataset_type vctk   # or timit
bash run_dataset.sh       # convenience wrapper
```

**Benchmarking:**
```bash
python benchmark_bcctn.py
bash run_benchmark.sh
```

## Key Hyperparameters (config/config.yaml)

Current canonical config (newest variants):

| Parameter | Current value | Notes |
|---|---|---|
| `model.kernel_num` | `[16,32,64,128,256,256]` | 6-stage encoder/decoder |
| `model.kernel_size` | 5 | Conv kernel size |
| `model.use_depthwise` | `true` | Depthwise separable convolution |
| `model.embed_dim` | 512 | Transformer embedding dim |
| `model.num_heads` | 32 | Attention heads |
| `model.attn_win` | 31 | Sliding attention window size |
| `model.win_len` | 128 | STFT window (current); 256 and 400 also tested |
| `model.win_inc` | 80 | STFT hop |
| `model.fft_len` | 128 | FFT size (current); 256 and 512 also tested |
| `model.rnn_layers` | 2 | CLSTM layers |
| `model.rnn_units` | 128 | CLSTM hidden units |
| `model.ild_weight` | 1 | ILD loss weight (γ) |
| `model.ipd_weight` | 10 | IPD loss weight (κ) |
| `model.snr_loss_weight` | 1 | SNR loss weight (α) |
| `model.stoi_weight` | 10 | STOI loss weight (β) |
| `training.batch_size` | 32 | |
| `training.n_epochs` | 150 | |
| `training.learning_rate` | 0.001 | Decays ×0.5 at steps [3, 8] |
| `training.fp8_qat` | `true` | FP8 QAT (e4m3 or e5m2) |

## Architecture Notes

- **Encoder-decoder** with 6 complex convolutional stages; each stage uses `ComplexConv2d` from the `complexpytorch` package.
- **Bottleneck**: Multi-head self-attention transformer over the frequency-time sequence.
- **Masks**: Individual complex ratio masks for left and right channels.
- **Loss**: `L = SNR + 10×STOI + ILD + 10×IPD`
- **Causal variants** replace standard convolution with causal padding; sliding-window variants chunk the time axis for low-latency streaming.

## Known Bottlenecks (hardware target)

1. Complex 5×1 convolutions (>80% of FLOPs)
2. Attention softmax over S×S (S ≈ 157 for 1-second audio at default hop)
3. STFT/iSTFT FFT operations

## Current Experiments (see current_progress.md)

- Reducing `kernel_size` from `(k, 2)` → `(k, 1)`
- Reducing `fft_len` from 256 → 128 (separable conv variants)
- Causal attention with sliding window
- Post-training quantization and weight sparsity analysis
