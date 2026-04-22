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

## W4A8 Quantization Analysis (Advisor Discussion 2026-04-10)

**結論**：W4A8 + per-group scalar factor 對本設計「雙贏」——壓縮 SRAM 佔用 + 緩解 DMA 瓶頸。

- **現況瓶頸**：MHA 權重 DRAM→WBUF 搬運。WBUF = 16 Banks × 64-bit × 256 Depth = 64KB。dim=256 的 MHA 四矩陣（Wq/Wk/Wv/Wo）fp8 下共 256KB，4-pass 複數路徑需搬兩組（MHA_re + MHA_im）= 512KB。
- **W4 效益**：每矩陣 256×256×4bit = 32KB，四個 = 128KB，兩組 = 256KB → DMA 量砍半。
- **Scalar factor 開銷**：group_size=32 時，每矩陣需 2048 scale × 1B = 2KB（~6% overhead）。硬體端在 MAC input 前加 per-group dequant，weight-stationary 策略下 scale 隨權重載入，pipeline 成本可控。
- **待驗證**：re-training 後語音品質（ILD/IPD 保持度）。

## Hardware-Aware Analysis Directions (6 個分析方向)

按「投入產出比」排序，每個方向對應獨立分析資料夾：

| 優先序 | 資料夾 | 方向 | 需要 re-training? |
|---|---|---|---|
| **T1** | `01_utilization_analysis/` | MAC Utilization & Pipeline 瓶頸分析 | No |
| **T1** | `03_dynamic_range/` | Weight Dynamic Range（per-layer, per-channel） | No |
| **T2** | `02_weights_bitwidth/` | Weight Bit-width 探索（FP vs INT, mixed-precision） | Yes (PTQ + QAT) |
| **T3** | `04_sparsity_analysis/` | Zero/Sparsity 結構分析（structured vs random） | No |
| **T3** | `06_attn_mised_precision/` | Attention Mix-Precision / Dense 分析 | Partial |
| **T4** | `05_conv_reuse/` | Conv Weight Reuse 優化 | No |

**執行策略**：
1. T1 先做 profiling（方向 1 + 3），產出數據基礎
2. T2 根據 dynamic range 結果決定 mixed-precision 策略（哪些層 W4-INT + per-group scale，哪些保持 W8）
3. T3 根據 sparsity 結構性決定是否值得硬體投入
4. T4 最後考慮（L/R shared encoder/decoder 已是一種 reuse，邊際收益小）

## Current Experiments (see current_progress.md)

- Reducing `kernel_size` from `(k, 2)` → `(k, 1)`
- Reducing `fft_len` from 256 → 128 (separable conv variants)
- Causal attention with sliding window
- Post-training quantization and weight sparsity analysis
