# Binaural Speech Enhancement — SE Processor Model

Implementation and hardware-oriented experiments based on:

> **Binaural Speech Enhancement Using Deep Complex Convolutional Transformer Networks**
> Vikas Tokala, Eric Grinstein, Mike Brookes, Simon Doclo, Jesper Jensen, Patrick A. Naylor
> ICASSP 2024

The model processes stereo audio to suppress noise while preserving spatial cues (ILD and IPD) using a complex-valued encoder-decoder with a multi-head attention transformer bottleneck.

---

## Repository Structure

| Directory | Description |
|---|---|
| `BCCTN/` | Baseline implementation (original paper architecture) |
| `BCCTN_sep_conv/` | Depthwise separable convolution variant |
| `BCCTN_sep_conv_causal/` | Causal (streaming-compatible) variant |
| `BCCTN_sep_conv_causal_slide_win/` | Causal + sliding attention window — **current focus** |
| `BCCTN_sep_conv_causal_slide_win_quant/` | Above + post-training quantization |
| `BCCTN_sep_conv_causal_slide_win_quant_256dim/` | Above + 256-dim embedding |
| `BCCTN_sep_conv_causal_slide_win_relu/` | Above + ReLU activation |
| `BCCTN_quant/` | Standalone quantization experiment |
| `BCCTN_quantization_pruning/` | Weight pruning + quantization experiments |
| `BCCTN_hardware/` | Hardware deployment: PTQ, activation dumps, weight export |
| `Dataset/` | Shared dataset root (not tracked in git) |

---

## Environment Setup

```bash
# Main environment (PyTorch 2.3.1, CUDA 12.1, Python 3.10)
mamba env create -f bcctn.yml
mamba activate bcctn

# Quantization environment (PyTorch 2.2.0, CUDA 12.4, Python 3.9)
mamba env create -f bcctn_quant.yml
mamba activate bcctn_quant
```

Use `mamba` instead of `conda` for faster dependency solving.

---

## Data Preparation

Run from inside any variant directory:

```bash
python prepare_data.py \
    --clean_dir /path/to/vctk \
    --noise_dir /path/to/noisex92 \
    --hrir_path /path/to/hrir \
    --output_dir ./Dataset \
    --dataset_type vctk   # or timit
```

This generates clean/noisy binaural pairs for training, validation, and testing. For TIMIT, separate directories are created for each SNR level (−6 to +15 dB).

---

## Training

All commands run from inside the target variant directory.

```bash
cd BCCTN_sep_conv_causal_slide_win
python train.py                         # uses config/ via Hydra
HYDRA_FULL_ERROR=1 python train.py      # verbose error output
CUDA_VISIBLE_DEVICES=1 python train.py
```

---

## Evaluation

```bash
bash run_eval.sh

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

Metrics reported: MBSTOI, SegSNR improvement, ILD error, IPD error.

---

## Model Architecture

- **Encoder**: 6 complex convolutional stages (`kernel_num = [16,32,64,128,256,256]`, depthwise separable)
- **Bottleneck**: Complex multi-head attention transformer (32 heads, 512-dim, sliding window size 31)
- **Decoder**: Mirrored 6-stage transposed convolution with skip connections
- **Output**: Individual complex ratio masks for left and right channels
- **STFT**: `win_len=128`, `win_inc=80`, `fft_len=128`

### Loss Function

```
L = α·SNR + β·STOI + γ·ILD + κ·IPD
  = 1·SNR + 10·STOI + 1·ILD + 10·IPD
```

---

## Key Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| `model.kernel_num` | `[16,32,64,128,256,256]` | 6-stage encoder/decoder |
| `model.kernel_size` | 5 | Conv kernel size |
| `model.use_depthwise` | `true` | Depthwise separable conv |
| `model.embed_dim` | 512 | Transformer embedding dim |
| `model.num_heads` | 32 | Attention heads |
| `model.attn_win` | 31 | Sliding attention window |
| `model.win_len` | 128 | STFT window length |
| `model.fft_len` | 128 | FFT size |
| `training.batch_size` | 32 | |
| `training.n_epochs` | 150 | |
| `training.learning_rate` | 0.001 | Decays ×0.5 at epochs 3, 8 |

---

## Citation

```bibtex
@inproceedings{tokala2024binaural,
  title={Binaural Speech Enhancement using Deep Complex Convolutional Transformer Networks},
  author={Tokala, Vikas and Grinstein, Eric and Brookes, Mike and Doclo, Simon and Jensen, Jesper and Naylor, Patrick A},
  booktitle={ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2024},
  organization={IEEE}
}
```

## License

MIT License — see [BCCTN/LICENSE](BCCTN/LICENSE).
