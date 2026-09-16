# 21_lbccn_baseline — LBCCN reproduction (comparison baseline for BCCTN)

Reproduction of **LBCCN**: Jingyuan Wang, Jie Zhang, Shihao Chen, Miao Sun,
"A Lightweight and Real-Time Binaural Speech Enhancement Model with Spatial
Cues Preservation," ICASSP 2025. arXiv: 2409.12444.

## Provenance

- `lbccn_official/` — clone of https://github.com/jywanng/LBCCN @ `12aa002`.
  The official repo ships **only** `model.py` + `loss.py` (+ example wavs).
  **No pretrained weights, no training script** → retraining is mandatory.
- `code/model.py` — verbatim copy of official `model.py` (38.0 K params,
  matches the paper's 38.0 K exactly). Note the official release has the
  ComplexConvGRU **commented out** (dual-path block is conv-only); kept as-is.
- `code/loss.py` — official loss with per-step `print()`s removed and the
  hardcoded 40-band slice exposed as `band_num` (default 40, identical math).
- `code/dataset.py` / `code/train.py` — BCCTN-style paired-wav loader +
  a minimal PyTorch Lightning harness (Adam lr 1e-4 per paper, batch 32,
  early stopping on val loss).

## Key differences vs the paper's own training setup (intentional)

| | LBCCN paper | This reproduction |
|---|---|---|
| Speech / noise | Librispeech + DEMAND (README errata; paper said NoiseX-92) | BCCTN dataset (VCTK speech, ISTS/other noises), same pairs BCCTN was trained on |
| Speaker azimuth | **fixed 55°** (errata; paper said 45°), CIPIC HRIR | random −90°…90°, BCCTN HRIR pipeline |
| SNR range | −10…10 dB | BCCTN trainset SNR distribution |
| Segments | 2 s @ 16 kHz | 2 s @ 16 kHz (identical) |

Training on our dataset is the point: it makes LBCCN directly comparable to
our BCCTN numbers (same data, same eval protocol). But note LBCCN was
*designed* for fixed-speaker conditions ("each training instance targets only
one direction"), so random-azimuth training is out-of-spec for them — report
this caveat alongside any comparison.

Official loss defaults used: `snr=0.5, stoi=5, ild=0.5, ipd=5` applied to both
voice and noise branches (the paper's k = 0.5 voice/noise split).

## Model summary

Waveform in/out, STFT 256/hop 128 (129 bins @ 16 kHz). Only the lowest
`band_num=40` bins (≈2.5 kHz) are enhanced; bins 40–128 pass through noisy.
Two decoders estimate complex RTF-ish coefficients a, b; clean estimates are
recovered as `vrc = (xlc − b·xrc)/(a − b)`, `vlc = a·vrc`, which ties L/R
estimates through a relative transfer function → preserves spatial cues.

## Usage

```bash
GPU=5 bash run_train.sh                  # full training (logs → logs/lbccn/)
bash run_train.sh --limit_batches 0.01 --epochs 2   # smoke
```

Paper reference numbers (their dataset, fixed 55°): MBSTOI 0.94 @ 0 dB vs
BCCTN 0.92; ΔPESQ 0.87 vs 0.97; params 38.0 K vs 11.1 M; MACs 216.3 M vs 12.7 G.

## Results (2026-07-11, VCTK testset, 750 pairs/SNR, −6…15 dB avg)

Trained 50 epochs (Adam 1e-4, batch 32), best ckpt epoch 48 (val loss 6.044):
`logs/lbccn/version_0/checkpoints/lbccn-epoch=48-validation_loss=6.044.ckpt`.
Full tables: `results/eval_ep48/`.

| Model | MBSTOI↑ | ΔPESQ↑ | STOI↑ | L_ILD↓ dB | L_IPD↓ rad |
|---|---|---|---|---|---|
| Noisy | 0.7861 | — | 0.8241 | 3.31 | 0.269 |
| BCCTN FP32 (fp32_512_ref_full) | 0.8535 | 0.598 | 0.8852 | 2.95 | 0.260 |
| BCCTN deploy W4A8 (dir18 strict_core) | 0.8355 | 0.374 | — | 3.31 | 0.278 |
| **LBCCN retrained (this dir)** | **0.7663** | 0.042 | 0.8387 | 3.03 | 0.294 |

**LBCCN fails to improve MBSTOI on our random-azimuth dataset** (below noisy
at every SNR; gap grows with SNR — SegSNR_impr goes negative ≥3 dB, L_IPD has
a ~14.7° floor at high SNR). Monaural STOI *does* improve (+0.015), so it
denoises; the binaural/high-band story is what fails. Consistent with LBCCN's
fixed-azimuth design assumption + only ≤2.5 kHz enhanced (40/129 bands, rest
noisy passthrough) + released code has the ConvGRU commented out. GAF-Net
(arXiv 2509.14076) reports LBCCN 0.85 MBSTOI under *their* random-azimuth
HUTUBS setup — they may have trained the paper architecture (GRU in) or a
different loss config; our run uses the official release as-is.
