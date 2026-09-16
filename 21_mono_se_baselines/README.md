# 21_mono_se_baselines — Mono-SE paradigm references for the chip-paper comparison table

Reimplementations of the neural networks from the two monaural SE chip papers in
`comparison_table/`, retrained on **our binaural VCTK dataset** and evaluated
**bilaterally** (each ear enhanced independently) on the exact deploy考卷
(MBSTOI / ILD / IPD, 750 pairs × 8 SNR, VCTK + TIMIT). This turns the "N/A"
MBSTOI cells of the comparison table into measured numbers.

**What is measured:** the *mono-SE paradigm*, not the silicon. Both networks are
reproduced from the paper descriptions, trained in **FP32** (no quantization —
favorable upper bound), on **matched data** (our binaural VCTK trainset, each ear
used as an independent mono sample: 61,938 train / 13,272 val samples), with a
modern **SI-SDR waveform loss**. Every choice favors the baseline.

## Models (`code/models/`)

| | `snu_bstgru.py` | `ntu_cnn.py` |
|---|---|---|
| Source chip paper | Park et al., JSSC 2024 (28nm) | Lee et al., JSSC 2020 (40nm, 2.17mW) |
| Underlying algorithm | TRU-Net (Choi et al., ICASSP 2021, ref [6]) + band-separated mods | Lai et al., APSIPA (ref [13]), simplified |
| Reproduced from | Fig. 9 per-layer table + Sec. II | Table I |
| Front-end | 512-pt STFT, hop 256, 128-D Mel magnitude | 256-pt STFT, hop 128, Hamming, 129 bins |
| Mask | sigmoid on Mel mag, pinv-Mel + noisy phase | sigmoid on linear mag, noisy phase, 7-frame context |
| Params (paper) | 56,322 (**56k** ✓) | 495,105 (~0.5M) |

Unspecified-in-paper choices (documented, favorable-to-baseline): SNU Hann
window + HTK Mel + pseudo-inverse Mel (paper: "pseudo-inverse Mel filtering");
NTU log-magnitude input, sigmoid mask, replicate time padding.

## Reproduce

```bash
bash run_train_both.sh                       # trains snu then ntu on GPU $CUDA_VISIBLE_DEVICES
CKPT=results/snu/best_snu.pt TAG=snu GPU_ID=1 bash run_bilateral_eval.sh
CKPT=results/ntu/best_ntu.pt TAG=ntu GPU_ID=1 bash run_bilateral_eval.sh
```

`code/eval_bilateral.py` is `20_pesq_finetune/code/eval.py` with only the model
loading swapped for a `BilateralWrapper` (mono net run on L and R
independently); all metrics, file pairing, and SNR breakdown are identical to
the deploy eval.

Sanity checks passed: SNU param count matches the paper's 56k exactly; identity
mask reconstructs noisy input at 20.9 dB SI-SDR (Mel/pinv path healthy); oracle
mel-IRM ceiling 13.7 dB vs oracle linear-IRM 13.9 dB (both paradigms have
near-identical oracle ceilings on our data — equal footing).

## Results (2026-07-09, full 750×8 SNR VCTK+TIMIT)

| | VCTK MBSTOI | TIMIT MBSTOI | VCTK ILD err | TIMIT ILD err |
|---|---|---|---|---|
| Noisy | 0.7861 | 0.7311 | 3.71 dB | 3.29 dB |
| FP32 binaural (09_tied_ft) | 0.8411 | 0.7966 | **3.21 (improves)** | **2.97 (improves)** |
| Deploy (binaural W4A8) | 0.8357 | 0.7898 | **3.31 (improves)** | **3.08 (improves)** |
| SNU repro (mono FP32) | 0.8383 | 0.8035 | 5.71 (worse than noisy) | 5.79 |
| NTU repro (mono FP32) | 0.8372 | 0.8006 | 5.47 | 5.50 |

MBSTOI does **not** separate the paradigms; ILD error does — the mono
baselines worsen ILD at every SNR point (up to ~2× noisy at -6 dB), the
binaural deploy model is the only one that improves it. Full per-SNR tables
and framing: `docs/RESULTS_MONO_SE_BASELINES_2026-07-09.md`.

Raw outputs: `results/bilateral_eval/{snu,ntu}/{vctk,timit}/snr_*/`; training
curves in `results/{snu,ntu}/train_log.jsonl`.
