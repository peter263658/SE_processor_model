# RESULTS: Binaural Spatial-Cue Metrics — ILD / IPD / ITD / IC

Date: 2026-07-10
Base: same checkpoints as [RESULTS_CHIP_COMPARISON_FINAL_2026-07-09.md](RESULTS_CHIP_COMPARISON_FINAL_2026-07-09.md) (supersedes nothing; adds ITD + IC columns)
Eval: VCTK + TIMIT, 750-pair limit, 32 metric workers, −6..15 dB average

## 0. TL;DR — the two new metrics do NOT strengthen the binaural story; ILD still does

- **ITD (GCC-PHAT, μs):** all methods — mono AND binaural — land 4–5.5 μs, below the human ITD JND (~10–20 μs). ITD does NOT separate binaural from mono (bilateral gain-mask processing preserves time structure). Use it only as a defensive "localization preserved (< JND)" checkmark, not as a competitive win.
- **IC (interaural coherence / MSC error):** mono baselines are *better* (lower) than our model, because independent per-ear denoising over-coheres L/R toward the coherent single-source clean target — the same mechanism that corrupts ILD. **Do NOT feature IC in the comparison table.**
- **ILD remains the unique discriminator:** deploy 3.31 vs SNU 5.71 / NTU 5.47 dB (mono +47–54%). ⭐

## 1. VCTK (enhanced, −6..15 avg)

| Method | Prec. | MBSTOI | STOI | PESQ wb | PESQ nb | ILD↓ dB | IPD↓ ° | ITD↓ μs | IC↓ |
|---|---|---|---|---|---|---|---|---|---|
| noisy | — | 0.7861 | 0.8241 | 1.2404 | 1.8597 | 3.714 | 15.43 | 8.81 | 0.3803 |
| SNU mono (bilat) | FP32 | 0.8383 | 0.8808 | 1.9494 | 2.6372 | **5.706** ✗ | 14.78 | 5.51 | 0.1931 |
| NTU mono (bilat) | FP32 | 0.8372 | 0.8761 | 1.8192 | 2.5396 | **5.469** ✗ | 17.08 | 5.07 | 0.2142 |
| Deploy ⭐ | W4A8 | 0.8357 | 0.8683 | 1.5932 | 2.2985 | **3.308** ✓ | 15.94 | 5.24 | 0.2368 |
| Rebalance | W4A8 | 0.8230 | 0.8605 | 1.8565 | 2.4956 | **3.349** ✓ | 16.92 | 4.23 | 0.2287 |
| Plain-QAT w12 | W4A8 | 0.8209 | 0.8599 | 1.8660 | 2.5033 | **3.492** ✓ | 17.28 | 4.81 | 0.2291 |

## 2. TIMIT (enhanced, −6..15 avg)

| Method | Prec. | MBSTOI | STOI | PESQ wb | PESQ nb | ILD↓ dB | IPD↓ ° | ITD↓ μs | IC↓ |
|---|---|---|---|---|---|---|---|---|---|
| noisy | — | 0.7311 | 0.7666 | 1.2667 | 1.7997 | 3.290 | 21.10 | 5.48 | 0.3467 |
| SNU mono (bilat) | FP32 | 0.8035 | 0.8503 | 1.9351 | 2.5724 | **5.794** ✗ | 20.16 | 4.65 | 0.2161 |
| NTU mono (bilat) | FP32 | 0.8006 | 0.8426 | 1.7642 | 2.4328 | **5.503** ✗ | 22.08 | 4.50 | 0.2409 |
| Deploy ⭐ | W4A8 | 0.7898 | 0.8228 | 1.6558 | 2.2299 | **3.085** ✓ | 20.39 | 4.08 | 0.2401 |
| Rebalance | W4A8 | 0.7798 | 0.8166 | 1.8403 | 2.4016 | **3.120** ✓ | 21.42 | 3.34 | 0.2428 |
| Plain-QAT w12 | W4A8 | 0.7759 | 0.8149 | 1.8482 | 2.4094 | **3.289** ✓ | 21.81 | 4.11 | 0.2453 |

## 3. Metric implementation (eval.py / eval_bilateral.py)

- **ITD**: `compute_itd_error` — GCC-PHAT cross-correlation of L/R, PHAT weighting, ±1 ms search, parabolic sub-sample peak; error = |ITD_clean − ITD_proc| (μs).
- **IC**: `compute_ic_error` — magnitude-squared coherence of L/R averaged over ≤1.5 kHz; error = |IC_clean − IC_proc|, referenced against clean.
- Sanity: GCC-PHAT recovers a known 519 μs ITD to 0 μs (preserved) / 382 μs (shifted); IC error ≈0 (coherent) / ≈0.99 (decohered).
- Ported identically into all three eval codebases: `20_pesq_finetune/code/eval.py`, `18_..._hw_handoff/code/eval.py`, `21_mono_se_baselines/code/eval_bilateral.py`.

## 4. Not yet carrying ITD/IC (optional follow-up evals)

KD+spatial · FP32-512 lineage (baseline / w12-sat / MetricGAN) · FP32-binaural ancestor (09_tied) · low-SNR −8/−10 zero-shot · VoiceBank+DEMAND variants. All would be a single sequential chain (~1–1.5 hr) if full coverage is wanted; none change the ILD-led conclusion above.
