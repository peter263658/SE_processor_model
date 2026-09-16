# RESULTS: Mono-SE baselines — shared-mask ablation + as-published training regime

Date: 2026-08-27 (supersedes the framing of [RESULTS_MONO_SE_BASELINES_2026-07-09.md](RESULTS_MONO_SE_BASELINES_2026-07-09.md); all numbers there remain valid)
Base candidate: deploy = `18_.../ckpt/strict_core_pow2_blockfloat_deploy_ep3_vctk0.8357_timit0.7898.ckpt`; baselines = `21_mono_se_baselines/results/{ntu,ntu_mse,ntu_aspub,snu}/best_*.pt`
Purpose: close the two holes the 07-09 doc left open — (A) can a mono network keep ILD by simply sharing one mask across the ears, and (B) where does the NTU JSSC'20 baseline land at the training regime its paper actually reports.

## 0. TL;DR

All on the existing isotropic考卷 (VCTK/TIMIT, 750 pairs × 8 SNR).

| Variant | Loss / training data | VCTK MBSTOI | TIMIT MBSTOI | VCTK ILD err | TIMIT ILD err | Verdict |
|---|---|---|---|---|---|---|
| Noisy | — | 0.7862 | 0.7311 | 3.7141 | 3.2902 | reference |
| ⭐⭐ Deploy (binaural W4A8) | SNR+STOI+ILD+IPD | 0.8357 | 0.7897 | **3.3079** | **3.0846** | only variant that improves ILD |
| FP32 binaural (09_tied_ft) | same | 0.8411 | 0.7966 | **3.2149** | **2.9743** | same-precision reference |
| NTU per-ear mask | SI-SDR, 30,969 mixes | 0.8372 | 0.8006 | 5.4689 | 5.5032 | ties MBSTOI, wrecks ILD |
| NTU **shared mask** | SI-SDR, 30,969 mixes | **0.8653** | **0.8361** | 3.6593 | 3.2911 | **beats deploy on MBSTOI**; ILD only preserved |
| NTU MSE loss | mag MSE, 30,969 mixes | 0.8474 | 0.8090 | 4.5838 | 4.4929 | MSE **beats** SI-SDR on MBSTOI |
| NTU **as-published** | mag MSE, 694 mixes @ ~0 dB | **0.8025** | **0.7678** | 5.0008 | 4.7012 | **below deploy** (−0.033 / −0.022) |
| SNU per-ear mask | SI-SDR, 30,969 mixes | 0.8383 | 0.8035 | 5.7056 | 5.7940 | as 07-09 |
| SNU **shared mask** | SI-SDR, 30,969 mixes | 0.8662 | 0.8402 | 3.8802 | 3.4039 | same pattern as NTU |

Three findings, two of them against the prior expectation:

1. **A shared mask beats the deploy model on every intelligibility metric** (MBSTOI +0.0296/+0.0464, PESQ +0.357, STOI +0.025) and restores ILD to the noisy level. It cannot *improve* ILD — mathematically impossible for a common per-bin gain — which is the one axis deploy still wins.
2. **Magnitude MSE beats SI-SDR on MBSTOI** (0.8474 vs 0.8372) and on ILD (4.58 vs 5.47), despite 0.50 dB worse val SI-SDR. The as-published ladder is therefore **not monotonic**.
3. **At the regime the NTU paper actually reports, the baseline lands at 0.8025 / 0.7678 — below deploy.** The 0.8372 "tie" in the 07-09 doc was an artifact of handing the baseline 45× more training data spanning all SNRs.

## 1. Setup

- Gates: deploy full VCTK 0.8357 / TIMIT 0.7898 (`ft_pow2_ep3_full`). All rows scored on the same考卷 with `eval_bilateral.py`, 750 pairs × 8 SNR, `metric_workers=32`.
- **(A) shared mask**: `NTUCnnNet.forward_pair_shared` / `SNUBstgruNet.forward_pair_shared` average the two ears' masks **in each network's own mask domain** (NTU linear magnitude, SNU Mel) and apply the result identically to both ears. Selected by `BCCTN_MONO_MASK_MODE=shared`; per-ear behaviour is unchanged by default.
- **(B) as-published regime**, from the NTU JSSC'20 paper Sec. VI-B: 620 clean utterances, one male speaker, five NOISEX types, **all mixed at 0 dB**; test at 0/±3/±6 dB; **the paper never states a loss function** (Sec. I only says "noisy speech data as the model input and clean speech data as the target"), so magnitude-spectrum MSE stands in — the objective of the Lai et al. APSIPA model it simplifies.
  - Mapped onto our data as `|SNR| ≤ 0.2 dB`, which selects **694 of the 30,969 train mixes** (val: 154 of 6,636), all at −0.2…+0.2 dB. Data scale (694 vs 620) and the single-SNR constraint both land close; speaker diversity is *not* restricted (109 speakers vs the paper's 1), which favours the baseline.
  - Checkpoint selection follows the objective being trained (`monitor = -val_mse` for MSE runs), so an SI-SDR-selected checkpoint cannot smuggle the modern loss back in.

## 2. Results

### 2.1 Experiment A — shared mask

Verification before the eval (`forward_pair_shared` is correct, the ILD residual is physics not a bug):

| check | result |
|---|---|
| L==R input, shared vs per-ear path | max abs diff **0.0** (both nets) |
| ILD deviation from noisy, **before** iSTFT | **1.14e-06 dB** — exact preservation |
| ILD deviation from noisy, per-ear mask, before iSTFT | 10.6991 dB |
| ILD deviation from noisy, shared mask, **after** iSTFT | 3.5616 dB |

The 3.56 dB after the iSTFT is overlap-add of a magnitude-masked, phase-unchanged (inconsistent) STFT — inherent to every mask-based method, not an implementation defect. It is why the measured ILD error lands at 3.6593 rather than exactly at noisy's 3.7141.

VCTK per-SNR MBSTOI:

| SNR | Noisy | Deploy | NTU per-ear | NTU shared | SNU per-ear | SNU shared |
|---|---|---|---|---|---|---|
| -6 | 0.5366 | 0.6649 | 0.6721 | 0.7269 | 0.6741 | 0.7276 |
| -3 | 0.6376 | 0.7310 | 0.7358 | 0.7829 | 0.7383 | 0.7841 |
| 0 | 0.7264 | 0.7962 | 0.7988 | 0.8341 | 0.8011 | 0.8350 |
| 3 | 0.7949 | 0.8403 | 0.8415 | 0.8711 | 0.8436 | 0.8724 |
| 6 | 0.8474 | 0.8767 | 0.8759 | 0.8972 | 0.8764 | 0.8983 |
| 9 | 0.8897 | 0.9073 | 0.9059 | 0.9219 | 0.9050 | 0.9218 |
| 12 | 0.9171 | 0.9260 | 0.9256 | 0.9375 | 0.9255 | 0.9381 |
| 15 | 0.9395 | 0.9431 | 0.9419 | 0.9510 | 0.9425 | 0.9520 |
| **AVG** | **0.7862** | **0.8357** | **0.8372** | **0.8653** | **0.8383** | **0.8662** |

Shared mask leads at **every** SNR point. Mechanism: averaging two estimates of the same underlying speech halves the mask-estimation variance, and identical processing keeps the ears consistent, which MBSTOI's EC stage rewards.

Other metrics also go to the shared mask — VCTK PESQ (wb) 1.9501 vs deploy 1.5932, VCTK STOI 0.8936 vs 0.8683, VCTK ITD 3.76 µs vs 5.24, VCTK IC 0.1816 vs 0.2368. **The only metric deploy wins is ILD** (3.3079 vs 3.6593 VCTK; 3.0846 vs 3.2911 TIMIT).

VCTK per-SNR ILD error:

| SNR | Noisy | Deploy | NTU per-ear | NTU shared |
|---|---|---|---|---|
| -6 | 6.0667 | 4.9692 | 9.0958 | 5.9688 |
| -3 | 5.2598 | 4.4118 | 8.0779 | 5.1673 |
| 0 | 4.5349 | 3.9165 | 6.7325 | 4.4525 |
| 3 | 3.9004 | 3.4662 | 5.7970 | 3.8345 |
| 6 | 3.2572 | 3.0103 | 4.7462 | 3.2097 |
| 9 | 2.6608 | 2.5615 | 3.8083 | 2.6324 |
| 12 | 2.2019 | 2.2150 | 3.0205 | 2.1859 |
| 15 | 1.8313 | 1.9129 | 2.4728 | 1.8229 |
| **AVG** | **3.7141** | **3.3079** | **5.4689** | **3.6593** |

Shared mask tracks noisy within ±0.1 dB at every point — it neither damages nor repairs. Deploy improves on noisy at every point from −6 to 9 dB (−1.10 dB at −6 dB).

**Scope limit that keeps this from being fatal:** averaging the two ears' masks requires both ears' spectra in one place, i.e. an ear-to-ear link. JSSC'20 and JSSC'24 are single-channel processors in physically separate devices. `ntu_shared` is therefore not a model of either chip — it is a new binaural baseline that a reviewer could propose, and now has a number.

### 2.2 Experiment B — as-published training regime ladder

val SI-SDR re-scored on **both** val sets, because `ntu_aspub`'s own val set is the 0 dB subset and is not comparable to the full-range one:

| Run | Loss | Train data | full val (−7…16 dB) | 0 dB val (\|snr\|≤0.2) |
|---|---|---|---|---|
| `ntu` | SI-SDR | 30,969 mixes, −7…16 dB | 12.268 | 9.125 |
| `ntu_mse` | mag MSE | 30,969 mixes, −7…16 dB | 11.773 | 8.631 |
| `ntu_aspub` | mag MSE | 694 mixes, ~0 dB | **9.634** | 7.522 |

On the common val set: loss change costs **0.50 dB**, the data regime a further **2.14 dB** — total **2.63 dB**. (The naive read of the raw training logs suggested 4.75 dB; roughly half of that was the val-set difference.)

VCTK per-SNR MBSTOI for the ladder:

| SNR | Noisy | Deploy | NTU SI-SDR | NTU MSE | NTU as-published |
|---|---|---|---|---|---|
| -6 | 0.5366 | 0.6649 | 0.6721 | 0.6945 | 0.6458 |
| -3 | 0.6376 | 0.7310 | 0.7358 | 0.7551 | 0.7145 |
| 0 | 0.7264 | 0.7962 | 0.7988 | 0.8137 | 0.7729 |
| 3 | 0.7949 | 0.8403 | 0.8415 | 0.8519 | 0.8110 |
| 6 | 0.8474 | 0.8767 | 0.8759 | 0.8824 | 0.8405 |
| 9 | 0.8897 | 0.9073 | 0.9059 | 0.9101 | 0.8640 |
| 12 | 0.9171 | 0.9260 | 0.9256 | 0.9282 | **0.8794** |
| 15 | 0.9395 | 0.9431 | 0.9419 | 0.9429 | **0.8923** |

**The as-published model degrades the signal at high SNR**: 0.8794 vs noisy 0.9171 at 12 dB, 0.8923 vs 0.9395 at 15 dB. Trained only at 0 dB, it over-suppresses whenever the input is cleaner than that — the textbook single-training-SNR generalisation failure, and a property of the published regime rather than of the network.

## 3. Cross-cuts

- **MBSTOI is ~94% magnitude.** Measured on 12 pairs @ 0 dB against the clean binaural reference: clean magnitude + clean phase (round-trip control) 1.0000; **clean magnitude + noisy phase 0.9445**; **noisy magnitude + clean phase 0.8389**; noisy 0.8031. Discarding all clean phase costs 0.056; perfect phase over a noisy magnitude buys 0.036. This is visible in the code — the better-ear branch reduces every frame to `H @ |X|²` per 1/3-octave band, so monaural phase is literally discarded; the EC branch keeps only the interaural cross-spectrum `ρ = Σ conj(X_R)·X_L`.
- **Which is why MSE beats SI-SDR here.** Magnitude MSE optimises exactly what MBSTOI scores; SI-SDR spends its capacity on waveform fine structure that the metric barely sees. Same reason MSE also improves ILD (4.58 vs 5.47) — accurate per-ear magnitudes give accurate interaural level ratios.
- **MBSTOI is *not* blind to ILD**, contrary to an intermediate hypothesis in this session. Scaling one ear by a constant offset on 12 noisy pairs @ 0 dB: +0 dB → 0.8031, +1 → 0.7953, +3 → **0.7570**, +6 → 0.6975, +10 → 0.6549. The EC stage's ±20 dB γ search does not neutralise a level offset. Any argument resting on "MBSTOI structurally cannot see ILD" is unsupported.
- **The ILD claim is the robust one.** Every per-ear mono variant damages ILD regardless of loss or data regime (4.58–5.71 against noisy 3.71); only the shared mask returns to noisy (3.66–3.88), and only the binaural models improve on it (3.21 FP32 / 3.31 W4A8). Unlike MBSTOI, this ordering does not move under any training condition tested.
- **Both surprises share a root cause.** They were measured on the isotropic考卷, whose interferer carries a broadband ILD of **0.23 dB** — there is essentially no spatial information for a binaural model to exploit. See `22_directional_interferer/` for the follow-up, and Westhausen et al. (TASLP 2024) for the published precedent that diffuse noise yields a mono/binaural tie.

## 4. Recommendation

The comparison table reports **absolute MBSTOI per system on one考卷**, and read
that way experiment B settles the headline row: on the isotropic考卷 we score
**0.8357 against the NTU baseline's 0.8025 at the training regime its own paper
reports** — +0.0331, with ILD 3.31 against 5.00. That is the row to lead with,
because it compares against what the paper actually describes (Sec. VI-B: 620
utterances, one male speaker, all mixed at 0 dB) rather than a baseline we
weakened.

It also carries a functional argument the mean hides: trained at a single SNR,
that model **degrades the signal in quiet**, on both test sets.

| SNR | VCTK noisy | NTU as-pub | net | TIMIT noisy | NTU as-pub | net |
|---|---|---|---|---|---|---|
| 0 | 0.7264 | 0.7729 | +0.0465 | 0.6548 | 0.7236 | +0.0688 |
| +6 | 0.8474 | 0.8405 | **−0.0069** | 0.8034 | 0.8184 | +0.0150 |
| +9 | 0.8897 | 0.8640 | **−0.0257** | 0.8463 | 0.8465 | +0.0002 |
| +12 | 0.9171 | 0.8794 | **−0.0377** | 0.8842 | 0.8677 | **−0.0165** |
| +15 | 0.9395 | 0.8923 | **−0.0472** | 0.9181 | 0.8885 | **−0.0296** |

Crossover at +6 dB on VCTK and +12 dB on TIMIT — the same pattern on both sets,
so it is not a single-corpus artifact. Our model improves at **every** SNR point
on both. For a hearing aid this is a functional defect: in a quiet room the chip
is worse than bypass. It follows entirely from the training regime the paper
describes, not from any choice of ours.

Present the paradigm as a range with each operating point labelled:

| Row | VCTK MBSTOI | VCTK ILD | What it is |
|---|---|---|---|
| Ours (W4A8 zero-multiplier) | **0.8357** | **3.31** | the chip |
| NTU [ref], as published | 0.8025 | 5.00 | its paper's training regime |
| NTU [ref], upper bound | 0.8372 | 5.47 | 45× data, all SNRs, modern loss, FP32 |
| NTU [ref] + ear-to-ear link | 0.8653 | 3.66 | shared mask; not realisable on that chip |

Against the fully-fed upper bound the isotropic考卷 is a tie (−0.0015) — say so,
and pair it with a directional condition where we win outright (1 noise
interferer 90°: 0.8419 against 0.8307, see
[22_directional_interferer](../../22_directional_interferer/docs/RESULTS_DIRECTIONAL_INTERFERER_2026-08-27.md)).
ILD needs no such pairing: we win every condition tested.

- Publish the shared-mask row with its ear-to-ear-link constraint stated. Any
  reviewer can propose it; being asked without a number is worse.
- `FEATURES_en.md:280` claims "ours leads on binaural intelligibility (MBSTOI
  0.836)". True against the as-published baseline, a tie against the fully-fed
  one — add the qualifier.
- Do not use "MBSTOI cannot see ILD" as an argument — disproved in §3.
- The magnitude-MSE result (§3) suggests an ablation on our own loss, which
  carries a waveform SNR term. Not run here.

## 5. Reproduction

```bash
cd /raid/R12K41024/21_mono_se_baselines
# (A) shared-mask evals — both nets, both datasets
GPU_ID=<free> bash run_shared_mask_evals.sh
# or one run: CKPT=results/ntu/best_ntu.pt TAG=ntu_shared MASK_MODE=shared GPU_ID=<free> bash run_bilateral_eval.sh

# (B) as-published ladder — trains ntu_mse + ntu_aspub, then evals both
GPU_ID=<free> bash run_aspub_regime.sh

# comparison table across every run
python collate_rows.py
```

Strictly sequential — concurrent GPU jobs get SIGTERMed on this box (see `run_full_pipeline.sh`). Raw per-SNR files: `results/bilateral_eval/{ntu,ntu_shared,ntu_mse,ntu_aspub,snu,snu_shared}/{vctk,timit}/snr_*/`. Training curves: `results/{ntu_mse,ntu_aspub}/train_log.jsonl`.
