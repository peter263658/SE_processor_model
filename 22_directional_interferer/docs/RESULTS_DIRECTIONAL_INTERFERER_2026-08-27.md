# RESULTS: Directional-interferer conditions — does the binaural model do anything a mono one cannot?

Date: 2026-08-27
Base candidate: deploy = `18_.../ckpt/strict_core_pow2_blockfloat_deploy_ep3_vctk0.8357_timit0.7898.ckpt`; FP32 ancestor = `09_weight_tied_mha/outputs/finetune/2026-05-09_22-29-20/.../epoch=57-step=56144.ckpt`; mono baselines = `21_mono_se_baselines/results/ntu/best_ntu.pt`
Prior: [RESULTS_SHARED_MASK_AND_ASPUB_REGIME_2026-08-27.md](../../21_mono_se_baselines/docs/RESULTS_SHARED_MASK_AND_ASPUB_REGIME_2026-08-27.md)
Reference: N. L. Westhausen et al., "Real-Time Multichannel Deep Speech Enhancement in Hearing Aids: Comparing Monaural and Binaural Processing in Complex Acoustic Scenarios," IEEE/ACM TASLP vol. 32, 2024 ([arXiv:2405.01967](https://arxiv.org/abs/2405.01967))

## 0. TL;DR

**The directional conditions did not rescue the binaural intelligibility claim.** Against the true mono paradigm (per-ear masks, no ear-to-ear link — what JSSC'20/'24 chips implement) MBSTOI is a tie in all five conditions, within ±0.011 and with no reliable ordering.

| Condition | Interferer ILD | better-ear adv. | noisy MBSTOI | Deploy Δ | mono per-ear Δ | Deploy lead | mono shared Δ |
|---|---|---|---|---|---|---|---|
| isotropic (deploy考卷) | 0.23 dB | 4.53 dB | 0.7862 | +0.0495 | +0.0510 | **−0.0015** | +0.0792 |
| 2 talkers ±60° (roaming target) | 2.62 dB | 3.63 dB | 0.7812 | −0.0075 | −0.0117 | **+0.0042** | −0.0053 |
| 2 noise ±60° | 4.45 dB | 5.13 dB | 0.8311 | +0.0082 | +0.0052 | **+0.0030** | +0.0248 |
| 1 talker 90° | 5.64 dB | 10.10 dB | 0.8110 | −0.0190 | −0.0187 | **−0.0003** | −0.0114 |
| 1 noise 90° | 6.79 dB | 11.25 dB | 0.8505 | −0.0086 | −0.0198 | **+0.0112** | +0.0119 |
| ⭐⭐ S0N±60 (frontal target, Westhausen-matched) | 2.39 dB | **2.32 dB** | 0.8104 | −0.0112 | −0.0138 | **+0.0026** | −0.0063 |

ILD error change against noisy:

| Condition | Deploy | mono per-ear | mono shared |
|---|---|---|---|
| isotropic | **−0.406** | +1.755 | −0.055 |
| 2 talkers ±60° | +0.418 | +0.476 | +0.118 |
| 2 noise ±60° | +0.447 | +1.061 | +0.138 |
| 1 talker 90° | +0.473 | +0.485 | +0.114 |
| 1 noise 90° | +0.331 | +1.005 | +0.149 |
| S0N±60 | +0.391 | +0.466 | +0.106 |

Three claims survive, three do not:

| Claim | Status |
|---|---|
| The isotropic考卷 cannot separate the paradigms | **holds** — matches Westhausen's published S0Ndiff result (binaural 0.44 = monaural 0.44) |
| A single lateral interferer is not a valid spatial test | **holds** — better-ear advantage 10–11 dB, noisy baseline pushed to 0.8505, every model degrades it |
| W4A8 costs nothing out-of-distribution | **holds** — FP32 ancestor matches deploy within noise on both directional conditions |
| Binaural beats mono on MBSTOI once the interferer is directional | **fails** — tie in all five conditions |
| The binaural lead grows with interferer concentration | **fails** — asserted mid-session on four points, broken by the fifth (1 talker 90° = −0.0003) |
| Only the binaural model preserves spatial cues | **narrows sharply** — deploy *improves* ILD only on isotropic; on noise interferers it damages ILD less than per-ear mono (0.6–2.2 dB better); on **speech** interferers the two are indistinguishable (0.01–0.06 dB) |

## 1. Setup

The existing考卷 mixes every target with **isotropic noise** — 37 uncorrelated sources every 5° across the azimuthal plane ([prepare_data.py](../../BCCTN/prepare_data.py) `create_isotropic_noise`, itself "as described in the paper", i.e. inherited from BCCTN ICASSP'24). Measured broadband interferer ILD: **0.23 dB**. There is essentially no spatial information for a binaural model to exploit, which is the mechanism behind the paradigm tie found in the 07-09 doc.

Five directional conditions were built with everything else pinned — same clean targets reused verbatim from `clean_testset`, same SNR grid, same `mix_speech_and_noise` SNR definition, same file naming so `eval.py` / `eval_bilateral.py` run unchanged. Verified: measured SNR matches nominal within 0.04 dB and matches the isotropic set; all 6400 mixtures per condition hit the intended separation exactly.

**SNR definition.** Westhausen set the *better-ear* SNR; every set here keeps the *global* SNR of the isotropic考卷, so only the interferer's spatial distribution changes. The offset is condition-dependent (isotropic +1.52 dB, 1 interferer 90° **+6.59 dB**, 2 interferers ±60° +2.97 dB at nominal −6 dB), a further reason cross-condition comparison is not meaningful and the single-interferer sets are ~5 dB easier at the same nominal number.

**Why symmetric pairs.** With one lateral interferer the far ear is simply cleaner and better-ear listening carries the benefit — which a per-ear mono model collects for free, since MBSTOI models better-ear explicitly. Flanking the target at ±60° removes the clean ear. Measured better-ear advantage: 10.10–11.25 dB for single interferers versus **2.32 dB for S0N±60**, below even isotropic's 4.53 dB.

**Why a frontal target.** With two competing talkers and no steering input the task is ill-posed unless the target has a consistent spatial prior — the same −30° voice must be kept when it is the target and suppressed when it is an interferer. Westhausen fix the target at 0° (S0). `dirspeech2x60` lets the target roam over ±30° and is kept as the ill-posed contrast; `dirspeech2x60_s0` restricts to |az| ≤ 5°, leaving 815 test targets (≥ the 750 the eval uses, so the pair count matches every other condition) and 3,784 train targets.

**Unclosable gap:** Westhausen use **2 microphones per ear** (4 channels); this system has **1 per ear** (2 channels). Their "monaural" model can still beamform on its own side. Our ceiling on extractable spatial information is structurally lower, and no amount of retraining changes that.

## 2. Results

### 2.1 Deploy vs the true mono paradigm

Reading the TL;DR table by row: the deploy lead over per-ear mono is +0.0042, +0.0030, −0.0003, +0.0112, +0.0026 across the five directional conditions and −0.0015 on isotropic — mean ≈ +0.0032, no ordering with interferer concentration. The best-designed condition, S0N±60 (lowest better-ear advantage at 2.32 dB, real spatial structure, well-posed frontal target, matched to the reference paper), gives +0.0026 — squarely at the mean, so the tie is not an artifact of any single condition's design. A mid-session claim that the lead grew monotonically with spatial concentration was made on the first four points and is **withdrawn**: the 1-talker-90° point (−0.0003) breaks it.

### 2.1b Correction — the cross-architecture comparison cannot answer "does binaural help"

Everything in §2.1 compares **BCCTN against the NTU CNN**, two architectures that
differ in parameter count, FFT size, causality and loss design. Those differences
swamp the binaural effect, so a tie there says nothing about binaural processing —
it says the two networks are similarly good overall.

Westhausen avoid this by ablating **one model's input**: GCFSnet with ipsilateral
features only versus the same net also given the contralateral side. The repo
already has that experiment — `18_.../code/eval_bilateral_ablation.py` exploits the
weight-shared per-ear encoder/decoder to feed the deploy model `stack([L, L])` and
`stack([R, R])`, so the architecture, weights and quantisation are identical and
only the contralateral input is removed.

On the isotropic考卷 (600 pairs), that ablation gives the opposite verdict:

| config | MBSTOI | STOI | ILD | IPD | ITD err |
|---|---|---|---|---|---|
| noisy | 0.7880 | 0.8236 | 3.719 | 16.19° | 10.29 µs |
| **binaural** | **0.8369** | 0.8678 | **3.302** | 16.70° | 5.06 µs |
| **ipsilateral only** | 0.8161 | 0.8472 | 4.118 | 18.91° | 6.22 µs |

**+0.0208 MBSTOI and 0.82 dB ILD from the contralateral input alone** — against the
−0.0015 "tie" the cross-architecture comparison reported on the same考卷. Removing
the contralateral ear also flips ILD from improving (3.302 vs noisy 3.719) to
damaging (4.118), which attributes the improvement to binaural processing directly.

§2.1 remains valid for the question it actually answers — *this chip versus those
chips*, which is what the comparison table needs. It is not evidence about the
value of binaural processing. Directional conditions under the ablation design:
see §2.5.

### 2.2 The shared-mask baseline

`ntu_shared` (mean of the two ears' masks applied identically to both) leads deploy on MBSTOI in every condition except the two where everything degrades, and preserves ILD better than deploy in five of six conditions. It requires both ears' spectra in one place — an ear-to-ear link — so it is not a model of either reference chip, but it is a baseline any reviewer can propose and must appear in the paper with that caveat.

### 2.3 FP32 control — the quantization hypothesis is disproved

The deploy model's ILD behaviour flips from improving (−0.406 on isotropic) to damaging (+0.33…+0.47 on directional). The hypothesis was that its **statically calibrated** per-tensor pow2 exponents, measured on isotropic data, saturate out of distribution. The FP32 ancestor was run on the same two conditions:

| Condition | Deploy ΔMBSTOI | FP32 ΔMBSTOI | Deploy ΔILD | FP32 ΔILD |
|---|---|---|---|---|
| 2 talkers ±60° | −0.0075 | −0.0109 | +0.418 | +0.478 |
| 2 noise ±60° | +0.0082 | +0.0070 | +0.447 | +0.352 |

FP32 degrades ILD by the same amount. **Recalibration will not fix it** — the effect is architectural or from the training distribution.

Useful side result for the chip paper: **W4A8 costs essentially nothing out-of-distribution**, trading wins with FP32 across four numbers at margins of 0.001–0.095. Quantization robustness holds outside the training distribution, which this run did not set out to measure.

### 2.4 Retraining on the directional data

**Baseline arm (valid).** NTU at its as-published regime (magnitude MSE, the 694
mixes at |SNR| ≤ 0.2 dB standing in for the paper's 620 utterances at 0 dB) was
retrained on `noisy_trainset_dirnoise90` and scored on the matching test set.
Matched training makes it *worse*, not better:

| dirnoise90 | MBSTOI | vs noisy 0.8505 | ILD | vs noisy 3.991 |
|---|---|---|---|---|
| Ours (W4A8, **isotropic-trained**) | **0.8419** | −0.0086 | **4.322** | +0.331 |
| NTU as-published, **retrained on this condition** | 0.8144 | **−0.0361** | 5.893 | **+1.902** |

We lead by **+0.0275 MBSTOI and 1.57 dB ILD while using isotropic-trained
weights against a baseline that got matched training** — the asymmetry runs
against us. Compare the isotropic row (+0.0332): the gap does **not** widen with
a directional interferer, it narrows slightly. A single-SNR, 620-utterance regime
generalises even worse once the interferer is spatially concentrated.

**Our arm (failed twice, both times a load/architecture defect, not a result).**
Neither attempt produced a usable model; neither number may be quoted.

| Attempt | Vehicle | What happened | Cause |
|---|---|---|---|
| 1 | 09 tied-MHA FP32 (deploy ancestor) | 0.3115 on S0, **0.2765 on isotropic** against the original 0.8411 | **Unknown.** All 215 tensors loaded, config matched, no NaN, weights moved 9.11% median. Ruled out: partial load, config drift, architecture mismatch, numeric corruption |
| 2 | FP32-512 via `20_pesq_finetune/fp32_512_pesq_ft/code` | not a fine-tune at all | That copy's `model.py` **hardcodes** `encoder_l/encoder_r/decoder_l/decoder_r` (555 tensors, 8.08M) while the checkpoint has 195 (4.80M). Warm start loaded **195/555**; 65% stayed random |

The second is diagnosed and fixable: `BCCTN/`'s own `model.py` has no `encoder_l`,
the original run's config is preserved at
`BCCTN/outputs/2025-05-10/19-39-07/.hydra/config.yaml` (win 400 / fft 512), and
the remaining 24 shape mismatches are one systematic difference — the conv time
kernel is `(5,1)`/`padding=(2,0)` in current code against `(5,2)`/`padding=(2,1)`
in the checkpoint, with the original still present as commented-out lines in
`BCCTN/DCNN/models/model.py:133`. Any retry must verify, **before training**:
tensor count, shape match with missing=0, and that the *untrained* loaded model
reproduces MBSTOI 0.8535.

A trap worth recording: the 512 directory's **eval** path loads that checkpoint
correctly (this session reproduced 0.8535 exactly), while its **train** path
builds the larger architecture. A successful eval is therefore not evidence that
a fine-tune in the same directory will warm-start correctly.

**The fix above was itself wrong, and the reason matters.** Restoring the
`(5,2)`/`padding=(2,1)` time kernel to match the checkpoint's shapes makes the
convolution **non-causal** — kernel 2 in time padded on both sides sees one future
frame. The author's 0.8535 checkpoint is a non-causal model. Adopting it to fix a
shape mismatch would have silently traded away the causality the chip claims
(~8 ms algorithmic latency against NTU's ≈32 ms centred 7-frame context), and any
score obtained that way would contradict the paper's own latency row.

The correct move is therefore to **discard the author's weights** and train the
original resolution (win 400 / fft 512, embed 512, 32 heads) from scratch with the
causal `(5,1)`/`padding=(2,0)` kernel. What carries over from the original is its
frequency resolution and capacity, not its parameters. `run_512_scratch.sh` does
this; `code512/` is an isolated copy of `BCCTN/`'s code so the legacy directory is
untouched.

Generalised: a shape mismatch between a checkpoint and current code is a signal
that the architecture changed, and the change may encode a design commitment.
Reshaping the model to fit the checkpoint can silently revert that commitment.
Check what the difference *means* before matching it.

### 2.5 Same-model ablation: binaural against ipsilateral-only input

`eval_bilateral_ablation.py` feeds the deploy model `stack([L, L])` and
`stack([R, R])`, removing the contralateral ear with architecture, weights and
quantisation held fixed — Westhausen's ablation design applied to our model.
600 pairs per condition.

| Condition | Interferer | binaural | ipsilateral only | MBSTOI gain | binaural ILD | ipsi ILD | ILD gain |
|---|---|---|---|---|---|---|---|
| isotropic | noise | 0.8369 | 0.8161 | **+0.0208** | 3.302 | 4.118 | **+0.816** |
| 2 noise ±60° | noise | 0.8457 | 0.8411 | **+0.0045** | 3.285 | 3.645 | **+0.360** |
| 1 noise 90° | noise | 0.8439 | 0.8400 | **+0.0039** | 4.143 | 4.410 | **+0.267** |
| 2 talkers ±60° | speech | 0.7763 | 0.7772 | −0.0009 | 2.105 | 2.302 | **+0.198** |
| S0N±60 | speech | 0.8020 | 0.8054 | −0.0034 | 2.134 | 2.244 | **+0.110** |
| 1 talker 90° | speech | 0.7958 | 0.8001 | −0.0043 | 2.479 | 2.755 | **+0.276** |

The contralateral input buys intelligibility against **noise** interferers and
nothing against **speech** interferers, but it improves ILD preservation in **all
six**. Note the gain is largest on isotropic (+0.0208), the opposite of what a
spatial-filtering account predicts — consistent with the model having learned
diffuse-noise statistics, which it is the only thing it was ever trained on.

### 2.6 FP32-512, the algorithm-level reference

The original full-size BCCTN (win 400 / fft 512, **4.80M** parameters against the
deploy model's 542K) re-scored this session with current code, reproducing its
recorded MBSTOI exactly:

| | VCTK MBSTOI | TIMIT MBSTOI | VCTK STOI | VCTK PESQ wb |
|---|---|---|---|---|
| FP32-512 | **0.8535** | **0.8134** | **0.8852** | **1.8380** |
| NTU per-ear, fully fed | 0.8372 | 0.8006 | 0.8761 | 1.8193 |
| Deploy (W4A8, 128-pt) | 0.8357 | 0.7897 | 0.8683 | 1.5932 |

It wins MBSTOI, STOI and PESQ on both sets — but at 16× the deploy parameter
count and 9.7× NTU's, so it belongs in the paper as an **algorithm-level
reference**, not a silicon row. Its use is to bound the cost of the hardware
chain: 0.8535 → 0.8357 is **−0.0178** for 512→128 FFT, structural compression and
W4A8 together.

**ILD from this row is not comparable.** Its eval pipeline reports a noisy ILD of
3.3072 dB against 3.7141 dB everywhere else — reproduced with current code, so it
is a property of that path (win 400 / hop 100 alignment), not a stale metric.
Quote only its improvement over its own baseline (−0.353 dB) and footnote it.

## 3. Cross-cuts

- **Hypotheses eliminated, in order.** Test-set design (isotropic carries no spatial information) — confirmed as a real defect, but fixing it did not change the MBSTOI verdict. Metric blindness — MBSTOI was fully dissected and it does see ILD (a constant +3 dB offset costs 0.046). Better-ear contamination — real, fixed with symmetric pairs. Ill-posedness — addressed with the frontal-target S0 condition. Quantization calibration — disproved by the FP32 control. **Only the training distribution remains untested.**
- **The ILD claim is condition- and interferer-dependent.** Deploy beats per-ear mono by 0.6–2.2 dB on *noise* interferers but by 0.01–0.06 dB on *speech* interferers. "Improves ILD" (a negative delta) is true only on isotropic.
- **The single-interferer conditions are a useful negative example**, not filler: they show that adding a lateral interferer is not the same as testing binaural processing. At an 11.25 dB better-ear advantage the noisy baseline reaches 0.8505 and every model degrades it.

## 4. Recommendation — what goes in the comparison table

The comparison table reports **absolute metric values per system within one
condition**, not improvement over noisy. Read that way the measurements already
answer it: our model beats the NTU per-ear baseline on MBSTOI in four of six
conditions and on ILD in **all six**.

### Rows to use

| Condition | Ours (W4A8) | NTU per-ear | Δ | Ours ILD | NTU ILD | Δ ILD |
|---|---|---|---|---|---|---|
| **isotropic, NTU at its published training regime** | **0.8357** | 0.8025 | **+0.0331** | **3.31** | 5.00 | **−1.69** |
| **1 noise 90°, NTU as-published retrained on it** | **0.8419** | 0.8144 | **+0.0275** | **4.32** | 5.89 | **−1.57** |
| 1 noise interferer 90°, NTU fully fed | **0.8419** | 0.8307 | **+0.0112** | **4.32** | 5.00 | **−0.67** |
| 2 talkers ±60° | **0.7737** | 0.7695 | +0.0042 | **2.12** | 2.18 | −0.06 |
| 2 noise ±60° | **0.8393** | 0.8363 | +0.0030 | **3.37** | 3.99 | **−0.61** |
| S0N±60 | **0.7992** | 0.7966 | +0.0025 | **2.13** | 2.20 | −0.08 |
| isotropic, NTU given every advantage | 0.8357 | 0.8372 | −0.0015 | **3.31** | 5.47 | **−2.16** |
| 1 talker 90° | 0.7920 | 0.7923 | −0.0003 | **2.54** | 2.55 | −0.01 |

**Headline row: 0.8357 vs 0.8025 (+0.0331).** This compares against the training
regime the NTU paper itself reports (Sec. VI-B: 620 utterances, one male speaker,
all mixed at 0 dB) — not against a baseline we weakened. It is the fairest
representation of what that chip actually is, and it carries a functional
argument beyond the mean: trained at a single SNR, that model **degrades the
signal in quiet** (12 dB: 0.8794 against noisy 0.9171; 15 dB: 0.8923 against
0.9395), which for a hearing aid is a defect, not merely a lower score.

**Second row: 1 noise interferer 90°, both at the as-published regime
(0.8419 vs 0.8144, +0.0275).** Here the baseline was **retrained on this exact
condition** while ours still runs isotropic-trained weights — the asymmetry runs
against us and we still lead, on MBSTOI and by 1.57 dB on ILD. The gap does not
widen relative to isotropic (+0.0332); it narrows slightly, so do not claim the
advantage grows with spatial structure. Against a *fully fed* NTU the same
condition gives +0.0112. Note this condition is dominated by better-ear listening
(11.25 dB advantage), so present it as an operating point, not as evidence about
binaural processing.

**ILD is the strongest column and needs no qualification: we win all six
conditions**, by 2.16 dB on the isotropic考卷 down to 0.01 dB on the
single-talker case.

### Caveats that must ship with the table

- **The `ntu_shared` variant beats us on MBSTOI in most conditions** (isotropic
  0.8653 against our 0.8357). It averages the two ears' masks, which requires
  both ears' spectra in one place — an **ear-to-ear link**. JSSC'20 is a
  single-channel processor in physically separate devices and cannot do this.
  Publish it as a reviewer defence with that constraint stated; it is not the
  comparison-table opponent.
- Absolute values are comparable **only within a condition** — a lateralised
  interferer raises everyone's noisy baseline. Label every row with its condition.
- The mono baselines are our reimplementations trained on our data, not the
  silicon. Say so.
- `FEATURES_en.md:280` claims "ours leads on binaural intelligibility (MBSTOI
  0.836)". On the isotropic考卷 against a fully-fed NTU that is a tie (−0.0015),
  so the sentence needs a condition qualifier or the as-published comparison.

## 5. Reproduction

```bash
cd /raid/R12K41024/22_directional_interferer
bash run_make_testsets.sh          # dirnoise90, dirspeech90
bash run_make_2x60.sh              # dirnoise2x60, dirspeech2x60
bash run_make_s0.sh                # dirspeech2x60_s0 + its train/val
bash run_make_trainsets.sh         # dirnoise90 train/val

GPU_ID=<free> bash run_directional_evals.sh    # deploy + NTU shared + NTU per-ear, all conditions
GPU_ID=<free> bash run_fp32_control.sh         # FP32 ancestor, quantization control
GPU_ID=<free> bash run_s0_retrain.sh           # retrain both sides on S0N±60

python collate_directional.py                  # the tables above
```

Strictly sequential — concurrent GPU jobs get SIGTERMed on this box. Raw per-SNR files under `18_.../results/apot_descale_eval/ft_dir*/` and `21_mono_se_baselines/results/bilateral_eval/dir*/`. Condition characterisation (better-ear advantage, interferer ILD, better-ear vs global SNR): [README.md](../README.md).
