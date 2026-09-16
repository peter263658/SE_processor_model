# 22_directional_interferer — does the binaural model do anything a mono one cannot?

## Why this exists

The deploy考卷 (`BCCTN/Dataset/noisy_testset`) mixes every target with **isotropic
noise**: 37 uncorrelated sources placed every 5° across the whole azimuthal plane
([prepare_data.py:248](../BCCTN/prepare_data.py) `create_isotropic_noise`).

A diffuse field is the acoustic condition in which binaural processing has the
*least* to exploit — there is no interferer direction to null, so the binaural
gain collapses to better-ear listening plus decorrelation, which a per-ear mono
model partly gets for free. That is the most likely reason
`21_mono_se_baselines` found MBSTOI **cannot separate the paradigms**: mono
per-ear ties the binaural deploy model (VCTK 0.8372 vs 0.8357), and mono with a
shared mask beats it outright (0.8653).

This directory rebuilds the same test pairs with the interferer **concentrated at
a single azimuth**, so the spatial variable is actually present.

## Design — one variable changed, everything else pinned

| | isotropic (existing) | directional (here) |
|---|---|---|
| clean targets | `clean_testset` | **the same files, verbatim** |
| SNR grid | −6…15 dB, 8 points | same |
| SNR definition | `mix_speech_and_noise` | the same function |
| file naming | `{base}_az{AT}_snr{S:+.1f}.wav` | same → evals run unchanged |
| interferer | 37 sources, −90…90° every 5° | **one source at AT ± 90°** |

Measured on `p225_p225_009_az-40` @ −6 dB — target at −40° (left):

| | broadband ILD |
|---|---|
| clean target | +4.54 dB |
| isotropic interferer | **+0.15 dB** (no spatial handle) |
| directional interferer (az +50°) | **−9.97 dB** (opposite side) |

## Alignment with Westhausen et al., TASLP 2024

The design target is N. L. Westhausen, H. Kayser, T. Jansen, B. T. Meyer,
"Real-Time Multichannel Deep Speech Enhancement in Hearing Aids: Comparing
Monaural and Binaural Processing in Complex Acoustic Scenarios," IEEE/ACM TASLP
vol. 32, 2024 ([arXiv:2405.01967](https://arxiv.org/abs/2405.01967)) — the paper
whose finding this experiment reproduces:

> all algorithms perform similarly **in diffuse noise**, [while] the binaural
> deep learning approach performs best **in the presence of spatial interferers**

Their conditions and how ours line up:

| | Westhausen 2024 | here | note |
|---|---|---|---|
| diffuse condition | S0Ndiff, cafeteria ambisonics | existing isotropic考卷 | both give a paradigm tie |
| spatial condition | **S0N±60**: two IFFM talkers at ±60° | **`dirspeech2x60`** | the headline condition |
| interferer type | competing speech (fluctuating) | competing talker | matched |
| mics per ear | 2 (4 channels) | **1 (2 channels)** | our spatial ceiling is lower |
| metric | MSBG + MBSTOI | plain MBSTOI | absolute values not comparable |
| SNR | better-ear SNR | global SNR | see below |
| their result | S0Ndiff 0.44 = 0.44; S0N±60 binaural 0.49 > monaural | — | |

### SNR definition

Westhausen sets the **better-ear** SNR; every set here keeps the **global** SNR
of the existing isotropic考卷, so that `isotropic → directional` changes only the
interferer's spatial distribution and nothing else. The two definitions diverge
by a condition-dependent amount, measured over 40 files per point:

| condition | better-ear advantage | better-ear SNR @ nominal −6 dB | interferer ILD |
|---|---|---|---|
| isotropic | 4.53 dB | −4.39 | **0.23 dB** |
| 1 interferer, 90° (noise) | **11.25 dB** | +0.45 | 6.79 dB |
| 1 interferer, 90° (speech) | **10.10 dB** | −0.36 | 5.64 dB |
| 2 interferers, ±60° (noise) | 5.13 dB | −3.01 | 4.45 dB |
| **2 interferers, ±60° (speech)** | **3.63 dB** | −3.97 | **2.62 dB** |

`dirspeech2x60` is the condition to read: spatial information is present
(interferer ILD 2.62 dB against isotropic's 0.23 dB) while the better-ear
advantage (3.63 dB) is *below* isotropic's 4.53 dB. Any gain there has to come
from binaural processing, because picking the cleaner ear cannot deliver it. The
single-interferer sets do the opposite — an 10-11 dB better-ear advantage, and
~5 dB easier at the same nominal SNR — which is why they are contrasts, not the
primary condition.

Consequences: model-vs-model comparison inside one condition is unaffected (the
files are identical); **cross-condition** comparison is affected, because a fixed
global SNR is a different difficulty per condition — another reason the
single-interferer sets are unsuitable as the primary condition, since they are
~5 dB easier at the same nominal SNR. Against Westhausen, a matched nominal
number means *their* condition is harder, so a higher score here is not evidence
of a better system.

**Why two symmetric interferers is the condition that matters.** With a single
lateral interferer the far ear is simply cleaner, so better-ear listening carries
most of the benefit — and a per-ear mono model collects it for free (MBSTOI
models better-ear explicitly). Flanking the target at ±60° leaves *no* clean ear,
better-ear cannot carry the score, and only genuine binaural processing helps.
`dirnoise90` / `dirspeech90` are kept as the single-interferer contrast.

Three conditions are built:
- **`dirspeech2x60`** — two competing talkers at AT±60°, targets restricted to
  |AT| ≤ 30° so the symmetric pair always fits the HRIR grid. **Primary.**
- **`dirnoise90`** — NOISEX-92 at one azimuth. Cleanest zero-shot test: the same
  noise corpus the models trained on, *only* the spatial distribution changed.
- **`dirspeech90`** — one competing talker. Isolates the single-vs-two
  interferer effect against `dirspeech2x60`.

## Reproduce

```bash
bash run_make_testsets.sh                        # CPU, builds both sets
GPU_ID=3 bash run_directional_evals.sh           # deploy + NTU shared + NTU per-ear
COND=dirnoise90 GPU_ID=3 bash run_directional_evals.sh   # one condition only
```

Evals write to `18_.../results/apot_descale_eval/ft_{cond}_deploy/` and
`21_mono_se_baselines/results/bilateral_eval/{cond}_ntu_{shared,independent}/`.

## Reading the results — caveats fixed in advance

1. **Do not compare across conditions.** A lateralized interferer raises the
   *noisy* baseline (one ear is simply cleaner), so directional MBSTOI will look
   higher for everyone. Only within-condition comparisons mean anything, and the
   deploy gates (VCTK 0.8357 / TIMIT 0.7898) belong to the isotropic考卷 only.
2. **Better-ear is not binaural unmasking.** MBSTOI models both. A mono per-ear
   model can ride the better-ear effect alone, so a win on the single-interferer
   conditions attributes to better-ear, not to binaural processing. The
   two-interferer condition is the one that isolates the binaural mechanism.
3. **This is zero-shot.** Every model here was trained on isotropic noise and has
   never seen a directional interferer. A null result therefore does *not* show
   binaural processing is useless — it would show our training distribution never
   gave the model a reason to learn spatial filtering. The follow-up in that case
   is to add directional interferers to *training*, not to drop the claim.
4. **Report both conditions.** Isotropic is where the paradigms tie; that stays
   in the paper. Publishing only the favourable condition would be
   cherry-picking, and the tie is defensible once the mechanism is explained.
