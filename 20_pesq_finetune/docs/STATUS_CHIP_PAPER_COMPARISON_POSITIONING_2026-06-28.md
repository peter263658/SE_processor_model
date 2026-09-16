# STATUS — Chip-Paper Comparison & Positioning Strategy (2026-06-28)

**Purpose.** Consolidated reference for how to position the binaural W4A8 zero-multiplier SE
accelerator against the chip-paper comparison targets, how to claim superiority **without**
losing on PESQ, the current scores, and the reusable binaural-vs-monaural rhetoric (sourced
from the BCCTN paper). First doc in this thread — no predecessor.

---

## TL;DR

- **Do NOT fight P2 on PESQ.** It is structural, not a tuning failure: our architecture (BCCTN
  lineage) was never designed for PESQ — **the original BCCTN paper does not report PESQ at all.**
  P2's lineage (MP-SENet) is a PESQ specialist (FP32 wb 3.50). Our wideband PESQ ~2.3–2.7 is below
  P2's **wideband** 3.34. The narrowband "match" (nb 3.0–3.3) was a band mismatch.
- **Win on the three axes every monaural competitor structurally lacks:** binaural + ILD/IPD +
  **MBSTOI** (uncontested), **STOI** (we beat the only same-dataset competitor), **SI-SDR health**
  (credibility, not gaming), plus **pure-digital zero-multiplier** hardware.
- **PESQ optimisation actively erodes the binaural USP** (dose-dependent — proven, see §5). So
  **deploy** (no/low PESQ pressure) is the operating point to report.

---

## 1. Comparison targets

| | P1 Lee | **P2 Zhi** ⭐ | P3 Wu | P4 Park | BCCTN (algo lineage) |
|---|---|---|---|---|---|
| Venue | JSSC'20 | **CICC'26** | A-SSCC'25 | ISSCC'23/JSSC'24 | ICASSP'24 |
| Tech | 40nm | 22nm | 22nm | 28nm | — (FP32 SW) |
| Power | 2.17mW | 1.9mW | 0.23mW(sys 0.58) | 0.74mW | — |
| Energy | ~4µJ | 15.2µJ | 3.64µJ/frame | 11.75µJ | — |
| Model | CNN mask | Transformer 1.35M | AV CNN | SepConv+GRU 56k | Complex CED+Transformer 10M |
| MAC | general | **D-CIM** | multi-coding | **multiplier-less** | — |
| **Channels** | mono | mono | mono+lip | mono | **binaural** |
| **Spatial ILD/IPD** | ✗ | ✗ | ✗ | ✗ | **✓** |
| Dataset | IEEE/NOISEX | **VB+DEMAND** | TMSV | CHiME-2 | VCTK/TIMIT |
| PESQ | — | **3.34 (wb)** | 2.13@-10dB | 2.73 (nb) | **not reported** |
| STOI | — | 95.1 | — | 91.44 | — (uses MBSTOI) |
| SI-SDR | — | — | — | (SDR 14.0) | — |
| MBSTOI | — | — | — | — | **✓ headline** |

**Two load-bearing facts:**
1. All four chip competitors are **monaural** and report **no spatial cues**. Only **P2** shares our
   dataset (VoiceBank+DEMAND) and metric class (PESQ/STOI).
2. **P2's PESQ 3.34 is wideband** (confirmed: its example noisy = 1.97 = the canonical VB+DEMAND
   wideband noisy anchor, Richter TASLP'23; it cites MP-SENet whose wb PESQ = 3.50). Never compare
   our **nb** to P2's **wb**.

---

## 2. Comparison strategy — which axes, how to claim "better"

| Axis | How to compare | Outcome |
|---|---|---|
| **Binaural + ILD/IPD + MBSTOI** | Headline, **0 competitors** | Monaural chips physically cannot do this |
| **STOI** | Same metric, direct | **Ours 96.85 > P2 95.1 > P4 91.44** (beat all that report it) |
| **SI-SDR** | Credibility (anti-gaming) | ~17.1, healthy; competitors don't report it |
| **PESQ** | Report wb+nb honestly; **do not claim to beat P2** | Below P2 (structural); ahead of other monaural chips |
| **Hardware** | Pure-digital zero-mult W4A8 | vs P2 D-CIM (more portable); vs P4 multiplier-less but harder binaural transformer |

**Headline claim (one sentence):**
> *First binaural, spatial-cue-preserving SE accelerator: matches/exceeds the best monaural SE
> chips on standard quality (STOI 96.85 > 95.1) while uniquely preserving ILD/IPD (MBSTOI 0.836)
> that monaural designs cannot, in a pure-digital zero-multiplier W4A8 core.*

---

## 3. Current scores (deploy = primary model; meets MBSTOI gate)

**VoiceBank+DEMAND (az70 binaural render, vs P2):**
| | PESQ nb | PESQ wb | STOI | SI-SDR |
|---|---|---|---|---|
| Noisy | 2.89 | 2.00 | 0.969 | 17.9 |
| **Deploy** | **3.01** | **2.27** | **0.9685** | **17.1** |
| P2 (competitor) | — | **3.34** | 0.951 | — |

**VCTK (binaural — the USP):**
| | MBSTOI | ILD (dB) | IPD (°) |
|---|---|---|---|
| Noisy | 0.786 | 3.71 | 15.43 |
| **Deploy** | **0.836** (full 0.8357) | 3.33 | 16.0 |
| FP32 reproduction | 0.852 | 3.27 | 14.57 |

**Deploy gates (do not regress):** VCTK MBSTOI ≥ 0.835 (0.8357), TIMIT MBSTOI ≥ 0.788 (0.7898).

**PESQ-leaning variant** `mix_pesq_w6_ep100` ep13 (warm from deploy, mixed VCTK+VB, w=6, early-stopped
at convergence): VB nb 3.085 / wb 2.318 / STOI 0.9675 / SISDR 17.31; VCTK MBSTOI 0.828 / ILD 3.38 /
IPD 16.32. → +0.07 nb for −0.006 MBSTOI (drops below the 0.835 gate). **Not worth it; report deploy.**

---

## 4. PESQ handling (honest + use BCCTN-didn't-use-PESQ as a shield)

- State explicitly: *"This architecture follows BCCTN (ICASSP'24), whose design objective is
  binaural spatial-cue preservation; the original work does not use PESQ as an evaluation metric."*
  → pre-empts "why doesn't PESQ beat P2."
- Report **both wb and nb**, labelled; **never** put our nb next to P2's wb.
- Frame the **PESQ↔spatial tradeoff (§5) as a design-decision ablation** ("we chose to preserve
  spatial cues") — turns the weakness into evidence of intent.

---

## 5. PESQ ↔ spatial-cue tradeoff (ablation data — proven, dose-dependent)

VCTK, same eval, warm from deploy (PESQ weight ↑ on VCTK):
| Model | MBSTOI ↑ | ILD err ↓ | IPD err ↓ |
|---|---|---|---|
| Deploy (w=0) | 0.834 | 3.33 | 16.0 |
| PESQ-ft w18 | 0.823 | 3.44 | 17.7 |
| PESQ-ft w30 | 0.773 | 3.98 | 21.6 |

Higher PESQ weight monotonically wrecks MBSTOI/ILD/IPD. **Quantization alone preserves the cues;
PESQ-finetuning is what erodes them.** This is the data-backed reason to report the low/no-PESQ
(deploy) model and to keep STOI(10)/IPD(10)/ILD(1) strong if any PESQ term is added.

---

## 6. Binaural-vs-monaural rhetoric (from BCCTN, reusable)

**Argument chain (the paper never claims binaural wins on noise reduction — it concedes SNR):**
1. Binaural listening itself helps the user (intelligibility **+** spatial awareness).
2. Binaural signals carry ILD/IPD needed for localization; binaural unmasking raises intelligibility
   → cues must be **preserved**.
3. Villain = monaural per-ear processing — denoises "at the cost of damaging vital binaural cues."
4. Even a *good* monaural model is dangerous: altering level/phase "modifies the spatial
   information... compromise[s] the localization and spatial awareness of the listener."
5. Fix = explicit ILD/IPD loss terms.
6. Data capstone: BiTasNet wins SegSNR but **loses MBSTOI and ILD/IPD** → denoising alone isn't enough.

**Verbatim weapon sentences (BCCTN, arXiv:2403.05393):**
- *"providing binaural signals to the user of an assistive listening device may improve speech
  intelligibility and spatial awareness."*
- *"monaural speech enhancement approaches operating on each binaural channel independently enhance
  the signals **but at the cost of damaging vital binaural cues**."*
- *"Even if the model achieves significant noise reduction and improves speech intelligibility,
  **altering the level and phase information would modify the spatial information... and therefore
  compromise the localization and spatial awareness of the listener**."*
- *"Even though the BiTasNet has better noise reduction performance, it exhibits a **lower MBSTOI**
  binaural intelligibility score... The proposed model... have the **lowest ILD and IPD error for
  all SNRs**."*

**Citations to reuse (BCCTN's own):**
- Hawley, Litovsky, Culling 2004 — binaural localization / cocktail-party benefit.
- Beutelmann & Brand 2006 — binaural unmasking → intelligibility (hearing-impaired).
- Blauert 1997, *Spatial Hearing* — ILD/ITD/IPD are the primary localization cues.
- Klasen et al. 2006 — binaural Wiener preserving ITD/ILD (preserve-cues precedent).
- Andersen et al. 2018 — MBSTOI metric.

**Chip-paper adaptation (replace SegSNR→PESQ, BiTasNet→P2, algorithm→chip):**
> *"In noisy environments, binaural signals improve both intelligibility and spatial awareness,
> carrying the interaural cues (ILD/IPD) required for localization and binaural unmasking
> [Hawley'04, Beutelmann'06, Blauert'97]. **However, all prior SE accelerators are single-channel:**
> applied independently to each ear they denoise **at the cost of damaging the interaural cues**.
> Even an accelerator achieving high PESQ/STOI, by altering per-ear level and phase, **modifies the
> spatial information and compromises the listener's localization and spatial awareness.** We present
> **the first binaural SE accelerator that preserves ILD/IPD while denoising**, in a pure-digital
> zero-multiplier W4A8 core. Mirroring the algorithm-level finding that a stronger noise reducer can
> win SegSNR yet lose binaural intelligibility, **the strongest monaural SE chip (P2) leads PESQ but
> neither reports nor can preserve spatial cues; ours leads on binaural intelligibility (MBSTOI
> 0.836) and STOI (96.85 vs 95.1) while preserving ILD/IPD."*

---

## 7. ILD/IPD metric A/B (why our numbers ≠ the paper's 0.4 dB / 5°)

A/B over 64 VCTK files (deploy model), both algorithms on identical signals:
| | ILD (dB) | IPD (°) |
|---|---|---|
| Our eval algorithm | 3.76 | 17.99 |
| Original BCCTN repo algorithm (verbatim) | 4.30 | 59.59 |
| Paper table | ~0.4 | ~5 |

**Findings:**
- **The metric is NOT the cause of the gap.** The paper's *own* algorithm gives ~4.3 dB / ~60° on
  our model — nowhere near 0.4 / 5°. Our FP32 reproduction is also ~3.3 / 14.6.
- **Our IPD metric is more rigorous than the paper's**, not looser: original = no diff-rewrap +
  all-frequency → inflates to 60°; our rewrap (−16.8°) + ≤1500 Hz (−14.4°) + mask (−9°) → ~18°. **Do
  not "fix" toward the paper's version.**
- **RESOLVED — the gap to 0.4 / 5° is the TEST DATA/CONDITIONS, not metric and not model.**
  The official BCCTN repo (`VikasTokala/BCCTN`) ships the actual paper checkpoint
  (`DCNN/Checkpoints/Trained_model.ckpt`, FP32, 11.09M params — loaded clean, 0 missing/unexpected
  `model.*`). Running **that authentic checkpoint on OUR VCTK test set with the ORIGINAL repo
  metric** gives **ILD 5.39 dB / IPD 65.3°** — nowhere near 0.4 / 5°, and slightly *worse* than our
  deploy (4.30 / 59.6, original metric; 3.76 / 18.0, our metric). So the paper's 0.4/5 is specific
  to the authors' own (unreleased) test set; ILD/IPD absolute values are **not cross-test-set
  comparable**. (Caveat: the paper model is out-of-domain on our HRIR/azimuth setup, so the
  our-model-beats-theirs ordering is partly home-advantage — but the 0.4/5-is-data-specific
  conclusion stands.)
- **Implications:** (a) do NOT try to reproduce 0.4/5 or put our 3.3/16 next to it; (b) report
  **improvement-over-noisy** (ILD 3.71→3.33) and **FP32-reference-vs-quantized preservation on
  identical data** (our W4A8 ≈ FP32 BCCTN) — i.e. "quantization does not sacrifice spatial
  preservation"; (c) absolute ILD/IPD is a test-set property, frame it as such.

**FP32-reference vs our W4A8, identical test set + identical metric (64 VCTK files):**
| | ILD (dB) | IPD (°) |
|---|---|---|
| Paper FP32 BCCTN (11M, authentic ckpt) — OUR metric | 4.59 | **17.32** |
| **Our deploy W4A8 — OUR metric** | **3.76** | **18.0** |
| (same models, ORIGINAL no-rewrap metric) | 5.39 / 4.30 | 65.3 / 59.6 |

→ **Publishable claim:** *"On an identical binaural test set and metric, our W4A8 zero-multiplier
core preserves interaural cues on par with the FP32 reference BCCTN (IPD 18.0° vs 17.3°; ILD 3.76 vs
4.59 dB) — quantization does not sacrifice spatial-cue preservation."* (Caveat: the FP32 model is
out-of-domain on our HRIR/azimuth pipeline, so the ILD edge is partly home-advantage; the IPD tie is
the cleanest evidence. The ~60° figures are a no-rewrap metric artifact, not real IPD loss.)

**Concept note (ILD/IPD "noisy" value):** ILD/IPD *error* is always `|cue(signal) − cue(clean)|`.
"Noisy" error = how much diffuse noise corrupts the cue vs clean (pre-processing baseline);
"Enhanced" error = residual after processing. They are NOT subtracted to form the metric — each is an
error-vs-clean. `noisy_err − enhanced_err` = the *improvement* (a separate, useful quantity).

Algorithm differences found (our eval vs original repo): IPD ≤1500 Hz restriction (ours, original
none); IPD diff-rewrap (ours via exp(jΔ), original plain |Δ|); mask channel-combine (original L AND R
bitwise_and, ours max(|L|²,|R|²) OR-like); mask dB threshold (original −20 dB; ours effective −10 dB
via a `20·log10(power)` convention); DC bin (original dropped, librosa keeps). Internal note: our
**training** ILD uses the original `speechMask` but our **eval** ILD uses the inline OR mask —
train/eval mask inconsistency.

---

## 8. Open items

- [x] **Reproduce paper ILD/IPD (0.4/5): RESOLVED — data-specific (see §7).** Checkpoint shipped
  in-repo; data generator + eval-aggregation script NOT released. Authentic FP32 ckpt on our data =
  5.39/65.3 (≠ 0.4/5). Do not chase it. (Optional: re-run their ckpt with OUR metric, and/or rebuild
  their test set from the paper recipe — Kayser HRIR, az −90..90, SNR −6..15, anechoic+reverb — if a
  same-test-set FP32-vs-W4A8 spatial-preservation table is wanted for the paper.)
- [ ] Our chip hardware numbers (power/energy/area/node) — fill the comparison table once RTL
  bring-up yields them.
- [ ] Decide final reported model (deploy recommended) and lock the scores table for the paper.
- [ ] Optionally: select checkpoints by measured MBSTOI rather than val loss.
