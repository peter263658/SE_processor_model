# STATUS: Chip-Paper Experiment Inventory & Open Questions

Date: 2026-07-10
Scope: BCCTN binaural speech-enhancement ASIC (W4A8 zero-multiplier core) — quality/positioning work for the chip paper.
Companion results: [RESULTS_CHIP_COMPARISON_FINAL_2026-07-09.md](RESULTS_CHIP_COMPARISON_FINAL_2026-07-09.md), [RESULTS_SPATIAL_CUE_ITD_IC_2026-07-10.md](RESULTS_SPATIAL_CUE_ITD_IC_2026-07-10.md)

## 1. One-paragraph project state

We have a **frozen, deployable W4A8 zero-multiplier binaural denoiser** (every Conv/MHA/BN/PReLU/softmax multiply realized as shift-add). Its quality gates are met (VCTK MBSTOI 0.8357, TIMIT 0.7898). The paper's thesis is: **we keep binaural spatial cues (ILD/IPD) that single-channel competitors structurally cannot**, at silicon-grade efficiency. The last quarter of work has (a) pushed PESQ via finetuning without regressing spatial cues, (b) established a rigorous FP32→W4A8 scaling story, and (c) built the head-to-head evidence that mono networks break localization. Algorithm side is essentially done; RTL bring-up (bit-exact vs golden) is the open hardware item.

## 2. What is established (all numbers measured, 750-pair VCTK / 160-pair TIMIT, −6..15 avg)

### 2.1 Deploy model (the chip) — W4A8, frozen
| set | MBSTOI | STOI | PESQ wb | PESQ nb | ILD↓ | IPD↓° | ITD↓μs | IC↓ |
|---|---|---|---|---|---|---|---|---|
| VCTK | 0.8357 | 0.868 | 1.593 | 2.299 | 3.31 | 15.9 | 5.24 | 0.237 |
| TIMIT | 0.7898 | 0.823 | 1.656 | 2.230 | 3.08 | 20.4 | 4.08 | 0.240 |

### 2.2 PESQ-finetuned W4A8 variants (same hardware datapath as deploy)
| variant | VCTK PESQ wb | VCTK MBSTOI | VCTK ILD↓ | note |
|---|---|---|---|---|
| Plain-QAT w12 | 1.866 | 0.821 | 3.49 | max PESQ, mild spatial cost |
| Rebalance (ILD4/IPD12) | 1.857 | 0.823 | 3.35 | ~dominates plain-QAT; SI-SDR +9.21 dB |
| KD+spatial | 1.760 | 0.829 | 3.37 | teacher-bounded, milder trade |
All three lift PESQ ~+0.26 over deploy while holding ILD within ~0.2 dB.

### 2.3 FP32-512 ceiling + measured quantization scaling
- FP32-512 original, torch_pesq-finetuned to saturation: **PESQ wb 2.291, Δ vs noisy +0.93** (> P4 silicon Δ+0.78).
- Measured deploy/FP32 ratio on enhanced PESQ ≈ **0.822**; Δ-survival through quantization ≈ **59%**. Predicted deploy-achievable band contained the QAT-measured point.

### 2.4 Competitor comparison (−6..9 dB axis, wideband PESQ; P2 excluded per decision)
| paper | enh PESQ | ΔPESQ | spatial cues | modality |
|---|---|---|---|---|
| P1 NTU JSSC'20 | 2.38 | +0.43 | ✗ | audio |
| P4 SNU JSSC'24 (silicon) | 2.73 | +0.78 | ✗ | audio |
| P3 A-SSCC'25 (silicon) | 2.13 @−10dB | — (no avg) | ✗ | **audio+video (lips)** |
| Ours (rebalance) | — | +0.51 | ✓ | audio |

### 2.5 Mono-paradigm head-to-head (reproduced SNU/NTU, run bilaterally, FP32)
- SNU: VCTK ILD **5.71 dB (+54% vs noisy 3.71)**; NTU: **5.47 (+47%)**. Our W4A8 deploy: **3.31 (improves on noisy)**.
- **32/32 ILD test points: mono degrades, ours preserves.** This is the load-bearing USP argument.

### 2.6 Spatial-cue metric study (this session)
Added ITD (GCC-PHAT) + IC (MSC). Finding: **ITD is preserved by everyone (<JND, not a discriminator); IC slightly favors mono (over-coherence artifact) — do not feature it. ILD (+IPD) remains the only discriminating spatial axis.**

### 2.7 Robustness / breadth
- Low-SNR zero-shot (train floor −7 dB): −10 dB MBSTOI **Δ+0.15**, audio-only, vs P3's video-aided 2.1@−10dB.
- VoiceBank+DEMAND: deploy nb 3.00; a VB-finetuned variant reaches nb **3.334** / wb 2.734 (P2 territory; P2 excluded from the table).

## 3. Open tasks (algorithm side)

| priority | task | cost | status |
|---|---|---|---|
| low | Backfill ITD/IC for KD+spatial, FP32 lineage, 09_tied ancestor, low-SNR, VB+DEMAND | ~1–1.5 hr, 1 GPU | not started; won't change conclusions |
| low | ILD_W=6 sweep to push ILD < deploy 3.31 | ~1 hr train + eval | prepared |
| low | sisdr_tied loss A/B (scale-tied binaural SI-SDR) | ~2 hr | prepared, not launched |
| low | Write SI-SDR into paper-style `_averages.txt` (computed but not summarized) | trivial | code touch |
| deferred | MetricGAN-on-deploy port | ~3 hr | deferred |

## 4. Open questions for other consultants

1. **Where are the silicon efficiency numbers?** This is a chip paper (JSSC/ISSCC/A-SSCC audience). Quality is covered exhaustively, but the competitive table has **no pJ/inference, mm², throughput, or TOPS/W vs P1/P4** yet. SSCC reviewers weight efficiency as much as quality. Is post-synthesis power/area from the RTL ready to put in the table? **Likely the single highest-leverage gap.**
2. **Headline model choice:** lead with **deploy** (strongest spatial USP, PESQ 1.59) or **rebalance** (PESQ 1.86, ILD ~equal)? Affects the whole narrative.
3. **Axis choice:** −6..9 (CHiME-2 alignment, fair) vs −6..15 (more favorable). Which does the venue expect?
4. **Is ILD + objective metrics enough to defend the spatial USP, or does it need a listening test** (MUSHRA / localization-error study)? ITD/IC did NOT yield a competitive win beyond ILD, so the objective spatial case rests almost entirely on ILD.
5. **Is a better-differentiating spatial metric available** (e.g., binaural MBSTOI advantage, spatial release from masking, BMLD, model-based azimuth error) where the binaural model genuinely beats mono by more than ILD does?
6. **Is chasing PESQ even the right objective** given the differentiator is spatial preservation, not raw PESQ (where P4 silicon still leads on absolute score)? Or is "competitive PESQ **plus** uniquely-preserved spatial cues **at** this power/area" the correct framing?

## 5. Reproduction
- Spatial-cue chain: `bash 20_pesq_finetune/run_itdic_eval_chain.sh` (auto-picks free GPU; SNU→NTU→deploy→rebalance→plain-QAT).
- Deploy eval: `deploy-eval-rerun-recipe` (dir-18 `run_apot_descale_eval.sh`, `ACT_FP8_MODE=calibrated_pow2_e4m3` + calib json).
- Finetune-variant eval: `20_pesq_finetune/code/run_pesq_eval.sh` (CKPT/TAG/GPU_ID).

## 6. External (Codex) review corrections — appended 2026-07-10

Two external review passes over §1–§5 and the w24/MetricGAN work. Corrections adopted:

### 6.1 Statistical corrections (framing MUST follow these)
- **w24_ep16 vs reproduced SNU (VCTK −6..9, paired cluster bootstrap):** PESQ **+0.0314**, 95% CI [+0.0178, +0.0450] — a real win, not rounding. MBSTOI **−0.0216**, 95% CI [−0.0240, −0.0193] — a real loss. **Never describe w24's MBSTOI as "tied/parity" with mono baselines.**
- **"ILD positive" must be dataset-scoped:** w24_ep19 TIMIT full ILD 3.2943 vs noisy 3.2902 (marginally worse). The w24 "ILD improves" claim holds on VCTK −6..9 only.
- **MBSTOI is an intelligibility gate, not a spatial-fidelity metric.** Its EC stage searches ±20 dB / ±1 ms compensation and falls back to better-ear; bilateral mono processing that destroys ILD can still score at parity (observed: deploy 0.8357 vs SNU 0.8383 / NTU 0.8372 while ILD is 3.31 vs 5.71/5.47). Spatial claims ride on ILD (+IPD); MBSTOI rides intelligibility.

### 6.2 Attribution renaming
- "Quantization scaling ratio" → **"FP32-to-deploy stack retention"**: the 0.822/0.867 ratios bundle FFT 512→128, model downsizing, structural compression (tied MHA/grouped/top-4) AND W4A8 — not pure quantization. On VCTK, 0.160 of the 0.245 baseline-to-baseline gap is already present at the FP32-128 lineage ancestor (1.838→1.678→1.593). Fair deltas: baseline-to-baseline **0.245**; PESQ-optimized-to-PESQ-optimized 2.291→1.856 = **0.435**. A capacity-matched FFT-only A/B was never run; do not attribute the gap to FFT alone.

### 6.3 Priority re-ordering (adopted)
1. **Gate-holding PESQ fine sweep (running):** w6 (PESQ_W=6, lr 1e-5) trajectory was MBSTOI 0.8273→0.8301→**0.8337** (ep2→3→4, still rising when the 5-epoch run ended) with wb 1.72–1.75 and **TIMIT gate already passed (0.7888)**. Chain: R1 continue w6-ep4 (PESQ6/STOI12/lr 5e-6), R2 PESQ4/lr 5e-6, R3 PESQ2/lr 1e-5, R4 PESQ4+sisdr_tied/lr 5e-6; every epoch screened full VCTK+TIMIT (scaled); candidates (v≥0.834 & t≥0.787) → **canonical dir-18 calibrated_pow2_e4m3 eval decides the gate** (dir-20 scaled evals are screens only). Realistic target: gate-passing wb ≈1.68–1.75 (vs deploy 1.593); beating SNU's Δ+0.562 while holding the gate has no supporting evidence yet.
2. **MetricGAN-on-deploy: standby, deprioritized.** FP32 MetricGAN's proven −6..9 gain is +0.098; beating SNU needs ≈+0.30. v2 fixes implemented & compiled (D warm-up epochs, PESQ-failure label exclusion, noisy-pair D calibration, D freeze in G-step, NaN-sanitize+clip 1.0, D-vs-true-PESQ monitoring (MAE/Pearson/Spearman), BCCTN_D_INIT_CKPT warm init — FP32 critic at fp32_512_pesq_ft/outputs/metricgan_w12 has 20 disc.* tensors).
3. KD lineage deprioritized (KD+spatial 1.760 < rebalance 1.857).

### 6.4 Loader fix for canonical gate evaluation
dir-18 `code/eval.py` and dir-20 `code/eval.py` now ignore ("loss.", "disc.", "base_loss.") state-dict keys (strict=False + explicit mismatch check), so PESQ-finetuned / MetricGAN ckpts can be judged on the canonical path.

## 7. Gate sweep + rescue final verdict — appended 2026-07-11

Sweep(R0–R4, 27 ckpts)+ §4(a) rescue 全部收官:**NO_GATE_PASS + RESCUE_NO_PASS**。完整判決、offset 定律(finetuned 權重 canonical 一律低於 scaled 0.0015–0.0030)、量測 Pareto(gate ⇒ wb <~1.71)、Tier-2 蓋棺(w24 僅 wb 贏 SNU、nb 仍輸)、贏家詛咒教訓,見 [RESULTS_GATE_SWEEP_RESCUE_2026-07-11.md](RESULTS_GATE_SWEEP_RESCUE_2026-07-11.md)。

採納建議:兩點故事 — deploy ep3(gate/空間主角,不動)+ `sw_w6cont_s12_v4e1`(PESQ-boosted 副點,canonical 0.8308/0.7886、wb 1.7092/1.7531、ILD 3.3363)。§6.3 之 1(gate-holding sweep)關閉;之 2(MetricGAN v2)轉 future work,為僅存的均值移動機制。
