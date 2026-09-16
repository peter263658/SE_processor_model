# TODO: Adversarial review checklist — gate-holding PESQ sweep + MetricGAN v2 standby

Date: 2026-07-10
For: external reviewer (Codex) — parallel adversarial review alongside the internal multi-agent review.
Goal under test: find a W4A8 checkpoint passing canonical VCTK MBSTOI >= 0.835 AND TIMIT >= 0.788 while maximizing wb PESQ.

## A. Artifacts to review (all new/changed today)

| # | file | what it does |
|---|---|---|
| A1 | `20_pesq_finetune/run_gate_sweep_train.sh` | trainer: 4 sequential finetunes (R1 w6cont PESQ6/STOI12/lr5e-6 from `pesq_w6_lr1e-5_ep5/.../all-epoch=4-*.ckpt`; R2 PESQ4/lr5e-6; R3 PESQ2/lr1e-5; R4 PESQ4+sisdr_tied/lr5e-6), self-heal resume from last.ckpt, writes SWEEPTRAIN_DONE |
| A2 | `20_pesq_finetune/run_gate_sweep_screen.sh` | screener: polls `all-epoch=*.ckpt` under 4 tags, full VCTK+TIMIT 750 scaled screen per epoch, summary + candidate selection (v>=0.834 && t>=0.787, top-3 by v_wb), canonical dir-18 eval per candidate, writes SWEEP_DONE |
| A3 | `18_w4a8_pow2_mixed_lmul_hw_handoff_20260530/code/eval.py` (~line 1097) | canonical loader now ignores `("loss.","disc.","base_loss.")` keys, strict=False + explicit mismatch raise |
| A4 | `20_pesq_finetune/code/eval.py` (~line 1130) | same ignore-list extension (was `loss.` only) |
| A5 | `20_pesq_finetune/code/DCNN/metricgan.py` (v2) | D warm-up epochs; PESQ-failure labels EXCLUDED (mask, not 0.0); (noisy,clean) pairs added to D; D frozen during G step; NaN-sanitize + clip_grad_norm 1.0 both steps; D-vs-true-PESQ monitor (MAE/Pearson/Spearman/true-PESQ mean) per val epoch |
| A6 | `20_pesq_finetune/code/train_metricgan_deploy.py` | builds generator via DCNNLightningModule (full W4A8 surgery) + `_load_checkpoint` warm-start; `BCCTN_D_INIT_CKPT` env loads disc.* (FP32 critic at `fp32_512_pesq_ft/outputs/metricgan_w12/.../last.ckpt`, 20 tensors) |
| A7 | `20_pesq_finetune/code/run_metricgan_deploy.sh` | wrapper: full deploy override stack + GAN_W/D_WARMUP/LR_D knobs; SMOKE=1 mini-run |

## B. Claims to attack (most valuable review targets)

1. **Screen→canonical consistency.** Screens run dir-20 `activation_fp8_mode=scaled`; the gate verdict runs dir-18 `calibrated_pow2_e4m3` + `activation_calibration_p999_b64.json` (calibrated FOR deploy ep3). For finetuned weights, is the static calib table still valid enough that a scaled-screen pass at 0.834 plausibly survives canonical? Deploy ep3 itself matched (0.8357 both paths) — but does that generalize to finetuned ckpts, and should the screen threshold be tighter/looser?
2. **Candidate selection logic** (A2 bottom): stag round-trip `sw_w6cont_s12_v0e4 → run/version/epoch`; awk field alignment vs printf columns; `sort -k3 -rn` on v_wb; empty-candidate handling under `set -u`; failed-screen retry (presence-of-averages-file as the done marker).
3. **Trainer resume arithmetic** (A1 `run_one`): `done_n` counts all-epoch ckpts across version_* dirs; remaining = target − done_n; warm-start (weights-only, fresh Adam) — is restarting mid-run at lr 5e-6 from last.ckpt scientifically acceptable for these short runs?
4. **R1 design**: continuing w6-ep4 with STOI_W raised 10→12 and lr 1e-5→5e-6, PESQ_W kept 6. Is there a failure mode where MBSTOI rises but PESQ collapses below usefulness (w6 wb was 1.72)? Would STOI_W=11 or keeping 10 be safer? (The MBSTOI trend 0.8273→0.8337 was already rising at STOI_W=10.)
5. **R4 sisdr_tied**: `snr_loss_mode=sisdr_tied` replaces the SNR term with scale-tied binaural SI-SDR (one α for concatenated L/R) at SNR_W=1. Verify loss.py's implementation sign/weighting can't dominate at lr 5e-6, and that it doesn't fight the STOI anchor.
6. **Canonical eval of dir-20 ckpts (A3).** dir-18's DCNNLightningModule + `run_apot_descale_eval.sh` overrides must build the EXACT parameter/buffer set of dir-20 training ckpts, or strict-check raises / silently mis-evals. Diff the two `DCNN/trainer.py` QAT wrap paths if in doubt.
7. **MetricGAN v2 (A5)**: with `d_label_subset=8` and clean-anchor `(clean,clean)→1.0`, can D collapse to predicting ~1 for everything (the clean anchor is 1/3 of terms)? Is Adam for D with spectral-norm layers at lr 5e-4 stable? Is the D monitor's Spearman-on-128-samples meaningful?
8. **GPU discipline**: trainer writes `logs/gate_sweep/train_gpu`; screener avoids it. Machine shared (GPUs 0/2/4/6 occupied by another user). Max 2 GPUs total (1 train + 1 eval). Any path that violates this?

## C. Known-accepted limitations (don't re-report)

- Screens are scaled-mode by design; only canonical judges the gate.
- The "all-" ckpt series is used (save_top_k would drop the gate-optimal epoch — this already ate w6's epoch=4 non-all ckpt).
- Beating reproduced-SNU ΔPESQ (+0.562 @ −6..9) while holding the gate has no supporting evidence yet; the deliverable tiers are (1) gate-passing PESQ-max ~1.68–1.75, (2) three-axis win if it materializes.
- MetricGAN v2 is standby, not launched (Codex priority call); it has passed py_compile but NOT a GPU smoke since the v2 rewrite.

## D. Reference numbers (for sanity-checking any re-derivation)

- Deploy ep3 canonical: VCTK 0.8357/1.5932/ILD 3.308; TIMIT 0.7898/1.6558/3.085.
- w6 (scaled): ep2 0.8273/1.7280, ep3 0.8301/1.7407, ep4 0.8337/1.7176 (VCTK); ep4 TIMIT 0.7888/1.7469.
- w24_ep16 (scaled): VCTK 0.8188/1.9392; −6..9 Δwb +0.593; SNU-repro −6..9 Δwb +0.562, noisy ILD (−6..9) 4.279.
- Noisy VCTK full: MBSTOI 0.7861 / wb 1.2404 / ILD 3.714.
