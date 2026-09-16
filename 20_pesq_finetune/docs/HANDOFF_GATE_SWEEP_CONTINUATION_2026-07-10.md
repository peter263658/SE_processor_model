# HANDOFF: Gate-holding PESQ sweep — continuation checklist (for GPT/Codex)

Date: 2026-07-10 (evening; Claude quota exhausted mid-run, resets 21:10 Taipei)
Base candidate: deploy ep3 `18_w4a8_pow2_mixed_lmul/outputs/d1c3_pow2_finetune/d1c3_apotclip16_attnm8_outlierm4_15ep/.../epoch=2-step=726.ckpt`
Handoff: [REVIEW_CHECKLIST_GATE_SWEEP_2026-07-10.md](REVIEW_CHECKLIST_GATE_SWEEP_2026-07-10.md), [STATUS_EXPERIMENT_INVENTORY_2026-07-10.md](STATUS_EXPERIMENT_INVENTORY_2026-07-10.md) §6

## 1. Goal and success criteria

Find a W4A8 checkpoint that passes the **canonical** deploy gates while maximizing wb PESQ:
- **Gate (hard):** canonical VCTK MBSTOI >= 0.835 AND TIMIT MBSTOI >= 0.788. Canonical = dir-18 `run_apot_descale_eval.sh` with `ACT_FP8_MODE=calibrated_pow2_e4m3` + `BCCTN_ACTIVATION_CALIB_PATH=$PKG18/ckpt/activation_calibration_p999_b64.json`, VCTK_ONLY=0. dir-20 evals (`run_pesq_eval.sh`) are scaled-mode **screens only** — measured to read LOW by ~0.0016 (VCTK) / 0.0019 (TIMIT) vs canonical (deploy ep3: scaled 0.8341/0.7879 vs canonical 0.8357/0.7898).
- Tier 1 deliverable: gate-passing wb PESQ ~1.70+ (deploy is 1.593). Tier 2 (unlikely per analysis): also beat reproduced-SNU −6..9 Δwb +0.562.

## 2. What is running right now (launched ~20:00 2026-07-10)

Two detached processes (survive shell exit; self-healing):
- **Trainer** `20_pesq_finetune/run_gate_sweep_train.sh` (GPU 5, sequential):
  R0 `sw_w6cont_exact` — exact-recipe warm-start from w6 ep4 (same loss weights+lr; Adam/scheduler reset — not a literal optimizer continuation) (PESQ6/STOI10/ILD1/IPD10/SNR1, lr 1e-5, 4 ep) ← most likely tier-1 completer
  R1 `sw_w6cont_s12` — recovery ablation (PESQ6/STOI12, lr 5e-6, 6 ep)
  R2 `sw_w4_lr5e6` (PESQ4, lr 5e-6, 6 ep) · R3 `sw_w2_lr1e5` (PESQ2, lr 1e-5, 5 ep) · R4 `sw_w4_sisdr` (PESQ4 + snr_loss_mode=sisdr_tied, lr 5e-6, 6 ep), all from deploy ep3.
  Writes `SWEEPTRAIN_DONE` when finished trying; `SWEEPTRAIN_FAILED` if any run fell short of its epoch target.
- **Screener** `20_pesq_finetune/run_gate_sweep_screen.sh` (GPU 7):
  (a) FIRST canonical-evals the existing **w6 ep4 anchor** (`pesq_w6_lr1e-5_ep5/.../all-epoch=4-step=2420-val=5.1325.ckpt`, scaled 0.8337/0.7888 → est. canonical ~0.8353/~0.7907 — **may already pass**; result at `18_.../results/apot_descale_eval/ft_canon_w6ep4_anchor/`).
  (b) Screens every `all-epoch=*.ckpt` of the 5 tags (full VCTK+TIMIT, LIMIT=750) → `logs/gate_sweep/sweep_summary.txt`.
  (c) Candidates: scaled vMBSTOI>=0.8325 && tMBSTOI>=0.7855; global top-3 by v_wb + best per run + highest-MBSTOI fallback (wb>1.65).
  (d) Canonical-evals candidates, parses gates → `logs/gate_sweep/canonical_summary.tsv` + `logs/gate_sweep/BEST_GATE_PASS` (winner line: stag/vMBSTOI/tMBSTOI/v_wb/t_wb/vILD/PASS/ckpt) or `NO_GATE_PASS`. Writes `SWEEP_DONE` (+`SWEEP_FAILED` if retries were exhausted anywhere).
  ETA: ~6–9 hr total (trainer ~5 hr; screener trails, then ~5–8 canonical evals x 20 min).

## 3. Monitoring commands

```bash
cd /raid/R12K41024/20_pesq_finetune
tail -5 logs/gate_sweep/train_chain.log         # trainer progress
tail -5 logs/gate_sweep/screen_chain.log        # screener progress
column -t logs/gate_sweep/sweep_summary.txt     # per-epoch screen table (grows)
cat logs/gate_sweep/canonical_summary.tsv       # final gate verdicts
cat logs/gate_sweep/BEST_GATE_PASS              # the winner (if any)
ls SWEEPTRAIN_DONE SWEEP_DONE SWEEP_FAILED SWEEPTRAIN_FAILED 2>/dev/null
```
GPU rule: max 1 training + 1 eval GPU (2 total). GPUs 0/2/4/6 belong to another user. If a process dies (the box has a Claude-client-teardown SIGTERM gotcha), just relaunch the same script — both are idempotent/resumable (`setsid nohup bash run_gate_sweep_train.sh &`, same for screener).

## 4. Decision tree on completion

1. **`BEST_GATE_PASS` exists** → that line is the new PESQ-boosted operating point. Actions:
   - Re-verify: re-run its canonical eval once more (same command family as §1) to confirm stability.
   - Compute −6..9 Δwb + ILD from its screen CSV (`results/pesq_eval/<stag>/vctk_paper_style_results.csv`) for the comparison table.
   - Update `docs/RESULTS_CHIP_COMPARISON_FINAL_2026-07-09.md` (append §2.10) + `STATUS_..._2026-07-10.md` §6. Deploy ep3 stays the gate/spatial headline; this becomes the "PESQ-boosted, gate-holding" variant.
2. **`ft_canon_w6ep4_anchor` passes both gates** (check even before SWEEP_DONE) → tier-1 is already done with an EXISTING ckpt; the sweep then only tries to beat its wb 1.718.
3. **`NO_GATE_PASS`** → the honest conclusion is a measured Pareto: gate ⇒ PESQ ~<1.7 on this architecture. Next options, in order: (a) pick best FAIL-by-<0.001 point and continue it 2-3 epochs with STOI_W=12; (b) MetricGAN-on-deploy standby track (below); (c) accept deploy + rebalance two-point story.
4. **`SWEEP_FAILED` / chains dead** → inspect the two chain logs; each failed unit names itself (`failcount_*`, `!! canonical ... failed`). Relaunch scripts as-is (idempotent).

## 5. Standby track: MetricGAN-on-deploy v2 (do NOT start while sweep runs — GPU budget)

All Codex-review fixes are implemented & compiled, NOT GPU-smoked since the v2 rewrite:
- `code/DCNN/metricgan.py` — D warm-up (`+model.d_warmup_epochs`), PESQ-failure label exclusion, (noisy,clean) D calibration pairs, D frozen in G-step, NaN-sanitize+clip(1.0), D-vs-truth monitor (dmon_mae/pearson/spearman/val_true_pesq), snr_loss_mode passthrough.
- `code/train_metricgan_deploy.py` — generator built through DCNNLightningModule (full W4A8 surgery) + warm-start; `BCCTN_D_INIT_CKPT` env loads disc.* (FP32 critic: `fp32_512_pesq_ft/outputs/metricgan_w12/.../last.ckpt`, 20 tensors).
- `code/run_metricgan_deploy.sh` — SMOKE=1 mini-run; batch 32.
- `run_metricgan_deploy_chain.sh` — resume arithmetic fixed (counts ckpts across version dirs), batch 32.
Before any long run: SMOKE=1 first; then D-warm-up-only sanity (D_WARMUP=2, N_EPOCHS=2) watching dmon_pearson (stop if Pearson/Spearman <= 0 or prediction std collapses); then short GAN_W sweep {2,4,8}, screen per epoch, canonical only for near-gate points.

## 6. Backlog (unordered, not blocking)

- ITD/IC backfill for KD+spatial / FP32-512 lineage / 09_tied / low-SNR / VB+DEMAND rows (metrics already in all three eval.py; ~1.5 hr chain).
- SI-SDR not written into `_averages.txt` (computed per-file only).
- Silicon efficiency table (pJ/inf, mm², TOPS/W vs P1/P4) — biggest paper gap, hardware-side.
- Doc language rules: docs append-only; new results doc named `RESULTS_<NAME>_<YYYY-MM-DD>.md`; numbers 4-decimal; gates always named.

## 7. Key numbers cheat-sheet

| point | VCTK MBSTOI / wb (scaled screen) | TIMIT MBSTOI / wb | note |
|---|---|---|---|
| deploy ep3 | 0.8341 / 1.5895 (canonical 0.8357/1.5932) | 0.7879 / 1.6415 (canon 0.7898/1.6558) | gate model |
| w6 ep4 anchor | 0.8337 / 1.7176 | 0.7888 / 1.7469 | canonical running now |
| w24 ep16 | 0.8188 / 1.9392 | 0.7748 / 1.9110 | beats SNU PESQ, fails gate |
| reproduced SNU | −6..9 Δwb +0.562; ILD 5.71 (destroys) | — | comparison target |
| noisy VCTK | 0.7861 / 1.2404, ILD 3.714 | 0.7311 / 1.2667, ILD 3.290 | |

## 8. Night status (appended 2026-07-10 ~19:30) — hostile-box shutdown, morning restart plan

From ~16:59 an unidentified mechanism killed our compute jobs within 1-5 min of start:
first only eval.py (any lineage incl. user-tmux, any GPU, 32 or 8 metric workers — always
killed during the metric fan-out phase), and from ~18:20 train.py attempts too (R2/R3/R4
burned all 4 supervisor retries before banking a single epoch). The SAME eval config ran
clean for hours in the morning. All chains were deliberately STOPPED at ~19:30.

**Banked (safe on disk):** `sw_w6cont_exact` 3 epochs; `sw_w6cont_s12` 4/6 epochs. R2–R4: none.
Anchor canonical (`ft_canon_w6ep4_anchor`) never completed — still the top-priority verdict.
Sentinels present: `SWEEPTRAIN_DONE` + `SWEEPTRAIN_FAILED` (trainer exhausted retries on R2–R4).

**Morning restart (run when the box behaves — ideally user-typed in tmux):**
```bash
cd /raid/R12K41024/20_pesq_finetune
rm -f SWEEPTRAIN_DONE SWEEPTRAIN_FAILED   # let the trainer finish R1 (+2 ep) and R2-R4
bash run_gate_sweep_train.sh &            # resumes; skips R0 (target met)
bash run_gate_sweep_screen.sh &           # anchor canonical first, then screens + verdict
```
Both idempotent. Screener currently at METRIC_WORKERS=8 (32 was fine in the morning; restore
for speed once the box is confirmed calm). Ask the admin what process guard was active on the
box after ~17:00 on 2026-07-10 — it kills this user's multi-process python jobs within minutes
(2 TB RAM box, no OOM records; kills also hit user-tmux-launched processes).

## 9. Session-handoff snapshot (2026-07-10 ~20:00) — READ THIS FIRST in a new session

### 9.1 Killer: current best theory = CONCURRENCY-TRIGGERED guard (Codex)
Every death tonight fits one pattern: **whenever a SECOND compute job of this user starts, one
of the two is SIGTERMed within seconds-to-minutes** (decisive case: user's foreground tmux
anchor eval ran clean for 5.5 min alone, user started the trainer at 18:58:38, anchor was
SIGTERMed 24 s later at 18:59:02 at VCTK −3 566/750). Host itself is clean (no cron/oomd/cgroup
limits/OOM/Xid; /dev/shm 1%). Harness child-reaping alone is refuted (user-typed foreground
victim). **Operating rule until falsified: run exactly ONE compute job at a time.** If a SOLO
job still gets killed → theory falsified; capture the killer with
`strace -p <python_pid> -e trace=signal -o /tmp/sigtrace.log` (si_pid names the sender) and take
the exact timestamp to the admin.

### 9.2 Calibration false alarm — RETRACTED (do not re-fix)
w6-ep4 under the deploy static calib table at SNR −6: MBSTOI 0.6573 vs deploy-canonical 0.6649
vs w6-scaled 0.6629 — differences ≤0.008; anchor even has the best PESQ (1.1954) and SegSNR-impr
(3.10). Static table transfers cleanly; NO per-candidate recalibration needed. Real signal to
track: w6 weights carry a small spatial drift (ILD +0.34 dB, IPD +0.8° vs deploy, present in
scaled mode too) — weigh vs PESQ gain at selection time.

### 9.3 What is running / queued (strict serial queue, all USER-launched in tmux)
1. RUNNING: trainer (`bash run_gate_sweep_train.sh`, user-launched 18:58, sole job). Fixed
   tonight: run_one success is judged by BANKED all-epoch ckpt count, never exit code
   (Lightning exits 0 on graceful SIGTERM). Look for `banked=N/target` lines and
   `TRAINER COMPLETE`; ETA ~21:30.
2. NEXT: Task 1 anchor canonical, alone (§8-style command, TAG=w6ep4_anchor,
   **METRIC_WORKERS=32**, GPU any free of 3/1/7/5). Verdict: VCTK ≥0.835 AND TIMIT ≥0.788.
3. THEN: `cp -r .../ft_w6ep4_anchor .../ft_canon_w6ep4_anchor` (lets the screener skip its own
   anchor pass).
4. THEN: screener overnight, alone (`bash run_gate_sweep_screen.sh`; internally serial; now
   restored to METRIC_WORKERS=32; flock + child `9>&-` + timeout wraps + sentinel self-clean).
   Do NOT start the nanny during clean tests (`run_gate_sweep_nanny.sh` stays off).
5. Morning: read `logs/gate_sweep/canonical_summary.tsv` + `BEST_GATE_PASS` (or `NO_GATE_PASS`)
   → §4 decision tree; then final comparison tables + RESULTS doc update.

### 9.4 Roles
- The USER launches all compute jobs (tmux foreground). Claude: read-only monitoring (`tail`,
  `grep`, `ps`), analysis, doc/code edits. Claude must NOT spawn compute jobs or kill processes
  while the serialization experiment is running.
- External reviewer (Codex) is active; its findings tonight were high-quality (concurrency
  theory, exit-code bug, sentinel semantics) — keep routing review checklists to it
  (see REVIEW_CHECKLIST_GATE_SWEEP_2026-07-10.md).
