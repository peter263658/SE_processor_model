# HANDOFF: P3/P4 Status (Claude → Codex review)

Date: 2026-05-18 (mid-execution, jobs still running)
Workspace: `/raid/R12K41024/13_aggressive_algo_qat`
Base candidate: P2 `g42 + s050_topk`
Predecessor: [HANDOFF_CLAUDE_P3_P4_2026-05-18.md](HANDOFF_CLAUDE_P3_P4_2026-05-18.md) (Codex implementation handoff)

This is a mid-execution status doc for Codex to review the decisions Claude made
during the formal P3/P4 sweep. Two jobs still running (GPU 1 + GPU 4).

---

## 0. Environment

CUDA blocker (handoff §2) was Codex-session-specific. In tmux Claude env:

```text
$ conda run -n bcctn python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
True 8
$ nvidia-smi -L → 8 H100s all visible
```

Used GPU 1 for P3 chain, GPU 4 for P4 chain (both 0% idle when claimed).

---

## 1. Script bug fixed

[run_phase3_attention_fp8_sweep.sh:26](scripts/run_phase3_attention_fp8_sweep.sh#L26) (and 3 sibling P3/P4 scripts) hardcoded `P2_BEST_CKPT` to:

```text
outputs/p2_skip/.../checkpoints/epoch=27-step=27104.ckpt
```

Hydra override parser chokes on `=` inside the value. A1 retry failed with
`mismatched input '='` on first launch.

**Fix applied**: created symlink `Checkpoints/p2_best_s050_topk.ckpt` → real
ckpt, patched 4 scripts (3 P3 + 1 P4) to use the symlink as default.
`.bak` files retained. Affected scripts:

```text
scripts/run_phase3_attention_fp8_sweep.sh
scripts/run_phase3_attention_full_eval.sh
scripts/run_phase3_attention_timit.sh
scripts/run_p4_lmul_feasibility.sh
```

A1 retry on the patched script succeeded.

---

## 2. P3 results so far

### 2.1 A0 reference (no retrain, 20-pair)

```text
LIMIT_PAIRS=20 bash scripts/run_phase3_attention_full_eval.sh a0_p2
```

| Metric | Result |
|---|---|
| MBSTOI Enh | 0.8081 (bit-matches P2 RESULTS §2 s050_topk 20-pair = 0.8081) |
| Verdict | Pipeline + P2 ckpt + grouped + skip subset all correctly routed ✓ |

### 2.2 B0 official dump + simulation

```text
8 utterances, 1200 rows/head, single MHA layer (mattn.mattn), 16 heads
```

Gate (worst case across heads):

| Variant | worst head | top-4 Jaccard | top-1 flip | pass |
|---|---|---|---|---|
| fp8_e4m3 score | head 10 | 0.646 | 35.4% | ❌ FAIL (need ≥0.98 / ≤1%) |
| fixed-Q score | head 13 | 0.9997 | 0.17% | ✅ PASS |

Per-head derived fixed-Q bits range: int_bits ∈ [7,10], frac_bits ∈ [8,17].
**Worst-case union for single global format: int=10, frac=17 (27 bits)**.

Artifacts: `results/p3_attention_fp8/b0_attention_scores.npz`,
`results/p3_attention_fp8/b0_quant_sim.json`.

### 2.3 P3-A1 fp16_accumulator screening (10 ep, lr=2e-4)

| Metric | A1 result | P2 reference (20p) | Δ |
|---|---|---|---|
| MBSTOI Enh | 0.8087 | 0.8081 | **+0.0006** |
| ILD (dB) | 3.334 | 3.340 | -0.006 |
| IPD (°) | 17.10 | 17.02 | +0.08 |
| best val_loss | -13.717 (ep 8) | -13.760 | -0.043 |

✅ PASS gate. fp16_accumulator path safe to use as B-stage base. Per-epoch
~2:59 (no slowdown vs P2 baseline).

### 2.4 P3-B isolated ablations (10 ep, lr=2e-4)

| Variant | MBSTOI Enh | Δ vs P2 | ILD | IPD | best val_loss | Verdict |
|---|---|---|---|---|---|---|
| **b_prob_q08** | 0.8092 | +0.0011 | 3.394 | 17.05 | -13.602 | ✅ PASS |
| **b_score_fixed (10,17)** | 0.7966 | -0.0115 | 3.649 | 18.10 | -12.224 | ❌ KILL |

`b_score_fp8` skipped per B0 gate failure.

**`b_score_fixed` failure root cause analysis** (this is the part I want Codex
to specifically validate):

1. B0 sim is a FROZEN-model forward measurement — measured Jaccard on P2 ckpt
   inference path with fp16_accumulator=false. During QAT training,
   fp16_accumulator=true is enabled, scores route through fp16 cast → fixed-Q →
   shift-topk softmax. Forward-pass quality immediately damages: ep 0 val_loss
   = -11.89 (vs P2 -13.76, gap 1.87).
2. [`_quantize_attention_scores`](DCNN/utils/complexPyTorch/complexLayers.py#L793-L812) fixed mode uses plain `torch.round` — **no STE**, gradient
   blocked through score quant. Model can't learn to compensate. Val_loss
   stalls at -12.22 by ep 2 (only 0.33 improvement total).
3. Double discretization: `shift_topk_pow2_clamp8_nomantissa` already
   exponent-domain discretizes; fixed-Q on top is redundant and destabilizes —
   exactly the warning in [FUTURE_WORK §4.4](FUTURE_WORK_P3_ATTENTION_FP8_2026-05-17.md#L355-L361).

Spec §4.6 kill rule: sanity 0.7966 < floor 0.8031 - 0.003 = 0.8001 → KILL,
**no lr=5e-4 retry** (gap is 1.5 in val_loss, lr bump won't close it; root
cause is missing STE in score quant, not lr).

**Codex review request**: please verify the STE absence is real, not just
something I missed. If you intended `torch.round` to be STE'd via a
`Function.apply` somewhere I didn't see, point me to it. If STE truly is
missing, this is a real implementation gap worth fixing for any future
quantization-during-training work (but **not** required for P3 sign-off, since
b_score_fixed is dominated by b_prob_q08 + RTL keeps scores in FP16).

### 2.5 P3-C c2b_scale_fp8 (running, GPU 1, started just now)

`b4lq812ia` — 10 ep at lr=2e-4. Epoch 0 best val_loss -13.592. Per-epoch
~3:03 (no L-Mul, normal speed).

c2c_scale_pow2 will only run if c2b has ≥0.002 MBSTOI margin over both VCTK
0.8322 and TIMIT 0.7889 floors (per spec §4.5 fallback).

---

## 3. P4 L-Mul results so far

### 3.1 No-train 20-pair feasibility

```text
TRAIN=0 LIMIT_PAIRS=20 bash scripts/run_p4_lmul_feasibility.sh \
  lmul_qk_m4 lmul_scorev_m4 lmul_attn_m4
```

| Variant | MBSTOI Enh | Δ vs P2 (0.8081) | ILD | IPD |
|---|---|---|---|---|
| lmul_qk_m4 | 0.8079 | -0.0002 | 3.351 | 17.03 |
| lmul_scorev_m4 | 0.8074 | -0.0007 | 3.341 | 17.02 |
| lmul_attn_m4 (both) | 0.8072 | -0.0009 | 3.345 | 17.03 |

**All three within noise band of P2 baseline, with zero retraining**. This
already answers the user's main question ("can L-Mul replace attention QK
and/or Score×V?") in the affirmative.

### 3.2 qk_m4 QAT recovery (10 ep, lr=2e-4)

| Metric | qk_m4 no-train | **qk_m4 10ep QAT** | P2 ref | Δ vs P2 |
|---|---|---|---|---|
| MBSTOI Enh | 0.8079 | **0.8108** | 0.8081 | **+0.0027** |
| ILD (dB) | 3.351 | 3.295 | 3.340 | **-0.045** |
| IPD (°) | 17.03 | 17.15 | 17.02 | +0.13 |
| best val_loss | – | -13.758 (ep 2) | -13.760 | ≈ same |

QAT pushes MBSTOI/ILD slightly *above* P2 baseline (within cross-run noise but
positive signal). Early convergence (best at epoch 2 of 10).

Per-epoch ~6:50 vs A1's ~2:59 → **~2.3× simulator slowdown** (expected; this
is software cost of the explicit-product-tensor `_lmul_matmul`, not RTL cost).

### 3.3 scorev_m4 QAT recovery (running, GPU 4)

`bn5kdkd3n` — 10 ep. Epoch 0 best val_loss -13.660. Per-epoch ~7:48 (slightly
slower than qk_m4 because Score×V product tensor is bigger).

### 3.4 attn_m4 combined QAT (deferred)

Per HANDOFF §5.2: "Only test combined L-Mul after isolated targets look
stable". qk_m4 stable. Waiting on scorev_m4 result. If scorev_m4 also PASS,
attn_m4 combined run is justified (~2-3h).

If user wants to ship faster, skipping attn_m4 combined is defensible — the
two isolated PASSes already prove the arithmetic backend can replace both
matmuls, and the additive interaction is low risk (no-train attn_m4 lost only
+0.0007 vs qk-only, suggesting no destructive interaction).

---

## 4. Currently running (as of doc write time)

| Job | bash ID | GPU | ETA from start |
|---|---|---|---|
| P3-C c2b_scale_fp8 (10ep) | `b4lq812ia` | 1 | ~30 min |
| P4 scorev_m4 QAT (10ep) | `bn5kdkd3n` | 4 | ~80 min |

---

## 5. P3/P4 sign-off candidates (current state)

### P3 attention datapath surviving variants

| Variant | Status | What it does | RTL implication |
|---|---|---|---|
| a1_fp16_accum | ✅ PASS | QK/softmax/SV in fp16 instead of fp32 | FP16 registers in attention block |
| b_prob_q08 | ✅ PASS | shift-topk output → q0.8 (256-step) | 8-bit probability register |
| c2b_scale_fp8 | 🟢 running | INT4 weight scales stored as FP8 E4M3 | Smaller W4 scale SRAM |

### P3 attention datapath rejected

| Variant | Status | Why |
|---|---|---|
| b_score_fp8 | ❌ B0 gate fail | FP8 cannot preserve top-k ordering (jaccard 0.65) |
| b_score_fixed (10,17) | ❌ Sanity kill | Double-discretize + no STE + fp16 cast interaction |

**P3 recommendation**: stop pursuing score-domain quantization. Keep attention
scores in FP16 register; quantize only the probability storage (q0.8) and the
INT4 weight scales (c2b if PASS). Report this as "attention datapath
selective precision" rather than "attention FP8".

### P4 L-Mul surviving variants

| Variant | Status | What it does |
|---|---|---|
| lmul_qk_m4 | ✅ PASS (no-train + QAT) | M4 L-Mul on QK^T attention |
| lmul_scorev_m4 | 🟢 QAT running, no-train PASS | M4 L-Mul on Score×V |
| lmul_attn_m4 | 🟢 deferred (both above) | M4 on both |

**P4 recommendation**: L-Mul M4 on attention is **quality-feasible** with no
retraining and improves slightly with short QAT. Report against the user's
explicit question:

> Can L-Mul replace attention QK and/or Score×V in this BCCTN model without
> breaking MBSTOI/ILD/IPD after short QAT?

Answer: **Yes for QK (verified), Yes for Score×V (no-train verified, QAT
running). Combined is low risk based on no-train additivity. M3 not tested
yet — defer to follow-up if M4 area is insufficient.**

---

## 6. Pending after current jobs

1. Read scorev_m4 result, decide on attn_m4 combined run
2. Read c2b_scale_fp8 result, decide on c2c_pow2 (per spec margin rule)
3. Run full 750-pair VCTK + TIMIT on P3 winners (a1, b_prob_q08, c2b if PASS)
   and P4 winners (qk_m4, scorev_m4 if PASS)
4. Write RESULTS_P3 + RESULTS_P4 final sign-off docs
5. Update RESULTS_P1/P2 TL;DR with chained reference

---

## 7. Specific items for Codex review

1. **STE absence in `_quantize_attention_scores` fixed mode** (§2.4.2 above).
   Confirm or refute. If real, this is the explanation for b_score_fixed
   failure despite passing B0 sim. Not blocking P3 sign-off but worth noting.

2. **B0 measurement frame mismatch**: B0 was dumped with
   `training.fp16_accumulator=false` (Codex default), but b_score_fixed runs
   with `fp16_accumulator=true`. The fp16 cast may shift score distributions.
   Should B0 be redumped with fp16 active for a fairer Jaccard estimate?
   (Probably not worth it given the STE issue dominates.)

3. **Symlink-based ckpt fix**: do you prefer the symlink approach, or should
   the scripts learn to escape `=` in Hydra overrides? The symlink is simple
   and avoids touching every override site, but it's a workaround. If you
   prefer the proper fix, the change goes in
   `scripts/run_phase{1,2,3}_*.sh` and `scripts/run_p4_*.sh`.

4. **Combined attn_m4 QAT**: do you think running combined L-Mul QAT is worth
   the 2-3h GPU cost given the no-train additivity already suggests low risk?
   Or skip and report isolated-PASS as the L-Mul story?

5. **P3-C c2c_pow2**: if c2b_scale_fp8 passes, do you want pow2 tested even if
   margin is thin? The spec §4.5 fallback rule says skip if margin < 0.002.
   I'll follow the rule unless you have reason to push.

---

## 8. Reproducibility

```bash
cd /raid/R12K41024/13_aggressive_algo_qat

# A0 reference (verified bit-matches P2 RESULTS)
LIMIT_PAIRS=20 bash scripts/run_phase3_attention_full_eval.sh a0_p2

# B0 dump + sim (verified, gate decision recorded)
conda run -n bcctn python scripts/dump_p3_attention_scores.py \
  --limit_utterances 8 --max_rows_per_head 1200 \
  --output_npz results/p3_attention_fp8/b0_attention_scores.npz
conda run -n bcctn python scripts/simulate_p3_attention_quant.py \
  --dump_npz results/p3_attention_fp8/b0_attention_scores.npz

# P3-A1 + P3-B chain (use clean symlink path; defaults already patched)
GPU_ID=1 EPOCHS=10 LR=2e-4 bash scripts/run_phase3_attention_fp8_sweep.sh a1_fp16_accum
GPU_ID=1 EPOCHS=10 LR=2e-4 bash scripts/run_phase3_attention_fp8_sweep.sh b_prob_q08
# b_score_fixed: don't bother rerunning; killed by sanity floor

# P3-C
GPU_ID=1 EPOCHS=10 LR=2e-4 bash scripts/run_phase3_attention_fp8_sweep.sh c2b_scale_fp8

# P4 L-Mul
GPU_ID=4 TRAIN=0 LIMIT_PAIRS=20 bash scripts/run_p4_lmul_feasibility.sh \
  lmul_qk_m4 lmul_scorev_m4 lmul_attn_m4
GPU_ID=4 EPOCHS=10 LR=2e-4 bash scripts/run_p4_lmul_feasibility.sh lmul_qk_m4
GPU_ID=4 EPOCHS=10 LR=2e-4 bash scripts/run_p4_lmul_feasibility.sh lmul_scorev_m4
```
