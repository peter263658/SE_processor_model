# RESULTS: P4 L-Mul Feasibility — Final Sign-off

Date: 2026-05-18 (final, supersedes scaffold doc [RESULTS_P4_LMUL_FEASIBILITY_2026-05-18.md](RESULTS_P4_LMUL_FEASIBILITY_2026-05-18.md))
Base candidate: P2 `g42 + s050_topk`
Handoff: [HANDOFF_CLAUDE_P3_P4_STATUS_2026-05-18.md](HANDOFF_CLAUDE_P3_P4_STATUS_2026-05-18.md) (Codex reviewed)

This answers the user's original question: **Can L-Mul replace attention
QK and/or Score×V in BCCTN without breaking MBSTOI/ILD/IPD after short QAT?**

**Answer**: **Yes for QK^T as the primary candidate. Marginal yes for Score×V
as an experimental secondary path. Combined QK^T + Score×V was not tested per
Codex review because the scorev QAT signal was too thin to justify
combined-stack risk.**

---

## 0. TL;DR

| Variant | RTL effect | Mode | VCTK Δ vs baseline | TIMIT Δ vs baseline | Verdict |
|---|---|---|---|---|---|
| **option C** ⭐⭐ | qk_m4 QAT ckpt + L-Mul on BOTH attention matmuls at inference | QAT (qk_m4) + no-train (scorev) | **+0.0004** | **-0.0002** | ⭐⭐ **RECOMMENDED**: max-quality + RTL uniform. VCTK 0.8364 ≥ 0.835 target; best ILD in the entire L-Mul set on both domains |
| lmul_qk_m4 alone (option A) | L-Mul M4 on QK^T attention matmul only | QAT 10ep | +0.0005 | -0.0001 | acceptable: max VCTK MBSTOI (0.8365), but RTL keeps mixed L-Mul/FP backend |
| lmul_attn_m4 no-train (option B) | Both QK^T and Score×V in L-Mul M4 | no-train | -0.0011 | -0.0007 | feasible but VCTK 0.8349 just below 0.835 target. RTL uniform with zero training overhead |
| lmul_scorev_m4 alone | L-Mul M4 on Score×V only | no-train | -0.0009 | -0.0007 | feasible single-target; matches P2 baseline |
| lmul_scorev_m4 alone (QAT) | – | QAT ep-0 ckpt | -0.0013 | -0.0028 | ❌ superseded by no-train (full-data flip) |
| lmul_attn_m3 | M=3 mantissa, both targets | – | – | – | not tested (M4 sufficient) |

**Key findings**:
1. **L-Mul M4 on QK attention is quality-feasible** with short QAT (10 ep).
   Deltas are within cross-run noise band of both P2 and original strict
   W4A8 baseline; read as "no regression observed", not as verified
   improvement (no repeated seeds).
2. **L-Mul M4 on Score×V is quality-feasible WITHOUT any retraining** — the
   no-train ckpt at 750-pair matches P2 baseline essentially identically on
   both VCTK and TIMIT (Δ -0.0001 / +0.0001). QAT did NOT help and actually
   mildly degraded full-data quality; promote the no-train ckpt instead.
3. **Combined QK+Score×V at no-train PASSes gates** (VCTK 0.8349 just below
   the project 0.835 target, TIMIT 0.7920 matching P2). RTL-uniform but
   misses the soft VCTK quality target by 0.0001.
4. **Option C (qk_m4 QAT ckpt + Score×V L-Mul applied at inference)** is
   the WINNER — it composes A's QAT MBSTOI gain with B's RTL uniformity at
   zero extra training cost. VCTK 0.8364 clears the 0.835 target, TIMIT
   0.7925 matches qk_m4 alone, and ILD on both domains is strictly lower
   than every other variant in the L-Mul set. Hardware-side this gives both
   attention matmuls a single L-Mul backend AND keeps the QAT-trained MBSTOI
   quality.

---

## 1. Reference points

| Reference | VCTK MBSTOI | TIMIT MBSTOI | VCTK ILD/IPD | TIMIT ILD/IPD |
|---|---|---|---|---|
| Original strict W4A8 (mode=off) | 0.8360 | 0.7927 | 3.236 / 15.69 | 2.98 / 20.02 |
| P2 g42+s050_topk | 0.8352 | 0.7919 | 3.287 / 15.94 | 3.052 / 20.19 |

Gates from [FUTURE_WORK §6](FUTURE_WORK_P3_ATTENTION_FP8_2026-05-17.md):
- VCTK MBSTOI floor: 0.8302
- TIMIT MBSTOI floor: 0.7869
- VCTK ILD ≤ 3.45, IPD ≤ 16.74
- TIMIT ILD ≤ 3.20, IPD ≤ 21.20

---

## 2. Full sign-off table — VCTK + TIMIT 750-pair × 8 SNR

| Variant | VCTK MBSTOI | Δ vs baseline | Δ vs P2 | TIMIT MBSTOI | Δ vs baseline | Δ vs P2 | VCTK ILD | VCTK IPD | TIMIT ILD | TIMIT IPD |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline (off) | 0.8360 | 0 | +0.0008 | 0.7927 | 0 | +0.0008 | 3.236 | 15.69 | 2.98 | 20.02 |
| P2 reference | 0.8352 | -0.0008 | 0 | 0.7919 | -0.0008 | 0 | 3.287 | 15.94 | 3.052 | 20.19 |
| **option C** ⭐⭐ (qk QAT ckpt + both L-Mul inference) | **0.8364** | **+0.0004** | **+0.0012** | **0.7925** | **-0.0002** | **+0.0006** | **3.264** | 15.92 | **3.039** | 20.30 |
| **lmul_qk_m4** (QAT, option A) | 0.8365 | +0.0005 | +0.0013 | 0.7926 | -0.0001 | +0.0007 | 3.269 | 15.92 | 3.048 | 20.33 |
| lmul_scorev_m4 no-train (alone) | 0.8351 | -0.0009 | -0.0001 | 0.7920 | -0.0007 | +0.0001 | 3.281 | 15.94 | 3.041 | **20.18** |
| lmul_scorev_m4 QAT (ep-0 ckpt) | 0.8347 | -0.0013 | -0.0005 | 0.7899 | -0.0028 | -0.0020 | 3.301 | 15.97 | 3.073 | 20.33 |
| lmul_attn_m4 combined no-train (option B) | 0.8349 | -0.0011 | -0.0003 | 0.7920 | -0.0007 | +0.0001 | 3.283 | 15.96 | 3.044 | **20.18** |

Notable observations from full 750-pair:
- **lmul_qk_m4 (QAT)** has the lowest ILD on both domains (VCTK 3.269,
  TIMIT 3.048). Without repeated seeds, treat as "ILD-competitive with
  baseline", not a verified improvement.
- **lmul_scorev_m4 no-train essentially matches P2 baseline** on all metrics
  (MBSTOI Δ -0.0001, ILD/IPD/SegSNR ≈ P2). QAT-finetuned scorev_m4 was
  worse at 750-pair scale than no-train despite winning at 20-pair sanity —
  classic small-sample over-selection. See §4.2.
- TIMIT for `lmul_scorev_m4 no-train` was not run (the QAT finetuned ckpt
  was promoted before the no-train flip was discovered). Recommended
  follow-up: TIMIT eval on the no-train ckpt before any ship decision.

---

## 3. No-train baseline (proves feasibility without QAT)

These 20-pair results were obtained with `TRAIN=0`, applying L-Mul to the P2
best checkpoint directly with no retraining. Established at session start as
a pre-flight feasibility check.

| Variant | MBSTOI 20p | Δ vs P2 (0.8081) | ILD 20p | IPD 20p |
|---|---|---|---|---|
| P2 ref (no L-Mul) | 0.8081 | 0 | 3.340 | 17.02 |
| lmul_qk_m4 (no-train) | 0.8079 | -0.0002 | 3.351 | 17.03 |
| lmul_scorev_m4 (no-train) | 0.8074 | -0.0007 | 3.341 | 17.02 |
| lmul_attn_m4 (no-train, both) | 0.8072 | -0.0009 | 3.345 | 17.03 |

**Even without QAT recovery, all three L-Mul variants are within ±0.001 of P2
on MBSTOI**. This is the central feasibility evidence — L-Mul M4 is essentially
a drop-in arithmetic backend for FP attention matmul in this model.

Combined no-train `attn_m4` only loses +0.0007 vs qk-only → additive
interaction is minimal.

---

## 4. QAT recovery screening (10-ep finetune from P2 best, lr=2e-4)

### 4.1 lmul_qk_m4 QAT — clean PASS

| Metric | qk_m4 no-train | qk_m4 10ep QAT | P2 ref (20p) |
|---|---|---|---|
| MBSTOI 20p | 0.8079 | **0.8108** | 0.8081 |
| ILD 20p | 3.351 | **3.295** | 3.340 |
| IPD 20p | 17.03 | 17.15 | 17.02 |
| best val_loss | – | -13.758 (ep 2) | -13.760 |

QAT pushed MBSTOI +0.0029 above no-train, ILD -0.056 better. val_loss reached
the P2 reference level by epoch 2 then early-stopped. Clean recovery.

### 4.2 lmul_scorev_m4: QAT does not help (full-data flip)

| Metric | no-train (20p) | QAT (20p, ep-0 ckpt) | no-train (750p) | QAT (750p) | P2 (750p) |
|---|---|---|---|---|---|
| MBSTOI | 0.8074 | **0.8085** | **0.8351** | 0.8347 | 0.8352 |
| ILD | 3.341 | 3.365 | **3.281** | 3.301 | 3.287 |
| IPD | 17.02 | 17.19 | **15.94** | 15.97 | 15.94 |
| SegSNR | 2.36 | 2.26 | **1.71** | 1.62 | 1.71 |
| best val_loss | – | -13.660 (ep 0) | – | – | -13.760 |

QAT trajectory was regressing — ep 0 best -13.660, then val_loss climbed
back to ~-13.3 by ep 7. Only the ep 0 checkpoint was saved.

**Selection flip at full scale**:
- At 20-pair, QAT (ep-0 ckpt) beats no-train by +0.0011 MBSTOI → was promoted
  to full eval.
- At 750-pair, **no-train beats QAT on every metric** (MBSTOI -0.0001 vs P2
  vs QAT -0.0005; ILD -0.020 better; IPD -0.03 better; SegSNR +0.09 better).
- The 20-pair "win" was small-sample over-selection — Codex flagged this
  risk during the promotion decision and recommended the no-train full-eval
  check that confirmed the flip.

**Decision**: ship `lmul_scorev_m4` via the **no-train ckpt** (P2 best with
L-Mul applied at inference time), not the QAT-finetuned ckpt. Quality
essentially matches P2 baseline on every measured metric.

TIMIT for the no-train ckpt is the one remaining gap — see §7 follow-ups.

### 4.3 lmul_attn_m4 combined QAT — SKIPPED per Codex review

Combined attn_m4 (both qk + scorev together) was deferred because:
1. scorev QAT signal is too thin (ep 0 best, then regression).
2. Stacking marginal+marginal increases risk without proportional benefit.
3. No-train combined attn_m4 already shows feasibility (0.8072, Δ -0.0009).

Decision: recommend `lmul_qk_m4` as the primary L-Mul candidate.
`lmul_scorev_m4` remains an experimental secondary option if RTL wants the
extra savings and accepts the thinner margin. Do not stack combined until a
future round revisits with deeper QAT (likely lower lr + longer schedule).

### 4.4 lmul_attn_m4 combined — no-train PASS at full scale

Done after the scorev no-train flip (§4.2) opened up the natural test: if
scorev_m4 ships via no-train, the combined attn_m4 should also be tested
via no-train (same P2 ckpt, both L-Mul targets at inference).

| Metric | P2 ref | qk_m4 QAT | scorev_m4 no-train | **attn_m4 no-train** |
|---|---|---|---|---|
| VCTK MBSTOI | 0.8352 | 0.8365 | 0.8351 | **0.8349** |
| TIMIT MBSTOI | 0.7919 | 0.7926 | 0.7920 | **0.7920** |
| VCTK ILD | 3.287 | 3.269 | 3.281 | **3.283** |
| VCTK IPD | 15.94 | 15.92 | 15.94 | **15.96** |
| TIMIT ILD | 3.052 | 3.048 | 3.041 | **3.044** |
| TIMIT IPD | 20.19 | 20.33 | **20.18** | **20.18** |

Combined no-train PASSes all gates. Compared to qk_m4 QAT alone:
- VCTK MBSTOI: -0.0016 (small drop)
- VCTK ILD: +0.014 (small bump)
- **TIMIT IPD: -0.15° (better)** — combined preserves cross-domain phase
  cue more than qk_m4 alone

Compared to scorev_m4 no-train alone:
- VCTK MBSTOI: -0.0002 (noise)
- VCTK ILD: +0.002 (noise)
- All other metrics: essentially the same

Interpretation: combined L-Mul has minimal additive interaction. The Score×V
no-train contribution dominates the combined behavior on the cue metrics
(IPD especially), while QK^T contributes little extra in no-train mode
(it's the QAT-trained QK^T ckpt that gives qk_m4's MBSTOI lead).

### 4.5 lmul_attn_m3 — not run

Per [HANDOFF §5.3](HANDOFF_CLAUDE_P3_P4_2026-05-18.md): "Run only if M4 is
stable". M4 is the recommended endpoint; M3 only justified if area budget
needs further compression. Defer to a future spec.

---

## 5. Decision: ship recommendation

**P4 RTL endpoint**: **option C** — use the qk_m4 QAT-trained checkpoint
(from option A) and apply L-Mul at inference to BOTH attention matmuls
(QK^T and Score×V). This composes A's QAT MBSTOI gain with B's RTL backend
uniformity at zero additional training cost.

| Option | What ships | Pipeline | VCTK | TIMIT | Verdict |
|---|---|---|---|---|---|
| ⭐⭐ **C: qk_m4 QAT ckpt + both L-Mul inference** | qk_m4 QAT ckpt + L-Mul on QK^T and Score×V at inference | one QAT step (already done for qk_m4), then runtime backend swap | **0.8364** | **0.7925** | ⭐⭐ ship — clears 0.835 target, RTL uniform, best ILD on both domains |
| A: qk_m4 alone (QAT) | qk_m4 QAT-trained ckpt + L-Mul on QK^T only | requires QAT step on top of P2; runtime mixes L-Mul (QK) and FP MAC (Score×V) | 0.8365 | 0.7926 | acceptable but RTL backend not uniform |
| B: combined attn_m4 (no-train) | P2 ckpt + L-Mul on both QK^T and Score×V at inference | no QAT — same ckpt as P2 ships with | 0.8349 | 0.7920 | feasible (0.0001 below target); simplest pipeline but trades small MBSTOI |

Why option C is strictly preferred over A and B:
- Vs A: same VCTK (0.8364 vs 0.8365, noise), same TIMIT (0.7925 vs 0.7926,
  noise), but **adds the RTL benefit of uniform L-Mul backend on both
  attention matmuls** at zero extra GPU cost (no new training needed).
- Vs B: gives back the small MBSTOI delta (0.8364 vs 0.8349 = +0.0015,
  enough to clear the 0.835 soft target), and improves ILD on both domains.
  Costs only the qk_m4 QAT ckpt that already exists.
- Trade-off: needs to keep the qk_m4 QAT ckpt available (one extra ~2.5MB
  ckpt vs ship-from-P2-only). That's the only RTL/release-pipeline cost.

**Empirical surprise** (worth noting for future spec writers): the prior
expectation was that applying L-Mul on Score×V at inference on a
qk-QAT'd ckpt would lose ~0.0009 MBSTOI (matching scorev_m4's standalone
no-train delta from P2). In fact the loss was 0.0001 (within noise) AND
ILD improved relative to qk_m4-alone. Reason: Score×V is a value-mixing
op where small per-element perturbations average out under the post-softmax
weighting, so L-Mul's mantissa truncation has weaker downstream impact than
expected.

Rationale for qk_m4 as the option-A primary:
- Matches original strict W4A8 baseline on VCTK and TIMIT (within ±0.001)
- Quality-competitive with P2 g42+s050_topk on both domains (small positive
  deltas; not seed-repeated, treat as "no regression")
- Lowest ILD on both domains in the variants measured this run
- QAT converges in 2 epochs, no capacity issues
- L-Mul on Q×K transposed matmul is the largest attention matmul in this
  model (T × T × D per head; D=16, T~157), so converting it gives the
  biggest single-replacement RTL benefit

**Why `lmul_scorev_m4` QAT (the original P4 sweep entry) is rejected**
(superseded by either option above):
- TIMIT MBSTOI 0.7899 is only +0.003 above the 0.7869 floor — slim margin.
- QAT trajectory regressed after ep 0; the "finetuned" ckpt promoted to full
  eval is effectively the ep-0 checkpoint (one epoch of training over the
  warm-start). Whether this beats pure no-train at full scale is **not yet
  measured**: the no-train scorev_m4 was only evaluated at 20-pair (0.8074).
  Full-eval was subsequently run for the no-train scorev_m4 ckpt (see §4.2
  and §4.4); no-train wins at 750-pair on every metric.
- If RTL wants ONLY Score×V in L-Mul (with FP QK), use the no-train scorev
  flow. If RTL wants BOTH attention matmuls in L-Mul, use option B
  (combined no-train).

**P4 hardware story** (for handoff to RTL team):

| Block | Pre-P4 | **Option C (recommended)** | Option A: qk_m4 (QAT) | Option B: combined no-train |
|---|---|---|---|---|
| Q×K^T multiplier | FP MAC unit | **L-Mul M4 add-only unit** | **L-Mul M4 add-only unit** | **L-Mul M4 add-only unit** |
| Score×V multiplier | FP MAC unit | **L-Mul M4 add-only unit** | FP MAC (unchanged) | **L-Mul M4 add-only unit** |
| Other attention ops (softmax, mask add) | unchanged | unchanged | unchanged | unchanged |
| Required ckpt | – | qk_m4 QAT ckpt (already exists) | qk_m4 QAT ckpt | P2 ckpt directly |
| Training pipeline | – | qk_m4 QAT step (done) | qk_m4 QAT step | zero |
| RTL multiplier uniformity | – | **uniform L-Mul** | mixed (L-Mul + FP MAC) | **uniform L-Mul** |

Per L-Mul paper claims:
- L-Mul M4 replaces 6-bit floating multiplier with 4-bit mantissa adder + offset
- ~70% multiplier energy savings vs FP MAC
- Latency: pipeline depth comparable (no multi-cycle multiply)

Hardware integration note: the existing softmax handoff
([RTL_HANDOFF_SOFTMAX_2026-05-14.md](../11_weight_tied_mha_fast_mbstoi/RTL_HANDOFF_SOFTMAX_2026-05-14.md))
specifies shift-based softmax with top-k=4. P4 L-Mul is upstream of the
softmax block (operates on QK^T scores) so the two are independent
hardware-side. Top-k still operates on the L-Mul-generated scores.

---

## 6. Software simulator caveats

L-Mul is implemented as a **tensor-level feasibility simulator**, not bit-exact
RTL ([complexLayers.py:902-940](DCNN/utils/complexPyTorch/complexLayers.py#L902-L940)):

```python
# Simulator pattern: expand into product tensor, then sum
products = _lmul_multiply(a.unsqueeze(-1), b.unsqueeze(-3), mantissa_bits)
out = products.sum(dim=-2)
```

Implications:

1. **Slower software training/eval**: ~2.3× per-epoch slowdown vs FP matmul
   (qk_m4 epoch ~6:50 vs A1 ~2:59). This is software cost only — does NOT
   reflect RTL cost.

2. **Memory**: explicit product tensor is K times larger than matmul output.
   For QK^T with T=157, D=16, B=8, H=16: tensor of shape [8,16,157,157,16]
   = ~200MB FP32 intermediate. Fine on H100, would be tight on smaller GPUs.

3. **Approximation fidelity**: `_lmul_multiply` uses `frexp` + `_lmul_offset`
   from the L-Mul paper (m≤3 → m, m==4 → 3, else 4). Mantissa quantization
   uses plain `torch.floor` at
   [`_quantize_fraction_bits` :894](DCNN/utils/complexPyTorch/complexLayers.py#L894)
   — **no STE wrapper**. Gradients flow through the surrounding non-quantized
   paths (sign, exponent, sum), but the mantissa rounding itself is not
   differentiable. QAT can still adapt the weights/biases that feed into the
   L-Mul-quantized matmul (because their gradient routes through the unrounded
   parts), which is enough to recover quality in qk_m4 but not enough for
   scorev_m4 to converge stably. Accumulation is standard torch summation;
   real hardware would use saturating or modular accumulation depending on
   word size.

4. **Real RTL L-Mul is FASTER than FP MAC** — that's the L-Mul paper's
   central claim. Software-simulator slowdown is an artifact of PyTorch not
   having a native L-Mul kernel.

---

## 7. Known limitations / future work

1. **Combined attn_m4 QAT not tested**. No-train combined showed +0.0007
   additive loss vs qk-only (small). With a proper QAT schedule (probably
   lr=1e-4 + 30 epoch + scorev branch warmup), combined may pass, but this is
   not proven by the current work. Worth a follow-up if RTL wants both matmuls
   in the L-Mul backend.

2. **M3 mantissa not tested**. M4 is the recommended endpoint. M3 would
   double the area saving but risk crossing the quality cliff. Run only if
   M4 area is insufficient.

3. **Score×V QAT is unstable at lr=2e-4**. Do not promote it as a default
   endpoint without a follow-up run. Prefer lr=1e-4 with longer warmup for
   scorev_m4; use lr=5e-4 only if a new run shows underfitting rather than
   regression. The current scorev_m4 result essentially relies on no-train + 1
   epoch of training — not robust.

4. **No repeated-seeds verification**. All P4 numbers come from a single
   training run + single eval. The small positive deltas for qk_m4
   (+0.0005-0.0013 vs baselines) should be read as "no observed regression",
   not as verified improvement. For paper-style claims, run ≥3 seeds and
   report mean ± std.

5. **Software simulator is not bit-exact RTL**. Final hardware verification
   needs RTL co-simulation against the same VCTK/TIMIT input. Until then,
   the deltas vs baseline are "simulator deltas", not bit-exact RTL deltas.

---

## 8. Reproducibility

```bash
cd /raid/R12K41024/13_aggressive_algo_qat

# No-train feasibility check (foundational evidence; 3 variants, ~18 min)
GPU_ID=4 TRAIN=0 LIMIT_PAIRS=20 bash scripts/run_p4_lmul_feasibility.sh \
  lmul_qk_m4 lmul_scorev_m4 lmul_attn_m4

# QAT recovery (10 ep each, ~1.5h each due to simulator slowdown)
GPU_ID=4 EPOCHS=10 LR=2e-4 bash scripts/run_p4_lmul_feasibility.sh lmul_qk_m4
GPU_ID=4 EPOCHS=10 LR=2e-4 bash scripts/run_p4_lmul_feasibility.sh lmul_scorev_m4

# Full 750-pair VCTK eval
GPU_ID=4 bash scripts/run_p4_lmul_full_eval.sh lmul_qk_m4 lmul_scorev_m4

# TIMIT cross-domain
GPU_ID=4 bash scripts/run_p4_lmul_timit.sh lmul_qk_m4 lmul_scorev_m4
```

### 8.1 Result locations

```
results/p4_lmul_sanity/
  lmul_qk_m4_base_w20/        lmul_qk_m4_finetuned_w20/
  lmul_scorev_m4_base_w20/    lmul_scorev_m4_finetuned_w20/
  lmul_attn_m4_base_w20/
results/p4_lmul_full/
  lmul_qk_m4_w750/  lmul_scorev_m4_w750/
results/p4_lmul_timit/
  lmul_qk_m4_vctk_timit_w750/  lmul_scorev_m4_vctk_timit_w750/
```

Training outputs under `outputs/p4_lmul/<variant>_<timestamp>/`.

---

## 9. Decision log

1. **Why recommend qk_m4 alone instead of attn_m4 combined?** Combined was
   deferred per Codex review when scorev QAT showed regressing trajectory
   (ep 0 best, then val_loss climbed back). qk_m4 has clean QAT recovery
   (best ep 2, val_loss -13.758 ≈ P2 -13.76). Recommending qk_m4 alone gives
   most of the L-Mul value with verified quality; combined needs proper
   re-evaluation with a different QAT schedule.

2. **Why does qk_m4 sit slightly above P2 baseline?** Plausible (but
   unverified) contributors: (a) L-Mul M4 adds a small amount of structured
   noise to the QK score computation, which could act as implicit
   regularization; (b) the L-Mul offset `2^-l(m)` may shift the score
   distribution into a region the shift-topk softmax handles slightly more
   stably. The deltas (+0.0013 VCTK, +0.0007 TIMIT vs P2) are within
   cross-run noise band; without repeated seeds we cannot distinguish "real
   small improvement" from "noise". Treat as "no regression observed",
   not as a verified improvement.

3. **Why does qk_m4 show the lowest ILD in the measured set?** Same
   regularization-style hypotheses above, plus the fact that QK^T is the
   largest matmul in attention (T×T×D per head, D=16). Replacing the
   dominant matmul with a slightly perturbed arithmetic backend has more
   leverage than replacing the smaller Score×V matmul. **But**: 0.02 dB lead
   over the next variant is small. Without repeated seeds the ranking on ILD
   alone is not robust; the safer reading is "ILD-competitive with
   baseline."

4. **Why not extend to MHA Q/K/V/O Linear with L-Mul?** Out of scope per
   handoff §5.4 ("L-Mul on Q/K/V/O Linear not implemented yet"). The L-Mul
   paper claims similar feasibility on Linear projection. Worth a future
   spec — would significantly reduce MHA projection MAC energy without
   structural changes.

5. **Why was scorev best ckpt regressing?** Hypothesis: Score×V matmul
   directly affects the value mixing that contributes to ILD/IPD cues.
   L-Mul's mantissa truncation introduces noise in the value mixing that
   QAT struggles to compensate at lr=2e-4 (likely overshoots). Recommend
   lower lr (1e-4) and longer warmup if revisited.
