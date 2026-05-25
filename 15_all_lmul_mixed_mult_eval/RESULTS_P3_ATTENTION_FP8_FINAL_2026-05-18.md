# RESULTS: P3 Attention Datapath — Final Sign-off

Date: 2026-05-18 (final, supersedes scaffold doc [RESULTS_P3_ATTENTION_FP8_2026-05-18.md](RESULTS_P3_ATTENTION_FP8_2026-05-18.md))
Base candidate: P2 `g42 + s050_topk`
Spec: [FUTURE_WORK_P3_ATTENTION_FP8_2026-05-17.md](FUTURE_WORK_P3_ATTENTION_FP8_2026-05-17.md)
Handoff: [HANDOFF_CLAUDE_P3_P4_STATUS_2026-05-18.md](HANDOFF_CLAUDE_P3_P4_STATUS_2026-05-18.md) (Codex reviewed)

---

## 0. TL;DR

| Variant | RTL effect | VCTK Δ vs baseline | TIMIT Δ vs baseline | Verdict (isolated) |
|---|---|---|---|---|
| **a1_fp16_accum** | FP16 in QK/softmax/Score×V | -0.0008 | -0.0013 | ✅ isolated PASS |
| **b_prob_q08** | 8-bit probability storage | -0.0014 | -0.0008 | ✅ isolated PASS |
| **c2b_scale_fp8** | W4 per-group scale stored as FP8 | -0.0011 | -0.0021 | ✅ isolated PASS |
| b_score_fp8 | FP8 attention score | – | – | ❌ B0 gate fail (Jaccard 0.65) |
| b_score_fixed (10,17) | Q10.17 fixed-point score | – | – | ❌ sanity kill (0.7966) |
| c2c_scale_pow2 | W4 scale as pow2 | – | – | ❌ skipped (TIMIT margin 0.0017 < 0.002) |

**P3 sign-off**:
- **Isolated**: three variants (a1, b_prob_q08, c2b) all PASS full VCTK +
  TIMIT gates individually on top of P2 g42+s050_topk.
- **Cumulative endpoint candidate**: the three variants are structurally
  orthogonal (compute dtype / probability register / W4 scale SRAM), so
  stacking them is the recommended RTL endpoint — **but the combined stack
  has not been measured yet**. Final ship decision needs combined-stack
  verification (see §6 verification rule).
- None reduce MAC count; all reduce datapath/storage cost inside the
  attention block and W4 weight scale SRAM.

**Score-domain quantization is rejected** (both FP8 and fixed-Q). Keep
attention scores in FP16 register. The shift_topk_pow2_clamp8_nomantissa
softmax already discretizes scores in exponent-domain; double-discretization
breaks training.

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
| a1_fp16_accum | 0.8352 | -0.0008 | 0 | 0.7914 | -0.0013 | -0.0005 | 3.280 | 15.95 | 3.052 | 20.32 |
| b_prob_q08 | 0.8346 | -0.0014 | -0.0006 | 0.7919 | -0.0008 | 0 | 3.324 | 15.98 | 3.071 | 20.26 |
| c2b_scale_fp8 | 0.8349 | -0.0011 | -0.0003 | 0.7906 | -0.0021 | -0.0013 | 3.316 | 15.91 | 3.085 | 20.20 |

All three surviving variants pass all gates as **isolated** ablations on top of
P2. **Combined stack (a1 + b_prob_q08 + c2b together) was not measured**;
treat it as a candidate endpoint, not a measured result. Structural
orthogonality argues for low interaction risk:
- a1 changes intermediate compute dtype, doesn't change weights or post-softmax storage
- b_prob_q08 changes post-softmax storage, orthogonal to a1
- c2b changes W4 scale storage, orthogonal to attention path

But structural argument is not measurement. See §6 verification rule before
treating the combined stack as a shipped result.

---

## 3. 20-pair sanity (screening, all already promoted to full)

| Variant | MBSTOI 20p | Δ vs P2 (0.8081) | ILD 20p | IPD 20p | best val_loss |
|---|---|---|---|---|---|
| P2 reference | 0.8081 | 0 | 3.340 | 17.02 | -13.760 |
| a1_fp16_accum | 0.8087 | +0.0006 | 3.334 | 17.10 | -13.717 |
| b_prob_q08 | 0.8092 | +0.0011 | 3.394 | 17.05 | -13.602 |
| c2b_scale_fp8 | 0.8076 | -0.0005 | 3.397 | 17.06 | -13.679 |

Sanity ordering matches full-data ordering for direction-of-effect (b_prob_q08
best at sanity, ties top of full). val_loss range -13.6 to -13.7 stays close
to P2 reference -13.76 → no capacity issues across the surviving variants.

---

## 4. B0 attention score dump + simulation (gate for score quantization)

Dump scope:
- 8 utterances from VCTK validation set
- ≥1200 rows/head per MHA layer (target ≥1000)
- 1 MHA layer (`mattn.mattn`), 16 heads
- Output: [results/p3_attention_fp8/b0_attention_scores.npz](results/p3_attention_fp8/b0_attention_scores.npz)

### 4.1 FP8 score quantization (REJECTED)

Per-head worst case:

| metric | head 10 worst | gate | pass? |
|---|---|---|---|
| top-4 Jaccard | 0.646 | ≥ 0.98 | ❌ |
| top-1 flip rate | 35.4% | ≤ 1% | ❌ |

FP8 E4M3 cannot preserve attention top-k ordering with required stability.
Per spec §4.3: skip `b_score_fp8`.

### 4.2 Fixed-Q score (REJECTED at training stage, despite B0 PASS)

Per-head derived bits range: int_bits ∈ [7,10], frac_bits ∈ [8,17].
**Worst-case union for single global format: int=10, frac=17** (27 bits total).

B0 simulation gate: **all heads PASS** (worst head 13: Jaccard 0.9997, flip 0.17%).

**But training fails** — see §4.3.

### 4.3 Why fixed-Q (10,17) failed at QAT despite passing B0

20-pair sanity result: MBSTOI 0.7966 (Δ -0.0115 vs P2, below floor 0.8031).
val_loss at epoch 0 = -11.892 (vs P2 -13.76, gap 1.87). Improves to -12.224
by epoch 2 then stalls.

Three root causes (Codex review confirmed):

1. **Frame mismatch**: B0 dump was on the *frozen* P2 model in fp32 attention
   path (default `fp16_accumulator=false`). Training runs with
   `fp16_accumulator=true` → scores route through fp16 cast → fixed-Q →
   shift-topk softmax. fp16 cast shifts the score distribution that B0
   measured.

2. **STE absence in `_quantize_attention_scores` fixed mode**
   ([complexLayers.py:807](DCNN/utils/complexPyTorch/complexLayers.py#L807)):
   plain `torch.round` blocks gradient. Model can't learn to compensate for
   the introduced quantization noise. Same gap exists in `q0.8` prob quant
   at [:830](DCNN/utils/complexPyTorch/complexLayers.py#L830) but doesn't
   manifest because shift-topk softmax already discretizes prob heavily.

3. **Double discretization**: shift_topk_pow2_clamp8_nomantissa softmax is
   already aggressive exponent-domain discretization (per
   [softmax handoff §4.1](../11_weight_tied_mha_fast_mbstoi/RTL_HANDOFF_SOFTMAX_2026-05-14.md)).
   Adding fixed-Q on top of that creates redundant + destabilizing
   quantization in the same stage. Exactly the warning in
   [FUTURE_WORK §4.4](FUTURE_WORK_P3_ATTENTION_FP8_2026-05-17.md#L355).

**Decision**: kill `b_score_fixed`. No 5e-4 retry attempted; root cause is
STE absence, lr bump won't close the 1.5 val_loss gap.

**STE patch is NOT in scope for P3 sign-off** because the score-domain
quantization path is dominated by the b_prob_q08 + a1 stack. If a future
project decides to revisit score quantization, the patch is:
[complexLayers.py:807](DCNN/utils/complexPyTorch/complexLayers.py#L807) and
[:830](DCNN/utils/complexPyTorch/complexLayers.py#L830) need
`torch.autograd.Function`-based STE wrappers around `torch.round`.

### 4.4 b_score_fixed STE-patch retry (post-Codex patch verification)

Codex landed STE wrappers at the three referenced sites on 2026-05-18 after
the initial P3 sign-off. To validate the patch and re-test the score-fixed
path, b_score_fixed was retried with identical config (Q10.17 from B0,
lr=2e-4, 10 ep) but now with `_ste_round` providing backward passthrough.

| Metric | Pre-patch (no STE) | **Post-patch (with STE)** | P2 ref (20p) |
|---|---|---|---|
| MBSTOI 20p | 0.7966 | **0.7969** | 0.8081 |
| ILD | 3.643 | 3.643 | 3.340 |
| IPD | 18.10° | 18.14° | 17.02 |
| best val_loss | -12.224 (ep 2) | **-12.344 (ep 3)** | -13.760 |

**Interpretation**:
- STE patch is verified working: val_loss converges 0.12 deeper and reaches
  best one epoch later (ep 3 vs ep 2). Gradient flow through `torch.round`
  is now restored.
- **But the fundamental quality gap is unchanged**: MBSTOI 0.7969 vs 0.7966
  is within run-to-run noise, still 0.0062 below the 0.8031 floor and 0.0112
  below P2 reference. STE is necessary but not sufficient.
- **Root cause upgrade**: the failure was NOT primarily an STE/gradient-flow
  issue. It is a **structural double-discretization conflict** between Q10.17
  fixed-point score storage and the shift_topk_pow2_clamp8_nomantissa softmax.
  The softmax already exponent-domain discretizes scores; adding Q10.17 on
  top creates two interacting quantization grids whose mismatch can't be
  learned around even with proper gradient flow. This exactly matches the
  warning in [FUTURE_WORK §4.4](FUTURE_WORK_P3_ATTENTION_FP8_2026-05-17.md#L355).

**Outcome**: b_score_fixed remains rejected, but the rejection now rests on
**structural** rather than **implementation** grounds. Score-domain
quantization should NOT be revisited under shift-topk softmax.

**STE patch value (looking forward)**:
- Useful for any score-quant work that uses a different softmax (e.g.,
  standard FP softmax + score FP8) where double-discretization is absent
- Required dependency if anyone wants to try score quant with the existing
  shift-topk path replaced by a softer normalization variant
- The mantissa-floor STE applied at [:927](DCNN/utils/complexPyTorch/complexLayers.py#L927)
  is relevant to L-Mul QAT — see [RESULTS_P4 §7 follow-ups](RESULTS_P4_LMUL_FEASIBILITY_FINAL_2026-05-18.md)

---

## 5. c2c_pow2 fallback decision (spec §4.5)

Rule:
> Run c2c only if c2b has at least about 0.002 MBSTOI margin over both
> full-data floors: VCTK ≥ 0.8322, TIMIT ≥ 0.7889.

c2b margins:
- VCTK: 0.8349 - 0.8322 = **0.0027 ≥ 0.002 ✓**
- TIMIT: 0.7906 - 0.7889 = **0.0017 < 0.002 ✗**

**Verdict**: TIMIT margin fails → **stop at c2b, do not run c2c_pow2**. c2b is
the P3 scale endpoint.

If a future spec wants pow2 weight scales, recommended path: do not stack on
top of full P2+P3 stack. Instead retest pow2 directly on P1 g42 (without
s050_topk skip cut) where there's more quality headroom.

---

## 6. Decision: endpoint recommendation

**P3 RTL endpoint candidate** (requires combined-stack verification before
final ship): stack of three variants on top of P2 `g42 + s050_topk`:

1. `training.fp16_accumulator = true` (a1)
2. `model.attention_datapath.prob_quant = q0.8` (b_prob_q08)
3. `training.int4_scale_format = fp8_e4m3` (c2b)

**Verification rule (unified)**:
- **Paper / formal sign-off**: combined stack must pass **full 750-pair
  VCTK + TIMIT** at gates listed in §1.
- **RTL bring-up / tape-out precheck**: combined stack must pass **20-pair
  sanity** (MBSTOI ≥ 0.8031) as a cheap go/no-go before committing area.

Each was tested isolated; combined hasn't been verified but the three are
structurally orthogonal (compute dtype / probability storage / W4 scale
storage do not share state). Recommended verification sequence: start with a
combined-stack 20-pair sanity; if it passes, run full VCTK + TIMIT. Use a
longer 30-epoch finetune only if the 10-epoch combined sanity shows recovery
but lacks enough margin.

**P3 hardware story** (candidate handoff to RTL team after combined sanity):

| Block | Pre-P3 | Post-P3 |
|---|---|---|
| Attention scratch dtype (Q/K/V, QK^T, softmax intermediates, Score×V) | fp32 / fp16 mixed | **FP16 throughout** (a1) |
| Attention score storage | FP16 register (32 entries from sliding window) | unchanged (b_score_fp8/fixed both rejected) |
| Attention probability storage | FP32 (post-softmax) | **q0.8 (8-bit fixed-point register)** (b_prob_q08) |
| W4 weight per-group scale | FP32 in SRAM | **FP8 E4M3 in SRAM** (c2b) |

MAC count is unchanged by P3. The savings are in datapath/register width and
W4 scale SRAM footprint.

---

## 7. Known limitations / future work

1. **STE absent in score/prob quantization paths**. P3 b_prob_q08 still PASSes
   gate because shift-topk softmax already discretizes prob output; the lack
   of STE doesn't manifest. But STE should be added before any future score-
   quant experiment.

2. **Combined-stack untested** — the critical gap before any "ship" claim.
   Only the three individual variants on top of P2 are measured. Per §6
   verification rule, paper sign-off needs combined 30-ep finetune + full
   VCTK + TIMIT; tape-out precheck needs at least combined 20-pair sanity.

3. **B0 only covers `mattn.mattn`** (the single MHA layer in this model).
   If model grows to multi-MHA-layer in the future, B0 scope must extend.

4. **Mask audit not fully reported**. B0 dump captured mask values per-layer
   but the JSON summary doesn't break out unique mask values. Worth a follow-
   up dump if anyone wants to re-evaluate score-domain quantization.

---

## 8. Reproducibility

```bash
cd /raid/R12K41024/13_aggressive_algo_qat

# A0 reference (sanity; reproduces P2 0.8081 bit-identical at 20p)
LIMIT_PAIRS=20 bash scripts/run_phase3_attention_full_eval.sh a0_p2

# B0 dump + simulation
conda run -n bcctn python scripts/dump_p3_attention_scores.py \
  --limit_utterances 8 --max_rows_per_head 1200 \
  --output_npz results/p3_attention_fp8/b0_attention_scores.npz
conda run -n bcctn python scripts/simulate_p3_attention_quant.py \
  --dump_npz results/p3_attention_fp8/b0_attention_scores.npz

# Surviving variants: 10-ep screening
GPU_ID=1 EPOCHS=10 LR=2e-4 bash scripts/run_phase3_attention_fp8_sweep.sh a1_fp16_accum
GPU_ID=1 EPOCHS=10 LR=2e-4 bash scripts/run_phase3_attention_fp8_sweep.sh b_prob_q08
GPU_ID=1 EPOCHS=10 LR=2e-4 bash scripts/run_phase3_attention_fp8_sweep.sh c2b_scale_fp8

# Full eval
GPU_ID=1 bash scripts/run_phase3_attention_full_eval.sh a1_fp16_accum b_prob_q08 c2b_scale_fp8

# TIMIT cross-domain
GPU_ID=1 bash scripts/run_phase3_attention_timit.sh a1_fp16_accum b_prob_q08 c2b_scale_fp8
```

### 8.1 Result file locations

```
results/p3_attention_fp8/
  b0_attention_scores.npz           # B0 dump
  b0_quant_sim.json                 # B0 gate decision
results/p3_attention_fp8_sanity/
  a1_fp16_accum_w20/  b_prob_q08_w20/  c2b_scale_fp8_w20/  b_score_fixed_w20/
results/p3_attention_fp8_full/
  a1_fp16_accum_w750/  b_prob_q08_w750/  c2b_scale_fp8_w750/
results/p3_attention_fp8_timit/
  a1_fp16_accum_vctk_timit_w750/  b_prob_q08_vctk_timit_w750/  c2b_scale_fp8_vctk_timit_w750/
```

Training outputs under `outputs/p3_attention_fp8/<variant>_<timestamp>/`.

---

## 9. Decision log

1. **Why recommend all three of a1 + b_prob_q08 + c2b as the candidate
   endpoint instead of picking one?** They are structurally orthogonal
   (compute dtype / prob register width / W4 scale SRAM). Each is essentially
   free in isolation. Pick-one would leave value on the table for no quality
   benefit. **Note**: this is a *candidate* endpoint — combined-stack
   verification (one 30-ep finetune + full eval per §6 rule) is required
   before any actual ship claim.

2. **Why reject b_score_fp8 and b_score_fixed?** B_score_fp8: B0 gate failed
   (Jaccard 0.65 << 0.98). B_score_fixed: passed B0 but failed training due to
   missing STE + double-discretize interaction with shift-topk softmax. Both
   indicate the same finding: **attention scores should stay in FP16
   register; softmax discretization is the right place to spend the precision
   budget, not the score storage**.

3. **Why skip c2c_pow2?** Spec §4.5 explicit rule: TIMIT margin failed (0.0017
   < 0.002). Stop at c2b.

4. **Why was B0 dumped without fp16_accumulator=true?** Codex's dump script
   defaults to the P2 configuration which has `fp16_accumulator=false`. This
   is a known measurement frame mismatch that contributed to b_score_fixed
   failure. Not worth redumping since score-domain quant is rejected anyway.
   If revisited, redump with `fp16_accumulator=true` would give more
   representative score distributions.

5. **Why not retry b_score_fixed at lr=5e-4 per spec §4.2 fallback?** val_loss
   gap is 1.5 below P2 reference, which is far beyond what a lr bump can
   close. Root cause analysis pointed to STE absence, not optimization
   difficulty. Kill cleanly per kill rule (sanity 0.7966 < floor - 0.003 =
   0.8001).
