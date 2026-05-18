# RTL Handoff: BCCTN Optimization Stack (P1 → P4)

Date: 2026-05-19
Status: **ship recommendation = Option C** (verified at full 750-pair VCTK +
TIMIT). Full candidate stack with all P3 storage flags was measured and
**fails the 0.835 VCTK soft target** (clears hard floors with thin margin);
P3 q08+c2b are therefore deferred from the ship endpoint. See §0.1.
Workspace: `/raid/R12K41024/13_aggressive_algo_qat`
Predecessor handoffs / RESULTS docs:
- Softmax: [RTL_HANDOFF_SOFTMAX_2026-05-14.md](../11_weight_tied_mha_fast_mbstoi/RTL_HANDOFF_SOFTMAX_2026-05-14.md)
- P1: [RESULTS_P1_GROUPED_2026-05-17.md](../12_decoder_grouped_pointwise/RESULTS_P1_GROUPED_2026-05-17.md)
- P2: [RESULTS_P2_SKIP_2026-05-18.md](../12_decoder_grouped_pointwise/RESULTS_P2_SKIP_2026-05-18.md)
- P3: [RESULTS_P3_ATTENTION_FP8_FINAL_2026-05-18.md](RESULTS_P3_ATTENTION_FP8_FINAL_2026-05-18.md)
- P4: [RESULTS_P4_LMUL_FEASIBILITY_FINAL_2026-05-18.md](RESULTS_P4_LMUL_FEASIBILITY_FINAL_2026-05-18.md)

This is the integration handoff for the RTL team. It consolidates the
verified ship endpoint across the optimization phases, explicitly separates
deferred P3 storage options, and lists the hardware-side changes by block.

---

## 0. TL;DR

**Ship recommendation**: **Option C** (the verified L-Mul endpoint without
P3 storage flags). The fuller candidate stack with q08 + c2b stacked on
top was measured and fails the 0.835 VCTK soft target; q08 and c2b are
therefore **deferred** from the ship endpoint. In isolation they clear the
hard floors but miss the 0.835 VCTK soft target.

**Ship stack (Option C, measured)**:

```
Original strict W4A8 baseline (MBSTOI 0.8360 VCTK / 0.7927 TIMIT)
  + W4A8 strict QAT (W4 weights, A8 activations, per-group scalar)   [Apr 2026 baseline]
  + Tied MHA (MHA_re ≡ MHA_im)                                       [Apr 2026 baseline]
  + Softmax = shift_topk_pow2_clamp8_nomantissa (top-k=4)             [P0, Apr 2026]
  + Decoder grouped pointwise: groups_x=4, groups_skip=2 (P1 g42)     [P1]
  + Decoder skip subset: skip_select_ratio=0.5, mode=topk_l1 (P2)    [P2]
  + FP16 accumulator in attention QK/softmax/SV                       [P3 a1]
  + L-Mul M4 on BOTH attention matmuls (QK^T and Score×V)             [P4 option C]

Deferred (clear hard floors in isolation, but miss the 0.835 VCTK soft target
and regress badly when stacked on Option C):
  - Probability storage q0.8 (8-bit fixed-point)         [P3 b_prob_q08]
  - INT4 weight scale stored as FP8 E4M3                 [P3 c2b]
```

**Ship ckpt**: `Checkpoints/p4_qk_m4_qat_best.ckpt`
(symlink → `outputs/p4_lmul/lmul_qk_m4_2026-05-18_14-53-28/.../epoch=2-step=2904.ckpt`)

This ckpt was trained with `lmul_targets=[qk]` only. Score×V L-Mul is
applied at **inference time only** (empirically robust to the at-inference
swap; see [RESULTS_P4 §5](RESULTS_P4_LMUL_FEASIBILITY_FINAL_2026-05-18.md)
"Option C empirical surprise").

**Skip indices**: `Checkpoints/p2_skip_indices_g42_topk_l1_r050.pt`

### 0.1 What is measured

| Configuration | VCTK MBSTOI | TIMIT MBSTOI | Above 0.835 / 0.7869? |
|---|---|---|---|
| P1 g42 alone (isolated) | 0.8361 | 0.7923 | both ✓ |
| P2 g42+s050_topk alone (isolated) | 0.8352 | 0.7919 | both ✓ |
| P3 a1 isolated on P2 | 0.8352 | 0.7914 | both ✓ |
| P3 b_prob_q08 isolated on P2 | 0.8346 | 0.7919 | VCTK ✗ / TIMIT ✓ |
| P3 c2b isolated on P2 | 0.8349 | 0.7906 | VCTK ✗ / TIMIT ✓ |
| **Option C** (qk_m4 QAT ckpt + a1 + L-Mul[qk,score_v]) ⭐ | **0.8364** | **0.7925** | **both ✓** |
| Full candidate stack (Option C + b_prob_q08 + c2b) | 0.8326 | 0.7889 | VCTK ✗ / TIMIT ✓ |

Full-stack measurement raw output:
[results/final_stack_p1p2p3all_p4optC_vctk_timit_w750/](results/final_stack_p1p2p3all_p4optC_vctk_timit_w750/)
(log: `logs/final_stack_eval_*.log`)

**Interpretation**:
- **Option C clears the 0.835 VCTK target** (0.8364) and has comfortable
  margin on TIMIT (0.7925 vs floor 0.7869). All gates pass.
- **Full candidate stack** (Option C + q08 + c2b) drops to VCTK 0.8326 and
  TIMIT 0.7889 — passes hard floors but **fails the 0.835 VCTK target**,
  and TIMIT margin shrinks to 0.0020 (right at the kill rule threshold).
- The combined Δ (-0.0038 VCTK, -0.0036 TIMIT vs Option C) is significantly
  worse than what isolated runs predicted (q08 isolated lost -0.0006 VCTK;
  c2b isolated lost -0.0003 VCTK — additive predict says -0.0009, observed
  -0.0038). The components **do not stack linearly**.
- Likely cause: q0.8 prob discretization + FP8 W4 scale storage shift +
  L-Mul mantissa truncation all act on the same attention numerical pathway,
  and their joint distribution shift exceeds what any single change adds.
  This is exactly the cumulative-verification risk Codex flagged.

### 0.2 Deferred from ship endpoint

q0.8 prob storage and FP8 W4 scale storage clear the hard floors in isolation
but miss the 0.835 VCTK soft target individually, and are **rejected from the
cumulative ship stack** because of the larger joint degradation observed.
Their RTL benefits (8-bit prob register, 4× compressed W4 scale SRAM) are
SRAM-width nice-to-haves, not on the critical multiplier-energy or MAC-count
path. The L-Mul attention multipliers (P4 Option C) carry the bulk of the
area/energy win.

If a future round wants q08 and/or c2b back in the stack, suggested
mitigation:
- Test q08-on-Option-C and c2b-on-Option-C individually (without each
  other) to identify which contributes the destructive interaction
- Consider a joint QAT (`lmul_targets=[qk,score_v]` + `prob_quant=q0.8` from
  the start of finetune, with STE patches now in place) — may recover the
  joint margin
- Pick whichever subset preserves the 0.835 target

### 0.3 Quality gates (P2-derived hard floors / stereo bounds)

All measured rows above clear the hard gates:
- VCTK MBSTOI floor 0.8302 (clearance +0.0062)
- TIMIT MBSTOI floor 0.7869 (clearance +0.0056)
- VCTK ILD ≤ 3.45 (clearance 0.19), IPD ≤ 16.74 (clearance 0.82)
- TIMIT ILD ≤ 3.20 (clearance 0.16), IPD ≤ 21.20 (clearance 0.90)

---

## 1. Compute / MAC summary

### 1.1 Verified MAC reductions

The decoder pointwise reductions (P1 + P2) are MAC-count reductions verified
by analytical counters under [scripts/count_p1_macs.py](scripts/count_p1_macs.py)
and [scripts/count_p2_macs.py](scripts/count_p2_macs.py). The full-model MAC
columns count Conv2d / ConvTranspose2d only (not MHA Linear, not attention
scores). See P2 RESULTS §3 for caveats.

| Stage | dec_pw MAC vs baseline | conv-only total MAC vs baseline |
|---|---|---|
| baseline | 0 | 0 |
| P1 g42 | -62.16% | -38.70% |
| P1 g42 + P2 s050_topk | -74.74% | -46.52% |

### 1.2 Attention multiplier energy (P4 estimate)

P4 L-Mul replaces FP multiplier with mantissa-adder + offset. Per L-Mul
paper (arxiv:2410.00907), M4 mantissa configuration saves ~70% multiplier
energy vs FP MAC at equivalent attention throughput.

| Configuration | QK^T multiplier | Score×V multiplier | Attention block multiplier energy |
|---|---|---|---|
| baseline | FP MAC | FP MAC | reference |
| **Option C (ship)** | **L-Mul M4** | **L-Mul M4** | **~-70% (both matmuls)** |

This is a multiplier-energy reduction, not a MAC-count reduction. Total
attention MACs are unchanged.

---

## 2. Block-by-block hardware changes

### 2.1 STFT / iSTFT
**Unchanged.** No P1-P4 work touches the STFT block. Hop=80, fft_len=128,
hamming window, onesided + DC drop. See [feature_extractors.py](DCNN/feature_extractors.py).

### 2.2 Encoder (6 stages, depthwise-separable complex conv)
**Unchanged.** P1 + P2 only touch the decoder pointwise. Encoder is still
the depthwise (5×1) + pointwise (1×1) chain from
[ComplexDepthwiseSeparableConv2d](DCNN/utils/complexPyTorch/complexLayers.py).
W4A8 strict quantization still applies.

### 2.3 Decoder (6 stages)

Replace `cat([x, skip], dim=1) → pointwise(1×1) → depthwise_T(5×1) → BN → PReLU`
with **branch-asymmetric grouped pointwise** + **skip channel subset**:

```
x   ┐
    ├─ pointwise_x   (groups=4, in=in_x,    out=out) ┐
                                                       ├─ add → depthwise_T → BN → PReLU
skip[:, kept_indices] (kept = in_skip * 0.5, group-aware select)
    └─ pointwise_skip (groups=2, in=kept,   out=out) ┘
```

Per-layer effective groups (last layer mask head has out=1 → forced groups=1):

| Decoder iter | in_x (cmplx) | in_skip (cmplx) | out (cmplx) | groups_x_eff | groups_skip_eff | kept skip (ratio=0.5) |
|---|---|---|---|---|---|---|
| 0 (deepest) | 128 | 128 | 128 | 4 | 2 | 64 |
| 1 | 128 | 128 | 64 | 4 | 2 | 64 |
| 2 | 64 | 64 | 32 | 4 | 2 | 32 |
| 3 | 32 | 32 | 16 | 4 | 2 | 16 |
| 4 | 16 | 16 | 8 | 4 | 2 | 8 |
| 5 (mask head) | 8 | 8 | 1 | 1 | 1 | 4 |

**Skip subset selection (P2 topk_l1)**:
- Indices are **frozen at training time** from a one-pass valset L1-magnitude
  scan over the P2 g42 ckpt's encoder outputs (`scripts/precompute_skip_indices.py`).
- For each decoder layer, sort each skip channel group by mean |activation|,
  keep top half within each group, preserve group-wise ordering.
- Indices live in `Checkpoints/p2_skip_indices_g42_topk_l1_r050.pt` and are
  also stored as state-dict buffers `decoder.skip_indices_{stage}` in the
  shipped ckpt so RTL just reads them at boot.

RTL impact:
- Pointwise multiplier count per layer reduced by `1/groups` factor.
- Skip path bandwidth halved (only 50% of encoder skip channels are read by
  the decoder pointwise input).
- No structural change to depthwise transpose, BN, PReLU.

### 2.4 MHA (single layer, dim=256, heads=16, head_dim=16)

Multi-front change. RTL block diagram changes from:

```
[Q proj] [K proj] [V proj]
      ↓        ↓       ↓
       \      /        |
      QK^T (FP MAC, fp32 accum)
        ↓
    × 1/sqrt(D) (multiply)
        ↓
    + mask (-inf)
        ↓
    standard FP softmax (exp + div)
        ↓
    [probs, fp32]
        ↓
    matmul with V (FP MAC, fp32 accum)
        ↓
[O proj]
```

to:

```
[Q proj] [K proj] [V proj]   ← tied weights (MHA_re ≡ MHA_im, halves weight SRAM)
      ↓        ↓       ↓
       \      /        |
      cast to FP16
        ↓
      QK^T (L-Mul M4)            ← P4: 4-bit mantissa adder + offset
        ↓
    × 1/sqrt(D) = × 1/4 = shift right 2    ← P3 (D=16 → pow2 native)
        ↓
    + sliding window mask                  ← unchanged
        ↓
    shift_topk_pow2_clamp8_nomantissa     ← Softmax handoff
        ↓
    top-k=4 dense scatter (4 nonzero entries per row)
        ↓
    [probs in FP16 register]              ← q0.8 deferred (combined-stack regression)
        ↓
    matmul with V (L-Mul M4)              ← P4: 4-bit mantissa adder
        ↓
      cast back to original dtype
        ↓
[O proj]
```

Key per-block RTL changes:

| Sub-block | Pre-P3 | Ship (Option C) | Deferred (combined regression) |
|---|---|---|---|
| Q/K/V/O Linear weights | FP32 / W4A8 | **tied (re ≡ im), W4A8, per-head per-group scale** | – |
| Q/K/V/O scratch dtype | fp32 | **FP16** (P3 a1) | – |
| Q×K^T multiplier | FP MAC | **L-Mul M4 adder + offset** (P4 option C) | – |
| 1/√D scale | FP mul | **pure barrel shift (>>2)** (qk_scale_mode=pow2, native pow2 D=16) | – |
| Mask | additive -inf | unchanged (sliding window, attn_win=31) | – |
| Softmax | exp + div | **shift-topk pow2 clamp8 nomantissa** (top-k=4) | – |
| Probability register | FP16 | **FP16** (q08 deferred — see §0.1) | q0.8 (8-bit) shown to regress in combined stack |
| Score×V multiplier | FP MAC | **L-Mul M4 adder + offset** (P4 option C) | – |
| Top-k=4 sparse AV | dense (32 entries) | **4-entry gather + 4-MAC weighted sum per row** (softmax handoff) | – |
| W4 weight scale storage | FP32 in SRAM | **FP32** (c2b deferred — see §0.1) | FP8 E4M3 shown to regress in combined stack |

W4 quantization scale storage (ship endpoint = c2b deferred):
- INT4 weights with **per-head per-group scalar** (group_size=32)
- Scale tensor remains FP32 in SRAM. c2b FP8 is deferred per §0.1; isolated
  c2b clears hard floors but misses the 0.835 VCTK soft target, and the
  combined stack with q08+OptC drops VCTK by 0.0038. c2b can be revisited
  later if the 4× scale-tensor storage compression is needed.
- Per-MAC dequant happens at the input of the multiplier (existing W4A8
  dataflow, unchanged)

### 2.5 Bottleneck

The decoder grouping doesn't reach into the MHA bottleneck. MHA is the
single complex MultiheadAttention layer between encoder stage 5 and decoder
stage 0. No structural change beyond what's listed in §2.4.

---

## 3. Software config flags (ship configuration = Option C as measured)

For the **functional software reference** (Python forward pass that the RTL
team can compare against during co-simulation; NOT bit-exact RTL — see §5.6),
use this Hydra override set on top of the default
[config/config.yaml](config/config.yaml):

```yaml
# Quantization
training:
  mixed_precision_qat: true
  qat_scope: full
  top_k_attention: 4
  int4_group_size: 32
  int4_scale_format: fp32             # ship = default; c2b (FP8) deferred per §0.1
  fp16_accumulator: true              # P3 a1

# Model
model:
  tied_mha: true                      # base (Apr 2026)
  attention_softmax:
    mode: shift_topk_pow2_clamp8_nomantissa   # Softmax handoff (May 2026)

  decoder_grouped_pointwise:
    mode: grouped                     # P1 g42 entry point
    groups_x: 4                       # P1
    groups_skip: 2                    # P1
    channel_shuffle: false            # deferred
    skip_select_ratio: 0.5            # P2 s050
    skip_select_mode: topk_l1         # P2
    skip_indices_path: Checkpoints/p2_skip_indices_g42_topk_l1_r050.pt

  attention_datapath:
    enabled: true                     # P3 + P4
    qk_scale_mode: pow2               # P3 (D=16, natively pow2)
    score_quant: none                 # P3: score-quant rejected (see §5.3)
    prob_quant: none                  # ship = default; q0.8 deferred per §0.1
    lmul_enabled: true                # P4
    lmul_targets: [qk, score_v]       # P4 option C
    lmul_mantissa_bits: 4             # P4 M4
```

CKPT: `Checkpoints/p4_qk_m4_qat_best.ckpt`

Important details:
- `score_quant=none` keeps the FP16 score register; both `fp8_e4m3` and
  `fixed` (Q10.17) were tested and rejected
  ([RESULTS_P3 §4.1–4.4](RESULTS_P3_ATTENTION_FP8_FINAL_2026-05-18.md)).
- `prob_quant=none` and `int4_scale_format=fp32` are the ship defaults; the
  `q0.8` / `fp8_e4m3` alternatives clear hard floors individually but miss
  the 0.835 VCTK soft target, and the combined Option C + q0.8 + fp8 scale
  stack drops VCTK to 0.8326 (below 0.835 target). See §0.1 for the
  measurement and §0.2 for the deferral rationale.

---

## 4. Quality numbers — full sign-off table

VCTK 750-pair × 8 SNR + TIMIT 750-pair × 8 SNR.

| Stack | VCTK MBSTOI | TIMIT MBSTOI | VCTK ILD/IPD | TIMIT ILD/IPD |
|---|---|---|---|---|
| Original strict W4A8 baseline | 0.8360 | 0.7927 | 3.236 / 15.69 | 2.98 / 20.02 |
| + softmax shift_topk_pow2_clamp8_nomantissa | (already in baseline) | (already in baseline) | – | – |
| + P1 g42 (grouped pointwise) | 0.8361 | 0.7923 | 3.263 / 16.04 | 3.013 / 20.32 |
| + P2 s050_topk (skip subset 50%) | 0.8352 | 0.7919 | 3.287 / 15.94 | 3.052 / 20.19 |
| + P3 a1 (FP16 accumulator) [isolated] | 0.8352 | 0.7914 | 3.280 / 15.95 | 3.052 / 20.32 |
| + P3 b_prob_q08 (q0.8 prob) [isolated] | 0.8346 | 0.7919 | 3.324 / 15.98 | 3.071 / 20.26 |
| + P3 c2b (FP8 W4 scale) [isolated] | 0.8349 | 0.7906 | 3.316 / 15.91 | 3.085 / 20.20 |
| **+ P4 Option C (L-Mul both)** ⭐ **SHIP** | **0.8364** | **0.7925** | **3.264 / 15.92** | **3.039 / 20.30** |
| + P4 Option C + P3 q08 + P3 c2b (full candidate stack) | 0.8326 | 0.7889 | 3.267 / 15.97 | 2.997 / 20.30 |

**Important provenance** (per Codex review 2026-05-19):
- "**Option C** row" (the SHIP row): measured with the qk_m4 QAT ckpt and
  the following inference-time overrides: `fp16_accumulator=true`,
  `lmul_targets=[qk,score_v]`, `lmul_mantissa_bits=4`. `prob_quant` was
  `none`, `int4_scale_format` was `fp32`.
  Raw output: [results/p4_lmul_timit/lmul_optC_qkqat_plus_scorev_vctk_timit_w750/](results/p4_lmul_timit/lmul_optC_qkqat_plus_scorev_vctk_timit_w750/)
- "**Full candidate stack** row": measured with the same ckpt and the §3
  override set (adds `prob_quant=q0.8`, `int4_scale_format=fp8_e4m3`,
  `qk_scale_mode=pow2`). VCTK drops to 0.8326 (below 0.835 target); see §0.1.
  Raw output: [results/final_stack_p1p2p3all_p4optC_vctk_timit_w750/](results/final_stack_p1p2p3all_p4optC_vctk_timit_w750/)
- The P3 individual rows (a1, b_prob_q08, c2b) were measured separately on
  top of P2, each with only its own flag enabled (no Option C).

---

## 5. Open items / unverified combinations

### 5.1 P3 storage combinations

The three P3 variants (a1_fp16_accum, b_prob_q08, c2b_scale_fp8) were tested
individually on top of P2 g42+s050_topk. All clear the hard floors, but
b_prob_q08 and c2b_scale_fp8 miss the 0.835 VCTK soft target individually.

The P3 storage options stacked on Option C **were measured**:

```text
Option C + b_prob_q08 + c2b_scale_fp8
  VCTK 0.8326 / TIMIT 0.7889
```

This clears hard floors but fails the 0.835 VCTK target, so q08 and c2b are
deferred from the ship endpoint. The unmeasured item is narrower: a **P3-only
combined** run on top of P2 without L-Mul was not run, because the ship path
already rejects q08/c2b when combined with Option C.

If a future round wants q08 and/or c2b back, do not use the current
`scripts/run_phase3_attention_fp8_sweep.sh` command as-is: it only has
individual P3 variants in `variant_overrides`. Add an explicit combined
variant or run a direct `eval.py` / QAT command that records all overrides in
the log.

### 5.2 Option C composition was not jointly QAT'd

Option C uses the qk_m4 QAT'd ckpt, then adds Score×V L-Mul at inference. The
model was QAT'd assuming Score×V was FP MAC; the runtime swap introduces an
unmodeled discrepancy. Empirically the quality is preserved (VCTK 0.8364 vs
qk_m4 alone 0.8365), but a joint QAT (`lmul_targets=[qk, score_v]` from the
start of finetune) was not run. **STE patches (now landed) make this
experiment safer to retry** if anyone wants to verify joint QAT matches
Option C's inference-time composition.

### 5.3 score_quant=fixed (Q10.17) rejected structurally

Two attempts:
- Pre-STE-patch: hard fail (sanity 0.7966 << floor 0.8031)
- Post-STE-patch (this run): **same fail (sanity 0.7969)**, despite
  gradient flow confirmed (val_loss -12.224 → -12.344)

The blocker is structural: **double-discretization between Q10.17 fixed-point
score storage and the shift_topk_pow2_clamp8_nomantissa softmax** (which is
already an exponent-domain discretizer). RTL should **NOT revisit
score-domain quantization while shift-topk softmax is in use**. Keep scores
in FP16 register.

### 5.4 L-Mul M3 not tested

`lmul_attn_m3` (3-bit mantissa instead of 4) would halve mantissa-adder area
on top of M4 savings. Not tested in this work. Run only if RTL area budget
is insufficient with M4.

### 5.5 Repeated-seeds verification absent

All P1-P4 numbers in §4 are from single training + single eval runs. The
small positive deltas (e.g., Option C VCTK +0.0004 vs baseline) are within
cross-run noise band — read them as "no quality regression observed", not
as verified improvements. For paper-style claims, ≥3 seeds with mean ± std
would be needed.

### 5.6 No bit-exact RTL co-simulation yet

The Python forward pass with the §3 config is a **functional software
reference** / **golden PyTorch reference**, not bit-exact RTL. The L-Mul
simulator [`_lmul_multiply` :930](DCNN/utils/complexPyTorch/complexLayers.py#L930)
explicitly notes it's a "tensor-level feasibility simulator, not bit-exact
RTL". The W4A8 + per-group scale dequantization is also implemented as
fake-quant, not the saturating / accumulator-width logic the RTL will use.

Final hardware verification needs RTL co-simulation against the same VCTK /
TIMIT input set; until that's done, any sub-0.001 MBSTOI claim in §0.1 / §4
should be treated as "PyTorch simulator delta", not as a hardware-side
claim.

---

## 6. Decision history reference

| Decision | Source doc | One-line summary |
|---|---|---|
| Replace exp+div softmax | [Softmax handoff](../11_weight_tied_mha_fast_mbstoi/RTL_HANDOFF_SOFTMAX_2026-05-14.md) | shift_topk_pow2_clamp8_nomantissa with top-k=4 |
| Decoder grouped pointwise | [RESULTS_P1](../12_decoder_grouped_pointwise/RESULTS_P1_GROUPED_2026-05-17.md) | g42 (gx=4, gs=2) — branch-asymmetric |
| Decoder skip subset | [RESULTS_P2](../12_decoder_grouped_pointwise/RESULTS_P2_SKIP_2026-05-18.md) | s050_topk (ratio=0.5, topk_l1) |
| FP16 attention accumulator | [RESULTS_P3 §2](RESULTS_P3_ATTENTION_FP8_FINAL_2026-05-18.md) | a1 clears hard floors isolated |
| q0.8 probability storage | [RESULTS_P3 §2](RESULTS_P3_ATTENTION_FP8_FINAL_2026-05-18.md) | b_prob_q08 clears hard floors isolated; misses 0.835 VCTK soft target |
| FP8 W4 scale | [RESULTS_P3 §2](RESULTS_P3_ATTENTION_FP8_FINAL_2026-05-18.md) | c2b clears hard floors isolated; misses 0.835 VCTK soft target |
| Score-domain quant REJECTED | [RESULTS_P3 §4](RESULTS_P3_ATTENTION_FP8_FINAL_2026-05-18.md) | b_score_fp8 B0 fail; b_score_fixed structural fail (even with STE) |
| L-Mul on both attention matmuls | [RESULTS_P4 §5](RESULTS_P4_LMUL_FEASIBILITY_FINAL_2026-05-18.md) | Option C: qk_m4 QAT ckpt + scorev L-Mul inference |

---

## 7. Reproducibility

### 7.1 As-measured Option C (the one with the §0.1 "Option C as evaluated" numbers)

This is what produced VCTK 0.8364 / TIMIT 0.7925. Note `prob_quant` and
`int4_scale_format` are at their defaults (off / fp32), not the candidate
stack values.

```bash
cd /raid/R12K41024/13_aggressive_algo_qat
conda activate bcctn

CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n bcctn python eval.py \
  --model_checkpoint Checkpoints/p4_qk_m4_qat_best.ckpt \
  --vctk_test_dir /raid/R12K41024/BCCTN/Dataset/noisy_testset \
  --vctk_clean_dir /raid/R12K41024/BCCTN/Dataset/clean_testset \
  --timit_test_dir /raid/R12K41024/BCCTN/Dataset/noisy_testset_timit \
  --timit_clean_dir /raid/R12K41024/BCCTN/Dataset/clean_testset_timit \
  --output_dir results/p4_lmul_timit/lmul_optC_qkqat_plus_scorev_vctk_timit_w750 \
  --paper_style_eval --batch_size 8 --limit_pairs 750 --metric_workers 64 \
  --hydra_overrides \
    training.qat_scope=full \
    training.top_k_attention=4 \
    training.fp16_accumulator=true \
    model.attention_softmax.mode=shift_topk_pow2_clamp8_nomantissa \
    model.decoder_grouped_pointwise.mode=grouped \
    model.decoder_grouped_pointwise.groups_x=4 \
    model.decoder_grouped_pointwise.groups_skip=2 \
    model.decoder_grouped_pointwise.skip_select_ratio=0.5 \
    model.decoder_grouped_pointwise.skip_select_mode=topk_l1 \
    model.decoder_grouped_pointwise.skip_indices_path=Checkpoints/p2_skip_indices_g42_topk_l1_r050.pt \
    model.attention_datapath.enabled=true \
    model.attention_datapath.lmul_enabled=true \
    'model.attention_datapath.lmul_targets=[qk,score_v]' \
    model.attention_datapath.lmul_mantissa_bits=4
```

### 7.2 Full candidate stack (Option C + q08 + c2b) — measured, NOT shipped

This stack was measured per Codex review (bash `bmzipexpq`, log
`logs/final_stack_eval_*.log`) and **failed the 0.835 VCTK target** (VCTK
0.8326, TIMIT 0.7889). Kept here for reference and for future revisits if a
joint-QAT schedule can recover the joint margin.

```bash
cd /raid/R12K41024/13_aggressive_algo_qat
conda activate bcctn

# Quality reproduction (VCTK + TIMIT full 750-pair)
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n bcctn python eval.py \
  --model_checkpoint Checkpoints/p4_qk_m4_qat_best.ckpt \
  --vctk_test_dir /raid/R12K41024/BCCTN/Dataset/noisy_testset \
  --vctk_clean_dir /raid/R12K41024/BCCTN/Dataset/clean_testset \
  --timit_test_dir /raid/R12K41024/BCCTN/Dataset/noisy_testset_timit \
  --timit_clean_dir /raid/R12K41024/BCCTN/Dataset/clean_testset_timit \
  --output_dir results/final_stack_p1p2p3all_p4optC_vctk_timit_w750 \
  --paper_style_eval --batch_size 8 --limit_pairs 750 --metric_workers 64 \
  --hydra_overrides \
    training.qat_scope=full \
    training.top_k_attention=4 \
    training.fp16_accumulator=true \
    training.int4_scale_format=fp8_e4m3 \
    model.tied_mha=true \
    model.attention_softmax.mode=shift_topk_pow2_clamp8_nomantissa \
    model.decoder_grouped_pointwise.mode=grouped \
    model.decoder_grouped_pointwise.groups_x=4 \
    model.decoder_grouped_pointwise.groups_skip=2 \
    model.decoder_grouped_pointwise.skip_select_ratio=0.5 \
    model.decoder_grouped_pointwise.skip_select_mode=topk_l1 \
    model.decoder_grouped_pointwise.skip_indices_path=Checkpoints/p2_skip_indices_g42_topk_l1_r050.pt \
    model.attention_datapath.enabled=true \
    model.attention_datapath.qk_scale_mode=pow2 \
    model.attention_datapath.prob_quant=q0.8 \
    model.attention_datapath.lmul_enabled=true \
    'model.attention_datapath.lmul_targets=[qk,score_v]' \
    model.attention_datapath.lmul_mantissa_bits=4
```

Measured output: **VCTK 0.8326 / TIMIT 0.7889** (fails 0.835 target;
clears hard floors with thin margin). Raw output:
[results/final_stack_p1p2p3all_p4optC_vctk_timit_w750/](results/final_stack_p1p2p3all_p4optC_vctk_timit_w750/).

---

## 8. Per-block file index for RTL implementers

| Block | Python reference (functional forward) |
|---|---|
| STFT / iSTFT | [DCNN/feature_extractors.py](DCNN/feature_extractors.py) |
| Encoder | [DCNN/models/model.py:127-183](DCNN/models/model.py#L127-L183) |
| Decoder + grouped pointwise + skip select | [DCNN/models/model.py:297-448](DCNN/models/model.py#L297-L448) |
| Split conv | [DCNN/utils/complexPyTorch/complexLayers.py:1425](DCNN/utils/complexPyTorch/complexLayers.py#L1425) (`ComplexDepthwiseSeparableConvTranspose2dSplit`) |
| MHA forward | [complexLayers.py:1158](DCNN/utils/complexPyTorch/complexLayers.py#L1158) (`_mha_attention_forward`) |
| Softmax (shift_topk_pow2_clamp8_nomantissa) | [complexLayers.py:1082](DCNN/utils/complexPyTorch/complexLayers.py#L1082) (`_shift_topk_softmax`) |
| QK scale (pow2) | [complexLayers.py:877](DCNN/utils/complexPyTorch/complexLayers.py#L877) (`_apply_qk_scale`) |
| Probability q0.8 quant | [complexLayers.py:845](DCNN/utils/complexPyTorch/complexLayers.py#L845) (`_quantize_attention_probs`) |
| L-Mul multiplier | [complexLayers.py:930](DCNN/utils/complexPyTorch/complexLayers.py#L930) (`_lmul_multiply`) |
| L-Mul matmul | [complexLayers.py:966](DCNN/utils/complexPyTorch/complexLayers.py#L966) (`_lmul_matmul`) |
| STE wrappers (`_RoundSTE` / `_FloorSTE`) | [complexLayers.py:762](DCNN/utils/complexPyTorch/complexLayers.py#L762) — applied at `_quantize_attention_scores` :835, `_quantize_attention_probs` :858, `_quantize_fraction_bits` :927 |
| W4A8 mixed-precision QAT (incl. FP8 scale storage) | [DCNN/utils/mixed_precision_qat.py](DCNN/utils/mixed_precision_qat.py) |

---

## 9. Review status / residual caveats

Codex review pass (2026-05-19):
- §0 / §3 / §4 now consistently define the ship endpoint as **Option C without
  q08 or c2b**.
- Option C and full-candidate quality numbers were cross-checked against the
  raw `*_averages.txt` files listed in §4.
- Full-candidate provenance exists in `logs/final_stack_eval_20260519_003758.log`.
  A dedicated Option C eval log was not found; §7.1 is the reproduction command
  to rerun and archive if log-level provenance is required.
- §8 line references were spot-checked after the STE patches.

Remaining hardware-side validation is unchanged: run RTL co-simulation against
the same VCTK / TIMIT input set before treating sub-0.001 MBSTOI differences as
hardware-side claims.
