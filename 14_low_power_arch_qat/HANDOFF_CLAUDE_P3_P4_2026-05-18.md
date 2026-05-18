# HANDOFF: P3 Attention FP8 + P4 L-Mul

Date: 2026-05-18  
Workspace: `/raid/R12K41024/13_aggressive_algo_qat`  
Base candidate: P2 `g42 + s050_topk`

This handoff is for Claude to continue the formal experiments after Codex
implemented the P3/P4 scaffolding and smoke-tested it.

---

## 0. Immediate Status

Code and scripts are implemented. Formal quality sign-off is not complete.

Completed:

- P3 attention datapath scaffold:
  - `model.attention_datapath`
  - score quantization: `none / fp8_e4m3 / fixed`
  - probability quantization: `none / q0.8 / fp8_e4m3 / pow2`
  - QK scale mode: `pow2 / fp16 / fp8_e4m3`
  - INT4 scale format: `fp32 / fp8_e4m3 / pow2`
  - B0 dump/simulate scripts
- P4 L-Mul scaffold:
  - attention `QK^T`
  - attention `Score x V`
  - variants: `lmul_qk_m4`, `lmul_scorev_m4`, `lmul_attn_m4`, `lmul_attn_m3`
- Smoke tests:
  - Python syntax checks passed.
  - Shell syntax checks passed.
  - Tensor-level score/prob/L-Mul tests passed.
  - Model init with P2 grouped+skip and P3 datapath passed.
  - B0 dump/simulate smoke passed.
  - L-Mul `lmul_qk_m4` 1-pair no-train smoke ran end-to-end.

Not complete:

- Official P3-A1 10-epoch screening.
- Official P3 B0 dump with >=8 utterances and >=1000 rows/head.
- Official P3-B/C 20-pair, full VCTK, or TIMIT results.
- Official P4 L-Mul 20-pair / QAT recovery / full VCTK / TIMIT results.

Important: the previous Codex session could not see CUDA even inside the
`bcctn` conda env, so formal GPU runs were not started.

---

## 1. Must-Read Files

Read in this order:

```text
FUTURE_WORK_P3_ATTENTION_FP8_2026-05-17.md
RESULTS_P2_SKIP_2026-05-18.md
RESULTS_P3_ATTENTION_FP8_2026-05-18.md
RESULTS_P4_LMUL_FEASIBILITY_2026-05-18.md
```

Use `RESULTS_P2_SKIP_2026-05-18.md` as the source of truth for the P2 primary
candidate.

---

## 2. Environment Blocker First

Codex verified that the conda env was correct:

```text
python /home/R12K41024/miniforge3/envs/bcctn/bin/python
torch 2.3.1
torch_cuda 12.1
```

But CUDA was unavailable:

```text
cuda_available False
device_count 0
```

Also observed:

```text
/etc/profile.d/set_mig_gpu.sh: line 17: syntax error near unexpected token `)'
nvidia-smi has failed because it couldn't communicate with the NVIDIA driver
```

Before running formal experiments, fix GPU/MIG visibility and confirm:

```bash
cd /raid/R12K41024/13_aggressive_algo_qat

conda run --no-capture-output -n bcctn python -c \
  "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"

nvidia-smi -L
```

Expected:

```text
True <positive device count>
```

If `set_mig_gpu.sh` is still broken, do not assume experiments are valid.

---

## 3. Current Code Anchors

P3/P4 config:

```text
config/config.yaml
  model.attention_datapath
  training.fp16_accumulator
  training.int4_scale_format
```

Attention datapath:

```text
DCNN/utils/complexPyTorch/complexLayers.py
  _quantize_attention_scores
  _quantize_attention_probs
  _apply_qk_scale
  _lmul_multiply
  _lmul_matmul
  _mha_attention_forward
```

QAT wiring:

```text
DCNN/utils/mixed_precision_qat.py
  _quantize_scale
  _set_topk_on_complex_mha
  enable_mixed_precision_qat(..., attention_datapath=...)

DCNN/trainer.py
  passes model.attention_datapath into enable_mixed_precision_qat
```

Scripts:

```text
scripts/dump_p3_attention_scores.py
scripts/simulate_p3_attention_quant.py
scripts/run_phase3_attention_fp8_sweep.sh
scripts/run_phase3_attention_full_eval.sh
scripts/run_phase3_attention_timit.sh
scripts/run_p4_lmul_feasibility.sh
```

---

## 4. P3 Formal Run Plan

### 4.1 P3-A0 Reference Eval

This is no-retraining P2 reference through the P3 eval script.

```bash
cd /raid/R12K41024/13_aggressive_algo_qat

LIMIT_PAIRS=20 bash scripts/run_phase3_attention_full_eval.sh a0_p2
```

For full 750-pair VCTK later:

```bash
bash scripts/run_phase3_attention_full_eval.sh a0_p2
```

### 4.2 P3-A1 FP16 Accumulator Screening

Start with 5-10 epoch screening from P2 best:

```bash
EPOCHS=10 LR=2e-4 bash scripts/run_phase3_attention_fp8_sweep.sh a1_fp16_accum
```

If no recovery trend, retry once:

```bash
EPOCHS=10 LR=5e-4 bash scripts/run_phase3_attention_fp8_sweep.sh a1_fp16_accum
```

Promote only if the 20-pair result is recovering and not below the kill rule
from `FUTURE_WORK_P3_ATTENTION_FP8_2026-05-17.md`.

### 4.3 Official B0 Attention Score Dump

Run before any score quantization training:

```bash
conda run --no-capture-output -n bcctn python scripts/dump_p3_attention_scores.py \
  --limit_utterances 8 \
  --max_rows_per_head 1200 \
  --output_npz results/p3_attention_fp8/b0_attention_scores.npz
```

Then simulate:

```bash
conda run --no-capture-output -n bcctn python scripts/simulate_p3_attention_quant.py \
  --dump_npz results/p3_attention_fp8/b0_attention_scores.npz
```

Decision:

- If FP8 score gate passes worst-case head/layer, run `b_score_fp8`.
- If FP8 score gate fails, skip `b_score_fp8` and prefer `b_score_fixed` or
  FP16 score registers.
- Use the derived fixed-Q bits from the simulation output for `b_score_fixed`.

### 4.4 P3-B Isolated Ablations

Run only after B0:

```bash
# Only if B0 FP8 score gate passes
EPOCHS=10 LR=2e-4 bash scripts/run_phase3_attention_fp8_sweep.sh b_score_fp8

# If B0 recommends fixed-Q. Fill these from the B0 simulation JSON.
EPOCHS=10 LR=2e-4 SCORE_FIXED_INT_BITS=<B0_INT> SCORE_FIXED_FRAC_BITS=<B0_FRAC> \
  bash scripts/run_phase3_attention_fp8_sweep.sh b_score_fixed

# Mandatory probability storage ablation
EPOCHS=10 LR=2e-4 bash scripts/run_phase3_attention_fp8_sweep.sh b_prob_q08

# Optional sanity only
EPOCHS=10 LR=2e-4 bash scripts/run_phase3_attention_fp8_sweep.sh b_prob_fp8
```

### 4.5 P3-C Scale Ablations

Only after P3-B passes:

```bash
EPOCHS=10 LR=2e-4 bash scripts/run_phase3_attention_fp8_sweep.sh c2b_scale_fp8
```

Run pow2 only if C2b has enough margin:

```bash
EPOCHS=10 LR=2e-4 bash scripts/run_phase3_attention_fp8_sweep.sh c2c_scale_pow2
```

### 4.6 Full VCTK + TIMIT

For any variant that passes 20-pair screening:

```bash
bash scripts/run_phase3_attention_full_eval.sh <variant>
bash scripts/run_phase3_attention_timit.sh <variant>
```

Recommended initial promotion candidates:

```text
a1_fp16_accum
b_score_fixed or b_score_fp8
b_prob_q08
c2b_scale_fp8
```

---

## 5. P4 L-Mul Formal Run Plan

The user specifically wants to know whether L-Mul is feasible in this work.
Do not mix L-Mul with score/prob quantization until both pass in isolation.

### 5.1 No-Train 20-Pair Feasibility

```bash
cd /raid/R12K41024/13_aggressive_algo_qat

TRAIN=0 LIMIT_PAIRS=20 bash scripts/run_p4_lmul_feasibility.sh \
  lmul_qk_m4 lmul_scorev_m4 lmul_attn_m4
```

Interpretation:

- `lmul_qk_m4`: tests score generation sensitivity.
- `lmul_scorev_m4`: tests value mixing sensitivity.
- `lmul_attn_m4`: tests both together.

### 5.2 QAT Recovery Screening

If no-train 20-pair does not collapse:

```bash
EPOCHS=10 LR=2e-4 bash scripts/run_p4_lmul_feasibility.sh lmul_qk_m4
EPOCHS=10 LR=2e-4 bash scripts/run_p4_lmul_feasibility.sh lmul_scorev_m4
```

If needed, retry once:

```bash
EPOCHS=10 LR=5e-4 bash scripts/run_p4_lmul_feasibility.sh lmul_qk_m4
EPOCHS=10 LR=5e-4 bash scripts/run_p4_lmul_feasibility.sh lmul_scorev_m4
```

Only test combined L-Mul after isolated targets look stable:

```bash
EPOCHS=10 LR=2e-4 bash scripts/run_p4_lmul_feasibility.sh lmul_attn_m4
```

### 5.3 More Aggressive M3

Run only if M4 is stable:

```bash
TRAIN=0 LIMIT_PAIRS=20 bash scripts/run_p4_lmul_feasibility.sh lmul_attn_m3
```

### 5.4 Full VCTK/TIMIT for L-Mul

No dedicated full/TIMIT script exists yet for P4. Use either:

1. Extend `scripts/run_phase3_attention_full_eval.sh` / `run_phase3_attention_timit.sh`
   with L-Mul variants; or
2. Add `scripts/run_p4_lmul_full_eval.sh` and `scripts/run_p4_lmul_timit.sh`
   mirroring the P3 scripts.

Do this only after 20-pair + QAT screening passes.

---

## 6. Gates

Reference values from P2:

```text
P2 g42+s050_topk:
  VCTK MBSTOI = 0.8352
  TIMIT MBSTOI = 0.7919
  20-pair sanity MBSTOI = 0.8081
  VCTK ILD/IPD = 3.2871 dB / 15.94 deg
  TIMIT ILD/IPD = 3.0520 dB / 20.19 deg
```

Quality gates:

```text
20-pair sanity:
  MBSTOI >= 0.8031

Full VCTK:
  target around 0.835-0.836
  hard floor: 0.8302

TIMIT:
  hard floor: 0.7869

Binaural cue ceilings:
  VCTK ILD <= 3.45 dB
  VCTK IPD <= 16.74 deg
  TIMIT ILD <= 3.20 dB
  TIMIT IPD <= 21.20 deg
```

For B0:

```text
Worst-case head/layer:
  top-4 Jaccard >= 0.98
  top-1 flip rate <= 1%
```

---

## 7. Existing Smoke Results

B0 smoke only:

```text
dump: /tmp/p3_b0_smoke.npz
summary: /tmp/p3_b0_smoke.json
simulation: /tmp/p3_b0_smoke_quant_sim.json
```

This was only 1 utterance and 16 rows/head. Do not use as an official gate.

Observation from tiny smoke:

```text
FP8 score failed the gate on tiny sample.
Fixed-Q passed on tiny sample.
```

Treat this only as a warning that official B0 may reject score FP8.

L-Mul smoke:

```text
TRAIN=0 LIMIT_PAIRS=1 bash scripts/run_p4_lmul_feasibility.sh lmul_qk_m4
```

Output:

```text
results/p4_lmul_sanity/lmul_qk_m4_base_w1/
```

Smoke average:

```text
Average MBSTOI Enhanced: 0.8629
Average ILD Error Enhanced: 4.1879 dB
Average IPD Error Enhanced: 14.69 deg
```

This proves `QK^T` L-Mul M4 runs end-to-end without immediate numerical
collapse. It is not a quality result because it is one pair.

---

## 8. Reporting Instructions

For P3:

- Report as datapath/storage/multiplier-width improvement.
- Do not report as new decoder MAC reduction.
- Keep deltas vs:
  - P2 best `g42+s050_topk`
  - original strict W4A8 baseline

For P4 L-Mul:

- State clearly that current simulator is not bit-exact RTL.
- Keep L-Mul isolated from score/prob quantization until isolated tests pass.
- First answer the user's main question:

```text
Can L-Mul replace attention QK and/or Score x V in this BCCTN model without
breaking MBSTOI/ILD/IPD after short QAT?
```

Structured Q/K/V/O Linear pruning should remain after L-Mul feasibility unless
the user explicitly asks to combine them.

