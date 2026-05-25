# Handoff: All-MAC L-Mul Mixed W4A8 Eval

## Goal

Evaluate whether the mixed hardware multiplier plan can replace all
MAC-bearing products with L-Mul M4:

- INT4 weight x FP8 E4M3 activation paths from W4A8 QAT.
- FP8 weight x FP8 E4M3 activation paths in decoder outlier pointwise layers.
- Attention QKV/out projections, QK, and Score x V.

This is eval-only. It uses the existing final-stack checkpoint and does not
train/recover the added all-MAC L-Mul noise.

## New Folder

```bash
cd /raid/R12K41024/15_all_lmul_mixed_mult_eval
```

This folder was copied from `14_low_power_arch_qat` with old outputs/results
excluded.

## Code Changes

- `DCNN/utils/all_lmul_multiply.py`
  - Patches `nn.Conv2d`, `nn.ConvTranspose2d`, and `nn.Linear` forward methods
    to use the existing L-Mul product primitive plus normal accumulation.
  - Handles grouped/depthwise Conv2d and ConvTranspose2d.
  - When enabled, it also forces `ComplexMultiheadAttention` into custom
    forward with `qk`, `score_v`, and projection L-Mul active.

- `DCNN/utils/complexPyTorch/complexLayers.py`
  - Adds `_lmul_linear(...)`.
  - Adds `attention_datapath.lmul_all_projection_ops`, so QKV and attention
    out projection can use L-Mul instead of direct `F.linear`.

- `DCNN/trainer.py`
  - Applies the all-MAC L-Mul patch after W4A8/FP8 parametrization when
    `model.low_power.all_lmul_multiply.enabled=true`.

- `config/config.yaml`
  - Adds:
    - `model.attention_datapath.lmul_all_projection_ops`
    - `model.low_power.all_lmul_multiply.enabled`
    - `model.low_power.all_lmul_multiply.mantissa_bits`
    - `model.low_power.all_lmul_multiply.bit_exact`

- `scripts/run_all_lmul_mixed_eval.sh`
  - Runs two variants on the same checkpoint/input set:
    - `baseline_final`
    - `all_lmul_m4_bitexact`

## Commands

20-pair VCTK sanity:

```bash
cd /raid/R12K41024/15_all_lmul_mixed_mult_eval
GPU_ID=4 LIMIT_PAIRS=20 BATCH_SIZE=1 METRIC_WORKERS=64 \
  RESULT_ROOT=results/all_lmul_mixed_eval_w20 \
  bash scripts/run_all_lmul_mixed_eval.sh
```

QAT rescue with memory-safe chunking/checkpointing:

```bash
cd /raid/R12K41024/15_all_lmul_mixed_mult_eval
GPU_ID=4 EPOCHS=5 LR=1e-4 TRAIN_BATCH_SIZE=4 ACCUMULATE_GRAD_BATCHES=4 \
  LMUL_LINEAR_CHUNK_SIZE=64 LMUL_CHECKPOINT_CHUNKS=true \
  LIMIT_PAIRS=20 BATCH_SIZE=1 METRIC_WORKERS=64 \
  bash scripts/run_all_lmul_mixed_qat_rescue.sh
```

The command above uses physical batch 4 with gradient accumulation 4, so the
optimizer sees effective batch 16 without materializing batch 16 all at once.
If memory allows, try `TRAIN_BATCH_SIZE=8 ACCUMULATE_GRAD_BATCHES=2` for the
same effective batch with fewer optimizer-side steps.

Full VCTK + TIMIT signoff-style eval:

```bash
cd /raid/R12K41024/15_all_lmul_mixed_mult_eval
GPU_ID=4 RUN_TIMIT=1 LIMIT_PAIRS=750 BATCH_SIZE=1 METRIC_WORKERS=64 \
  RESULT_ROOT=results/all_lmul_mixed_eval_vctk_timit_w750 \
  bash scripts/run_all_lmul_mixed_eval.sh
```

If GPU is actually active, the log should say `Evaluating on cuda`, not
`Evaluating on cpu`. Codex smoke ran on CPU because CUDA/NVML was unavailable
inside this session.

## Output Locations

The script writes:

```text
results/all_lmul_mixed_eval*/baseline_final_w*/
results/all_lmul_mixed_eval*/all_lmul_m4_bitexact_w*/
logs/all_lmul_mixed_baseline_final_*.log
logs/all_lmul_mixed_all_lmul_m4_bitexact_*.log
```

Quick result extraction:

```bash
rg -n "Average MBSTOI Enhanced|Average ILD Error Enhanced|Average IPD Error Enhanced" \
  results/all_lmul_mixed_eval*/*/*_averages.txt logs/all_lmul_mixed_*.log
```

## Codex Smoke Result

CPU smoke completed with `LIMIT_PAIRS=1`, VCTK paper-style only.

| Variant | VCTK MBSTOI | PESQ | SegSNR | ILD | IPD |
|---|---:|---:|---:|---:|---:|
| baseline_final | 0.8156 | 1.5135 | 0.97 | 5.9096 dB | 15.69 deg |
| all_lmul_m4_bitexact | 0.7992 | 1.3145 | -0.48 | 5.4597 dB | 16.20 deg |
| delta | -0.0164 | -0.1990 | -1.45 | -0.4499 dB | +0.51 deg |

Do not treat this as quality signoff; it is a path/smoke check only. It did
confirm the all-MAC patch is active:

```text
L-Mul all-MAC patch enabled: 48 Conv2d, 12 ConvTranspose2d, 3 Linear,
1 attention modules; mantissa_bits=4, bit_exact=True
```

## Pass/Fail Guidance

Use same gates as the final-stack handoff:

- VCTK MBSTOI target: >= 0.835.
- TIMIT MBSTOI floor: >= 0.7869.
- Compare `all_lmul_m4_bitexact` against `baseline_final` from the same run.
- If 20-pair drops by more than about 0.003 MBSTOI, full 750-pair is unlikely
  to be safe without QAT recovery.

## Scope Note

This patch targets MAC-bearing Conv/ConvTranspose/Linear/attention products.
It does not replace small elementwise affine multiplications from BatchNorm or
PReLU; those are separate low-power knobs and are normally fused or handled by
the existing pow2 experiments.
