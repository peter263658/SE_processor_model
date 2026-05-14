# 09 Weight-Tied MHA Softmax RTL Handoff

日期：2026-05-14

目標：以 `09_weight_tied_mha` 的最新 strict 版本為基礎，先在軟體端模擬 RTL-friendly / exp-free attention probability path，再比較 VCTK paper-style metrics，確認是否值得往 finetune 或硬體實作走。

## Base Branch / Checkpoint

本次改動以 `09_weight_tied_mha` 為唯一基礎。

Strict baseline 設定沿用：

```text
model.tied_mha=true
training.qat_scope=full
training.top_k_attention=4
training.int4_group_size=32
```

預設 checkpoint：

```text
/raid/R12K41024/09_weight_tied_mha/outputs/finetune_strict/2026-05-10_02-40-16/logs/lightning_logs/version_0/checkpoints/epoch=45-step=44528.ckpt
```

目前沒有啟動 full eval、finetune、hub background job 或 watcher。

## Changed Files

### Core code

```text
09_weight_tied_mha/DCNN/utils/complexPyTorch/complexLayers.py
09_weight_tied_mha/DCNN/models/model.py
09_weight_tied_mha/DCNN/utils/mixed_precision_qat.py
09_weight_tied_mha/config/config.yaml
09_weight_tied_mha/verify_custom_attention.py
```

### New experiment folders

```text
09_weight_tied_mha/softmax_rtl_shift_based/
09_weight_tied_mha/softmax_rtl_taylor_relu/
```

## Softmax Modes

Hydra override:

```text
model.attention_softmax.mode=<mode>
```

Supported modes:

| Mode | Meaning | Hardware angle |
|---|---|---|
| `softmax` | Original PyTorch-style softmax path | Baseline |
| `shift` | Approximate `exp(x)` as `2^(1.5x)`, linear fractional term, denominator rounded to power-of-two | Adder / shifter friendly |
| `taylor2` | `1 + score + 0.5 * score^2`, clamp negative weights, row normalize | Exp-free, smoother than ReLU |
| `relu2` | `ReLU(score)^2`, row normalize, uniform fallback if row is all zero | Lowest arithmetic cost, highest model-risk |

Implementation detail:

- Approximate modes are implemented inside `_approx_attention_probs`.
- `shift` currently mimics a more RTL-like path by rounding denominator to a power-of-two reciprocal.
- `taylor2` and `relu2` use row normalization after masking.
- Approximate modes force `ComplexMultiheadAttention` into the existing custom forward path, so strict W4A8 and top-k stay in the same evaluation route.

## Interaction With Strict W4A8 / Top-k

The strict eval scripts still pass:

```text
training.qat_scope=full
training.top_k_attention=4
```

Attention order in custom forward:

```text
QKV projection
QK^T / sqrt(D)
causal sliding-window mask
approx probability path: softmax / shift / taylor2 / relu2
top-k masking and renormalization
Score x V
out projection
```

This preserves the existing 09 strict setup while only swapping the probability-generation path.

## New Scripts

### Shift-based

```text
09_weight_tied_mha/softmax_rtl_shift_based/run_eval_shift_strict.sh
09_weight_tied_mha/softmax_rtl_shift_based/run_finetune_shift_strict.sh
```

### Taylor2 / ReLU2

```text
09_weight_tied_mha/softmax_rtl_taylor_relu/run_eval_exp_free_strict.sh
09_weight_tied_mha/softmax_rtl_taylor_relu/run_finetune_exp_free_strict.sh
```

The scripts are meant to be launched with `bash`, so executable bit is not required.

## Verification Already Done

From `/raid/R12K41024`:

```bash
python -m py_compile \
    09_weight_tied_mha/DCNN/utils/complexPyTorch/complexLayers.py \
    09_weight_tied_mha/DCNN/models/model.py \
    09_weight_tied_mha/DCNN/utils/mixed_precision_qat.py \
    09_weight_tied_mha/verify_custom_attention.py
```

Result: passed.

```bash
bash -n \
    09_weight_tied_mha/softmax_rtl_shift_based/run_eval_shift_strict.sh \
    09_weight_tied_mha/softmax_rtl_shift_based/run_finetune_shift_strict.sh \
    09_weight_tied_mha/softmax_rtl_taylor_relu/run_eval_exp_free_strict.sh \
    09_weight_tied_mha/softmax_rtl_taylor_relu/run_finetune_exp_free_strict.sh
```

Result: passed.

From `/raid/R12K41024/09_weight_tied_mha`:

```bash
conda run --no-capture-output -n bcctn python verify_custom_attention.py
```

Result:

```text
[real] OK
[complex] OK
[top-k] output shape ok
[shift] output shape/finite ok
[taylor2] output shape/finite ok
[relu2] output shape/finite ok
All sanity checks passed.
```

Hydra override smoke check:

```bash
conda run --no-capture-output -n bcctn python -c "from hydra import compose, initialize; from hydra.core.global_hydra import GlobalHydra; GlobalHydra.instance().clear(); initialize(config_path='config', version_base=None); cfg=compose(config_name='config', overrides=['training.qat_scope=full','training.top_k_attention=4','model.attention_softmax.mode=shift']); print(cfg.model.attention_softmax.mode, cfg.training.qat_scope, cfg.training.top_k_attention)"
```

Result:

```text
shift full 4
```

## Recommended Next Steps

### 1. Fast smoke eval first

Use a small `LIMIT_PAIRS` before spending full GPU time:

```bash
cd /raid/R12K41024/09_weight_tied_mha

LIMIT_PAIRS=20 bash softmax_rtl_shift_based/run_eval_shift_strict.sh
LIMIT_PAIRS=20 bash softmax_rtl_taylor_relu/run_eval_exp_free_strict.sh taylor2
LIMIT_PAIRS=20 bash softmax_rtl_taylor_relu/run_eval_exp_free_strict.sh relu2
```

Check:

```text
softmax_rtl_shift_based/results/shift_strict/vctk_paper_style_results_averages.txt
softmax_rtl_taylor_relu/results/taylor2_strict/vctk_paper_style_results_averages.txt
softmax_rtl_taylor_relu/results/relu2_strict/vctk_paper_style_results_averages.txt
```

### 2. Full eval if smoke passes

```bash
cd /raid/R12K41024/09_weight_tied_mha

bash softmax_rtl_shift_based/run_eval_shift_strict.sh
bash softmax_rtl_taylor_relu/run_eval_exp_free_strict.sh taylor2
bash softmax_rtl_taylor_relu/run_eval_exp_free_strict.sh relu2
```

### 3. Short finetune only for promising modes

Default scripts use:

```text
EPOCHS=30
LR=1e-4
```

Commands:

```bash
cd /raid/R12K41024/09_weight_tied_mha

EPOCHS=30 LR=1e-4 bash softmax_rtl_shift_based/run_finetune_shift_strict.sh
EPOCHS=30 LR=1e-4 bash softmax_rtl_taylor_relu/run_finetune_exp_free_strict.sh taylor2
EPOCHS=30 LR=1e-4 bash softmax_rtl_taylor_relu/run_finetune_exp_free_strict.sh relu2
```

Suggested decision rule:

- If `shift` is close to strict baseline without finetune, prioritize it for RTL.
- If `taylor2` is worse but recoverable after short finetune, keep it as the pure exp-free candidate.
- If `relu2` drops badly at eval-time, only continue if the hardware-area target is aggressive enough to justify retraining risk.

## Baseline Reference

Existing 09 strict PTQ result:

```text
09_weight_tied_mha/results/ptq_strict/vctk_paper_style_results_averages.txt
```

Baseline averages:

```text
Average MBSTOI Enhanced: 0.8373
Average SegSNR Enhanced: 1.50
Average SegSNR Improvement: 2.33
Average ILD Error Enhanced: 3.2505 dB
Average IPD Error Enhanced: 15.68 deg
```

Use these as the first comparison target.

## Notes / Cautions

- `taylor2` and `relu2` are not exact softmax replacements; eval-time drop may be large without finetune.
- The current `taylor2` is intentionally simple and hardware-readable. It may need score centering or clamp tuning if results are unstable.
- The current `shift` mode is closest to a direct softmax replacement and should be the first hardware candidate.
- `10_stft_fixpoint_ptq/README.md` already had unrelated local modifications before this work and was not touched by this softmax RTL change.

