# Shift-Based RTL Softmax Simulation

This folder holds launch scripts for the exp-free-ish shift-based attention
probability path in `09_weight_tied_mha`.

Mode:

```text
model.attention_softmax.mode=shift
```

Optional fused mode:

```text
model.attention_softmax.mode=shift_topk_pow2
```

The implementation is in `../DCNN/utils/complexPyTorch/complexLayers.py`:

- subtract row max after the causal/sliding-window mask
- approximate `exp(x)` as `2^(1.5x)`
- approximate the fractional `2^f` term as `1 + f`
- round the denominator to a power-of-two reciprocal

`shift_topk_pow2` moves the existing strict `top_k_attention=4` selection before
the shift weights are computed. It only computes four weights per row and uses a
power-of-two denominator over those four weights, so the final normalize stage is
also a shift. For equivalence/debug, `shift_topk_exact` uses the same fused
top-k support but an exact top-k denominator.

The scripts preserve the strict 09 setup:

```text
training.qat_scope=full
training.top_k_attention=4
model.tied_mha=true
```

Examples:

```bash
# Original validated shift mode.
bash softmax_rtl_shift_based/run_eval_shift_strict.sh

# Fused top-k-first variant.
MODE=shift_topk_pow2 bash softmax_rtl_shift_based/run_eval_shift_strict.sh
```
