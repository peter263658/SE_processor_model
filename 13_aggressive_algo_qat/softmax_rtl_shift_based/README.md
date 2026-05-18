# Shift-Based RTL Softmax Simulation

This folder holds launch scripts for the exp-free-ish shift-based attention
probability path in `11_weight_tied_mha_fast_mbstoi`.

Mode:

```text
model.attention_softmax.mode=shift
```

Fused top-k variants:

```text
model.attention_softmax.mode=shift_topk_pow2
model.attention_softmax.mode=shift_topk_pow2_floor
model.attention_softmax.mode=shift_topk_pow2_ceil
model.attention_softmax.mode=shift_topk_pow2_clamp8
model.attention_softmax.mode=shift_topk_pow2_nomantissa
model.attention_softmax.mode=shift_topk_pow2_clamp8_nomantissa
```

The implementation is in `../DCNN/utils/complexPyTorch/complexLayers.py`:

- subtract row max after the causal/sliding-window mask
- approximate `exp(x)` as `2^(1.5x)`
- approximate the fractional `2^f` term as `1 + f`
- round the denominator to a power-of-two reciprocal

Variant details:

- `shift_topk_pow2`: top-k first, nearest power-of-two denominator.
- `shift_topk_pow2_floor`: top-k first, floor power-of-two denominator.
- `shift_topk_pow2_ceil`: top-k first, ceil power-of-two denominator.
- `shift_topk_pow2_clamp8`: top-k first, clamp exponent range to `[-8, 0]`.
- `shift_topk_pow2_nomantissa`: top-k first, use `2^k` only and drop `1+frac`.
- `shift_topk_pow2_clamp8_nomantissa`: combine clamp `[-8, 0]` and no mantissa.

Short aliases are also accepted: `floor`, `ceil`, `clamp8`, `mantissa_free`,
`clamp8_nomantissa`.

The scripts preserve the strict 09 setup:

```text
training.qat_scope=full
training.top_k_attention=4
model.tied_mha=true
```

Example:

```bash
METRIC_WORKERS=64 MODE=shift_topk_pow2_floor bash softmax_rtl_shift_based/run_eval_shift_strict.sh
METRIC_WORKERS=64 MODE=clamp8_nomantissa bash softmax_rtl_shift_based/run_eval_shift_strict.sh
```
