# Taylor2 / ReLU2 RTL Attention Simulation

This folder holds launch scripts for the lower-area exp-free attention
probability paths in `09_weight_tied_mha`.

Modes:

```text
model.attention_softmax.mode=taylor2
model.attention_softmax.mode=relu2
```

The implementation is in `../DCNN/utils/complexPyTorch/complexLayers.py`:

- `taylor2`: use `1 + score + 0.5 * score^2`, clamp negative weights to zero,
  then normalize by the row sum.
- `relu2`: use `ReLU(score)^2`, then normalize by the row sum. If a row has no
  positive scores, it falls back to uniform over the valid causal/sliding window.

The scripts preserve the strict 09 setup:

```text
training.qat_scope=full
training.top_k_attention=4
model.tied_mha=true
```

