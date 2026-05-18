# 09 — Weight-Tied ComplexMHA (N1 from HANDOFF_2026-05-09)

This folder tests whether tying `MHA_re ≡ MHA_im` in the 4-pass complex multihead
attention preserves binaural-SE quality (MBSTOI / ILD / IPD) under W4A8 mixed-precision
QAT, while halving the MHA parameter count.

## Architecture change (vs. 08)

| | 08 baseline | 09 tied |
|---|---|---|
| `MHA_re` weights | full | full |
| `MHA_im` weights | independent | **shared with `MHA_re`** |
| MHA params | 526,336 | **263,168** (50% saving) |
| Real-MHA passes / inference | 4 | **2** |
| Softmax serializations / inference | 4 | **2** |

### Math

```
ComplexMHA(x) = (1+i) · [M(x.real) + i · M(x.imag)]
              = (a − b) + i (a + b)   where  a = M(x.real),  b = M(x.imag)
```

Implementation: [`DCNN/utils/complexPyTorch/complexLayers.py`](DCNN/utils/complexPyTorch/complexLayers.py)
`ComplexMultiheadAttention(..., tied=True)` skips creating `MultiheadAttention_im`
and uses `_forward_tied` which computes the `(a−b, a+b)` pair.

## Files

| File | Purpose |
|---|---|
| `config/config.yaml`                    | Inherits 08 config; adds `model.tied_mha: true`. |
| `checkpoint_surgery_for_tied.py`        | Convert untied baseline ckpt → tied init by averaging `MHA_re`/`MHA_im`. |
| `baseline_tied_init.ckpt`               | Output of surgery; init for fine-tune. |
| `run_train_finetune.sh`                 | Fine-tune from `baseline_tied_init.ckpt` on **GPU 0**. lr=5e-4, 80 epochs. |
| `run_train_scratch.sh`                  | Train from random init on **GPU 2**. lr=1e-3, 150 epochs. |
| `run_eval.sh`                           | Eval one checkpoint with paper-style metrics. |
| `run_eval_compare.sh`                   | Auto-discover both 09 ckpts, eval each, merge with 08 baseline + PTQ runs. |
| `merge_eval_comparison.py`              | CSV merger producing the side-by-side table. |

## Run

```bash
# 1. (already done) Prep tied init checkpoint
python checkpoint_surgery_for_tied.py \
    --src epoch=134-step=130680.ckpt \
    --dst baseline_tied_init.ckpt

# 2. Launch both trainings (different GPUs, parallel)
bash run_train_finetune.sh    # GPU 0
bash run_train_scratch.sh     # GPU 2

# 3. After training, eval + merge into one comparison table
bash run_eval_compare.sh
```

## Compute & memory savings (MHA only)

| Metric | Untied | Tied | Δ |
|---|---:|---:|---:|
| MHA parameters                 | 526,336 | 263,168 | −50% |
| MHA fp8 weight bytes           | 514 KB  | 257 KB  | −257 KB |
| Real-MHA passes / forward      | 4       | 2       | −50% |
| Softmax stalls (per ASIC spec) | 4       | 2       | −50% |
| `mattn` block compute (per step) | ~1.5 M ops | ~0.95 M ops | −37% |

Encoder / decoder are unchanged; the saving above is local to the MHA block.

## Open question — what this experiment answers

Tying restricts the function class (loses one degree of freedom for `MHA_im` to
diverge from `MHA_re`). The empirical question is whether MBSTOI / ILD / IPD
under matched (VCTK) and reverberant (TIMIT) test conditions stay within
acceptable tolerance. The comparison table from `run_eval_compare.sh` gives the
direct answer against the 08 baseline and its PTQ variants.
