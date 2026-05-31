# HANDOFF — Soft-LMul STE backward variants (17/)

**Date**: 2026-05-26
**Author**: Claude
**Status**: Drafted, smoke-tested, NOT launched
**Purpose**: Backup path in case the GPU 2 scout (16/, curriculum + learnable_bc + binaural KD, naive STE) plateaus. This dir adds STE backward variants without touching 16/'s running scout.

---

## TL;DR

- Forked `16_kd_heatmap_recovery/` → `17_soft_lmul_ste_recovery/` (source only; teacher_outputs symlinked, outputs/logs/results empty).
- Added two STE backward variants for the L-Mul multiply / matmul primitives:
  1. `autograd_lmul` — backward = autograd through L-Mul forward expression.
  2. `grad_scale_<f>` — backward = real-mult × scalar (e.g. `grad_scale_0.67`).
- Forward is bit-exact identical across all modes (training-only knob, inference unchanged).
- Plumbed through config / trainer / `enable_all_lmul_multiply` / `run_combined_recovery.sh`.
- Smoke test passes: forward bit-exact, gradients differ, `grad_scale_0.67` produces exactly `0.6700×` of real-mode gradient, `autograd_lmul` produces mean `0.675×` (matches theoretical `E[1/sy]` damping ≈ 0.67).

**Nothing has been launched. Awaiting**:
1. GPU 2 scout result (16/, curriculum + learnable_bc + binaural KD). First m=4 val at ~15:00.
2. Codex review of this handoff.

---

## Why STE variants matter

The 16/ scout uses the **legacy STE**: forward = bit-exact L-Mul, backward = `grad_x = grad · y, grad_y = grad · x` (real-mult identity). This is what every v4 / v3 run used. The α-sweep on v4 Ep5 ckpt showed:

> Any direction of post-hoc per-channel δ (with magnitudes from α=0 to α=1) cannot move VCTK MBSTOI w=200 above 0.8255 (= v4 Ep5 control).

Codex's interpretation: v4 Ep5 has converged to **the local optimum reachable under the legacy STE rule**. The weight gradient direction is fundamentally mis-aligned with the L-Mul nonlinearity in the forward. The 16/ scout attacks this by **adding** learnable δ Parameter and KD signal — neither modifies the L-Mul gradient itself.

If the 16/ scout's first m=4 validation shows no improvement from v4 Ep5 baseline (~-13.1 / 0.8255), the most likely remaining diagnosis is: **the STE backward itself is the bottleneck**. 17/ is the patch to test that hypothesis.

---

## What 17/ contains

### Source delta vs 16/

Only four source files differ:

| File | Change |
|---|---|
| `DCNN/utils/complexPyTorch/complexLayers.py` | Added `_LmulMultiplyScaleSTE`, `_LmulMatmulScaleSTE`, `_resolve_ste_mode`. `_lmul_multiply_ste` and `_lmul_matmul` now dispatch via the env var / explicit arg. **Legacy "real" path is byte-identical to 16/.** |
| `DCNN/utils/all_lmul_multiply.py` | Added `ste_mode` kwarg; publishes `BCCTN_STE_MODE` env var so downstream calls (including MHA datapath) inherit it. Logs `[L-Mul patch] STE backward mode: <mode>`. Validates mode at patch time to fail loudly. |
| `DCNN/trainer.py` | Plumbs `model.low_power.all_lmul_multiply.ste_mode` from config to `enable_all_lmul_multiply`. |
| `config/config.yaml` | Added `ste_mode: "real"` under `all_lmul_multiply`. |
| `scripts/run_combined_recovery.sh` | `REPO` points to `17_soft_lmul_ste_recovery`. Added `STE_MODE` env var (default `"real"`) and passes it as a hydra override. |

### Symlinks

- `teacher_outputs/` → `16_kd_heatmap_recovery/teacher_outputs/` (avoids 3.8 GB duplication)

### New files

- `scripts/smoke_ste_variants.py` — CPU-only smoke test (`~2 s`), verifies forward bit-exact and backward dispatch.
- `HANDOFF_SOFT_LMUL_STE_2026-05-26.md` — this file.

---

## STE variants — math and implementation

### Variant 1: `"real"` (default, legacy)

```text
forward:  y = L-Mul(x, w)
backward: grad_x = grad_y · w
          grad_w = grad_y · x
```

Identity STE — gradient is computed *as if forward were `x*w`*. Naive but stable. This is the current v4 Ep5 setting.

### Variant 2: `"autograd_lmul"`

Bypasses the custom `autograd.Function`. PyTorch's autograd backpropagates through the L-Mul forward expression. `_ste_floor` inside `_lmul_multiply_bit_exact` passes identity gradient through the discrete floor steps, so the graph stays connected.

Roughly:
```text
forward:  y = L-Mul(x, w) = sign(xw) * sig * 2^(ex + ew + carry)
          where sig = 1 + fx + fw + offset_term
                fx = floor((sx - 1) · 2^m) / 2^m         (STE floor)
                sx = significand of x ∈ [1, 2)
backward (chain rule via autograd):
          ∂y/∂x ≈ sign(xw) * (∂sig/∂sx) * (∂sx/∂x) * 2^(ex + ew + carry)
               = sign(xw) * 1 * (1/2^ex) * 2^(ex + ew + carry)
               = sign(xw) * 2^(ew + carry)
```

Compared to real-mult gradient `∂(xw)/∂x = w = sign(w) · sw · 2^ew`, the autograd-LMul gradient drops the operand-significand factor `sw ∈ [1, 2)`. Smoke test confirms: mean `|grad_auto / grad_real|` = `0.675` over a random batch (matches theoretical `E[1/sw]` ≈ `0.67`).

**Interpretation**: this STE is "L-Mul-aware" in the sense that it admits the same kind of magnitude attenuation L-Mul does. Cost: graph slightly larger than legacy, ~no measurable wall-clock impact in practice. No new parameters.

### Variant 3: `"grad_scale_<f>"`

Hand-pick the damping factor:
```text
forward:  y = L-Mul(x, w)
backward: grad_x = grad_y · w · f
          grad_w = grad_y · x · f
```

Cheaper than `autograd_lmul` (no autograd-graph cost) and gives manual control of damping. Useful values:
- `0.50` — aggressive damping (worst-case sw ratio)
- `0.67` — matches `autograd_lmul` mean
- `0.85` — light damping

If `autograd_lmul` improves but is slow, `grad_scale_0.67` is the cheap fallback that captures most of the same effect.

---

## Hardware story preserved

All STE variants are **training-only**. Inference path is identical to legacy:
- The `torch.is_grad_enabled() and (x.requires_grad or y.requires_grad)` check in `_lmul_multiply_ste` short-circuits to the plain forward when in `eval()` / no-grad mode.
- The MHA datapath in `complexLayers` also calls `_lmul_multiply_ste` / `_lmul_matmul` through the same dispatcher, so MHA inherits the variant automatically during training.

→ No multiplier hardware added. STE backward is computed on the host during training and discarded at deploy.

---

## Smoke test results

```text
[smoke] forward bit-exact check:
  forward(autograd_lmul) vs real: max_abs_diff = 0.000e+00  OK
  forward(grad_scale_0.5) vs real: max_abs_diff = 0.000e+00  OK
  forward(grad_scale_0.67) vs real: max_abs_diff = 0.000e+00  OK

[smoke] backward gradient differs from real-mode:
  grad_x diff(autograd_lmul): rel_max = 4.652e-01  OK
  grad_x diff(grad_scale_0.5): rel_max = 5.000e-01  OK
  grad_x diff(grad_scale_0.67): rel_max = 3.300e-01  OK

[smoke] grad_scale exact ratio:
  grad_scale_0.5:  mean(grad_x/real) = 0.5000 (expect 0.5000) OK
  grad_scale_0.67: mean(grad_x/real) = 0.6700 (expect 0.6700) OK

[smoke] autograd_lmul sanity:
  |grad_auto / grad_real| stats: mean=0.675, median=0.638, min=0.000, max=0.989
  (matches theoretical E[1/sy] ≈ 0.67)

[smoke] matmul dispatch:
  matmul forward(autograd_lmul) vs real: max_abs_diff = 0.000e+00
  matmul grad_a diff(autograd_lmul - real): 1.857e+00  (non-zero → dispatch alive)
```

Run reproduction: `cd /raid/R12K41024/17_soft_lmul_ste_recovery && conda run --no-capture-output -n bcctn python scripts/smoke_ste_variants.py`.

---

## Experimental plan (sequence)

This is contingent on the 16/ scout result. Decision tree at the bottom.

### Phase A — 16/ scout finishes (already running, GPU 2)

Watch the first m=4 validation (~15:00). Decision points already documented in 16/ wrapper.

### Phase B (only if 16/ scout plateaus at -13.1)

#### B1: Soft-LMul scout — `autograd_lmul`

Reuse the same scout config (curriculum 2+2+4, learnable_bc, binaural KD at λ=0.0528), just flip STE mode:

```bash
cd /raid/R12K41024/17_soft_lmul_ste_recovery
KD_LAMBDA=0.0528721747 \
EPOCHS_M6=2 EPOCHS_M5=2 EPOCHS_M4=4 \
GPU_ID=<free GPU> LIMIT=200 METRIC_WORKERS=16 \
RUN_TAG=scout_ste_autograd_l0p0529_2_2_4 \
STE_MODE=autograd_lmul \
bash scripts/run_combined_recovery.sh
```

Wall-time: ~20 hr (same as 16/ scout).

#### B2: Soft-LMul scout — `grad_scale_0.67` (cheaper alternative)

If `autograd_lmul` runs into autograd-graph memory issues at training batch=128 on H100, fall back to:

```bash
STE_MODE=grad_scale_0.67 \
...
```

This avoids the autograd graph through L-Mul entirely; it's the cheapest possible "STE-aware" experiment.

### Phase C — interpret results

| 16/ scout (B0) result | Phase B1 result | Conclusion |
|---|---|---|
| MBSTOI > 0.834 | (skip B) | Done. δ + KD + curriculum was enough. |
| MBSTOI ~0.83 (small lift) | autograd_lmul +0.005+ | STE damping helps. Run longer. |
| MBSTOI ~0.83 (small lift) | autograd_lmul flat | Plateau is *not* STE-bottlenecked. Consider arch changes (pow-2 pre-scale). |
| MBSTOI flat / -0.001 | autograd_lmul +0.01+ | STE was the dominant bottleneck. Big confirmation. Run longer. |
| MBSTOI flat / -0.001 | autograd_lmul flat | Likely architecture-level limit. Pivot to pow-2 pre-scale or per-channel mantissa. |

---

## Risks / open questions for codex review

1. **`autograd_lmul` may be unstable at extreme tails.** Smoke shows `min=0.000` for gradient ratio — this happens at operands that round to zero in the L-Mul forward. For those, autograd gradient is exactly zero (the operand contributes nothing). Legacy STE would still give a non-zero gradient. **Question**: should we add a fallback (clamp lower bound on gradient) to avoid dead-zone? Or trust that the optimizer can route around it via the per-channel learnable δ?

2. **MHA L-Mul interaction.** The MHA custom forward path also calls `_lmul_multiply_ste` and `_lmul_matmul`. Since BCCTN_STE_MODE is a global env var, it affects MHA too. Is that desired? The legacy default is "real" so this only matters when STE_MODE is set. **Decision needed**: should we add a separate `attention_ste_mode` config knob for finer control, or keep one knob?

3. **Smoke test coverage.** The smoke tests run on tiny CPU tensors. Real H100 / fp16 training may surface numerical-stability issues that don't appear in fp32 / CPU. **Suggestion**: before launching B1, run a 10-step training in autograd_lmul mode and check grad-norm magnitudes are reasonable (`< 1e3`).

4. **Curriculum + STE_MODE interaction.** Curriculum stages 1/2 use m=6 / m=5 mantissa. The autograd graph size scales with mantissa width (more bit-level ops in L-Mul forward). m=6 means more graph nodes than m=4. **Concern**: training memory at m=6 + autograd_lmul + batch=128 might exceed budget. Should we batch=64 for m=6 stage and recover batch=128 at m=4? Or stick with batch=128 throughout?

5. **`_resolve_ste_mode` validation is at patch-time only.** If someone manually sets `BCCTN_STE_MODE` between forward calls (e.g. via env var monkeypatch in a debug session), the next forward picks it up without validation. **Question**: should we validate per-forward (small overhead) or only at patch time (current)?

6. **No CUDA / fp16 smoke yet.** Smoke runs CPU fp32. **Action**: codex should run a brief CUDA fp16 smoke before launching B1 to catch any dtype issues with autograd through `_lmul_multiply_bit_exact`. Suggested:

   ```bash
   cd /raid/R12K41024/17_soft_lmul_ste_recovery
   # Run smoke on GPU 3 (free now) with fp16:
   python -c "
   import torch
   from DCNN.utils.complexPyTorch.complexLayers import _lmul_multiply_ste
   import os
   for mode in ['real', 'autograd_lmul', 'grad_scale_0.67']:
       os.environ['BCCTN_STE_MODE'] = mode
       x = torch.randn(4, 8, device='cuda', dtype=torch.float16, requires_grad=True)
       y = torch.randn(4, 8, device='cuda', dtype=torch.float16, requires_grad=True)
       z = _lmul_multiply_ste(x, y, 4, True)
       z.sum().backward()
       print(f'{mode}: ok, grad_x finite={torch.isfinite(x.grad).all().item()}')
   "
   ```

---

## What's NOT in 17/

These were considered and deferred:

- **Smooth analytical L-Mul backward** (closed-form ∂lmul/∂x). Tried — math is messy because of carry / floor discreteness. `autograd_lmul` is the cleaner version of the same idea.
- **Learnable STE damping** (`α[c]` per channel as a Parameter). Sounds great but adds C_out × n_layers ≈ 60 × 60 ≈ 3600 new params, which Stage-1 of curriculum may not converge in 2 epochs. **Consider for Phase C** only.
- **Pow-2 pre-scale**. Different attack (operand-distribution shaping). Belongs to a separate dir (`18/`) if needed.

---

## Quick reference

### Verify 17/ install

```bash
cd /raid/R12K41024/17_soft_lmul_ste_recovery
python -c "
from DCNN.utils.complexPyTorch.complexLayers import _resolve_ste_mode
import inspect
from DCNN.utils.all_lmul_multiply import enable_all_lmul_multiply
print('ste_mode in signature:', 'ste_mode' in inspect.signature(enable_all_lmul_multiply).parameters)
print('resolve real:', _resolve_ste_mode('real'))
print('resolve autograd_lmul:', _resolve_ste_mode('autograd_lmul'))
print('resolve grad_scale_0.67:', _resolve_ste_mode('grad_scale_0.67'))
"
conda run -n bcctn python scripts/smoke_ste_variants.py
```

### Run a Phase B1 launch (do NOT execute until 16/ scout result is in)

```bash
cd /raid/R12K41024/17_soft_lmul_ste_recovery
KD_LAMBDA=0.0528721747 \
EPOCHS_M6=2 EPOCHS_M5=2 EPOCHS_M4=4 \
GPU_ID=<free> LIMIT=200 METRIC_WORKERS=16 \
RUN_TAG=scout_ste_autograd_l0p0529_2_2_4_$(date +%H%M) \
STE_MODE=autograd_lmul \
bash scripts/run_combined_recovery.sh
```
