"""GPU smoke for L-Mul STE backward variants — 10 training-shape steps × 3 modes.

Goal: catch H100-specific failures that the CPU smoke can miss, before
committing 20 hr of curriculum training in 17/.

What this checks per STE mode:
  - no NaN/Inf in forward / gradients
  - gradient L2-norm stays below 1e3 (sanity bound)
  - peak GPU memory stays bounded (extrapolatable to b=128 m=6)
  - wall-time per step (for comparing autograd_lmul vs real overhead)

Modes tested:  real (baseline)  |  autograd_lmul  |  grad_scale_0.67

The mock model mirrors the BCCTN multiply surface:
  Conv2d + depthwise Conv2d + pointwise Conv2d + ConvTranspose2d + Linear,
  patched in-place via enable_all_lmul_multiply with learnable_bc=True so
  the smoke also covers the δ-parameter optimisation path.

Usage:
  cd /raid/R12K41024/18_w4a8_pow2_mixed_lmul
  CUDA_VISIBLE_DEVICES=<free GPU> conda run --no-capture-output -n bcctn \
      python scripts/smoke_ste_gpu.py

The script runs in ~30 s on a free H100. Returns exit code 1 on any failure.
"""
from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from DCNN.utils.all_lmul_multiply import enable_all_lmul_multiply  # noqa: E402


def make_mock_model(in_ch: int = 2, mid_ch: int = 32) -> nn.Module:
    """Conv + depthwise + pointwise + ConvTranspose + Linear stack."""
    return nn.Sequential(
        nn.Conv2d(in_ch, mid_ch, 5, padding=2, bias=True),
        nn.ReLU(),
        nn.Conv2d(mid_ch, mid_ch, 5, padding=2, groups=mid_ch, bias=True),  # depthwise
        nn.Conv2d(mid_ch, mid_ch, 1, bias=True),                            # pointwise
        nn.ReLU(),
        nn.ConvTranspose2d(mid_ch, mid_ch, 4, stride=2, padding=1, bias=True),
        nn.Conv2d(mid_ch, mid_ch, 1, bias=True),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(mid_ch, mid_ch, bias=True),
        nn.Linear(mid_ch, 1, bias=True),
    )


@dataclass
class StepStats:
    mode: str
    mantissa_bits: int
    batch: int
    n_steps: int
    grad_norms: List[float] = field(default_factory=list)
    step_times_s: List[float] = field(default_factory=list)
    forward_max_abs: List[float] = field(default_factory=list)
    peak_mem_mb: float = 0.0
    nan_count: int = 0
    inf_count: int = 0


def run_steps(mode: str, *, mantissa_bits: int = 6, batch: int = 32,
              n_steps: int = 10, device: str = "cuda",
              spatial: tuple = (64, 32)) -> StepStats:
    """Build a fresh patched model, run n_steps fwd+bwd in given STE mode."""
    # IMPORTANT: set BCCTN_STE_MODE before constructing the patched model so
    # the env-var-publishing logic inside enable_all_lmul_multiply respects
    # the explicit kwarg.
    os.environ["BCCTN_STE_MODE"] = mode
    try:
        model = make_mock_model().to(device, dtype=torch.float32)
        model = enable_all_lmul_multiply(
            model,
            mantissa_bits=mantissa_bits,
            bit_exact=True,
            ste_mode=mode,
            learnable_bias_correction=True,  # exercise δ optimisation path too
            learnable_bc_init=0.0,
        )
        opt = torch.optim.Adam(model.parameters(), lr=1e-5)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        stats = StepStats(mode=mode, mantissa_bits=mantissa_bits,
                          batch=batch, n_steps=n_steps)
        H, W = spatial
        for _ in range(n_steps):
            x = torch.randn(batch, 2, H, W, device=device, dtype=torch.float32)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            y = model(x)
            loss = (y ** 2).mean()
            if not torch.isfinite(loss).item():
                stats.nan_count += 1
                continue
            loss.backward()
            grads = [p.grad.detach() for p in model.parameters() if p.grad is not None]
            gn = float(torch.norm(torch.stack([g.norm() for g in grads])))
            if not math.isfinite(gn):
                stats.inf_count += 1
            stats.grad_norms.append(gn)
            stats.forward_max_abs.append(float(y.detach().abs().max()))
            opt.step()
            torch.cuda.synchronize()
            stats.step_times_s.append(time.perf_counter() - t0)
        stats.peak_mem_mb = torch.cuda.max_memory_allocated(device) / 1e6
        return stats
    finally:
        os.environ.pop("BCCTN_STE_MODE", None)


def fmt(s: StepStats) -> str:
    g, t = s.grad_norms, s.step_times_s
    if not g:
        return f"mode={s.mode!s:18s} m={s.mantissa_bits} ALL FAILED (no finite steps)"
    return (f"mode={s.mode!s:18s} m={s.mantissa_bits} b={s.batch} "
            f"peak_mem={s.peak_mem_mb:7.1f}MB  "
            f"gn[mean={sum(g)/len(g):.2e}, max={max(g):.2e}]  "
            f"step_t[mean={sum(t)/len(t)*1000:6.1f}ms]  "
            f"fwd_max={max(s.forward_max_abs):.2e}  "
            f"nan={s.nan_count} inf={s.inf_count}")


def main():
    if not torch.cuda.is_available():
        print("FAIL: CUDA not available", file=sys.stderr)
        sys.exit(2)
    dev_name = torch.cuda.get_device_name(0)
    print(f"[gpu-smoke] device=cuda ({dev_name})")
    print(f"[gpu-smoke] config: mantissa_bits=6 (m=6 stage worst case), "
          f"batch=32, n_steps=10, spatial=64x32, fp32")
    print(f"[gpu-smoke] learnable_bc enabled (δ optimisation exercised)")
    print()

    results: List[StepStats] = []
    for mode in ("real", "autograd_lmul", "lmul_exact", "grad_scale_0.67"):
        s = run_steps(mode, mantissa_bits=6, batch=32, n_steps=10)
        results.append(s)
        print(fmt(s))
    print()

    # Verdict
    print("[gpu-smoke] verdict:")
    failed = False
    real = next((r for r in results if r.mode == "real"), None)
    for s in results:
        if not s.grad_norms:
            print(f"  {s.mode:18s} FAIL  no finite steps")
            failed = True
            continue
        max_gn = max(s.grad_norms)
        ok_no_nan = s.nan_count == 0 and s.inf_count == 0
        ok_gn = max_gn < 1e3
        # Wall-time vs real (informational, not a fail criterion)
        if real and real is not s and real.step_times_s:
            slowdown = (sum(s.step_times_s)/len(s.step_times_s)) / \
                       (sum(real.step_times_s)/len(real.step_times_s))
            slowdown_str = f"  slowdown_vs_real={slowdown:.2f}x"
        else:
            slowdown_str = ""
        tag = "OK" if (ok_no_nan and ok_gn) else "FAIL"
        if not (ok_no_nan and ok_gn):
            failed = True
        print(f"  {s.mode:18s} {tag}  "
              f"gn_max={max_gn:.2e} (limit 1e3), "
              f"nan={s.nan_count}, inf={s.inf_count}"
              f"{slowdown_str}")
    print()

    # Extrapolation note
    if real:
        print(f"[gpu-smoke] memory extrapolation to b=128:")
        for s in results:
            est_b128 = s.peak_mem_mb * (128 / s.batch)
            print(f"  {s.mode:18s} peak_mem@b32={s.peak_mem_mb:6.1f}MB  "
                  f"→ est@b128={est_b128:7.1f}MB")
        print(f"[gpu-smoke] H100 80GB headroom: should be << 60_000 MB at b=128 m=6")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
