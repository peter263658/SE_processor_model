"""Smoke test for the L-Mul STE backward variants.

Asserts:
  1. Forward output is bit-exact identical across STE modes (mode = training-only).
  2. Backward gradients differ between "real" and the variants (proves dispatch alive).
  3. "grad_scale_<f>" gradient = real-mode gradient × f, exact to numerical noise.
  4. "autograd_lmul" gradient ≠ real-mode gradient AND has finite magnitude (no NaN/Inf).

Run on CPU, no GPU required. Takes ~2s.

Usage:
  cd /raid/R12K41024/17_soft_lmul_ste_recovery
  python scripts/smoke_ste_variants.py
"""
from __future__ import annotations

import os
import sys

import torch

# Make repo imports work without conda env activation
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from DCNN.utils.complexPyTorch.complexLayers import (  # noqa: E402
    _lmul_multiply_ste,
    _lmul_matmul,
    _resolve_ste_mode,
)


def _mk_inputs(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(4, 8, generator=g, dtype=torch.float32, requires_grad=True)
    y = torch.randn(4, 8, generator=g, dtype=torch.float32, requires_grad=True)
    return x, y


def _grad(mode, x, y, mantissa_bits=4):
    """Run forward + backward in a given STE mode, return (out, grad_x, grad_y)."""
    os.environ["BCCTN_STE_MODE"] = mode
    try:
        x_ = x.detach().clone().requires_grad_(True)
        y_ = y.detach().clone().requires_grad_(True)
        out = _lmul_multiply_ste(x_, y_, mantissa_bits=mantissa_bits, bit_exact=True)
        out.sum().backward()
        return out.detach(), x_.grad.clone(), y_.grad.clone()
    finally:
        os.environ.pop("BCCTN_STE_MODE", None)


def main():
    print("[smoke] resolve checks:")
    for m in ("real", "autograd_lmul", "lmul_exact", "grad_scale_0.5", "grad_scale_0.67", "grad_scale_1.0"):
        assert _resolve_ste_mode(m) == m, m
        print(f"  {m:24s} OK")
    print()

    # Reject typos
    try:
        _resolve_ste_mode("foobar")
        print("  FAIL: expected ValueError for typo")
        sys.exit(1)
    except ValueError as e:
        print(f"  reject typo OK: {str(e)[:60]}")
    print()

    x, y = _mk_inputs()

    out_real, gx_real, gy_real = _grad("real", x, y)
    out_auto, gx_auto, gy_auto = _grad("autograd_lmul", x, y)
    out_exact, gx_exact, gy_exact = _grad("lmul_exact", x, y)
    out_s050, gx_s050, gy_s050 = _grad("grad_scale_0.5", x, y)
    out_s067, gx_s067, gy_s067 = _grad("grad_scale_0.67", x, y)

    # 1. Forward bit-exact across modes
    print("[smoke] forward bit-exact check:")
    for tag, out in [("autograd_lmul", out_auto), ("lmul_exact", out_exact),
                     ("grad_scale_0.5", out_s050), ("grad_scale_0.67", out_s067)]:
        diff = (out - out_real).abs().max().item()
        ok = diff == 0.0
        print(f"  forward({tag}) vs real: max_abs_diff = {diff:.3e}  {'OK' if ok else 'FAIL'}")
        assert ok, f"forward not bit-exact for {tag}"
    print()

    # 2. Backward gradient differs between modes
    print("[smoke] backward gradient differs from real-mode:")
    for tag, gx in [("autograd_lmul", gx_auto), ("grad_scale_0.5", gx_s050),
                    ("grad_scale_0.67", gx_s067)]:
        rel = (gx - gx_real).abs().max().item() / max(gx_real.abs().max().item(), 1e-12)
        ok = rel > 1e-3
        print(f"  grad_x diff({tag}): rel_max = {rel:.3e}  {'OK' if ok else 'FAIL'}")
        assert ok, f"backward should differ from real for {tag}"
    print()

    # 3. grad_scale exact ratio
    print("[smoke] grad_scale exact ratio:")
    for tag, gx, gy, expected in [
        ("0.5", gx_s050, gy_s050, 0.5),
        ("0.67", gx_s067, gy_s067, 0.67),
    ]:
        ratio_x = (gx / gx_real.clamp(min=1e-12).sign() / gx_real.abs().clamp(min=1e-12)) \
                  .sign().detach()  # not numerically robust for zeros — use elementwise
        # Use elementwise direct ratio with masked zeros
        mask = gx_real.abs() > 1e-8
        r_x = (gx[mask] / gx_real[mask])
        r_y = (gy[mask] / gy_real[mask])
        ok_x = torch.allclose(r_x, torch.full_like(r_x, expected), atol=1e-5)
        ok_y = torch.allclose(r_y, torch.full_like(r_y, expected), atol=1e-5)
        print(f"  grad_scale_{tag}: mean(grad_x/real)={r_x.mean().item():.4f} "
              f"(expect {expected:.4f}) {'OK' if ok_x else 'FAIL'}")
        print(f"  grad_scale_{tag}: mean(grad_y/real)={r_y.mean().item():.4f} "
              f"(expect {expected:.4f}) {'OK' if ok_y else 'FAIL'}")
        assert ok_x and ok_y, f"grad_scale_{tag} ratio off"
    print()

    # 4. autograd_lmul finite and not real-mode
    print("[smoke] autograd_lmul sanity:")
    print(f"  grad_x is finite: {torch.isfinite(gx_auto).all().item()}")
    print(f"  grad_y is finite: {torch.isfinite(gy_auto).all().item()}")
    assert torch.isfinite(gx_auto).all() and torch.isfinite(gy_auto).all()
    mask = gx_real.abs() > 1e-8
    auto_ratio = (gx_auto[mask] / gx_real[mask]).abs()
    print(f"  |grad_auto / grad_real| stats: "
          f"mean={auto_ratio.mean().item():.3f}, "
          f"median={auto_ratio.median().item():.3f}, "
          f"min={auto_ratio.min().item():.3f}, "
          f"max={auto_ratio.max().item():.3f}")
    print("  (expect mean roughly 0.5–1.0 — autograd-through-LMul damps by ~sy factor)")
    print()

    # 4b. lmul_exact: analytic pow2floor gradient. grad_exact = grad·pow2floor(other),
    #     so the per-element damping grad_exact/grad_real = pow2floor(y)/y = 1/(1+m_y)
    #     MUST lie in (0.5, 1] for every element, with mean ≈ ln2 ≈ 0.693.
    print("[smoke] lmul_exact analytic-gradient check:")
    mxr = gx_real.abs() > 1e-8
    myr = gy_real.abs() > 1e-8
    er_x = (gx_exact[mxr] / gx_real[mxr])
    er_y = (gy_exact[myr] / gy_real[myr])
    print(f"  grad_x/real: mean={er_x.mean().item():.4f} (expect ln2≈0.693), "
          f"min={er_x.min().item():.4f}, max={er_x.max().item():.4f}")
    print(f"  grad_y/real: mean={er_y.mean().item():.4f}, "
          f"min={er_y.min().item():.4f}, max={er_y.max().item():.4f}")
    assert er_x.min().item() > 0.499 and er_x.max().item() <= 1.001, \
        "lmul_exact grad_x ratio must lie in (0.5, 1]"
    assert er_y.min().item() > 0.499 and er_y.max().item() <= 1.001, \
        "lmul_exact grad_y ratio must lie in (0.5, 1]"
    assert 0.60 < er_x.mean().item() < 0.78, "mean damping should be near ln2≈0.69"
    corr = torch.corrcoef(torch.stack([gx_exact.flatten(), gx_auto.flatten()]))[0, 1].item()
    print(f"  ratio ∈ (0.5,1] OK, mean ≈ ln2 OK; vs autograd_lmul corr={corr:.4f} "
          f"(exact mean={er_x.mean().item():.3f} / auto mean={auto_ratio.mean().item():.3f})")
    print("  (the two differ only on the clamp-saturated top-mantissa-bin minority,")
    print("   where autograd zeroes the grad but pow2floor does not — lmul_exact is cleaner)")
    assert corr > 0.80, f"lmul_exact should correlate strongly with autograd_lmul (corr={corr})"
    print()

    # 5. Matmul dispatch parity
    print("[smoke] matmul dispatch:")
    g = torch.Generator().manual_seed(1)
    a = torch.randn(2, 3, 4, generator=g, requires_grad=True)
    b = torch.randn(2, 4, 5, generator=g, requires_grad=True)
    outs = {}
    for mode in ("real", "autograd_lmul", "lmul_exact", "grad_scale_0.67"):
        os.environ["BCCTN_STE_MODE"] = mode
        try:
            a_ = a.detach().clone().requires_grad_(True)
            b_ = b.detach().clone().requires_grad_(True)
            o = _lmul_matmul(a_, b_, mantissa_bits=4, bit_exact=True)
            o.sum().backward()
            outs[mode] = (o.detach(), a_.grad.clone(), b_.grad.clone())
        finally:
            os.environ.pop("BCCTN_STE_MODE", None)
    fwd_diff_auto = (outs["autograd_lmul"][0] - outs["real"][0]).abs().max().item()
    fwd_diff_s067 = (outs["grad_scale_0.67"][0] - outs["real"][0]).abs().max().item()
    print(f"  matmul forward(autograd_lmul) vs real: max_abs_diff = {fwd_diff_auto:.3e}")
    print(f"  matmul forward(grad_scale_0.67) vs real: max_abs_diff = {fwd_diff_s067:.3e}")
    assert fwd_diff_auto == 0.0 and fwd_diff_s067 == 0.0
    mm_grad_diff_auto = (outs["autograd_lmul"][1] - outs["real"][1]).abs().max().item()
    mm_grad_diff_s067 = (outs["grad_scale_0.67"][1] - outs["real"][1]).abs().max().item()
    print(f"  matmul grad_a diff(autograd_lmul - real): {mm_grad_diff_auto:.3e}")
    print(f"  matmul grad_a diff(grad_scale_0.67 - real): {mm_grad_diff_s067:.3e}")
    assert mm_grad_diff_auto > 0 and mm_grad_diff_s067 > 0
    fwd_diff_exact = (outs["lmul_exact"][0] - outs["real"][0]).abs().max().item()
    mm_grad_diff_exact = (outs["lmul_exact"][1] - outs["real"][1]).abs().max().item()
    mm_corr = torch.corrcoef(torch.stack([
        outs["lmul_exact"][1].flatten(), outs["autograd_lmul"][1].flatten()]))[0, 1].item()
    print(f"  matmul forward(lmul_exact) vs real: max_abs_diff = {fwd_diff_exact:.3e}")
    print(f"  matmul grad_a diff(lmul_exact - real): {mm_grad_diff_exact:.3e}  "
          f"corr(lmul_exact, autograd_lmul) = {mm_corr:.4f}")
    assert fwd_diff_exact == 0.0 and mm_grad_diff_exact > 0
    assert mm_corr > 0.80, f"matmul lmul_exact should correlate with autograd_lmul ({mm_corr})"
    print()

    print("[smoke] ALL OK")


if __name__ == "__main__":
    main()
