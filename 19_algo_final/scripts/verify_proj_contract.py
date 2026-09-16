#!/usr/bin/env python3
"""Pin down the EXACT mattn.input_proj / output_proj computation contract.

Resolves the hardware engineer's "R^2 ~= 0 for every layout guess" by reproducing
the shipped fp32 golden output of mattn.input_proj / output_proj from the deploy
model's effective complex weight, and showing which index/combine convention is
the correct one (and that the wrong ones give R^2 ~= 0).

    cd /raid/R12K41024/19_algo_final
    conda run --no-capture-output -n bcctn python scripts/verify_proj_contract.py
"""
from __future__ import annotations
import os, sys
import numpy as np
import torch

PKG = "/raid/R12K41024/18_w4a8_pow2_mixed_lmul_hw_handoff_20260530"
S2H = os.path.join(PKG, "software_to_hardware_18")
THIS = "/raid/R12K41024/19_algo_final"
CODE = os.path.join(THIS, "code")
LIO = os.path.join(THIS, "hw_patterns_final", "debug_reference", "layer_io_fp32", "layers")

os.environ.setdefault("BCCTN_ACTIVATION_FP8_MODE", "calibrated_pow2_e4m3")
os.environ.setdefault("BCCTN_ACTIVATION_CALIB_PATH",
                      os.path.join(PKG, "ckpt", "activation_calibration_p999_b64.json"))
os.environ.setdefault("BCCTN_ACTIVATION_POW2_ROUND", "floor")
os.environ.setdefault("BCCTN_OUTLIER_DELAYED_DESCALE", "1")
os.environ.setdefault("BCCTN_OUTLIER_DESCALE_APOT_MIN", "-16")
os.environ.setdefault("BCCTN_OUTLIER_DESCALE_APOT_MAX", "-6")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_cache")
sys.path.insert(0, S2H); sys.path.insert(0, CODE)
import export_helpers as eh  # noqa


def hexf32(path):
    w = np.array([int(l, 16) for l in open(path).read().split()], dtype=np.uint32)
    return w.view(np.float32)


def load_cio(call):  # complex layer io  -> (in_complex[T,256], out_complex[T,256])
    d = os.path.join(LIO, call)
    xi = hexf32(os.path.join(d, "input_0.real.fp32.hex")) + 1j * hexf32(os.path.join(d, "input_0.imag.fp32.hex"))
    yo = hexf32(os.path.join(d, "output_0.real.fp32.hex")) + 1j * hexf32(os.path.join(d, "output_0.imag.fp32.hex"))
    return xi.reshape(-1, 256), yo.reshape(-1, 256)


def r2(a, b):
    a = a.reshape(-1); b = b.reshape(-1)
    ss_res = np.sum(np.abs(a - b) ** 2)
    ss_tot = np.sum(np.abs(b - b.mean()) ** 2)
    return 1.0 - ss_res / (ss_tot + 1e-30)


def cos(a, b):
    a = a.reshape(-1); b = b.reshape(-1)
    return float(np.real(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def main():
    model, *_ = eh.build_model()
    model.eval()
    # locate the GroupedLinear projection modules
    mods = dict(model.named_modules())
    name_in = [n for n in mods if n.endswith("mattn.input_proj")][0]
    name_out = [n for n in mods if n.endswith("mattn.output_proj")][0]
    print("input_proj module :", name_in, type(mods[name_in]).__name__)
    print("output_proj module:", name_out, type(mods[name_out]).__name__)

    for tag, mname, call in [("input_proj", name_in, "mattn_input_proj__0"),
                             ("output_proj", name_out, "mattn_output_proj__0")]:
        m = mods[mname]
        W = m.weight.detach()                  # effective (INT4 fake-quant) complex weight
        b = m.bias.detach() if m.bias is not None else None
        Wr = W.real.to(torch.float64).numpy(); Wi = W.imag.to(torch.float64).numpy()
        print(f"\n===== {tag} =====  W.shape={tuple(W.shape)} complex  "
              f"(out_features={W.shape[0]}, in_features={W.shape[1]})  bias={None if b is None else tuple(b.shape)}")
        xin, yout = load_cio(call)             # fp32 golden I/O (PRE-act-quant input)
        xr = xin.real; xi = xin.imag
        br = (b.real.to(torch.float64).numpy() if b is not None else 0.0)
        bi = (b.imag.to(torch.float64).numpy() if b is not None else 0.0)

        # ---- CORRECT contract:  out = x @ W^T  (F.linear), complex ----
        #   out_re = x_r @ W_r^T - x_i @ W_i^T
        #   out_im = x_r @ W_i^T + x_i @ W_r^T
        out_re = xr @ Wr.T - xi @ Wi.T + br
        out_im = xr @ Wi.T + xi @ Wr.T + bi
        out = out_re + 1j * out_im
        print(f"  [CORRECT] out=x@W^T (W[oc,ic]), re=xr·Wr-xi·Wi, im=xr·Wi+xi·Wr")
        print(f"            R2={r2(out, yout):.6f}  cos={cos(out, yout):.6f}  "
              f"maxabs={np.max(np.abs(out - yout)):.4e}")

        # ---- WRONG variants (what gives R^2~=0) ----
        wrong = {
            "x@W (no transpose, W[ic,oc])": (xr @ Wr - xi @ Wi) + 1j * (xr @ Wi + xi @ Wr),
            "conj combine re=xr·Wr+xi·Wi": (xr @ Wr.T + xi @ Wi.T) + 1j * (xr @ Wi.T - xi @ Wr.T),
            "real-only (drop imag W)": (xr @ Wr.T) + 1j * (xi @ Wr.T),
        }
        for label, o in wrong.items():
            o = o + (br + 1j * bi)
            print(f"  [wrong] {label:38s} R2={r2(o, yout):+.4f} cos={cos(o, yout):+.4f}")

        # ---- per-head scale structure ----
        # head = oc // (out_features // 16); 16 heads, head_dim rows each
        oc = W.shape[0]; nh = 16; rpr = oc // nh
        print(f"  per-head scale: {nh} heads, {rpr} output-rows/head -> head(oc)=oc//{rpr}; "
              f"fc_r and fc_i carry independent per-head APoT scales (shape [16,1,1]).")


if __name__ == "__main__":
    main()
