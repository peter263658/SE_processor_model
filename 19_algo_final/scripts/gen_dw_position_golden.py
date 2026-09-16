#!/usr/bin/env python3
"""Per-position depthwise golden for dw bring-up (Consult B3 (ii)/(iii)).

Dumps, for enc1 dw (forward Conv2d stride-2) and dec1/dec4 dw (ConvTranspose2d
stride-2), the per-output-freq-position tap->input mapping + I/O, and VERIFIES
the mapping reproduces the deploy layer_io output from input x effective-weight.

Conv2d (enc, stride2,pad2,k5):   out[c,q] = Σ_k w[c,k]·in[c, 2q-2+k]   (zero-pad)
ConvTranspose2d (dec, stride2,pad2,k5,outpad1):
                                 out[c,q] = Σ_{(p,k): q=2p-2+k} w[c,k]·in[c,p]
(depthwise: per-channel; freq axis only; time t is independent.)

Out: hw_patterns_final/dw_position_golden/{enc1,dec1,dec4}/
"""
from __future__ import annotations
import os, sys, json
import numpy as np, torch
import torch.nn.functional as F

PKG = "/raid/R12K41024/18_w4a8_pow2_mixed_lmul_hw_handoff_20260530"
THIS = "/raid/R12K41024/19_algo_final"
for k, v in {"BCCTN_ACTIVATION_FP8_MODE": "calibrated_pow2_e4m3",
             "BCCTN_ACTIVATION_CALIB_PATH": PKG + "/ckpt/activation_calibration_p999_b64.json",
             "BCCTN_ACTIVATION_POW2_ROUND": "floor", "BCCTN_OUTLIER_DELAYED_DESCALE": "1",
             "NUMBA_CACHE_DIR": "/tmp/nb", "MPLCONFIGDIR": "/tmp/mpl",
             "BCCTN_APOT_A_MIN": "-16", "BCCTN_APOT_A_MAX": "-1", "BCCTN_APOT_B_MIN": "-16",
             "BCCTN_APOT_B_MAX": "-1", "BCCTN_OUTLIER_DESCALE_APOT_MIN": "-16",
             "BCCTN_OUTLIER_DESCALE_APOT_MAX": "-6"}.items():
    os.environ.setdefault(k, v)
sys.path.insert(0, PKG + "/software_to_hardware_18"); sys.path.insert(0, THIS + "/code")
import export_helpers as eh
LIO = THIS + "/hw_patterns_final/debug_reference/layer_io_fp32/layers"
OUT = THIS + "/hw_patterns_final/dw_position_golden"


def hx(p):
    return np.array([int(x, 16) for x in open(p).read().split()], np.uint32).view(np.float32).astype(np.float64)


def io(layer):
    d = f"{LIO}/{layer.replace(chr(46),chr(95))}__0"
    return hx(f"{d}/input_0.fp32.hex"), hx(f"{d}/output_0.fp32.hex")


def whx(path, arr):
    a = np.ascontiguousarray(arr, np.float32)
    a.tofile(path + ".bin")
    with open(path + ".hex", "w") as f:
        for w in a.reshape(-1).view(np.uint32):
            f.write(f"{int(w):08X}\n")


def fwd_map(Fin, stride=2, pad=2, k=5):   # Conv2d: out q -> in (2q-2+k)
    Fout = (Fin + 2 * pad - k) // stride + 1
    m = {q: [(2 * q - pad + kk, kk) for kk in range(k) if 0 <= 2 * q - pad + kk < Fin] for q in range(Fout)}
    return Fout, m


def tconv_map(Fin, stride=2, pad=2, k=5, outpad=1):  # ConvTranspose2d: out q <- (p,kk) with q=stride*p-pad+kk
    Fout = (Fin - 1) * stride - 2 * pad + k + outpad
    m = {q: [] for q in range(Fout)}
    for p in range(Fin):
        for kk in range(k):
            q = stride * p - pad + kk
            if 0 <= q < Fout:
                m[q].append((p, kk))
    return Fout, m


def main():
    model, *_ = eh.build_model(); model.eval()
    mods = dict(model.named_modules())
    specs = [("enc1", "encoder.model.1.0.depthwise_r", 8, 32, "conv"),
             ("dec1", "decoder.model.1.0.depthwise_r", 64, 2, "tconv"),
             ("dec4", "decoder.model.4.0.depthwise_r", 8, 16, "tconv")]
    for tag, name, C, Fin, kind in specs:
        m = mods[[n for n in mods if n.endswith(name)][0]]
        w = m.weight.detach().to(torch.float64).numpy()   # [C,1,5,1]
        wk = w.reshape(C, 5)                                # per-channel 5 taps
        bias = m.bias.detach().to(torch.float64).numpy() if m.bias is not None else np.zeros(C)  # dec dw: bias=True; enc dw: none
        xi, xo = io(name)
        T = 399
        xin = xi.reshape(C, Fin, T); xref = xo.reshape(C, -1, T)
        Fout = xref.shape[1]
        if kind == "conv":
            Fo, posmap = fwd_map(Fin)
        else:
            Fo, posmap = tconv_map(Fin)
        assert Fo == Fout, (tag, Fo, Fout)
        # reproduce from input x weight via posmap (no bias on dw)
        rep = np.zeros((C, Fout, T))
        for q, taps in posmap.items():
            for (p, kk) in taps:
                rep[:, q, :] += wk[:, kk][:, None] * xin[:, p, :]
        rep += bias[:, None, None]                          # dec dw bias (0 for enc)
        err = np.max(np.abs(rep - xref))
        d = os.path.join(OUT, tag); eh.ensure(d)
        whx(os.path.join(d, "weight_Cx5"), wk)
        whx(os.path.join(d, "bias_C"), bias)
        whx(os.path.join(d, "input_CxFinxT"), xin)
        whx(os.path.join(d, "output_CxFoutxT"), xref)
        json.dump({"layer": name, "kind": "Conv2d_fwd_stride2" if kind == "conv" else "ConvTranspose2d_stride2_outpad1",
                   "C": C, "F_in": Fin, "F_out": Fout, "kernel": 5, "stride": 2, "pad": 2,
                   "output_padding": 0 if kind == "conv" else 1,
                   "tap_map_out_q_to_(in_p,k)": {str(q): taps for q, taps in posmap.items()},
                   "formula": ("out[c,q,t]=Σ_k w[c,k]·in[c,2q-2+k,t] (zero-pad)" if kind == "conv"
                               else "out[c,q,t]=Σ_{(p,k):q=2p-2+k} w[c,k]·in[c,p,t]"),
                   "verify_max_abs_err_vs_layer_io": float(err),
                   "has_bias": bool(m.bias is not None),
                   "note": "depthwise per-channel, freq axis only, t independent; weight is the INT4 effective "
                           "(fake-quant) weight; DEC dw has per-channel bias (out += bias_C), ENC dw has none"},
                  open(os.path.join(d, "dw_position_golden.json"), "w"), indent=1)
        print(f"  {tag} ({name}): {C}ch {Fin}->{Fout} freq, {kind}; reproduce-from-tapmap max_err={err:.3e}")
        print(f"     tap_map: " + "; ".join(f"q{q}<-{[(p,k) for p,k in posmap[q]]}" for q in range(min(Fout, 5))) + (" ..." if Fout > 5 else ""))
    print(f"[gen] wrote {OUT}/{{enc1,dec1,dec4}}/  (weight, input, output hex + tap-map JSON)")


if __name__ == "__main__":
    main()
