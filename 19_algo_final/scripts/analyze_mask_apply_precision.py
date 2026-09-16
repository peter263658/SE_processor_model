#!/usr/bin/env python3
"""Mask-apply (mode C) multiplier-precision analysis (Consult 2026-06-09).

Captures the real apply_mask I/O (spec = STFT out, mask = decoder out, masked =
iSTFT in) from a deploy forward, pins the operand formats/grids, then sweeps the
multiplier precision -> iSTFT -> waveform error vs the deploy golden, to find the
minimum multiplier that holds the design tolerance.

    cd /raid/R12K41024/19_algo_final
    conda run --no-capture-output -n bcctn python scripts/analyze_mask_apply_precision.py
"""
from __future__ import annotations
import os, sys, json
import numpy as np
import torch

PKG = "/raid/R12K41024/18_w4a8_pow2_mixed_lmul_hw_handoff_20260530"
S2H = os.path.join(PKG, "software_to_hardware_18")
THIS = "/raid/R12K41024/19_algo_final"
CODE = os.path.join(THIS, "code")
OUT = os.path.join(THIS, "hw_patterns_final", "mask_apply_golden")
for k, v in {
    "BCCTN_APOT_A_MIN": "-16", "BCCTN_APOT_A_MAX": "-1", "BCCTN_APOT_B_MIN": "-16", "BCCTN_APOT_B_MAX": "-1",
    "BCCTN_ACTIVATION_FP8_MODE": "calibrated_pow2_e4m3",
    "BCCTN_ACTIVATION_CALIB_PATH": os.path.join(PKG, "ckpt", "activation_calibration_p999_b64.json"),
    "BCCTN_ACTIVATION_POW2_ROUND": "floor", "BCCTN_OUTLIER_DELAYED_DESCALE": "1",
    "BCCTN_OUTLIER_DESCALE_APOT_MIN": "-16", "BCCTN_OUTLIER_DESCALE_APOT_MAX": "-6",
    "NUMBA_CACHE_DIR": "/tmp/numba_cache", "MPLCONFIGDIR": "/tmp/mpl_cache",
}.items():
    os.environ.setdefault(k, v)
sys.path.insert(0, CODE); sys.path.insert(0, S2H)
import export_helpers as eh                                # noqa
import DCNN.models.binaural_attention_model as bam         # noqa

CAP = []
_orig_apply = bam.apply_mask


def cap_apply(x, specs, mode):
    out = _orig_apply(x, specs, mode)
    CAP.append({"mask": x.detach().clone(), "spec": specs.detach().clone(), "masked": out.detach().clone()})
    return out


def grid_frac_bits(v):
    """smallest f such that v*2^f is (near) integer for all v (fixed-point Q?.f)."""
    v = v[np.isfinite(v) & (v != 0)]
    for f in range(0, 24):
        s = v * (2.0 ** f)
        if np.max(np.abs(s - np.round(s))) < 1e-6:
            return f
    return None


def q_round(v, frac_bits, int_bits=None):
    step = 2.0 ** (-frac_bits)
    q = np.round(v / step) * step
    if int_bits is not None:
        lim = 2.0 ** (int_bits - 1)
        q = np.clip(q, -lim, lim - step)
    return q


def to_e4m3(v):
    return torch.tensor(v, dtype=torch.float64).to(torch.float8_e4m3fn).to(torch.float64).numpy()


def to_fp16(v):
    return torch.tensor(v, dtype=torch.float64).to(torch.float16).to(torch.float64).numpy()


def main():
    eh.ensure(OUT)
    model, *_ = eh.build_model(); model.eval()
    bam.apply_mask = cap_apply
    import soundfile as sf
    a, _ = sf.read("/raid/R12K41024/BCCTN/Dataset/noisy_testset/snr_0dB/p225_p225_009_az-40_snr+0.0.wav", dtype="float32", always_2d=True)
    a = a[:, :2] if a.shape[1] >= 2 else np.repeat(a, 2, 1)
    a = a[:32000] if a.shape[0] >= 32000 else np.concatenate([a, np.zeros((32000 - a.shape[0], 2), np.float32)])
    x = torch.from_numpy(a.T).unsqueeze(0).contiguous()
    with torch.no_grad():
        model(x)
    istft = model.istft

    c = CAP[0]  # L
    sr = c["spec"].real[0].to(torch.float64).numpy(); si = c["spec"].imag[0].to(torch.float64).numpy()   # [64,399]
    mr = c["mask"].real[0].to(torch.float64).numpy(); mi = c["mask"].imag[0].to(torch.float64).numpy()
    Gr = c["masked"].real[0].to(torch.float64).numpy(); Gi = c["masked"].imag[0].to(torch.float64).numpy()  # [65,399]

    print(f"spec  re[{sr.min():.4f},{sr.max():.4f}] im[{si.min():.4f},{si.max():.4f}]  frac_bits={grid_frac_bits(np.r_[sr.ravel(),si.ravel()])}")
    print(f"mask  re[{mr.min():.4f},{mr.max():.4f}] im[{mi.min():.4f},{mi.max():.4f}]  frac_bits={grid_frac_bits(np.r_[mr.ravel(),mi.ravel()])}")
    mall = np.r_[mr.ravel(), mi.ravel()]
    print(f"mask grid: on e4m3={np.mean(to_e4m3(mall)==mall):.3f}  on fp16={np.mean(to_fp16(mall)==mall):.3f}")
    sall = np.r_[sr.ravel(), si.ravel()]
    print(f"spec grid: on fp16={np.mean(to_fp16(sall)==sall):.3f}  Q1.15={np.mean(q_round(sall,15)==sall):.3f}  Q2.14={np.mean(q_round(sall,14)==sall):.3f}  Q.16={np.mean(q_round(sall,16)==sall):.3f}")

    # reference alignment (fp64)
    rr = sr * mr - si * mi; ii = sr * mi + si * mr
    print(f"\n[align] fp64 modeC(spec,mask) vs captured masked[1:65]: max_abs re={np.max(np.abs(rr-Gr[1:65])):.2e} im={np.max(np.abs(ii-Gi[1:65])):.2e}  DC(row0)=0:{(Gr[0]==0).all() and (Gi[0]==0).all()}")

    # golden output waveform (deploy)
    y_ref = istft(c["masked"]).detach().to(torch.float64).numpy().reshape(-1)

    def run_cfg(name, qs, qm, prod_round=None):
        """qs/qm: (frac_bits,int_bits) fixed or 'fp16'/'e4m3'/None(fp32); prod_round: frac_bits to round products/sum."""
        def Q(v, spec):
            if spec is None: return v
            if spec == "fp16": return to_fp16(v)
            if spec == "e4m3": return to_e4m3(v)
            return q_round(v, spec[0], spec[1])
        SR, SI, MR, MI = Q(sr, qs), Q(si, qs), Q(mr, qm), Q(mi, qm)
        def mul(a, b):
            p = a * b
            return q_round(p, prod_round) if prod_round is not None else p
        R = mul(SR, MR) - mul(SI, MI); I = mul(SR, MI) + mul(SI, MR)
        if prod_round is not None:
            R = q_round(R, prod_round); I = q_round(I, prod_round)
        Rp = np.vstack([np.zeros((1, 399)), R]); Ip = np.vstack([np.zeros((1, 399)), I])  # DC pad
        masked = torch.complex(torch.tensor(Rp).unsqueeze(0), torch.tensor(Ip).unsqueeze(0))
        y = istft(masked).detach().to(torch.float64).numpy().reshape(-1)
        spec_err = max(np.max(np.abs(R - Gr[1:65])), np.max(np.abs(I - Gi[1:65])))
        wav_max = np.max(np.abs(y - y_ref)); wav_rms = np.sqrt(np.mean((y - y_ref) ** 2)) / (np.sqrt(np.mean(y_ref ** 2)) + 1e-30)
        # Q1.15 LSB of waveform = 2^-15
        print(f"  {name:42s} spec_err={spec_err:.3e}  wav_max={wav_max:.3e} ({wav_max/2**-15:.2f} Q15-LSB)  wav_relRMS={wav_rms:.2e}")
        return dict(name=name, spec_err=spec_err, wav_max=wav_max, wav_relrms=wav_rms)

    print("\n[sweep] multiplier precision -> iSTFT -> waveform vs deploy golden (spec kept fp32 unless noted):")
    res = []
    res.append(run_cfg("deploy fp32 (baseline)", None, None))
    res.append(run_cfg("mask e4m3 x spec fp32, prod fp32", None, "e4m3"))
    res.append(run_cfg("mask fp16 x spec fp16, prod fp16-ish(Q.15)", (15, 2), "fp16", prod_round=15))
    res.append(run_cfg("mask Q1.7  x spec Q1.15 -> Q1.15", (15, 2), (7, 2), prod_round=15))
    res.append(run_cfg("mask Q1.9  x spec Q1.15 -> Q1.15", (15, 2), (9, 2), prod_round=15))
    res.append(run_cfg("mask Q1.11 x spec Q1.15 -> Q1.15", (15, 2), (11, 2), prod_round=15))
    res.append(run_cfg("mask Q1.13 x spec Q1.15 -> Q1.15", (15, 2), (13, 2), prod_round=15))
    res.append(run_cfg("mask Q1.15 x spec Q1.15 -> Q1.15", (15, 2), (15, 2), prod_round=15))
    res.append(run_cfg("mask Q1.11 x spec Q1.15 -> Q1.18(wide)", (15, 2), (11, 2), prod_round=18))

    # worked per-bin golden (frame with largest |masked|, a few bins)
    fr = int(np.argmax(np.max(np.abs(Gr[1:65]), axis=0)))
    rows = []
    for binc in range(8):
        rows.append({"bin": binc + 1, "frame": fr,
                     "spec_re": float(sr[binc, fr]), "spec_im": float(si[binc, fr]),
                     "mask_re": float(mr[binc, fr]), "mask_im": float(mi[binc, fr]),
                     "masked_re": float(Gr[binc + 1, fr]), "masked_im": float(Gi[binc + 1, fr])})
    json.dump({"op": "modeC: masked_re=sr*mr-si*mi, masked_im=sr*mi+si*mr; DC bin0=0 then iSTFT",
               "spec_frac_bits": grid_frac_bits(sall), "mask_on_e4m3": float(np.mean(to_e4m3(mall) == mall)),
               "sweep": res, "worked_bins_frame": fr, "worked": rows}, open(os.path.join(OUT, "mask_apply_golden.json"), "w"), indent=1)
    print(f"\n[gen] wrote {OUT}/mask_apply_golden.json  (worked frame={fr}, bins 1..8)")


if __name__ == "__main__":
    main()
