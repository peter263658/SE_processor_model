#!/usr/bin/env python3
"""Full fp16-mask mask-apply golden (Consult §B-6 / RELAY #3).

Captures the real apply_mask I/O, then emits the HW bit-exact target: the
fp16-mask masked spectrum, per (bin 0..63, frame 0..398), as hex codes:
  spec_re/im  : Q2.14 int16 code  (STFT bus)
  mask_re/im  : fp16  uint16 code (decoder fp16-PASS output)
  masked_re/im: Q2.14 int16 code  (fp16-mask result -> iSTFT input)

Reference arithmetic (RTL must match):
  mr,mi = fp16(mask)            ; sr,si = Q2.14(spec)
  masked_re = Q2.14_RNE( sr*mr - si*mi )   ; masked_im = Q2.14_RNE( sr*mi + si*mr )
  products + the one subtract/add are evaluated full-precision, then a SINGLE
  round-half-to-EVEN to Q2.14 (step 2^-14). DC bin 0 = 0 (F.pad) -> 65 bins to iSTFT.

    cd /raid/R12K41024/19_algo_final
    conda run --no-capture-output -n bcctn python scripts/gen_mask_apply_golden.py
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
_orig = bam.apply_mask
FRAC = 14
QMIN, QMAX = -(2 ** 15), 2 ** 15 - 1   # int16 Q2.14 code range


def cap(x, specs, mode):
    out = _orig(x, specs, mode)
    CAP.append({"mask": x.detach().clone(), "spec": specs.detach().clone()})
    return out


def q214_code(v):
    """value -> Q2.14 int16 code, round-half-to-even, saturating."""
    c = np.rint(np.asarray(v, np.float64) * (2.0 ** FRAC))   # np.rint = round-half-to-even
    return np.clip(c, QMIN, QMAX).astype(np.int16)


def q214_val(code):
    return code.astype(np.float64) / (2.0 ** FRAC)


def fp16_code(v):
    return torch.tensor(np.asarray(v, np.float64), dtype=torch.float64).to(torch.float16).view(torch.uint16).numpy()


def fp16_val(v):
    return torch.tensor(np.asarray(v, np.float64), dtype=torch.float64).to(torch.float16).to(torch.float64).numpy()


def write_hex(path, arr_u16):
    with open(path, "w") as f:
        for w in np.asarray(arr_u16).reshape(-1):
            f.write(f"{int(w) & 0xFFFF:04X}\n")


def main():
    eh.ensure(OUT)
    model, *_ = eh.build_model(); model.eval()
    bam.apply_mask = cap
    import soundfile as sf
    a, _ = sf.read("/raid/R12K41024/BCCTN/Dataset/noisy_testset/snr_0dB/p225_p225_009_az-40_snr+0.0.wav", dtype="float32", always_2d=True)
    a = a[:, :2] if a.shape[1] >= 2 else np.repeat(a, 2, 1)
    a = a[:32000] if a.shape[0] >= 32000 else np.concatenate([a, np.zeros((32000 - a.shape[0], 2), np.float32)])
    with torch.no_grad():
        model(torch.from_numpy(a.T).unsqueeze(0).contiguous())

    for ci, side in [(0, "L"), (1, "R")]:
        c = CAP[ci]
        sr = c["spec"].real[0].to(torch.float64).numpy(); si = c["spec"].imag[0].to(torch.float64).numpy()  # [64,399]
        mr = c["mask"].real[0].to(torch.float64).numpy(); mi = c["mask"].imag[0].to(torch.float64).numpy()
        # quantize operands
        sr_c, si_c = q214_code(sr), q214_code(si)
        mr_c, mi_c = fp16_code(mr), fp16_code(mi)
        SR, SI = q214_val(sr_c), q214_val(si_c)              # Q2.14 spec values
        MR, MI = fp16_val(mr), fp16_val(mi)                  # fp16 mask values
        # fp16-mask masked spectrum -> Q2.14 (single RNE round)
        mre_c = q214_code(SR * MR - SI * MI)
        mim_c = q214_code(SR * MI + SI * MR)
        d = os.path.join(OUT, side)
        eh.ensure(d)
        # row-major [bin 0..63, frame 0..398] (bin-major)
        for nm, arr in [("spec_re", sr_c), ("spec_im", si_c)]:
            write_hex(os.path.join(d, f"{nm}.q214.hex"), arr.view(np.uint16)); arr.tofile(os.path.join(d, f"{nm}.q214.int16.bin"))
        for nm, arr in [("mask_re", mr_c), ("mask_im", mi_c)]:
            write_hex(os.path.join(d, f"{nm}.fp16.hex"), arr); np.asarray(arr, np.uint16).tofile(os.path.join(d, f"{nm}.fp16.bin"))
        for nm, arr in [("masked_re", mre_c), ("masked_im", mim_c)]:
            write_hex(os.path.join(d, f"{nm}.q214.hex"), arr.view(np.uint16)); arr.tofile(os.path.join(d, f"{nm}.q214.int16.bin"))
        # self-check vs fp64 modeC of the (already-quantized) operands
        ref_re = SR * MR - SI * MI; ref_im = SR * MI + SI * MR
        err = max(np.max(np.abs(q214_val(mre_c) - ref_re)), np.max(np.abs(q214_val(mim_c) - ref_im)))
        print(f"[{side}] bins0..63 x 399fr  spec|max|={max(np.abs(sr).max(),np.abs(si).max()):.3f} "
              f"mask|max|={max(np.abs(mr).max(),np.abs(mi).max()):.3f}  masked q214 round-err<=0.5LSB:{err<=0.5/2**FRAC+1e-12} ({err*2**FRAC:.3f} LSB)")
        if ci == 0:
            meta = {"side": side, "layout": "[bin 0..63][frame 0..398] bin-major (64x399)",
                    "formula": "masked_re=sr*mr-si*mi, masked_im=sr*mi+si*mr ; DC bin0=0 (pad) -> 65 bins to iSTFT",
                    "spec_format": "Q2.14 int16 (step 2^-14, range [-2,2))", "mask_format": "fp16",
                    "masked_format": "Q2.14 int16", "rounding": "round-half-to-EVEN (RNE) to Q2.14, single round after the full-precision products+add",
                    "accumulator_note": "products sr*mr etc. + the subtract/add evaluated >= Q2.14+guard (fp32/wide); fp16-accumulate would NOT match this golden bit-exactly",
                    "dc": "masked bin 0 = 0; spec/mask bins 0..63 -> masked bins 1..64",
                    "prereq": "decoder.model.5.0.pointwise_x (sub-ops 28/29) MUST emit fp16-PASS (no e4m3 cast); e4m3 mask -> 1.9% RMS"}
            json.dump(meta, open(os.path.join(OUT, "mask_apply_golden_meta.json"), "w"), indent=1)
    print(f"[gen] wrote {OUT}/{{L,R}}/ + mask_apply_golden_meta.json")


if __name__ == "__main__":
    main()
