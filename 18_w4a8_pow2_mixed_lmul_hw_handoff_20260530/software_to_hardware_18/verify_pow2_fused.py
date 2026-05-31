"""Verify the exported block-float fusion is exact: eager pow2 forward == fused
HW path on the 12 FP8-outlier L-Mul layers.

Eager (what the validated pow2 model computes):
    out = APoT(d_w) * LMul(input, code),   input = act_code * 2^k   (k = floor pow2)
Fused HW path (activation = raw code, single post-acc descale = exponent offset):
    out = (APoT(d_w) * 2^k) * LMul(act_code, code)

Because 2^k is a single power of two, LMul(act_code*2^k, code) == 2^k*LMul(act_code,code)
EXACTLY, so the two must be bit-identical. relL2 ~ 0 proves the exported
descale_pow2_exp offset is a faithful, lossless representation of the deployed math.
"""
from __future__ import annotations
import os, re, math, glob
import numpy as np, soundfile as sf, torch
import export_helpers as eh
from DCNN.utils.all_lmul_multiply import _lmul_conv2d, _apot_descale_value

E4M3_MAX = 448.0
WAV = sorted(glob.glob("/raid/R12K41024/BCCTN/Dataset/noisy_testset/snr_0dB/*.wav"))[0]


def main():
    os.environ["BCCTN_ACTIVATION_FP8_MODE"] = "calibrated_pow2_e4m3"
    os.environ.setdefault("BCCTN_ACTIVATION_CALIB_PATH", eh.DEFAULT_ACT_CALIB)
    os.environ["BCCTN_ACTIVATION_POW2_ROUND"] = "floor"
    os.environ["BCCTN_OUTLIER_DELAYED_DESCALE"] = "1"
    os.environ["BCCTN_OUTLIER_DESCALE_APOT_MIN"] = "-16"
    os.environ["BCCTN_OUTLIER_DESCALE_APOT_MAX"] = "-6"

    bf = eh.blockfloat_activation_table(os.environ["BCCTN_ACTIVATION_CALIB_PATH"])
    model, *_ = eh.build_model()  # DEFAULT_CKPT = pow2 ep3
    model.eval()
    names = set(getattr(model, "_fp8_preserved_names", []))

    caps, hooks = [], []
    def mk(nm):
        def pre(mod, args):
            if isinstance(args[0], torch.Tensor):
                caps.append((nm, args[0].detach().clone(), mod))
        return pre
    for nm, mod in model.named_modules():
        if nm in names and isinstance(mod, torch.nn.Conv2d):
            hooks.append(mod.register_forward_pre_hook(mk(nm)))

    a, sr = sf.read(WAV, dtype="float32", always_2d=True)
    if a.shape[1] == 1: a = np.repeat(a, 2, 1)
    a = a[:32000, :2]
    if a.shape[0] < 32000: a = np.concatenate([a, np.zeros((32000 - a.shape[0], 2), np.float32)])
    x = torch.from_numpy(a.T).unsqueeze(0).contiguous()
    with torch.no_grad():
        model(x)
    for h in hooks: h.remove()

    e4m3 = torch.float8_e4m3fn
    worst = 0.0
    n = 0
    for nm, inp, mod in caps:
        sm = re.search(r"(decoder\.model\.\d+\.0)\.", nm)
        part = "real" if "_r_" in nm else "imag"
        k = bf[f"{sm.group(1)}.split_pointwise_cat.{part}"]["scale_pow2_exp"]
        # runtime hook: input = e4m3(x*2^k)/2^k = code * 2^-k. Recover the raw code:
        act_code = inp * (2.0 ** k)               # = code_fp8 (fp8-representable)
        rt = (act_code.to(e4m3).float() - act_code).abs().max().item()
        w = mod.weight.detach().float(); wmax = w.abs().max().clamp_min(1e-12)
        code = w * (E4M3_MAX / wmax); d_w = wmax / E4M3_MAX
        st, pa, di, gr = mod.stride, mod.padding, mod.dilation, mod.groups
        with torch.no_grad():
            eager = _lmul_conv2d(inp, code, None, st, pa, di, gr, 4, True, 0, False) * _apot_descale_value(d_w, -16, -6)
            # fused: raw code into L-Mul, descale = APoT(d_w) * 2^-k  (single exponent offset)
            fused = _lmul_conv2d(act_code, code, None, st, pa, di, gr, 4, True, 0, False) * (_apot_descale_value(d_w, -16, -6) * (2.0 ** (-k)))
        rl = ((fused - eager).norm() / (eager.norm() + 1e-12)).item()
        worst = max(worst, rl); n += 1
        if rl > 1e-6:
            print(f"  {nm.replace('decoder.model.',''):28s} k={k:+d} fp8_rt={rt:.1e} relL2={rl:.3e}  <-- nonzero")
    print(f"[verify-pow2-fused] {n} outlier calls; worst relL2(fused,eager) = {worst:.3e}")
    print("[verify-pow2-fused] ~0 => exported pow2 descale offset reproduces the deployed math exactly.")


if __name__ == "__main__":
    main()
