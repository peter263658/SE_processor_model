"""Probe: eager calibrated descale vs TRUE fused single-descale on the 12 FP8-outlier
L-Mul layers.

Background
----------
The calibrated_e4m3 deploy ckpt scored 0.8354 with the EAGER activation descale
(activation is descaled at its quantization boundary, then enters L-Mul as a
general float). The real hardware does ONE post-accumulation descale: the
activation enters the MAC as the raw fp8 *code*, and  descale_combined =
descale_a * descale_w  is applied once after accumulation.

For INT4 layers this is bit-exact (real multiply, pure reassociation). For the
12 FP8xFP8 *L-Mul* outliers it is NOT exact, because L-Mul is approximate and
folding the per-tensor activation descale into / out of the L-Mul operand
changes the mantissa-add result. This probe measures that gap per conv-call:

  eager  = APoT(d_w)            * LMul(input,          code)     # what model did
  fused  = APoT(d_w * descale_a)* LMul(input/descale_a, code)    # true HW path

input = act_code * descale_a  (the calibrated-descaled activation the conv sees).
act_code = input / descale_a is the fp8 code; descale_a recovered from saturation
(max|input| == 448*descale_a when the p99.9 calibration clips some values).
We verify the recovery is exact by round-tripping act_code through e4m3.
"""
from __future__ import annotations

import os
import re
import json
import glob
import numpy as np
import soundfile as sf
import torch

import export_helpers as eh
from DCNN.utils.all_lmul_multiply import _lmul_conv2d, _apot_descale_value


def _load_calib_descales(path):
    """stage -> {'real': descale, 'imag': descale} for split_pointwise_cat boundaries."""
    data = json.load(open(path)).get("keys", {})
    out = {}
    for k, v in data.items():
        m = re.search(r"(decoder\.model\.\d+\.0)\.split_pointwise_cat\.(real|imag)$", k)
        if m:
            out.setdefault(m.group(1), {})[m.group(2)] = float(v["descale"])
    return out


def _recover_descale_a(inp, candidates, e4m3):
    """Pick the calibrated descale that makes inp/descale land exactly on the fp8 grid."""
    best = (None, float("inf"))
    for da in candidates:
        if da <= 0:
            continue
        code = inp / da
        rt = (code.to(e4m3).float() - code).abs().max().item()
        if rt < best[1]:
            best = (da, rt)
    return best  # (descale_a, round_trip_err)

WAV = sorted(glob.glob("/raid/R12K41024/BCCTN/Dataset/noisy_testset/snr_0dB/*.wav"))[0]
DD_MIN = int(os.environ.get("BCCTN_OUTLIER_DESCALE_APOT_MIN", "-16"))
DD_MAX = int(os.environ.get("BCCTN_OUTLIER_DESCALE_APOT_MAX", "-6"))
# widened range for the FUSED product (descale_a*descale_w reaches ~2^-17)
FUSED_MIN = int(os.environ.get("BCCTN_FUSED_DESCALE_APOT_MIN", "-20"))
FUSED_MAX = int(os.environ.get("BCCTN_FUSED_DESCALE_APOT_MAX", "-6"))
E4M3_MAX = 448.0


def relL2(a, b):
    return (a - b).norm().item() / (b.norm().item() + 1e-12)


def main():
    os.environ["BCCTN_ACTIVATION_FP8_MODE"] = "calibrated_e4m3"
    os.environ.setdefault(
        "BCCTN_ACTIVATION_CALIB_PATH",
        os.path.join(eh.PACKAGE_ROOT, "ckpt", "activation_calibration_p999_b64.json"),
    )
    os.environ["BCCTN_OUTLIER_DELAYED_DESCALE"] = "1"
    os.environ["BCCTN_OUTLIER_DESCALE_APOT_MIN"] = str(DD_MIN)
    os.environ["BCCTN_OUTLIER_DESCALE_APOT_MAX"] = str(DD_MAX)

    ckpt = os.path.join(eh.PACKAGE_ROOT, "ckpt",
                        "strict_core_calibrated_e4m3_deploy_ep2_vctk0.8354_timit0.7897.ckpt")
    model, *_ = eh.build_model(ckpt_path=ckpt)
    model = model.eval()
    names = set(getattr(model, "_fp8_preserved_names", []))
    print(f"[probe] {len(names)} outlier names; calib={os.environ['BCCTN_ACTIVATION_CALIB_PATH']}")

    # capture input to each outlier conv per call
    caps = []  # list of (name, call_idx, input_tensor, module)
    counters = {}
    hooks = []
    def mk(nm, module):
        def pre(mod, args):
            i = counters.get(nm, 0); counters[nm] = i + 1
            x = args[0]
            if isinstance(x, torch.Tensor):
                caps.append((nm, i, x.detach().clone(), mod))
            return None
        return pre
    for nm, mod in model.named_modules():
        if nm in names and isinstance(mod, torch.nn.Conv2d):
            hooks.append(mod.register_forward_pre_hook(mk(nm, mod)))

    audio, sr = sf.read(WAV, dtype="float32", always_2d=True)
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    audio = audio[:32000, :2]
    if audio.shape[0] < 32000:
        audio = np.concatenate([audio, np.zeros((32000 - audio.shape[0], 2), np.float32)])
    x = torch.from_numpy(audio.T).unsqueeze(0).contiguous()
    with torch.no_grad():
        model(x)
    for h in hooks:
        h.remove()
    print(f"[probe] captured {len(caps)} outlier conv calls\n")

    e4m3 = torch.float8_e4m3fn
    calib = _load_calib_descales(os.environ["BCCTN_ACTIVATION_CALIB_PATH"])
    rows = []
    for nm, ci, inp, mod in caps:
        w = mod.weight.detach().float()
        wmax = w.abs().max().clamp_min(1e-12)
        code = w * (E4M3_MAX / wmax)
        d_w = (wmax / E4M3_MAX)
        # exact descale_a: pick the stage's real/imag calibrated descale that
        # makes inp/descale land on the fp8 grid (handles per-call real/imag).
        sm = re.search(r"(decoder\.model\.\d+\.0)\.", nm)
        cand = list(calib.get(sm.group(1), {}).values()) if sm else []
        cand = cand or [inp.abs().max().item() / E4M3_MAX]
        descale_a, rt = _recover_descale_a(inp, cand, e4m3)
        act_code = inp / descale_a
        st, pa, di, gr = mod.stride, mod.padding, mod.dilation, mod.groups
        with torch.no_grad():
            acc_e = _lmul_conv2d(inp, code, None, st, pa, di, gr, 4, True, 0, False)
            eager = acc_e * _apot_descale_value(d_w, DD_MIN, DD_MAX)
            acc_f = _lmul_conv2d(act_code, code, None, st, pa, di, gr, 4, True, 0, False)
            comb = _apot_descale_value(d_w * descale_a, FUSED_MIN, FUSED_MAX)
            fused = acc_f * comb
        rows.append((nm, ci, float(descale_a), rt, relL2(fused, eager),
                     (fused - eager).abs().max().item(), eager.abs().max().item()))

    print(f"{'layer':32s} {'call':>4s} {'descale_a':>10s} {'fp8_rt':>9s} {'relL2(f,e)':>11s} {'maxabsΔ':>10s}")
    worst = 0.0
    for nm, ci, da, rt, rl, mx, em in rows:
        worst = max(worst, rl)
        flag = "  <-- big" if rl > 0.01 else ""
        print(f"{nm.replace('decoder.model.',''):32s} {ci:>4d} {da:>10.4e} {rt:>9.2e} {rl:>11.4e} {mx:>10.3e}{flag}")
    print(f"\n[probe] worst relL2(fused,eager) across all outlier calls = {worst:.4e}")
    print("[probe] (fp8_rt ~0 confirms act_code recovery is exact; "
          "relL2 is the L-Mul reassociation gap the FT was NOT trained on.)")


if __name__ == "__main__":
    main()
