#!/usr/bin/env python3
"""
Cross-validate the L-Mul golden mirror (tb_pe_lane.lmul_product_bits) against the
ACTUAL software reference, complexLayers._lmul_multiply_bit_exact, in the
fp16-normal result region. This closes the gap that mode 1/2 golden was only a
mirror of the RTL: it proves the mirror (hence the RTL) encodes the true SW spec.

Run (env has torch):
    PYTHONPATH=../../10_ALGO/code conda run -n bcctn_test python tb_pe_lane_xcheck_lmul.py
"""
import os, sys, struct, random
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "10_ALGO", "code"))

import torch
from DCNN.utils.complexPyTorch.complexLayers import _lmul_multiply_bit_exact
from tb_pe_lane import lmul_product_bits, synth_fp16_codes, decode_operand_float


def f_to_fp16_bits(v):
    try:
        return struct.unpack(">H", struct.pack(">e", v))[0]
    except (OverflowError, struct.error):
        return None


def check(label, pairs, EXP_W, FRAC_W, HAS_INF, m):
    gs, vxs, vys, keep = [], [], [], []
    for xb, yb in pairs:
        g = lmul_product_bits(xb, yb, EXP_W, FRAC_W, HAS_INF, m)
        sx, vx, zx, spx = decode_operand_float(xb, EXP_W, FRAC_W, HAS_INF)
        sy, vy, zy, spy = decode_operand_float(yb, EXP_W, FRAC_W, HAS_INF)
        e = (g >> 10) & 0x1F
        skip = zx or zy or spx or spy or e == 0 or e == 0x1F  # documented deviations
        gs.append(g); keep.append(not skip)
        vxs.append(vx if vx is not None else 0.0)
        vys.append(vy if vy is not None else 0.0)
    t = _lmul_multiply_bit_exact(torch.tensor(vxs, dtype=torch.float32),
                                 torch.tensor(vys, dtype=torch.float32), m)
    cmp = mism = 0
    for i, (g, k) in enumerate(zip(gs, keep)):
        if not k:
            continue
        cmp += 1
        tb = f_to_fp16_bits(float(t[i].item()))
        if tb != g:
            mism += 1
            if mism <= 12:
                print(f"  MISMATCH x={pairs[i][0]:04x} y={pairs[i][1]:04x} "
                      f"mirror={g:04x} torch={tb} ({float(t[i].item())})")
    print(f"[{label}] compared={cmp}  skipped(special/FTZ/inf)={len(gs)-cmp}  mismatch={mism}")
    return mism


def main():
    rng = random.Random(7)
    p1 = [(x, y) for x in range(256) for y in range(256)]            # fp8: exhaustive
    m1 = check("mode1 FP8 L-Mul m4 (exhaustive 256x256)", p1, 4, 3, 0, 4)
    f16 = synth_fp16_codes(400, rng)
    p2 = [(rng.choice(f16), rng.choice(f16)) for _ in range(40000)]  # fp16: synth cross
    m2 = check("mode2 FP16 L-Mul m8 (synth 40000)", p2, 5, 10, 1, 8)
    if m1 or m2:
        sys.exit(f"[FATAL] L-Mul mirror != torch: mode1={m1} mode2={m2}")
    print("[xcheck] PASS — L-Mul mirror == torch _lmul_multiply_bit_exact (fp16-normal region)")


if __name__ == "__main__":
    main()
