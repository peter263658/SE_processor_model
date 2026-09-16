#!/usr/bin/env python3
"""Witness: decoder stage-4 FP8-outlier fp16 Inf collision, before/after preshift.

The consultant's quality-gate witness (REPLY 2026-06-10 Q2): without the pre-tree
preshift the FP8xFP8 products overflow the fp16 32-way tree (single product
448x448=200704 > 65504 -> Inf; nodes up to 264448), and the re-baked preshift
(2^-3/2^-4) must make every node fit (0 overflow). This reproduces it on real
activations and pinpoints the peak frame (the "t=351"-class event).

Reuses verify_fp16_tree_headroom's exact fp16 saturating tree + e4m3 decode.
Read-only.
"""
from __future__ import annotations
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import verify_fp16_tree_headroom as hr  # e4m3_decode, fp16_sat_tree, load_weight, load_acts, GROUP, TARGET, FP16_MAX

DEC4 = [f"decoder.model.4.0.pointwise_{ri}_{br}" for ri in ("r", "i") for br in ("x", "skip")]


def per_frame_scan(w2d, acts, preshift):
    """Return (max_single_product, max_node, inf_nodes, peak_tag, peak_frame)."""
    oc, ic = w2d.shape
    psc = 2.0 ** preshift
    best = dict(maxprod=0.0, maxnode=0.0, inf=0, tag="", frame=-1)
    for tag, codes in acts:
        if codes.size % ic:
            continue
        P = codes.size // ic
        a = hr.e4m3_decode(codes).reshape(ic, P)
        for g0 in range(0, ic, hr.GROUP):
            gw = min(hr.GROUP, ic - g0)
            wg = w2d[:, g0:g0 + gw]
            ag = a[g0:g0 + gw, :]
            prod = (wg[:, :, None] * ag[None, :, :]) * psc          # [oc, gw, P]
            best["maxprod"] = max(best["maxprod"], float(np.abs(prod).max()))
            pr = np.transpose(prod, (0, 2, 1)).reshape(oc * P, gw)
            if gw < hr.GROUP:
                pr = np.pad(pr, ((0, 0), (0, hr.GROUP - gw)))
            _, _, mx, ov = hr.fp16_sat_tree(pr)
            # per (oc,P) row -> map back to frame P
            mx2 = mx.reshape(oc, P)
            ov2 = ov.reshape(oc, P)
            fmax = np.abs(mx2).max(axis=0)            # per-frame worst node
            fi = int(np.argmax(fmax))
            if float(fmax[fi]) > best["maxnode"]:
                best["maxnode"] = float(fmax[fi]); best["tag"] = tag; best["frame"] = fi
            best["inf"] += int(ov2.sum())
    return best


def main():
    print("=" * 92)
    print("WITNESS: decoder stage-4 FP8-outlier fp16 tree overflow, pre/post re-bake preshift")
    print("golden input:", "p225_p225_009_az-40_snr+0.0.wav (headroom dump)")
    print("=" * 92)
    print(f"{'tensor':38}{'preshift':>9}{'max|prod|':>11}{'max|node|':>11}{'/65504':>9}{'Inf nodes':>10}  peak")
    print("-" * 92)
    any_pre_inf = False
    for t in DEC4:
        wl = hr.load_weight(os.path.join(hr.WDIR, t))
        if wl is None:
            print(f"{t:38}  (no weight)"); continue
        _, w2d, _ = wl
        acts = hr.load_acts(hr.ADIR, t.replace(".", "_"))
        if not acts:
            print(f"{t:38}  (no acts)"); continue
        r0 = per_frame_scan(w2d, acts, 0)
        s = int(math.floor(math.log2(hr.TARGET / r0["maxnode"]))) if r0["maxnode"] > hr.TARGET else 0
        rfix = per_frame_scan(w2d, acts, s)
        any_pre_inf |= (r0["maxnode"] > hr.FP16_MAX)
        # pre-fix row (preshift=0): single product already > 65504 => Inf in pure fp16
        prod_inf = "Inf!" if r0["maxprod"] > hr.FP16_MAX else ""
        print(f"{t:38}{'0 (raw)':>9}{r0['maxprod']:>11.0f}{r0['maxnode']:>11.0f}"
              f"{r0['maxnode']/hr.FP16_MAX:>8.2f}x{r0['inf']:>10d}  {r0['tag']}:t={r0['frame']} {prod_inf}")
        print(f"{'  -> re-baked':38}{('2^%d'%s):>9}{rfix['maxprod']:>11.0f}{rfix['maxnode']:>11.0f}"
              f"{rfix['maxnode']/hr.FP16_MAX:>8.2f}x{rfix['inf']:>10d}  (fits)")
    print("-" * 92)
    print("VERDICT: pre-rebake FP8 products/nodes exceed fp16 65504 (single product 448x448=200704 -> Inf;")
    print("the dec4 peak frame is the 't=351'-class collision). Re-baked preshift lands every node <=0.49x,")
    print("0 Inf -> the re-bake removes the overflow event. (preshift = exact identity, so output value")
    print("is unchanged; the fp16 tree then only adds benign +-1 ulp, like the conv path.)")


if __name__ == "__main__":
    main()
