#!/usr/bin/env python3
"""Re-bake FP8-outlier descale codes for the pre-tree preshift fix.

Per RTL_FIX_FP16_TREE_PRESHIFT_2026-06-07 §3 + REQUEST_FP16_PPU_EVAL_2026-06-10
追加: move a coarse per-tensor pow2 (`preshift`, <=0) BEFORE the fp16 32-way MAC
tree so the FP8xFP8 pre-descale partials stop overflowing fp16 (target <=0.49x of
65504), and store the RESIDUAL descale (= full_descale / 2^preshift) in the WBUF
weight-scale codes.

Key identity: residual = full * 2^(-preshift). In the APoT 2-term encoding
(value = sign*(2^a_exp +/- 2^b_exp)) this is a PURE exponent offset:
    a_exp += (-preshift) ,  b_exp += (-preshift)        # preshift<=0 -> exps grow
    has_second, residual_sign : unchanged
With pe_pair applying 2^preshift to the products and the dealer applying the
residual, net descale = full -> the deployed output is BIT-UNCHANGED (MBSTOI
unaffected); only the fp16 headroom changes. The fused (act-k) variants shift by
the SAME offset because act-k fusion is additive in the exponent.

Targets the ACTIVE image: 19_algo_final/hw_patterns_final/tb_pattern/weights/.
Backs up originals, writes a preshift_manifest.json, annotates
fused_descale_summary.json, and self-validates (round-trip identity + post-fix
maxnode <= TARGET). Idempotent-guarded: refuses to double-bake if a backup exists.

Usage:
    conda run -n bcctn python scripts/rebake_fp8_preshift.py [--apply] [--force]
Without --apply it is a DRY RUN (computes + validates, writes nothing).
"""
from __future__ import annotations
import argparse
import json
import math
import os
import shutil
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import verify_fp16_tree_headroom as hr  # load_weight, load_acts, scan_pointwise, TARGET ...

WEIGHTS = os.path.join(ROOT, "hw_patterns_final", "tb_pattern", "weights")
SUMMARY = os.path.join(WEIGHTS, "fused_descale_summary.json")
BAK = os.path.join(WEIGHTS, ".prebake_bak")
MANIFEST = os.path.join(WEIGHTS, "preshift_manifest.json")

FP8_OUTLIER = [
    f"decoder.model.{s}.0.pointwise_{ri}_{branch}"
    for s in (0, 3, 4) for ri in ("r", "i") for branch in ("x", "skip")
]  # 12 tensors

A_FILES = ["fp8_descale_apot_a_exp.int16.bin",
           "fp8_descale_apot_a_exp_fused_act_kreal.int16.bin",
           "fp8_descale_apot_a_exp_fused_act_kimag.int16.bin"]
B_FILES = ["fp8_descale_apot_b_exp.int16.bin",
           "fp8_descale_apot_b_exp_fused_act_kreal.int16.bin",
           "fp8_descale_apot_b_exp_fused_act_kimag.int16.bin"]


def _decode_apot(a_exp, b_exp, has_second, residual_sign):
    """descale = 2^a_exp [+/- 2^b_exp]; primary always +, residual_sign on 2nd term."""
    val = 2.0 ** float(a_exp)
    if has_second:
        val += (1.0 if residual_sign >= 0 else -1.0) * (2.0 ** float(b_exp))
    return float(val)


def _read(path, dt):
    return np.fromfile(path, dtype=dt)


def compute_preshift(tensor):
    """preshift = floor(log2(TARGET / maxnode)) (<=0), 0 if no overflow risk."""
    wl = hr.load_weight(os.path.join(hr.WDIR, tensor))
    if wl is None:
        raise RuntimeError(f"no weight for {tensor} in {hr.WDIR}")
    mode, w2d, _ = wl
    acts = hr.load_acts(hr.ADIR, tensor.replace(".", "_"))
    if not acts:
        raise RuntimeError(f"no activations for {tensor} in {hr.ADIR}")
    r0 = hr.scan_pointwise(tensor, w2d, acts, preshift=0)
    maxnode = r0["maxnode"]
    if maxnode <= hr.TARGET:
        return 0, maxnode, maxnode
    s = int(math.floor(math.log2(hr.TARGET / maxnode)))   # <= -1
    rfix = hr.scan_pointwise(tensor, w2d, acts, preshift=s)
    return s, maxnode, rfix["maxnode"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="write changes (else dry run)")
    ap.add_argument("--force", action="store_true", help="re-bake even if backup exists")
    args = ap.parse_args()

    if os.path.isdir(BAK) and args.apply and not args.force:
        print(f"[rebake] ABORT: backup already exists at {BAK} -> image looks already "
              f"re-baked. Use --force to override (will NOT re-backup).")
        sys.exit(2)

    print(f"[rebake] {'APPLY' if args.apply else 'DRY-RUN'}  target={WEIGHTS}")
    print(f"{'tensor':40} {'preshift':>8} {'pre|node|':>11} {'post|node|':>11} "
          f"{'a':>4}->{'a+|s|':>5} {'b':>4}->{'b+|s|':>5} {'descale x':>10}")
    manifest = {}
    all_ok = True
    for t in FP8_OUTLIER:
        d = os.path.join(WEIGHTS, t)
        s, pre, post = compute_preshift(t)
        shift = -s  # >= 0
        a0 = int(_read(os.path.join(d, A_FILES[0]), np.int16)[0])
        b0 = int(_read(os.path.join(d, B_FILES[0]), np.int16)[0])
        hs = int(_read(os.path.join(d, "fp8_descale_apot_has_second.u8.bin"), np.uint8)[0])
        sg = int(_read(os.path.join(d, "fp8_descale_apot_residual_sign.int8.bin"), np.int8)[0])
        old_descale = _decode_apot(a0, b0, hs, sg)
        new_descale = _decode_apot(a0 + shift, b0 + shift, hs, sg)
        ratio = new_descale / old_descale if old_descale else float("nan")
        # round-trip identity check (exact pow2 scaling)
        ok = abs(ratio - 2.0 ** shift) < 1e-9 * max(1.0, 2.0 ** shift) and post <= hr.TARGET
        all_ok &= ok
        manifest[t] = {
            "preshift": s, "maxnode_pre": round(pre, 1), "maxnode_post": round(post, 1),
            "a_exp": a0, "a_exp_residual": a0 + shift, "b_exp": b0, "b_exp_residual": b0 + shift,
            "has_second": hs, "residual_sign": sg,
            "descale_scale_factor": round(ratio, 6),
        }
        flag = "" if ok else "  <<< CHECK"
        print(f"{t:40} {('2^%d'%s) if s else '—':>8} {pre:>11.1f} {post:>11.1f} "
              f"{a0:>4}->{a0+shift:>5} {b0:>4}->{b0+shift:>5} {('x%g'%(2.0**shift)):>10}{flag}")

        if args.apply and s != 0:
            tb = os.path.join(BAK, t)
            os.makedirs(tb, exist_ok=True)
            for fn in A_FILES + B_FILES:
                p = os.path.join(d, fn)
                if not os.path.exists(p):
                    continue
                if not os.path.exists(os.path.join(tb, fn)):  # backup once
                    shutil.copy2(p, os.path.join(tb, fn))
                arr = _read(p, np.int16)
                (arr + np.int16(shift)).astype(np.int16).tofile(p)

    print(f"\n[rebake] round-trip + headroom checks: {'ALL PASS' if all_ok else 'FAILED'}")
    if not all_ok:
        sys.exit(1)

    if args.apply:
        os.makedirs(BAK, exist_ok=True)
        with open(MANIFEST, "w") as f:
            json.dump({"note": "pre-tree preshift fix; WBUF scale codes are RESIDUAL "
                       "(full - preshift). RTL descriptor preshift_{r,i} = this preshift. "
                       "preshift==0 tensors untouched.", "tensors": manifest}, f, indent=2)
        print(f"[rebake] wrote {MANIFEST}")
        if os.path.exists(SUMMARY):
            summ = json.load(open(SUMMARY))
            tgt = summ.get("fused", summ)
            for t, m in manifest.items():
                if t in tgt and isinstance(tgt[t], dict):
                    tgt[t]["preshift"] = m["preshift"]
                    tgt[t]["wscale_is_residual"] = (m["preshift"] != 0)
            json.dump(summ, open(SUMMARY, "w"), indent=2)
            print(f"[rebake] annotated {SUMMARY} with preshift fields")
        print(f"[rebake] originals backed up under {BAK}")
        print("[rebake] NEXT (builder/RTL): re-pack wbuf_unified_init.bin from these "
              "weights, set descriptor preshift_{r,i} from preshift_manifest.json, regen ROM.")
    else:
        print("[rebake] dry-run only; re-run with --apply to write.")


if __name__ == "__main__":
    main()
