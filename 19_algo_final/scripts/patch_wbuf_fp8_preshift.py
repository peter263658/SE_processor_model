#!/usr/bin/env python3
"""In-place patch wbuf_unified_init.bin with the re-baked FP8-outlier descale codes.

Companion to rebake_fp8_preshift.py. That script shifted the per-tensor fused
APoT a/b exponents in tb_pattern/weights/. This script propagates the shift into
the packed SRAM image by overwriting ONLY the 12 weight_scale const_blocks
(region R2, apot_6a, 3 bytes = 2x12-bit kreal|kimag) at their memory_map offsets.

Bytes-only: offset/size are unchanged (exponent shift stays in the 5-bit biased
range), so memory_map.json needs no edit. Uses the builder's own apot_code /
pack_codes so the encoding is bit-identical to a full re-pack.

Self-validating: the CURRENT wbuf bytes at each offset must decode to the
PRE-bake (.prebake_bak) a/b — proves the offset, 12-bit layout, and bias=24 are
right before any write.

  conda run -n bcctn python scripts/patch_wbuf_fp8_preshift.py [--apply]
Without --apply it is a DRY RUN.
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S2H = "/raid/R12K41024/18_w4a8_pow2_mixed_lmul_hw_handoff_20260530/software_to_hardware_18"
sys.path.insert(0, S2H)
from build_sram_images import apot_code, apot_decode, pack_codes, unpack_codes  # noqa: E402

IMG = os.path.join(ROOT, "hw_patterns_final", "sram_images")
WBUF = os.path.join(IMG, "wbuf_unified_init.bin")
MMAP = os.path.join(IMG, "memory_map.json")
WEIGHTS = os.path.join(ROOT, "hw_patterns_final", "tb_pattern", "weights")
BAK = os.path.join(WEIGHTS, ".prebake_bak")
MANIFEST = os.path.join(WEIGHTS, "preshift_manifest.json")
WBUF_BAK = WBUF + ".pre_fp8preshift_bak"
BIAS = 24


def _i16(path):
    return int(np.fromfile(path, dtype=np.int16)[0])


def _codes_from_fused(ab_dir, meta_dir=None):
    """[kreal_code, kimag_code] from a/b in ab_dir + has/sign in meta_dir (default
    ab_dir). has_second/residual_sign are unchanged by the re-bake, so for the
    pre-bake codes we read a/b from .prebake_bak but meta from the live dir."""
    meta_dir = meta_dir or ab_dir
    has = int(np.fromfile(os.path.join(meta_dir, "fp8_descale_apot_has_second.u8.bin"), np.uint8)[0])
    rs = int(np.fromfile(os.path.join(meta_dir, "fp8_descale_apot_residual_sign.int8.bin"), np.int8)[0])
    sign_lo = 1 if rs < 0 else 0
    codes = []
    for br in ("kreal", "kimag"):
        a = _i16(os.path.join(ab_dir, f"fp8_descale_apot_a_exp_fused_act_{br}.int16.bin"))
        b = _i16(os.path.join(ab_dir, f"fp8_descale_apot_b_exp_fused_act_{br}.int16.bin"))
        codes.append(apot_code(a, b, has, sign_lo, BIAS))
    return codes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    manifest = json.load(open(MANIFEST))["tensors"]
    mm = json.load(open(MMAP))
    blocks = {b["dotted"]: b for b in mm["const_blocks"]
              if b.get("kind") == "weight_scale" and b.get("dotted") in manifest}
    wbuf = bytearray(open(WBUF, "rb").read())

    print(f"[wbuf-patch] {'APPLY' if args.apply else 'DRY-RUN'}  {WBUF} ({len(wbuf)} B)")
    print(f"{'tensor':38} {'off':>7} {'cur codes':>14} -> {'new codes':>14}  shift  chk")
    patches = []
    all_ok = True
    for dotted, m in manifest.items():
        s = m["preshift"]
        blk = blocks[dotted]
        off, n = blk["wbuf_offset"], blk["wbuf_bytes"]
        cur = bytes(wbuf[off:off + n])
        cur_codes = unpack_codes(cur, 2, "12bit")
        # decode current -> a/b ; must equal the PRE-bake fused a/b
        old = (_codes_from_fused(os.path.join(BAK, dotted), os.path.join(WEIGHTS, dotted))
               if s != 0 else list(cur_codes))
        new = _codes_from_fused(os.path.join(WEIGHTS, dotted))
        # sanity: current image holds the OLD codes
        chk_cur = (list(cur_codes) == list(old))
        # sanity: new == old shifted by +|s| in both a and b (or unchanged if s==0)
        shift = -s
        ok_shift = True
        for cc, nc in zip(old, new):
            ca, cb, ch, cs = apot_decode(cc, BIAS)
            na, nb, nh, ns = apot_decode(nc, BIAS)
            ok_shift &= (na == ca + shift and nb == cb + shift and nh == ch and ns == cs)
        ok = chk_cur and ok_shift
        all_ok &= ok
        new_bytes = pack_codes(new, "12bit")
        patches.append((off, n, new_bytes))
        print(f"{dotted:38} {off:>7} {str(list(cur_codes)):>14} -> {str(list(new)):>14}  "
              f"{('2^%d'%s) if s else '0':>5}  {'ok' if ok else 'FAIL'}")

    print(f"\n[wbuf-patch] sanity (current==prebake & new==old<<|s|): {'ALL PASS' if all_ok else 'FAILED'}")
    if not all_ok:
        sys.exit(1)

    if args.apply:
        if not os.path.exists(WBUF_BAK):
            shutil.copy2(WBUF, WBUF_BAK)
            print(f"[wbuf-patch] backed up -> {WBUF_BAK}")
        changed = 0
        for off, n, nb in patches:
            assert len(nb) == n
            for i in range(n):
                if wbuf[off + i] != nb[i]:
                    changed += 1
                wbuf[off + i] = nb[i]
        open(WBUF, "wb").write(bytes(wbuf))
        # verify bytes-only vs backup
        orig = open(WBUF_BAK, "rb").read()
        new_img = open(WBUF, "rb").read()
        diff_idx = [i for i in range(len(orig)) if orig[i] != new_img[i]]
        patched_ranges = set()
        for off, n, _ in patches:
            patched_ranges.update(range(off, off + n))
        only_in_blocks = all(i in patched_ranges for i in diff_idx)
        print(f"[wbuf-patch] wrote {WBUF}: {changed} bytes changed across {len(patches)} blocks")
        print(f"[wbuf-patch] diff vs backup = {len(diff_idx)} bytes; all inside the 12 "
              f"weight_scale blocks: {only_in_blocks}")
        print(f"[wbuf-patch] memory_map.json untouched -> offsets/sizes unchanged (bytes-only).")
        if not only_in_blocks:
            print("[wbuf-patch] WARNING: diff bytes outside the descale blocks!"); sys.exit(1)
    else:
        print("[wbuf-patch] dry-run only; re-run with --apply to write.")


if __name__ == "__main__":
    main()
