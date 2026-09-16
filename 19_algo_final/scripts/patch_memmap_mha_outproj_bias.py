#!/usr/bin/env python3
"""Map the MHA out_proj bias (sub-op 34) into the deploy SRAM image (RELAY #1).

Appends one const_block (MHA out_proj bias b_o[256] fp16, plain-add β, SAME for
both passes) into the WBUF slack of the (already proj-fused-patched) image, so
gen_ce_subop_rom.py can attach a bias pointer to sub-op 34's PPU β path.

Non-destructive: backs up to *.pre_outprojbias.bak (re-run restores first), keeps
the existing proj-fused const, updates const_blocks/const_prefetch/counts/used+slack.

    cd /raid/R12K41024/19_algo_final
    conda run --no-capture-output -n bcctn python scripts/patch_memmap_mha_outproj_bias.py
"""
from __future__ import annotations
import os, json, shutil
import numpy as np

HW = "/raid/R12K41024/19_algo_final/hw_patterns_final"
SR = os.path.join(HW, "sram_images")
MM = os.path.join(SR, "memory_map.json")
IMG = os.path.join(SR, "wbuf_unified_init.bin")
BIAS = os.path.join(HW, "tb_pattern", "weights", "mattn.mattn.MultiheadAttention_re", "out_proj", "bias.fp16.bin")


def main():
    for f in (MM, IMG):
        bak = f + ".pre_outprojbias.bak"
        if os.path.exists(bak):
            shutil.copy2(bak, f)
        else:
            shutil.copy2(f, bak)

    mm = json.load(open(MM))
    img = bytearray(open(IMG, "rb").read())
    cap = mm["wbuf_capacity_bytes"]
    base = (mm["wbuf_used_bytes"] + 3) & ~3
    bo = np.fromfile(BIAS, np.float16)            # [256], real-valued
    assert bo.size == 256
    blob = bo.tobytes()
    assert len(blob) == 512
    end = base + 512
    assert end <= cap, f"WBUF overflow {end}>{cap}"
    img[base:end] = blob

    cb = {
        "dotted": "mattn.mattn.MultiheadAttention_re/out_proj", "kind": "bn_bias",
        "subkind": "mha_out_proj_beta", "region": "R6_proj_fused", "decode": "fp16",
        "n_elems": 256, "layout": "b_o[256] real (NOT complex)",
        "add": "plain per-pass: out_pass = fp16_add_sat(oacc_pass, b_o); SAME const for real & imag passes",
        "note": "tied MHA: net cancels in recombine real (a-b), doubles in imag (a+b=...+2*b_o)",
        "wbuf_offset": base, "wbuf_bytes": 512, "subop": 34,
    }
    mm["const_blocks"].append(cb)
    mm["const_prefetch"].append({"unit": "mattn.mattn.MultiheadAttention_re/out_proj:bn_bias",
                                 "src_wbuf_offset": base, "bytes": 512, "decode": "fp16"})
    mm["regions"]["proj_fused"]["end"] = end
    mm["regions"]["proj_fused"]["bytes"] = end - mm["regions"]["proj_fused"]["start"]
    mm["wbuf_used_bytes"] = end
    mm["wbuf_slack_bytes"] = cap - end
    mm["counts"]["bn_bias_bytes"] += 512
    mm["counts"].setdefault("mha_out_proj_bias", {})
    mm["counts"]["mha_out_proj_bias"] = {"n_elems": 256, "bytes": 512, "wbuf_offset": base, "subop": 34}

    open(IMG, "wb").write(bytes(img))
    json.dump(mm, open(MM, "w"), indent=1)
    print(f"[patch] MHA out_proj bias mapped at [{base},{end}) (512 B); new slack {mm['wbuf_slack_bytes']} B")

    # integrity + readback
    img2 = np.fromfile(IMG, np.uint8)
    rb = np.frombuffer(bytes(img2[base:end]), np.float16).astype(np.float64)
    print(f"[verify] readback bias == export: {np.array_equal(rb, bo.astype(np.float64))}  |max|={np.abs(rb).max():.4f}")
    bak = np.fromfile(IMG + ".pre_outprojbias.bak", np.uint8)
    print(f"[verify] image size {img2.size}==262144:{img2.size==262144}; bytes[:{base}) unchanged:{np.array_equal(img2[:base], bak[:base])}")


if __name__ == "__main__":
    main()
