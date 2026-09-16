#!/usr/bin/env python3
"""Map the MHA in_proj bias (768 = Q256|K256|V256) into the image (sub-op 31).

Completes the bias set: input_proj β @251872, output_proj β @252896,
MHA out_proj b_o @253920, and now MHA in_proj bias. After this all four biases are
WBUF-resident -> live const feed needs no TB-preload.

Non-destructive (*.pre_inprojbias.bak; re-run restores first).
    cd /raid/R12K41024/19_algo_final
    conda run --no-capture-output -n bcctn python scripts/patch_memmap_mha_inproj_bias.py
"""
from __future__ import annotations
import os, json, shutil
import numpy as np

HW = "/raid/R12K41024/19_algo_final/hw_patterns_final"
SR = os.path.join(HW, "sram_images")
MM = os.path.join(SR, "memory_map.json")
IMG = os.path.join(SR, "wbuf_unified_init.bin")
BIAS = os.path.join(HW, "tb_pattern", "weights", "mattn.mattn.MultiheadAttention_re", "in_proj_weight", "bias.fp16.bin")


def main():
    for f in (MM, IMG):
        bak = f + ".pre_inprojbias.bak"
        if os.path.exists(bak):
            shutil.copy2(bak, f)
        else:
            shutil.copy2(f, bak)
    mm = json.load(open(MM)); img = bytearray(open(IMG, "rb").read())
    cap = mm["wbuf_capacity_bytes"]
    base = (mm["wbuf_used_bytes"] + 3) & ~3
    bi = np.fromfile(BIAS, np.float16)                 # [768] = Q256|K256|V256
    assert bi.size == 768
    blob = bi.tobytes(); assert len(blob) == 1536
    end = base + 1536; assert end <= cap, f"WBUF overflow {end}>{cap}"
    img[base:end] = blob

    cb = {"dotted": "mattn.mattn.MultiheadAttention_re/in_proj_weight", "kind": "bn_bias",
          "subkind": "mha_in_proj_bias", "region": "R6_proj_fused", "decode": "fp16",
          "n_elems": 768, "layout": "Q[256]||K[256]||V[256] (head-major within each)",
          "add": "plain post-acc per-pass: qkv += bias; SAME const for real & imag passes",
          "wbuf_offset": base, "wbuf_bytes": 1536, "subop": 31}
    mm["const_blocks"].append(cb)
    mm["const_prefetch"].append({"unit": "mattn.mattn.MultiheadAttention_re/in_proj_weight:bn_bias",
                                 "src_wbuf_offset": base, "bytes": 1536, "decode": "fp16"})
    mm["regions"]["proj_fused"]["end"] = end
    mm["regions"]["proj_fused"]["bytes"] = end - mm["regions"]["proj_fused"]["start"]
    mm["wbuf_used_bytes"] = end; mm["wbuf_slack_bytes"] = cap - end
    mm["counts"]["bn_bias_bytes"] += 1536
    mm["counts"]["mha_in_proj_bias"] = {"n_elems": 768, "bytes": 1536, "wbuf_offset": base, "subop": 31}
    open(IMG, "wb").write(bytes(img)); json.dump(mm, open(MM, "w"), indent=1)

    img2 = np.fromfile(IMG, np.uint8)
    rb = np.frombuffer(bytes(img2[base:end]), np.float16).astype(np.float64)
    bak = np.fromfile(IMG + ".pre_inprojbias.bak", np.uint8)
    print(f"[patch] MHA in_proj bias @[{base},{end}) (1536 B, 768 fp16); slack {mm['wbuf_slack_bytes']} B")
    print(f"[verify] readback==export:{np.array_equal(rb, bi.astype(np.float64))} |max|={np.abs(rb).max():.4f}; "
          f"image {img2.size}B; bytes[:{base}) unchanged:{np.array_equal(img2[:base], bak[:base])}")
    print("[summary] biases now WBUF-resident: input_proj β@251872, output_proj β@252896, "
          f"out_proj b_o@253920, in_proj bias@{base}")


if __name__ == "__main__":
    main()
