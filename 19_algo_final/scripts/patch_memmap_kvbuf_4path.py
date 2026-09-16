#!/usr/bin/env python3
"""Pin the KVBUF 4-path layout + addressing contract into memory_map.json
(Consult CONSULT_KVBUF_4PATH_2026-06-09 item 1).

KVBUF = 4 banks x 64b x 1024 depth = 32 KiB (dual-port). e4m3 (1 byte) K/V.
4 paths, 8 KiB (256 rows @ 32 B) each = 32 KiB exact:
  K_re rows 0..255, K_im 256..511, V_re 512..767, V_im 768..1023.
Per-path element layout [head][frame][dim]: byte = head*512 + frame_slot*16 + dim.
32-frame streaming ring; kvbuf_init_zero.bin (32768 B all-zero) partitioning unchanged.

Non-destructive (memory_map.json.pre_kvbuf.bak).
    cd /raid/R12K41024/19_algo_final
    conda run --no-capture-output -n bcctn python scripts/patch_memmap_kvbuf_4path.py
"""
from __future__ import annotations
import os, json, shutil

SR = "/raid/R12K41024/19_algo_final/hw_patterns_final/sram_images"
MM = os.path.join(SR, "memory_map.json")
KVZERO = os.path.join(SR, "kvbuf_init_zero.bin")

H, FR, D = 16, 32, 16          # heads, ring frames, head_dim
ROW = 32                       # KVBUF bytes/row (4 banks x 8 B)
PATH_BYTES = H * FR * D        # 16*32*16 = 8192 (e4m3 1 byte)
HEAD_STRIDE = FR * D           # 512
FRAME_STRIDE = D               # 16


def main():
    bak = MM + ".pre_kvbuf.bak"
    if os.path.exists(bak):
        shutil.copy2(bak, MM)
    else:
        shutil.copy2(MM, bak)
    mm = json.load(open(MM))

    paths = ["K_re", "K_im", "V_re", "V_im"]
    descale = {"K_re": -5, "K_im": -5, "V_re": -5, "V_im": -5}   # corrected: shared k=-5
    regions = {}
    for i, p in enumerate(paths):
        off = i * PATH_BYTES
        regions[p] = {
            "byte_offset": off, "bytes": PATH_BYTES,
            "row_start": off // ROW, "row_end": off // ROW + PATH_BYTES // ROW - 1,
            "dtype": "e4m3", "descale_pow2": descale[p],
            "tensor": p[0], "pass": "A_real" if p.endswith("_re") else "B_imag",
        }
    kv = {
        "buffer": "KVBUF", "banks": 4, "width_bits": 64, "depth": 1024,
        "capacity_bytes": 32768, "row_bytes": ROW, "dtype": "e4m3 (1 byte)",
        "fits": "4 paths x 8192 B = 32768 B = capacity (zero slack)",
        "regions": regions,
        "element_addressing": {
            "formula": "byte = path_base + head*512 + frame_slot*16 + dim",
            "head_stride_bytes": HEAD_STRIDE, "frame_stride_bytes": FRAME_STRIDE, "dim_stride_bytes": 1,
            "heads": H, "ring_frames": FR, "head_dim": D,
        },
        "pass_to_path": {
            "passA_x_real": {"writes": ["K_re", "V_re"], "QK_reads": "K_re", "scoreV_reads": "V_re"},
            "passB_x_imag": {"writes": ["K_im", "V_im"], "QK_reads": "K_im", "scoreV_reads": "V_im"},
        },
        "ring": {
            "frames": FR,
            "write": "on frame t: K/V_t -> ring slot = ptr (all heads share ptr)",
            "read": "window entry local_idx d -> physical frame_slot = (ptr - d) mod 32",
            "local_idx_0": "current/newest frame (= ptr)",
            "advance": "ptr = (ptr + 1) mod 32 AFTER frame t's QK + scoreV complete",
        },
        "descale_on_read": {"K": "x 2^-5", "V": "x 2^-5", "pass_independent": True,
                            "note": "k_Q=k_K=k_V=-5 verified same in both passes; only K/V DATA is pass-split"},
        "init": "kvbuf_init_zero.bin (32768 B all-zero); no resize; partitioning = the 4 regions above",
    }
    mm["kvbuf"] = kv
    json.dump(mm, open(MM, "w"), indent=1)

    # checks
    assert PATH_BYTES * 4 == 32768
    assert os.path.getsize(KVZERO) == 32768
    print("[patch] KVBUF 4-path pinned into memory_map.json['kvbuf']")
    for p in paths:
        r = regions[p]
        print(f"  {p}: rows {r['row_start']}..{r['row_end']}  byte {r['byte_offset']}..{r['byte_offset']+r['bytes']-1}  ({r['bytes']} B)  descale x2^{r['descale_pow2']}")
    print(f"  total {PATH_BYTES*4} B == KVBUF 32768 B (zero slack); kvbuf_init_zero.bin = {os.path.getsize(KVZERO)} B (no resize)")
    print("  addr: byte = path_base + head*512 + ((ptr - local_idx) mod 32)*16 + dim")


if __name__ == "__main__":
    main()
