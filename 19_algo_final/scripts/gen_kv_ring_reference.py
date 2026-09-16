#!/usr/bin/env python3
"""Multi-frame KV ring reference (Consult item 4): verify the streaming-ring
pointer + local_idx->physical mapping picks the right historical frames.

For the t*=351 window (32 frames 320..351), with ptr advancing 0..31:
  frame f -> ring_slot = f-320 (write order) ; at query 351 (ptr=31):
  local_idx d -> physical ring_slot (ptr-d) mod 32 = 31-d -> frame 351-d.
Verifies K_block (frame/ring order 320..351) [31-d] == qstar_Kwindow (local_idx order)[d],
and dumps K/V e4m3 ring codes + the ring map for a stream-write-then-window-read test.
"""
from __future__ import annotations
import os, json
import numpy as np, torch

MIO = "/raid/R12K41024/19_algo_final/hw_patterns_final/mha_module_io"
OUT = "/raid/R12K41024/19_algo_final/hw_patterns_final/kv_ring_reference"
T0, BLK, QSTAR = 320, 32, 351
KDESC, VDESC = -5, -5   # corrected: deploy shared k=-5


def f16(p):
    return np.fromfile(p + ".bin", np.float16).astype(np.float64)


def e4_code(v, k):  # value = e4m3 * 2^k  ->  code = e4m3_encode(value * 2^-k)
    return torch.tensor(np.asarray(v) * (2.0 ** (-k)), dtype=torch.float64).to(torch.float8_e4m3fn).view(torch.uint8).numpy()


def write_hex(path, u8):
    with open(path, "w") as f:
        for b in np.asarray(u8).reshape(-1):
            f.write(f"{int(b):02X}\n")


def main():
    os.makedirs(OUT, exist_ok=True)
    ptr = QSTAR % BLK            # 31
    ring_map = []
    for f in range(T0, QSTAR + 1):
        local_idx = QSTAR - f
        ring_slot = (ptr - local_idx) % BLK
        ring_map.append({"frame": f, "write_slot": f - T0, "local_idx": local_idx, "read_slot_at_q351": ring_slot})
    # self-consistency: read_slot == write_slot for every frame (the ring is filled in frame order)
    consistent = all(r["write_slot"] == r["read_slot_at_q351"] for r in ring_map)

    # numeric check across all heads (real pass): K_block[31-d] == qstar_Kwindow[d]
    ok = True
    for h in range(16):
        kb = f16(f"{MIO}/real_pass/02_qk/head_{h:02d}/K_block_32x16_localidx_fp16") if False else \
             f16(f"{MIO}/real_pass/02_qk/head_{h:02d}/K_block_32x16_fp16").reshape(32, 16)   # frame order 320..351
        kw = f16(f"{MIO}/real_pass/02_qk/head_{h:02d}/qstar_Kwindow_32x16_localidx_fp16").reshape(32, 16)  # local_idx order
        # qstar_Kwindow[d] should equal K_block[31-d]
        if not np.array_equal(kw, kb[::-1]):
            ok = False
    print(f"ring map self-consistent (write_slot==read_slot): {consistent}")
    print(f"K_block[31-d]==qstar_Kwindow[d] across all 16 heads: {ok}  -> (ptr-local_idx)%32 picks the right frame")

    # dump K_re/V_re e4m3 ring codes (head 0), frame/ring order (slot 0=frame320 .. 31=frame351)
    for nm, desc in [("K_re", KDESC), ("V_re", VDESC)]:
        src = "K" if nm.startswith("K") else "V"
        blk = f16(f"{MIO}/real_pass/02_qk/head_00/{src}_block_32x16_fp16").reshape(32, 16)
        code = e4_code(blk, desc)   # [32 slots,16 dim] e4m3 codes, slot = frame-320
        write_hex(os.path.join(OUT, f"{nm}_head00_ring_e4m3.hex"), code)
        code.astype(np.uint8).tofile(os.path.join(OUT, f"{nm}_head00_ring_e4m3.u8.bin"))

    json.dump({"window": {"query_frame": QSTAR, "frames": [T0, QSTAR], "ring_frames": BLK, "ptr_at_query": ptr},
               "ring_map_frame_slot_localidx": ring_map,
               "read_formula": "physical_slot = (ptr - local_idx) mod 32 ; local_idx 0 = newest = ptr",
               "write": "frame f written to slot (f mod 32); ptr advances after each frame",
               "verified": {"ring_self_consistent": consistent,
                            "K_block[31-d]==qstar_Kwindow[d]_all_heads": ok,
                            "note": "K_block_32x16 = frame/ring order (slot0=f320..slot31=f351); "
                                    "qstar_Kwindow_localidx = local_idx order (d0=f351..d31=f320)"},
               "dumped": "K_re/V_re head00 e4m3 ring codes (slot order); all 16 heads' fp16 K/V in mha_module_io/02_qk/",
               "expected_QK": "mha_module_io/real_pass/02_qk/head_HH/qstar_scores_row32_localidx_postscale_* "
                              "(QK at query 351 over the window, local_idx order)"},
              open(os.path.join(OUT, "kv_ring_reference.json"), "w"), indent=1)
    print(f"[gen] wrote {OUT}")


if __name__ == "__main__":
    main()
