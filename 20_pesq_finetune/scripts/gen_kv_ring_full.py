#!/usr/bin/env python3
"""Full KV ring-order e4m3 golden: all 16 heads x both passes (4 paths).

ring/slot order = frame order in the t*=351 window (slot 0 = frame 320 ... slot 31 =
frame 351). Pass A(real)->K_re/V_re, Pass B(imag)->K_im/V_im. e4m3 codes (KVBUF storage):
  K code = e4m3_encode(K_value * 2^7)   (descale x2^-7 on read)
  V code = e4m3_encode(V_value * 2^6)   (descale x2^-6 on read)

Out: hw_patterns_final/kv_ring_reference/{K_re,K_im,V_re,V_im}/head_HH_ring_e4m3.{hex,u8}
     (32 slots x 16 dim each). Lays out exactly as the KVBUF path region expects
     (byte = head*512 + slot*16 + dim within the path).
"""
from __future__ import annotations
import os
import numpy as np, torch

MIO = "/raid/R12K41024/19_algo_final/hw_patterns_final/mha_module_io"
OUT = "/raid/R12K41024/19_algo_final/hw_patterns_final/kv_ring_reference"
PASS_SRC = {"real_pass": "_re", "imag_pass": "_im"}
DESC = {"K": -5, "V": -5}   # deploy shared k_Q=k_K=k_V=-5 (corrected from -7/-6)


def f16(p):
    return np.fromfile(p + ".bin", np.float16).astype(np.float64)


def e4_code(v, k):
    return torch.tensor(np.asarray(v) * (2.0 ** (-k)), dtype=torch.float64).to(torch.float8_e4m3fn).view(torch.uint8).numpy()


def main():
    nbytes = {}
    for pass_name, suf in PASS_SRC.items():
        for src in ("K", "V"):
            path = f"{src}{suf}"          # K_re/K_im/V_re/V_im
            d = os.path.join(OUT, path)
            os.makedirs(d, exist_ok=True)
            allheads = []
            for h in range(16):
                blk = f16(f"{MIO}/{pass_name}/02_qk/head_{h:02d}/{src}_block_32x16_fp16").reshape(32, 16)  # slot(frame) order
                code = e4_code(blk, DESC[src]).astype(np.uint8)   # [32,16]
                with open(os.path.join(d, f"head_{h:02d}_ring_e4m3.hex"), "w") as f:
                    for b in code.reshape(-1):
                        f.write(f"{int(b):02X}\n")
                code.tofile(os.path.join(d, f"head_{h:02d}_ring_e4m3.u8.bin"))
                allheads.append(code)
            # one packed per-path file in {head][slot][dim} order = the KVBUF region layout
            packed = np.stack(allheads).reshape(-1).astype(np.uint8)   # [16*32*16]=8192
            packed.tofile(os.path.join(d, "path_ring_e4m3.u8.bin"))
            with open(os.path.join(d, "path_ring_e4m3.hex"), "w") as f:
                for b in packed:
                    f.write(f"{int(b):02X}\n")
            nbytes[path] = packed.size
            # roundtrip check: dequant code*2^k == fp16 source
            chk = torch.tensor(np.stack(allheads).reshape(16, 32, 16), dtype=torch.uint8).view(torch.float8_e4m3fn).to(torch.float64).numpy() * (2.0 ** DESC[src])
            src_all = np.stack([f16(f"{MIO}/{pass_name}/02_qk/head_{h:02d}/{src}_block_32x16_fp16").reshape(32, 16) for h in range(16)])
            print(f"  {path}: 16 heads x 32 slot x 16 dim = {packed.size} B (=8192=path region)  dequant==fp16 src: {np.array_equal(chk, src_all)}")
    print(f"[gen] wrote {OUT}/{{K_re,K_im,V_re,V_im}}/  (per-head + path_ring packed; layout byte=head*512+slot*16+dim)")
    print(f"  path bytes: {nbytes}  (each == 8192 = KVBUF path region)")


if __name__ == "__main__":
    main()
