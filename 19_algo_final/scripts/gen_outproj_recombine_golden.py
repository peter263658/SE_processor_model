#!/usr/bin/env python3
"""out_proj β (bias) golden + recombine golden as hex (Consult §C-9/§C-10).

No model run needed — uses the exported MHA out_proj bias + the layer_io goldens.

Emits to hw_patterns_final/mha_out_proj_recombine_golden/:
  beta.fp16.{hex,bin}            : MHA out_proj bias b_o [256] fp16 (SAME for both
                                    passes; out_pass = fp16_add_sat(oacc_pass, b_o)).
  recombine worked example (t*=351): a,b,(a-b),(a+b) hex (a=real-pass, b=imag-pass
                                    post-bias out_proj fp16).
  outproj_in_re_code.u8 / _im    : recombine result e4m3-cast at i_cast_k=+3
                                    (output_proj input k=-3), all 399 frames x 256.
"""
from __future__ import annotations
import os, json
import numpy as np
import torch

HW = "/raid/R12K41024/19_algo_final/hw_patterns_final"
W = os.path.join(HW, "tb_pattern", "weights", "mattn.mattn.MultiheadAttention_re", "out_proj")
L = os.path.join(HW, "debug_reference", "layer_io_fp32", "layers", "mattn_output_proj__0")
MIO = os.path.join(HW, "mha_module_io")
OUT = os.path.join(HW, "mha_out_proj_recombine_golden")
OUTPROJ_IN_CAST_K = 3   # i_cast_k feeding output_proj (consumer block-float k = -3)


def hx(p):
    return np.array([int(x, 16) for x in open(p).read().split()], np.uint32).view(np.float32).astype(np.float64)


def f32bin(p):
    return np.fromfile(p + ".bin", np.float32).astype(np.float64)


def fp16_hex(v):
    return torch.tensor(np.asarray(v, np.float64), dtype=torch.float64).to(torch.float16).view(torch.uint16).numpy()


def e4_code(v, cast_k):
    return torch.tensor(np.asarray(v, np.float64) * (2.0 ** cast_k), dtype=torch.float64).to(torch.float8_e4m3fn).view(torch.uint8).numpy()


def write_hex(path, words, width=4):
    fmt = "{:0%dX}" % width
    with open(path, "w") as f:
        for w in np.asarray(words).reshape(-1):
            f.write(fmt.format(int(w) & ((1 << (4 * width)) - 1)) + "\n")


def main():
    os.makedirs(OUT, exist_ok=True)
    # ---- β (bias) golden ----
    bo = np.fromfile(os.path.join(W, "bias.fp16.bin"), np.float16).astype(np.float64)  # [256]
    np.asarray(bo, np.float16).tofile(os.path.join(OUT, "beta.fp16.bin"))
    write_hex(os.path.join(OUT, "beta.fp16.hex"), fp16_hex(bo))
    print(f"[beta] MHA out_proj bias b_o [256] fp16: |max|={np.abs(bo).max():.4f} -> beta.fp16.hex (both passes use it)")

    # ---- recombine worked example (t*=351), a/b = post-bias out_proj fp16 ----
    a = f32bin(os.path.join(MIO, "real_pass", "05_out_proj", "output256_fp32"))  # a = M(x.real)
    b = f32bin(os.path.join(MIO, "imag_pass", "05_out_proj", "output256_fp32"))  # b = M(x.imag)
    out_re = a - b; out_im = a + b
    for nm, v in [("a_real_pass", a), ("b_imag_pass", b), ("recombine_re_a_minus_b", out_re), ("recombine_im_a_plus_b", out_im)]:
        write_hex(os.path.join(OUT, f"t351_{nm}.fp16.hex"), fp16_hex(v))
    print(f"[recombine] t*=351 worked: a,b,(a-b),(a+b) fp16 hex. out_re=a-b, out_im=a+b.")

    # ---- recombine result -> output_proj input e4m3 codes (i_cast_k=+3), all frames ----
    opi_re = hx(os.path.join(L, "input_0.real.fp32.hex")).reshape(399, 256)  # already = recombine (a-b), e4m3 k=-3 grid
    opi_im = hx(os.path.join(L, "input_0.imag.fp32.hex")).reshape(399, 256)
    cre = e4_code(opi_re, OUTPROJ_IN_CAST_K); cim = e4_code(opi_im, OUTPROJ_IN_CAST_K)
    cre.astype(np.uint8).tofile(os.path.join(OUT, "outproj_in_re_code.u8.bin"))
    cim.astype(np.uint8).tofile(os.path.join(OUT, "outproj_in_im_code.u8.bin"))
    write_hex(os.path.join(OUT, "outproj_in_re_code.hex"), cre, width=2)
    write_hex(os.path.join(OUT, "outproj_in_im_code.hex"), cim, width=2)
    # self-check: dequant code*2^-3 == opi (recombine is on the e4m3 k=-3 grid)
    deq = torch.tensor(cre, dtype=torch.uint8).view(torch.float8_e4m3fn).to(torch.float64).numpy() * (2.0 ** -OUTPROJ_IN_CAST_K)
    print(f"[outproj_in] recombine->output_proj boundary e4m3 codes (i_cast_k=+3), [399,256] re/im. "
          f"dequant==golden: {np.max(np.abs(deq - opi_re)):.2e}")

    json.dump({"beta": "MHA out_proj bias b_o[256] fp16; out_pass=fp16_add_sat(oacc_pass,b_o); SAME both passes; "
                       "plain add. Net: cancels in real (a-b), doubles in imag (a+b=...+2*b_o) -- automatic.",
               "recombine": "out_re=a-b, out_im=a+b; a=real-pass, b=imag-pass post-bias out_proj fp16",
               "outproj_input_cast": f"recombine (fp16) -> e4m3 at i_cast_k=+{OUTPROJ_IN_CAST_K} (output_proj input block-float k=-3)",
               "output_proj_output_cast_k": -1},
              open(os.path.join(OUT, "meta.json"), "w"), indent=1)
    print(f"[gen] wrote {OUT}")


if __name__ == "__main__":
    main()
