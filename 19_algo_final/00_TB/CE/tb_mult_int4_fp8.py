#!/usr/bin/env python3
"""
tb_mult_int4_fp8.py — golden generator for tb_mult_int4_fp8.sv

Sweeps all 16 x 256 = 4096 (INT4_W, FP8_A) raw-byte pairs and emits one
FP16 hex per line into pattern_lane_w4a8/golden_fp16.dat.

Self-contained: only depends on numpy (no ml_dtypes), since FP8 E4M3 has
only 256 distinct codes — we decode them manually.

Convention (must match RTL mult_int4_fp8.sv):
    INT4 W : 2's complement, raw 0..15.
             raw 4'b1000 (-8) is don't-care; we still write a placeholder
             (0x0000), the SV TB skips this row.
    FP8 A  : E4M3 (1+4+3, bias=7).
             - (e=0,  m=0)        : +/-0
             - (e=0,  m!=0)       : subnormal, value = (-1)^s * m * 2^-9
             - (e=15, m=7)        : NaN (the *only* NaN encoding)
             - otherwise          : normal, value = (-1)^s * (1 + m/8) * 2^(e-7)
    FP16 O : IEEE 754 half (1+5+10, bias=15).
             - NaN input          -> canonical FP16 qNaN  0x7E00
             - Either operand 0   -> +/-0 with sign = sign_w XOR sign_a
             - Otherwise          -> bit-exact normal (lane is exact)
"""
import os
import math
import numpy as np

CANON_QNAN_FP16 = 0x7E00


def int4_to_signed(raw: int) -> int:
    return raw - 16 if raw >= 8 else raw


def fp8_e4m3_to_float(raw: int):
    """Decode 1 byte of FP8 E4M3.  Returns python float, or math.nan."""
    s = (raw >> 7) & 0x1
    e = (raw >> 3) & 0xF
    m = raw & 0x7
    sign = -1.0 if s else 1.0

    if e == 0 and m == 0:
        return sign * 0.0
    if e == 0xF and m == 0x7:
        return math.nan
    if e == 0:
        # subnormal: value = (-1)^s * m * 2^-9
        return sign * float(m) * (2.0 ** -9)
    # normal: value = (-1)^s * (1 + m/8) * 2^(e-7)
    return sign * (1.0 + m / 8.0) * (2.0 ** (e - 7))


def fp16_bits(x) -> int:
    """Pack a python float as IEEE FP16 raw bits (little-endian)."""
    return int.from_bytes(np.float16(x).tobytes(), "little")


def expected(w_raw: int, a_raw: int) -> int:
    a_val = fp8_e4m3_to_float(a_raw)

    # NaN propagation (canonical qNaN regardless of sign)
    if isinstance(a_val, float) and math.isnan(a_val):
        return CANON_QNAN_FP16

    # don't-care row: SV TB skips, value irrelevant
    if w_raw == 0b1000:
        return 0x0000

    w_signed = int4_to_signed(w_raw)
    sw = (w_raw >> 3) & 0x1
    sa = (a_raw >> 7) & 0x1

    # signed-zero output if either operand is zero
    if w_signed == 0 or a_val == 0.0:
        return (sw ^ sa) << 15

    return fp16_bits(float(w_signed) * a_val)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(here, "pattern_lane_w4a8")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "golden_fp16.dat")

    n = 0
    with open(out_path, "w") as f:
        for w in range(16):
            for a in range(256):
                f.write(f"{expected(w, a):04x}\n")
                n += 1

    print(f"[tb_mult_int4_fp8] wrote {n} entries -> {out_path}")


if __name__ == "__main__":
    main()
