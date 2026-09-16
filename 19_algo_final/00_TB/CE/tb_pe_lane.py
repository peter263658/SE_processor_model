#!/usr/bin/env python3
"""
tb_pe_lane.py — golden generator for pe_lane.sv (mixed-precision zero-mult PE lane).

Operand sources (REAL export codes where they exist, in
10_ALGO/hw_patterns_final/tb_pattern/):
    mode 0 W4A8       : INT4 weight codes (w_int4.packed.bin) x FP8 e4m3 act codes
                        + an EXHAUSTIVE 15x256 sweep
    mode 1 LMUL_FP8   : FP8 outlier weight codes (w_fp8.e4m3.bin) x FP8 e4m3 act codes
    mode 2 LMUL_ATTN  : FP16 x FP16   -- synthesised (no fp16 operand dump in tb_pattern)
    mode 3 W4A16      : INT4 weight codes (real) x FP16 (synthesised)

Golden by mode:
    INT4 modes (0,3): INDEPENDENT ground truth = round-to-nearest-even fp16(q * y)
                      via struct '>e' (decode + multiply + fp16 cast). Additionally
                      cross-checked at gen time against int4_product_bits(), the
                      bit-level mirror of pe_lane's RNE datapath (catches an RTL spec
                      transcription bug before simulation).
    L-Mul modes (1,2): mirror of complexLayers.py::_lmul_multiply_bit_exact (same
                      integer-code datapath as the RTL). Validate against torch
                      separately (see tb_pe_lane_xcheck_lmul.py).

Output: pattern_pe_lane/stim.dat  --  "<mode> <op_x> <op_y> <golden>" (hex/line),
read by tb_pe_lane.sv via $fscanf.
"""
import os
import struct
import random

HERE = os.path.dirname(os.path.abspath(__file__))
PAT  = os.path.join(HERE, "..", "..", "10_ALGO", "hw_patterns_final", "tb_pattern")
OUT  = os.path.join(HERE, "pattern_pe_lane")

MODE_W4A8, MODE_LMUL_FP8, MODE_LMUL_ATTN, MODE_W4A16 = 0, 1, 2, 3
QNAN15, INF15 = 0x7E00, 0x7C00

# INT4 CSD table : mag -> (shA, shB, useB, sub) ; mirrors pe_int4_csd
CSD = {0:(0,0,0,0), 1:(0,0,0,0), 2:(1,0,0,0), 3:(1,0,1,0),
       4:(2,0,0,0), 5:(2,0,1,0), 6:(2,1,1,0), 7:(3,0,1,1)}


# ---------------- mirror of pe_fp_unpack (frexp convention) ----------------------
def unpack_float(bits, EXP_W, FRAC_W, HAS_INF):
    s = (bits >> (EXP_W + FRAC_W)) & 1
    e = (bits >> FRAC_W) & ((1 << EXP_W) - 1)
    f = bits & ((1 << FRAC_W) - 1)
    BIAS = (1 << (EXP_W - 1)) - 1
    e_all1 = (e == (1 << EXP_W) - 1); e_zero = (e == 0)
    f_all1 = (f == (1 << FRAC_W) - 1); f_zero = (f == 0)
    zero    = e_zero and f_zero
    special = e_all1 if HAS_INF else (e_all1 and f_all1)
    if e_zero:
        p = 0
        for i in range(FRAC_W):
            if (f >> i) & 1:
                p = i
        sig = (f << (FRAC_W - p)) & ((1 << (FRAC_W + 1)) - 1)
        exp = 1 - BIAS - 2 * FRAC_W + p
    else:
        sig = (1 << FRAC_W) | f
        exp = e - BIAS - FRAC_W
    return s, exp, sig, zero, special


# ---------------- bit-level mirror of pe_lane INT4 path (RNE) ---------------------
def int4_product_bits(q_raw, y_bits, Y_EXP, Y_FRAC, Y_INF):
    s_y, exp_y, sig_y, zero_y, special_y = unpack_float(y_bits, Y_EXP, Y_FRAC, Y_INF)
    signed = q_raw - 16 if q_raw >= 8 else q_raw
    out_s  = (1 if signed < 0 else 0) ^ s_y
    mag    = abs(signed)
    if special_y:
        return (out_s << 15) | QNAN15
    if mag == 0 or zero_y:
        return (out_s << 15)
    shA, shB, useB, sub = CSD[mag]
    A = sig_y << shA
    B = (sig_y << shB) if useB else 0
    M = (A - B) if sub else (A + B)
    L = M.bit_length() - 1
    expf = exp_y + L + 15
    if L > 10:                                  # round-to-nearest-even
        sh = L - 10
        kept = M >> sh
        g  = (M >> (sh - 1)) & 1
        st = (M & ((1 << (sh - 1)) - 1)) != 0
        if g and (st or (kept & 1)):
            kept += 1
            if kept >> 11:
                kept >>= 1; expf += 1
        frac = kept & 0x3FF
    else:
        frac = (M << (10 - L)) & 0x3FF
    if expf <= 0:
        return (out_s << 15)
    if expf >= 31:
        return (out_s << 15) | INF15
    return (out_s << 15) | ((expf & 0x1F) << 10) | frac


# ---------------- mirror of L-Mul (complexLayers._lmul_multiply_bit_exact) --------
def lmul_offset(m):
    return m if m <= 3 else (3 if m == 4 else 4)

def lmul_product_bits(x_bits, y_bits, EXP_W, FRAC_W, HAS_INF, m):
    s_x, ex, sig_x, zx, spx = unpack_float(x_bits, EXP_W, FRAC_W, HAS_INF)
    s_y, ey, sig_y, zy, spy = unpack_float(y_bits, EXP_W, FRAC_W, HAS_INF)
    sign = s_x ^ s_y
    if zx or zy or spx or spy:
        return 0x0000
    exX = ex + FRAC_W; exY = ey + FRAC_W
    fracx = sig_x & ((1 << FRAC_W) - 1); fracy = sig_y & ((1 << FRAC_W) - 1)
    if FRAC_W >= m:
        fx = fracx >> (FRAC_W - m); fy = fracy >> (FRAC_W - m)
    else:
        fx = fracx << (m - FRAC_W); fy = fracy << (m - FRAC_W)
    scale  = 1 << m
    offset = round(scale * (2.0 ** (-lmul_offset(m))))
    sig_pre = scale + fx + fy + offset
    carry  = 1 if sig_pre >= 2 * scale else 0
    sig_cd = (sig_pre >> 1) if carry else sig_pre
    frac   = (sig_cd - scale) << (10 - m)
    expf   = exX + exY + carry + 15
    if expf <= 0:
        return 0x0000
    if expf >= 31:
        return (sign << 15) | INF15
    return (sign << 15) | ((expf & 0x1F) << 10) | (frac & 0x3FF)


# ---------------- INDEPENDENT INT4 ground truth: fp16(q * y), RNE ----------------
def fp8_e4m3_to_float(raw):
    s = (raw >> 7) & 1; e = (raw >> 3) & 0xF; m = raw & 0x7
    sign = -1.0 if s else 1.0
    if e == 0 and m == 0:   return sign * 0.0
    if e == 0xF and m == 7: return None                       # NaN
    if e == 0:              return sign * m * (2.0 ** -9)      # subnormal
    return sign * (1.0 + m / 8.0) * (2.0 ** (e - 7))          # normal

def decode_operand_float(bits, EXP_W, FRAC_W, HAS_INF):
    if (EXP_W, FRAC_W) == (5, 10):                            # fp16
        s = (bits >> 15) & 1; e = (bits >> 10) & 0x1F; f = bits & 0x3FF
        special = (e == 0x1F); zero = (e == 0 and f == 0)
        v = None if special else struct.unpack(">e", struct.pack(">H", bits & 0xFFFF))[0]
        return s, v, zero, special
    s = (bits >> 7) & 1; e = (bits >> 3) & 0xF; m = bits & 0x7  # fp8 e4m3
    special = (e == 0xF and m == 7); zero = (e == 0 and m == 0)
    return s, fp8_e4m3_to_float(bits), zero, special

def int4_golden_bits(q_raw, y_bits, EXP_W, FRAC_W, HAS_INF):
    s_y, v_y, zero_y, special_y = decode_operand_float(y_bits, EXP_W, FRAC_W, HAS_INF)
    signed = q_raw - 16 if q_raw >= 8 else q_raw
    out_s  = (1 if signed < 0 else 0) ^ s_y
    if special_y:
        return (out_s << 15) | QNAN15
    if signed == 0 or zero_y:
        return (out_s << 15)
    val = signed * v_y
    try:
        b = struct.unpack(">H", struct.pack(">e", val))[0]
    except OverflowError:
        return (out_s << 15) | INF15
    if ((b >> 10) & 0x1F) == 0:                               # fp16 subnormal -> RTL FTZ
        return (out_s << 15)
    return b


# ---------------- export readers / synth ----------------------------------------
def read_bytes(path):
    with open(path, "rb") as fh:
        return list(fh.read())

def unpack_int4_nibbles(byts):
    out = []
    for b in byts:
        for nib in (b & 0xF, (b >> 4) & 0xF):
            if nib != 0x8:                                   # -8 don't-care
                out.append(nib)
    return out

def first_existing(*paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None

def synth_fp16_codes(n, rng):
    codes = []
    for s in (0, 1):
        for e in (10, 12, 13, 14, 15, 16, 17, 18, 20):
            for f in (0, 1, 0x155, 0x200, 0x2AA, 0x3FF, 0x080, 0x37F):
                codes.append((s << 15) | (e << 10) | f)
    while len(codes) < n:
        codes.append((rng.randint(0, 1) << 15) | (rng.randint(6, 24) << 10) | rng.randint(0, 0x3FF))
    return codes[:n]


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = random.Random(20260531)
    vectors = []
    assert_fail = 0

    def emit_int4(mode, q, y, Y_EXP, Y_FRAC, Y_INF, op_y_field):
        nonlocal assert_fail
        g = int4_golden_bits(q, y, Y_EXP, Y_FRAC, Y_INF)          # independent truth
        if int4_product_bits(q, y, Y_EXP, Y_FRAC, Y_INF) != g:    # RTL-spec mirror
            assert_fail += 1
        vectors.append((mode, q & 0xF, op_y_field, g))

    # real FP8 activation codes (mode 0 / 1)
    act_path = first_existing(
        os.path.join(PAT, "activations_fp8", "decoder_model_0_0_pointwise_r_x__0", "input_0.e4m3.u8.bin"),
        os.path.join(PAT, "activations_fp8", "decoder_model_0_0_depthwise_r__0", "input_0.e4m3.u8.bin"))
    act_codes = read_bytes(act_path) if act_path else [rng.randint(0, 255) for _ in range(4096)]

    # ---- mode 0 : EXHAUSTIVE 15x256 (independent golden) ------------------------
    for q in range(16):
        if q == 0b1000:
            continue
        for a in range(256):
            emit_int4(MODE_W4A8, q, a, 4, 3, 0, a & 0xFF)

    # ---- mode 0 : real INT4 weight codes x real FP8 act -------------------------
    int4_path = first_existing(
        os.path.join(PAT, "weights", "encoder.model.4.0.pointwise_r", "w_int4.packed.bin"),
        os.path.join(PAT, "weights", "decoder.model.0.0.depthwise_r", "w_int4.packed.bin"))
    int4_codes = unpack_int4_nibbles(read_bytes(int4_path)) if int4_path else \
                 [c for c in range(16) if c != 8] * 64
    for i in range(min(4000, len(int4_codes))):
        q = int4_codes[i % len(int4_codes)]
        a = act_codes[i % len(act_codes)]
        emit_int4(MODE_W4A8, q, a, 4, 3, 0, a & 0xFF)

    # ---- mode 1 : FP8 outlier weight (real) x FP8 act (real) --------------------
    fp8w_path = first_existing(
        os.path.join(PAT, "weights", "decoder.model.0.0.pointwise_r_x", "w_fp8.e4m3.bin"),
        os.path.join(PAT, "weights", "decoder.model.3.0.pointwise_r_x", "w_fp8.e4m3.bin"))
    fp8w = read_bytes(fp8w_path) if fp8w_path else [rng.randint(0, 255) for _ in range(4096)]
    for i in range(min(4000, len(fp8w))):
        x = fp8w[i % len(fp8w)]; y = act_codes[i % len(act_codes)]
        vectors.append((MODE_LMUL_FP8, x & 0xFF, y & 0xFF, lmul_product_bits(x, y, 4, 3, 0, 4)))

    # ---- mode 2 : FP16 x FP16 L-Mul m=8 (synth) ---------------------------------
    f16 = synth_fp16_codes(400, rng)
    for _ in range(4000):
        x = f16[rng.randrange(len(f16))]; y = f16[rng.randrange(len(f16))]
        vectors.append((MODE_LMUL_ATTN, x & 0xFFFF, y & 0xFFFF, lmul_product_bits(x, y, 5, 10, 1, 8)))

    # ---- mode 3 : INT4 weight (real) x FP16 (synth) -----------------------------
    for _ in range(4000):
        q = int4_codes[rng.randrange(len(int4_codes))]; y = f16[rng.randrange(len(f16))]
        emit_int4(MODE_W4A16, q, y, 5, 10, 1, y & 0xFFFF)

    if assert_fail:
        raise SystemExit(f"[FATAL] INT4 RTL-spec mirror disagrees with independent fp16(q*y) "
                         f"on {assert_fail} vectors — fix pe_lane INT4 datapath")

    with open(os.path.join(OUT, "stim.dat"), "w") as fh:
        for mode, x, y, g in vectors:
            fh.write(f"{mode:01x} {x:04x} {y:04x} {g:04x}\n")

    per = {}
    for m, *_ in vectors:
        per[m] = per.get(m, 0) + 1
    print(f"[tb_pe_lane] wrote {len(vectors)} vectors -> {OUT}/stim.dat  per-mode={per}")
    print(f"             act_src  = {act_path}")
    print(f"             int4_src = {int4_path}")
    print(f"             fp8w_src = {fp8w_path}")
    print(f"             INT4 independent-vs-mirror agree on all {per.get(0,0)+per.get(3,0)} vectors")


if __name__ == "__main__":
    main()
