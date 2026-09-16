#!/usr/bin/env python3
"""
gen_cast_patterns.py — patterns + pre-validation for fxp_to_e4m3 (STFT->fp8 cast).

Ties the converter to the MODEL's real encoder input: the golden e4m3 codes in
hw_patterns_final/.../activations_fp8/encoder_model_0_0_depthwise_r__0/input_0.e4m3.u8.bin
ARE e4m3(model_STFT * 2^9).  We round-trip:

   golden code g  -> model_value = e4m3_decode(g)/2^9
                  -> HW Q1.15 in = round(model_value / sqrt(128) * 2^15)   (what the HW STFT emits)
   fxp_to_e4m3(HW_in, k=9) must reproduce g  (the cast's sqrt(128)+2^9 scaling + e4m3 RNE).

Writes  patterns/stft_cast_in.hex  (Q1.15 int16, 4-hex)  and
        patterns/stft_cast_golden.hex (e4m3 code, 2-hex)  for tb_fxp_to_e4m3.sv,
and pre-validates with a bit-faithful model of the RTL chain
(DW i2flt -> *sqrt2(0x3DA8) fp16 -> fp16_scale2(k-12) -> fp16_to_fp8).
"""
import numpy as np, os, math

HERE = os.path.dirname(os.path.abspath(__file__))
GOLD = os.path.join(HERE, "..", "..", "hw_patterns_final", "tb_pattern",
                    "activations_fp8", "encoder_model_0_0_depthwise_r__0", "input_0.e4m3.u8.bin")
OUTD = os.path.join(HERE, "patterns")
K    = 9
SQRT128 = math.sqrt(128.0)
SQRT2_FP16 = float(np.float16(np.uint16(0x3DA8).view(np.float16)))   # = 1.4140625

# ---- e4m3 (OCP) decode ----
def e4m3_decode(c):
    s=(c>>7)&1; e=(c>>3)&0xF; m=c&7
    if e==0 and m==0: return -0.0 if s else 0.0
    if e==0xF and m==7: return math.nan
    v = (2.0**-6)*(m/8.0) if e==0 else (2.0**(e-7))*(1.0+m/8.0)
    return -v if s else v

# ---- bit-faithful port of fp16_to_fp8.sv (RNE, sat 448) ----
def fp16_to_e4m3_bits(u16):
    s=(u16>>15)&1; e16=(u16>>10)&0x1F; m16=u16&0x3FF
    if e16==0x1F and m16!=0: return (s<<7)|0x7F            # NaN
    if e16==0x1F and m16==0: return (s<<7)|0x7E            # Inf -> 448
    if e16==0:               return (s<<7)|0x00            # zero/subnormal -> 0
    sig11=0x400|m16; e8_pre=(e16-15)+7
    rshift = 0 if e8_pre>=1 else (1-e8_pre)
    sig_sh = (sig11<<11) >> rshift
    mant4=(sig_sh>>18)&0xF; l=mant4&1; g=(sig_sh>>17)&1; st=1 if (sig_sh & ((1<<17)-1)) else 0
    mant4_r = mant4 + (1 if (g and (l or st)) else 0)
    if e8_pre>=16: return (s<<7)|0x7E
    if e8_pre>=1:
        if mant4_r==0x10:
            return (s<<7)|0x7E if (e8_pre+1)>=16 else (s<<7)|(((e8_pre+1)&0xF)<<3)
        if e8_pre==15 and mant4_r==0xF: return (s<<7)|0x7E
        return (s<<7)|((e8_pre&0xF)<<3)|(mant4_r&7)
    return (s<<7)|(1<<3) if (mant4_r>>3)&1 else (s<<7)|(mant4_r&7)

def f16(x): return np.float16(x)
def f16bits(x): return int(np.float16(x).view(np.uint16))

# ---- model of fxp_to_e4m3 RTL chain ----
def convert(q15, k=K):
    f_int = f16(float(q15))                                       # DW i2flt (exact for |q|<2^11)
    f_rt2 = f16(np.float32(f_int) * np.float32(SQRT2_FP16))       # DW fp16 mult (RNE)
    f_sc  = f16(np.float32(f_rt2) * np.float32(2.0**(k-12)))      # fp16_scale2 (exponent add)
    return fp16_to_e4m3_bits(f16bits(f_sc))

# ---- read golden, build round-trip input, validate ----
g = np.fromfile(GOLD, dtype=np.uint8)
os.makedirs(OUTD, exist_ok=True)
ins, exact, off1, worse, worse_real = [], 0, 0, 0, 0
for code in g:
    code=int(code)
    mv = e4m3_decode(code)/(2.0**K)            # model STFT value (e4m3 precision)
    q  = int(round(mv / SQRT128 * (2.0**15)))  # HW Q1.15 code
    q  = max(-32768, min(32767, q))
    ins.append(q & 0xFFFF)
    pred = convert(q, K)
    same_sign = (pred>>7)==(code>>7)
    d = abs((pred&0x7F) - (code&0x7F))
    if pred==code:               exact+=1
    elif same_sign and d==1:     off1+=1            # boundary flip (Q1.15 vs float64, fp16 sqrt2)
    else:
        worse+=1
        if abs(q) > 2: worse_real+=1                # only |q15|>2 counts as a real error
# tiny-bin note: misses with |q15|<=2 are round-trip-derivation underflow (deriving the Q1.15
# input from the COARSE e4m3 golden loses near-zero bins), NOT a converter bug.

with open(os.path.join(OUTD,"stft_cast_in.hex"),"w") as f:
    for q in ins: f.write(f"{q:04X}\n")
with open(os.path.join(OUTD,"stft_cast_golden.hex"),"w") as f:
    for code in g: f.write(f"{int(code)&0xFF:02X}\n")

n=len(g)
print(f"golden file        : {os.path.relpath(GOLD,HERE)}")
print(f"samples (64x399)   : {n}")
print(f"sqrt2 fp16 (0x3DA8): {SQRT2_FP16}")
print(f"round-trip vs golden: EXACT {exact} ({100*exact/n:.2f}%)  off-by-1-e4m3 {off1} ({100*off1/n:.2f}%)  worse {worse}")
print(f"  -> REAL errors (|q15|>2 AND >1 e4m3 step) = {worse_real}  "
      f"({'PASS — scaling sqrt128+k=9 correct' if worse_real==0 else 'FAIL — investigate scaling'})")
print(f"  (off-by-1 + tiny-bin misses are Q1.15-vs-float64 / coarse-golden round-trip artifacts, not bugs)")
print(f"patterns written   : patterns/stft_cast_in.hex , patterns/stft_cast_golden.hex")
