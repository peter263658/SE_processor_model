import numpy as np
import os

# --- 設定參數 ---
WIDTH = 16
FRAC_BITS_WIN = 15  # Q1.15
FRAC_BITS_IN  = 14  # Q2.14 (Input)
FRAC_BITS_OUT = 15  # Q1.15 (Output)

# 計算硬體對應參數
HARDWARE_SHIFT = FRAC_BITS_WIN + FRAC_BITS_IN - FRAC_BITS_OUT
ROUND_BIAS = 1 << (HARDWARE_SHIFT - 1)

N_FFT = 128
KEEP_BINS = 64 # 硬體輸出 64 bins = FFT bins 1..64（去掉 DC bin0、保留 Nyquist bin64），對齊 model 的 Stft.narrow(start=1)
NUM_TEST_VECTORS = 4* N_FFT 
OUTPUT_DIR = "./patterns"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- 工具函式 ---
def float_to_fixed(x, frac_bits):
    scaling = 1 << frac_bits
    x_scaled = np.round(x * scaling)
    min_val = -(1 << (WIDTH - 1))
    max_val = (1 << (WIDTH - 1)) - 1
    x_int = np.clip(x_scaled, min_val, max_val).astype(np.int32)
    return x_int

def int_to_hex(x):
    return f"{(x & 0xFFFF):04X}"

def hardware_mul_round(a, b, shift, bias):
    prod = a.astype(np.int64) * b.astype(np.int64)
    prod_rounded = prod + bias
    output = prod_rounded >> shift
    min_val = -(1 << (WIDTH - 1))
    max_val = (1 << (WIDTH - 1)) - 1
    output = np.clip(output, min_val, max_val).astype(np.int32)
    return output

# --- e4m3 cast model (matches fxp_to_e4m3.sv: i2flt -> *sqrt2 -> scale2(k-12) -> fp16_to_fp8) ---
K_ENC      = 9                                                       # encoder.0.0.depthwise input k
SQRT2_FP16 = float(np.uint16(0x3DA8).view(np.float16))              # 1.4140625
def _f16(x):     return np.float16(x)
def _f16b(x):    return int(np.float16(x).view(np.uint16))
def _fp16_to_e4m3(u16):                                             # port of fp16_to_fp8.sv
    s=(u16>>15)&1; e=(u16>>10)&0x1F; m=u16&0x3FF
    if e==0x1F and m!=0: return (s<<7)|0x7F
    if e==0x1F and m==0: return (s<<7)|0x7E
    if e==0:             return (s<<7)|0x00
    sig=0x400|m; ep=(e-15)+7; rsh=0 if ep>=1 else (1-ep)
    sh=(sig<<11)>>rsh; mant=(sh>>18)&0xF; l=mant&1; g=(sh>>17)&1; st=1 if (sh&((1<<17)-1)) else 0
    mr=mant+(1 if (g and (l or st)) else 0)
    if ep>=16: return (s<<7)|0x7E
    if ep>=1:
        if mr==0x10: return (s<<7)|0x7E if (ep+1)>=16 else (s<<7)|(((ep+1)&0xF)<<3)
        if ep==15 and mr==0xF: return (s<<7)|0x7E
        return (s<<7)|((ep&0xF)<<3)|(mr&7)
    return (s<<7)|(1<<3) if (mr>>3)&1 else (s<<7)|(mr&7)
def fxp_to_e4m3(q15, k=K_ENC):                                      # Q1.15 HW STFT code -> e4m3 code
    fi  = _f16(float(q15))                                          # int -> fp16
    fr2 = _f16(np.float32(fi)*np.float32(SQRT2_FP16))               # * sqrt(2)
    fsc = _f16(np.float32(fr2)*np.float32(2.0**(k-12)))             # * 2^(k-12)  (folds sqrt128/2^15)
    return _fp16_to_e4m3(_f16b(fsc))

# --- 1. 生成輸入數據 ---
np.random.seed(42)
input_re_float = np.random.uniform(-0.9, 0.9, NUM_TEST_VECTORS)
input_im_float = np.random.uniform(-0.9, 0.9, NUM_TEST_VECTORS)
input_re_fixed = float_to_fixed(input_re_float, FRAC_BITS_IN)
input_im_fixed = float_to_fixed(input_im_float, FRAC_BITS_IN)

print(f"Generating Input Vectors...")
with open(f"{OUTPUT_DIR}/input_re.hex", "w") as f_re, open(f"{OUTPUT_DIR}/input_im.hex", "w") as f_im:
    for r, i in zip(input_re_fixed, input_im_fixed):
        f_re.write(f"{int_to_hex(r)}\n")
        f_im.write(f"{int_to_hex(i)}\n")

# --- 2. Window 處理 (模擬硬體) ---
window_float = np.hamming(N_FFT)
window_fixed = float_to_fixed(window_float, FRAC_BITS_WIN)
window_tiled = np.tile(window_fixed, NUM_TEST_VECTORS // N_FFT)

golden_win_re = hardware_mul_round(input_re_fixed, window_tiled, HARDWARE_SHIFT, ROUND_BIAS)
golden_win_im = hardware_mul_round(input_im_fixed, window_tiled, HARDWARE_SHIFT, ROUND_BIAS)

# --- 3. 計算 FFT Golden (自然順序, 64點) ---
golden_fft_re_final = []
golden_fft_im_final = []
scale_out = 1.0 / (1 << FRAC_BITS_OUT)

for frame in range(NUM_TEST_VECTORS // N_FFT):
    start_idx = frame * N_FFT
    end_idx = start_idx + N_FFT
    
    # 轉浮點做 FFT
    frame_re = golden_win_re[start_idx:end_idx] * scale_out
    frame_im = golden_win_im[start_idx:end_idx] * scale_out
    complex_input = frame_re + 1j * frame_im
    
    # FFT & 硬體 Scaling (除以 128)
    fft_out = np.fft.fft(complex_input) / N_FFT 
    
    # *** 關鍵: 取 bins 1..64 (去 DC、留 Nyquist)，對齊 RTL Bit_Reorder 與 model narrow(start=1) ***
    fft_out = fft_out[1:KEEP_BINS+1]
    
    # 轉定點
    fft_out_re_fixed = float_to_fixed(np.real(fft_out), FRAC_BITS_OUT)
    fft_out_im_fixed = float_to_fixed(np.imag(fft_out), FRAC_BITS_OUT)
    
    golden_fft_re_final.extend(fft_out_re_fixed)
    golden_fft_im_final.extend(fft_out_im_fixed)

print(f"Generating Golden FFT Output (Natural Order, bins 1..64: DC removed, Nyquist kept)...")
with open(f"{OUTPUT_DIR}/golden_fft_re.hex", "w") as f_re, open(f"{OUTPUT_DIR}/golden_fft_im.hex", "w") as f_im:
    for r, i in zip(golden_fft_re_final, golden_fft_im_final):
        f_re.write(f"{int_to_hex(r)}\n")
        f_im.write(f"{int_to_hex(i)}\n")

# --- 4. e4m3 golden = cast(STFT golden, k=9) ; for the STFT+cast end-to-end check ---
print(f"Generating Golden e4m3 (cast of STFT golden, k={K_ENC})...")
with open(f"{OUTPUT_DIR}/golden_fft_e4m3_re.hex","w") as f_re, open(f"{OUTPUT_DIR}/golden_fft_e4m3_im.hex","w") as f_im:
    for r, i in zip(golden_fft_re_final, golden_fft_im_final):
        f_re.write(f"{fxp_to_e4m3(int(r)) & 0xFF:02X}\n")
        f_im.write(f"{fxp_to_e4m3(int(i)) & 0xFF:02X}\n")

print("Done! Patterns generated in ./patterns/")