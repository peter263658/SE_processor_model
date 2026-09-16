#!/usr/bin/env python3
"""
gen_istft_patterns.py — directed patterns for the ISTFT path (STFT_TOP i_stft_mode=ISTFT).

Per frame: a masked onesided spectrum (64 bins 1..64, Q1.15, DC dropped) -> the chip's
conj_sym_feeder + FFT128(as IFFT, ÷128) + istft_reorder should output the 128 natural-order
real time samples = real(IFFT(conj-sym spectrum)).  (synth window + OLA + normalization are
OFF-CHIP, NOT checked here — this verifies the on-chip IFFT transform.)

Golden = float IFFT of the EXACT quantized spectrum the chip receives (so the only RTL-vs-
golden gap is fixed-point FFT128 rounding, same as the forward path's ±2 LSB).

Writes patterns/istft_in_re.hex, istft_in_im.hex (64/frame), istft_golden.hex (128/frame).
"""
import numpy as np, os
HERE=os.path.dirname(os.path.abspath(__file__)); OUTD=os.path.join(HERE,"patterns")
os.makedirs(OUTD,exist_ok=True)
N=128; NB=64; FRAMES=4
def q15(x):  # float -> Q1.15 int16 (saturating)
    v=np.round(np.asarray(x)*32768.0); v=np.clip(v,-32768,32767); return v.astype(np.int64)
def hx(v): return f"{int(v)&0xFFFF:04X}"

np.random.seed(11)
fin_re=open(f"{OUTD}/istft_in_re.hex","w"); fin_im=open(f"{OUTD}/istft_in_im.hex","w")
fg   =open(f"{OUTD}/istft_golden.hex","w")
maxabs_golden=0
for f in range(FRAMES):
    # directed: random Q1.15 masked spectrum, bins 1..64 (bin64=Nyquist must be REAL).
    # (a real masked spectrum is ÷128-scale -> tiny IFFT; we use larger bins for test SNR.
    #  the transform is linear, so this validates the IFFT at any scale.)
    bre = q15(np.random.uniform(-0.5,0.5,NB))
    bim = q15(np.random.uniform(-0.5,0.5,NB))
    bim[NB-1] = 0                                        # bin64 (Nyquist) real
    # conj_sym_feeder model (RTL-exact): emit[k]=conj(Xfull[k]); Xfull[0]=0, [1..64]=bins, mirror conj
    bf_re = bre/32768.0; bf_im = bim/32768.0            # decode chip's Q1.15 input back to float
    emit=np.zeros(N,dtype=complex)
    for k in range(N):
        if   k==0:    emit[k]=0
        elif k<=64:   emit[k]=bf_re[k-1] - 1j*bf_im[k-1]      # conj(bin k)
        else:         emit[k]=bf_re[127-k] + 1j*bf_im[127-k]  # bin(128-k)
    gold = np.real(np.fft.fft(emit))/N                  # = real(IFFT(spectrum)) , chip output
    maxabs_golden=max(maxabs_golden, np.max(np.abs(gold)))
    for b in range(NB):
        fin_re.write(hx(bre[b])+"\n"); fin_im.write(hx(bim[b])+"\n")
    for t in range(N):
        fg.write(hx(q15(gold[t]))+"\n")
fin_re.close(); fin_im.close(); fg.close()
print(f"frames={FRAMES}  spec/frame={NB}  samples/frame={N}")
print(f"max|golden time sample|={maxabs_golden:.4f} (Q1.15-safe: {'yes' if maxabs_golden<1 else 'NO'})")
print("patterns: istft_in_re.hex, istft_in_im.hex (64/frame), istft_golden.hex (128/frame)")
