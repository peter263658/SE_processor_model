#!/usr/bin/env python3
"""Executed-real-MAC counter via forward hooks: official 11.09 M ckpt vs fft-512 reproduction.
Counts every real Conv2d/ConvTranspose2d/Linear call (complex layers execute 4 real convs -> literal x4)
and MultiheadAttention (in_proj T*E*3E + QK T*T*E + AV T*T*E + out_proj T*E*E per real MHA; full T x T).
Input: 2.0 s stereo @16 kHz. Reports MAC per second and per hop-frame.
Cross-check: HW-sized conv MAC/frame (0.237 G/s / 200 fps = 1.19 M) equals 19_algo_final/scripts/mac_profiler.py
dense-baseline encoder+decoder x4 (405,760 + 784,640); its attention is full T x T here (PyTorch masks do
not skip work), whereas mac_profiler counts the windowed / top-4 hardware attention.
Run (CPU):  CUDA_VISIBLE_DEVICES= ~/miniforge3/envs/bcctn/bin/python code/mac_profile_official.py"""
import sys, io, contextlib, importlib, torch, torch.nn as nn
sys.path.insert(0, "/raid/R12K41024/23_official_bcctn_ckpt_eval/code")
from eval_official import build_official_model, _purge_dcnn_modules
R = "/raid/R12K41024"; SR = 16000; SEC = 2.0
COMMON = dict(rnn_layers=2, rnn_units=128, win_type="hann", kernel_size=5, kernel_num=[16,32,64,128,256,256],
              bidirectional=False, binaural=True, attention=True, use_clstm=True, use_cbn=True, masking_mode="E")

def count_macs(model, x):
    tot = {"conv": 0, "linear": 0, "mha": 0}
    hs = []
    def conv_hook(m, inp, out):
        cin = m.in_channels // m.groups; k = m.kernel_size[0] * m.kernel_size[1]
        if isinstance(m, nn.ConvTranspose2d):   # input-driven
            tot["conv"] += m.out_channels * cin * k * inp[0].shape[0] * inp[0].shape[2] * inp[0].shape[3]
        else:
            tot["conv"] += out.numel() * cin * k
    def lin_hook(m, inp, out):
        f = 4 if m.weight.is_complex() else 1   # complex Linear = 4 real MACs per complex MAC
        tot["linear"] += out.numel() * m.in_features * f
    def mha_hook(m, inp, out):
        q = inp[0]; B, T, E = q.shape
        tot["mha"] += B * (T*E*3*E + T*T*E + T*T*E + T*E*E)
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)): hs.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.MultiheadAttention): hs.append(m.register_forward_hook(mha_hook))
        elif isinstance(m, nn.Linear) and not m.__class__.__name__.startswith("NonDynamically"): hs.append(m.register_forward_hook(lin_hook))
    with torch.no_grad(): model(x)
    for h in hs: h.remove()
    return tot

x = torch.randn(1, 2, int(SR * SEC))
rows = []
m = build_official_model(f"{R}/BCCTN/DCNN/Checkpoints/Trained_model.ckpt", torch.device("cpu"))
rows.append(("official 11.09 M (fft512/hop100, MHA@1024)", 100, count_macs(m, x)))
_purge_dcnn_modules(); sys.path.insert(0, f"{R}/BCCTN"); D0 = importlib.import_module("DCNN.models.binaural_attention_model").BinauralAttentionDCNN; sys.path.pop(0)
with contextlib.redirect_stdout(io.StringIO()):
    m2 = D0(**COMMON, win_len=400, win_inc=100, fft_len=512, embed_dim=512, num_heads=32).eval()
rows.append(("repro fft512 / embed512 / (5,1) conv = 3,973,288", 100, count_macs(m2, x)))
_purge_dcnn_modules(); sys.path.insert(0, f"{R}/main_model"); D1 = importlib.import_module("DCNN.models.binaural_attention_model").BinauralAttentionDCNN; sys.path.pop(0)
with contextlib.redirect_stdout(io.StringIO()):
    m3 = D1(**COMMON, win_len=128, win_inc=80, fft_len=128, embed_dim=256, num_heads=16, use_depthwise=True).eval()
rows.append(("HW-sized 830 K (fft128/hop80, dense QKV/decoder, attn_win as coded)", 80, count_macs(m3, x)))
print(f"{'model':70s} {'MAC/s':>10s} {'MAC/frame(hop)':>15s}   conv / linear / mha  (MAC/s)")
for name, hop, t in rows:
    per_s = sum(t.values()) / SEC; per_frame = per_s * hop / SR
    print(f"{name:70s} {per_s/1e9:8.3f} G {per_frame/1e6:11.2f} M   {t['conv']/SEC/1e9:.3f} / {t['linear']/SEC/1e9:.3f} / {t['mha']/SEC/1e9:.3f} G")
