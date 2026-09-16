#!/usr/bin/env python3
"""Parameter reduction, official BCCTN checkpoint (11.09 M) -> hardware-sized working point (830,300).

Official = VikasTokala/BCCTN `Trained_model.ckpt` (BCCTN/DCNN/Checkpoints/). Its transformer is a
complex MHA directly at embed 1024 (no projections) + a real Linear 1024->1024; conv kernels (5,2).
Intermediate points are instantiated from the repo code (BCCTN/ = projected transformer + (5,1) conv;
main_model/ = same + depthwise flag). Counts are trainable parameters only (BN running stats and the
loss module's STOI buffers are excluded); the official ckpt tensor total is 11,093,257.

Run (CPU):  CUDA_VISIBLE_DEVICES= ~/miniforge3/envs/bcctn/bin/python scripts/count_params_official_to_hw.py
"""
import contextlib, importlib, io, sys
import torch

R = "/raid/R12K41024"
OFFICIAL = f"{R}/BCCTN/DCNN/Checkpoints/Trained_model.ckpt"
COMMON = dict(rnn_layers=2, rnn_units=128, win_type="hann", kernel_size=5, kernel_num=[16, 32, 64, 128, 256, 256],
              bidirectional=False, binaural=True, attention=True, use_clstm=True, use_cbn=True, masking_mode="E")
BUF = ("running_mean", "running_var", "num_batches_tracked")


def load(repo):
    for m in [k for k in sys.modules if k.startswith("DCNN")]:
        del sys.modules[m]
    sys.path.insert(0, f"{R}/{repo}")
    mod = importlib.import_module("DCNN.models.model")
    sys.path.pop(0)
    return mod.DCNN


def build(D, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return D(**{**COMMON, **kw})


def groups_from_model(m):
    g = {"transformer": 0, "encoder": 0, "decoder": 0}
    for n, p in m.named_parameters():
        g["transformer" if n.startswith("mattn") else n.split(".")[0]] += p.numel()
    return g


def groups_from_ckpt(path):
    sd = torch.load(path, map_location="cpu")["state_dict"]
    g = {"transformer": 0, "encoder": 0, "decoder": 0}
    for k, v in sd.items():
        if not k.startswith("model.") or k.endswith(BUF):
            continue
        k = k[6:]
        g["transformer" if k.startswith("mattn") else k.split(".")[0]] += v.numel()
    return g


D0, D1 = load("BCCTN"), load("main_model")
chain = [
    ("S0 official ckpt: MHA@1024 (no proj) + Linear 1024^2, conv (5,2), fft 512, embed 1024/32 heads", groups_from_ckpt(OFFICIAL)),
    ("T1 transformer block: Linear 1024->512 + complex MHA@512 + Linear 512->1024 (repo design), fft 512",
     None),  # = S0 conv + repo transformer at fft512/embed512 -> computed below
    ("C1 conv kernel (5,2) -> (5,1)  [= repo reproduction 3,973,288]",
     groups_from_model(build(D0, win_len=400, win_inc=100, fft_len=512, embed_dim=512, num_heads=32))),
    ("F  fft 512 -> 128 (win 400->128, hop 100->80): transformer input 1024 -> 256",
     groups_from_model(build(D0, win_len=128, win_inc=80, fft_len=128, embed_dim=512, num_heads=32))),
    ("A  embed 512 -> 256, heads 32 -> 16",
     groups_from_model(build(D0, win_len=128, win_inc=80, fft_len=128, embed_dim=256, num_heads=16))),
    ("D  full conv -> depthwise-separable (5,1) + pointwise  [= hardware-sized 830,300]",
     groups_from_model(build(D1, win_len=128, win_inc=80, fft_len=128, embed_dim=256, num_heads=16, use_depthwise=True))),
]
# T1 = official conv + repo transformer (fft 512 / embed 512)
chain[1] = (chain[1][0], {"transformer": chain[2][1]["transformer"], "encoder": chain[0][1]["encoder"], "decoder": chain[0][1]["decoder"]})
# cross-check: main_model non-depthwise encoder == official encoder (both (5,2))
mmF = groups_from_model(build(D1, win_len=128, win_inc=80, fft_len=128, embed_dim=256, num_heads=16, use_depthwise=False))
assert mmF["encoder"] == chain[0][1]["encoder"], (mmF["encoder"], chain[0][1]["encoder"])

tot0 = sum(chain[0][1].values())
totN = sum(chain[-1][1].values())
print(f"official params {tot0:,}  ->  hardware-sized {totN:,}   ({100*(totN/tot0-1):+.1f}%)\n")
print("| Step | Transformer | Encoder | Decoder | Total | Δ step | Δ / original | share of total saving |")
print("|---|---:|---:|---:|---:|---:|---:|---:|")
prev = None
for name, g in chain:
    t = sum(g.values())
    d = "" if prev is None else f"{t-prev:+,}"
    pct = "" if prev is None else f"{100*(t-prev)/tot0:+.1f}%"
    share = "" if prev is None else f"{100*(prev-t)/(tot0-totN):.1f}%"
    print(f"| {name} | {g['transformer']:,} | {g['encoder']:,} | {g['decoder']:,} | {t:,} | {d} | {pct} | {share} |")
    prev = t

# order sensitivity of F vs A (they interact through the in/out projections)
a_first = sum(groups_from_model(build(D0, win_len=400, win_inc=100, fft_len=512, embed_dim=256, num_heads=16)).values())
c1 = sum(chain[2][1].values()); f = sum(chain[3][1].values()); a = sum(chain[4][1].values())
print(f"\norder check: F then A = {f-c1:+,} then {a-f:+,};  A then F = {a_first-c1:+,} then {a-a_first:+,};  sum {a-c1:+,} either way")
