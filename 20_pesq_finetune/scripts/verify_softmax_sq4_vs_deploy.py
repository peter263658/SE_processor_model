#!/usr/bin/env python3
"""Quantify the downstream cost of the Softmax_v3 sq4 score quantization vs the
deploy full-fp16 shift-topk softmax (the gated model).

deploy: _shift_topk_softmax sees fp16 scores (score_quant=none) -> x=score-max in
        fp16 -> floor(1.5x).  RTL Softmax_v3 == tb_SM.py: truncates score to sq4
        (Q.4, 1/16) FIRST -> near-tie (delta<1/16) collapses to 2^0 (flatter).

Runs the deploy MHA twice (deploy softmax vs sq4-truncated softmax) on the golden
wav and reports the packet/context/output divergence.

    cd /raid/R12K41024/19_algo_final
    conda run --no-capture-output -n bcctn python scripts/verify_softmax_sq4_vs_deploy.py
"""
from __future__ import annotations
import os, sys
import numpy as np
import torch

PKG = "/raid/R12K41024/18_w4a8_pow2_mixed_lmul_hw_handoff_20260530"
S2H = os.path.join(PKG, "software_to_hardware_18")
THIS = "/raid/R12K41024/19_algo_final"
CODE = os.path.join(THIS, "code")
GOLDEN = os.path.join(THIS, "hw_patterns_final", "debug_reference", "layer_io_fp32",
                      "output_waveform.fp32.hex")
for k, v in {
    "BCCTN_APOT_A_MIN": "-16", "BCCTN_APOT_A_MAX": "-1", "BCCTN_APOT_B_MIN": "-16", "BCCTN_APOT_B_MAX": "-1",
    "BCCTN_ACTIVATION_FP8_MODE": "calibrated_pow2_e4m3",
    "BCCTN_ACTIVATION_CALIB_PATH": os.path.join(PKG, "ckpt", "activation_calibration_p999_b64.json"),
    "BCCTN_ACTIVATION_POW2_ROUND": "floor", "BCCTN_OUTLIER_DELAYED_DESCALE": "1",
    "BCCTN_OUTLIER_DESCALE_APOT_MIN": "-16", "BCCTN_OUTLIER_DESCALE_APOT_MAX": "-6",
    "NUMBA_CACHE_DIR": "/tmp/numba_cache", "MPLCONFIGDIR": "/tmp/mpl_cache",
}.items():
    os.environ.setdefault(k, v)
sys.path.insert(0, CODE); sys.path.insert(0, S2H)
import export_helpers as eh                              # noqa
import torch.nn.functional as F                          # noqa
import DCNN.utils.complexPyTorch.complexLayers as CL     # noqa

SQ4 = False
CAP = []
_orig_shift_topk = CL._shift_topk_softmax


def shift_topk_sq4(attn, valid_mask, top_k, **kw):
    if SQ4:
        finite = torch.isfinite(attn)
        # Q.4 truncate (floor to 1/16) — RTL Softmax_v3 / tb_SM.py
        trunc = torch.floor(attn.to(torch.float32) * 16.0) / 16.0
        attn = torch.where(finite, trunc.to(attn.dtype), attn)
    return _orig_shift_topk(attn, valid_mask, top_k, **kw)


def _cap_mha(mha, x, attn_mask, top_k, activation_quantizer=None, fp16_accumulator=False,
             softmax_mode="softmax", attention_datapath=None, mha_projection=None):
    c = {}
    B, T, E = x.shape; H = mha.num_heads; D = E // H; od = x.dtype
    xq = activation_quantizer(x) if activation_quantizer is not None else x
    mp = mha_projection or {}; g = int(mp.get("qkv_groups", 1) or 1)
    w = mha.in_proj_weight; m = CL._mha_in_proj_group_mask(w, E, g)
    if m is not None: w = w * m
    q, k, v = F.linear(xq, w, mha.in_proj_bias).chunk(3, -1)
    q = q.reshape(B, T, H, D).transpose(1, 2); k = k.reshape(B, T, H, D).transpose(1, 2); v = v.reshape(B, T, H, D).transpose(1, 2)
    if activation_quantizer is not None: q = activation_quantizer(q); k = activation_quantizer(k); v = activation_quantizer(v)
    if fp16_accumulator: q = q.to(torch.float16); k = k.to(torch.float16); v = v.to(torch.float16)
    lb = int(CL._attention_datapath_value(attention_datapath, "lmul_mantissa_bits", 4))
    be = bool(CL._attention_datapath_value(attention_datapath, "lmul_bit_exact", False))
    ww = CL._infer_causal_window_from_mask(attn_mask, T, T)
    qk = CL._lmul_windowed_qk_matmul(q, k.transpose(-2, -1), ww, mantissa_bits=lb, bit_exact=be,
                                     chunk_size=int(CL._attention_datapath_value(attention_datapath, "lmul_qk_chunk_size", 32)))
    attn = CL._apply_qk_scale(qk, D, attention_datapath)
    if attn_mask is not None: attn = attn + (attn_mask.to(attn.dtype) if fp16_accumulator else attn_mask)
    attn = CL._quantize_attention_scores(attn, attention_datapath)
    cfg = CL._shift_topk_config(CL._canonical_softmax_mode(softmax_mode))
    attn = CL._shift_topk_softmax(attn, torch.isfinite(attn), top_k=top_k, **cfg)
    c["probs"] = attn.clone()
    attn = CL._quantize_attention_probs(attn, attention_datapath)
    pcfg = CL._prob_packet_config(attention_datapath); pk = CL._packetize_q08_attention_probs(attn, pcfg); attn = pk["probs"]
    c["idx"] = pk["indices"].clone(); c["codes"] = pk["codes_topk"].clone()
    out = CL._q08_packet_lmul_matmul(pk, v, mantissa_bits=lb, bit_exact=be)
    c["scorev"] = out.clone()
    out = out.transpose(1, 2).contiguous().reshape(B, T, E)
    if fp16_accumulator: out = out.to(od)
    ow = mha.out_proj.weight; om = CL._linear_group_mask(ow, int(mp.get("out_proj_groups", 1) or 1))
    if om is not None: ow = ow * om
    out = F.linear(out, ow, mha.out_proj.bias); c["out"] = out.clone(); CAP.append(c); return out


def run(model, x, sq4):
    global SQ4, CAP
    SQ4 = sq4; CAP = []
    CL._shift_topk_softmax = shift_topk_sq4
    CL._mha_attention_forward = _cap_mha
    with torch.no_grad():
        y = model(x)
    return y, list(CAP)


def main():
    model, *_ = eh.build_model(); model.eval()
    import soundfile as sf
    a, _ = sf.read("/raid/R12K41024/BCCTN/Dataset/noisy_testset/snr_0dB/p225_p225_009_az-40_snr+0.0.wav", dtype="float32", always_2d=True)
    a = a[:, :2] if a.shape[1] >= 2 else np.repeat(a, 2, 1)
    a = a[:32000] if a.shape[0] >= 32000 else np.concatenate([a, np.zeros((32000 - a.shape[0], 2), np.float32)])
    x = torch.from_numpy(a.T).unsqueeze(0).contiguous()

    y0, c0 = run(model, x, False)
    g = np.array([int(l, 16) for l in open(GOLDEN).read().split()], np.uint32).view(np.float32)
    o0 = y0.cpu().to(torch.float32).numpy().reshape(-1)
    print(f"[deploy fp16-delta] output vs shipped golden: exact={np.array_equal(g,o0)} max_abs={np.max(np.abs(g-o0)):.2e}")
    y1, c1 = run(model, x, True)
    o1 = y1.cpu().to(torch.float32).numpy().reshape(-1)

    def cat(c, key): return torch.cat([c[0][key].reshape(-1), c[1][key].reshape(-1)]).to(torch.float32)
    print("\n==== Softmax: deploy fp16-delta vs sq4 (Q.4 truncate) ====")
    p0, p1 = cat(c0, "probs"), cat(c1, "probs")
    nz = (p0 != 0) | (p1 != 0)
    twox = (nz & ((torch.maximum(p0, p1) >= 1.999 * torch.minimum(p0, p1).clamp_min(1e-9)) & (torch.minimum(p0, p1) > 0)))
    print(f"  softmax dense : entries changed {int((p0!=p1).sum())}/{int(nz.sum())}  ~2x entries={int(twox.sum())}  max_abs={float((p0-p1).abs().max()):.3f}")
    i0 = torch.stack([c0[p]["idx"] for p in (0, 1)]); i1 = torch.stack([c1[p]["idx"] for p in (0, 1)])
    same = (torch.sort(i0, -1).values == torch.sort(i1, -1).values).all(-1)
    print(f"  top-4 keyset  : identical {int(same.sum())}/{same.numel()} ({100*float(same.sum())/same.numel():.2f}%)  top1 flip={int((i0[...,0]!=i1[...,0]).sum())}/{i0[...,0].numel()}")
    sv0, sv1 = cat(c0, "scorev"), cat(c1, "scorev")
    print(f"  Score×V ctx   : max_abs={float((sv0-sv1).abs().max()):.4f}  cos={float(torch.dot(sv0,sv1)/(sv0.norm()*sv1.norm())):.6f}")
    print(f"  full waveform : max_abs={np.max(np.abs(o0-o1)):.4e}  cos={float(np.dot(o0,o1)/(np.linalg.norm(o0)*np.linalg.norm(o1))):.7f}  rel_rms={np.sqrt(np.mean((o0-o1)**2))/np.sqrt(np.mean(o0**2)):.4e}")


if __name__ == "__main__":
    main()
