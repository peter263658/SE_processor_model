#!/usr/bin/env python3
"""Cold-start (warm-up) attention golden for t* < 32 (Consult B5).

Window not full: query t* sees keys [0, t*] (t*+1 valid; <4 valid for t*<3). Dumps
the same fields as mha_module_io, for t*=10 (11 valid) and t*=2 (3 valid <4), all 16
heads, both passes — so the per-frame validity mask + shift-top4 softmax (top-k and
pow2-denom over <32 / <4 valid) can be bit-exact-verified.

Uses the deploy path (fp32-sum QK, == shipped golden), same as mha_module_io.
Out: hw_patterns_final/mha_module_io_warmup/t{02,10}/{real,imag}_pass/...
"""
from __future__ import annotations
import os, sys, json
import numpy as np, torch

PKG = "/raid/R12K41024/18_w4a8_pow2_mixed_lmul_hw_handoff_20260530"
THIS = "/raid/R12K41024/19_algo_final"
for k, v in {"BCCTN_ACTIVATION_FP8_MODE": "calibrated_pow2_e4m3",
             "BCCTN_ACTIVATION_CALIB_PATH": PKG + "/ckpt/activation_calibration_p999_b64.json",
             "BCCTN_ACTIVATION_POW2_ROUND": "floor", "BCCTN_OUTLIER_DELAYED_DESCALE": "1",
             "NUMBA_CACHE_DIR": "/tmp/nb", "MPLCONFIGDIR": "/tmp/mpl", "BCCTN_APOT_A_MIN": "-16",
             "BCCTN_APOT_A_MAX": "-1", "BCCTN_APOT_B_MIN": "-16", "BCCTN_APOT_B_MAX": "-1",
             "BCCTN_OUTLIER_DESCALE_APOT_MIN": "-16", "BCCTN_OUTLIER_DESCALE_APOT_MAX": "-6"}.items():
    os.environ.setdefault(k, v)
sys.path.insert(0, THIS + "/code"); sys.path.insert(0, PKG + "/software_to_hardware_18")
import export_helpers as eh
import torch.nn.functional as F
import DCNN.utils.complexPyTorch.complexLayers as CL
OUT = THIS + "/hw_patterns_final/mha_module_io_warmup"
CAP = []


def _cap(mha, x, attn_mask, top_k, activation_quantizer=None, fp16_accumulator=False,
         softmax_mode="softmax", attention_datapath=None, mha_projection=None):
    c = {}
    B, T, E = x.shape; H = mha.num_heads; D = E // H; od = x.dtype
    xq = activation_quantizer(x) if activation_quantizer else x
    mp = mha_projection or {}; g = int(mp.get("qkv_groups", 1) or 1)
    w = mha.in_proj_weight; m = CL._mha_in_proj_group_mask(w, E, g); w = w * m if m is not None else w
    q, k, v = F.linear(xq, w, mha.in_proj_bias).chunk(3, -1)
    q = q.reshape(B, T, H, D).transpose(1, 2); k = k.reshape(B, T, H, D).transpose(1, 2); v = v.reshape(B, T, H, D).transpose(1, 2)
    if activation_quantizer: q = activation_quantizer(q); k = activation_quantizer(k); v = activation_quantizer(v)
    if fp16_accumulator: q = q.to(torch.float16); k = k.to(torch.float16); v = v.to(torch.float16)
    c["k"] = k.clone(); c["v"] = v.clone()
    lb = int(CL._attention_datapath_value(attention_datapath, "lmul_mantissa_bits", 4))
    be = bool(CL._attention_datapath_value(attention_datapath, "lmul_bit_exact", False))
    ww = CL._infer_causal_window_from_mask(attn_mask, T, T)
    qk = CL._lmul_windowed_qk_matmul(q, k.transpose(-2, -1), ww, mantissa_bits=lb, bit_exact=be,
                                     chunk_size=int(CL._attention_datapath_value(attention_datapath, "lmul_qk_chunk_size", 32)))
    c["qk_raw"] = qk.clone()
    attn = CL._apply_qk_scale(qk, D, attention_datapath); c["attn_scaled"] = attn.clone()
    if attn_mask is not None:
        attn = attn + (attn_mask.to(attn.dtype) if fp16_accumulator else attn_mask)
    c["attn_masked"] = attn.clone()
    attn = CL._quantize_attention_scores(attn, attention_datapath)
    cfg = CL._shift_topk_config(CL._canonical_softmax_mode(softmax_mode))
    attn = CL._shift_topk_softmax(attn, torch.isfinite(attn), top_k=top_k, **cfg)
    c["probs"] = attn.clone()
    attn = CL._quantize_attention_probs(attn, attention_datapath)
    pcfg = CL._prob_packet_config(attention_datapath); pk = CL._packetize_q08_attention_probs(attn, pcfg); attn = pk["probs"]
    c["codes"] = pk["codes_topk"].clone(); c["idx"] = pk["indices"].clone(); c["valid"] = pk["valid_slot"].clone()
    out = CL._q08_packet_lmul_matmul(pk, v, mantissa_bits=lb, bit_exact=be); c["scorev"] = out.clone()
    out = out.transpose(1, 2).contiguous().reshape(B, T, E)
    if fp16_accumulator: out = out.to(od)
    ow = mha.out_proj.weight; om = CL._linear_group_mask(ow, int(mp.get("out_proj_groups", 1) or 1)); ow = ow * om if om is not None else ow
    out = F.linear(out, ow, mha.out_proj.bias); c["dims"] = (H, D); CAP.append(c); return out


def dump(p, arr, dt):
    eh.ensure(os.path.dirname(p)); np.ascontiguousarray(arr, dt).tofile(p + ".bin")
    fmt = "{:04X}" if dt in (np.float16, np.uint16) else ("{:02X}" if dt == np.uint8 else "{:08X}")
    view = np.uint16 if dt in (np.float16, np.uint16) else (np.uint8 if dt == np.uint8 else np.uint32)
    with open(p + ".hex", "w") as f:
        for w in np.ascontiguousarray(arr, dt).reshape(-1).view(view):
            f.write(fmt.format(int(w)) + "\n")


def main():
    model, *_ = eh.build_model(); model.eval()
    import soundfile as sf
    a, _ = sf.read("/raid/R12K41024/BCCTN/Dataset/noisy_testset/snr_0dB/p225_p225_009_az-40_snr+0.0.wav", dtype="float32", always_2d=True)
    a = a[:, :2][:32000]
    CL._mha_attention_forward = _cap
    with torch.no_grad():
        model(torch.from_numpy(a.T).unsqueeze(0).contiguous())
    H, D = CAP[0]["dims"]
    passes = {0: "real_pass", 1: "imag_pass"}
    for QS in (2, 10):
        nvalid = QS + 1   # keys [0..QS]
        for pi in (0, 1):
            c = CAP[pi]; b = 0
            base = os.path.join(OUT, f"t{QS:02d}", passes[pi])
            for h in range(H):
                hd = os.path.join(base, f"head_{h:02d}")
                # window keys [0..QS] in local_idx order (0=newest=QS .. QS=oldest=key0)
                Kw = c["k"][b, h, 0:QS + 1, :].flip(0).to(torch.float32).numpy()    # [nvalid,16]
                Vw = c["v"][b, h, 0:QS + 1, :].flip(0).to(torch.float32).numpy()
                row_raw = c["qk_raw"][b, h, QS, 0:QS + 1].flip(0).to(torch.float32).numpy()
                row_scl = c["attn_scaled"][b, h, QS, 0:QS + 1].flip(0).to(torch.float32).numpy()
                dense = c["probs"][b, h, QS, 0:QS + 1].flip(0).to(torch.float32).numpy()
                dump(os.path.join(hd, "Kwindow_localidx_fp16"), Kw, np.float16)
                dump(os.path.join(hd, "Vwindow_localidx_fp16"), Vw, np.float16)
                dump(os.path.join(hd, "scores_row_localidx_prescale_fp16"), row_raw, np.float16)
                dump(os.path.join(hd, "scores_row_localidx_postscale_fp16"), row_scl, np.float16)
                dump(os.path.join(hd, "softmax_dense_localidx_fp32"), dense, np.float32)
                dump(os.path.join(hd, "ctx16_fp32"), c["scorev"][b, h, QS, :].to(torch.float32).numpy(), np.float32)
                codes = np.clip(np.rint(c["codes"][b, h, QS].numpy()), 0, 255).astype(int)
                lidx = (QS - c["idx"][b, h, QS].numpy()).astype(int)
                valid = c["valid"][b, h, QS].numpy().astype(int)
                json.dump({"query_abs": QS, "n_valid_keys": nvalid, "top_k_slots_used": int(valid.sum()),
                           "prob_code": codes.tolist(), "local_idx": lidx.tolist(), "valid_slot": valid.tolist(),
                           "nonzero_softmax": int((dense > 0).sum())},
                          open(os.path.join(hd, "packet.json"), "w"), indent=1)
            if pi == 0 and h == 0:
                pass
        print(f"  t*={QS}: {nvalid} valid keys; head0 real_pass packet -> {OUT}/t{QS:02d}/")
    # quick summary
    for QS in (2, 10):
        pk = json.load(open(os.path.join(OUT, f"t{QS:02d}", "real_pass", "head_00", "packet.json")))
        print(f"  t*={QS} head0: n_valid={pk['n_valid_keys']} top4_used={pk['top_k_slots_used']} "
              f"softmax_nz={pk['nonzero_softmax']} codes={pk['prob_code']} lidx={pk['local_idx']} valid={pk['valid_slot']}")
    print(f"[gen] wrote {OUT}/  (t02, t10; 16 heads x 2 passes)")


if __name__ == "__main__":
    main()
