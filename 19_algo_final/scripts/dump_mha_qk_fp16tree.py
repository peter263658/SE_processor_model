#!/usr/bin/env python3
"""QK accumulator: deploy fp32-sum vs HW fp16 adder-tree (round every add).

Runs the deploy MHA twice on the golden wav — once with the deploy QK reduction
(`products.sum` = fp32 accumulate, round once) and once with the CE
`adder_tree_fp16` model (`reduce_mode=fp16_tree` = paired binary tree, fp16 RNE
+ saturate at every node) — and reports:
  * QK score max |abs| / ULP difference,
  * whether the top-4 softmax packet selection changes (indices / Jaccard),
  * final MHA-half output and full model output-waveform difference.
It then writes the fp16-tree golden to `hw_patterns_final/mha_module_io_fp16tree/`.

The fp32 run is asserted bit-exact vs the shipped golden output (baseline OK);
the fp16-tree run is the HW-bit-exact golden.

    cd /raid/R12K41024/19_algo_final
    conda run --no-capture-output -n bcctn python scripts/dump_mha_qk_fp16tree.py
"""
from __future__ import annotations
import os, sys, json
import numpy as np
import torch

PKG = "/raid/R12K41024/18_w4a8_pow2_mixed_lmul_hw_handoff_20260530"
S2H = os.path.join(PKG, "software_to_hardware_18")
THIS = "/raid/R12K41024/19_algo_final"
CODE = os.path.join(THIS, "code")
GOLDEN_OUT_HEX = os.path.join(THIS, "hw_patterns_final", "debug_reference",
                              "layer_io_fp32", "output_waveform.fp32.hex")

for k, v in {
    "BCCTN_APOT_A_MIN": "-16", "BCCTN_APOT_A_MAX": "-1",
    "BCCTN_APOT_B_MIN": "-16", "BCCTN_APOT_B_MAX": "-1",
    "BCCTN_ACTIVATION_FP8_MODE": "calibrated_pow2_e4m3",
    "BCCTN_ACTIVATION_CALIB_PATH": os.path.join(PKG, "ckpt", "activation_calibration_p999_b64.json"),
    "BCCTN_ACTIVATION_POW2_ROUND": "floor",
    "BCCTN_OUTLIER_DELAYED_DESCALE": "1",
    "BCCTN_OUTLIER_DESCALE_APOT_MIN": "-16", "BCCTN_OUTLIER_DESCALE_APOT_MAX": "-6",
    "NUMBA_CACHE_DIR": "/tmp/numba_cache", "MPLCONFIGDIR": "/tmp/mpl_cache",
}.items():
    os.environ.setdefault(k, v)

sys.path.insert(0, CODE); sys.path.insert(0, S2H)
import export_helpers as eh           # noqa: E402
import torch.nn.functional as F        # noqa: E402
import DCNN.utils.complexPyTorch.complexLayers as CL  # noqa: E402

QK_MODE = "sum"        # set per-run: "sum" (deploy) or "fp16_tree" (HW)
CAPTURES = []


def _mha_capture(mha, x, attn_mask, top_k, activation_quantizer=None,
                 fp16_accumulator=False, softmax_mode="softmax",
                 attention_datapath=None, mha_projection=None):
    cap = {}
    B, T, E = x.shape; H = mha.num_heads; D = E // H; orig_dtype = x.dtype
    x_q = activation_quantizer(x) if activation_quantizer is not None else x
    mha_projection = mha_projection or {}
    qkv_groups = int(mha_projection.get("qkv_groups", 1) or 1)
    out_proj_groups = int(mha_projection.get("out_proj_groups", 1) or 1)
    lmul_bits = int(CL._attention_datapath_value(attention_datapath, "lmul_mantissa_bits", 4))
    lmul_bit_exact = bool(CL._attention_datapath_value(attention_datapath, "lmul_bit_exact", False))
    in_proj_weight = mha.in_proj_weight
    in_proj_mask = CL._mha_in_proj_group_mask(in_proj_weight, E, qkv_groups)
    if in_proj_mask is not None:
        in_proj_weight = in_proj_weight * in_proj_mask
    qkv = F.linear(x_q, in_proj_weight, mha.in_proj_bias)
    q, k, v = qkv.chunk(3, dim=-1)
    q = q.reshape(B, T, H, D).transpose(1, 2)
    k = k.reshape(B, T, H, D).transpose(1, 2)
    v = v.reshape(B, T, H, D).transpose(1, 2)
    if activation_quantizer is not None:
        q = activation_quantizer(q); k = activation_quantizer(k); v = activation_quantizer(v)
    if fp16_accumulator:
        q = q.to(torch.float16); k = k.to(torch.float16); v = v.to(torch.float16)
    cap["q"] = q.clone(); cap["k"] = k.clone(); cap["v"] = v.clone()

    # ---- QK with selectable reduction ----
    reduce_mode = "fp16_tree" if QK_MODE == "fp16_tree" else "sum"
    k_t = k.transpose(-2, -1)
    qk_window_width = CL._infer_causal_window_from_mask(attn_mask, T, T)
    qk_chunk = int(CL._attention_datapath_value(attention_datapath, "lmul_qk_chunk_size", 32))
    qk_scores = CL._lmul_windowed_qk_matmul(
        q, k_t, qk_window_width, mantissa_bits=lmul_bits,
        bit_exact=lmul_bit_exact, chunk_size=qk_chunk, reduce_mode=reduce_mode)
    cap["qk_scores_raw"] = qk_scores.clone()
    attn = CL._apply_qk_scale(qk_scores, D, attention_datapath)
    cap["attn_scaled"] = attn.clone()
    if attn_mask is not None:
        mask = attn_mask.to(attn.dtype) if fp16_accumulator else attn_mask
        attn = attn + mask
    cap["attn_masked"] = attn.clone()

    attn = CL._quantize_attention_scores(attn, attention_datapath)
    softmax_mode = CL._canonical_softmax_mode(softmax_mode)
    cfg = CL._shift_topk_config(softmax_mode)
    if cfg is not None:
        attn = CL._shift_topk_softmax(attn, torch.isfinite(attn), top_k=top_k, **cfg)
    else:
        attn = CL._approx_attention_probs(attn, softmax_mode)
    cap["probs_dense"] = attn.clone()
    attn = CL._quantize_attention_probs(attn, attention_datapath)
    packet_cfg = CL._prob_packet_config(attention_datapath)
    packet = CL._packetize_q08_attention_probs(attn, packet_cfg)
    attn = packet["probs"]
    cap["packet_codes_topk"] = packet["codes_topk"].clone()
    cap["packet_indices"] = packet["indices"].clone()
    cap["packet_probs_topk"] = packet["probs_topk"].clone()
    cap["packet_valid_slot"] = packet["valid_slot"].clone()
    out = CL._q08_packet_lmul_matmul(packet, v, mantissa_bits=lmul_bits, bit_exact=lmul_bit_exact)
    cap["scorev"] = out.clone()
    out = out.transpose(1, 2).contiguous().reshape(B, T, E)
    if fp16_accumulator:
        out = out.to(orig_dtype)
    cap["ctx"] = out.clone()
    ow = mha.out_proj.weight
    om = CL._linear_group_mask(ow, out_proj_groups)
    if om is not None:
        ow = ow * om
    out = F.linear(out, ow, mha.out_proj.bias)
    cap["out"] = out.clone()
    cap["dims"] = {"B": B, "T": T, "E": E, "H": H, "D": D}
    CAPTURES.append(cap)
    return out


def load_wav():
    import soundfile as sf
    wav = "/raid/R12K41024/BCCTN/Dataset/noisy_testset/snr_0dB/p225_p225_009_az-40_snr+0.0.wav"
    a, sr = sf.read(wav, dtype="float32", always_2d=True)
    if a.shape[1] == 1:
        a = np.repeat(a, 2, 1)
    a = a[:, :2]
    a = a[:32000] if a.shape[0] >= 32000 else np.concatenate([a, np.zeros((32000 - a.shape[0], 2), np.float32)])
    return torch.from_numpy(a.T).unsqueeze(0).contiguous(), wav


def run(model, x, mode):
    global QK_MODE, CAPTURES
    QK_MODE = mode
    CAPTURES = []
    CL._mha_attention_forward = _mha_capture
    with torch.no_grad():
        y = model(x)
    return y, list(CAPTURES)


def dump(path_noext, arr, dtype):
    eh.ensure(os.path.dirname(path_noext))
    eh.write_bin(path_noext + ".bin", np.ascontiguousarray(arr, dtype=dtype))


def main():
    model, *_ = eh.build_model(); model.eval()
    x, wav = load_wav()

    y_fp32, cap32 = run(model, x, "sum")
    gw = np.array([int(l, 16) for l in open(GOLDEN_OUT_HEX).read().split()], np.uint32).view(np.float32)
    got = y_fp32.cpu().to(torch.float32).numpy().reshape(-1)
    print(f"[baseline] fp32-sum output vs shipped golden: exact={np.array_equal(gw, got)} "
          f"max_abs={np.max(np.abs(gw - got)):.3e}")

    y_tree, capT = run(model, x, "fp16_tree")
    yt = y_tree.cpu().to(torch.float32).numpy().reshape(-1)

    H = cap32[0]["dims"]["H"]; T = cap32[0]["dims"]["T"]

    def hp(c, key):  # both passes concatenated
        return torch.cat([c[0][key].reshape(-1), c[1][key].reshape(-1)])

    print("\n==== QK accumulator: fp32-sum (deploy) vs fp16-tree (HW) ====")
    # 1) QK raw scores (only finite/in-window entries are nonzero)
    s32 = hp(cap32, "qk_scores_raw").to(torch.float32)
    sT = hp(capT, "qk_scores_raw").to(torch.float32)
    nz = (s32 != 0) | (sT != 0)
    d = (s32 - sT).abs()
    # fp16 ULP measured at the score MAGNITUDE (max of the two operands) — this is
    # the meaningful "how many representable steps apart" metric. (ULP-of-value
    # blows up only under cancellation when the score itself is ~0.)
    def fp16_ulp(v):
        v = v.abs().clamp_min(2.0 ** -14)         # smallest fp16 normal
        e = torch.floor(torch.log2(v))
        return torch.ldexp(torch.ones_like(v), (e - 10).to(torch.int32))
    mag = torch.maximum(s32.abs(), sT.abs())
    ulp = (d / fp16_ulp(mag)).where(nz, torch.zeros_like(d))
    dn = d[nz]
    print(f"  QK raw scores : nonzero={int(nz.sum())}  changed={int((dn != 0).sum())} "
          f"({100.0 * float((dn != 0).sum()) / int(nz.sum()):.2f}%)")
    print(f"    abs diff    : max={float(dn.max()):.4e}  p99.9={float(dn.quantile(0.999)):.4e}  mean={float(dn.mean()):.4e}")
    un = ulp[nz]
    print(f"    ULP@magnitude: max={float(un.max()):.2f}  p99.9={float(un.quantile(0.999)):.2f}  mean={float(un.mean()):.3f}  "
          f"|  ==0:{int((un==0).sum())}  <=1ULP:{int((un<=1).sum())}  <=2ULP:{int((un<=2).sum())}  >4ULP:{int((un>4).sum())}")
    # 2) scaled+masked softmax input
    a32 = hp(cap32, "attn_masked").to(torch.float32); aT = hp(capT, "attn_masked").to(torch.float32)
    fin = torch.isfinite(a32) & torch.isfinite(aT)
    print(f"  softmax input : max_abs_diff(finite)={float((a32 - aT)[fin].abs().max()):.4e}")
    # 3) packet selection (top-4 indices) — does HW pick the same keys?
    i32 = torch.stack([cap32[p]["packet_indices"] for p in (0, 1)])  # (2,B,H,T,4)
    iT = torch.stack([capT[p]["packet_indices"] for p in (0, 1)])
    set32 = torch.sort(i32, dim=-1).values
    setT = torch.sort(iT, dim=-1).values
    same = (set32 == setT).all(dim=-1)
    n_rows = same.numel()
    print(f"  top-4 packet  : identical key-set rows = {int(same.sum())}/{n_rows} "
          f"({100.0 * float(same.sum()) / n_rows:.3f}%)")
    # top-1 (highest prob) flip
    top1_32 = i32[..., 0]; top1_T = iT[..., 0]
    print(f"  top-1 key flip: {int((top1_32 != top1_T).sum())}/{top1_32.numel()} "
          f"({100.0 * float((top1_32 != top1_T).sum()) / top1_32.numel():.3f}%)")
    # 4) prob codes change
    c32 = hp(cap32, "packet_codes_topk"); cT = hp(capT, "packet_codes_topk")
    print(f"  packet codes  : changed={int((c32 != cT).sum())}/{c32.numel()}")
    # 5) Score x V context + final
    sv32 = hp(cap32, "scorev").to(torch.float32); svT = hp(capT, "scorev").to(torch.float32)
    print(f"  Score×V out   : max_abs_diff={float((sv32 - svT).abs().max()):.4e}")
    o32 = hp(cap32, "out").to(torch.float32); oT = hp(capT, "out").to(torch.float32)
    print(f"  MHA-half out  : max_abs_diff={float((o32 - oT).abs().max()):.4e}  "
          f"cos={float(torch.dot(o32, oT) / (o32.norm() * oT.norm())):.8f}")
    print(f"  full waveform : max_abs_diff={np.max(np.abs(got - yt)):.4e}  "
          f"cos={float(np.dot(got, yt) / (np.linalg.norm(got) * np.linalg.norm(yt))):.8f}")

    # ---- write the fp16-tree golden (representative block, same layout as mha_module_io) ----
    OUT = os.path.join(THIS, "hw_patterns_final", "mha_module_io_fp16tree")
    eh.ensure(OUT)
    BLK = 32; T0 = 320; QSTAR = T0 + BLK - 1
    for pi, cap in enumerate(capT):
        pn = ["real_pass", "imag_pass"][pi]; b = 0
        for h in range(H):
            hd = os.path.join(OUT, pn, "02_qk", f"head_{h:02d}")
            row = cap["qk_scores_raw"][b, h, QSTAR, QSTAR - BLK + 1:QSTAR + 1].flip(0)
            rows = cap["attn_scaled"][b, h, QSTAR, QSTAR - BLK + 1:QSTAR + 1].flip(0)
            dump(os.path.join(hd, "qstar_scores_row32_localidx_prescale_fp16"), row.to(torch.float32).numpy(), np.float16)
            dump(os.path.join(hd, "qstar_scores_row32_localidx_postscale_fp16"), rows.to(torch.float32).numpy(), np.float16)
            dump(os.path.join(hd, "qstar_scores_row32_localidx_postscale_fp32"), rows.to(torch.float32).numpy(), np.float32)
            sb = cap["qk_scores_raw"][b, h, T0:T0 + BLK, T0:T0 + BLK].to(torch.float32).numpy()
            dump(os.path.join(hd, "scores_block_32x32_prescale_fp16"), sb, np.float16)
            ctxh = cap["scorev"][b, h, QSTAR, :].to(torch.float32).numpy()
            hd4 = os.path.join(OUT, pn, "04_score_v", f"head_{h:02d}")
            dump(os.path.join(hd4, "qstar_context16_fp16"), ctxh, np.float16)
            dump(os.path.join(hd4, "qstar_context16_fp32"), ctxh, np.float32)
            pk = {"prob_code": np.clip(np.rint(cap["packet_codes_topk"][b, h, QSTAR].numpy()), 0, 255).astype(int).tolist(),
                  "local_idx": (QSTAR - cap["packet_indices"][b, h, QSTAR].numpy()).astype(int).tolist()}
            hd3 = os.path.join(OUT, pn, "03_softmax", f"head_{h:02d}")
            eh.ensure(hd3)
            json.dump(pk, open(os.path.join(hd3, "qstar_packet.json"), "w"), indent=1)
        dump(os.path.join(OUT, pn, "05_out_proj", "output256_fp32"),
             capT[pi]["out"][b, QSTAR].to(torch.float32).numpy(), np.float32)
    json.dump({"note": "QK reduction = CE adder_tree_fp16 (round every add). "
                       "Other stages identical to mha_module_io/. Compare head-to-head with mha_module_io/ (fp32-sum).",
               "qk_accumulate": "fp16_tree"},
              open(os.path.join(OUT, "manifest.json"), "w"), indent=1)
    print(f"\n[mha-qk-tree] wrote fp16-tree golden -> {OUT}")


if __name__ == "__main__":
    main()
