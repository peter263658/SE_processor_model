#!/usr/bin/env python3
"""Step-by-step deploy softmax golden for RTL Softmax_v3 refactor (Consult Q3).

Captures the EXACT deploy `_shift_topk_softmax` intermediates (not a recompute) for
the representative query t*=351, all 16 heads, both passes:
  topk scores -> row_max -> x=score-max (fp16) -> 1.5x (fp16 RNE) -> clamp[-8,0]
  -> k=floor -> weight=2^k -> denom=round_pow2(sum) -> prob -> q0.8 code.

Also answers the fp16-vs-exact ×1.5 question: compares deploy's fp16-rounded 1.5x
(then floor) against an exact `x + x>>1` (then floor) over ALL rows/heads, to tell
the advisor whether `x + x>>1` in wide fixed-point is bit-exact or needs fp16 RNE.

    cd /raid/R12K41024/19_algo_final
    conda run --no-capture-output -n bcctn python scripts/gen_softmax_stepwise_golden.py
"""
from __future__ import annotations
import os, sys, json
import numpy as np
import torch

PKG = "/raid/R12K41024/18_w4a8_pow2_mixed_lmul_hw_handoff_20260530"
S2H = os.path.join(PKG, "software_to_hardware_18")
THIS = "/raid/R12K41024/19_algo_final"
CODE = os.path.join(THIS, "code")
OUT = os.path.join(THIS, "hw_patterns_final", "mha_module_io", "softmax_stepwise")
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
import DCNN.utils.complexPyTorch.complexLayers as CL     # noqa

QSTAR = 351
CAP = []
_orig = CL._shift_topk_softmax


def capture_shift_topk(attn, valid_mask, top_k, pow2_denom=True, denom_rounding="nearest",
                       clamp_min=-16.0, use_mantissa=True):
    # verbatim deploy body + intermediate capture (no behavioral change; returns orig)
    k = min(int(top_k), attn.size(-1))
    masked = attn.masked_fill(~valid_mask, float("-inf"))
    topk_scores, topk_idx = torch.topk(masked, k=k, dim=-1)
    topk_valid = torch.isfinite(topk_scores)
    row_max = topk_scores[..., :1].masked_fill(~topk_valid[..., :1], 0.0)
    x = (topk_scores - row_max).masked_fill(~topk_valid, 0.0)
    scaled = (x * 1.5).clamp(min=clamp_min, max=0.0)
    integer = torch.floor(scaled)
    mantissa = (1.0 + (scaled - integer)) if use_mantissa else torch.ones_like(scaled)
    weights = torch.ldexp(mantissa, integer.to(torch.int32)).masked_fill(~topk_valid, 0.0)
    denom = weights.sum(dim=-1, keepdim=True)
    denom_norm = CL._round_power_of_two(denom, weights.dtype, denom_rounding) if pow2_denom else denom
    # exact x+x>>1 (fp64) floor — to compare vs deploy fp16-rounded 1.5x
    x64 = x.to(torch.float64)
    integer_exact = torch.floor((x64 * 1.5).clamp(min=clamp_min, max=0.0))
    CAP.append({
        "topk_scores": topk_scores.detach().clone(), "topk_idx": topk_idx.detach().clone(),
        "row_max": row_max.detach().clone(), "x": x.detach().clone(),
        "scaled": scaled.detach().clone(), "integer": integer.detach().clone(),
        "weights": weights.detach().clone(), "denom": denom.detach().clone(),
        "denom_norm": denom_norm.detach().clone(), "topk_valid": topk_valid.detach().clone(),
        "integer_exact": integer_exact.detach().clone(),
    })
    return _orig(attn, valid_mask, top_k, pow2_denom=pow2_denom, denom_rounding=denom_rounding,
                 clamp_min=clamp_min, use_mantissa=use_mantissa)


def main():
    eh.ensure(OUT)
    model, *_ = eh.build_model(); model.eval()
    import soundfile as sf
    a, _ = sf.read("/raid/R12K41024/BCCTN/Dataset/noisy_testset/snr_0dB/p225_p225_009_az-40_snr+0.0.wav",
                   dtype="float32", always_2d=True)
    a = a[:, :2] if a.shape[1] >= 2 else np.repeat(a, 2, 1)
    a = a[:32000] if a.shape[0] >= 32000 else np.concatenate([a, np.zeros((32000 - a.shape[0], 2), np.float32)])
    x = torch.from_numpy(a.T).unsqueeze(0).contiguous()
    CL._shift_topk_softmax = capture_shift_topk
    with torch.no_grad():
        model(x)

    # _shift_topk_softmax is called once per pass (tied -> 2 calls: real, imag).
    assert len(CAP) >= 2, len(CAP)
    H = CAP[0]["x"].shape[1]
    PROB_SCALE = 256

    # fp16-vs-exact ×1.5 audit over ALL rows/heads/passes
    mism = 0; tot = 0
    for c in CAP:
        v = c["topk_valid"]
        d = (c["integer"] != c["integer_exact"]) & v
        mism += int(d.sum()); tot += int(v.sum())
    print(f"[1.5x audit] deploy fp16-rounded(1.5x)->floor vs exact(x+x>>1)->floor: "
          f"{mism}/{tot} top-k entries differ across all rows/heads/passes")

    passes = {0: "real_pass", 1: "imag_pass"}
    table = {}
    for pi in (0, 1):
        c = CAP[pi]; b = 0
        rows = []
        for h in range(H):
            ts = c["topk_scores"][b, h, QSTAR].to(torch.float64).numpy()
            idx = c["topk_idx"][b, h, QSTAR].numpy()
            rmax = float(c["row_max"][b, h, QSTAR, 0])
            xv = c["x"][b, h, QSTAR].to(torch.float64).numpy()
            scaled = c["scaled"][b, h, QSTAR].to(torch.float64).numpy()
            kk = c["integer"][b, h, QSTAR].to(torch.int64).numpy()
            wt = c["weights"][b, h, QSTAR].to(torch.float64).numpy()
            denom = float(c["denom"][b, h, QSTAR, 0]); dnorm = float(c["denom_norm"][b, h, QSTAR, 0])
            prob = wt / max(dnorm, 1e-12)
            code = np.clip(np.rint(prob * PROB_SCALE), 0, 255).astype(int)
            local_idx = (QSTAR - idx).astype(int)
            rows.append({
                "head": h, "row_max_fp16": rmax,
                "topk_localidx": local_idx.tolist(), "topk_score_fp16": [float(s) for s in ts],
                "x_fp16": [float(t) for t in xv], "scaled_1p5x_fp16": [float(s) for s in scaled],
                "k_floor": kk.tolist(), "weight_2pk": [float(w) for w in wt],
                "denom_raw": denom, "denom_pow2": dnorm,
                "prob": [float(p) for p in prob], "prob_code_q08": code.tolist(),
            })
        table[passes[pi]] = rows
        # also write a flat hex/dec golden per pass: k, code per head (topk order)
        kall = np.array([r["k_floor"] for r in rows], np.int16).reshape(-1)
        call = np.array([r["prob_code_q08"] for r in rows], np.uint8).reshape(-1)
        eh.write_bin(os.path.join(OUT, f"{passes[pi]}_k_floor.int16.bin"), kall)
        eh.write_bin(os.path.join(OUT, f"{passes[pi]}_prob_code_q08.u8.bin"), call)

    json.dump({"query_abs": QSTAR, "prob_scale": PROB_SCALE,
               "softmax_mode": "shift_topk_pow2_clamp8_nomantissa",
               "formula": "x=score-row_max(fp16); scaled=clamp(fp16(1.5x),-8,0); k=floor(scaled); "
                          "w=2^k; denom=round_pow2(sum w); prob=w/denom; code=round(prob*256)",
               "one_five_x_audit": {"differ": mism, "total": tot,
                                    "note": "0 => exact (x+x>>1)->floor matches deploy on golden"},
               "passes": table}, open(os.path.join(OUT, "softmax_stepwise_golden.json"), "w"), indent=1)

    # cross-check: recomputed code == shipped deploy packet code
    base = os.path.join(THIS, "hw_patterns_final", "mha_module_io")
    ok = 0; tot_h = 0
    for pi in (0, 1):
        for h in range(H):
            dep = json.load(open(os.path.join(base, passes[pi], "03_softmax", f"head_{h:02d}", "qstar_packet.json")))
            mine = table[passes[pi]][h]
            # match by local_idx -> code
            dcode = dict(zip(dep["local_idx"], dep["prob_code"]))
            mcode = dict(zip(mine["topk_localidx"], mine["prob_code_q08"]))
            tot_h += 1; ok += int(dcode == mcode)
    print(f"[cross-check] stepwise code == shipped deploy packet: {ok}/{tot_h} head-passes")
    print(f"[gen] wrote {OUT}")
    # print advisor's head-1 example (real_pass)
    r1 = table["real_pass"][1]
    print("\nhead 1 (real_pass) example:")
    print("  local_idx ", r1["topk_localidx"]); print("  score_fp16", [round(s, 5) for s in r1["topk_score_fp16"]])
    print("  x=score-max", [round(t, 5) for t in r1["x_fp16"]]); print("  1.5x clamp", [round(s, 5) for s in r1["scaled_1p5x_fp16"]])
    print("  k=floor   ", r1["k_floor"]); print("  prob_code ", r1["prob_code_q08"], " denom_pow2", r1["denom_pow2"])


if __name__ == "__main__":
    main()
