#!/usr/bin/env python3
"""Generate act-k-FUSED per-head APoT codes + bias (β) for mattn.input_proj /
output_proj, and VERIFY end-to-end against the fp32 golden by reconstructing the
projection from INT4 weight codes × e4m3 activation codes × fused descale + bias.

Closes the two RTL gaps (docs/MHA_GOLDEN_HANDOFF_2026-06-08.md Part D):
  D1  act-k fusion: emit scale_apot_{a,b}_exp_fused_act_kreal/kimag (like decoder)
  D2  bias        : emit bn_bias_{r,i}_fused (= b_r, b_i) for the PPU β path

Per-PEG fused descale (P0=fc_r·x_real, P1=fc_i·x_imag, P2=fc_r·x_imag, P3=fc_i·x_real):
  out_re = (ws_r·2^kre)·Σ q_r·xr_e4m3  −  (ws_i·2^kim)·Σ q_i·xi_e4m3  + b_r
  out_im = (ws_r·2^kim)·Σ q_r·xi_e4m3  +  (ws_i·2^kre)·Σ q_i·xr_e4m3  + b_i
  head(oc)=oc//16 ; ws = APoT 2^a±2^b ; fuse = (a+k, b+k).

Outputs codes to hw_patterns_final/proj_fused_codes/<layer>/<half>/ and additively
into the tb_pattern weight dirs (new filenames only). Reconstruction uses fp32
accumulation (HW uses the conv fp16 tree — orthogonal; this isolates the codes).

    cd /raid/R12K41024/19_algo_final
    conda run --no-capture-output -n bcctn python scripts/gen_proj_fused_codes_and_verify.py
"""
from __future__ import annotations
import os, json
import numpy as np
import torch

THIS = "/raid/R12K41024/19_algo_final"
HW = os.path.join(THIS, "hw_patterns_final")
TBW = os.path.join(HW, "tb_pattern", "weights")
DBW = os.path.join(HW, "debug_reference", "export_reference_debug", "weights")
ACT = os.path.join(HW, "tb_pattern", "activations_fp8")
LIO = os.path.join(HW, "debug_reference", "layer_io_fp32", "layers")
OUT = os.path.join(HW, "proj_fused_codes")

APOT_BIAS = 24  # apot6a: code[11:7]=a+24, [6:2]=b+24, [1]=has_second, [0]=res_sign


def rd(path, dt):
    return np.fromfile(path, dt)


def e4m3_decode(u8):
    return torch.tensor(u8, dtype=torch.uint8).view(torch.float8_e4m3fn).to(torch.float64).numpy()


def hexf32(path):
    return np.array([int(x, 16) for x in open(path).read().split()], np.uint32).view(np.float32)


def apot_value(a_exp, b_exp, has_second, res_sign):
    """2^a + (has_second ? res_sign·2^b : 0).  res_sign in {-1,0,+1}."""
    v = np.power(2.0, a_exp.astype(np.float64))
    second = has_second.astype(bool)
    v = v + np.where(second, res_sign.astype(np.float64) * np.power(2.0, b_exp.astype(np.float64)), 0.0)
    return v


def pack_apot6a(a_exp, b_exp, has_second, res_sign):
    a = (a_exp + APOT_BIAS).astype(np.int32)
    b = (b_exp + APOT_BIAS).astype(np.int32)
    assert a.min() >= 0 and a.max() <= 31, f"a+24 out of 5-bit range: {a.min()}..{a.max()}"
    assert b.min() >= 0 and b.max() <= 31, f"b+24 out of 5-bit range: {b.min()}..{b.max()}"
    rs_bit = (res_sign < 0).astype(np.int32)        # 1 => minus 2^b
    code = (a << 7) | (b << 2) | (has_second.astype(np.int32) << 1) | rs_bit
    return code.astype(np.uint16)


def write(path_noext, arr, dt):
    os.makedirs(os.path.dirname(path_noext), exist_ok=True)
    a = np.ascontiguousarray(arr, dt)
    a.tofile(path_noext + ".bin")
    # hex sidecar (same convention as the package)
    fmt = {np.dtype(np.int16): "{:04X}", np.dtype(np.uint16): "{:04X}",
           np.dtype(np.float16): "{:04X}", np.dtype(np.float32): "{:08X}"}[a.dtype]
    view = {np.dtype(np.int16): np.uint16, np.dtype(np.uint16): np.uint16,
            np.dtype(np.float16): np.uint16, np.dtype(np.float32): np.uint32}[a.dtype]
    with open(path_noext + ".hex", "w") as f:
        for v in a.reshape(-1).view(view):
            f.write(fmt.format(int(v)) + "\n")


def load_half(layer, half):
    tb = os.path.join(TBW, layer, half)
    a = rd(os.path.join(tb, "scale_apot_a_exp.int16.bin"), np.int16).astype(np.int64)
    b = rd(os.path.join(tb, "scale_apot_b_exp.int16.bin"), np.int16).astype(np.int64)
    hs = rd(os.path.join(tb, "scale_apot_has_second.u8.bin"), np.uint8)
    rsg = rd(os.path.join(tb, "scale_apot_residual_sign.int8.bin"), np.int8).astype(np.int64)
    bias = rd(os.path.join(tb, "bias.fp16.bin"), np.float16).astype(np.float64)
    qs8 = rd(os.path.join(DBW, layer, half, "w_int4.raw_s8.bin"), np.int8).astype(np.float64).reshape(256, 256)
    return dict(a=a, b=b, hs=hs, rs=rsg, bias=bias, q=qs8)


def recover_k(layer):
    out = {}
    for half, name in [("re", "real"), ("im", "imag")]:
        xf = hexf32(os.path.join(LIO, f"{layer.replace('.', '_')}__0", f"input_0.{name}.fp32.hex")).astype(np.float64)
        u8 = rd(os.path.join(ACT, f"{layer.replace('.', '_')}__0", f"input_0.{name}.e4m3.u8.bin"), np.uint8)
        e4 = e4m3_decode(u8)
        nz = (e4 != 0) & (xf != 0)
        k = int(np.round(np.median(np.log2(np.abs(xf[nz] / e4[nz])))))
        assert np.max(np.abs(e4 * (2.0 ** k) - xf)) == 0.0, f"{layer} {name}: e4m3*2^{k} != fp32 (exact check)"
        out[half] = (k, e4.reshape(-1, 256), xf.reshape(-1, 256))
    return out


def golden_out(layer):
    d = os.path.join(LIO, f"{layer.replace('.', '_')}__0")
    return (hexf32(os.path.join(d, "output_0.real.fp32.hex")).astype(np.float64).reshape(-1, 256),
            hexf32(os.path.join(d, "output_0.imag.fp32.hex")).astype(np.float64).reshape(-1, 256))


def r2(a, b):
    a, b = a.reshape(-1), b.reshape(-1)
    return 1.0 - np.sum((a - b) ** 2) / (np.sum((b - b.mean()) ** 2) + 1e-30)


def per_head(vec16, oc=256, nh=16):
    return np.repeat(vec16, oc // nh)        # head(oc)=oc//16


def process(layer):
    print(f"\n================ {layer} ================")
    fc_r, fc_i = load_half(layer, "fc_r"), load_half(layer, "fc_i")
    kinfo = recover_k(layer)
    kre, xr_e4, _ = kinfo["re"]
    kim, xi_e4, _ = kinfo["im"]
    print(f"  act-k: k_real={kre}  k_imag={kim}   (e4m3·2^k == fp32 golden input, exact)")

    # ---- generate fused codes (a+k, b+k); has_second/res_sign unchanged ----
    layers_codes = {}
    for half, d in [("fc_r", fc_r), ("fc_i", fc_i)]:
        for kname, k in [("kreal", kre), ("kimag", kim)]:
            a_f = (d["a"] + k).astype(np.int16)
            b_f = (d["b"] + k).astype(np.int16)
            code = pack_apot6a(a_f, b_f, d["hs"], d["rs"])
            base = os.path.join(OUT, layer, half)
            # match decoder filenames (a_exp/b_exp fused), + a packed 12-bit code
            write(os.path.join(base, f"scale_apot_a_exp_fused_act_{kname}.int16"), a_f, np.int16)
            write(os.path.join(base, f"scale_apot_b_exp_fused_act_{kname}.int16"), b_f, np.int16)
            write(os.path.join(base, f"scale_apot_code_fused_act_{kname}.u16"), code, np.uint16)
            # additive drop into the shipped tb_pattern dir (new filenames only)
            tb_base = os.path.join(TBW, layer, half)
            write(os.path.join(tb_base, f"scale_apot_a_exp_fused_act_{kname}.int16"), a_f, np.int16)
            write(os.path.join(tb_base, f"scale_apot_b_exp_fused_act_{kname}.int16"), b_f, np.int16)
            layers_codes[(half, kname)] = (a_f, b_f)
        # copy shared fields for convenience
        write(os.path.join(OUT, layer, half, "scale_apot_has_second.u8"), d["hs"], np.uint16)  # store as u16 hex-friendly
    # ---- bias (β) for the PPU: proj is a plain complex linear -> direct ----
    b_r, b_i = fc_r["bias"], fc_i["bias"]      # (256,)
    write(os.path.join(OUT, layer, "bn_bias_r_fused.fp16"), b_r.astype(np.float16), np.float16)
    write(os.path.join(OUT, layer, "bn_bias_i_fused.fp16"), b_i.astype(np.float16), np.float16)
    write(os.path.join(TBW, layer, "bn_bias_r_fused.fp16"), b_r.astype(np.float16), np.float16)
    write(os.path.join(TBW, layer, "bn_bias_i_fused.fp16"), b_i.astype(np.float16), np.float16)
    # identity BN-alpha code (α=2^0): a=0,b=0,has_second=0,res_sign=0
    alpha_id = int((0 + APOT_BIAS) << 7 | (0 + APOT_BIAS) << 2)

    # ---- descale per-oc from the GENERATED fused codes ----
    def descale(half_d, k):
        v16 = apot_value(half_d["a"] + k, half_d["b"] + k, half_d["hs"], half_d["rs"])  # (16,)
        return per_head(v16)                          # (256,)
    ds_r_kre = descale(fc_r, kre); ds_i_kim = descale(fc_i, kim)   # P0, P1
    ds_r_kim = descale(fc_r, kim); ds_i_kre = descale(fc_i, kre)   # P2, P3

    # ---- reconstruct (fp32-domain accumulation) ----
    qr, qi = fc_r["q"], fc_i["q"]                    # (oc=256, ic=256)
    P0 = xr_e4 @ qr.T          # (T,256) Σ q_r·xr_e4m3
    P1 = xi_e4 @ qi.T
    P2 = xi_e4 @ qr.T
    P3 = xr_e4 @ qi.T
    out_re = P0 * ds_r_kre[None, :] - P1 * ds_i_kim[None, :] + b_r[None, :]
    out_im = P2 * ds_r_kim[None, :] + P3 * ds_i_kre[None, :] + b_i[None, :]
    g_re, g_im = golden_out(layer)
    out = out_re + 1j * out_im; g = g_re + 1j * g_im
    print(f"  [VERIFY fused codes + bias]  R2={r2(out, g):.8f}  "
          f"max_abs={np.max(np.abs(out - g)):.3e}  (vs fp32 golden output {g.shape})")

    # ---- negative control: RAW apot (no act-k fusion) ----
    rs0 = per_head(apot_value(fc_r["a"], fc_r["b"], fc_r["hs"], fc_r["rs"]))
    is0 = per_head(apot_value(fc_i["a"], fc_i["b"], fc_i["hs"], fc_i["rs"]))
    bad_re = P0 * rs0[None, :] - P1 * is0[None, :] + b_r[None, :]
    bad_im = P2 * rs0[None, :] + P3 * is0[None, :] + b_i[None, :]
    print(f"  [neg ctrl RAW apot, no k]    R2={r2(bad_re + 1j * bad_im, g):+.4f}  (should be ~0: off by 2^k)")
    # ---- negative control: drop bias ----
    nb = (out_re - b_r[None, :]) + 1j * (out_im - b_i[None, :])
    print(f"  [neg ctrl drop bias]         R2={r2(nb, g):+.6f}  (bias |max| re={np.abs(b_r).max():.3f} im={np.abs(b_i).max():.3f})")

    return dict(layer=layer, k_real=kre, k_imag=kim, R2=float(r2(out, g)),
                alpha_identity_code_hex=f"0x{alpha_id:03X}",
                fused_a_exp={f"{h}_{kn}": int(layers_codes[(h, kn)][0][0]) if np.unique(layers_codes[(h, kn)][0]).size == 1
                             else layers_codes[(h, kn)][0].tolist()
                             for h in ("fc_r", "fc_i") for kn in ("kreal", "kimag")})


def main():
    os.makedirs(OUT, exist_ok=True)
    summary = [process("mattn.input_proj"), process("mattn.output_proj")]
    man = {"note": "act-k-fused per-head APoT codes + PPU bias (β) for the outer complex "
                   "projections. PEG map: P0=fc_r·x_real(kreal) P1=fc_i·x_imag(kimag) "
                   "P2=fc_r·x_imag(kimag) P3=fc_i·x_real(kreal). fuse=(a+k,b+k); "
                   "has_second/res_sign unchanged. bias direct (beta_re=b_r, beta_im=b_i). "
                   "Files also dropped additively into tb_pattern/weights/<layer>/.",
           "apot6a_code": "[11:7]=a+24 [6:2]=b+24 [1]=has_second [0]=res_sign(1=minus)",
           "layers": summary}
    json.dump(man, open(os.path.join(OUT, "manifest.json"), "w"), indent=2)
    print(f"\n[gen] wrote fused codes + bias -> {OUT}  (and additively into tb_pattern)")
    print("[gen] summary:", json.dumps({s["layer"]: {"k": [s["k_real"], s["k_imag"]], "R2": round(s["R2"], 8)} for s in summary}))


if __name__ == "__main__":
    main()
