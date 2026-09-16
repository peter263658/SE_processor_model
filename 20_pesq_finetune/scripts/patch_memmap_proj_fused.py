#!/usr/bin/env python3
"""§3 patch: add act-k-fused split-k weight-scale + proj bias (β) const_blocks for
mattn.input_proj / output_proj into the deploy SRAM image.

Non-destructive: backs up memory_map.json + wbuf_unified_init.bin to *.pre_projfused.bak
(re-run restores from .bak first), appends the new const data into WBUF slack
[wbuf_used_bytes, capacity), leaves the existing raw-16 proj blocks in place (tagged
superseded), and updates const_blocks / const_prefetch / counts / used+slack.

Layout written (12-bit apot_6a codes, LSB-first contiguous — matches existing image):
  per branch fc_r/fc_i:  weight_scale_splitk = kreal[16] || kimag[16]  (32 codes = 48 B)
  per layer:             bn_bias proj        = beta_re[256] || beta_im[256] (fp16 = 1024 B)

Then READS BACK the patched image and reconstructs both projections vs the fp32
golden → R²=1 (proves the on-image bytes are correct).

    cd /raid/R12K41024/19_algo_final
    conda run --no-capture-output -n bcctn python scripts/patch_memmap_proj_fused.py
"""
from __future__ import annotations
import os, json, shutil
import numpy as np
import torch

HW = "/raid/R12K41024/19_algo_final/hw_patterns_final"
SR = os.path.join(HW, "sram_images")
MM = os.path.join(SR, "memory_map.json")
IMG = os.path.join(SR, "wbuf_unified_init.bin")
TBW = os.path.join(HW, "tb_pattern", "weights")
DBW = os.path.join(HW, "debug_reference", "export_reference_debug", "weights")
ACT = os.path.join(HW, "tb_pattern", "activations_fp8")
LIO = os.path.join(HW, "debug_reference", "layer_io_fp32", "layers")
APOT_BIAS = 24
PROJ = {"mattn.input_proj": (-7, -6), "mattn.output_proj": (-3, -3)}


def restore_or_backup():
    for f in (MM, IMG):
        bak = f + ".pre_projfused.bak"
        if os.path.exists(bak):
            shutil.copy2(bak, f)            # re-run: start from pristine
        else:
            shutil.copy2(f, bak)


def apot_code(a, b, hs, rs):               # build 12-bit apot_6a code
    a = (a + APOT_BIAS); b = (b + APOT_BIAS)
    assert 0 <= a <= 31 and 0 <= b <= 31, (a, b)
    return (a << 7) | (b << 2) | (int(hs) << 1) | (1 if rs < 0 else 0)


def pack12(codes):                          # contiguous 12-bit, LSB-first
    bits = []
    for c in codes:
        bits += [(c >> i) & 1 for i in range(12)]
    while len(bits) % 8:
        bits.append(0)
    return np.packbits(np.array(bits, np.uint8), bitorder="little").tobytes()


def unpack12(buf, n):
    bits = np.unpackbits(np.frombuffer(buf, np.uint8), bitorder="little")
    return [int("".join(str(x) for x in bits[i * 12:(i + 1) * 12][::-1]), 2) for i in range(n)]


def apot_value(code):
    a = ((code >> 7) & 0x1F) - APOT_BIAS
    b = ((code >> 2) & 0x1F) - APOT_BIAS
    hs = (code >> 1) & 1
    rs = code & 1
    return (2.0 ** a) + ((-(2.0 ** b) if rs else (2.0 ** b)) if hs else 0.0)


def raw_apot(layer, half):
    d = os.path.join(TBW, layer, half)
    a = np.fromfile(f"{d}/scale_apot_a_exp.int16.bin", np.int16).astype(int)
    b = np.fromfile(f"{d}/scale_apot_b_exp.int16.bin", np.int16).astype(int)
    hs = np.fromfile(f"{d}/scale_apot_has_second.u8.bin", np.uint8).astype(int)
    rs = np.fromfile(f"{d}/scale_apot_residual_sign.int8.bin", np.int8).astype(int)
    return a, b, hs, rs


def splitk_codes(layer, half, kre, kim):
    a, b, hs, rs = raw_apot(layer, half)
    kr = [apot_code(a[i] + kre, b[i] + kre, hs[i], rs[i]) for i in range(16)]
    ki = [apot_code(a[i] + kim, b[i] + kim, hs[i], rs[i]) for i in range(16)]
    return kr + ki                          # 32 codes: kreal[16] || kimag[16]


def e4(u8):
    return torch.tensor(u8, dtype=torch.uint8).view(torch.float8_e4m3fn).to(torch.float64).numpy()


def hexf32(p):
    return np.array([int(x, 16) for x in open(p).read().split()], np.uint32).view(np.float32).astype(np.float64)


def main():
    restore_or_backup()
    mm = json.load(open(MM))
    img = bytearray(open(IMG, "rb").read())
    cap = mm["wbuf_capacity_bytes"]
    base = (mm["wbuf_used_bytes"] + 3) & ~3      # 4-byte align the appended region
    cur = base
    new_blocks, new_prefetch = [], []

    # mark old raw proj weight_scale blocks superseded
    raw_off = {}
    for cb in mm["const_blocks"]:
        if cb.get("kind") == "weight_scale" and (cb["dotted"].startswith("mattn.input_proj")
                                                 or cb["dotted"].startswith("mattn.output_proj")):
            raw_off[cb["dotted"]] = cb["wbuf_offset"]
            cb["superseded_by"] = "weight_scale_splitk (act-k fused); raw=no act-k, do NOT use"

    # 1) split-k fused weight scale (48 B / branch)
    for layer, (kre, kim) in PROJ.items():
        for half in ("fc_r", "fc_i"):
            codes = splitk_codes(layer, half, kre, kim)
            blob = pack12(codes)
            assert len(blob) == 48
            img[cur:cur + 48] = blob
            dotted = f"{layer}/{half}"
            new_blocks.append({
                "dotted": dotted, "kind": "weight_scale_splitk", "region": "R6_proj_fused",
                "decode": "apot_6a", "pack": "12bit", "n_codes": 32,
                "layout": "kreal[16]||kimag[16]", "head_stride": 1, "k_real": kre, "k_imag": kim,
                "wbuf_offset": cur, "wbuf_bytes": 48, "supersedes_raw_offset": raw_off.get(dotted),
                "peg_map": {"P0": "kreal[head]", "P1(other half)": "kimag[head]",
                            "P2": "kimag[head]", "P3(other half)": "kreal[head]"},
            })
            new_prefetch.append({"unit": f"{dotted}:weight_scale_splitk",
                                 "src_wbuf_offset": cur, "bytes": 48, "decode": "apot_6a"})
            cur += 48

    # 2) proj bias beta (per-oc complex, fp16, plain add): beta_re[256] || beta_im[256]
    for layer in PROJ:
        br = np.fromfile(f"{TBW}/{layer}/fc_r/bias.fp16.bin", np.float16)
        bi = np.fromfile(f"{TBW}/{layer}/fc_i/bias.fp16.bin", np.float16)
        blob = br.tobytes() + bi.tobytes()
        assert len(blob) == 1024
        img[cur:cur + 1024] = blob
        new_blocks.append({
            "dotted": layer, "kind": "bn_bias", "subkind": "proj_complex_beta", "region": "R6_proj_fused",
            "decode": "fp16", "n_elems": 512, "layout": "beta_re[256]||beta_im[256]",
            "add": "plain (beta_re->real acc, beta_im->imag acc; NOT b_r∓b_i)",
            "wbuf_offset": cur, "wbuf_bytes": 1024,
        })
        new_prefetch.append({"unit": f"{layer}:proj_beta", "src_wbuf_offset": cur,
                             "bytes": 1024, "decode": "fp16"})
        cur += 1024

    end = cur
    assert end <= cap, f"overflow WBUF: need {end} > cap {cap}"

    mm["const_blocks"] += new_blocks
    mm["const_prefetch"] += new_prefetch
    mm["regions"]["proj_fused"] = {"start": base, "end": end, "bytes": end - base}
    mm["wbuf_used_bytes"] = end
    mm["wbuf_slack_bytes"] = cap - end
    mm["counts"]["weight_scale_codes"] += 4 * 32
    mm["counts"]["bn_bias_bytes"] += 2 * 1024
    mm["counts"]["proj_fused"] = {
        "weight_scale_splitk_blocks": 4, "codes_per_block": 32,
        "bias_blocks": 2, "bias_bytes_per_block": 1024,
        "appended_region": [base, end], "appended_bytes": end - base,
    }

    open(IMG, "wb").write(bytes(img))
    json.dump(mm, open(MM, "w"), indent=1)
    print(f"[patch] appended proj-fused const at [{base},{end}) = {end - base} B; "
          f"new slack {mm['wbuf_slack_bytes']} B")

    # ---------------- read-back verification from the PATCHED image ----------------
    img2 = np.fromfile(IMG, np.uint8)
    blk = {cb["dotted"] + ":" + cb["kind"]: cb for cb in mm["const_blocks"]
           if cb.get("region") == "R6_proj_fused"}
    print("\n[verify] reconstruct from PATCHED image vs fp32 golden:")
    for layer, (kre, kim) in PROJ.items():
        tag = layer.replace(".", "_")
        xr = e4(np.fromfile(f"{ACT}/{tag}__0/input_0.real.e4m3.u8.bin", np.uint8)).reshape(-1, 256)
        xi = e4(np.fromfile(f"{ACT}/{tag}__0/input_0.imag.e4m3.u8.bin", np.uint8)).reshape(-1, 256)
        qr = np.fromfile(f"{DBW}/{layer}/fc_r/w_int4.raw_s8.bin", np.int8).astype(np.float64).reshape(256, 256)
        qi = np.fromfile(f"{DBW}/{layer}/fc_i/w_int4.raw_s8.bin", np.int8).astype(np.float64).reshape(256, 256)

        def descale(half, which):           # which: 'kreal' or 'kimag'
            cb = blk[f"{layer}/{half}:weight_scale_splitk"]
            codes = unpack12(bytes(img2[cb["wbuf_offset"]:cb["wbuf_offset"] + 48]), 32)
            sub = codes[0:16] if which == "kreal" else codes[16:32]
            return np.repeat(np.array([apot_value(c) for c in sub]), 16)   # per-oc

        cbb = blk[f"{layer}:bn_bias"]
        bias = np.frombuffer(bytes(img2[cbb["wbuf_offset"]:cbb["wbuf_offset"] + 1024]), np.float16).astype(np.float64)
        b_r, b_i = bias[:256], bias[256:]
        P0, P1, P2, P3 = xr @ qr.T, xi @ qi.T, xi @ qr.T, xr @ qi.T
        out_re = P0 * descale("fc_r", "kreal") - P1 * descale("fc_i", "kimag") + b_r
        out_im = P2 * descale("fc_r", "kimag") + P3 * descale("fc_i", "kreal") + b_i
        g_re = hexf32(f"{LIO}/{tag}__0/output_0.real.fp32.hex").reshape(-1, 256)
        g_im = hexf32(f"{LIO}/{tag}__0/output_0.imag.fp32.hex").reshape(-1, 256)
        o = out_re + 1j * out_im; g = g_re + 1j * g_im
        r2 = 1 - np.sum(np.abs(o - g) ** 2) / np.sum(np.abs(g - g.mean()) ** 2)
        print(f"  {layer}: R2={r2.real:.8f}  max_abs={np.max(np.abs(o - g)):.3e}")
    print("\n[patch] memory_map.json + wbuf_unified_init.bin patched (backups *.pre_projfused.bak)")


if __name__ == "__main__":
    main()
