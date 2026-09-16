#!/usr/bin/env python3
"""
verify_fp16_tree_headroom.py — per-layer check of the 32-way FP16 saturating
adder tree (PEG.sv::adder_tree_fp16 / pe_pair) used in the CE W4A8 datapath.

Closes the open item in RTL_HANDOFF_PE_LANE_DATAPATH_2026-06-06.md §5.1:
  "每層 max |32-group pre-descale partial sum|" — extends handoff §3 (which only
  did encoder conv pointwise) to ALL e4m3-reconstructable W4A8/FP8 layers.

For every (weight tensor, captured e4m3 activation) it rebuilds the PRE-DESCALE
group partial sum exactly as the hardware tree sees it:

    leaf[ic] = q[oc,ic] * e4m3_decode(act_raw_code[ic,pos])      # exact in FP16
    T_g      = FP16_saturating_tree( leaf over a 32-ic group )   # pre-descale

and reports, per layer:
  * max |tree node|  (fp32 truth)            -> overflow risk vs FP16 max 65504
  * #nodes that would exceed 65504           -> would Inf/saturate in pure FP16
  * max rel-err  FP16-saturating-tree vs fp32 exact   -> the §F2 accumulation cost

NO descale / dealer / OACC here — descale is post-tree, so the tree only ever sees
the pre-descale (amplified) products. That is the only place the FP16 cliff lives.

Pure numpy; reads hw_patterns_final/. Run from repo root or scripts/.
"""
import json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAT  = os.path.join(ROOT, "hw_patterns_final")
WDIR = os.path.join(PAT, "debug_reference", "export_reference_debug", "weights")
ADIR = os.path.join(PAT, "tb_pattern", "activations_fp8")
FP16_MAX = 65504.0
GROUP    = 32
TARGET   = 32768.0   # pre-tree partial budget (2x margin under FP16 max) the preshift aims for

# ---------------------------------------------------------------- e4m3 decode
def e4m3_decode(codes):
    """OCP e4m3 (1-4-3, bias 7, max 448, NaN @ 0x7F/0xFF) -> fp32 value array."""
    c = codes.astype(np.int32)
    s = (c >> 7) & 1
    e = (c >> 3) & 0xF
    m = c & 7
    sign = np.where(s == 1, -1.0, 1.0)
    sub  = (2.0 ** -6) * (m / 8.0)
    nor  = (2.0 ** (e - 7)) * (1.0 + m / 8.0)
    val  = np.where(e == 0, sub, nor)
    val  = np.where((e == 0xF) & (m == 7), np.nan, val)     # NaN
    return (sign * val).astype(np.float32)

# ------------------------------------------------- FP16 saturating adder tree
def fp16_sat_tree(leaves):
    """
    leaves: (N, 32) fp32, each exactly representable in fp16 (the per-element
    products). Replicates adder_tree_fp16: pairwise (0,1)(2,3)... reduction with
    RNE-to-fp16 + saturate(Inf -> +-65504) at every node.

    Returns:
      final32   (N,) fp32 exact-order tree sum (truth, fp16-order but unrounded)
      final16   (N,) fp32 holding the fp16-saturating tree result
      maxnode32 (N,) max |node value| over the whole tree (fp32 truth) per row
      ovf       (N,) bool: any node's true magnitude > FP16_MAX (would saturate)
    """
    cur32 = leaves.astype(np.float32).copy()
    cur16 = leaves.astype(np.float32).copy()       # == cur32 at leaf level (exact)
    maxnode = np.abs(cur32).max(axis=1)
    ovf     = np.zeros(leaves.shape[0], dtype=bool)
    n = leaves.shape[1]
    while n > 1:
        a32, b32 = cur32[:, 0::2], cur32[:, 1::2]
        cur32 = a32 + b32                                   # fp32 truth (this order)
        a16, b16 = cur16[:, 0::2], cur16[:, 1::2]
        s = a16 + b16                                       # fp32 sum of fp16 nodes
        with np.errstate(over="ignore"):
            s16 = s.astype(np.float16).astype(np.float32)   # RNE to fp16 (Inf handled below)
        s16 = np.where(np.isinf(s16), np.sign(s) * FP16_MAX, s16)   # saturate
        cur16 = s16
        node_abs = np.abs(cur32)
        maxnode = np.maximum(maxnode, node_abs.max(axis=1))
        ovf |= (node_abs > FP16_MAX).any(axis=1)
        n //= 2
    return cur32[:, 0], cur16[:, 0], maxnode, ovf

# ------------------------------------------------------------- weight loaders
def load_weight(layer_dir):
    """Return (mode, w2d[oc,ic_or_tap], is_depthwise) or None. mode in int4/fp8."""
    mpath = os.path.join(layer_dir, "meta.json")
    if not os.path.exists(mpath):
        return None
    meta = json.load(open(mpath))
    ws   = meta.get("weight_shape")
    dw   = bool(meta.get("is_depthwise"))
    raw  = os.path.join(layer_dir, "w_int4.raw_s8.bin")
    fp8  = os.path.join(layer_dir, "w_fp8.e4m3.code_value.fp32.bin")
    if os.path.exists(raw):
        w = np.fromfile(raw, dtype=np.int8).astype(np.float32)
        mode = "int4"
    elif os.path.exists(fp8):
        w = np.fromfile(fp8, dtype=np.float32)              # e4m3 code value (pre-descale)
        mode = "fp8"
    else:
        return None
    if ws is None:
        return None
    w = w.reshape(ws)
    if dw:                       # [oc,1,k,1] -> [oc, k]
        w2d = w.reshape(ws[0], -1)
    else:                        # [oc,ic,1,1] / [oc,ic] -> [oc, ic]
        w2d = w.reshape(ws[0], ws[1])
    return mode, w2d, dw

# --------------------------------------------------------- activation loaders
def load_acts(act_root, base):
    """Return list of (tag, value[C,P]) over every call-index dir for `base`."""
    out = []
    if not os.path.isdir(act_root):
        return out
    for d in sorted(os.listdir(act_root)):
        if not (d == base or d.startswith(base + "__")):
            continue
        p = os.path.join(act_root, d)
        if not os.path.isdir(p):
            continue
        files = os.listdir(p)
        for fn in files:
            if not fn.endswith(".e4m3.u8.bin"):
                continue
            codes = np.fromfile(os.path.join(p, fn), dtype=np.uint8)
            out.append((d + ":" + fn.split('.')[0], codes))
    return out

# --------------------------------------------------------------- per-layer run
def scan_pointwise(name, w2d, acts, preshift=0):
    """w2d:[oc,ic]; acts:list of (tag, codes flat = C*P). Contract ic in 32-groups.
    preshift = per-TENSOR pre-tree pow2 (signed): every product is scaled by 2^preshift
    before the FP16 tree, modelling pe_pair's i_preshift (the FP16-overflow fix). 0 = off."""
    oc, ic = w2d.shape
    psc = 2.0 ** preshift
    res = dict(maxnode=0.0, maxleaf=0.0, ovf=0, relerr=0.0, abserr=0.0, n=0, peakat="", preshift=preshift)
    for tag, codes in acts:
        if codes.size % ic != 0:                # activation channels must match ic
            # try treat as [P, ic] (linear) vs [ic, P] (conv) — both flatten the same
            continue
        P = codes.size // ic
        a = e4m3_decode(codes).reshape(ic, P) if codes.size == ic * P else None
        # conv act is [1,ic,1,P] -> ic-major; linear act [1,P,ic] -> ic is last.
        # Detect by file layout: conv files decode to ic*P with ic outer already.
        if a is None:
            continue
        # If this looks like a linear [P,ic] (mattn), reshape accordingly:
        if "input_proj" in name or "output_proj" in name:
            a = e4m3_decode(codes).reshape(P, ic).T          # -> [ic, P]
        for g0 in range(0, ic, GROUP):
            gw = min(GROUP, ic - g0)
            wg = w2d[:, g0:g0+gw]                            # [oc, gw]
            ag = a[g0:g0+gw, :]                              # [gw, P]
            prod = (wg[:, :, None] * ag[None, :, :]) * psc   # [oc, gw, P], pre-tree pow2
            prod = np.transpose(prod, (0, 2, 1)).reshape(oc * P, gw)
            if gw < GROUP:
                prod = np.pad(prod, ((0, 0), (0, GROUP - gw)))
            res["maxleaf"] = max(res["maxleaf"], float(np.abs(prod).max()))
            f32, f16, mx, ov = fp16_sat_tree(prod)
            if mx.max() > res["maxnode"]:
                res["maxnode"] = float(mx.max()); res["peakat"] = f"{tag} g{g0//GROUP}"
            res["ovf"]   += int(ov.sum())
            res["abserr"] = max(res["abserr"], float(np.abs(f16 - f32).max()))
            big = np.abs(f32) > 1.0                          # meaningful denom only
            if big.any():
                rel = np.abs(f16[big] - f32[big]) / np.abs(f32[big])
                res["relerr"] = max(res["relerr"], float(rel.max()))
            res["n"]     += f32.size
    return res

def scan_depthwise(name, w2d, acts):
    """w2d:[oc,K]; 5-tap accumulate (OACC path, NOT the 32-tree). Report bound."""
    oc, K = w2d.shape
    res = dict(maxsum=0.0, n=0)
    for tag, codes in acts:
        if codes.size % oc != 0:
            continue
        P = codes.size // oc
        a = e4m3_decode(codes).reshape(oc, P)
        ap = np.pad(a, ((0, 0), (K - 1, 0)))                 # causal left pad
        acc = np.zeros((oc, P), dtype=np.float32)
        for t in range(K):
            acc += w2d[:, t:t+1] * ap[:, t:t+P]
        res["maxsum"] = max(res["maxsum"], float(np.abs(acc).max()))
        res["n"] += acc.size
    return res

def fixed_scan(name, w2d, acts, mode):
    """Run the tree with the FIX: per-TENSOR descale paths (FP8 outlier pw) get a
    per-tensor pre-tree pow2 preshift sized to land the partial <= TARGET; per-GROUP
    INT4 pointwise CANNOT pre-shift (雷②) so stays preshift=0 (it never overflows)."""
    r0 = scan_pointwise(name, w2d, acts, preshift=0)
    can_preshift = (mode == "fp8")            # FP8 outlier pw descale is per-tensor
    if can_preshift and r0["maxnode"] > TARGET:
        s = int(np.floor(np.log2(TARGET / r0["maxnode"])))     # <= -1
        return scan_pointwise(name, w2d, acts, preshift=s), r0
    return r0, r0

# ----------------------------------------------------------------------- main
def main():
    pw_rows, dw_rows, skipped = [], [], []
    layers = sorted(os.listdir(WDIR))
    for L in layers:
        ldir = os.path.join(WDIR, L)
        if not os.path.isdir(ldir):
            continue
        base = L.replace(".", "_")
        # mattn.input_proj / output_proj : weights live in fc_r / fc_i subdirs
        subdirs = [d for d in ("fc_r", "fc_i") if os.path.isdir(os.path.join(ldir, d))]
        if subdirs:
            for sd in subdirs:
                wl = load_weight(os.path.join(ldir, sd))
                if wl is None:
                    skipped.append(L + "/" + sd); continue
                mode, w2d, _ = wl
                acts = load_acts(ADIR, base)                 # real+imag both included
                if not acts:
                    skipped.append(L + "/" + sd + " (no act)"); continue
                r, r0 = fixed_scan(L, w2d, acts, mode)
                pw_rows.append((f"{L}.{sd}", mode, w2d.shape, r, r0))
            continue
        wl = load_weight(ldir)
        if wl is None:
            skipped.append(L + " (no loadable weight: internal-attn/fp16)"); continue
        mode, w2d, dw = wl
        acts = load_acts(ADIR, base)
        if not acts:
            skipped.append(L + " (no e4m3 act in dump)"); continue
        if dw:
            r = scan_depthwise(L, w2d, acts)
            dw_rows.append((L, mode, w2d.shape, r))
        else:
            r, r0 = fixed_scan(L, w2d, acts, mode)
            pw_rows.append((L, mode, w2d.shape, r, r0))

    # ---- report
    print("=" * 100)
    print("FP16 32-way tree headroom — PRE-DESCALE group partial (vs FP16 max 65504)")
    print("golden input: p225_p225_009_az-40_snr+0.0.wav  | all call-indices, all 32-ic groups")
    print("=" * 100)
    print("\n### POINTWISE / PROJ  (true 32-ic contraction -> the FP16 tree)  [WITH pre-tree preshift fix]")
    print("  pre|node| = pre-fix max tree node;  preshift = per-tensor pow2 applied in pe_pair;")
    print("  post|node| = max tree node after fix;  ovf# = overflow nodes AFTER fix")
    hdr = (f"{'layer':<40}{'mode':<5}{'shape':<12}{'pre|node|':>11}{'preshift':>9}"
           f"{'post|node|':>11}{'/65504':>9}{'ovf#':>7}")
    print(hdr); print("-" * len(hdr))
    worst = 0.0; total_ovf = 0; pre_worst = 0.0
    for name, mode, shp, r, r0 in sorted(pw_rows, key=lambda x: -x[3]["maxnode"]):
        ratio = r["maxnode"] / FP16_MAX
        ps = r.get("preshift", 0)
        pstxt = f"2^{ps}" if ps != 0 else "—"
        flag = "  <<< OVERFLOW" if r["ovf"] > 0 else ""
        print(f"{name:<40}{mode:<5}{str(tuple(shp)):<12}{r0['maxnode']:>11.1f}{pstxt:>9}"
              f"{r['maxnode']:>11.1f}{ratio:>8.2f}x{r['ovf']:>7d}{flag}")
        worst = max(worst, r["maxnode"]); total_ovf += r["ovf"]
        pre_worst = max(pre_worst, r0["maxnode"])
    print("-" * len(hdr))
    worst_errpk_i = max((r["abserr"]/max(r["maxnode"],1e-9))
                        for _,m,_,r,_ in pw_rows if m == "int4")
    print(f"PRE-fix worst max|node| = {pre_worst:.1f} ({pre_worst/FP16_MAX:.1f}x)  ->  "
          f"POST-fix worst max|node| = {worst:.1f} ({worst/FP16_MAX:.2f}x)   overflow nodes AFTER fix = {total_ovf}")
    print(f"  INT4 (per-group, preshift=0) worst fp16-accum err/peak = {worst_errpk_i:.2e}  "
          f"(~{-np.log2(max(worst_errpk_i,1e-12)):.0f} effective bits kept).")

    print("\n### DEPTHWISE  (5-tap accumulate in OACC, NOT the 32-tree; shown for completeness)")
    hdr2 = f"{'layer':<42}{'mode':<6}{'shape':<12}{'max|5tap sum|':>15}{'/65504':>9}"
    print(hdr2); print("-" * len(hdr2))
    for name, mode, shp, r in sorted(dw_rows, key=lambda x: -x[3]["maxsum"]):
        print(f"{name:<42}{mode:<6}{str(tuple(shp)):<12}{r['maxsum']:>15.1f}{r['maxsum']/FP16_MAX:>8.3f}x")

    print("\n### NOT COVERED by this e4m3 dump (need fp16-intermediate reconstruction):")
    for s in skipped:
        print("   -", s)

    print("\nVERDICT:",
          f"WITH the per-tensor pre-tree preshift fix, {total_ovf} overflow nodes remain "
          + ("(STILL OVERFLOWS — investigate)." if total_ovf > 0
             else f"-> NO overflow. Every layer now <= {worst:.0f} ({worst/FP16_MAX:.2f}x of 65504). "
                  f"INT4 untouched (per-group, preshift=0); only FP8 per-tensor layers pre-shifted."))

if __name__ == "__main__":
    main()
