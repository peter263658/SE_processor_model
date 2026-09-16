#!/usr/bin/env python3
"""Build SRAM init images for the strict-18 deploy core.

Implements docs/SRAM_MEMORY_MAP_SPEC_2026-06-03.md:
  - WBUF (256 KiB, single-port UHD): full model = weight code region + packed
    constants region. Constants are prefetched to a PPU-local cache at runtime
    (Option 1.5), so this image holds BOTH weights and constants.
  - KVBUF / UFBUF / SKIPBUF: zero-init images at their configured sizes.

Weight codes come from the RTL-preferred compact files in export_out_strict18
(in_proj -> [12,64,64] blocks, decoder grouped, FP8 outlier e4m3). FUSED APoT
scales come from hw_patterns_final/tb_pattern/weights (export_out_strict18 only
has base scales). Everything is byte-verified after writing.

Usage:
  python build_sram_images.py \
      --export-root  ../export_out_strict18 \
      --fused-root   ../hw_patterns_final/tb_pattern/weights \
      --blocks-root  ../export_out_strict18/blocks \
      --act-root     ../export_out_strict18/activation_blockfloat \
      --out          ../hw_patterns_final/sram_images \
      --scale-pack   12bit            # or 16bit (RTL-simpler, padded)
      --prefetch     subop            # or stage
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---- frozen SRAM config (spec §1) -----------------------------------------
SRAM = {
    "WBUF":  dict(banks=16, width=64, depth=2048),  # 256 KiB
    "KVBUF": dict(banks=4,  width=64, depth=1024),  # 32 KiB
    "UFBUF": dict(banks=8,  width=64, depth=128),   # 8 KiB
    "SKIPBUF": dict(banks=4, width=64, depth=128),  # 4 KiB
}
def cap_bytes(cfg): return cfg["banks"] * cfg["width"] // 8 * cfg["depth"]

WBUF_CAP = cap_bytes(SRAM["WBUF"])  # 262144
WORD_B = SRAM["WBUF"]["width"] // 8         # 8
ROW_B = SRAM["WBUF"]["banks"] * WORD_B      # 128 B per depth row
TENSOR_ALIGN = 64                            # spec §3

# ---- execution order (spec §4) --------------------------------------------
ENC_STAGES = list(range(6))
DEC_STAGES = list(range(6))


def _f(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 0


def read_bytes(path: str) -> bytes:
    with open(path, "rb") as fh:
        return fh.read()


def read_i16(path: str) -> np.ndarray:
    return np.frombuffer(read_bytes(path), dtype=np.int16) if _f(path) else np.array([], np.int16)


def read_i8(path: str) -> np.ndarray:
    return np.frombuffer(read_bytes(path), dtype=np.int8) if _f(path) else np.array([], np.int8)


def read_u8(path: str) -> np.ndarray:
    return np.frombuffer(read_bytes(path), dtype=np.uint8) if _f(path) else np.array([], np.uint8)


# ---- 12-bit / 16-bit APoT packing -----------------------------------------
def apot_code(a_exp: int, b_exp: int, has_second: int, sign_lo: int,
              bias: int = 24) -> int:
    """Pack one 12-bit code (spec §6).

    [11:7] a_exp+bias (5b), [6:2] b_exp+bias (5b), [1] has_second, [0] sign_lo.
    sign_lo meaning is region-dependent (residual sign for APoT 6a; overall slope
    sign for PReLU 6b). Caller passes the already-resolved bit (1 = negative).
    """
    a = a_exp + bias
    b = b_exp + bias
    if not (0 <= a < 32 and 0 <= b < 32):
        raise ValueError(f"APoT exp out of 5-bit biased range: a={a_exp} b={b_exp} (bias {bias})")
    return (a << 7) | (b << 2) | ((has_second & 1) << 1) | (sign_lo & 1)


def apot_decode(code: int, bias: int = 24) -> Tuple[int, int, int, int]:
    """Inverse of apot_code -> (a_exp, b_exp, has_second, sign_lo). For verifier."""
    a = (code >> 7) & 0x1F
    b = (code >> 2) & 0x1F
    has = (code >> 1) & 1
    s = code & 1
    return a - bias, b - bias, has, s


def unpack_codes(blob: bytes, n: int, pack: str) -> List[int]:
    """Inverse of pack_codes -> list of n 12-bit codes. For verifier."""
    if pack == "16bit":
        return np.frombuffer(blob, dtype=np.uint16)[:n].tolist()
    out: List[int] = []
    bits = 0
    nbits = 0
    it = iter(blob)
    while len(out) < n:
        while nbits < 12:
            bits |= next(it) << nbits
            nbits += 8
        out.append(bits & 0xFFF)
        bits >>= 12
        nbits -= 12
    return out


def pack_codes(codes: List[int], pack: str) -> bytes:
    """Pack a list of 12-bit codes into a byte stream.
    12bit: dense bit-packed (2 codes per 3 bytes). 16bit: zero-padded u16 LE."""
    if pack == "16bit":
        return np.asarray(codes, dtype=np.uint16).tobytes()
    # 12-bit dense
    bits = 0
    nbits = 0
    out = bytearray()
    for c in codes:
        bits |= (c & 0xFFF) << nbits
        nbits += 12
        while nbits >= 8:
            out.append(bits & 0xFF)
            bits >>= 8
            nbits -= 8
    if nbits:
        out.append(bits & 0xFF)
    return bytes(out)


# ---- weight tensor discovery (spec §4) ------------------------------------
def weight_tensor_path(export_root: str, dotted: str) -> Tuple[str, str]:
    """Return (path, kind) of the RTL-preferred weight-code file for a tensor dir."""
    d = os.path.join(export_root, "weights", dotted)
    blocks = os.path.join(d, "w_int4.blocks.packed.bin")
    packed = os.path.join(d, "w_int4.packed.bin")
    fp8 = os.path.join(d, "w_fp8.e4m3.bin")
    if _f(blocks):
        return blocks, "int4_blocks"
    if _f(fp8):
        return fp8, "fp8"
    if _f(packed):
        return packed, "int4"
    raise FileNotFoundError(f"no weight-code file for {dotted} in {d}")


def ordered_weight_tensors(export_root: str) -> List[str]:
    """Dotted names in execution order (spec §4), only those present in export."""
    names: List[str] = []

    def add(dotted: str):
        d = os.path.join(export_root, "weights", dotted)
        if os.path.isdir(d):
            names.append(dotted)

    for s in ENC_STAGES:
        for part in ("depthwise_r", "depthwise_i", "pointwise_r", "pointwise_i"):
            add(f"encoder.model.{s}.0.{part}")
    add("mattn.input_proj/fc_r")
    add("mattn.input_proj/fc_i")
    add("mattn.mattn.MultiheadAttention_re/in_proj_weight")
    add("mattn.mattn.MultiheadAttention_re/out_proj")
    add("mattn.output_proj/fc_r")
    add("mattn.output_proj/fc_i")
    for s in DEC_STAGES:
        for part in ("pointwise_r_x", "pointwise_i_x", "pointwise_r_skip", "pointwise_i_skip",
                     "depthwise_r", "depthwise_i"):
            add(f"decoder.model.{s}.0.{part}")
    return names


# ---- APoT scale collection (fused, spec §6) -------------------------------
def collect_weight_scale_codes(fused_root: str, export_root: str, dotted: str,
                               bias: int) -> List[int]:
    """Per weight tensor, return packed APoT codes from FUSED files when present,
    else base files. Handles per-branch kreal/kimag (concatenated) and FP8 descale."""
    fdir = os.path.join(fused_root, dotted)
    bdir = os.path.join(export_root, "weights", dotted)

    def load(tag: str) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        # tag = '' (base) or '_fused' or '_fused_act_kreal' etc.; FP8 uses fp8_descale_*
        for pre in ("scale_apot", "fp8_descale_apot"):
            a = os.path.join(fdir, f"{pre}_a_exp{tag}.int16.bin")
            b = os.path.join(fdir, f"{pre}_b_exp{tag}.int16.bin")
            if _f(a) and _f(b):
                has = os.path.join(bdir, f"{pre}_has_second.u8.bin")
                rs = os.path.join(bdir, f"{pre}_residual_sign.int8.bin")
                return read_i16(a), read_i16(b), read_u8(has), read_i8(rs)
        return None

    chunks: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    # priority: single fused -> per-branch (kreal then kimag) -> base
    got = load("_fused")
    if got:
        chunks.append(got)
    else:
        for br in ("_fused_act_kreal", "_fused_act_kimag"):
            g = load(br)
            if g:
                chunks.append(g)
        if not chunks:
            g = load("")  # base (e.g. MHA non-fused)
            if g:
                chunks.append(g)
    if not chunks:
        return []

    codes: List[int] = []
    for a_exp, b_exp, has, rs in chunks:
        n = len(a_exp)
        for i in range(n):
            h = int(has[i]) if i < len(has) else 1
            r = int(rs[i]) if i < len(rs) else 0
            sign_lo = 1 if r < 0 else 0          # residual sign (6a)
            codes.append(apot_code(int(a_exp[i]), int(b_exp[i]), h, sign_lo, bias))
    return codes


def collect_bn_codes(blocks_root: str, kind: str, stage: int, bias: int) -> Tuple[List[int], bytes]:
    """Return (BN APoT codes r+i, BN bias fp16 bytes r+i) for one stage."""
    st = os.path.join(blocks_root, kind, f"stage_{stage}")
    codes: List[int] = []
    bias_bytes = bytearray()
    for ri in ("r", "i"):
        a = read_i16(os.path.join(st, f"bn_scale_{ri}_apot_a_exp.int16.bin"))
        b = read_i16(os.path.join(st, f"bn_scale_{ri}_apot_b_exp.int16.bin"))
        has = read_u8(os.path.join(st, f"bn_scale_{ri}_apot_has_second.u8.bin"))
        rs = read_i8(os.path.join(st, f"bn_scale_{ri}_apot_residual_sign.int8.bin"))
        ov = read_i8(os.path.join(st, f"bn_scale_{ri}_apot_overall_sign.int8.bin"))
        for i in range(len(a)):
            # spec §6a: BN overall_sign is constant +1 in this ckpt; assert it so a
            # future gamma<0 retrain trips here instead of silently losing the sign.
            if len(ov) and int(ov[i]) != 1:
                raise ValueError(f"BN overall_sign != 1 at {kind} stage{stage} {ri}[{i}] "
                                 f"-> retrain flipped gamma; extend code format (spec §6a)")
            r = int(rs[i]) if i < len(rs) else 0
            sign_lo = 1 if r < 0 else 0          # residual sign (6a)
            codes.append(apot_code(int(a[i]), int(b[i]), int(has[i]) if i < len(has) else 0,
                                   sign_lo, bias))
        bb = read_bytes(os.path.join(st, f"bn_bias_{ri}_fused.fp32.bin"))
        bias_f32 = np.frombuffer(bb, dtype=np.float32)
        bias_bytes += bias_f32.astype(np.float16).tobytes()
    return codes, bytes(bias_bytes)


def collect_prelu_codes(blocks_root: str, kind: str, stage: int, bias: int) -> List[int]:
    """PReLU code per spec §6b: single pow2, bit0 = OVERALL slope sign (always
    meaningful), has_second=0, b_exp field zeroed (biased)."""
    st = os.path.join(blocks_root, kind, f"stage_{stage}")
    codes: List[int] = []
    for ri in ("r", "i"):
        e = read_i16(os.path.join(st, f"prelu_{ri}_pow2_exp.int16.bin"))
        s = read_i8(os.path.join(st, f"prelu_{ri}_pow2_sign.int8.bin"))
        for i in range(len(e)):
            overall_sign = 1 if (i < len(s) and int(s[i]) < 0) else 0   # 6b: bit0 = slope sign
            codes.append(apot_code(int(e[i]), -bias, 0, overall_sign, bias))
    return codes


# ---- builder ---------------------------------------------------------------
def align_up(n: int, a: int) -> int:
    return (n + a - 1) // a * a


def build_wbuf(args) -> Dict[str, Any]:
    bias = 24
    image = bytearray()
    mm: Dict[str, Any] = {"regions": {}, "tensors": [], "const_blocks": [], "config": SRAM}

    # ---------- R0/R1: weight code region ----------
    wstart = 0
    tensors = ordered_weight_tensors(args.export_root)
    for dotted in tensors:
        path, kind = weight_tensor_path(args.export_root, dotted)
        data = read_bytes(path)
        off = align_up(len(image), TENSOR_ALIGN)
        image.extend(b"\x00" * (off - len(image)))
        image.extend(data)
        mm["tensors"].append(dict(dotted=dotted, kind=kind, offset=off,
                                  length=len(data), src=os.path.relpath(path, args.export_root)))
    weight_end = len(image)
    mm["regions"]["weight_code"] = dict(start=wstart, end=weight_end, bytes=weight_end - wstart)

    # ---------- constants region: align to row boundary ----------
    const_base = align_up(len(image), ROW_B)
    image.extend(b"\x00" * (const_base - len(image)))
    mm["wbuf_const_base_row"] = const_base // ROW_B

    region_start = len(image)

    # Each const block is packed INDEPENDENTLY and placed at an explicit WBUF byte
    # offset, so it is directly DMA-able to the PPU local cache (spec §5) and the
    # verifier can decode it in isolation. const_prefetch carries the full contract.
    n_weight_scale = 0
    bn_codes_total = 0
    bn_bias_total = 0
    prelu_total = 0

    def place_block(meta: Dict[str, Any], payload: bytes):
        off = len(image)
        image.extend(payload)
        meta.update(wbuf_offset=off, wbuf_bytes=len(payload))
        mm["const_blocks"].append(meta)
        return meta

    # ----- R2: weight-side APoT scale, one block per weight tensor -----
    r2_lo = len(image)
    for dotted in tensors:
        codes = collect_weight_scale_codes(args.fused_root, args.export_root, dotted, bias)
        n_weight_scale += len(codes)
        place_block(dict(dotted=dotted, kind="weight_scale", region="R2",
                         decode="apot_6a", n_codes=len(codes), pack=args.scale_pack),
                    pack_codes(codes, args.scale_pack))
    mm["regions"]["weight_apot_scale"] = dict(start=r2_lo, end=len(image), bytes=len(image) - r2_lo)

    # ----- R3/R4/R5: BN scale, BN bias, PReLU — one block-set per stage -----
    r3_lo = len(image)
    for kind in ("encoder", "decoder"):
        for s in range(6):
            st = os.path.join(args.blocks_root, kind, f"stage_{s}")
            if not os.path.isdir(st):
                continue
            dotted = f"{kind}.model.{s}.0"
            bn_codes, bn_bias = collect_bn_codes(args.blocks_root, kind, s, bias)
            pr_codes = collect_prelu_codes(args.blocks_root, kind, s, bias)
            bn_codes_total += len(bn_codes)
            bn_bias_total += len(bn_bias)
            prelu_total += len(pr_codes)
            place_block(dict(dotted=dotted, kind="bn_scale", region="R3",
                             decode="apot_6a", n_codes=len(bn_codes), pack=args.scale_pack),
                        pack_codes(bn_codes, args.scale_pack))
            place_block(dict(dotted=dotted, kind="bn_bias", region="R4",
                             decode="fp16", n_elems=len(bn_bias) // 2),
                        bytes(bn_bias))
            place_block(dict(dotted=dotted, kind="prelu", region="R5",
                             decode="apot_6b", n_codes=len(pr_codes), pack=args.scale_pack),
                        pack_codes(pr_codes, args.scale_pack))
    mm["regions"]["bn_prelu"] = dict(start=r3_lo, end=len(image), bytes=len(image) - r3_lo)

    # ----- R5b: activation block-float k table (one block) -----
    act_k = read_i16(os.path.join(args.act_root, "scale_pow2_exp.int16.bin"))
    actk_payload = np.asarray([(int(v) + 32) & 0xFF for v in act_k], dtype=np.uint8).tobytes()
    place_block(dict(dotted="activation_k", kind="activation_k", region="R5b",
                     decode="u8_plus32", n_elems=len(act_k)), actk_payload)

    mm["regions"]["constants"] = dict(start=region_start, end=len(image),
                                      bytes=len(image) - region_start)

    # ---------- const_prefetch list (the RTL contract, spec §5) ----------
    # sub-op granularity: each const block is one prefetch unit (weight_scale and
    # BN/PReLU/bias prefetched separately). stage granularity: union per stage.
    prefetch: List[Dict[str, Any]] = []
    if args.prefetch == "subop":
        for cb in mm["const_blocks"]:
            prefetch.append(dict(unit=cb["dotted"] + ":" + cb["kind"],
                                 src_wbuf_offset=cb["wbuf_offset"], bytes=cb["wbuf_bytes"],
                                 decode=cb["decode"]))
    else:  # stage: group all blocks sharing a dotted key (contiguous by construction)
        from collections import OrderedDict
        groups: "OrderedDict[str, List[Dict[str,Any]]]" = OrderedDict()
        for cb in mm["const_blocks"]:
            groups.setdefault(cb["dotted"], []).append(cb)
        for dotted, blks in groups.items():
            lo = min(b["wbuf_offset"] for b in blks)
            hi = max(b["wbuf_offset"] + b["wbuf_bytes"] for b in blks)
            prefetch.append(dict(unit=dotted, src_wbuf_offset=lo, bytes=hi - lo,
                                 members=[b["kind"] for b in blks]))
    # assign PPU-cache dest offsets (double-buffered ping-pong handled by RTL; we
    # just report the max single-unit size the cache must hold)
    max_unit = max((p["bytes"] for p in prefetch), default=0)
    mm["const_prefetch"] = prefetch
    mm["prefetch"] = dict(
        granularity=args.prefetch,
        max_prefetch_unit_bytes=max_unit,
        recommended_cache_bytes=(2048 if args.prefetch == "subop" else 4096),
        double_buffer_cache_bytes=2 * (2048 if args.prefetch == "subop" else 4096),
    )
    mm["counts"] = dict(weight_scale_codes=n_weight_scale, bn_codes=bn_codes_total,
                        bn_bias_bytes=bn_bias_total, prelu_codes=prelu_total,
                        act_k=len(act_k), n_weight_tensors=len(tensors))

    # ---------- finalize to capacity ----------
    used = len(image)
    if used > WBUF_CAP:
        raise SystemExit(f"WBUF image {used} B EXCEEDS capacity {WBUF_CAP} B")
    image.extend(b"\x00" * (WBUF_CAP - used))
    mm["wbuf_used_bytes"] = used
    mm["wbuf_capacity_bytes"] = WBUF_CAP
    mm["wbuf_slack_bytes"] = WBUF_CAP - used
    return dict(image=bytes(image), memory_map=mm)


def write_config_vh(path: str, mm: Dict[str, Any]) -> None:
    lines = ["// Auto-generated by build_sram_images.py — do not edit",
             "// See docs/SRAM_MEMORY_MAP_SPEC_2026-06-03.md", ""]
    for name, cfg in SRAM.items():
        lines += [f"parameter {name}_BANKS = {cfg['banks']};",
                  f"parameter {name}_WIDTH = {cfg['width']};",
                  f"parameter {name}_DEPTH = {cfg['depth']};", ""]
    lines += [f"parameter WBUF_CONST_BASE_ROW = {mm['wbuf_const_base_row']};",
              f"parameter WBUF_USED_BYTES = {mm['wbuf_used_bytes']};",
              f"parameter PPU_CONST_CACHE_BYTES = {mm['prefetch']['recommended_cache_bytes']};",
              ""]
    with open(path, "w") as fh:
        fh.write("\n".join(lines))


def _src_weight_scale(args, dotted, bias):
    return collect_weight_scale_codes(args.fused_root, args.export_root, dotted, bias)


def write_pkg_svh(path: str, mm: Dict[str, Any]) -> None:
    """Compile memory_map.json into RTL-friendly localparams (for TB/RTL that
    can't parse JSON). One entry per weight tensor + per const block."""
    def ident(s: str) -> str:
        return s.replace(".", "_").replace("/", "_").upper()

    L = ["// Auto-generated by build_sram_images.py — do not edit.",
         "// Source of truth: memory_map.json. See SRAM_MEMORY_MAP_SPEC_2026-06-03.md.",
         "`ifndef MEMORY_MAP_PKG_SVH", "`define MEMORY_MAP_PKG_SVH", "",
         f"localparam int WBUF_CAP_BYTES   = {mm['wbuf_capacity_bytes']};",
         f"localparam int WBUF_USED_BYTES  = {mm['wbuf_used_bytes']};",
         f"localparam int WBUF_CONST_BASE  = {mm['regions']['constants']['start']};",
         f"localparam int PPU_CACHE_BYTES  = {mm['prefetch']['recommended_cache_bytes']};",
         f"localparam int N_WEIGHT_TENSORS = {len(mm['tensors'])};",
         f"localparam int N_CONST_BLOCKS   = {len(mm['const_blocks'])};", ""]
    L.append("// --- weight tensors: {offset, length} ---")
    for t in mm["tensors"]:
        n = ident(t["dotted"])
        L.append(f"localparam int W_{n}_OFF = {t['offset']}; localparam int W_{n}_LEN = {t['length']};")
    L.append("")
    L.append("// --- const blocks: {offset, bytes, decode} ---")
    DEC = {"apot_6a": 0, "apot_6b": 1, "fp16": 2, "u8_plus32": 3}
    for cb in mm["const_blocks"]:
        n = ident(cb["dotted"] + "_" + cb["kind"])
        L.append(f"localparam int C_{n}_OFF = {cb['wbuf_offset']}; "
                 f"localparam int C_{n}_BYTES = {cb['wbuf_bytes']}; "
                 f"localparam int C_{n}_DEC = {DEC.get(cb['decode'], 0)};  // {cb['decode']}")
    L += ["", "`endif"]
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")


def verify(out_dir: str, args, mm: Dict[str, Any]) -> List[str]:
    """Decode the image back and compare to source — weights (byte) AND constants
    (field-level), per spec §8. Catches APoT bit-order / sign / fp16 / k errors."""
    bias = 24
    errs: List[str] = []
    image = read_bytes(os.path.join(out_dir, "wbuf_unified_init.bin"))
    if len(image) != WBUF_CAP:
        errs.append(f"image size {len(image)} != {WBUF_CAP}")

    # ---- weights: byte-exact ----
    for t in mm["tensors"]:
        path, _ = weight_tensor_path(args.export_root, t["dotted"])
        if image[t["offset"]:t["offset"] + t["length"]] != read_bytes(path):
            errs.append(f"weight mismatch {t['dotted']}")
        if any(bad in t["src"] for bad in ("debug", "effective", ".fp32", "raw_s8")):
            errs.append(f"forbidden src {t['src']}")
        if "in_proj_weight" in t["dotted"] and t["length"] != 24576:
            errs.append(f"in_proj length {t['length']} != 24576 (dense leak?)")

    # ---- constants: decode each block, compare to source fields ----
    blocks_root = args.blocks_root
    for cb in mm["const_blocks"]:
        blob = image[cb["wbuf_offset"]:cb["wbuf_offset"] + cb["wbuf_bytes"]]
        dec = cb["decode"]
        if dec == "apot_6a":
            if cb["kind"] == "weight_scale":
                want = _src_weight_scale(args, cb["dotted"], bias)
            else:  # bn_scale
                kind, st = _stage_kv(cb["dotted"])
                want = collect_bn_codes(blocks_root, kind, st, bias)[0]
            got = unpack_codes(blob, cb["n_codes"], cb["pack"])
            if got != want:
                errs.append(f"const APoT mismatch {cb['dotted']}:{cb['kind']}")
        elif dec == "apot_6b":
            kind, st = _stage_kv(cb["dotted"])
            want = collect_prelu_codes(blocks_root, kind, st, bias)
            got = unpack_codes(blob, cb["n_codes"], cb["pack"])
            if got != want:
                errs.append(f"PReLU code mismatch {cb['dotted']} (sign loss?)")
            # explicit sign round-trip check
            for c in got:
                _, _, has, s = apot_decode(c, bias)
                if has != 0:
                    errs.append(f"PReLU has_second!=0 {cb['dotted']}")
        elif dec == "fp16":
            kind, st = _stage_kv(cb["dotted"])
            _, want_bias = collect_bn_codes(blocks_root, kind, st, bias)
            if blob != want_bias:
                errs.append(f"BN bias fp16 mismatch {cb['dotted']}")
        elif dec == "u8_plus32":
            want = read_i16(os.path.join(args.act_root, "scale_pow2_exp.int16.bin"))
            got = np.frombuffer(blob, dtype=np.uint8).astype(np.int16) - 32
            if got.tolist() != want.tolist():
                errs.append("activation-k mismatch")

    # ---- counts + prefetch ----
    if mm["counts"]["weight_scale_codes"] != 3700:
        errs.append(f"weight_scale_codes {mm['counts']['weight_scale_codes']} != 3700")
    cap = mm["prefetch"]["recommended_cache_bytes"]
    peak = mm["prefetch"]["max_prefetch_unit_bytes"]
    if peak > cap:
        errs.append(f"prefetch peak {peak} > cache cap {cap}")
    # every prefetch unit must point inside the constants region
    cregion = mm["regions"]["constants"]
    for p in mm["const_prefetch"]:
        if not (cregion["start"] <= p["src_wbuf_offset"] < cregion["end"]):
            errs.append(f"prefetch unit {p['unit']} offset outside constants region")
    return errs


def _stage_kv(dotted: str) -> Tuple[str, int]:
    # "encoder.model.3.0" -> ("encoder", 3)
    kind = dotted.split(".")[0]
    stage = int(dotted.split(".model.")[1].split(".")[0])
    return kind, stage


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, ".."))
    ap.add_argument("--export-root", default=os.path.join(root, "export_out_strict18"))
    ap.add_argument("--fused-root", default=os.path.join(root, "hw_patterns_final/tb_pattern/weights"))
    ap.add_argument("--blocks-root", default=os.path.join(root, "export_out_strict18/blocks"))
    ap.add_argument("--act-root", default=os.path.join(root, "export_out_strict18/activation_blockfloat"))
    ap.add_argument("--out", default=os.path.join(root, "hw_patterns_final/sram_images"))
    ap.add_argument("--scale-pack", choices=["12bit", "16bit"], default="12bit")
    ap.add_argument("--prefetch", choices=["subop", "stage"], default="subop")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    built = build_wbuf(args)
    mm = built["memory_map"]

    with open(os.path.join(args.out, "wbuf_unified_init.bin"), "wb") as fh:
        fh.write(built["image"])
    for name, cfg in (("kvbuf", "KVBUF"), ("ufbuf", "UFBUF"), ("skipbuf", "SKIPBUF")):
        with open(os.path.join(args.out, f"{name}_init_zero.bin"), "wb") as fh:
            fh.write(b"\x00" * cap_bytes(SRAM[cfg]))

    with open(os.path.join(args.out, "memory_map.json"), "w") as fh:
        json.dump(mm, fh, indent=2)
    write_config_vh(os.path.join(args.out, "sram_config.vh"), mm)
    write_pkg_svh(os.path.join(args.out, "memory_map_pkg.svh"), mm)

    errs = verify(args.out, args, mm)
    with open(os.path.join(args.out, "README.md"), "w") as fh:
        fh.write(f"""# SRAM init images (strict-18 deploy)

Generated by `software_to_hardware_18/build_sram_images.py`.
Spec: `docs/SRAM_MEMORY_MAP_SPEC_2026-06-03.md`.

## Regenerate (needs numpy from the bcctn env; base Python fails)
```
conda activate bcctn
python software_to_hardware_18/build_sram_images.py \\
    --scale-pack {args.scale_pack} --prefetch {args.prefetch}
```

scale_pack={args.scale_pack}  prefetch={args.prefetch}

| file | size | purpose |
|---|---|---|
| wbuf_unified_init.bin | {WBUF_CAP} B (256 KiB) | full model: weight codes + packed constants |
| kvbuf_init_zero.bin | {cap_bytes(SRAM['KVBUF'])} B | K/V ring (zero init) |
| ufbuf_init_zero.bin | {cap_bytes(SRAM['UFBUF'])} B | activation scratch (zero init) |
| skipbuf_init_zero.bin | {cap_bytes(SRAM['SKIPBUF'])} B | skip subset (zero init) |
| memory_map.json | — | per-tensor/const offsets + const_prefetch list |
| memory_map_pkg.svh | — | RTL localparams compiled from the JSON |
| sram_config.vh | — | SRAM params + WBUF_CONST_BASE_ROW + PPU_CONST_CACHE_BYTES |

WBUF used {mm['wbuf_used_bytes']} B, slack {mm['wbuf_slack_bytes']} B ({100*mm['wbuf_used_bytes']/WBUF_CAP:.1f}% full).
const_base byte={mm['regions']['constants']['start']}, PPU cache {mm['prefetch']['recommended_cache_bytes']} B
(max prefetch unit {mm['prefetch']['max_prefetch_unit_bytes']} B).
Verify (weights byte-exact + constants field-decode): {"PASS" if not errs else "FAIL: " + "; ".join(errs)}
""")

    print(f"[build] WBUF used {mm['wbuf_used_bytes']} / {WBUF_CAP} B "
          f"(slack {mm['wbuf_slack_bytes']} B, {100*mm['wbuf_used_bytes']/WBUF_CAP:.1f}% full)")
    print(f"[build] regions: " + ", ".join(f"{k}={v['bytes']}B" for k, v in mm["regions"].items()))
    print(f"[build] counts: {mm['counts']}")
    print(f"[build] prefetch: {mm['prefetch']}")
    print(f"[verify] {'PASS' if not errs else 'FAIL'}")
    for e in errs:
        print("   -", e)
    raise SystemExit(1 if errs else 0)


if __name__ == "__main__":
    main()
