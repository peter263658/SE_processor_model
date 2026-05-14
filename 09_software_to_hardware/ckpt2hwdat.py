"""
ckpt2hwdat.py
=============
Convert a trained Mixed-Precision W4A8 QAT checkpoint into
hardware-friendly binary files (weights, scales, fused BN/PReLU/√D constants)
arranged per-layer so they can be loaded independently by the RTL testbench.

Output layout (under --outdir, default `./export_out/`):

    manifest.json                        Global pointer / summary.
    layer_config.json                    Copy of what dump_hw_config.py emits.
    weights/<dotted>/w_int4.packed.bin   Nibble-packed INT4 (2 elements / byte).
    weights/<dotted>/w_int4.raw_s8.bin   Same values as int8 in [-7,7], for test.
    weights/<dotted>/w_fp8.e4m3.bin      FP8 raw bytes (outlier PW only).
    weights/<dotted>/scale.fp16.bin      FP16 scale tensor.
    weights/<dotted>/bias.fp16.bin       (Linear/MHA only) raw bias after √D fuse.
    weights/<dotted>/meta.json           Shape & dtype & per-field description.
    blocks/<kind>/<stage>/bn_scale_r.fp16.bin    BN scale (real output channel).
    blocks/<kind>/<stage>/bn_bias_r.fp16.bin     Fused conv-bias + BN-bias (real).
    blocks/<kind>/<stage>/bn_scale_i.fp16.bin
    blocks/<kind>/<stage>/bn_bias_i.fp16.bin
    blocks/<kind>/<stage>/prelu_alpha_r.fp32.bin (scalar)
    blocks/<kind>/<stage>/prelu_alpha_i.fp32.bin
    blocks/<kind>/<stage>/meta.json
    misc/stft_window.fp16.bin            Hann window (length = win_len).
    misc/istft_window.fp16.bin           Same as stft_window for symmetric iSTFT.
    misc/meta.json                       n_fft, hop, embed_dim, num_heads, etc.

Fusions performed (see FUSIONS.md for full derivation):
  * BN + complex conv bias → two post-MAC per-channel FMAs
    (bn_scale_r/i, bn_bias_r/i). NOT merged into the INT4 conv weight
    because the two BNs (real/imag) are orthogonal on the complex MAC.
  * 1/sqrt(head_dim) → pre-divided into Q rows of MHA in_proj scale
    and in_proj_bias. INT4 integers for Q are unchanged.
  * PReLU α stays as a per-block scalar (num_parameters=1). Blocks
    without PReLU (last decoder block) have alpha=1.0 and will be
    marked has_prelu=false in block meta.

Usage:
    python ckpt2hwdat.py --outdir ./export_out
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
REPO_09 = "/raid/R12K41024/09_weight_tied_mha"
FINETUNE_STRICT_RUN = os.path.join(
    REPO_09, "outputs/finetune_strict/2026-05-10_02-40-16"
)
DEFAULT_CKPT = os.path.join(
    FINETUNE_STRICT_RUN,
    "logs/lightning_logs/version_0/checkpoints/epoch=45-step=44528.ckpt",
)
DEFAULT_CONFIG = os.path.join(FINETUNE_STRICT_RUN, ".hydra/config.yaml")

if REPO_09 not in sys.path:
    sys.path.insert(0, REPO_09)

from omegaconf import OmegaConf  # noqa: E402
from DCNN.models.binaural_attention_model import BinauralAttentionDCNN  # noqa: E402
from DCNN.utils.mixed_precision_qat import enable_mixed_precision_qat  # noqa: E402

import export_helpers as eh  # noqa: E402
import dump_hw_config as dhc  # noqa: E402


# ──────────────────────────────────────────────────────────────
# Write utilities
# ──────────────────────────────────────────────────────────────

def _ensure(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _w_bin(path: str, arr: np.ndarray) -> None:
    arr.tofile(path)
    _w_sidecars(path, arr)


def _sidecar_base(path: str) -> str:
    return path[:-4] if path.endswith(".bin") else path


def _w_sidecars(path: str, arr: np.ndarray) -> None:
    """Write plain readmemh-friendly hex and debug decimal sidecars.

    Hex is one storage word per line: 8-bit for byte arrays, 16-bit for fp16,
    and 32-bit for fp32. Decimal is one numeric value per line.
    """
    a = np.asarray(arr)
    flat = a.reshape(-1)
    base = _sidecar_base(path)

    if a.dtype == np.float16:
        hex_words = flat.view(np.uint16)
        fmt = "{:04X}"
    elif a.dtype == np.float32:
        hex_words = flat.view(np.uint32)
        fmt = "{:08X}"
    elif a.dtype == np.int8:
        hex_words = flat.view(np.uint8)
        fmt = "{:02X}"
    elif a.dtype == np.uint8:
        hex_words = flat
        fmt = "{:02X}"
    elif a.dtype == np.int16:
        hex_words = flat.view(np.uint16)
        fmt = "{:04X}"
    elif a.dtype == np.uint16:
        hex_words = flat
        fmt = "{:04X}"
    elif a.dtype == np.int32:
        hex_words = flat.view(np.uint32)
        fmt = "{:08X}"
    elif a.dtype == np.uint32:
        hex_words = flat
        fmt = "{:08X}"
    else:
        # Fall back to byte-wise storage for uncommon dtypes.
        hex_words = a.view(np.uint8).reshape(-1)
        fmt = "{:02X}"

    with open(base + ".hex", "w") as f:
        for v in hex_words:
            f.write(fmt.format(int(v)) + "\n")

    with open(base + ".dec", "w") as f:
        if np.issubdtype(a.dtype, np.floating):
            for v in flat.astype(np.float64):
                f.write(f"{float(v):.10e}\n")
        else:
            for v in flat:
                f.write(f"{int(v)}\n")


def _w_dec_values(path: str, t: torch.Tensor) -> None:
    flat = t.detach().cpu().reshape(-1)
    with open(path, "w") as f:
        for v in flat:
            f.write(f"{float(v):.10e}\n")


def _w_fp16(path: str, t: torch.Tensor) -> None:
    _w_bin(path, t.detach().to(torch.float16).cpu().numpy())


def _w_fp32(path: str, t: torch.Tensor) -> None:
    _w_bin(path, t.detach().to(torch.float32).cpu().numpy())


def _w_json(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# ──────────────────────────────────────────────────────────────
# Per-conv exporter
# ──────────────────────────────────────────────────────────────

def export_conv_int4(module: nn.Module, dotted: str, outdir: str,
                     scale_gran: str, group_size: int = 256,
                     num_heads: Optional[int] = None) -> Dict[str, Any]:
    """INT4 export for a Conv2d / ConvTranspose2d / real-valued Linear."""
    w = eh.raw_weight(module)          # fp32
    if scale_gran == "per_channel":
        q, scale = eh.int4_per_channel(w)
    elif scale_gran == "per_group":
        q, scale, used_per_group = eh.int4_per_group_last_dim(w, group_size)
    elif scale_gran == "per_head":
        assert num_heads is not None
        q, scale = eh.int4_per_head_linear(w, num_heads)
    else:
        raise ValueError(f"unknown scale_gran={scale_gran}")

    d = os.path.join(outdir, "weights", dotted)
    _ensure(d)
    _w_bin(os.path.join(d, "w_int4.packed.bin"), eh.pack_int4_nibbles(q))
    _w_bin(os.path.join(d, "w_int4.raw_s8.bin"), q.cpu().numpy().astype(np.int8))
    _w_fp16(os.path.join(d, "scale.fp16.bin"), scale.squeeze().float())

    meta = {
        "dotted": dotted,
        "weight_format": "int4",
        "scale_granularity": scale_gran,
        "weight_shape": list(w.shape),
        "weight_dtype": "int4_packed",
        "scale_shape": list(scale.squeeze().shape),
        "scale_dtype": "fp16",
        "num_elements": int(q.numel()),
        "packed_bytes": int((q.numel() + 1) // 2),
    }
    if scale_gran == "per_group":
        meta["group_size"] = group_size
        meta["effective_scale_granularity"] = "per_group" if used_per_group else "per_channel"
    if scale_gran == "per_head":
        meta["num_heads"] = num_heads

    if getattr(module, "bias", None) is not None:
        _w_fp16(os.path.join(d, "bias.fp16.bin"), module.bias)
        meta["bias_length"] = module.bias.numel()
        meta["bias_dtype"] = "fp16"

    _w_json(os.path.join(d, "meta.json"), meta)
    return meta


def export_conv_fp8(module: nn.Module, dotted: str, outdir: str) -> Dict[str, Any]:
    """FP8 E4M3 export for outlier decoder pointwise conv."""
    w = eh.raw_weight(module)
    fp8_bytes = eh.fp8_e4m3_bytes(w)
    d = os.path.join(outdir, "weights", dotted)
    _ensure(d)
    _w_bin(os.path.join(d, "w_fp8.e4m3.bin"), fp8_bytes)
    _w_dec_values(os.path.join(d, "w_fp8.e4m3.value.dec"),
                  w.to(torch.float8_e4m3fn).to(torch.float32))

    meta = {
        "dotted": dotted,
        "weight_format": "fp8_e4m3",
        "scale_granularity": "per_tensor",
        "weight_shape": list(w.shape),
        "weight_dtype": "fp8_e4m3",
        "num_elements": int(w.numel()),
    }
    if getattr(module, "bias", None) is not None:
        _w_fp16(os.path.join(d, "bias.fp16.bin"), module.bias)
        meta["bias_length"] = module.bias.numel()
        meta["bias_dtype"] = "fp16"

    _w_json(os.path.join(d, "meta.json"), meta)
    return meta


# ──────────────────────────────────────────────────────────────
# Complex-Linear exporter (weight is complex64, has .real and .imag halves)
# ──────────────────────────────────────────────────────────────

def export_complex_linear(module: nn.Module, dotted: str, outdir: str,
                          num_heads: int) -> Dict[str, Any]:
    """Split a ComplexLinear (complex64 weight) into two real Linear exports
    under '{dotted}/fc_r' and '{dotted}/fc_i'.
    """
    w = eh.raw_weight(module)         # complex64 [out, in]
    w_r = w.real.contiguous()
    w_i = w.imag.contiguous()

    q_r, s_r = eh.int4_per_head_linear(w_r, num_heads)
    q_i, s_i = eh.int4_per_head_linear(w_i, num_heads)

    d_r = os.path.join(outdir, "weights", f"{dotted}/fc_r")
    d_i = os.path.join(outdir, "weights", f"{dotted}/fc_i")
    _ensure(d_r); _ensure(d_i)

    _w_bin(os.path.join(d_r, "w_int4.packed.bin"), eh.pack_int4_nibbles(q_r))
    _w_bin(os.path.join(d_r, "w_int4.raw_s8.bin"), q_r.cpu().numpy().astype(np.int8))
    _w_fp16(os.path.join(d_r, "scale.fp16.bin"), s_r.squeeze().float())

    _w_bin(os.path.join(d_i, "w_int4.packed.bin"), eh.pack_int4_nibbles(q_i))
    _w_bin(os.path.join(d_i, "w_int4.raw_s8.bin"), q_i.cpu().numpy().astype(np.int8))
    _w_fp16(os.path.join(d_i, "scale.fp16.bin"), s_i.squeeze().float())

    if module.bias is not None:
        b = module.bias.detach()
        _w_fp16(os.path.join(d_r, "bias.fp16.bin"), b.real)
        _w_fp16(os.path.join(d_i, "bias.fp16.bin"), b.imag)

    meta = {
        "dotted": dotted,
        "weight_dtype_logical": "complex64",
        "weight_shape": list(w.shape),
        "weight_format": "int4",
        "scale_granularity": "per_head",
        "num_heads": num_heads,
        "halves": ["fc_r", "fc_i"],
        "fc_r_scale_shape": list(s_r.squeeze().shape),
        "fc_i_scale_shape": list(s_i.squeeze().shape),
    }
    _w_json(os.path.join(outdir, "weights", dotted, "meta.json"), meta)
    return meta


# ──────────────────────────────────────────────────────────────
# MHA exporter (in_proj + out_proj, with 1/sqrt(D) fuse)
# ──────────────────────────────────────────────────────────────

def export_mha(mha: nn.MultiheadAttention, dotted: str, outdir: str,
               num_heads: int, embed_dim: int) -> Dict[str, Any]:
    # in_proj_weight
    w = eh.raw_in_proj_weight(mha)
    q_w, scale = eh.int4_per_head_mha(w, num_heads)
    bias = mha.in_proj_bias.detach() if mha.in_proj_bias is not None else None

    q_w_fused, scale_fused, bias_fused = eh.fuse_qk_scale_into_in_proj(
        q_w, scale, bias, embed_dim, num_heads
    )

    ip_dir = os.path.join(outdir, "weights", f"{dotted}/in_proj_weight")
    _ensure(ip_dir)
    _w_bin(os.path.join(ip_dir, "w_int4.packed.bin"), eh.pack_int4_nibbles(q_w_fused))
    _w_bin(os.path.join(ip_dir, "w_int4.raw_s8.bin"), q_w_fused.cpu().numpy().astype(np.int8))
    _w_fp16(os.path.join(ip_dir, "scale.fp16.bin"), scale_fused.squeeze().float())
    if bias_fused is not None:
        _w_fp16(os.path.join(ip_dir, "bias.fp16.bin"), bias_fused)
    meta_ip = {
        "dotted": f"{dotted}/in_proj_weight",
        "weight_shape": list(w.shape),
        "weight_format": "int4",
        "scale_granularity": "per_head_qkv_fused",
        "num_heads": num_heads,
        "scale_groups": 3 * num_heads,
        "q_rows_prescaled_by_inv_sqrt_D": True,
        "inv_sqrt_D": 1.0 / math.sqrt(embed_dim // num_heads),
        "bias_length": bias_fused.numel() if bias_fused is not None else 0,
    }
    _w_json(os.path.join(ip_dir, "meta.json"), meta_ip)

    # out_proj (nn.Linear with weight (E, E), bias (E,))
    out_proj = mha.out_proj
    export_conv_int4(out_proj, f"{dotted}/out_proj", outdir,
                     scale_gran="per_head", num_heads=num_heads)

    return {
        "dotted": dotted,
        "in_proj": meta_ip,
        "out_proj_dotted": f"{dotted}/out_proj",
    }


# ──────────────────────────────────────────────────────────────
# Block-level export (BN + PReLU)
# ──────────────────────────────────────────────────────────────

def export_block(block: eh.BlockInfo, outdir: str) -> Dict[str, Any]:
    d = os.path.join(outdir, "blocks", block.kind, f"stage_{block.stage_idx}")
    _ensure(d)

    fuse = eh.fuse_bn_into_post_mac(
        block.bn_r, block.bn_i,
        block.pointwise_r.bias if block.kind == "encoder" and block.pointwise_r is not None and block.pointwise_r.bias is not None else
            (block.depthwise_r.bias if block.kind == "decoder" and block.depthwise_r is not None and block.depthwise_r.bias is not None else None),
        block.pointwise_i.bias if block.kind == "encoder" and block.pointwise_i is not None and block.pointwise_i.bias is not None else
            (block.depthwise_i.bias if block.kind == "decoder" and block.depthwise_i is not None and block.depthwise_i.bias is not None else None),
        block.num_out_channels,
    )

    _w_fp16(os.path.join(d, "bn_scale_r.fp16.bin"), fuse.bn_scale_r)
    _w_fp16(os.path.join(d, "bn_bias_r.fp16.bin"),  fuse.bn_bias_r)
    _w_fp16(os.path.join(d, "bn_scale_i.fp16.bin"), fuse.bn_scale_i)
    _w_fp16(os.path.join(d, "bn_bias_i.fp16.bin"),  fuse.bn_bias_i)

    if block.prelu_r is not None:
        _w_fp32(os.path.join(d, "prelu_alpha_r.fp32.bin"), block.prelu_r.weight)
        _w_fp32(os.path.join(d, "prelu_alpha_i.fp32.bin"), block.prelu_i.weight)

    meta = {
        "kind": block.kind,
        "stage_idx": block.stage_idx,
        "dotted": block.dotted,
        "num_out_channels": block.num_out_channels,
        "has_bn": block.bn_r is not None,
        "has_prelu": block.prelu_r is not None,
        "is_outlier_fp8": block.is_outlier_fp8,
        "bn_attaches_to": "pointwise" if block.kind == "encoder" else "depthwise",
        "post_mac_order": [
            "bn_scale_{r,i} * mac_out_{r,i}",
            "+ bn_bias_{r,i}",
            "PReLU(result, alpha_{r,i})" if block.prelu_r is not None else "(no activation)",
        ],
        "bn_eps": (float(block.bn_r.eps) if block.bn_r is not None else None),
        "prelu_num_parameters": (block.prelu_r.weight.numel() if block.prelu_r is not None else 0),
    }
    _w_json(os.path.join(d, "meta.json"), meta)
    return meta


# ──────────────────────────────────────────────────────────────
# STFT/iSTFT window export
# ──────────────────────────────────────────────────────────────

def export_stft(model: nn.Module, outdir: str, model_cfg: Dict[str, Any]) -> Dict[str, Any]:
    d = os.path.join(outdir, "misc")
    _ensure(d)
    win_len = model_cfg.get("win_len", 128)
    fft_len = model_cfg.get("fft_len", 128)
    hop = model_cfg.get("win_inc", 80)
    win_type = model_cfg.get("win_type", "hamming")

    if str(win_type).lower() == "hamming":
        win = torch.hamming_window(win_len, periodic=False, dtype=torch.float32)
    elif str(win_type).lower() == "hann":
        win = torch.hann_window(win_len, periodic=False, dtype=torch.float32)
    else:
        raise ValueError(f"unsupported STFT win_type={win_type}")
    _w_fp16(os.path.join(d, "stft_window.fp16.bin"), win)
    _w_fp16(os.path.join(d, "istft_window.fp16.bin"), win)

    masking_mode = model_cfg.get("masking_mode", getattr(model, "masking_mode", "C"))
    meta = {
        "win_len": int(win_len),
        "win_inc": int(hop),
        "fft_len": int(fft_len),
        "win_type": win_type,
        "window_periodic": False,
        "center": False,
        "normalized": True,
        "num_freq_bins_before_dc_drop": int(fft_len // 2 + 1),
        "num_freq_bins_after_dc_drop": int(fft_len // 2),
        "dc_bin_removed_before_model": True,
        "masking_mode": masking_mode,
        "notes": "STFT matches torch.stft with hamming_window(periodic=False), center=False, normalized=True.",
    }
    _w_json(os.path.join(d, "meta.json"), meta)
    return meta


# ──────────────────────────────────────────────────────────────
# Entry
# ──────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT,
                    help="Lightning checkpoint to export. Defaults to 09 tied-MHA finetune strict best.")
    ap.add_argument("--config", default=DEFAULT_CONFIG,
                    help="Hydra config. Defaults to the finetune strict run config.")
    ap.add_argument("--outdir", default="./export_out")
    ap.add_argument("--no-qat", action="store_true",
                    help="Skip mixed-precision parametrization registration "
                    "(use when exporting a pre-QAT checkpoint).")
    args = ap.parse_args()

    if os.path.isdir(args.outdir):
        shutil.rmtree(args.outdir)
    _ensure(args.outdir)

    cfg = OmegaConf.load(args.config)
    model_cfg = OmegaConf.to_container(cfg.model)
    training_cfg = OmegaConf.to_container(cfg.training)

    num_heads = model_cfg.get("num_heads", 16)
    embed_dim = model_cfg.get("embed_dim", 256)
    int4_group_size = training_cfg.get("int4_group_size", 256)

    # Build skeleton and register parametrizations so state_dict keys match.
    model = BinauralAttentionDCNN(**model_cfg).eval()
    if not args.no_qat:
        mp_cfg = {
            "encoder_dw":          {"weight_bits": 4, "weight_scale": "per_channel"},
            "encoder_pw":          {"weight_bits": 4, "weight_scale": "per_group",   "group_size": int4_group_size},
            "decoder_dw":          {"weight_bits": 4, "weight_scale": "per_channel"},
            "decoder_pw_outlier":  {"weight_bits": 8, "weight_format": "fp8_e4m3"},
            "decoder_pw_normal":   {"weight_bits": 4, "weight_scale": "per_group",   "group_size": int4_group_size},
            "mha":                 {"weight_bits": 4, "weight_scale": "per_head",    "num_heads": num_heads},
        }
        enable_mixed_precision_qat(model, config=mp_cfg,
                                   activation_format=training_cfg.get("fp8_dtype", "e4m3"),
                                   top_k=training_cfg.get("top_k_attention", 4),
                                   fp16_accumulator=training_cfg.get("fp16_accumulator", False))

    sd = eh.load_state_dict_from_ckpt(args.ckpt)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[load_state_dict] missing={len(missing)}  unexpected={len(unexpected)}")
    if len(missing) > 0:
        print(f"  (first 5 missing) {missing[:5]}")
    if len(unexpected) > 0:
        print(f"  (first 5 unexpected) {unexpected[:5]}")

    # Per-layer config — reuse dump_hw_config emitter.
    layer_cfg = dhc.build_config(
        model, num_heads, embed_dim, int4_group_size,
        top_k_attention=training_cfg.get("top_k_attention", 4),
        model_cfg=model_cfg,
        fp16_accumulator=training_cfg.get("fp16_accumulator", False),
    )
    _w_json(os.path.join(args.outdir, "layer_config.json"), layer_cfg)

    # Walk blocks and export.
    blocks = eh.discover_blocks(model)
    block_metas = []
    layer_metas = []

    for b in blocks:
        block_metas.append(export_block(b, args.outdir))

        # Depthwise (always INT4 per-channel)
        if b.depthwise_r is not None:
            layer_metas.append(export_conv_int4(
                b.depthwise_r, f"{b.dotted}.depthwise_r", args.outdir,
                scale_gran="per_channel",
            ))
            layer_metas.append(export_conv_int4(
                b.depthwise_i, f"{b.dotted}.depthwise_i", args.outdir,
                scale_gran="per_channel",
            ))

        # Pointwise (INT4 per-group requested normally, FP8 for outlier)
        if b.pointwise_r is not None:
            if b.is_outlier_fp8:
                layer_metas.append(export_conv_fp8(
                    b.pointwise_r, f"{b.dotted}.pointwise_r", args.outdir))
                layer_metas.append(export_conv_fp8(
                    b.pointwise_i, f"{b.dotted}.pointwise_i", args.outdir))
            else:
                layer_metas.append(export_conv_int4(
                    b.pointwise_r, f"{b.dotted}.pointwise_r", args.outdir,
                    scale_gran="per_group", group_size=int4_group_size,
                ))
                layer_metas.append(export_conv_int4(
                    b.pointwise_i, f"{b.dotted}.pointwise_i", args.outdir,
                    scale_gran="per_group", group_size=int4_group_size,
                ))

    # Attention block
    matt = model.mattn
    # input_proj is declared as nn.Linear(complex64), weight is complex.
    if matt.input_proj.weight.is_complex():
        layer_metas.append(export_complex_linear(
            matt.input_proj, "mattn.input_proj", args.outdir, num_heads))
    else:
        layer_metas.append(export_conv_int4(
            matt.input_proj, "mattn.input_proj", args.outdir,
            scale_gran="per_head", num_heads=num_heads))

    if matt.output_proj.weight.is_complex():
        layer_metas.append(export_complex_linear(
            matt.output_proj, "mattn.output_proj", args.outdir, num_heads))
    else:
        layer_metas.append(export_conv_int4(
            matt.output_proj, "mattn.output_proj", args.outdir,
            scale_gran="per_head", num_heads=num_heads))

    cmha = matt.mattn
    layer_metas.append(export_mha(
        cmha.MultiheadAttention_re, "mattn.mattn.MultiheadAttention_re",
        args.outdir, num_heads, embed_dim))
    if hasattr(cmha, "MultiheadAttention_im"):
        layer_metas.append(export_mha(
            cmha.MultiheadAttention_im, "mattn.mattn.MultiheadAttention_im",
            args.outdir, num_heads, embed_dim))

    # STFT
    stft_meta = export_stft(model, args.outdir, model_cfg)

    # Manifest
    manifest = {
        "source_ckpt": os.path.abspath(args.ckpt),
        "config": os.path.abspath(args.config),
        "model": {
            "embed_dim": embed_dim,
            "num_heads": num_heads,
            "head_dim": embed_dim // num_heads,
            "int4_group_size": int4_group_size,
            "softmax_topk_k": training_cfg.get("top_k_attention", 4),
            "qat_scope": training_cfg.get("qat_scope", "full"),
            "tied_mha": bool(getattr(cmha, "tied", False)),
        },
        "fusions_applied": [
            "BN (real) + complex conv real-output bias → (bn_scale_r, bn_bias_r)",
            "BN (imag) + complex conv imag-output bias → (bn_scale_i, bn_bias_i)",
            "1/sqrt(head_dim) → Q scale[0:num_heads] + Q bias[0:embed_dim]",
            "nibble pack INT4 (2 values / byte, low nibble first)",
        ],
        "blocks": block_metas,
        "layers": layer_metas,
        "stft": stft_meta,
    }
    _w_json(os.path.join(args.outdir, "manifest.json"), manifest)

    print(f"[OK] exported to {args.outdir}")
    print(f"  {len(block_metas)} blocks  |  {len(layer_metas)} weight dirs")
    print(f"  FP8 outlier blocks: {sum(1 for b in block_metas if b['is_outlier_fp8'])}")


if __name__ == "__main__":
    main()
