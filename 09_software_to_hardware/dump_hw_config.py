"""
dump_hw_config.py
=================
Emit a per-layer precision / shape / fusion configuration JSON for the
hardware team. **No weights are written by this script** — only metadata.

Each entry in the output JSON describes one torch module that the MAC
array will have to execute: precision, scale granularity, packed-weight
shape, expected I/O activation dtype, and which post-MAC fusion slots
(BN scale, BN bias, PReLU α, 1/√D) apply.

Usage:
    python dump_hw_config.py --out layer_config.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

REPO_09 = "/raid/R12K41024/09_weight_tied_mha"
FINETUNE_STRICT_RUN = os.path.join(
    REPO_09, "outputs/finetune_strict/2026-05-10_02-40-16"
)
DEFAULT_CONFIG = os.path.join(FINETUNE_STRICT_RUN, ".hydra/config.yaml")

if REPO_09 not in sys.path:
    sys.path.insert(0, REPO_09)

from omegaconf import OmegaConf  # noqa: E402
from DCNN.models.binaural_attention_model import BinauralAttentionDCNN  # noqa: E402
from DCNN.utils.mixed_precision_qat import enable_mixed_precision_qat  # noqa: E402

import export_helpers as eh  # noqa: E402


# ──────────────────────────────────────────────────────────────
# Config dict emitters
# ──────────────────────────────────────────────────────────────

def conv_weight_shape(m: nn.Module) -> List[int]:
    return list(m.weight.shape) if not hasattr(m, "parametrizations") \
        or "weight" not in m.parametrizations \
        else list(m.parametrizations["weight"].original.shape)


def conv_cfg(module: nn.Module, dotted: str,
             weight_format: str, scale_gran: str,
             group_size: Optional[int] = None,
             num_heads: Optional[int] = None,
             is_depthwise: bool = False,
             is_pointwise: bool = False,
             is_transposed: bool = False) -> Dict[str, Any]:
    shp = conv_weight_shape(module)
    ks = module.kernel_size if hasattr(module, "kernel_size") else None
    stride = module.stride if hasattr(module, "stride") else None
    padding = module.padding if hasattr(module, "padding") else None
    cfg = {
        "module_type": type(module).__name__,
        "dotted_name": dotted,
        "weight_shape": shp,
        "weight_format": weight_format,       # "int4" or "fp8_e4m3"
        "scale_granularity": scale_gran,      # "per_channel" / "per_group" / "per_head" / "per_tensor"
        "act_format": "fp8_e4m3",
        "accumulator": "fp16",
        "is_depthwise": is_depthwise,
        "is_pointwise": is_pointwise,
        "is_transposed": is_transposed,
        "is_complex_half": True,              # this module is the real or imag half of a complex op
    }
    if ks is not None:
        cfg["kernel_size"] = list(ks) if hasattr(ks, "__iter__") else [ks]
    if stride is not None:
        cfg["stride"] = list(stride) if hasattr(stride, "__iter__") else [stride]
    if padding is not None:
        cfg["padding"] = list(padding) if hasattr(padding, "__iter__") else [padding]
    if group_size is not None:
        cfg["group_size"] = group_size
    if num_heads is not None:
        cfg["num_heads"] = num_heads
    if is_depthwise:
        cfg["groups"] = module.groups
    if hasattr(module, "bias") and module.bias is not None:
        cfg["has_conv_bias"] = True
    else:
        cfg["has_conv_bias"] = False
    return cfg


def block_cfg(block: eh.BlockInfo) -> Dict[str, Any]:
    """Metadata describing the post-MAC fusion for one encoder/decoder block."""
    return {
        "kind": block.kind,
        "stage_idx": block.stage_idx,
        "dotted": block.dotted,
        "num_out_channels": block.num_out_channels,
        "has_bn": block.bn_r is not None,
        "has_prelu": block.prelu_r is not None,
        "is_outlier_fp8": block.is_outlier_fp8,
        # Which conv's output the BN fuses into:
        "bn_attaches_to": "pointwise" if block.kind == "encoder" else "depthwise",
        # Post-MAC operations hardware must execute (in order):
        # 1. scale_{r,i}  2. + bias_{r,i}  3. PReLU (if present)
        "post_mac_fma_real": {
            "scale_name": f"blocks/{block.kind}/stage_{block.stage_idx}/bn_scale_r.fp16.bin",
            "bias_name":  f"blocks/{block.kind}/stage_{block.stage_idx}/bn_bias_r.fp16.bin",
            "length": block.num_out_channels,
            "dtype": "fp16",
        },
        "post_mac_fma_imag": {
            "scale_name": f"blocks/{block.kind}/stage_{block.stage_idx}/bn_scale_i.fp16.bin",
            "bias_name":  f"blocks/{block.kind}/stage_{block.stage_idx}/bn_bias_i.fp16.bin",
            "length": block.num_out_channels,
            "dtype": "fp16",
        },
        "prelu": {
            "alpha_r_file": f"blocks/{block.kind}/stage_{block.stage_idx}/prelu_alpha_r.fp32.bin",
            "alpha_i_file": f"blocks/{block.kind}/stage_{block.stage_idx}/prelu_alpha_i.fp32.bin",
            "num_parameters": 1,
            "dtype": "fp32",
        } if block.prelu_r is not None else None,
    }


def build_config(model: nn.Module, num_heads: int, embed_dim: int,
                 int4_group_size: int, top_k_attention: Optional[int] = 4,
                 model_cfg: Optional[Dict[str, Any]] = None,
                 fp16_accumulator: bool = False) -> Dict[str, Any]:
    model_cfg = model_cfg or {}
    tied_mha = bool(getattr(getattr(model, "mattn", None).mattn, "tied", False)) \
        if hasattr(getattr(model, "mattn", None), "mattn") else False
    cfg_out: Dict[str, Any] = {
        "model": {
            "name": "BinauralAttentionDCNN",
            "embed_dim": embed_dim,
            "num_heads": num_heads,
            "head_dim": embed_dim // num_heads,
            "int4_group_size": int4_group_size,
            "activation_format": "fp8_e4m3",
            "accumulator": "fp16",
            "qk_scale_fused_into_q": True,       # 1/sqrt(D) pre-folded
            "softmax_topk_k": top_k_attention,
            "topk_renorm_after_softmax": top_k_attention is not None,
            "tied_mha": tied_mha,
            "fp16_accumulator": fp16_accumulator,
        },
        "blocks": [],
        "layers": [],
        "attention": {},
        "stft": {},
    }

    blocks = eh.discover_blocks(model)
    for b in blocks:
        cfg_out["blocks"].append(block_cfg(b))

        # Depthwise: per-channel INT4.
        if b.depthwise_r is not None:
            for half, sub in (("r", b.depthwise_r), ("i", b.depthwise_i)):
                cfg_out["layers"].append(conv_cfg(
                    sub, f"{b.dotted}.depthwise_{half}",
                    weight_format="int4", scale_gran="per_channel",
                    is_depthwise=True,
                    is_transposed=(b.kind == "decoder"),
                ))

        # Pointwise: INT4 per-group requested normally; FP8 for decoder outlier stages.
        if b.pointwise_r is not None:
            pw_fmt = "fp8_e4m3" if b.is_outlier_fp8 else "int4"
            pw_gran = "per_tensor" if b.is_outlier_fp8 else "per_group"
            for half, sub in (("r", b.pointwise_r), ("i", b.pointwise_i)):
                layer = conv_cfg(
                    sub, f"{b.dotted}.pointwise_{half}",
                    weight_format=pw_fmt, scale_gran=pw_gran,
                    is_pointwise=True,
                    group_size=(int4_group_size if pw_fmt == "int4" else None),
                )
                if pw_fmt == "int4":
                    last = conv_weight_shape(sub)[-1]
                    layer["effective_scale_granularity"] = (
                        "per_group" if (last % int4_group_size == 0 and last != 1) else "per_channel"
                    )
                cfg_out["layers"].append(layer)

    # ── Attention block ──
    try:
        matt = model.mattn
    except AttributeError:
        matt = None

    if matt is not None:
        head_dim = embed_dim // num_heads
        sqrt_d = math.sqrt(head_dim)

        # input_proj (ComplexLinear: complex weight, split into fc_r / fc_i)
        # NOTE: current codebase declares it as a plain nn.Linear with complex64 dtype.
        ip = matt.input_proj
        op = matt.output_proj
        in_cfg = {
            "module_type": type(ip).__name__,
            "dotted_name": "mattn.input_proj",
            "weight_shape": list(ip.weight.shape) if not hasattr(ip, "parametrizations") or "weight" not in ip.parametrizations else list(ip.parametrizations["weight"].original.shape),
            "weight_dtype": "complex64",
            "weight_format": "int4",
            "scale_granularity": "per_head",
            "num_heads": num_heads,
            "act_format": "fp8_e4m3",
            "accumulator": "fp16",
            "has_bias": ip.bias is not None,
        }
        out_cfg_ = {
            "module_type": type(op).__name__,
            "dotted_name": "mattn.output_proj",
            "weight_shape": list(op.weight.shape) if not hasattr(op, "parametrizations") or "weight" not in op.parametrizations else list(op.parametrizations["weight"].original.shape),
            "weight_dtype": "complex64",
            "weight_format": "int4",
            "scale_granularity": "per_head",
            "num_heads": num_heads,
            "act_format": "fp8_e4m3",
            "accumulator": "fp16",
            "has_bias": op.bias is not None,
        }

        # ComplexMultiheadAttention: tied strict model has only MultiheadAttention_re.
        cmha = matt.mattn
        mha_re = cmha.MultiheadAttention_re
        mha_im = getattr(cmha, "MultiheadAttention_im", None)

        def _mha_cfg(mha, dotted):
            in_proj_shape = list(mha.in_proj_weight.shape) if not hasattr(mha, "parametrizations") or "in_proj_weight" not in mha.parametrizations else list(mha.parametrizations["in_proj_weight"].original.shape)
            return {
                "module_type": "MultiheadAttention",
                "dotted_name": dotted,
                "in_proj_weight_shape": in_proj_shape,
                "in_proj_bias_length": in_proj_shape[0],
                "in_proj_groups_Q_K_V": 3 * num_heads,
                "in_proj_scale_granularity": "per_head_qkv_fused",
                "q_rows_prescaled_by_inv_sqrt_D": True,
                "out_proj_weight_shape": list(mha.out_proj.weight.shape) if not hasattr(mha.out_proj, "parametrizations") or "weight" not in mha.out_proj.parametrizations else list(mha.out_proj.parametrizations["weight"].original.shape),
                "out_proj_scale_granularity": "per_head",
                "num_heads": num_heads,
                "head_dim": head_dim,
                "weight_format": "int4",
                "act_format": "fp8_e4m3",
                "accumulator": "fp16",
                "qk_scale_inv_sqrt_D": 1.0 / sqrt_d,
                "softmax_followed_by_topk": top_k_attention is not None,
                "topk_k": top_k_attention,
                "fp16_accumulator": fp16_accumulator,
            }

        cfg_out["attention"] = {
            "tied_mha": tied_mha,
            "input_proj": in_cfg,
            "output_proj": out_cfg_,
            "mha_shared": _mha_cfg(mha_re, "mattn.mattn.MultiheadAttention_re"),
            "mha_imag_half": (
                _mha_cfg(mha_im, "mattn.mattn.MultiheadAttention_im")
                if mha_im is not None else None
            ),
            "complex_flow": (
                "2-pass tied: a=M(x.real), b=M(x.imag), output=(a-b)+j(a+b)"
                if tied_mha else "4-pass (re×re, im×im, re×im, im×re)"
            ),
            "window_size": getattr(matt, "window_size", None),
        }

    # ── STFT / iSTFT ──
    try:
        stft_mod = model.stft
        cfg_out["stft"] = {
            "win_len": getattr(stft_mod, "win_len", None),
            "win_inc": getattr(stft_mod, "win_inc", None),
            "fft_len": getattr(stft_mod, "fft_len", None),
            "win_type": model_cfg.get("win_type", "hamming"),
            "window_periodic": False,
            "center": False,
            "normalized": True,
            "dc_bin_removed_before_model": True,
            "window_file": "misc/stft_window.fp16.bin",
            "masking_mode": getattr(model, "masking_mode", "E"),
        }
    except AttributeError:
        pass

    return cfg_out


# ──────────────────────────────────────────────────────────────
# Entry
# ──────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CONFIG,
                    help="Hydra config file used to build the model skeleton.")
    ap.add_argument("--out", default="layer_config.json",
                    help="Output JSON path.")
    ap.add_argument("--no-qat", action="store_true",
                    help="Do not attach mixed-precision parametrizations (for a model without QAT).")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    model_cfg = OmegaConf.to_container(cfg.model)
    model = BinauralAttentionDCNN(**model_cfg).eval()

    num_heads = model_cfg.get("num_heads", 16)
    embed_dim = model_cfg.get("embed_dim", 256)
    training_cfg = OmegaConf.to_container(cfg.training)
    int4_group_size = training_cfg.get("int4_group_size", 256)

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

    hw_cfg = build_config(
        model, num_heads, embed_dim, int4_group_size,
        top_k_attention=training_cfg.get("top_k_attention", 4),
        model_cfg=model_cfg,
        fp16_accumulator=training_cfg.get("fp16_accumulator", False),
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(hw_cfg, f, indent=2)
    print(f"wrote {args.out}")
    print(f"  blocks: {len(hw_cfg['blocks'])}  layers: {len(hw_cfg['layers'])}")
    n_outlier = sum(1 for b in hw_cfg["blocks"] if b["is_outlier_fp8"])
    print(f"  outlier FP8 blocks: {n_outlier}")


if __name__ == "__main__":
    main()
