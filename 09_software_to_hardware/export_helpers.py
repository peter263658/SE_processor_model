"""
Shared helpers for exporting the Mixed-Precision W4A8 BCCTN model to
hardware-friendly binaries.

Responsibilities:
  * Recompute INT4 symmetric quantization (integer values + scale) to
    match what 07_mixed_precision_qat/DCNN/utils/mixed_precision_qat.py
    does at runtime.
  * Fold BatchNorm + conv-bias into post-MAC per-channel (scale, bias).
  * Fold 1/sqrt(head_dim) into Q rows of MHA in_proj_weight / bias.
  * Pack INT4 integers two-per-byte for weight SRAM.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# ──────────────────────────────────────────────────────────────
# INT4 symmetric quant — mirror mixed_precision_qat.py
# ──────────────────────────────────────────────────────────────

INT4_QMAX = 7  # 2^(4-1) - 1
SCALE_EPS = 1e-12


def int4_per_channel(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """One scalar scale per output channel (first dim).
    Returns (int8_values in [-7,7], scale shaped to broadcast)."""
    shape = [w.shape[0]] + [1] * (w.dim() - 1)
    amax = w.abs().flatten(1).amax(dim=1).view(shape)
    scale = (amax / INT4_QMAX).clamp_(min=SCALE_EPS)
    q = torch.round(w / scale).clamp_(-INT4_QMAX, INT4_QMAX).to(torch.int8)
    return q, scale


def int4_per_group_last_dim(w: torch.Tensor, group_size: int) -> Tuple[torch.Tensor, torch.Tensor, bool]:
    """Per-group along last dim. Returns (int8 values in original shape,
    scale with broadcast shape, used_per_group flag).

    Matches mixed_precision_qat.py: if last dim not divisible by group_size,
    falls back to per-channel.
    """
    last = w.shape[-1]
    if last % group_size != 0 or last == 1:
        q, s = int4_per_channel(w)
        return q, s, False
    grouped = w.reshape(w.shape[:-1] + (last // group_size, group_size))
    amax = grouped.abs().amax(dim=-1, keepdim=True)
    scale = (amax / INT4_QMAX).clamp_(min=SCALE_EPS)
    q = torch.round(grouped / scale).clamp_(-INT4_QMAX, INT4_QMAX).to(torch.int8)
    return q.reshape_as(w), scale, True


def int4_per_head_mha(w: torch.Tensor, num_heads: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """MHA in_proj_weight: shape (3*E, E). Split rows into (3*H) groups.
    Returns (int8 values same shape as w, scale shape (3*H, 1, 1))."""
    rows, cols = w.shape
    total_groups = 3 * num_heads
    assert rows % total_groups == 0
    head_dim = rows // total_groups
    grouped = w.reshape(total_groups, head_dim, cols)
    amax = grouped.abs().flatten(1).amax(dim=1).view(total_groups, 1, 1)
    scale = (amax / INT4_QMAX).clamp_(min=SCALE_EPS)
    q = torch.round(grouped / scale).clamp_(-INT4_QMAX, INT4_QMAX).to(torch.int8)
    return q.reshape_as(w), scale


def int4_per_head_linear(w: torch.Tensor, num_heads: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """MHA out_proj or real-valued Linear: shape (out, in). Split first dim
    into num_heads groups. Returns int8 same shape, scale shape (H,1,1)."""
    rows, cols = w.shape
    if rows % num_heads == 0:
        head_dim = rows // num_heads
        grouped = w.reshape(num_heads, head_dim, cols)
        amax = grouped.abs().flatten(1).amax(dim=1).view(num_heads, 1, 1)
        scale = (amax / INT4_QMAX).clamp_(min=SCALE_EPS)
        q = torch.round(grouped / scale).clamp_(-INT4_QMAX, INT4_QMAX).to(torch.int8)
        return q.reshape_as(w), scale
    return int4_per_channel(w)


# ──────────────────────────────────────────────────────────────
# INT4 packing (2 nibbles per byte, little-endian)
# ──────────────────────────────────────────────────────────────

def pack_int4_nibbles(q_int8: torch.Tensor) -> np.ndarray:
    """Pack a signed INT4 tensor (values in [-7,7]) into uint8 bytes,
    two nibbles per byte. Low nibble = first element, high nibble = second.
    """
    flat = q_int8.flatten().to(torch.int8).numpy()
    n = flat.size
    if n % 2 == 1:
        flat = np.concatenate([flat, np.zeros(1, dtype=np.int8)])
    lo = (flat[0::2] & 0x0F).astype(np.uint8)
    hi = (flat[1::2] & 0x0F).astype(np.uint8)
    return (lo | (hi << 4)).astype(np.uint8)


# ──────────────────────────────────────────────────────────────
# BatchNorm fusion (post-MAC per-channel scale + fused bias)
# ──────────────────────────────────────────────────────────────

@dataclass
class BnFuse:
    """Result of folding BN + the split complex conv biases into post-MAC
    per-channel (scale, bias) pairs — one pair for the real output channel,
    one for the imaginary output channel.

    Hardware applies AFTER the complex MAC:
        out_real[c] = bn_scale_r[c] * mac_real[c] + bn_bias_r[c]
        out_imag[c] = bn_scale_i[c] * mac_imag[c] + bn_bias_i[c]
    """
    bn_scale_r: torch.Tensor  # (C,) fp32
    bn_bias_r: torch.Tensor   # (C,)
    bn_scale_i: torch.Tensor
    bn_bias_i: torch.Tensor


def fuse_bn_into_post_mac(
    bn_r: Optional[nn.BatchNorm2d],
    bn_i: Optional[nn.BatchNorm2d],
    conv_bias_r: Optional[torch.Tensor],
    conv_bias_i: Optional[torch.Tensor],
    num_channels: int,
) -> BnFuse:
    """Produce per-channel post-MAC scale+bias that absorbs:

      * BN mean/var/gamma/beta on real output
      * BN mean/var/gamma/beta on imag output
      * The effective complex conv bias:
          - real output gets (b_r - b_i)
          - imag output gets (b_r + b_i)

    If BN is absent (e.g. last decoder block), bn_scale=1 and bn_bias=
    the effective conv bias alone.
    """
    device = conv_bias_r.device if conv_bias_r is not None else torch.device("cpu")
    zeros = torch.zeros(num_channels, device=device)
    b_r = conv_bias_r if conv_bias_r is not None else zeros
    b_i = conv_bias_i if conv_bias_i is not None else zeros
    eff_bias_real = b_r - b_i
    eff_bias_imag = b_r + b_i

    if bn_r is None:
        ones = torch.ones(num_channels, device=device)
        return BnFuse(ones.clone(), eff_bias_real.clone(),
                      ones.clone(), eff_bias_imag.clone())

    eps = float(bn_r.eps)
    s_r = bn_r.weight / torch.sqrt(bn_r.running_var + eps)
    bias_r = s_r * (eff_bias_real - bn_r.running_mean) + bn_r.bias

    s_i = bn_i.weight / torch.sqrt(bn_i.running_var + eps)
    bias_i = s_i * (eff_bias_imag - bn_i.running_mean) + bn_i.bias

    return BnFuse(s_r.detach().clone(),  bias_r.detach().clone(),
                  s_i.detach().clone(),  bias_i.detach().clone())


# ──────────────────────────────────────────────────────────────
# Attention 1/sqrt(D) fusion into Q rows
# ──────────────────────────────────────────────────────────────

def fuse_qk_scale_into_in_proj(
    in_proj_weight_int8: torch.Tensor,   # shape (3E, E), int8
    in_proj_scale: torch.Tensor,         # shape (3H, 1, 1), fp32
    in_proj_bias: Optional[torch.Tensor],
    embed_dim: int,
    num_heads: int,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Divide the Q portion by sqrt(head_dim):
      - Pre-divide scale[0:H] by sqrt(D). INT4 integers unchanged (because
        round(W/scale) is invariant under (W,scale) → (kW, k·scale)).
      - Pre-divide bias[0:E] by sqrt(D).
    Hardware then computes Q·K^T directly, without the divide.
    """
    D = embed_dim // num_heads
    inv = 1.0 / math.sqrt(D)
    scale_fused = in_proj_scale.clone()
    scale_fused[0:num_heads] = scale_fused[0:num_heads] * inv
    if in_proj_bias is not None:
        bias_fused = in_proj_bias.clone()
        bias_fused[0:embed_dim] = bias_fused[0:embed_dim] * inv
    else:
        bias_fused = None
    return in_proj_weight_int8, scale_fused, bias_fused


# ──────────────────────────────────────────────────────────────
# Checkpoint loading
# ──────────────────────────────────────────────────────────────

def load_state_dict_from_ckpt(ckpt_path: str) -> Dict[str, torch.Tensor]:
    """Load either a raw state_dict or a PyTorch-Lightning checkpoint."""
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "state_dict" in obj:
        sd = obj["state_dict"]
    else:
        sd = obj
    # Strip Lightning "model." prefix on module keys.
    new_sd = {}
    for k, v in sd.items():
        if k.startswith("model."):
            new_sd[k[len("model."):]] = v
        else:
            new_sd[k] = v
    return new_sd


def raw_weight(module: nn.Module) -> torch.Tensor:
    """Return the underlying fp32 weight, skipping any registered parametrization."""
    if hasattr(module, "parametrizations") and "weight" in module.parametrizations:
        return module.parametrizations["weight"].original.detach()
    return module.weight.detach()


def raw_in_proj_weight(mha: nn.MultiheadAttention) -> torch.Tensor:
    if hasattr(mha, "parametrizations") and "in_proj_weight" in mha.parametrizations:
        return mha.parametrizations["in_proj_weight"].original.detach()
    return mha.in_proj_weight.detach()


# ──────────────────────────────────────────────────────────────
# FP8 cast (for outlier decoder PW layers)
# ──────────────────────────────────────────────────────────────

def cast_to_fp8_e4m3(w: torch.Tensor) -> torch.Tensor:
    """Fake-cast to FP8 E4M3 and back to fp32 for export. Returns fp32
    tensor whose values are representable in E4M3. For true hardware
    export, the raw bytes are obtainable by .to(torch.float8_e4m3fn).view(torch.uint8).
    """
    return w.to(torch.float8_e4m3fn).to(torch.float32)


def fp8_e4m3_bytes(w: torch.Tensor) -> np.ndarray:
    """Return the raw 1-byte-per-element FP8 E4M3 representation."""
    return w.to(torch.float8_e4m3fn).view(torch.uint8).cpu().numpy()


# ──────────────────────────────────────────────────────────────
# Model & block discovery
# ──────────────────────────────────────────────────────────────

DECODER_OUTLIER_STAGES = {0, 3, 4}  # strict tied-MHA handoff keeps these decoder PW layers in FP8


@dataclass
class BlockInfo:
    """One Conv-BN-PReLU block (encoder or decoder stage)."""
    kind: str              # "encoder" | "decoder"
    stage_idx: int         # index into self.model (0-based within encoder/decoder)
    dotted: str            # torch path prefix, e.g. "encoder.model.0.0"
    # Sub-modules (real & imag for depthwise and pointwise)
    depthwise_r: Optional[nn.Module]
    depthwise_i: Optional[nn.Module]
    pointwise_r: Optional[nn.Module]
    pointwise_i: Optional[nn.Module]
    # BN (Optional: last decoder block has no BN)
    bn_r: Optional[nn.BatchNorm2d]
    bn_i: Optional[nn.BatchNorm2d]
    # PReLU (Optional)
    prelu_r: Optional[nn.PReLU]
    prelu_i: Optional[nn.PReLU]
    num_out_channels: int  # output channels of the pointwise (encoder) / depthwise (decoder)
    is_outlier_fp8: bool   # decoder outlier PW stays FP8


def discover_blocks(model: nn.Module) -> List[BlockInfo]:
    """Walk the Encoder / Decoder modules and pack each stage into a
    BlockInfo. Order in the returned list follows forward order: all
    encoder stages, then all decoder stages.
    """
    blocks: List[BlockInfo] = []

    enc = model.encoder if hasattr(model, "encoder") else model
    dec = model.decoder if hasattr(model, "decoder") else None

    # Encoder: each nn.Sequential(conv_layer, BN, PReLU)
    for idx, block in enumerate(enc.model):
        conv = block[0]      # ComplexDepthwiseSeparableConv2d
        bn = block[1] if len(block) > 1 else None
        pr = block[2] if len(block) > 2 else None
        pw_r = conv.pointwise_r
        pw_i = conv.pointwise_i
        dw_r = conv.depthwise_r
        dw_i = conv.depthwise_i
        num_out = pw_r.out_channels
        blocks.append(BlockInfo(
            kind="encoder",
            stage_idx=idx,
            dotted=f"encoder.model.{idx}.0",
            depthwise_r=dw_r, depthwise_i=dw_i,
            pointwise_r=pw_r, pointwise_i=pw_i,
            bn_r=(bn.bn_r if bn is not None else None),
            bn_i=(bn.bn_i if bn is not None else None),
            prelu_r=(pr.r_prelu if pr is not None else None),
            prelu_i=(pr.i_prelu if pr is not None else None),
            num_out_channels=num_out,
            is_outlier_fp8=False,
        ))

    if dec is not None:
        for idx, block in enumerate(dec.model):
            conv = block[0]   # ComplexDepthwiseSeparableConvTranspose2d
            bn = block[1] if len(block) > 1 else None
            pr = block[2] if len(block) > 2 else None
            # In decoder, pointwise first, depthwise (ConvTranspose2d) second
            pw_r = conv.pointwise_r
            pw_i = conv.pointwise_i
            dw_r = conv.depthwise_r
            dw_i = conv.depthwise_i
            # BN sits after depthwise (post-MAC on depthwise output channels)
            num_out = dw_r.out_channels
            blocks.append(BlockInfo(
                kind="decoder",
                stage_idx=idx,
                dotted=f"decoder.model.{idx}.0",
                depthwise_r=dw_r, depthwise_i=dw_i,
                pointwise_r=pw_r, pointwise_i=pw_i,
                bn_r=(bn.bn_r if bn is not None else None),
                bn_i=(bn.bn_i if bn is not None else None),
                prelu_r=(pr.r_prelu if pr is not None else None),
                prelu_i=(pr.i_prelu if pr is not None else None),
                num_out_channels=num_out,
                is_outlier_fp8=(idx in DECODER_OUTLIER_STAGES),
            ))

    return blocks
