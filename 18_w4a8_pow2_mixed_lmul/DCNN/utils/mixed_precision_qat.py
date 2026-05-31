"""
Mixed-Precision W4A8 QAT Module
================================
Extends fp8_qat.py with:
  * INT4 symmetric fake quantization (per-channel / per-group / per-head)
  * Per-layer precision routing (Encoder/Decoder/MHA) driven by a config dict
  * FP8 preservation for outlier decoder PW layers
  * FP8 activation quantization (reuses fp8_qat helpers)
  * Top-k attention wiring on ComplexMultiheadAttention instances

Design:
  * Weights are quantized via torch.nn.utils.parametrize — STE is baked into
    the fake-quant autograd Function so gradients flow through unchanged.
  * Activations are quantized with a forward-pre-hook that casts inputs to
    FP8 (E4M3 by default).
  * MHA in_proj_weight is sliced into 3 * num_heads groups (Q/K/V × heads);
    each group gets its own scalar scale (per-head grouping).
  * Conv PW / Linear per-group: groups run along the last (in_channels) dim.
  * Conv DW / per-channel: one scale per output channel.

References: 07_mixed_precision_qat_handoff.md §2.1 precision table.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.utils.parametrize import register_parametrization

from DCNN.utils.fp8_qat import (
    FP8Parametrization,
    FP8ComplexParametrization,
    _fp8_quantize_tensor,
    fp8_quantize,
)


# ──────────────────────────────────────────────────────────────
# INT4 symmetric STE
# ──────────────────────────────────────────────────────────────

class _Int4QuantSTE(torch.autograd.Function):
    """Symmetric INT4 fake quant with a user-supplied scale tensor.

    Scale is broadcast over `x`. Pass `None` shapes through unchanged so
    the STE gradient is the identity.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: torch.Tensor, qmax: int):
        q = torch.round(x / scale).clamp_(-qmax, qmax)
        return q * scale

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


def _int4_qmax(num_bits: int = 4) -> int:
    return 2 ** (num_bits - 1) - 1  # INT4 → 7


def _quantize_scale(scale: torch.Tensor, scale_format: str) -> torch.Tensor:
    """Optional fake quantization for INT4 scale tensors (P3-C2)."""
    fmt = str(scale_format or "fp32").lower()
    if fmt in ("fp32", "fp16", "none", "off"):
        return scale
    if fmt in ("fp8_e4m3", "e4m3"):
        return scale.to(torch.float8_e4m3fn).to(scale.dtype).clamp(min=1e-12)
    if fmt in ("fp8_e5m2", "e5m2"):
        return scale.to(torch.float8_e5m2).to(scale.dtype).clamp(min=1e-12)
    if fmt in ("pow2", "power_of_two"):
        exponent = torch.round(torch.log2(scale.clamp_min(1e-12)))
        return torch.pow(torch.full_like(scale, 2.0), exponent).clamp(min=1e-12)
    if fmt in ("apot", "apot2", "apot_2", "two_pow2", "two_term_pow2", "apot_clip", "apot_clipped"):
        safe = scale.clamp_min(1e-12)
        primary_exp = torch.round(torch.log2(safe))
        if fmt in ("apot_clip", "apot_clipped"):
            a_min = float(os.environ.get("BCCTN_APOT_A_MIN", "-16"))
            a_max = float(os.environ.get("BCCTN_APOT_A_MAX", "-1"))
            primary_exp = primary_exp.clamp(min=a_min, max=a_max)
        primary = torch.pow(torch.full_like(scale, 2.0), primary_exp)
        residual = safe - primary
        abs_residual = residual.abs()
        has_second = abs_residual > 1e-12
        second_exp = torch.round(torch.log2(abs_residual.clamp_min(1e-12)))
        if fmt in ("apot_clip", "apot_clipped"):
            b_min = float(os.environ.get("BCCTN_APOT_B_MIN", "-16"))
            b_max = float(os.environ.get("BCCTN_APOT_B_MAX", "-1"))
            second_exp = second_exp.clamp(min=b_min, max=b_max)
            has_second = has_second & (second_exp < primary_exp)
        second_mag = torch.pow(torch.full_like(scale, 2.0), second_exp)
        second = torch.sign(residual) * torch.where(
            has_second, second_mag, torch.zeros_like(second_mag)
        )
        return (primary + second).clamp(min=1e-12)
    raise ValueError(f"Unsupported INT4 scale_format: {scale_format}")


# ──────────────────────────────────────────────────────────────
# Scale computation helpers (per-channel / per-group / per-head)
# ──────────────────────────────────────────────────────────────

def _scale_per_channel(w: torch.Tensor, qmax: int) -> torch.Tensor:
    """One scale per output channel (first dim). Returns broadcast-shaped scale."""
    shape = [w.shape[0]] + [1] * (w.dim() - 1)
    amax = w.abs().flatten(1).amax(dim=1).view(shape)
    return (amax / qmax).clamp(min=1e-12)


def _scale_per_group_last_dim(w: torch.Tensor, group_size: int, qmax: int) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    """Per-group scale along the input-channels axis.

    For Conv2d weight (out, in, kH, kW) the in-channels axis is dim=1.
    For Linear weight (out, in) the in-channels axis is dim=-1 (= dim=1).
    PW conv has kW=1, so grouping the trailing axis is a no-op — group along
    in_ch to actually exploit per-group scales.
    """
    if w.dim() == 4:
        out_ch, in_ch, kH, kW = w.shape
        if in_ch < group_size or in_ch % group_size != 0:
            return _scale_per_channel(w, qmax), w.shape
        n_groups = in_ch // group_size
        new_shape = (out_ch, n_groups, group_size, kH, kW)
        grouped = w.reshape(new_shape)
        amax = grouped.abs().amax(dim=(2, 3, 4), keepdim=True)  # (out, n_groups, 1, 1, 1)
        scale = (amax / qmax).clamp(min=1e-12)
        return scale, new_shape

    last = w.shape[-1]
    if last % group_size != 0:
        return _scale_per_channel(w, qmax), w.shape
    new_shape = w.shape[:-1] + (last // group_size, group_size)
    grouped = w.reshape(new_shape)
    amax = grouped.abs().amax(dim=-1, keepdim=True)
    scale = (amax / qmax).clamp(min=1e-12)
    return scale, new_shape


class Int4PerChannelParametrization(nn.Module):
    """Per-output-channel INT4 fake quant for Conv/Linear real weights."""

    def __init__(self, num_bits: int = 4, scale_format: str = "fp32"):
        super().__init__()
        self.qmax = _int4_qmax(num_bits)
        self.scale_format = scale_format

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        scale = _quantize_scale(_scale_per_channel(w, self.qmax), self.scale_format)
        return _Int4QuantSTE.apply(w, scale, self.qmax)


class Int4PerGroupParametrization(nn.Module):
    """Per-group INT4 fake quant along the last (input) dim."""

    def __init__(self, group_size: int = 256, num_bits: int = 4,
                 scale_format: str = "fp32"):
        super().__init__()
        self.group_size = group_size
        self.qmax = _int4_qmax(num_bits)
        self.scale_format = scale_format

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        scale, grouped_shape = _scale_per_group_last_dim(w, self.group_size, self.qmax)
        scale = _quantize_scale(scale, self.scale_format)
        if grouped_shape == w.shape:
            # per-channel fallback
            return _Int4QuantSTE.apply(w, scale, self.qmax)
        grouped = w.reshape(grouped_shape)
        q = _Int4QuantSTE.apply(grouped, scale, self.qmax)
        return q.reshape_as(w)


class Int4PerHeadMHAParametrization(nn.Module):
    """Per-head INT4 fake quant for nn.MultiheadAttention.in_proj_weight.

    in_proj_weight shape = (3 * E, E). Split rows into (3 * H) groups of
    head_dim rows each, compute a scalar scale per group.
    """

    def __init__(self, num_heads: int, num_bits: int = 4,
                 scale_format: str = "fp32"):
        super().__init__()
        self.num_heads = num_heads
        self.qmax = _int4_qmax(num_bits)
        self.scale_format = scale_format

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        rows, cols = w.shape
        total_groups = 3 * self.num_heads
        assert rows % total_groups == 0, \
            f"in_proj_weight rows={rows} not divisible by 3*num_heads={total_groups}"
        head_dim = rows // total_groups
        grouped = w.reshape(total_groups, head_dim, cols)
        amax = grouped.abs().flatten(1).amax(dim=1).view(total_groups, 1, 1)
        scale = _quantize_scale((amax / self.qmax).clamp(min=1e-12), self.scale_format)
        q = _Int4QuantSTE.apply(grouped, scale, self.qmax)
        return q.reshape_as(w)


class Int4PerHeadLinearParametrization(nn.Module):
    """Per-head INT4 fake quant for a single nn.Linear weight (E, E) in MHA
    flow — used on input_proj / output_proj / out_proj where rows split by
    num_heads head_dim blocks.

    Weight shape (out, in). Along whichever dim matches num_heads*head_dim,
    split into num_heads groups. We pick the first dim whose size is
    divisible by num_heads. If both are, we default to rows (out).
    """

    def __init__(self, num_heads: int, num_bits: int = 4,
                 scale_format: str = "fp32"):
        super().__init__()
        self.num_heads = num_heads
        self.qmax = _int4_qmax(num_bits)
        self.scale_format = scale_format

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        rows, cols = w.shape
        if rows % self.num_heads == 0:
            head_dim = rows // self.num_heads
            grouped = w.reshape(self.num_heads, head_dim, cols)
            amax = grouped.abs().flatten(1).amax(dim=1).view(self.num_heads, 1, 1)
        elif cols % self.num_heads == 0:
            head_dim = cols // self.num_heads
            grouped = w.reshape(rows, self.num_heads, head_dim)
            # per-(row, head) scale
            amax = grouped.abs().amax(dim=-1, keepdim=True)
        else:
            # fallback to per-channel
            return Int4PerChannelParametrization(
                num_bits=4, scale_format=self.scale_format
            )(w)
        scale = _quantize_scale((amax / self.qmax).clamp(min=1e-12), self.scale_format)
        q = _Int4QuantSTE.apply(grouped, scale, self.qmax)
        return q.reshape_as(w)


# ──────────────────────────────────────────────────────────────
# Activation hook (FP8 on inputs)
# ──────────────────────────────────────────────────────────────

def _make_fp8_activation_hook(fp8_dtype):
    def hook(module, args):
        new_args = []
        for a in args:
            if isinstance(a, torch.Tensor):
                new_args.append(fp8_quantize(a, fp8_dtype))
            else:
                new_args.append(a)
        return tuple(new_args)
    return hook


def _make_split_pointwise_activation_hook(fp8_dtype):
    """Quantize split decoder pointwise inputs with one legacy cat-scale.

    Without this hook, pointwise_x and pointwise_skip would each get their own
    Conv2d pre-hook and therefore their own dynamic activation scale. That is a
    different quantization experiment from grouped pointwise. This preserves the
    legacy cat([x, skip]) activation quantization boundary.
    """
    def hook(module, args):
        if len(args) < 2:
            return args
        x, skip = args[0], args[1]
        if not isinstance(x, torch.Tensor) or not isinstance(skip, torch.Tensor):
            return args
        if x.shape[0] != skip.shape[0] or x.shape[2:] != skip.shape[2:]:
            return args
        x_channels = x.shape[1]
        cat_q = fp8_quantize(torch.cat([x, skip], dim=1), fp8_dtype)
        return (cat_q[:, :x_channels], cat_q[:, x_channels:], *args[2:])
    return hook


# ──────────────────────────────────────────────────────────────
# Layer routing
# ──────────────────────────────────────────────────────────────

_ENCODER_RE = re.compile(r"\.encoder\.model\.(\d+)\.0\.(depthwise|pointwise)_[ri]$")
_DECODER_RE = re.compile(
    r"\.decoder\.model\.(\d+)\.0\.(pointwise|depthwise)_[ri](?:_(?:x|skip))?$"
)
_DECODER_SPLIT_POINTWISE_RE = re.compile(
    r"\.decoder\.model\.\d+\.0\.pointwise_[ri]_(?:x|skip)$"
)
_MHA_CORE_RE = re.compile(r"\.mattn\.mattn\.MultiheadAttention_(re|im)$")
_INPUT_PROJ_RE = re.compile(r"\.mattn\.input_proj\.(fc_r|fc_i)$|\.mattn\.input_proj$")
_OUTPUT_PROJ_RE = re.compile(r"\.mattn\.output_proj\.(fc_r|fc_i)$|\.mattn\.output_proj$")


def _is_depthwise_conv(m: nn.Module) -> bool:
    return isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)) and m.groups == m.in_channels \
        and m.in_channels > 1


def _is_pointwise_conv(m: nn.Module) -> bool:
    if not isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        return False
    k = m.kernel_size
    if isinstance(k, (tuple, list)):
        return all(x == 1 for x in k)
    return k == 1


DEFAULT_CONFIG: Dict[str, Any] = {
    "encoder_dw": {"weight_bits": 4, "weight_scale": "per_channel", "scale_format": "fp32"},
    "encoder_pw": {"weight_bits": 4, "weight_scale": "per_group", "group_size": 256, "scale_format": "fp32"},
    "decoder_dw": {"weight_bits": 4, "weight_scale": "per_channel", "scale_format": "fp32"},
    "decoder_pw_outlier": {"weight_bits": 8, "weight_format": "fp8_e4m3"},  # stage 0,3,4
    "decoder_pw_normal":  {"weight_bits": 4, "weight_scale": "per_group", "group_size": 256, "scale_format": "fp32"},
    "mha":                {"weight_bits": 4, "weight_scale": "per_head", "scale_format": "fp32"},
}

DECODER_OUTLIER_STAGES = {0, 3, 4}  # per handoff §2.1


def _weight_param_for_conv(full_name: str, module: nn.Module, config: Dict[str, Any]) -> Optional[nn.Module]:
    """Choose the right weight fake-quant for a Conv2d/ConvTranspose2d."""
    m_enc = _ENCODER_RE.search("." + full_name)
    m_dec = _DECODER_RE.search("." + full_name)

    if m_enc:
        sub_cfg = config["encoder_dw"] if _is_depthwise_conv(module) else config["encoder_pw"]
    elif m_dec:
        stage_idx = int(m_dec.group(1))
        if _is_depthwise_conv(module):
            sub_cfg = config["decoder_dw"]
        elif stage_idx in DECODER_OUTLIER_STAGES:
            sub_cfg = config["decoder_pw_outlier"]
        else:
            sub_cfg = config["decoder_pw_normal"]
    else:
        # Not an encoder/decoder layer we routed explicitly — default INT4 per-channel
        sub_cfg = {"weight_bits": 4, "weight_scale": "per_channel"}

    bits = sub_cfg.get("weight_bits", 4)
    if bits == 8:
        fmt = sub_cfg.get("weight_format", "fp8_e4m3")
        dtype_str = "e4m3" if fmt == "fp8_e4m3" else "e5m2"
        return FP8Parametrization(dtype_str)
    if bits == 4:
        mode = sub_cfg.get("weight_scale", "per_channel")
        scale_format = sub_cfg.get("scale_format", "fp32")
        if mode == "per_channel":
            return Int4PerChannelParametrization(num_bits=4, scale_format=scale_format)
        if mode == "per_group":
            gsz = sub_cfg.get("group_size", 256)
            return Int4PerGroupParametrization(
                group_size=gsz, num_bits=4, scale_format=scale_format
            )
    return None


# ──────────────────────────────────────────────────────────────
# Top-k wiring
# ──────────────────────────────────────────────────────────────

def _set_topk_on_complex_mha(model: nn.Module, top_k: Optional[int], fp8_dtype,
                             fp16_accumulator: bool = False,
                             attention_datapath: Optional[Dict[str, Any]] = None) -> int:
    """Walk model and wire top-k + activation quantizer on every
    ComplexMultiheadAttention so it uses the custom attention forward.

    fp16_accumulator toggles fp16 QK^T / softmax / Score×V inside the custom
    forward, matching the hardware datapath (07 handoff §2.1).
    """
    from DCNN.utils.complexPyTorch.complexLayers import ComplexMultiheadAttention

    def _act_q(t):
        return _fp8_quantize_tensor(t, fp8_dtype)

    n = 0
    datapath_cfg = dict(attention_datapath or {})
    datapath_enabled = bool(datapath_cfg.get("enabled", False))
    for m in model.modules():
        if isinstance(m, ComplexMultiheadAttention):
            softmax_mode = getattr(m, "softmax_mode", "softmax")
            # Enable the custom forward whenever top-k, fp16 accumulator, or an
            # approximate softmax is requested, since all bypass native MHA.
            enable_custom = (
                (top_k is not None)
                or fp16_accumulator
                or datapath_enabled
                or getattr(m, "projection_grouped_active", False)
                or softmax_mode != "softmax"
            )
            m.top_k = top_k
            m.use_custom_forward = enable_custom
            m.activation_quantizer = _act_q if enable_custom else None
            m.fp16_accumulator = fp16_accumulator
            m.attention_datapath = dict(datapath_cfg)
            n += 1
    return n


# ──────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────

def enable_mixed_precision_qat(
    model: nn.Module,
    config: Optional[Dict[str, Any]] = None,
    activation_format: str = "fp8_e4m3",
    top_k: Optional[int] = 4,
    fp16_accumulator: bool = False,
    attention_datapath: Optional[Dict[str, Any]] = None,
) -> nn.Module:
    """Enable W4A8 mixed-precision fake quantization on `model` in-place.

    Args:
        model:              nn.Module to modify.
        config:             Per-layer precision config. Defaults to DEFAULT_CONFIG.
        activation_format:  "fp8_e4m3" or "fp8_e5m2" for activation fake quant.
        top_k:              Set to 4 to enable top-k attention; None to disable.
        fp16_accumulator:   Force QK^T / softmax / Score×V into fp16 inside the
                            custom MHA forward (hardware datapath match).
        attention_datapath: Optional P3 attention score/probability datapath
                            simulation settings.

    Returns:
        Same model reference (modified).
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    fp8_dtype = torch.float8_e4m3fn if activation_format.lower() in ("fp8_e4m3", "e4m3") \
        else torch.float8_e5m2

    print("🔥 Enabling Mixed-Precision W4A8 QAT ...")
    n_conv = n_lin = n_mha = 0

    # Track FP8-preserved decoder names to report at the end
    fp8_preserved: List[str] = []

    for name, module in model.named_modules():
        if module.__class__.__name__ == "ComplexDepthwiseSeparableConvTranspose2dSplit":
            module.register_forward_pre_hook(_make_split_pointwise_activation_hook(fp8_dtype))

        # --- Conv2d / ConvTranspose2d ---
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            param = _weight_param_for_conv(name, module, cfg)
            if param is not None and hasattr(module, "weight") and module.weight is not None:
                register_parametrization(module, "weight", param)
                if isinstance(param, FP8Parametrization):
                    fp8_preserved.append(name)
            if not _DECODER_SPLIT_POINTWISE_RE.search("." + name):
                module.register_forward_pre_hook(_make_fp8_activation_hook(fp8_dtype))
            n_conv += 1

        # --- Linear ---
        elif isinstance(module, nn.Linear):
            w = module.weight
            if w is not None:
                if w.is_complex():
                    # complex Linear (input_proj / output_proj in MultiAttnBlock)
                    # → quantize real & imag parts; per-head along first dim divisible by num_heads
                    bits = cfg["mha"].get("weight_bits", 4)
                    scale_format = cfg["mha"].get("scale_format", "fp32")
                    if bits == 4:
                        num_heads = cfg["mha"].get("num_heads", 16)
                        register_parametrization(module, "weight",
                                                 ComplexInt4PerHeadParametrization(
                                                     num_heads,
                                                     scale_format=scale_format,
                                                 ))
                    else:
                        register_parametrization(module, "weight",
                                                 FP8ComplexParametrization("e4m3"))
                else:
                    # real Linear — typically MHA.out_proj (rows = E)
                    num_heads = cfg["mha"].get("num_heads", 16)
                    scale_format = cfg["mha"].get("scale_format", "fp32")
                    register_parametrization(module, "weight",
                                             Int4PerHeadLinearParametrization(
                                                 num_heads, 4, scale_format=scale_format
                                             ))
            module.register_forward_pre_hook(_make_fp8_activation_hook(fp8_dtype))
            n_lin += 1

        # --- MultiheadAttention (native nn.MultiheadAttention inside ComplexMHA) ---
        elif isinstance(module, nn.MultiheadAttention):
            num_heads = cfg["mha"].get("num_heads", module.num_heads)
            scale_format = cfg["mha"].get("scale_format", "fp32")
            if hasattr(module, "in_proj_weight") and module.in_proj_weight is not None:
                register_parametrization(module, "in_proj_weight",
                                         Int4PerHeadMHAParametrization(
                                             num_heads=num_heads,
                                             scale_format=scale_format,
                                         ))
            # out_proj is an nn.Linear sub-module — the Linear branch above will catch it.
            module.register_forward_pre_hook(_make_fp8_activation_hook(fp8_dtype))
            n_mha += 1

    n_topk = _set_topk_on_complex_mha(
        model, top_k, fp8_dtype, fp16_accumulator, attention_datapath
    )

    print(f"✅ Mixed-Precision QAT enabled: "
          f"{n_conv} Conv, {n_lin} Linear, {n_mha} MHA, top_k={top_k}, "
          f"fp16_accum={fp16_accumulator}, attention_datapath={attention_datapath or {}} "
          f"on {n_topk} ComplexMHA.")
    if fp8_preserved:
        print(f"   FP8 preserved ({len(fp8_preserved)} layers): {fp8_preserved}")
    # Expose the FP8-preserved (decoder_pw_outlier) layer names so a later
    # selective patch (enable_outlier_fp8_lmul) can L-Mul ONLY these, without
    # touching MHA / INT4 conv.
    model._fp8_preserved_names = list(fp8_preserved)
    return model


# ──────────────────────────────────────────────────────────────
# Complex Linear per-head INT4
# ──────────────────────────────────────────────────────────────

class ComplexInt4PerHeadParametrization(nn.Module):
    """Per-head INT4 fake quant applied independently to real and imag parts
    of a complex-valued Linear weight. Grouping logic matches
    Int4PerHeadLinearParametrization.
    """

    def __init__(self, num_heads: int, num_bits: int = 4,
                 scale_format: str = "fp32"):
        super().__init__()
        self.real_q = Int4PerHeadLinearParametrization(
            num_heads, num_bits, scale_format=scale_format
        )
        self.imag_q = Int4PerHeadLinearParametrization(
            num_heads, num_bits, scale_format=scale_format
        )

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        rq = self.real_q(w.real)
        iq = self.imag_q(w.imag)
        return torch.complex(rq, iq)
