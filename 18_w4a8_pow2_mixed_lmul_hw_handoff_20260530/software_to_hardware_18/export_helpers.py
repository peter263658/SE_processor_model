from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


INT4_QMAX = 7
SCALE_EPS = 1e-12
DECODER_OUTLIER_STAGES = {0, 3, 4}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DEFAULT_CODE_DIR = os.path.join(PACKAGE_ROOT, "code")
DEFAULT_CKPT = os.path.join(PACKAGE_ROOT, "ckpt", "strict_core_pow2_blockfloat_deploy_ep3_vctk0.8357_timit0.7898.ckpt")
DEFAULT_CONFIG = os.path.join(PACKAGE_ROOT, "ckpt", "strict_core_deploy_config.yaml")
DEFAULT_SKIP_INDICES = os.path.join(PACKAGE_ROOT, "ckpt", "p2_skip_indices_g42_topk_l1_r050.pt")
# Block-float activation: e4m3 raw code + per-tensor shared exponent (pow2 scale).
# The calibration table holds the per-boundary amax; the deployed scale is the
# FLOOR pow2 of 448/amax (matches DCNN.utils.fp8_qat._pow2_round default='floor').
DEFAULT_ACT_CALIB = os.path.join(PACKAGE_ROOT, "ckpt", "activation_calibration_p999_b64.json")


def ensure(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_json(path: str, obj: Dict[str, Any]) -> None:
    ensure(os.path.dirname(os.path.abspath(path)) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def _sidecar_base(path: str) -> str:
    return path[:-4] if path.endswith(".bin") else path


def write_bin(path: str, arr: np.ndarray) -> None:
    ensure(os.path.dirname(os.path.abspath(path)) or ".")
    a = np.asarray(arr)
    a.tofile(path)
    write_sidecars(path, a)


def write_sidecars(path: str, arr: np.ndarray) -> None:
    a = np.asarray(arr)
    flat = a.reshape(-1)
    base = _sidecar_base(path)

    if a.dtype == np.float16:
        words = flat.view(np.uint16)
        fmt = "{:04X}"
    elif a.dtype == np.float32:
        words = flat.view(np.uint32)
        fmt = "{:08X}"
    elif a.dtype == np.int8:
        words = flat.view(np.uint8)
        fmt = "{:02X}"
    elif a.dtype == np.uint8:
        words = flat
        fmt = "{:02X}"
    elif a.dtype == np.int16:
        words = flat.view(np.uint16)
        fmt = "{:04X}"
    elif a.dtype == np.uint16:
        words = flat
        fmt = "{:04X}"
    elif a.dtype == np.int32:
        words = flat.view(np.uint32)
        fmt = "{:08X}"
    elif a.dtype == np.uint32:
        words = flat
        fmt = "{:08X}"
    else:
        words = a.view(np.uint8).reshape(-1)
        fmt = "{:02X}"

    with open(base + ".hex", "w", encoding="utf-8") as f:
        for v in words:
            f.write(fmt.format(int(v)) + "\n")

    with open(base + ".dec", "w", encoding="utf-8") as f:
        if np.issubdtype(a.dtype, np.floating):
            for v in flat.astype(np.float64):
                f.write(f"{float(v):.10e}\n")
        else:
            for v in flat:
                f.write(f"{int(v)}\n")


def write_tensor(path: str, tensor: torch.Tensor, dtype: torch.dtype) -> None:
    t = tensor.detach().cpu().contiguous()
    if dtype == torch.float16:
        write_bin(path, t.to(torch.float16).numpy())
    elif dtype == torch.float32:
        write_bin(path, t.to(torch.float32).numpy())
    elif dtype == torch.int16:
        write_bin(path, t.to(torch.int16).numpy())
    elif dtype == torch.int8:
        write_bin(path, t.to(torch.int8).numpy())
    elif dtype == torch.uint8:
        write_bin(path, t.to(torch.uint8).numpy())
    else:
        raise ValueError(f"unsupported tensor export dtype {dtype}")


def pack_int4_nibbles(q_int8: torch.Tensor) -> np.ndarray:
    flat = q_int8.detach().cpu().flatten().to(torch.int8).numpy()
    if flat.size % 2 == 1:
        flat = np.concatenate([flat, np.zeros(1, dtype=np.int8)])
    lo = (flat[0::2] & 0x0F).astype(np.uint8)
    hi = (flat[1::2] & 0x0F).astype(np.uint8)
    return (lo | (hi << 4)).astype(np.uint8)


def raw_weight(module: nn.Module) -> torch.Tensor:
    if hasattr(module, "parametrizations") and "weight" in module.parametrizations:
        return module.parametrizations["weight"].original.detach()
    return module.weight.detach()


def raw_in_proj_weight(mha: nn.MultiheadAttention) -> torch.Tensor:
    if hasattr(mha, "parametrizations") and "in_proj_weight" in mha.parametrizations:
        return mha.parametrizations["in_proj_weight"].original.detach()
    return mha.in_proj_weight.detach()


def fp8_e4m3_bytes(w: torch.Tensor) -> np.ndarray:
    return w.detach().cpu().to(torch.float8_e4m3fn).view(torch.uint8).numpy()


def scaled_fp8_encode(w: torch.Tensor, dtype: torch.dtype = torch.float8_e4m3fn) -> Dict[str, Any]:
    """Replicate the source FP8Parametrization (DCNN/utils/fp8_qat.py FP8QuantizeSTE):
    a PER-TENSOR amax-scaled FP8, NOT a plain ``w.to(float8_e4m3fn)`` cast.

        max_val = 448.0 (e4m3) | 57344.0 (e5m2)
        scale   = max_val / amax(|w|)
        code    = fp8_dtype(w * scale)            # the e4m3 bits actually stored
        weight  = code / scale                    # == module.weight (bit-exact)

    The stored e4m3 ``code`` uses the full e4m3 range; the per-tensor ``descale``
    (= amax/max_val = 1/scale) is a single constant that the chip applies once per
    output on the fp16 accumulator (see HANDOFF §3.4 for the zero-multiplier path).
    """
    w = w.detach().to(torch.float32).cpu()
    max_val = 448.0 if dtype == torch.float8_e4m3fn else 57344.0
    amax = w.abs().max().clamp(min=1e-12)
    scale = max_val / amax                       # per-tensor scalar, == source scale
    code = (w * scale).to(dtype).contiguous()    # e4m3 tensor (stored bits)
    code_val = code.to(torch.float32)
    descale = 1.0 / scale                         # == amax / max_val
    recon = code_val / scale                      # == module.weight, bit-exact
    return {
        "code": code,
        "code_val": code_val,
        "scale": scale.reshape(1),
        "descale": descale.reshape(1),
        "amax": amax.reshape(1),
        "max_val": float(max_val),
        "recon": recon,
    }


def apot_encode(
    values: torch.Tensor,
    min_exp: int,
    max_exp: int,
    signed: bool,
) -> Dict[str, torch.Tensor]:
    x = values.detach().to(torch.float32).cpu()
    overall_sign = torch.sign(x) if signed else torch.ones_like(x)
    mag = x.abs() if signed else x.clamp_min(0)
    is_zero = mag == 0
    safe = mag.clamp_min(float(2.0 ** int(min_exp)))

    a_exp = torch.round(torch.log2(safe)).clamp(int(min_exp), int(max_exp)).to(torch.int16)
    primary = torch.pow(torch.full_like(safe, 2.0), a_exp.to(torch.float32))
    residual = safe - primary
    abs_residual = residual.abs()
    b_exp = torch.round(torch.log2(abs_residual.clamp_min(float(2.0 ** int(min_exp)))))
    b_exp = b_exp.clamp(int(min_exp), int(max_exp)).to(torch.int16)
    has_second = (abs_residual > SCALE_EPS) & (b_exp.to(torch.int32) < a_exp.to(torch.int32))
    residual_sign = torch.where(has_second, torch.sign(residual), torch.zeros_like(residual))
    second = torch.where(
        has_second,
        residual_sign * torch.pow(torch.full_like(safe, 2.0), b_exp.to(torch.float32)),
        torch.zeros_like(safe),
    )
    quant_abs = primary + second
    quant = overall_sign * quant_abs
    quant = torch.where(is_zero, torch.zeros_like(quant), quant)
    overall_sign = torch.where(is_zero, torch.zeros_like(overall_sign), overall_sign)

    return {
        "quantized": quant,
        "a_exp": a_exp,
        "b_exp": b_exp,
        "has_second": has_second.to(torch.uint8),
        "residual_sign": residual_sign.to(torch.int8),
        "overall_sign": overall_sign.to(torch.int8),
    }


def pow2_encode(values: torch.Tensor, min_exp: int, max_exp: int) -> Dict[str, torch.Tensor]:
    x = values.detach().to(torch.float32).cpu()
    sign = torch.sign(x)
    mag = x.abs()
    is_zero = mag == 0
    safe = mag.clamp_min(float(2.0 ** int(min_exp)))
    exp = torch.round(torch.log2(safe)).clamp(int(min_exp), int(max_exp)).to(torch.int16)
    quant = sign * torch.pow(torch.full_like(safe, 2.0), exp.to(torch.float32))
    quant = torch.where(is_zero, torch.zeros_like(quant), quant)
    sign = torch.where(is_zero, torch.zeros_like(sign), sign)
    return {"quantized": quant, "exp": exp, "sign": sign.to(torch.int8)}


def blockfloat_activation_table(calib_path: str, max_val: float = 448.0) -> Dict[str, Any]:
    """Per activation-boundary block-float exponent table for the pow2 deploy.

    For each calibration boundary key, the deployed activation scale is the FLOOR
    power of two of (max_val / calib_amax) -- EXACTLY matching the runtime
    ``DCNN.utils.fp8_qat._pow2_round`` (default mode='floor'). The activation is
    stored as raw e4m3 code = e4m3(x * 2^k); its descale is the pure exponent
    shift 2^-k. Hardware folds 2^-k into the layer's post-accumulation weight
    descale by adding (-k) to the weight APoT exponents -- a shift, no multiply.
    """
    with open(calib_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    keys = data.get("keys", data)
    table: Dict[str, Any] = {}
    for key, entry in keys.items():
        calib_amax = float(entry["calib_amax"])
        scale = max_val / max(calib_amax, 1e-30)
        k = int(math.floor(math.log2(scale)))          # floor -> matches runtime
        table[key] = {
            "calib_amax": calib_amax,
            "scale_general_float": scale,               # 448/amax (pre-pow2, debug)
            "scale_pow2_exp": k,                         # activation stored = e4m3(x * 2^k)
            "scale_pow2": float(2.0 ** k),
            "descale_pow2_exp": -k,                      # post-acc descale = 2^-k
            "descale_pow2": float(2.0 ** (-k)),
            "scaled_amax": calib_amax * (2.0 ** k),      # <= max_val (no overflow)
        }
    return table


def _scale_format_is_apot(scale_format: str) -> bool:
    return str(scale_format or "").lower() in {
        "apot", "apot2", "apot_2", "two_pow2", "two_term_pow2", "apot_clip", "apot_clipped"
    }


def quantize_scale_for_int4(
    raw_scale: torch.Tensor,
    scale_format: str,
    min_exp: int = -16,
    max_exp: int = -1,
) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
    fmt = str(scale_format or "fp32").lower()
    if fmt in {"fp32", "fp16", "none", "off"}:
        return raw_scale, None
    if fmt in {"pow2", "power_of_two"}:
        enc = pow2_encode(raw_scale, min_exp, max_exp)
        return enc["quantized"].clamp_min(SCALE_EPS), {
            "quantized": enc["quantized"],
            "a_exp": enc["exp"],
            "b_exp": enc["exp"],
            "has_second": torch.zeros_like(enc["sign"], dtype=torch.uint8),
            "residual_sign": torch.zeros_like(enc["sign"], dtype=torch.int8),
            "overall_sign": torch.ones_like(enc["sign"], dtype=torch.int8),
        }
    if _scale_format_is_apot(fmt):
        enc = apot_encode(raw_scale, min_exp, max_exp, signed=False)
        return enc["quantized"].clamp_min(SCALE_EPS), enc
    raise ValueError(f"unsupported int4 scale_format={scale_format}")


def int4_per_channel(
    w: torch.Tensor,
    scale_format: str = "apot_clip",
    min_exp: int = -16,
    max_exp: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[Dict[str, torch.Tensor]], str]:
    shape = [w.shape[0]] + [1] * (w.dim() - 1)
    raw_scale = (w.abs().flatten(1).amax(dim=1).view(shape) / INT4_QMAX).clamp_min(SCALE_EPS)
    scale_q, enc = quantize_scale_for_int4(raw_scale, scale_format, min_exp, max_exp)
    q = torch.round(w / scale_q).clamp(-INT4_QMAX, INT4_QMAX).to(torch.int8)
    return q, raw_scale, scale_q, enc, "per_channel"


def int4_per_group_input_dim(
    w: torch.Tensor,
    group_size: int,
    scale_format: str = "apot_clip",
    min_exp: int = -16,
    max_exp: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[Dict[str, torch.Tensor]], str]:
    if w.dim() == 4:
        out_ch, in_ch, k_h, k_w = w.shape
        if in_ch < group_size or in_ch % group_size != 0:
            return int4_per_channel(w, scale_format, min_exp, max_exp)
        grouped_shape = (out_ch, in_ch // group_size, group_size, k_h, k_w)
        grouped = w.reshape(grouped_shape)
        raw_scale = (grouped.abs().amax(dim=(2, 3, 4), keepdim=True) / INT4_QMAX).clamp_min(SCALE_EPS)
        scale_q, enc = quantize_scale_for_int4(raw_scale, scale_format, min_exp, max_exp)
        q = torch.round(grouped / scale_q).clamp(-INT4_QMAX, INT4_QMAX).to(torch.int8)
        return q.reshape_as(w), raw_scale, scale_q, enc, "per_group"

    last = w.shape[-1]
    if last < group_size or last % group_size != 0:
        return int4_per_channel(w, scale_format, min_exp, max_exp)
    grouped_shape = w.shape[:-1] + (last // group_size, group_size)
    grouped = w.reshape(grouped_shape)
    raw_scale = (grouped.abs().amax(dim=-1, keepdim=True) / INT4_QMAX).clamp_min(SCALE_EPS)
    scale_q, enc = quantize_scale_for_int4(raw_scale, scale_format, min_exp, max_exp)
    q = torch.round(grouped / scale_q).clamp(-INT4_QMAX, INT4_QMAX).to(torch.int8)
    return q.reshape_as(w), raw_scale, scale_q, enc, "per_group"


def int4_per_head_linear(
    w: torch.Tensor,
    num_heads: int,
    scale_format: str = "apot_clip",
    min_exp: int = -16,
    max_exp: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[Dict[str, torch.Tensor]], str]:
    rows, cols = w.shape
    if rows % num_heads == 0:
        head_dim = rows // num_heads
        grouped = w.reshape(num_heads, head_dim, cols)
        raw_scale = (grouped.abs().flatten(1).amax(dim=1).view(num_heads, 1, 1) / INT4_QMAX).clamp_min(SCALE_EPS)
        scale_q, enc = quantize_scale_for_int4(raw_scale, scale_format, min_exp, max_exp)
        q = torch.round(grouped / scale_q).clamp(-INT4_QMAX, INT4_QMAX).to(torch.int8)
        return q.reshape_as(w), raw_scale, scale_q, enc, "per_head_rows"
    if cols % num_heads == 0:
        head_dim = cols // num_heads
        grouped = w.reshape(rows, num_heads, head_dim)
        raw_scale = (grouped.abs().amax(dim=-1, keepdim=True) / INT4_QMAX).clamp_min(SCALE_EPS)
        scale_q, enc = quantize_scale_for_int4(raw_scale, scale_format, min_exp, max_exp)
        q = torch.round(grouped / scale_q).clamp(-INT4_QMAX, INT4_QMAX).to(torch.int8)
        return q.reshape_as(w), raw_scale, scale_q, enc, "per_head_cols"
    return int4_per_channel(w, scale_format, min_exp, max_exp)


def int4_per_head_mha(
    w: torch.Tensor,
    num_heads: int,
    scale_format: str = "apot_clip",
    min_exp: int = -16,
    max_exp: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
    rows, cols = w.shape
    total = 3 * num_heads
    if rows % total != 0:
        raise ValueError(f"in_proj rows={rows} not divisible by 3*num_heads={total}")
    head_dim = rows // total
    grouped = w.reshape(total, head_dim, cols)
    raw_scale = (grouped.abs().flatten(1).amax(dim=1).view(total, 1, 1) / INT4_QMAX).clamp_min(SCALE_EPS)
    scale_q, enc = quantize_scale_for_int4(raw_scale, scale_format, min_exp, max_exp)
    q = torch.round(grouped / scale_q).clamp(-INT4_QMAX, INT4_QMAX).to(torch.int8)
    return q.reshape_as(w), raw_scale, scale_q, enc


def linear_group_mask(weight_shape: Iterable[int], groups: int) -> Optional[torch.Tensor]:
    rows, cols = [int(x) for x in weight_shape]
    groups = int(groups or 1)
    if groups <= 1:
        return None
    if rows % groups != 0 or cols % groups != 0:
        raise ValueError(f"invalid linear groups={groups} for shape={(rows, cols)}")
    out_per = rows // groups
    in_per = cols // groups
    mask = torch.zeros((rows, cols), dtype=torch.int8)
    for g in range(groups):
        mask[g * out_per:(g + 1) * out_per, g * in_per:(g + 1) * in_per] = 1
    return mask


def mha_in_proj_group_mask(weight_shape: Iterable[int], embed_dim: int, groups: int) -> Optional[torch.Tensor]:
    groups = int(groups or 1)
    if groups <= 1:
        return None
    rows, cols = [int(x) for x in weight_shape]
    if rows != 3 * embed_dim or cols != embed_dim:
        raise ValueError(f"unexpected MHA in_proj shape {(rows, cols)} for embed_dim={embed_dim}")
    base = linear_group_mask((embed_dim, embed_dim), groups)
    return torch.cat([base, base, base], dim=0)


def _child(module: nn.Module, key: str) -> Optional[nn.Module]:
    return module._modules.get(key)


@dataclass
class BlockInfo:
    kind: str
    stage_idx: int
    dotted: str
    depthwise_r: Optional[nn.Module]
    depthwise_i: Optional[nn.Module]
    pointwise_items: List[Tuple[str, nn.Module]]
    bn_r: Optional[nn.Module]
    bn_i: Optional[nn.Module]
    prelu_r: Optional[nn.Module]
    prelu_i: Optional[nn.Module]
    num_out_channels: int
    is_outlier_fp8: bool


def discover_blocks(model: nn.Module) -> List[BlockInfo]:
    blocks: List[BlockInfo] = []
    enc = model.encoder
    dec = model.decoder

    for idx, block in enumerate(enc.model):
        conv = _child(block, "0")
        bn = _child(block, "1")
        pr = _child(block, "2")
        blocks.append(BlockInfo(
            kind="encoder",
            stage_idx=idx,
            dotted=f"encoder.model.{idx}.0",
            depthwise_r=conv.depthwise_r,
            depthwise_i=conv.depthwise_i,
            pointwise_items=[
                ("pointwise_r", conv.pointwise_r),
                ("pointwise_i", conv.pointwise_i),
            ],
            bn_r=bn.bn_r if bn is not None else None,
            bn_i=bn.bn_i if bn is not None else None,
            prelu_r=pr.r_prelu if pr is not None else None,
            prelu_i=pr.i_prelu if pr is not None else None,
            num_out_channels=conv.pointwise_r.out_channels,
            is_outlier_fp8=False,
        ))

    for idx, block in enumerate(dec.model):
        conv = _child(block, "0")
        bn = _child(block, "1")
        pr = _child(block, "2")
        if hasattr(conv, "pointwise_r_x"):
            pointwise_items = [
                ("pointwise_r_x", conv.pointwise_r_x),
                ("pointwise_i_x", conv.pointwise_i_x),
                ("pointwise_r_skip", conv.pointwise_r_skip),
                ("pointwise_i_skip", conv.pointwise_i_skip),
            ]
        else:
            pointwise_items = [
                ("pointwise_r", conv.pointwise_r),
                ("pointwise_i", conv.pointwise_i),
            ]
        blocks.append(BlockInfo(
            kind="decoder",
            stage_idx=idx,
            dotted=f"decoder.model.{idx}.0",
            depthwise_r=conv.depthwise_r,
            depthwise_i=conv.depthwise_i,
            pointwise_items=pointwise_items,
            bn_r=bn.bn_r if bn is not None else None,
            bn_i=bn.bn_i if bn is not None else None,
            prelu_r=pr.r_prelu if pr is not None else None,
            prelu_i=pr.i_prelu if pr is not None else None,
            num_out_channels=conv.depthwise_r.out_channels,
            is_outlier_fp8=idx in DECODER_OUTLIER_STAGES,
        ))
    return blocks


@dataclass
class BnAffine:
    alpha_r: torch.Tensor
    beta_r: torch.Tensor
    alpha_i: torch.Tensor
    beta_i: torch.Tensor
    alpha_r_code: Dict[str, torch.Tensor]
    alpha_i_code: Dict[str, torch.Tensor]


def _conv_effective_bias(
    bias_r: Optional[torch.Tensor],
    bias_i: Optional[torch.Tensor],
    num_channels: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = bias_r.device if bias_r is not None else torch.device("cpu")
    zeros = torch.zeros(num_channels, device=device)
    b_r = bias_r.detach() if bias_r is not None else zeros
    b_i = bias_i.detach() if bias_i is not None else zeros
    return b_r - b_i, b_r + b_i


def folded_bn_affine(
    block: BlockInfo,
    bn_min_exp: int = -9,
    bn_max_exp: int = 4,
    fuse_conv_bias: bool = True,
) -> BnAffine:
    if block.kind == "encoder":
        # BN follows encoder pointwise complex conv.
        pr = dict(block.pointwise_items)
        bias_r = getattr(pr["pointwise_r"], "bias", None)
        bias_i = getattr(pr["pointwise_i"], "bias", None)
    else:
        # BN follows decoder depthwise transpose complex conv.
        bias_r = getattr(block.depthwise_r, "bias", None)
        bias_i = getattr(block.depthwise_i, "bias", None)

    eff_r, eff_i = _conv_effective_bias(bias_r, bias_i, block.num_out_channels)

    if block.bn_r is None:
        ones = torch.ones(block.num_out_channels)
        zeros = torch.zeros(block.num_out_channels)
        beta_r = eff_r.cpu() if fuse_conv_bias else zeros
        beta_i = eff_i.cpu() if fuse_conv_bias else zeros
        return BnAffine(
            ones, beta_r, ones.clone(), beta_i,
            apot_encode(ones, bn_min_exp, bn_max_exp, signed=True),
            apot_encode(ones, bn_min_exp, bn_max_exp, signed=True),
        )

    def one_side(bn: nn.BatchNorm2d, eff_bias: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        gamma = bn.weight.detach().cpu()
        beta = bn.bias.detach().cpu()
        mean = bn.running_mean.detach().cpu()
        var = bn.running_var.detach().cpu()
        alpha = gamma * torch.rsqrt(var + float(bn.eps))
        code = apot_encode(alpha, bn_min_exp, bn_max_exp, signed=True)
        alpha_q = code["quantized"]
        folded_bias = beta - alpha_q * mean
        if fuse_conv_bias:
            folded_bias = folded_bias + alpha_q * eff_bias.detach().cpu()
        return alpha_q, folded_bias, code

    ar, br, cr = one_side(block.bn_r, eff_r)
    ai, bi, ci = one_side(block.bn_i, eff_i)
    return BnAffine(ar, br, ai, bi, cr, ci)


def strip_state_dict_for_raw_model(ckpt_path: str) -> Dict[str, torch.Tensor]:
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj
    out: Dict[str, torch.Tensor] = {}
    for key, value in sd.items():
        if key.startswith("loss."):
            continue
        if key.startswith("model."):
            key = key[len("model."):]
        out[key] = value
    return out


def load_config(config_path: str, skip_indices_path: Optional[str] = None):
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(config_path)
    if skip_indices_path:
        cfg.model.decoder_grouped_pointwise.skip_indices_path = os.path.abspath(skip_indices_path)
    elif os.path.exists(DEFAULT_SKIP_INDICES):
        cfg.model.decoder_grouped_pointwise.skip_indices_path = os.path.abspath(DEFAULT_SKIP_INDICES)
    return cfg


def build_model(
    config_path: str = DEFAULT_CONFIG,
    ckpt_path: Optional[str] = DEFAULT_CKPT,
    code_dir: str = DEFAULT_CODE_DIR,
    skip_indices_path: Optional[str] = DEFAULT_SKIP_INDICES,
) -> Tuple[nn.Module, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)

    os.environ.setdefault("BCCTN_APOT_A_MIN", "-16")
    os.environ.setdefault("BCCTN_APOT_A_MAX", "-1")
    os.environ.setdefault("BCCTN_APOT_B_MIN", "-16")
    os.environ.setdefault("BCCTN_APOT_B_MAX", "-1")

    from omegaconf import OmegaConf
    from DCNN.models.binaural_attention_model import BinauralAttentionDCNN
    from DCNN.utils.mixed_precision_qat import enable_mixed_precision_qat
    from DCNN.utils.all_lmul_multiply import enable_outlier_fp8_lmul

    cfg = load_config(config_path, skip_indices_path)
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    training_cfg = OmegaConf.to_container(cfg.training, resolve=True)

    model = BinauralAttentionDCNN(**model_cfg).eval()
    num_heads = int(model_cfg.get("num_heads", 16))
    group_size = int(training_cfg.get("int4_group_size", 32))
    scale_format = training_cfg.get("int4_scale_format", "apot_clip")
    mp_cfg = {
        "encoder_dw": {"weight_bits": 4, "weight_scale": "per_channel", "scale_format": scale_format},
        "encoder_pw": {"weight_bits": 4, "weight_scale": "per_group", "group_size": group_size, "scale_format": scale_format},
        "decoder_dw": {"weight_bits": 4, "weight_scale": "per_channel", "scale_format": scale_format},
        "decoder_pw_outlier": {"weight_bits": 8, "weight_format": "fp8_e4m3"},
        "decoder_pw_normal": {"weight_bits": 4, "weight_scale": "per_group", "group_size": group_size, "scale_format": scale_format},
        "mha": {"weight_bits": 4, "weight_scale": "per_head", "num_heads": num_heads, "scale_format": scale_format},
    }
    if training_cfg.get("qat_scope", "full") == "mha_only":
        fp8_entry = {"weight_bits": 8, "weight_format": "fp8_e4m3"}
        for k in ("encoder_dw", "encoder_pw", "decoder_dw", "decoder_pw_outlier", "decoder_pw_normal"):
            mp_cfg[k] = fp8_entry

    model = enable_mixed_precision_qat(
        model,
        config=mp_cfg,
        activation_format=training_cfg.get("fp8_dtype", "e4m3"),
        top_k=training_cfg.get("top_k_attention", 4),
        fp16_accumulator=training_cfg.get("fp16_accumulator", False),
        attention_datapath=model_cfg.get("attention_datapath", {}),
    )

    outlier_cfg = model_cfg.get("low_power", {}).get("outlier_fp8_lmul", {}) or {}
    if outlier_cfg.get("enabled", False):
        model = enable_outlier_fp8_lmul(
            model,
            mantissa_bits=outlier_cfg.get("mantissa_bits", 4),
            bit_exact=outlier_cfg.get("bit_exact", True),
        )

    load_report: Dict[str, Any] = {"ckpt": ckpt_path, "missing": [], "unexpected": []}
    if ckpt_path:
        sd = strip_state_dict_for_raw_model(ckpt_path)
        result = model.load_state_dict(sd, strict=False)
        load_report = {
            "ckpt": os.path.abspath(ckpt_path),
            "missing": list(result.missing_keys),
            "unexpected": list(result.unexpected_keys),
        }
    model.eval()
    return model, model_cfg, training_cfg, load_report
