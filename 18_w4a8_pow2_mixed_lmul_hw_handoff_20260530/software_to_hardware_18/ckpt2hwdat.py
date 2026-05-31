from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from typing import Any, Dict, Optional

import numpy as np
import torch

import export_helpers as eh


def _write_scale(dir_path: str, raw_scale: torch.Tensor, scale_q: torch.Tensor,
                 enc: Optional[Dict[str, torch.Tensor]], signed: bool = False) -> Dict[str, Any]:
    eh.write_tensor(os.path.join(dir_path, "scale_raw.fp32.bin"), raw_scale, torch.float32)
    eh.write_tensor(os.path.join(dir_path, "scale_quant.fp32.bin"), scale_q, torch.float32)
    eh.write_tensor(os.path.join(dir_path, "scale_quant.fp16.bin"), scale_q, torch.float16)
    meta = {
        "raw_scale_shape": list(raw_scale.shape),
        "quant_scale_shape": list(scale_q.shape),
        "quant_scale_debug_files": [
            "scale_raw.fp32.bin",
            "scale_quant.fp32.bin",
            "scale_quant.fp16.bin",
        ],
    }
    if enc is not None:
        eh.write_tensor(os.path.join(dir_path, "scale_apot_a_exp.int16.bin"), enc["a_exp"], torch.int16)
        eh.write_tensor(os.path.join(dir_path, "scale_apot_b_exp.int16.bin"), enc["b_exp"], torch.int16)
        eh.write_tensor(os.path.join(dir_path, "scale_apot_has_second.u8.bin"), enc["has_second"], torch.uint8)
        eh.write_tensor(os.path.join(dir_path, "scale_apot_residual_sign.int8.bin"), enc["residual_sign"], torch.int8)
        if signed:
            eh.write_tensor(os.path.join(dir_path, "scale_apot_overall_sign.int8.bin"), enc["overall_sign"], torch.int8)
        meta.update({
            "hardware_scale_format": "apot_signed" if signed else "apot_positive",
            "apot_files": [
                "scale_apot_a_exp.int16.bin",
                "scale_apot_b_exp.int16.bin",
                "scale_apot_has_second.u8.bin",
                "scale_apot_residual_sign.int8.bin",
            ] + (["scale_apot_overall_sign.int8.bin"] if signed else []),
        })
    else:
        meta["hardware_scale_format"] = "fp_scale_debug_only"
    return meta


def _write_int4_dir(
    module,
    dotted: str,
    outdir: str,
    q: torch.Tensor,
    raw_scale: torch.Tensor,
    scale_q: torch.Tensor,
    enc: Optional[Dict[str, torch.Tensor]],
    scale_granularity: str,
    extra_meta: Optional[Dict[str, Any]] = None,
    q_compact: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    d = os.path.join(outdir, "weights", dotted)
    eh.ensure(d)
    eh.write_bin(os.path.join(d, "w_int4.packed.bin"), eh.pack_int4_nibbles(q))
    eh.write_tensor(os.path.join(d, "w_int4.raw_s8.bin"), q, torch.int8)
    if q_compact is not None:
        eh.write_bin(os.path.join(d, "w_int4.blocks.packed.bin"), eh.pack_int4_nibbles(q_compact))
        eh.write_tensor(os.path.join(d, "w_int4.blocks.raw_s8.bin"), q_compact, torch.int8)
    scale_meta = _write_scale(d, raw_scale, scale_q, enc, signed=False)
    w = eh.raw_weight(module)
    meta = {
        "dotted": dotted,
        "module_type": type(module).__name__,
        "weight_shape": list(w.shape),
        "weight_format": "int4_symmetric_qmax7",
        "scale_granularity": scale_granularity,
        "scale_meta": scale_meta,
        "num_elements_dense": int(q.numel()),
        "packed_bytes_dense": int((q.numel() + 1) // 2),
        "bias_exported_here": False,
    }
    if q_compact is not None:
        meta["num_elements_compact_blocks"] = int(q_compact.numel())
        meta["packed_bytes_compact_blocks"] = int((q_compact.numel() + 1) // 2)
        meta["preferred_hw_weight_file"] = "w_int4.blocks.packed.bin"
    else:
        meta["preferred_hw_weight_file"] = "w_int4.packed.bin"
    if extra_meta:
        meta.update(extra_meta)
    eh.write_json(os.path.join(d, "meta.json"), meta)
    return meta


def export_int4_module(
    module,
    dotted: str,
    outdir: str,
    scale_granularity: str,
    group_size: int,
    num_heads: int,
    scale_format: str,
    mask: Optional[torch.Tensor] = None,
    extra_meta: Optional[Dict[str, Any]] = None,
    compact_blocks: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    w = eh.raw_weight(module).to(torch.float32).cpu()
    if scale_granularity == "per_channel":
        q, raw_scale, scale_q, enc, effective = eh.int4_per_channel(w, scale_format)
    elif scale_granularity == "per_group":
        q, raw_scale, scale_q, enc, effective = eh.int4_per_group_input_dim(w, group_size, scale_format)
    elif scale_granularity == "per_head":
        q, raw_scale, scale_q, enc, effective = eh.int4_per_head_linear(w, num_heads, scale_format)
    else:
        raise ValueError(f"unknown int4 scale_granularity={scale_granularity}")

    meta_extra = dict(extra_meta or {})
    meta_extra["effective_scale_granularity"] = effective
    if effective == "per_group":
        meta_extra["group_size"] = int(group_size)

    q_compact = compact_blocks
    if mask is not None:
        q = (q * mask.to(torch.int8)).to(torch.int8)
        meta_extra["block_diagonal_mask_applied_after_quant"] = True
        meta_extra["nonzero_int4_elements_after_mask"] = int((q != 0).sum().item())

    return _write_int4_dir(
        module, dotted, outdir, q, raw_scale, scale_q, enc,
        scale_granularity, meta_extra, q_compact=q_compact,
    )


def export_fp8_module(module, dotted: str, outdir: str, extra_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    # Decoder PW outliers use the source FP8Parametrization = PER-TENSOR amax-scaled
    # FP8 (scale=448/amax), NOT a plain w.to(e4m3) cast. Export the actual e4m3 code
    # bits = fp8(w*scale) plus the per-tensor descale so recon == module.weight.
    w = eh.raw_weight(module).detach().to(torch.float32).cpu()
    enc = eh.scaled_fp8_encode(w, torch.float8_e4m3fn)
    d = os.path.join(outdir, "weights", dotted)
    eh.ensure(d)
    # stored e4m3 code bits = fp8_e4m3(w * fp8_scale)
    eh.write_bin(os.path.join(d, "w_fp8.e4m3.bin"), enc["code"].view(torch.uint8).numpy())
    eh.write_tensor(os.path.join(d, "w_fp8.e4m3.code_value.fp32.bin"), enc["code_val"], torch.float32)
    # effective (reconstructed) weight == module.weight, for golden/sanity
    eh.write_tensor(os.path.join(d, "w_fp8.effective.fp32.bin"), enc["recon"], torch.float32)
    # per-tensor scale / descale (exact golden):  weight = code_value / scale = code_value * descale
    eh.write_tensor(os.path.join(d, "fp8_scale.fp32.bin"), enc["scale"], torch.float32)
    eh.write_tensor(os.path.join(d, "fp8_descale.fp32.bin"), enc["descale"], torch.float32)
    eh.write_tensor(os.path.join(d, "fp8_amax.fp32.bin"), enc["amax"], torch.float32)
    # APoT descale for the zero-multiplier datapath (shift-add / fold into L-Mul exponent).
    # Range is data-driven (epoch2 descale exponents live in [-14,-8]); [-16,-6] = 4-bit,
    # contains them with margin => no clamp => bit-identical to the trained descale.
    dd_min = int(os.environ.get("BCCTN_OUTLIER_DESCALE_APOT_MIN", "-16"))
    dd_max = int(os.environ.get("BCCTN_OUTLIER_DESCALE_APOT_MAX", "-6"))
    dap = eh.apot_encode(enc["descale"], dd_min, dd_max, signed=False)
    eh.write_tensor(os.path.join(d, "fp8_descale_apot_a_exp.int16.bin"), dap["a_exp"], torch.int16)
    eh.write_tensor(os.path.join(d, "fp8_descale_apot_b_exp.int16.bin"), dap["b_exp"], torch.int16)
    eh.write_tensor(os.path.join(d, "fp8_descale_apot_has_second.u8.bin"), dap["has_second"], torch.uint8)
    eh.write_tensor(os.path.join(d, "fp8_descale_apot_residual_sign.int8.bin"), dap["residual_sign"], torch.int8)
    meta = {
        "dotted": dotted,
        "module_type": type(module).__name__,
        "weight_shape": list(w.shape),
        "weight_format": "fp8_e4m3_per_tensor_scaled",
        "fp8_scaled": True,
        "fp8_scale_max_val": enc["max_val"],
        "fp8_amax": float(enc["amax"].item()),
        "fp8_scale": float(enc["scale"].item()),        # 448 / amax
        "fp8_descale": float(enc["descale"].item()),    # amax / 448
        "stored_code_is": "fp8_e4m3(weight * fp8_scale)",
        "reconstruction": "weight = e4m3_code_value * fp8_descale  (== code_value / fp8_scale, bit-exact vs module.weight)",
        "runtime_multiply": "lmul_m4_for_decoder_pw_outlier",
        "descale_hw_note": (
            "Per-tensor descale (amax/448) factors out of the per-element L-Mul into a single "
            "scaling of the fp16 accumulator per output. To keep the core multiplier-free, realize "
            "it as a pow2/APoT exponent offset folded into L-Mul (shift-add): see fp8_descale_apot_* "
            "and HANDOFF_HW_ZERO_MULTIPLIER_CORE.md §3.4. The exact fp32 fp8_descale is the golden "
            "reference; the APoT descale is the deployable shift-add form."
        ),
        "descale_apot_exp_range": [dd_min, dd_max],
        "descale_apot_files": [
            "fp8_descale_apot_a_exp.int16.bin",
            "fp8_descale_apot_b_exp.int16.bin",
            "fp8_descale_apot_has_second.u8.bin",
            "fp8_descale_apot_residual_sign.int8.bin",
        ],
        "bias_exported_here": False,
        "preferred_hw_weight_file": "w_fp8.e4m3.bin",
    }
    if extra_meta:
        meta.update(extra_meta)
    eh.write_json(os.path.join(d, "meta.json"), meta)
    return meta


def export_complex_linear(module, dotted: str, outdir: str, num_heads: int,
                          scale_format: str) -> Dict[str, Any]:
    w = eh.raw_weight(module).detach().cpu()
    metas = []
    for half, tensor in (("fc_r", w.real.contiguous()), ("fc_i", w.imag.contiguous())):
        q, raw_scale, scale_q, enc, effective = eh.int4_per_head_linear(
            tensor.to(torch.float32), num_heads, scale_format
        )
        # Small shim object so _write_int4_dir can read weight shape from the half.
        class _Half:
            weight = tensor
        meta = _write_int4_dir(
            _Half(), f"{dotted}/{half}", outdir, q, raw_scale, scale_q, enc,
            "per_head", {
                "complex_parent": dotted,
                "complex_half": half,
                "effective_scale_granularity": effective,
                "num_heads": int(num_heads),
            },
        )
        if module.bias is not None:
            b = module.bias.detach().cpu()
            bd = os.path.join(outdir, "weights", f"{dotted}/{half}")
            eh.write_tensor(os.path.join(bd, "bias.fp16.bin"), b.real if half == "fc_r" else b.imag, torch.float16)
            meta["bias_exported_here"] = True
            meta["bias_file"] = "bias.fp16.bin"
            eh.write_json(os.path.join(bd, "meta.json"), meta)
        metas.append(meta)
    parent = {
        "dotted": dotted,
        "weight_dtype_logical": "complex64",
        "weight_shape": list(w.shape),
        "halves": ["fc_r", "fc_i"],
        "scale_granularity": "per_head",
        "num_heads": int(num_heads),
    }
    eh.ensure(os.path.join(outdir, "weights", dotted))
    eh.write_json(os.path.join(outdir, "weights", dotted, "meta.json"), parent)
    return parent


def _mha_compact_blocks(q: torch.Tensor, embed_dim: int, groups: int) -> Optional[torch.Tensor]:
    groups = int(groups or 1)
    if groups <= 1:
        return None
    block = embed_dim // groups
    pieces = []
    for qkv_idx in range(3):
        base = qkv_idx * embed_dim
        for group_idx in range(groups):
            rows = slice(base + group_idx * block, base + (group_idx + 1) * block)
            cols = slice(group_idx * block, (group_idx + 1) * block)
            pieces.append(q[rows, cols].contiguous())
    return torch.stack(pieces, dim=0)


def export_mha(mha, dotted: str, outdir: str, num_heads: int, embed_dim: int,
               scale_format: str, qkv_groups: int, out_proj_groups: int) -> Dict[str, Any]:
    w = eh.raw_in_proj_weight(mha).to(torch.float32).cpu()
    q, raw_scale, scale_q, enc = eh.int4_per_head_mha(w, num_heads, scale_format)
    mask = eh.mha_in_proj_group_mask(q.shape, embed_dim, qkv_groups)
    if mask is not None:
        q_masked = (q * mask.to(torch.int8)).to(torch.int8)
    else:
        q_masked = q
    compact = _mha_compact_blocks(q_masked, embed_dim, qkv_groups)
    ip_dir = os.path.join(outdir, "weights", f"{dotted}/in_proj_weight")
    eh.ensure(ip_dir)
    eh.write_bin(os.path.join(ip_dir, "w_int4.packed.bin"), eh.pack_int4_nibbles(q_masked))
    eh.write_tensor(os.path.join(ip_dir, "w_int4.raw_s8.bin"), q_masked, torch.int8)
    if compact is not None:
        eh.write_bin(os.path.join(ip_dir, "w_int4.blocks.packed.bin"), eh.pack_int4_nibbles(compact))
        eh.write_tensor(os.path.join(ip_dir, "w_int4.blocks.raw_s8.bin"), compact, torch.int8)
    scale_meta = _write_scale(ip_dir, raw_scale, scale_q, enc, signed=False)
    if mha.in_proj_bias is not None:
        eh.write_tensor(os.path.join(ip_dir, "bias.fp16.bin"), mha.in_proj_bias.detach().cpu(), torch.float16)
    head_dim = embed_dim // num_heads
    meta_ip = {
        "dotted": f"{dotted}/in_proj_weight",
        "weight_shape": list(w.shape),
        "weight_format": "int4_symmetric_qmax7",
        "scale_granularity": "per_head_qkv",
        "num_heads": int(num_heads),
        "scale_groups": int(3 * num_heads),
        "qkv_groups": int(qkv_groups),
        "block_diagonal_mask_applied_after_quant": mask is not None,
        "preferred_hw_weight_file": "w_int4.blocks.packed.bin" if compact is not None else "w_int4.packed.bin",
        "compact_block_shape": list(compact.shape) if compact is not None else None,
        "qk_scale_runtime_pow2_exp": int(round(math.log2(1.0 / math.sqrt(head_dim)))),
        "qk_scale_fused_into_q_projection": False,
        "scale_meta": scale_meta,
        "bias_file": "bias.fp16.bin" if mha.in_proj_bias is not None else None,
    }
    eh.write_json(os.path.join(ip_dir, "meta.json"), meta_ip)

    op_mask = eh.linear_group_mask(tuple(mha.out_proj.weight.shape), out_proj_groups)
    out_meta = export_int4_module(
        mha.out_proj, f"{dotted}/out_proj", outdir,
        "per_head", group_size=32, num_heads=num_heads, scale_format=scale_format,
        mask=op_mask,
        extra_meta={"num_heads": int(num_heads), "out_proj_groups": int(out_proj_groups)},
    )
    if mha.out_proj.bias is not None:
        od = os.path.join(outdir, "weights", f"{dotted}/out_proj")
        eh.write_tensor(os.path.join(od, "bias.fp16.bin"), mha.out_proj.bias.detach().cpu(), torch.float16)
        out_meta["bias_exported_here"] = True
        out_meta["bias_file"] = "bias.fp16.bin"
        eh.write_json(os.path.join(od, "meta.json"), out_meta)
    return {"dotted": dotted, "in_proj": meta_ip, "out_proj": out_meta}


def export_block(block: eh.BlockInfo, outdir: str, bn_min_exp: int, bn_max_exp: int) -> Dict[str, Any]:
    d = os.path.join(outdir, "blocks", block.kind, f"stage_{block.stage_idx}")
    eh.ensure(d)
    aff = eh.folded_bn_affine(block, bn_min_exp, bn_max_exp, fuse_conv_bias=True)
    eh.write_tensor(os.path.join(d, "bn_scale_r_quant.fp32.bin"), aff.alpha_r, torch.float32)
    eh.write_tensor(os.path.join(d, "bn_scale_i_quant.fp32.bin"), aff.alpha_i, torch.float32)
    eh.write_tensor(os.path.join(d, "bn_bias_r_fused.fp32.bin"), aff.beta_r, torch.float32)
    eh.write_tensor(os.path.join(d, "bn_bias_i_fused.fp32.bin"), aff.beta_i, torch.float32)
    for prefix, code in (("bn_scale_r", aff.alpha_r_code), ("bn_scale_i", aff.alpha_i_code)):
        eh.write_tensor(os.path.join(d, f"{prefix}_apot_a_exp.int16.bin"), code["a_exp"], torch.int16)
        eh.write_tensor(os.path.join(d, f"{prefix}_apot_b_exp.int16.bin"), code["b_exp"], torch.int16)
        eh.write_tensor(os.path.join(d, f"{prefix}_apot_has_second.u8.bin"), code["has_second"], torch.uint8)
        eh.write_tensor(os.path.join(d, f"{prefix}_apot_residual_sign.int8.bin"), code["residual_sign"], torch.int8)
        eh.write_tensor(os.path.join(d, f"{prefix}_apot_overall_sign.int8.bin"), code["overall_sign"], torch.int8)

    prelu_meta = None
    if block.prelu_r is not None:
        pr = eh.pow2_encode(block.prelu_r.weight.detach().cpu(), -8, 0)
        pi = eh.pow2_encode(block.prelu_i.weight.detach().cpu(), -8, 0)
        eh.write_tensor(os.path.join(d, "prelu_r_pow2_exp.int16.bin"), pr["exp"], torch.int16)
        eh.write_tensor(os.path.join(d, "prelu_r_pow2_sign.int8.bin"), pr["sign"], torch.int8)
        eh.write_tensor(os.path.join(d, "prelu_i_pow2_exp.int16.bin"), pi["exp"], torch.int16)
        eh.write_tensor(os.path.join(d, "prelu_i_pow2_sign.int8.bin"), pi["sign"], torch.int8)
        eh.write_tensor(os.path.join(d, "prelu_r_quant.fp32.bin"), pr["quantized"], torch.float32)
        eh.write_tensor(os.path.join(d, "prelu_i_quant.fp32.bin"), pi["quantized"], torch.float32)
        prelu_meta = {
            "num_parameters_r": int(block.prelu_r.weight.numel()),
            "num_parameters_i": int(block.prelu_i.weight.numel()),
            "format": "signed_pow2",
            "exp_range": [-8, 0],
            "note": "Current source code uses scalar PReLU per real/imag block unless num_parameters is changed.",
        }

    meta = {
        "kind": block.kind,
        "stage_idx": block.stage_idx,
        "dotted": block.dotted,
        "num_out_channels": int(block.num_out_channels),
        "has_bn": block.bn_r is not None,
        "has_prelu": block.prelu_r is not None,
        "is_outlier_fp8": bool(block.is_outlier_fp8),
        "bn_scale_format": "signed_apot",
        "bn_scale_exp_range": [int(bn_min_exp), int(bn_max_exp)],
        "conv_bias_fused_into_bn_bias": True,
        "bn_bias_files": ["bn_bias_r_fused.fp32.bin", "bn_bias_i_fused.fp32.bin"],
        "prelu": prelu_meta,
    }
    eh.write_json(os.path.join(d, "meta.json"), meta)
    return meta


def export_skip_indices(model, outdir: str) -> Dict[str, Any]:
    d = os.path.join(outdir, "skip_indices")
    eh.ensure(d)
    items = []
    for idx in range(len(model.decoder.model)):
        name = f"skip_indices_{idx}"
        if not hasattr(model.decoder, name):
            continue
        values = getattr(model.decoder, name).detach().cpu().to(torch.int64)
        fn = f"decoder_stage_{idx}.u16.bin"
        eh.write_bin(os.path.join(d, fn), values.to(torch.uint16).numpy())
        items.append({
            "stage_idx": idx,
            "file": fn,
            "count": int(values.numel()),
            "min": int(values.min().item()) if values.numel() else None,
            "max": int(values.max().item()) if values.numel() else None,
        })
    meta = {"items": items, "format": "uint16 channel indices, stored group-wise"}
    eh.write_json(os.path.join(d, "meta.json"), meta)
    return meta


def export_stft(outdir: str, model_cfg: Dict[str, Any]) -> Dict[str, Any]:
    d = os.path.join(outdir, "misc")
    eh.ensure(d)
    win_len = int(model_cfg.get("win_len", 128))
    win_type = str(model_cfg.get("win_type", "hamming")).lower()
    if win_type == "hamming":
        win = torch.hamming_window(win_len, periodic=False, dtype=torch.float32)
    elif win_type == "hann":
        win = torch.hann_window(win_len, periodic=False, dtype=torch.float32)
    else:
        raise ValueError(f"unsupported win_type={win_type}")
    eh.write_tensor(os.path.join(d, "stft_window.fp16.bin"), win, torch.float16)
    eh.write_tensor(os.path.join(d, "istft_window.fp16.bin"), win, torch.float16)
    meta = {
        "win_len": win_len,
        "win_inc": int(model_cfg.get("win_inc", 80)),
        "fft_len": int(model_cfg.get("fft_len", 128)),
        "win_type": win_type,
        "window_periodic": False,
        "masking_mode": model_cfg.get("masking_mode", "C"),
        "scope_note": "DSP front/back-end is outside the zero-multiplier neural core claim.",
    }
    eh.write_json(os.path.join(d, "meta.json"), meta)
    return meta


import re as _re


def export_activation_blockfloat(outdir: str, calib_path: str) -> Dict[str, Any]:
    """Export the block-float activation exponent table for the pow2 deploy.

    Each activation boundary stores raw e4m3 code = e4m3(x * 2^k) plus the single
    shared exponent k (per tensor). On hardware the activation enters the MAC /
    L-Mul as that raw code; its descale 2^-k is fused into the layer's
    post-accumulation weight descale by ADDING (-k) to the weight APoT
    exponents (a shift, no multiply). See HANDOFF §3.4/§5.
    """
    table = eh.blockfloat_activation_table(calib_path)
    d = os.path.join(outdir, "activation_blockfloat")
    eh.ensure(d)
    ordered = sorted(table.keys())
    scale_exp = torch.tensor([table[k]["scale_pow2_exp"] for k in ordered], dtype=torch.int16)
    descale_exp = torch.tensor([table[k]["descale_pow2_exp"] for k in ordered], dtype=torch.int16)
    eh.write_tensor(os.path.join(d, "scale_pow2_exp.int16.bin"), scale_exp, torch.int16)
    eh.write_tensor(os.path.join(d, "descale_pow2_exp.int16.bin"), descale_exp, torch.int16)
    eh.write_json(os.path.join(d, "boundary_keys.json"), ordered)
    eh.write_json(os.path.join(d, "table.json"), table)
    exps = [v["scale_pow2_exp"] for v in table.values()]
    return {
        "format": "blockfloat_e4m3_shared_exponent",
        "calib_path": os.path.abspath(calib_path),
        "num_boundaries": len(table),
        "scale_pow2_exp_range": [int(min(exps)), int(max(exps))],
        "rounding": "floor(log2(448/calib_amax)) == DCNN.utils.fp8_qat._pow2_round(mode=floor)",
        "activation_storage": "raw e4m3 code = e4m3(x * 2^scale_pow2_exp); no per-activation runtime descale",
        "fusion_rule": "final_post_acc_descale_exp = weight_APoT_exp + descale_pow2_exp(input boundary) "
                       "= weight_APoT_exp - scale_pow2_exp; pure exponent offset (shift-add, no multiply)",
        "files": ["table.json", "boundary_keys.json",
                  "scale_pow2_exp.int16.bin", "descale_pow2_exp.int16.bin"],
        "exceptions": {
            "mha_out_proj": "input = Score x V fp16 accumulator (NOT block-float A8); INT4 weight "
                            "still shift-add (W4A16). No activation exponent for out_proj.",
            "qk_descale_placement": "QK activation descale (2^-k applied to Q and K) must be applied on "
                                    "the QK accumulator BEFORE softmax (it sets logit temperature); not "
                                    "deferred to score x V / attention output.",
        },
    }


def _resolve_act_boundary(dotted: str, table: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Map a layer dotted-name to its input activation block-float boundary(ies)."""
    def entry(key):
        v = table.get(key)
        if v is None:
            return None
        return {"boundary_key": key, "scale_pow2_exp": v["scale_pow2_exp"],
                "descale_pow2_exp": v["descale_pow2_exp"]}

    # decoder split pointwise: input is the shared cat boundary (real for _r_, imag for _i_)
    m = _re.search(r"(decoder\.model\.\d+\.0)\.pointwise_([ri])_(x|skip)$", dotted)
    if m:
        part = "real" if m.group(2) == "r" else "imag"
        return entry(f"{m.group(1)}.split_pointwise_cat.{part}")
    # encoder dw/pw + decoder dw: own per-conv boundary
    m = _re.search(r"((encoder|decoder)\.model\.\d+\.0\.(depthwise|pointwise)_[ri])$", dotted)
    if m:
        return entry(m.group(1))
    # complex MHA projections
    if dotted == "mattn.input_proj":
        return {"real": entry("mattn.input_proj.real"), "imag": entry("mattn.input_proj.imag")}
    if dotted == "mattn.output_proj":
        return {"real": entry("mattn.output_proj.real"), "imag": entry("mattn.output_proj.imag")}
    # internal MHA: in_proj / qk / score x V all use the single qkv boundary; out_proj = wide accumulator
    if dotted.startswith("mattn.mattn.MultiheadAttention"):
        e = entry("mattn.mattn.activation_quantizer")
        return {"in_proj_qkv": e,
                "out_proj": "wide_fp16_accumulator_no_blockfloat (Option 2)"}
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=eh.DEFAULT_CKPT)
    ap.add_argument("--config", default=eh.DEFAULT_CONFIG)
    ap.add_argument("--code-dir", default=eh.DEFAULT_CODE_DIR)
    ap.add_argument("--skip-indices", default=eh.DEFAULT_SKIP_INDICES)
    ap.add_argument("--act-calib", default=eh.DEFAULT_ACT_CALIB)
    ap.add_argument("--outdir", default=os.path.join(eh.PACKAGE_ROOT, "export_out_strict18"))
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if os.path.exists(args.outdir):
        if not args.overwrite:
            raise FileExistsError(f"{args.outdir} exists; pass --overwrite to replace it")
        shutil.rmtree(args.outdir)
    eh.ensure(args.outdir)

    model, model_cfg, training_cfg, load_report = eh.build_model(
        config_path=args.config,
        ckpt_path=args.ckpt,
        code_dir=args.code_dir,
        skip_indices_path=args.skip_indices,
    )
    num_heads = int(model_cfg.get("num_heads", 16))
    embed_dim = int(model_cfg.get("embed_dim", 256))
    group_size = int(training_cfg.get("int4_group_size", 32))
    scale_format = training_cfg.get("int4_scale_format", "apot_clip")
    bn_cfg = model_cfg.get("low_power", {}).get("bn_affine", {}) or {}
    bn_min = int(bn_cfg.get("pow2_min_exp", -9))
    bn_max = int(bn_cfg.get("pow2_max_exp", 4))

    blocks = eh.discover_blocks(model)
    block_metas = []
    layer_metas = []

    for block in blocks:
        block_metas.append(export_block(block, args.outdir, bn_min, bn_max))
        if block.depthwise_r is not None:
            for half, sub in (("r", block.depthwise_r), ("i", block.depthwise_i)):
                layer_metas.append(export_int4_module(
                    sub, f"{block.dotted}.depthwise_{half}", args.outdir,
                    "per_channel", group_size, num_heads, scale_format,
                    extra_meta={"conv_bias_fused_elsewhere": True, "is_depthwise": True},
                ))
        for suffix, sub in block.pointwise_items:
            dotted = f"{block.dotted}.{suffix}"
            if block.is_outlier_fp8:
                layer_metas.append(export_fp8_module(
                    sub, dotted, args.outdir,
                    extra_meta={"is_decoder_pw_outlier": True, "lmul_mantissa_bits": 4},
                ))
            else:
                layer_metas.append(export_int4_module(
                    sub, dotted, args.outdir,
                    "per_group", group_size, num_heads, scale_format,
                    extra_meta={"conv_bias_fused_elsewhere": True, "is_pointwise": True},
                ))

    matt = model.mattn
    layer_metas.append(export_complex_linear(matt.input_proj, "mattn.input_proj", args.outdir, num_heads, scale_format))
    layer_metas.append(export_complex_linear(matt.output_proj, "mattn.output_proj", args.outdir, num_heads, scale_format))

    cmha = matt.mattn
    qkv_groups = int(getattr(cmha, "mha_projection", {}).get("qkv_groups", 1) or 1)
    out_proj_groups = int(getattr(cmha, "mha_projection", {}).get("out_proj_groups", 1) or 1)
    layer_metas.append(export_mha(
        cmha.MultiheadAttention_re,
        "mattn.mattn.MultiheadAttention_re",
        args.outdir,
        num_heads,
        embed_dim,
        scale_format,
        qkv_groups,
        out_proj_groups,
    ))

    skip_meta = export_skip_indices(model, args.outdir)
    stft_meta = export_stft(args.outdir, model_cfg)

    # Block-float activation export (pow2 shared exponent) + per-layer input boundary.
    act_bf_meta = export_activation_blockfloat(args.outdir, args.act_calib)
    bf_table = eh.blockfloat_activation_table(args.act_calib)
    for lm in layer_metas:
        ia = _resolve_act_boundary(lm.get("dotted", ""), bf_table)
        if ia is not None:
            lm["input_activation"] = ia

    manifest = {
        "source_ckpt": os.path.abspath(args.ckpt),
        "config": os.path.abspath(args.config),
        "code_dir": os.path.abspath(args.code_dir),
        "load_report": {
            "missing_count": len(load_report["missing"]),
            "unexpected_count": len(load_report["unexpected"]),
            "missing_first10": load_report["missing"][:10],
            "unexpected_first10": load_report["unexpected"][:10],
        },
        "model": {
            "name": "BCCTN strict-core zero-multiplier deploy",
            "embed_dim": embed_dim,
            "num_heads": num_heads,
            "head_dim": embed_dim // num_heads,
            "int4_group_size": group_size,
            "int4_scale_format": scale_format,
            "qkv_groups": qkv_groups,
            "decoder_groups_x": model_cfg.get("decoder_grouped_pointwise", {}).get("groups_x"),
            "decoder_groups_skip": model_cfg.get("decoder_grouped_pointwise", {}).get("groups_skip"),
            "decoder_skip_select_ratio": model_cfg.get("decoder_grouped_pointwise", {}).get("skip_select_ratio"),
            "softmax": model_cfg.get("attention_softmax", {}),
            "attention_datapath": model_cfg.get("attention_datapath", {}),
            "low_power": model_cfg.get("low_power", {}),
        },
        "hardware_notes": [
            "Conv/DW/MHA projection INT4 q uses qmax=7.",
            "Conv pointwise scale groups follow the Conv2d input-channel axis, not the trailing kW axis.",
            "INT4 APoT scales are exported as exponent/sign code files; fp16/fp32 scale files are debug views.",
            "MHA QKV qkv_groups mask is applied after per-head quantization, matching current source behavior.",
            "MHA in_proj compact block file stores only diagonal 64x64 Q/K/V blocks for qkv_groups=4.",
            "BN alpha is APoT of gamma/sqrt(running_var+eps); conv bias is fused into the exported BN bias.",
            "Decoder PW outliers (12) are PER-TENSOR SCALED FP8: stored e4m3 = fp8(w*448/amax); "
            "effective weight = e4m3_code * (amax/448). The per-tensor descale is exported (exact fp32 "
            "fp8_descale + APoT fp8_descale_apot_*) and applies once per output on the fp16 accumulator.",
            "PReLU is exported as signed pow2. Current source uses scalar PReLU per real/imag block.",
            "QK 1/sqrt(head_dim) is not pre-fused; metadata gives the runtime pow2 shift exponent.",
            "ACTIVATIONS are block-float: e4m3 raw code + per-tensor shared exponent (see "
            "activation_blockfloat/). The activation enters every MAC / L-Mul as raw e4m3 code; there is "
            "NO runtime activation amax/descale. Each layer's post-acc descale = weight APoT exponent + "
            "the input boundary's descale_pow2_exp (a shift-add offset, no multiply). Because the activation "
            "scale is a single pow2, eager==fused exactly at every L-Mul site, so the validated MBSTOI "
            "(VCTK 0.8357 / TIMIT 0.7898) IS the deployed hardware number.",
            "EXCEPTION: MHA internal out_proj input is the Score x V fp16 accumulator (W4A16, not A8); INT4 "
            "weight still shift-add. QK activation descale must be applied on the QK accumulator BEFORE "
            "softmax (logit temperature). STFT/iSTFT/complex-mask are Q1.15 DSP, outside the zero-mult core.",
        ],
        "activation_blockfloat": act_bf_meta,
        "blocks": block_metas,
        "layers": layer_metas,
        "skip_indices": skip_meta,
        "stft": stft_meta,
    }
    eh.write_json(os.path.join(args.outdir, "manifest.json"), manifest)
    print(f"[OK] exported strict18 hw data to {args.outdir}")
    print(f"  blocks={len(block_metas)} layers={len(layer_metas)}")
    print(f"  load missing={len(load_report['missing'])} unexpected={len(load_report['unexpected'])}")


if __name__ == "__main__":
    main()
