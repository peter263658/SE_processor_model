from __future__ import annotations

import argparse
import os
from typing import Any, Dict

import export_helpers as eh


def _conv_shape(module) -> Dict[str, Any]:
    w = eh.raw_weight(module)
    return {
        "module_type": type(module).__name__,
        "weight_shape": list(w.shape),
        "groups": int(getattr(module, "groups", 1)),
        "kernel_size": list(getattr(module, "kernel_size", [])),
        "stride": list(getattr(module, "stride", [])),
        "padding": list(getattr(module, "padding", [])),
        "has_bias": getattr(module, "bias", None) is not None,
    }


def build_config(model, model_cfg: Dict[str, Any], training_cfg: Dict[str, Any]) -> Dict[str, Any]:
    num_heads = int(model_cfg.get("num_heads", 16))
    embed_dim = int(model_cfg.get("embed_dim", 256))
    group_size = int(training_cfg.get("int4_group_size", 32))
    scale_format = training_cfg.get("int4_scale_format", "apot_clip")
    blocks = []
    layers = []

    for block in eh.discover_blocks(model):
        blocks.append({
            "kind": block.kind,
            "stage_idx": block.stage_idx,
            "dotted": block.dotted,
            "num_out_channels": int(block.num_out_channels),
            "has_bn": block.bn_r is not None,
            "has_prelu": block.prelu_r is not None,
            "is_decoder_pw_outlier_fp8": bool(block.is_outlier_fp8),
            "bn_scale_format": "signed_apot[-9,4]",
            "conv_bias_fused_into_bn_bias": True,
            "prelu_format": "signed_pow2[-8,0]" if block.prelu_r is not None else None,
            "prelu_num_parameters_r": int(block.prelu_r.weight.numel()) if block.prelu_r is not None else 0,
            "prelu_num_parameters_i": int(block.prelu_i.weight.numel()) if block.prelu_i is not None else 0,
        })
        for half, sub in (("r", block.depthwise_r), ("i", block.depthwise_i)):
            info = _conv_shape(sub)
            info.update({
                "dotted_name": f"{block.dotted}.depthwise_{half}",
                "weight_format": "int4_symmetric_qmax7",
                "scale_granularity": "per_output_channel",
                "scale_format": scale_format,
                "activation_format": "fp8_e4m3_at_op_input",
                "is_depthwise": True,
            })
            layers.append(info)
        for suffix, sub in block.pointwise_items:
            info = _conv_shape(sub)
            info.update({
                "dotted_name": f"{block.dotted}.{suffix}",
                "weight_format": "fp8_e4m3_per_tensor_scaled" if block.is_outlier_fp8 else "int4_symmetric_qmax7",
                "scale_granularity": "per_tensor_amax_scaled_fp8" if block.is_outlier_fp8 else "per_group_input_channels",
                "group_size": None if block.is_outlier_fp8 else group_size,
                "scale_format": None if block.is_outlier_fp8 else scale_format,
                "activation_format": "fp8_e4m3_at_op_input",
                "is_pointwise": True,
                "runtime_product": "fp8_fp8_lmul_m4" if block.is_outlier_fp8 else "int4_code_x_fp8_shift_add",
            })
            layers.append(info)

    cmha = model.mattn.mattn
    qkv_groups = int(getattr(cmha, "mha_projection", {}).get("qkv_groups", 1) or 1)
    out_proj_groups = int(getattr(cmha, "mha_projection", {}).get("out_proj_groups", 1) or 1)
    attention = {
        "tied_mha": bool(getattr(cmha, "tied", False)),
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "head_dim": embed_dim // num_heads,
        "qkv_groups": qkv_groups,
        "out_proj_groups": out_proj_groups,
        "qkv_block_shape": [3, qkv_groups, embed_dim // qkv_groups, embed_dim // qkv_groups],
        "projection_weight_format": "int4_symmetric_qmax7",
        "projection_scale_format": f"positive_apot[-16,-1] per head ({3 * num_heads} groups for in_proj)",
        "qk_product": "lmul_m8",
        "score_v_product": "top4_packet_lmul_m8",
        "softmax": model_cfg.get("attention_softmax", {}),
        "prob_packet": model_cfg.get("attention_datapath", {}).get("prob_packet", {}),
        "qk_scale": {
            "mode": model_cfg.get("attention_datapath", {}).get("qk_scale_mode", "pow2"),
            "pow2_exp": -2,
            "fused_into_q_projection_in_export": False,
        },
    }

    return {
        "model": {
            "name": "18 strict-core BCCTN zero-multiplier neural core",
            "activation_format": "fp8_e4m3 at Conv/Linear/MHA op inputs",
            "int4_group_size": group_size,
            "int4_scale_format": scale_format,
            "decoder_grouped_pointwise": model_cfg.get("decoder_grouped_pointwise", {}),
            "low_power": model_cfg.get("low_power", {}),
        },
        "blocks": blocks,
        "layers": layers,
        "attention": attention,
        "skip_indices": "static top-50% skip channels; see skip_indices/meta.json after ckpt2hwdat.py",
        "out_of_scope": ["STFT", "iSTFT", "window multiply", "waveform-domain mask apply"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=eh.DEFAULT_CONFIG)
    ap.add_argument("--ckpt", default=eh.DEFAULT_CKPT)
    ap.add_argument("--code-dir", default=eh.DEFAULT_CODE_DIR)
    ap.add_argument("--skip-indices", default=eh.DEFAULT_SKIP_INDICES)
    ap.add_argument("--out", default=os.path.join(eh.PACKAGE_ROOT, "layer_config_strict18.json"))
    args = ap.parse_args()

    model, model_cfg, training_cfg, load_report = eh.build_model(
        config_path=args.config,
        ckpt_path=args.ckpt,
        code_dir=args.code_dir,
        skip_indices_path=args.skip_indices,
    )
    cfg = build_config(model, model_cfg, training_cfg)
    cfg["load_report"] = {
        "missing_count": len(load_report["missing"]),
        "unexpected_count": len(load_report["unexpected"]),
        "missing_first10": load_report["missing"][:10],
        "unexpected_first10": load_report["unexpected"][:10],
    }
    eh.write_json(args.out, cfg)
    print(f"[OK] wrote {args.out}")
    print(f"  blocks={len(cfg['blocks'])} layers={len(cfg['layers'])}")


if __name__ == "__main__":
    main()
