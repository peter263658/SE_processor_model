#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Config-aware model parameter inspector.

This script intentionally builds the model from config/config.yaml. Older
versions accepted --config but instantiated BinauralAttentionDCNN() with default
constructor arguments, which silently produced the wrong architecture for the
09+ experiments.
"""
import argparse
import csv
import os
from collections import OrderedDict
from copy import deepcopy

import torch
import yaml


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config/config.yaml", help="Path to config YAML")
    ap.add_argument("--ckpt", type=str, default=None, help="Optional checkpoint path")
    ap.add_argument("--out-csv", type=str, default="model_inspect.csv", help="Output CSV path")
    ap.add_argument("--set", dest="overrides", action="append", default=[], help="Dotted YAML override, e.g. model.tied_mha=false")
    ap.add_argument("--strict-load", action="store_true", help="Load checkpoint with strict=True")
    ap.add_argument("--untied-reference", action="store_true", help="Also count same config with model.tied_mha=false")
    return ap.parse_args()


def set_dotted(cfg, dotted_key):
    if "=" not in dotted_key:
        raise ValueError(f"override must be key=value, got {dotted_key!r}")
    key, raw = dotted_key.split("=", 1)
    parts = key.split(".")
    if parts[0] != "model":
        parts = ["model"] + parts
    cur = cfg
    for part in parts[:-1]:
        cur = cur.setdefault(part, {})
    cur[parts[-1]] = yaml.safe_load(raw)


def bytes_per_elem(dtype):
    if dtype == torch.complex64:
        return 8
    if dtype == torch.complex128:
        return 16
    if dtype in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
        return torch.finfo(dtype).bits // 8
    if dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.bool):
        return torch.iinfo(dtype).bits // 8
    raise ValueError(f"Unsupported dtype: {dtype}")


def build_model(model_cfg):
    from DCNN.models.binaural_attention_model import BinauralAttentionDCNN
    return BinauralAttentionDCNN(**deepcopy(model_cfg)).cpu().eval()


def load_checkpoint(model, ckpt_path, strict):
    if not ckpt_path:
        return
    print(f"[+] Load checkpoint: {ckpt_path}")
    sd = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" in sd:
        sd = sd["state_dict"]
    cleaned = OrderedDict((k.replace("module.", ""), v) for k, v in sd.items())
    result = model.load_state_dict(cleaned, strict=strict)
    if not strict:
        print(f"[i] non-strict load: missing={len(result.missing_keys)} unexpected={len(result.unexpected_keys)}")


def _flatten_for_stats(param):
    data = param.detach().reshape(-1).cpu()
    if data.is_complex():
        data = torch.view_as_real(data).reshape(-1)
    return data.float()


def summarize_layer(name, module):
    params = list(module.parameters(recurse=False))
    if not params:
        return None
    flat = torch.cat([_flatten_for_stats(p) for p in params])
    n = sum(p.numel() for p in params)
    dtype = params[0].dtype
    bpe = bytes_per_elem(dtype)
    return OrderedDict(
        layer=name,
        cls=module.__class__.__name__,
        n=n,
        dtype=str(dtype),
        mb=n * bpe / 1024**2,
        mb_q8=n / 1024**2,
        shapes=";".join(str(tuple(p.shape)) for p in params),
        w_min=float(flat.min()),
        w_max=float(flat.max()),
        w_mu=float(flat.mean()),
        w_sd=float(flat.std(unbiased=False)),
    )


def collect_rows(model):
    rows = []
    for name, module in model.named_modules():
        if not name:
            continue
        row = summarize_layer(name, module)
        if row:
            rows.append(row)
    return rows


def write_csv(rows, out_csv):
    if not rows:
        raise RuntimeError("no parameterized layers found")
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_config_summary(model_cfg):
    interesting = {
        "embed_dim": model_cfg.get("embed_dim"),
        "num_heads": model_cfg.get("num_heads"),
        "attn_win": model_cfg.get("attn_win"),
        "tied_mha": model_cfg.get("tied_mha"),
        "attention_softmax": model_cfg.get("attention_softmax"),
        "decoder_grouped_pointwise": model_cfg.get("decoder_grouped_pointwise"),
        "attention_datapath": model_cfg.get("attention_datapath"),
        "low_power": model_cfg.get("low_power"),
    }
    print("[i] Applied model config:")
    for key, value in interesting.items():
        if value is not None:
            print(f"    {key}: {value}")


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    for override in args.overrides:
        set_dotted(cfg, override)

    model_cfg = deepcopy(cfg.get("model", {}))
    print_config_summary(model_cfg)
    model = build_model(model_cfg)
    load_checkpoint(model, args.ckpt, args.strict_load)

    rows = collect_rows(model)
    write_csv(rows, args.out_csv)
    total = sum(row["n"] for row in rows)
    total_mb = sum(row["mb"] for row in rows)
    print(f"[+] Saved summary to {args.out_csv}")
    print(f"[+] Total parameter elements: {total:,} ({total_mb:.3f} MiB native dtype)")

    if args.untied_reference:
        ref_cfg = deepcopy(model_cfg)
        ref_cfg["tied_mha"] = False
        ref = build_model(ref_cfg)
        ref_total = sum(p.numel() for p in ref.parameters())
        delta = ref_total - total
        pct = 100.0 * delta / ref_total if ref_total else 0.0
        print(f"[+] Untied reference parameter elements: {ref_total:,}")
        print(f"[+] Tied reduction: {delta:,} ({pct:.2f}%)")


if __name__ == "__main__":
    main()
