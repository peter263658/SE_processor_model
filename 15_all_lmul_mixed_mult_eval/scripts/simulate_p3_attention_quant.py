#!/usr/bin/env python3
"""Offline P3-B0 score-quantization simulation.

Consumes the NPZ/JSON produced by dump_p3_attention_scores.py and reports
per-layer/per-head top-k stability for finite-only FP8 and derived fixed-Q
score formats.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch


REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from DCNN.utils.complexPyTorch.complexLayers import _quantize_attention_scores


def _topk_indices(rows: torch.Tensor, finite_mask_threshold: float, k: int):
    valid = torch.isfinite(rows) & (rows > finite_mask_threshold)
    enough = valid.sum(dim=-1) >= k
    masked = rows.masked_fill(~valid, float("-inf"))
    idx = torch.topk(masked, k=k, dim=-1).indices
    return idx, enough, valid


def _support_metrics(orig_idx: torch.Tensor, quant_idx: torch.Tensor, enough: torch.Tensor):
    if enough.sum().item() == 0:
        return {
            "rows": 0,
            "topk_jaccard": None,
            "topk_exact_match_rate": None,
            "top1_flip_rate": None,
            "row_max_order_flip_rate": None,
        }
    orig_idx = orig_idx[enough]
    quant_idx = quant_idx[enough]
    k = orig_idx.shape[-1]
    match = orig_idx.unsqueeze(-1).eq(quant_idx.unsqueeze(-2))
    intersection = match.any(dim=-1).sum(dim=-1).to(torch.float32)
    union = (2 * k) - intersection
    jaccard = intersection / union.clamp_min(1.0)
    exact = intersection.eq(k).to(torch.float32)
    top1_flip = orig_idx[:, 0].ne(quant_idx[:, 0]).to(torch.float32)
    return {
        "rows": int(enough.sum().item()),
        "topk_jaccard": float(jaccard.mean().item()),
        "topk_exact_match_rate": float(exact.mean().item()),
        "top1_flip_rate": float(top1_flip.mean().item()),
        "row_max_order_flip_rate": float(top1_flip.mean().item()),
    }


def _iter_head_chunks(npz, layer: Dict, head: str) -> Iterable[np.ndarray]:
    for key in layer["score_keys"].get(head, []):
        yield npz[key]


def _head_values(chunks: List[np.ndarray], finite_mask_threshold: float):
    vals = []
    for chunk in chunks:
        flat = chunk.reshape(-1)
        mask = np.isfinite(flat) & (flat > finite_mask_threshold)
        if mask.any():
            vals.append(flat[mask].astype(np.float64, copy=False))
    if not vals:
        return np.array([], dtype=np.float64)
    return np.concatenate(vals)


def _derive_fixed_bits(values: np.ndarray):
    if values.size == 0:
        return 2, 8
    p001, p999 = np.percentile(values, [0.1, 99.9])
    max_abs = max(abs(float(p001)), abs(float(p999)), 1e-12)
    int_bits = max(2, int(math.ceil(math.log2(max_abs))) + 1)

    unique = np.unique(values)
    diffs = np.diff(unique)
    positive = diffs[diffs > 0]
    if positive.size:
        min_step = max(float(positive.min()), 2.0 ** -24)
        frac_bits = max(0, int(math.ceil(math.log2(1.0 / min_step))))
    else:
        frac_bits = 8
    return int_bits, frac_bits


def _fixed_saturation_rate(values: np.ndarray, int_bits: int, frac_bits: int):
    if values.size == 0:
        return 0.0
    step = 2.0 ** (-frac_bits)
    qmin = -(2.0 ** (int_bits - 1))
    qmax = (2.0 ** (int_bits - 1)) - step
    return float(np.mean((values < qmin) | (values > qmax)))


def _simulate_chunks(chunks: List[np.ndarray], quant_cfg: Dict, finite_mask_threshold: float, k: int):
    merged = {
        "rows": 0,
        "topk_jaccard_sum": 0.0,
        "topk_exact_sum": 0.0,
        "top1_flip_sum": 0.0,
        "row_max_order_flip_sum": 0.0,
    }
    for chunk in chunks:
        rows = torch.from_numpy(chunk.astype(np.float32, copy=False))
        orig_idx, enough, _ = _topk_indices(rows, finite_mask_threshold, k)
        quantized = _quantize_attention_scores(rows, quant_cfg)
        quant_idx, _, _ = _topk_indices(quantized, finite_mask_threshold, k)
        metrics = _support_metrics(orig_idx, quant_idx, enough)
        n = metrics["rows"]
        if n <= 0:
            continue
        merged["rows"] += n
        merged["topk_jaccard_sum"] += metrics["topk_jaccard"] * n
        merged["topk_exact_sum"] += metrics["topk_exact_match_rate"] * n
        merged["top1_flip_sum"] += metrics["top1_flip_rate"] * n
        merged["row_max_order_flip_sum"] += metrics["row_max_order_flip_rate"] * n

    n = merged["rows"]
    if n <= 0:
        return {
            "rows": 0,
            "topk_jaccard": None,
            "topk_exact_match_rate": None,
            "top1_flip_rate": None,
            "row_max_order_flip_rate": None,
        }
    return {
        "rows": int(n),
        "topk_jaccard": merged["topk_jaccard_sum"] / n,
        "topk_exact_match_rate": merged["topk_exact_sum"] / n,
        "top1_flip_rate": merged["top1_flip_sum"] / n,
        "row_max_order_flip_rate": merged["row_max_order_flip_sum"] / n,
    }


def _passes(metrics: Dict, min_jaccard: float, max_top1_flip: float):
    if metrics["rows"] <= 0:
        return False
    return (
        metrics["topk_jaccard"] is not None
        and metrics["top1_flip_rate"] is not None
        and metrics["topk_jaccard"] >= min_jaccard
        and metrics["top1_flip_rate"] <= max_top1_flip
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump_npz", required=True)
    parser.add_argument("--summary_json", default=None)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--top_k", type=int, default=4)
    parser.add_argument("--min_jaccard", type=float, default=None)
    parser.add_argument("--max_top1_flip", type=float, default=None)
    parser.add_argument("--finite_mask_threshold", type=float, default=None)
    args = parser.parse_args()

    dump_npz = Path(args.dump_npz)
    summary_json = Path(args.summary_json) if args.summary_json else dump_npz.with_suffix(".json")
    output_json = Path(args.output_json) if args.output_json else dump_npz.with_name(
        dump_npz.stem + "_quant_sim.json"
    )

    with summary_json.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    npz = np.load(dump_npz)

    finite_mask_threshold = (
        float(args.finite_mask_threshold)
        if args.finite_mask_threshold is not None
        else float(summary.get("finite_mask_threshold", -10000.0))
    )
    min_jaccard = (
        float(args.min_jaccard)
        if args.min_jaccard is not None
        else 0.98
    )
    max_top1_flip = (
        float(args.max_top1_flip)
        if args.max_top1_flip is not None
        else 0.01
    )

    results = {
        "dump_npz": str(dump_npz),
        "summary_json": str(summary_json),
        "finite_mask_threshold": finite_mask_threshold,
        "top_k": int(args.top_k),
        "min_jaccard": min_jaccard,
        "max_top1_flip": max_top1_flip,
        "layers": {},
    }

    global_fp8_pass = True
    global_fixed_pass = True
    worst_fp8 = {"topk_jaccard": 1.0, "top1_flip_rate": 0.0, "layer": None, "head": None}
    worst_fixed = {"topk_jaccard": 1.0, "top1_flip_rate": 0.0, "layer": None, "head": None}

    for layer_name, layer in summary["layers"].items():
        layer_out = {"heads": {}}
        layer_fixed_int_bits = []
        layer_fixed_frac_bits = []
        for head in sorted(layer["score_keys"], key=lambda h: int(h)):
            chunks = list(_iter_head_chunks(npz, layer, head))
            values = _head_values(chunks, finite_mask_threshold)

            fp8_cfg = {
                "enabled": True,
                "score_quant": "fp8_e4m3",
                "finite_mask_threshold": finite_mask_threshold,
            }
            fp8_metrics = _simulate_chunks(chunks, fp8_cfg, finite_mask_threshold, args.top_k)
            fp8_pass = _passes(fp8_metrics, min_jaccard, max_top1_flip)
            global_fp8_pass = global_fp8_pass and fp8_pass

            int_bits, frac_bits = _derive_fixed_bits(values)
            fixed_attempts = []
            fixed_pass = False
            chosen = None
            for widened in range(3):
                trial_int_bits = int_bits + widened
                sat_rate = _fixed_saturation_rate(values, trial_int_bits, frac_bits)
                fixed_cfg = {
                    "enabled": True,
                    "score_quant": "fixed",
                    "score_fixed_int_bits": trial_int_bits,
                    "score_fixed_frac_bits": frac_bits,
                    "finite_mask_threshold": finite_mask_threshold,
                }
                fixed_metrics = _simulate_chunks(chunks, fixed_cfg, finite_mask_threshold, args.top_k)
                attempt = {
                    "int_bits": trial_int_bits,
                    "frac_bits": frac_bits,
                    "saturation_rate": sat_rate,
                    "metrics": fixed_metrics,
                    "pass": _passes(fixed_metrics, min_jaccard, max_top1_flip)
                    and sat_rate <= 0.01,
                }
                fixed_attempts.append(attempt)
                if attempt["pass"]:
                    fixed_pass = True
                    chosen = attempt
                    break

            global_fixed_pass = global_fixed_pass and fixed_pass
            fixed_for_recommendation = chosen if chosen else fixed_attempts[-1]
            layer_fixed_int_bits.append(int(fixed_for_recommendation["int_bits"]))
            layer_fixed_frac_bits.append(int(fixed_for_recommendation["frac_bits"]))

            for bucket, metrics in ((worst_fp8, fp8_metrics), (worst_fixed, chosen["metrics"] if chosen else fixed_attempts[-1]["metrics"])):
                if metrics["topk_jaccard"] is not None and metrics["topk_jaccard"] < bucket["topk_jaccard"]:
                    bucket["topk_jaccard"] = metrics["topk_jaccard"]
                    bucket["layer"] = layer_name
                    bucket["head"] = head
                if metrics["top1_flip_rate"] is not None and metrics["top1_flip_rate"] > bucket["top1_flip_rate"]:
                    bucket["top1_flip_rate"] = metrics["top1_flip_rate"]
                    bucket["layer"] = layer_name
                    bucket["head"] = head

            layer_out["heads"][head] = {
                "fp8_e4m3": {"metrics": fp8_metrics, "pass": fp8_pass},
                "fixed_q": {
                    "derived_int_bits": int_bits,
                    "derived_frac_bits": frac_bits,
                    "attempts": fixed_attempts,
                    "chosen": chosen,
                    "pass": fixed_pass,
                },
            }
        layer_out["recommended_fixed_q"] = {
            "score_fixed_int_bits": max(layer_fixed_int_bits) if layer_fixed_int_bits else None,
            "score_fixed_frac_bits": max(layer_fixed_frac_bits) if layer_fixed_frac_bits else None,
            "all_heads_pass": all(
                h["fixed_q"]["pass"] for h in layer_out["heads"].values()
            ) if layer_out["heads"] else False,
        }
        results["layers"][layer_name] = layer_out

    results["gate"] = {
        "fp8_score_pass": bool(global_fp8_pass),
        "fixed_score_pass": bool(global_fixed_pass),
        "worst_fp8": worst_fp8,
        "worst_fixed": worst_fixed,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"[B0 simulate] wrote {output_json}")
    print(f"[B0 simulate] fp8_score_pass={global_fp8_pass}")
    print(f"[B0 simulate] fixed_score_pass={global_fixed_pass}")
    print(f"[B0 simulate] worst_fp8={worst_fp8}")
    print(f"[B0 simulate] worst_fixed={worst_fixed}")


if __name__ == "__main__":
    main()
