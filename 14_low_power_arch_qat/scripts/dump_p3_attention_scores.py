#!/usr/bin/env python3
"""Dump P3-B0 post-mask, pre-softmax attention scores.

The dump is intentionally offline: it loads the P2 checkpoint, runs a small
set of noisy utterances through the model, captures every ComplexMHA score
tensor after mask addition and before softmax, then writes raw row samples to
NPZ plus per-layer/per-head distribution and mask summaries to JSON.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra


REPO_DIR = Path(__file__).resolve().parents[1]
DEFAULT_P2_CKPT = (
    REPO_DIR
    / "outputs/p2_skip/s050_topk_2026-05-17_21-27-05"
    / "logs/lightning_logs/version_0/checkpoints/epoch=27-step=27104.ckpt"
)
DEFAULT_P2_INDEX = REPO_DIR / "Checkpoints/p2_skip_indices_g42_topk_l1_r050.pt"


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_") or "layer"


def _json_float(x):
    if x is None:
        return None
    x = float(x)
    if math.isnan(x) or math.isinf(x):
        return str(x)
    return x


def _parse_snr(path: Path):
    match = re.search(r"_snr([-+]?\d+(?:\.\d+)?)", path.stem)
    return float(match.group(1)) if match else None


def _select_noisy_files(noisy_dir: Path, limit: int) -> List[Path]:
    files = sorted(noisy_dir.glob("**/*.wav"))
    if limit <= 0 or len(files) <= limit:
        return files

    by_snr: Dict[float, List[Path]] = {}
    unknown: List[Path] = []
    for path in files:
        snr = _parse_snr(path)
        if snr is None:
            unknown.append(path)
        else:
            by_snr.setdefault(snr, []).append(path)

    selected: List[Path] = []
    buckets = [by_snr[k] for k in sorted(by_snr)] or [unknown]
    while len(selected) < limit and any(buckets):
        for bucket in buckets:
            if bucket and len(selected) < limit:
                selected.append(bucket.pop(0))
    return selected[:limit]


class AttentionScoreCollector:
    def __init__(self, max_rows_per_head: int, finite_mask_threshold: float, hist_bins: int):
        self.max_rows_per_head = int(max_rows_per_head)
        self.finite_mask_threshold = float(finite_mask_threshold)
        self.hist_bins = int(hist_bins)
        self.layers: Dict[str, Dict] = {}
        self.arrays: Dict[str, np.ndarray] = {}

    def make_capture(self, layer_name: str):
        def capture(scores: torch.Tensor, mask: torch.Tensor | None):
            self.capture(layer_name, scores, mask)

        return capture

    def capture(self, layer_name: str, scores: torch.Tensor, mask: torch.Tensor | None):
        scores_np = scores.detach().float().cpu().numpy()
        if scores_np.ndim != 4:
            return
        batch, heads, rows, cols = scores_np.shape
        layer = self.layers.setdefault(
            layer_name,
            {
                "slug": _slug(layer_name),
                "num_heads": int(heads),
                "score_keys": {str(h): [] for h in range(heads)},
                "rows_collected": {str(h): 0 for h in range(heads)},
                "mask_value_counts": {},
                "capture_shapes": [],
            },
        )
        layer["capture_shapes"].append([int(batch), int(heads), int(rows), int(cols)])

        if mask is not None:
            self._record_mask_values(layer, mask.detach().float().cpu().numpy())

        by_head = scores_np.transpose(1, 0, 2, 3).reshape(heads, batch * rows, cols)
        for head in range(heads):
            have = layer["rows_collected"][str(head)]
            remaining = self.max_rows_per_head - have
            if remaining <= 0:
                continue
            take = min(remaining, by_head.shape[1])
            if take <= 0:
                continue
            key = f"{layer['slug']}__head{head:02d}__chunk{len(layer['score_keys'][str(head)]):04d}"
            self.arrays[key] = by_head[head, :take].astype(np.float32, copy=False)
            layer["score_keys"][str(head)].append(key)
            layer["rows_collected"][str(head)] = have + int(take)

    def _record_mask_values(self, layer: Dict, mask_np: np.ndarray):
        flat = mask_np.reshape(-1)
        neg_inf = int(np.isneginf(flat).sum())
        pos_inf = int(np.isposinf(flat).sum())
        finite = flat[np.isfinite(flat)]
        if neg_inf:
            layer["mask_value_counts"]["-inf"] = layer["mask_value_counts"].get("-inf", 0) + neg_inf
        if pos_inf:
            layer["mask_value_counts"]["inf"] = layer["mask_value_counts"].get("inf", 0) + pos_inf
        if finite.size:
            unique, counts = np.unique(finite, return_counts=True)
            for value, count in zip(unique[:64], counts[:64]):
                key = f"{float(value):.8g}"
                layer["mask_value_counts"][key] = layer["mask_value_counts"].get(key, 0) + int(count)

    def summarize(self) -> Dict:
        layer_summaries = {}
        for layer_name, layer in self.layers.items():
            heads = {}
            for head_str, keys in layer["score_keys"].items():
                chunks = [self.arrays[k] for k in keys]
                heads[head_str] = self._summarize_head(chunks)
            mask_values = layer["mask_value_counts"]
            layer_summaries[layer_name] = {
                "num_heads": layer["num_heads"],
                "rows_collected": layer["rows_collected"],
                "capture_shapes": layer["capture_shapes"],
                "mask_value_counts": mask_values,
                "mask_classification": self._classify_mask_values(mask_values),
                "score_keys": layer["score_keys"],
                "heads": heads,
            }
        return {
            "finite_mask_threshold": self.finite_mask_threshold,
            "hist_bins": self.hist_bins,
            "layers": layer_summaries,
        }

    def _summarize_head(self, chunks: List[np.ndarray]) -> Dict:
        if not chunks:
            return {"rows": 0, "finite_regular_count": 0}
        values = []
        literal_inf = finite_mask_like = finite_regular = 0
        for chunk in chunks:
            flat = chunk.reshape(-1)
            finite = np.isfinite(flat)
            regular = finite & (flat > self.finite_mask_threshold)
            literal_inf += int(np.isinf(flat).sum())
            finite_mask_like += int((finite & ~regular).sum())
            finite_regular += int(regular.sum())
            if regular.any():
                values.append(flat[regular])
        if not values:
            return {
                "rows": int(sum(c.shape[0] for c in chunks)),
                "finite_regular_count": 0,
                "literal_inf_count": literal_inf,
                "finite_mask_like_count": finite_mask_like,
            }
        vals = np.concatenate(values).astype(np.float64, copy=False)
        percentiles = np.percentile(vals, [0.1, 1, 50, 99, 99.9])
        hist_counts, hist_edges = np.histogram(vals, bins=self.hist_bins)
        unique_vals = np.unique(vals)
        diffs = np.diff(unique_vals)
        positive_diffs = diffs[diffs > 0]
        min_step = float(positive_diffs.min()) if positive_diffs.size else None
        return {
            "rows": int(sum(c.shape[0] for c in chunks)),
            "finite_regular_count": int(finite_regular),
            "literal_inf_count": int(literal_inf),
            "finite_mask_like_count": int(finite_mask_like),
            "min": _json_float(vals.min()),
            "max": _json_float(vals.max()),
            "p0.1": _json_float(percentiles[0]),
            "p1": _json_float(percentiles[1]),
            "p50": _json_float(percentiles[2]),
            "p99": _json_float(percentiles[3]),
            "p99.9": _json_float(percentiles[4]),
            "min_resolvable_step": _json_float(min_step),
            "histogram": {
                "counts": hist_counts.astype(int).tolist(),
                "edges": [_json_float(x) for x in hist_edges.tolist()],
            },
        }

    def _classify_mask_values(self, counts: Dict[str, int]) -> Dict[str, str]:
        out = {}
        for value in counts:
            if value in ("-inf", "inf"):
                out[value] = "literal_inf"
                continue
            try:
                numeric = float(value)
            except ValueError:
                out[value] = "unknown"
                continue
            if numeric <= self.finite_mask_threshold:
                out[value] = "finite_very_negative"
            else:
                out[value] = "finite_regular"
        return out


def _load_model(args):
    os.environ.setdefault("PYTHONPATH", str(REPO_DIR))
    import sys

    if str(REPO_DIR) not in sys.path:
        sys.path.insert(0, str(REPO_DIR))

    from DCNN.trainer import DCNNLightningModule

    overrides = [
        "training.qat_scope=full",
        "training.top_k_attention=4",
        f"training.fp16_accumulator={str(args.fp16_accumulator).lower()}",
        f"model.attention_softmax.mode={args.softmax_mode}",
        "model.decoder_grouped_pointwise.mode=grouped",
        "model.decoder_grouped_pointwise.groups_x=4",
        "model.decoder_grouped_pointwise.groups_skip=2",
        "model.decoder_grouped_pointwise.skip_select_ratio=0.5",
        "model.decoder_grouped_pointwise.skip_select_mode=topk_l1",
        f"model.decoder_grouped_pointwise.skip_indices_path={args.p2_index}",
        "model.attention_datapath.enabled=false",
    ]
    overrides.extend(args.hydra_overrides or [])

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(REPO_DIR / "config"), version_base="1.1"):
        config = compose(config_name="config", overrides=overrides)

    model = DCNNLightningModule(config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model.model


def _install_collectors(model, collector: AttentionScoreCollector):
    from DCNN.utils.complexPyTorch.complexLayers import ComplexMultiheadAttention

    installed = []
    for name, module in model.named_modules():
        if isinstance(module, ComplexMultiheadAttention):
            cfg = dict(getattr(module, "attention_datapath", {}) or {})
            cfg["score_capture"] = collector.make_capture(name)
            module.attention_datapath = cfg
            installed.append(name)
    return installed


def _read_noisy(path: Path) -> torch.Tensor | None:
    data, _ = sf.read(path, always_2d=True)
    if data.shape[1] < 2:
        return None
    data = data[:, :2].astype(np.float32, copy=False)
    return torch.from_numpy(data.T).unsqueeze(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(DEFAULT_P2_CKPT))
    parser.add_argument("--p2_index", default=str(DEFAULT_P2_INDEX))
    parser.add_argument("--noisy_dir", default="/raid/R12K41024/BCCTN/Dataset/noisy_testset")
    parser.add_argument("--output_npz", default=str(REPO_DIR / "results/p3_attention_fp8/b0_attention_scores.npz"))
    parser.add_argument("--summary_json", default=None)
    parser.add_argument("--limit_utterances", type=int, default=8)
    parser.add_argument("--max_rows_per_head", type=int, default=1200)
    parser.add_argument("--hist_bins", type=int, default=256)
    parser.add_argument("--finite_mask_threshold", type=float, default=-10000.0)
    parser.add_argument("--softmax_mode", default="shift_topk_pow2_clamp8_nomantissa")
    parser.add_argument("--fp16_accumulator", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hydra_overrides", nargs="*", default=[])
    args = parser.parse_args()

    output_npz = Path(args.output_npz)
    summary_json = Path(args.summary_json) if args.summary_json else output_npz.with_suffix(".json")
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    model = _load_model(args).to(args.device)
    collector = AttentionScoreCollector(
        max_rows_per_head=args.max_rows_per_head,
        finite_mask_threshold=args.finite_mask_threshold,
        hist_bins=args.hist_bins,
    )
    installed = _install_collectors(model, collector)
    if not installed:
        raise RuntimeError("No ComplexMultiheadAttention modules found for B0 capture.")

    files = _select_noisy_files(Path(args.noisy_dir), args.limit_utterances)
    if not files:
        raise RuntimeError(f"No wav files found under {args.noisy_dir}")

    with torch.no_grad():
        for path in files:
            noisy = _read_noisy(path)
            if noisy is None:
                continue
            noisy = noisy.to(args.device)
            _ = model(noisy)

    summary = collector.summarize()
    summary.update(
        {
            "checkpoint": str(args.checkpoint),
            "p2_index": str(args.p2_index),
            "noisy_dir": str(args.noisy_dir),
            "files": [str(p) for p in files],
            "installed_layers": installed,
            "fp16_accumulator": bool(args.fp16_accumulator),
            "softmax_mode": args.softmax_mode,
            "output_npz": str(output_npz),
        }
    )

    np.savez_compressed(output_npz, **collector.arrays)
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[B0 dump] wrote {output_npz}")
    print(f"[B0 dump] wrote {summary_json}")
    for layer_name, layer in summary["layers"].items():
        min_rows = min(int(v) for v in layer["rows_collected"].values())
        print(f"[B0 dump] {layer_name}: min rows/head={min_rows}")


if __name__ == "__main__":
    main()
