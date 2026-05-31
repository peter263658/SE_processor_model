#!/usr/bin/env python3
"""Verify R0 decoder split equivalence.

This is a unit-style FP32 reference check, not a metric eval. It strips QAT
parametrization keys from checkpoints before loading so activation fake-quant
hooks do not introduce branch-specific quantization differences.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path

import torch
from omegaconf import OmegaConf

from DCNN.models.binaural_attention_model import BinauralAttentionDCNN


DEFAULT_BASELINE_CKPT = (
    "outputs/finetune_strict/2026-05-10_02-40-16/"
    "logs/lightning_logs/version_0/checkpoints/epoch=45-step=44528.ckpt"
)
DEFAULT_SPLIT_CKPT = (
    "outputs/finetune_strict/2026-05-10_02-40-16/"
    "logs/lightning_logs/version_0/checkpoints/epoch=45-step=44528_split_g11.ckpt"
)


def checkpoint_to_raw_model_state(path: Path) -> OrderedDict:
    checkpoint = torch.load(path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    has_lightning_model_prefix = any(key.startswith("model.") for key in state_dict)
    raw = OrderedDict()
    for key, value in state_dict.items():
        if has_lightning_model_prefix:
            if not key.startswith("model."):
                continue
            key = key[len("model."):]
        key = key.replace(".parametrizations.weight.original", ".weight")
        key = key.replace(".parametrizations.in_proj_weight.original", ".in_proj_weight")
        raw[key] = value
    return raw


def load_model(config_path: Path, ckpt_path: Path, split_mode: str) -> BinauralAttentionDCNN:
    cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    model_cfg = dict(cfg["model"])
    model_cfg["attention_softmax"] = {"mode": "softmax"}
    model_cfg["decoder_grouped_pointwise"] = {
        "mode": split_mode,
        "groups_x": 1,
        "groups_skip": 1,
        "channel_shuffle": False,
        "skip_select_ratio": 1.0,
        "skip_select_mode": "topk_l1",
    }
    model = BinauralAttentionDCNN(**model_cfg)
    missing, unexpected = model.load_state_dict(
        checkpoint_to_raw_model_state(ckpt_path), strict=False
    )
    significant_missing = [key for key in missing if "num_batches_tracked" not in key]
    if significant_missing or unexpected:
        print("missing keys:", significant_missing)
        print("unexpected keys:", unexpected)
        raise RuntimeError(f"failed to load {ckpt_path}")
    model.eval()
    return model


def capture_decoder_outputs(model: BinauralAttentionDCNN):
    outputs = []
    handles = []

    def hook(_module, _inputs, output):
        outputs.append(output.detach().cpu())

    for block in model.decoder.model:
        handles.append(block.register_forward_hook(hook))
    return outputs, handles


def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--baseline-ckpt", default=DEFAULT_BASELINE_CKPT)
    parser.add_argument("--split-ckpt", default=DEFAULT_SPLIT_CKPT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-random", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--samples", type=int, default=16000)
    parser.add_argument("--tol", type=float, default=1e-5)
    args = parser.parse_args()

    torch.manual_seed(0)
    device = torch.device(args.device)
    baseline = load_model(Path(args.config), Path(args.baseline_ckpt), "off").to(device)
    split = load_model(Path(args.config), Path(args.split_ckpt), "split").to(device)

    worst_final = 0.0
    worst_decoder = 0.0
    for batch_idx in range(args.num_random):
        inputs = torch.randn(args.batch_size, 2, args.samples, device=device)
        baseline_outputs, baseline_handles = capture_decoder_outputs(baseline)
        split_outputs, split_handles = capture_decoder_outputs(split)
        with torch.no_grad():
            baseline_wav = baseline(inputs)
            split_wav = split(inputs)
        for handle in baseline_handles + split_handles:
            handle.remove()

        final_diff = max_abs_diff(baseline_wav.detach().cpu(), split_wav.detach().cpu())
        decoder_diffs = [
            max_abs_diff(a, b) for a, b in zip(baseline_outputs, split_outputs)
        ]
        batch_decoder = max(decoder_diffs) if decoder_diffs else 0.0
        worst_final = max(worst_final, final_diff)
        worst_decoder = max(worst_decoder, batch_decoder)
        print(
            f"batch={batch_idx} final_diff={final_diff:.3e} "
            f"decoder_max={batch_decoder:.3e}"
        )

    print(
        f"[decoder_split exact, groups=1] final_wav_max={worst_final:.3e} "
        f"decoder_max={worst_decoder:.3e}"
    )
    if worst_final > args.tol:
        raise SystemExit(f"FAIL: final wav diff {worst_final:.3e} > tol {args.tol:.3e}")
    print("[decoder_split exact, groups=1] PASS")


if __name__ == "__main__":
    main()
