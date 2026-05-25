#!/usr/bin/env python3
"""Precompute fixed decoder skip-channel subsets for P2.

topk_l1 mode:
  Run the checkpoint encoder over the validation set, score each decoder skip
  channel by mean L1 magnitude, and keep the top channels within each skip
  group. The saved index order is group-wise ascending, which preserves the
  grouped pointwise layout used by P1.

first_n mode:
  Generate the same group-aware channel counts, taking the first channels in
  each group. This is intended as a sanity baseline.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_cache")

import torch
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from DCNN.datasets import create_torch_dataloader  # noqa: E402
from DCNN.trainer import DCNNLightningModule  # noqa: E402


def parse_kernel_num(text: str) -> list[int]:
    values = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("kernel-num must contain at least one integer")
    return values


def effective_groups(requested_groups: int, in_channels: int, out_channels: int) -> int:
    requested_groups = max(1, int(requested_groups))
    max_groups = math.gcd(in_channels, out_channels)
    for groups in range(min(requested_groups, max_groups), 0, -1):
        if max_groups % groups == 0:
            return groups
    return 1


def keep_count(full_channels: int, ratio: float, groups: int) -> int:
    if not (0.0 < ratio <= 1.0):
        raise ValueError(f"ratio must be in (0, 1], got {ratio}")
    if full_channels % groups != 0:
        raise ValueError(f"groups={groups} does not divide channels={full_channels}")
    if ratio >= 1.0:
        return full_channels
    group_size = full_channels // groups
    keep_per_group = max(1, min(group_size, int(round(group_size * ratio))))
    return keep_per_group * groups


def first_n_indices(full_channels: int, ratio: float, groups: int) -> torch.Tensor:
    keep = keep_count(full_channels, ratio, groups)
    if keep == full_channels:
        return torch.arange(full_channels, dtype=torch.long)
    group_size = full_channels // groups
    keep_per_group = keep // groups
    pieces = []
    for group_idx in range(groups):
        start = group_idx * group_size
        pieces.append(torch.arange(start, start + keep_per_group, dtype=torch.long))
    return torch.cat(pieces, dim=0)


def topk_groupwise_indices(scores: torch.Tensor, ratio: float, groups: int) -> torch.Tensor:
    full_channels = int(scores.numel())
    keep = keep_count(full_channels, ratio, groups)
    if keep == full_channels:
        return torch.arange(full_channels, dtype=torch.long)
    group_size = full_channels // groups
    keep_per_group = keep // groups
    pieces = []
    for group_idx in range(groups):
        start = group_idx * group_size
        group_scores = scores[start:start + group_size]
        chosen_local = torch.topk(group_scores, keep_per_group, largest=True).indices
        chosen = (chosen_local + start).sort().values
        pieces.append(chosen.to(torch.long))
    return torch.cat(pieces, dim=0)


def decoder_stage_info(kernel_num: list[int], groups_skip: int):
    full_kernel_num = [2] + kernel_num
    info = []
    for stage, idx in enumerate(range(len(full_kernel_num) - 1, 0, -1)):
        full_skip = full_kernel_num[idx] // 2
        out_channels = full_kernel_num[idx - 1] // 2
        groups_eff = effective_groups(groups_skip, full_skip, out_channels)
        info.append({
            "stage": stage,
            "idx": idx,
            "full_skip_channels": full_skip,
            "out_channels": out_channels,
            "groups_skip_eff": groups_eff,
        })
    return info


def configure_model(args):
    cfg = OmegaConf.load(args.config)
    cfg.model.attention_softmax.mode = args.attention_softmax_mode
    cfg.model.decoder_grouped_pointwise.mode = "grouped"
    cfg.model.decoder_grouped_pointwise.groups_x = args.groups_x
    cfg.model.decoder_grouped_pointwise.groups_skip = args.groups_skip
    cfg.model.decoder_grouped_pointwise.skip_select_ratio = 1.0
    cfg.model.decoder_grouped_pointwise.skip_select_mode = "topk_l1"
    cfg.model.decoder_grouped_pointwise.skip_indices_path = None
    cfg.training.batch_size = args.batch_size
    cfg.training.n_workers = args.n_workers
    cfg.training.pin_memory = bool(args.pin_memory)
    cfg.training.qat_scope = args.qat_scope
    cfg.training.top_k_attention = args.top_k_attention
    return cfg


def load_lightning_model(args, device):
    cfg = configure_model(args)
    module = DCNNLightningModule(cfg)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    result = module.load_state_dict(checkpoint["state_dict"], strict=False)
    if result.missing_keys:
        print(f"[load] missing keys: {len(result.missing_keys)}; first={result.missing_keys[:5]}")
    if result.unexpected_keys:
        print(f"[load] unexpected keys: {len(result.unexpected_keys)}; first={result.unexpected_keys[:5]}")
    module = module.to(device)
    module.eval()
    return module


@torch.no_grad()
def collect_l1_scores(args, stage_info):
    device = torch.device(
        args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[precompute] device={device}")
    module = load_lightning_model(args, device)
    cfg = configure_model(args)
    loader = create_torch_dataloader(cfg, "validation")
    model = module.model

    sums = [
        torch.zeros(item["full_skip_channels"], dtype=torch.float64)
        for item in stage_info
    ]
    counts = [0 for _ in stage_info]

    for batch_idx, batch in enumerate(loader):
        if args.limit_batches and batch_idx >= args.limit_batches:
            break
        noisy = batch[0].to(device, non_blocking=True)
        cspecs_l = model.stft(noisy[:, 0])
        cspecs_r = model.stft(noisy[:, 1])
        enc_l = model.encoder(cspecs_l.unsqueeze(1))
        enc_r = model.encoder(cspecs_r.unsqueeze(1))

        for enc_out in (enc_l, enc_r):
            for stage, item in enumerate(stage_info):
                skip = enc_out[-1 - stage]
                if skip.shape[1] != item["full_skip_channels"]:
                    raise RuntimeError(
                        f"stage {stage}: skip channels={skip.shape[1]}, "
                        f"expected {item['full_skip_channels']}"
                    )
                sums[stage] += skip.detach().abs().sum(dim=(0, 2, 3)).double().cpu()
                counts[stage] += skip.shape[0] * skip.shape[2] * skip.shape[3]

        if (batch_idx + 1) % args.log_every == 0:
            print(f"[precompute] processed {batch_idx + 1} validation batches")

    if not any(counts):
        raise RuntimeError("no validation batches were processed")

    return [stage_sum / max(1, counts[stage]) for stage, stage_sum in enumerate(sums)]


def build_payload(args):
    stage_info = decoder_stage_info(args.kernel_num, args.groups_skip)
    if args.mode == "topk_l1":
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required for topk_l1")
        scores = collect_l1_scores(args, stage_info)
        indices = [
            topk_groupwise_indices(score, args.ratio, item["groups_skip_eff"])
            for score, item in zip(scores, stage_info)
        ]
    else:
        scores = [None for _ in stage_info]
        indices = [
            first_n_indices(item["full_skip_channels"], args.ratio, item["groups_skip_eff"])
            for item in stage_info
        ]

    layers = []
    for item, idx in zip(stage_info, indices):
        layer = dict(item)
        layer["keep_skip_channels"] = int(idx.numel())
        layers.append(layer)
        print(
            f"stage={item['stage']} idx={item['idx']} full_skip={item['full_skip_channels']} "
            f"keep={idx.numel()} out={item['out_channels']} groups_skip={item['groups_skip_eff']} "
            f"indices={idx.tolist()}"
        )

    return {
        "format": "decoder_skip_indices_v1",
        "mode": args.mode,
        "ratio": args.ratio,
        "groups_x": args.groups_x,
        "groups_skip": args.groups_skip,
        "kernel_num": args.kernel_num,
        "indices": indices,
        "scores": scores,
        "layers": layers,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode", choices=("topk_l1", "first_n"), default="topk_l1")
    parser.add_argument("--ratio", type=float, required=True)
    parser.add_argument("--groups_x", type=int, default=4)
    parser.add_argument("--groups_skip", type=int, default=2)
    parser.add_argument(
        "--kernel-num",
        type=parse_kernel_num,
        default=parse_kernel_num("16,32,64,128,256,256"),
    )
    parser.add_argument("--config", default=str(REPO / "config" / "config.yaml"))
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--n_workers", type=int, default=4)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--limit_batches", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--qat_scope", default="full")
    parser.add_argument("--top_k_attention", type=int, default=4)
    parser.add_argument(
        "--attention_softmax_mode",
        default="shift_topk_pow2_clamp8_nomantissa",
    )
    args = parser.parse_args()

    payload = build_payload(args)
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
