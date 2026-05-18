#!/usr/bin/env python3
"""Convert a P1 split/grouped decoder checkpoint into a P2 skip-subset checkpoint.

Only decoder pointwise skip-branch weights change shape:

    pointwise_{r,i}_skip: [out, full_skip/groups, 1, 1]
        -> [out, keep_skip/groups, 1, 1]

The x branch, depthwise transpose, BN/PReLU, encoder, and MHA weights are kept
unchanged. Skip indices are copied into the checkpoint as decoder buffers so
strict eval loading sees the same model state as training.
"""

from __future__ import annotations

import argparse
import math
from collections import OrderedDict
from pathlib import Path

import torch


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


def find_weight_key(state_dict: OrderedDict, prefix: str, stem: str) -> tuple[str, str]:
    candidates = (
        f"{prefix}{stem}.parametrizations.weight.original",
        f"{prefix}{stem}.weight",
    )
    for key in candidates:
        if key in state_dict:
            return key, key[len(prefix + stem):]
    raise KeyError(f"could not find {prefix}{stem} weight key")


def stage_indices_from_payload(payload, stage: int) -> torch.Tensor:
    if isinstance(payload, (list, tuple)):
        return torch.as_tensor(payload[stage], dtype=torch.long).flatten()
    if isinstance(payload, dict):
        if "indices" in payload:
            return stage_indices_from_payload(payload["indices"], stage)
        for key in (stage, str(stage), f"stage_{stage}", f"decoder_stage_{stage}"):
            if key in payload:
                return torch.as_tensor(payload[key], dtype=torch.long).flatten()
    raise KeyError(f"could not find skip indices for decoder stage {stage}")


def validate_groupwise_indices(
    indices: torch.Tensor,
    full_channels: int,
    ratio: float,
    groups: int,
    stage: int,
) -> torch.Tensor:
    keep = keep_count(full_channels, ratio, groups)
    if indices.numel() != keep:
        raise ValueError(
            f"stage={stage}: got {indices.numel()} indices, expected {keep} "
            f"for full_channels={full_channels}, ratio={ratio}, groups={groups}"
        )
    if indices.min().item() < 0 or indices.max().item() >= full_channels:
        raise ValueError(f"stage={stage}: indices out of range 0..{full_channels - 1}")
    if indices.unique().numel() != indices.numel():
        raise ValueError(f"stage={stage}: duplicate indices")
    if groups > 1:
        group_size = full_channels // groups
        keep_per_group = keep // groups
        for group_idx in range(groups):
            chunk = indices[group_idx * keep_per_group:(group_idx + 1) * keep_per_group]
            lo = group_idx * group_size
            hi = lo + group_size
            if not torch.all((chunk >= lo) & (chunk < hi)):
                raise ValueError(
                    f"stage={stage}: indices must be stored group-wise; "
                    f"group {group_idx} expected [{lo}, {hi})"
                )
    return indices


def subset_grouped_weight(
    weight: torch.Tensor,
    indices: torch.Tensor,
    full_in_channels: int,
    old_groups: int,
) -> torch.Tensor:
    if weight.ndim != 4:
        raise ValueError(f"expected Conv2d weight with 4 dims, got {tuple(weight.shape)}")
    out_channels, stored_in_channels, kh, kw = weight.shape
    if old_groups == 1:
        if stored_in_channels != full_in_channels:
            raise ValueError(
                f"dense skip weight input={stored_in_channels}, expected {full_in_channels}"
            )
        return weight.index_select(1, indices.to(weight.device)).contiguous()

    if out_channels % old_groups != 0 or full_in_channels % old_groups != 0:
        raise ValueError(
            f"old_groups={old_groups} invalid for weight={tuple(weight.shape)}, "
            f"full_in={full_in_channels}"
        )
    full_in_per_group = full_in_channels // old_groups
    if stored_in_channels != full_in_per_group:
        raise ValueError(
            f"grouped skip weight stores {stored_in_channels} inputs/group, "
            f"expected {full_in_per_group}"
        )

    keep = int(indices.numel())
    if keep % old_groups != 0:
        raise ValueError(f"keep={keep} must divide old_groups={old_groups}")
    keep_per_group = keep // old_groups
    out_per_group = out_channels // old_groups
    blocks = weight.reshape(old_groups, out_per_group, full_in_per_group, kh, kw)
    pieces = []
    for group_idx in range(old_groups):
        chunk = indices[group_idx * keep_per_group:(group_idx + 1) * keep_per_group]
        local = chunk - group_idx * full_in_per_group
        pieces.append(blocks[group_idx].index_select(1, local.to(weight.device)))
    return torch.cat(pieces, dim=0).contiguous()


def detect_state_prefix(state_dict: OrderedDict) -> str:
    if any(key.startswith("model.decoder.") for key in state_dict):
        return "model."
    if any(key.startswith("decoder.") for key in state_dict):
        return ""
    raise KeyError("checkpoint does not look like a DCNN/BinauralAttentionDCNN state_dict")


def convert_state_dict(
    state_dict: OrderedDict,
    skip_payload,
    kernel_num: list[int],
    ratio: float,
    groups_x: int,
    groups_skip: int,
) -> OrderedDict:
    full_kernel_num = [2] + kernel_num
    state_prefix = detect_state_prefix(state_dict)
    converted = OrderedDict(state_dict)

    for stage, idx in enumerate(range(len(full_kernel_num) - 1, 0, -1)):
        full_skip = full_kernel_num[idx] // 2
        out_channels = full_kernel_num[idx - 1] // 2
        old_groups = effective_groups(groups_skip, full_skip, out_channels)
        raw_indices = stage_indices_from_payload(skip_payload, stage)
        indices = validate_groupwise_indices(raw_indices, full_skip, ratio, old_groups, stage)
        keep_skip = int(indices.numel())
        new_groups = effective_groups(groups_skip, keep_skip, out_channels)
        if new_groups != old_groups:
            raise ValueError(
                f"stage={stage}: new groups {new_groups} differ from P1 groups {old_groups}; "
                "choose a group-aware ratio/index file"
            )

        prefix = f"{state_prefix}decoder.model.{stage}.0."
        for part in ("r", "i"):
            key, _suffix = find_weight_key(converted, prefix, f"pointwise_{part}_skip")
            converted[key] = subset_grouped_weight(
                converted[key], indices, full_skip, old_groups
            )

        buffer_key = f"{state_prefix}decoder.skip_indices_{stage}"
        converted[buffer_key] = indices.clone()
        print(
            f"stage={stage} idx={idx}: full_skip={full_skip} keep_skip={keep_skip} "
            f"out={out_channels} groups_skip={old_groups}"
        )

    return converted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="input_ckpt", required=True)
    parser.add_argument("--out", dest="output_ckpt", required=True)
    parser.add_argument("--skip-indices", required=True)
    parser.add_argument("--ratio", type=float, required=True)
    parser.add_argument("--groups_x", type=int, default=4)
    parser.add_argument("--groups_skip", type=int, default=2)
    parser.add_argument(
        "--kernel-num",
        type=parse_kernel_num,
        default=parse_kernel_num("16,32,64,128,256,256"),
    )
    args = parser.parse_args()

    input_path = Path(args.input_ckpt)
    output_path = Path(args.output_ckpt)
    checkpoint = torch.load(input_path, map_location="cpu")
    skip_payload = torch.load(args.skip_indices, map_location="cpu")

    if "state_dict" in checkpoint:
        checkpoint["state_dict"] = convert_state_dict(
            checkpoint["state_dict"],
            skip_payload,
            args.kernel_num,
            args.ratio,
            args.groups_x,
            args.groups_skip,
        )
        output_obj = checkpoint
    else:
        output_obj = convert_state_dict(
            checkpoint,
            skip_payload,
            args.kernel_num,
            args.ratio,
            args.groups_x,
            args.groups_skip,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_obj, output_path)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
