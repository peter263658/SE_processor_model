#!/usr/bin/env python3
"""Convert a legacy decoder checkpoint to split/grouped pointwise keys.

The conversion keeps the legacy decoder block numbering:

    model.decoder.model.<stage>.0.*

Only decoder pointwise weights are replaced:

    pointwise_r.weight -> pointwise_r_x.weight + pointwise_r_skip.weight

If groups_x/groups_skip are greater than one, the split dense weights are
warm-started by taking the block diagonal expected by grouped Conv2d.
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


def grouped_block_diagonal(weight: torch.Tensor, groups: int) -> torch.Tensor:
    if groups == 1:
        return weight.clone()
    if weight.ndim != 4:
        raise ValueError(f"expected Conv2d weight with 4 dims, got shape={tuple(weight.shape)}")

    out_channels, in_channels, kh, kw = weight.shape
    if out_channels % groups != 0 or in_channels % groups != 0:
        raise ValueError(
            f"groups={groups} does not divide weight shape={tuple(weight.shape)}"
        )

    out_per_group = out_channels // groups
    in_per_group = in_channels // groups
    blocks = weight.reshape(groups, out_per_group, groups, in_per_group, kh, kw)
    diagonal = [blocks[group_idx, :, group_idx] for group_idx in range(groups)]
    return torch.cat(diagonal, dim=0).contiguous()


def find_weight_key(state_dict: OrderedDict, prefix: str, stem: str) -> tuple[str, str]:
    candidates = (
        f"{prefix}{stem}.parametrizations.weight.original",
        f"{prefix}{stem}.weight",
    )
    for key in candidates:
        if key in state_dict:
            return key, key[len(prefix + stem):]
    raise KeyError(f"could not find {prefix}{stem} weight key")


def convert_state_dict(
    state_dict: OrderedDict,
    kernel_num: list[int],
    groups_x: int,
    groups_skip: int,
) -> OrderedDict:
    full_kernel_num = [2] + kernel_num
    converted = OrderedDict()
    skip_keys: set[str] = set()
    new_weights: dict[str, torch.Tensor] = {}

    for stage, idx in enumerate(range(len(full_kernel_num) - 1, 0, -1)):
        in_x = full_kernel_num[idx] // 2
        in_skip = full_kernel_num[idx] // 2
        out_channels = full_kernel_num[idx - 1] // 2
        gx_eff = effective_groups(groups_x, in_x, out_channels)
        gs_eff = effective_groups(groups_skip, in_skip, out_channels)
        prefix = f"model.decoder.model.{stage}.0."

        for part in ("r", "i"):
            old_key, suffix = find_weight_key(state_dict, prefix, f"pointwise_{part}")
            old_weight = state_dict[old_key]
            expected_in = in_x + in_skip
            if old_weight.shape[1] != expected_in:
                raise ValueError(
                    f"{old_key} has input channels {old_weight.shape[1]}, "
                    f"expected {expected_in} for idx={idx}"
                )

            x_weight = old_weight[:, :in_x].clone()
            skip_weight = old_weight[:, in_x:in_x + in_skip].clone()
            new_weights[f"{prefix}pointwise_{part}_x{suffix}"] = grouped_block_diagonal(
                x_weight, gx_eff
            )
            new_weights[f"{prefix}pointwise_{part}_skip{suffix}"] = grouped_block_diagonal(
                skip_weight, gs_eff
            )
            skip_keys.add(old_key)

        print(
            f"stage={stage} idx={idx}: in_x={in_x} in_skip={in_skip} "
            f"out={out_channels} groups_x={gx_eff} groups_skip={gs_eff}"
        )

    for key, value in state_dict.items():
        if key not in skip_keys:
            converted[key] = value

    for key, value in new_weights.items():
        converted[key] = value
    return converted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="input_ckpt", required=True)
    parser.add_argument("--out", dest="output_ckpt", required=True)
    parser.add_argument("--groups_x", type=int, default=1)
    parser.add_argument("--groups_skip", type=int, default=1)
    parser.add_argument(
        "--kernel-num",
        type=parse_kernel_num,
        default=parse_kernel_num("16,32,64,128,256,256"),
        help="comma-separated model.kernel_num without the leading input channel",
    )
    args = parser.parse_args()

    input_path = Path(args.input_ckpt)
    output_path = Path(args.output_ckpt)
    checkpoint = torch.load(input_path, map_location="cpu")
    if "state_dict" in checkpoint:
        checkpoint["state_dict"] = convert_state_dict(
            checkpoint["state_dict"], args.kernel_num, args.groups_x, args.groups_skip
        )
        output_obj = checkpoint
    else:
        output_obj = convert_state_dict(
            checkpoint, args.kernel_num, args.groups_x, args.groups_skip
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_obj, output_path)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
