#!/usr/bin/env python3
"""Low-power architecture accounting for the 2026-05-19 future-work spec."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


BASE_MACS = {
    "encoder_conv": 202_880,
    "decoder_conv_p2": 122_624,
    "attn_input_proj": 262_144,
    "attn_qkv_tied_2pass": 393_216,
    "attn_out_proj_tied_2pass": 131_072,
    "attn_output_proj": 262_144,
    "qk_scorev": 18_432,
}


def grouped_attention_macs(input_groups, qkv_groups, out_groups, output_groups):
    return {
        "attn_input_proj": math.ceil(BASE_MACS["attn_input_proj"] / input_groups),
        "attn_qkv_tied_2pass": math.ceil(BASE_MACS["attn_qkv_tied_2pass"] / qkv_groups),
        "attn_out_proj_tied_2pass": math.ceil(BASE_MACS["attn_out_proj_tied_2pass"] / out_groups),
        "attn_output_proj": math.ceil(BASE_MACS["attn_output_proj"] / output_groups),
    }


def cycles_for(macs, lanes, utilization):
    return math.ceil(macs / max(1.0, lanes * utilization))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_groups", type=int, default=1)
    parser.add_argument("--qkv_groups", type=int, default=1)
    parser.add_argument("--out_groups", type=int, default=1)
    parser.add_argument("--output_groups", type=int, default=1)
    parser.add_argument("--lanes", type=int, nargs="+", default=[64, 96, 128, 256])
    parser.add_argument("--clocks_mhz", type=float, nargs="+", default=[5, 10, 20])
    parser.add_argument("--utilization", type=float, default=0.85)
    parser.add_argument("--output_json", default=None)
    args = parser.parse_args()

    attn = grouped_attention_macs(
        args.input_groups, args.qkv_groups, args.out_groups, args.output_groups
    )
    mixed_macs = (
        BASE_MACS["encoder_conv"]
        + BASE_MACS["decoder_conv_p2"]
        + sum(attn.values())
    )
    total_macs = mixed_macs + BASE_MACS["qk_scorev"]

    packet_bits = {
        "q08_top4_packet_bits_per_row": 4 * (5 + 8),
        "q08_dense_bits_per_row": 32 * 8,
        "fp16_dense_bits_per_row": 32 * 16,
    }
    weight_encoding = {
        "dense_w4_plus_fp8_scale_bits_per_weight": 4.25,
        "block_mask_16_break_even_max_nonzeros": 11,
        "block_mask_16_break_even_sparsity_gt": 0.25,
        "scalar_value_index_4b_value_8b_index_sparsity_gt": 0.65,
    }

    operating_points = []
    for lanes in args.lanes:
        cycles = cycles_for(total_macs, lanes, args.utilization)
        for clock in args.clocks_mhz:
            budget = int(clock * 5_000)
            operating_points.append(
                {
                    "lanes": lanes,
                    "clock_mhz": clock,
                    "cycles_per_frame": cycles,
                    "frame_budget_cycles": budget,
                    "duty_cycle": cycles / budget,
                    "meets_5ms_hop": cycles <= budget,
                }
            )

    report = {
        "groups": {
            "input_proj": args.input_groups,
            "qkv": args.qkv_groups,
            "out_proj": args.out_groups,
            "output_proj": args.output_groups,
        },
        "macs": {
            "encoder_conv": BASE_MACS["encoder_conv"],
            "decoder_conv_p2": BASE_MACS["decoder_conv_p2"],
            **attn,
            "qk_scorev": BASE_MACS["qk_scorev"],
            "mixed_pe_macs": mixed_macs,
            "total_mac_equivalent": total_macs,
        },
        "packet_bits": packet_bits,
        "weight_encoding": weight_encoding,
        "utilization_assumption": args.utilization,
        "operating_points": operating_points,
    }

    print(json.dumps(report, indent=2))
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
