#!/usr/bin/env python3
"""Generate FP16 patterns for Softmax_v3.sv.

The golden model matches the RTL primary mode from
RTL_HANDOFF_SOFTMAX_2026-05-14.md:

    score-domain top-k=4
    -> fixed-point 1.5x, clamp [-8, 0], floor
    -> Q0.8 pow2 weights
    -> nearest-pow2 denominator
    -> dense sparse-vector scatter
"""

from __future__ import annotations

import argparse
import math
import random
import struct
from pathlib import Path


SEQ_LEN = 32
TOP_K = 4
PROB_FRAC = 8
SCORE_Q_FRAC = 4
SCORE_Q_WIDTH = 16


def fp16_bits(value: float) -> int:
    return int.from_bytes(struct.pack(">e", float(value)), byteorder="big")


def fp16_hex(value: float) -> str:
    return f"{fp16_bits(value):04x}"


def bits_to_fp16_float(bits: int) -> float:
    return struct.unpack(">e", bits.to_bytes(2, byteorder="big"))[0]


def round_fp16(value: float) -> float:
    return bits_to_fp16_float(fp16_bits(value))


def fp16_bits_to_sq4(bits: int) -> int:
    sign = (bits >> 15) & 0x1
    exp = (bits >> 10) & 0x1F
    frac = bits & 0x3FF
    max_val = (1 << (SCORE_Q_WIDTH - 1)) - 1
    min_val = -(1 << (SCORE_Q_WIDTH - 1))

    if exp == 0:
        scaled = 0
    elif exp == 0x1F:
        scaled = min_val if sign else max_val
    else:
        mant = (1 << 10) | frac
        shift_amt = (exp - 15) + SCORE_Q_FRAC - 10
        if shift_amt >= 0:
            scaled = mant << shift_amt
        else:
            scaled = mant >> (-shift_amt)
        if sign:
            scaled = -scaled

    return max(min(scaled, max_val), min_val)


def round_log2_pow2_exp(value: int) -> int:
    # nearest pow2 in the LOG domain (deploy: 2^round(log2(sum))): round up iff
    # value >= 2^(msb+0.5) <=> value^2 >= 2^(2msb+1). The old next-bit rule was the
    # ARITHMETIC midpoint (1.5*2^msb) and diverges from the deploy for mantissa in
    # [sqrt2, 1.5) — exposed by the t*=10 warm-up golden (wsum q8=368 -> denom 512).
    if value <= 0:
        return 0

    msb = value.bit_length() - 1
    exp = msb
    if value * value >= (1 << (2 * msb + 1)):
        exp += 1
    return exp


def q8_to_fp16_hex(q: int) -> str:
    return fp16_hex(q / float(1 << PROB_FRAC))


def shift_topk_pow2_clamp8_nomantissa(
    score_bits: list[int], mask: list[bool]
) -> list[int]:
    out_q = [0 for _ in range(SEQ_LEN)]
    valid_indices = [idx for idx, valid in enumerate(mask) if valid]
    if not valid_indices:
        return out_q

    scores = [bits_to_fp16_float(bits) for bits in score_bits]
    top_idx = sorted(valid_indices, key=lambda idx: (-scores[idx], idx))[:TOP_K]
    row_max = scores[top_idx[0]]

    # fp16-delta (BIT-EXACT to deploy _shift_topk_softmax + Softmax_v3.sv): centered delta and 1.5x are
    # computed in fp16 (NOT the old sq4 Q?.4, which collapsed near-max ties). The 1.5x is the fp16 RNE
    # add x + x/2 (round to fp16 BEFORE floor), NOT a wide-exact shift-add.
    weights: list[tuple[int, int]] = []
    for idx in top_idx:
        x = round_fp16(scores[idx] - row_max)            # fp16 sub (Sterbenz: exact for nearby scores)
        half = round_fp16(x * 0.5)                        # x/2 (exponent decrement; exact for normals)
        scaled = round_fp16(x + half)                     # RNE_fp16(1.5x)
        scaled = max(-8.0, min(0.0, scaled))              # clamp [-8, 0]
        k_i = max(-8, min(0, math.floor(scaled)))         # floor toward -inf
        weight_q = 1 << (PROB_FRAC + k_i)
        weights.append((idx, weight_q))

    denom_q = sum(weight for _, weight in weights)
    denom_exp = round_log2_pow2_exp(denom_q)

    for idx, weight_q in weights:
        prob_q = (weight_q << PROB_FRAC) >> denom_exp
        prob_q = min(prob_q, 1 << PROB_FRAC)
        out_q[idx] = prob_q

    return out_q


def build_cases(num_random: int, seed: int) -> list[tuple[list[int], list[bool], list[int], str]]:
    rng = random.Random(seed)
    cases: list[tuple[list[int], list[bool], list[int], str]] = []

    fixed_specs = [
        (32, -6.0, 6.0, "full_uniform_wide"),
        (32, -1.0, 1.0, "full_uniform_small"),
        (32, -12.0, -1.0, "full_all_negative"),
        (32, 1.0, 12.0, "full_all_positive"),
        (1, -2.0, 2.0, "early_valid_1"),
        (2, -2.0, 2.0, "early_valid_2"),
        (3, -2.0, 2.0, "early_valid_3"),
        (4, -2.0, 2.0, "early_valid_4"),
    ]

    for valid_count, lo, hi, name in fixed_specs:
        scores = [round_fp16(rng.uniform(lo, hi)) for _ in range(SEQ_LEN)]
        bits = [fp16_bits(value) for value in scores]
        mask = [idx < valid_count for idx in range(SEQ_LEN)]
        golden_q = shift_topk_pow2_clamp8_nomantissa(bits, mask)
        cases.append((bits, mask, golden_q, name))

    for idx in range(num_random):
        valid_count = rng.randint(1, SEQ_LEN)
        scores = [round_fp16(rng.gauss(0.0, 2.5)) for _ in range(SEQ_LEN)]
        bits = [fp16_bits(value) for value in scores]
        mask = [lane < valid_count for lane in range(SEQ_LEN)]
        golden_q = shift_topk_pow2_clamp8_nomantissa(bits, mask)
        cases.append((bits, mask, golden_q, f"random_{idx:03d}_valid_{valid_count:02d}"))

    return cases


def write_patterns(out_dir: Path, cases: list[tuple[list[int], list[bool], list[int], str]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "data.mem").open("w", encoding="ascii") as f_data, (
        out_dir / "mask.mem"
    ).open("w", encoding="ascii") as f_mask, (out_dir / "golden.mem").open(
        "w", encoding="ascii"
    ) as f_golden, (out_dir / "pattern_report.txt").open("w", encoding="ascii") as f_report:
        f_report.write("Softmax_v3 shift-topk pattern report\n")
        f_report.write(f"num_cases={len(cases)} seq_len={SEQ_LEN} top_k={TOP_K}\n\n")

        for case_id, (bits, mask, golden_q, name) in enumerate(cases):
            f_report.write(f"case {case_id}: {name}\n")
            f_report.write("idx mask input_hex input_float golden_hex golden_float\n")
            f_mask.write("".join("1" if mask[idx] else "0" for idx in reversed(range(SEQ_LEN))) + "\n")

            for idx in range(SEQ_LEN):
                data_hex = f"{bits[idx]:04x}"
                golden_hex = q8_to_fp16_hex(golden_q[idx])
                f_data.write(f"{data_hex}\n")
                f_golden.write(f"{golden_hex}\n")
                f_report.write(
                    f"{idx:02d} {int(mask[idx])} {data_hex} "
                    f"{bits_to_fp16_float(bits[idx]): .8e} "
                    f"{golden_hex} {bits_to_fp16_float(int(golden_hex, 16)): .8e}\n"
                )
            f_report.write("\n")

    legacy = {
        "data.dat": "data.mem",
        "golden.dat": "golden.mem",
        "mask.dat": "mask.mem",
    }
    for old_name, new_name in legacy.items():
        (out_dir / old_name).write_text(
            (out_dir / new_name).read_text(encoding="ascii"), encoding="ascii"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("pattern"))
    parser.add_argument("--num-random", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cases = build_cases(args.num_random, args.seed)
    write_patterns(args.out_dir, cases)
    print(f"Wrote {len(cases)} Softmax_v3 shift-topk cases to {args.out_dir}")


if __name__ == "__main__":
    main()
