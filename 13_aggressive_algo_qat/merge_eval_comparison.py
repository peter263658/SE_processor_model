#!/usr/bin/env python3
"""
Aggregate paper-style eval CSVs from multiple variants (08 PTQ runs + 09 tied
fine-tune + 09 tied scratch) into one comparison table.

Per-SNR metrics surfaced:
    MBSTOI       : speech intelligibility (higher better)
    ΔSegSNR      : SegSNR improvement (higher better)
    L_ILD        : ILD error (lower better)
    L_IPD (deg)  : IPD error in degrees (lower better)

Usage:
    python merge_eval_comparison.py \
        --dataset vctk \
        --outfile compare_table_vctk.csv \
        baseline=/raid/.../08_mixed_precision_qat_from_scratch/results_May21/vctk_paper_style/vctk_paper_style_results.csv \
        09_tied_ft=/raid/.../09_weight_tied_mha/results/finetune/vctk_paper_style_results.csv \
        09_tied_sc=/raid/.../09_weight_tied_mha/results/scratch/vctk_paper_style_results.csv

Each positional arg is `<label>=<csv_path>`. Order is preserved in the output.
"""
from __future__ import annotations
import argparse
import math
from pathlib import Path
import sys

import pandas as pd


METRICS = [
    ("MBSTOI",      "MBSTOI",      lambda v: v),                       # higher better
    ("DeltaSegSNR", "SegSNR_impr", lambda v: v),                       # higher better
    ("L_ILD",       "L_ILD",       lambda v: v),                       # lower better
    # eval.py writes L_IPD already in degrees (paper_style CSV uses "deg" lines).
    # No conversion needed.
    ("L_IPD_deg",   "L_IPD",       lambda v: v),                       # lower better
]


def parse_kv(arg: str) -> tuple[str, Path]:
    if "=" not in arg:
        raise argparse.ArgumentTypeError(
            f"expected <label>=<csv_path>, got {arg!r}"
        )
    label, path = arg.split("=", 1)
    return label, Path(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("vctk", "timit"), default="vctk",
                    help="Used as a label only.")
    ap.add_argument("--outfile", type=Path, default=None,
                    help="Optional output CSV (long format).")
    ap.add_argument("variants", nargs="+", type=parse_kv,
                    help="<label>=<csv_path>, repeatable.")
    args = ap.parse_args()

    # ── Load all variants ──
    frames: dict[str, pd.DataFrame] = {}
    for label, path in args.variants:
        if not path.exists():
            print(f"[skip] {label}: {path} (file missing)", file=sys.stderr)
            continue
        df = pd.read_csv(path)
        if "SNR" not in df.columns:
            print(f"[skip] {label}: {path} (no SNR column)", file=sys.stderr)
            continue
        frames[label] = df

    if not frames:
        sys.exit("No usable variant CSVs found.")

    # ── Wide table: rows = (SNR, Metric), cols = variants ──
    snrs = sorted({s for df in frames.values() for s in df["SNR"].tolist()})

    rows = []
    for snr in snrs:
        for metric_label, source_col, transform in METRICS:
            row = {"SNR": snr, "Metric": metric_label}
            for label, df in frames.items():
                sub = df[df["SNR"] == snr]
                if sub.empty or source_col not in sub.columns:
                    row[label] = None
                else:
                    val = float(sub.iloc[0][source_col])
                    row[label] = transform(val)
            rows.append(row)

    table = pd.DataFrame(rows)

    # ── Pretty print ──
    pd.options.display.float_format = lambda v: f"{v:.4f}" if v is not None and abs(v) < 100 else f"{v:.2f}"
    title = f"Paper-style eval comparison — {args.dataset.upper()}"
    print(title)
    print("=" * len(title))
    print(table.to_string(index=False))

    if args.outfile:
        args.outfile.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.outfile, index=False)
        print(f"\nSaved: {args.outfile}")


if __name__ == "__main__":
    main()
