#!/usr/bin/env python3
"""
Build a paper_style_results.csv directly from per-SNR `average_results.txt`
files. Avoids re-running eval when per-SNR data is already on disk (e.g. when
a VCTK eval finished but the run was killed before TIMIT completed and the
final consolidated CSV could be written).

Usage:
    python aggregate_per_snr_to_paper_csv.py \
        --per_snr_root results/scratch/vctk \
        --output      results/scratch/vctk_paper_style_results.csv
"""
from __future__ import annotations
import argparse
import csv
import re
from pathlib import Path
import sys


COLUMN_MAP = {
    "MBSTOI_noisy"  : "Average MBSTOI Noisy",
    "MBSTOI"        : "Average MBSTOI",
    "SegSNR_noisy"  : "Average SegSNR Noisy",
    "SegSNR"        : "Average SegSNR",
    "SegSNR_impr"   : "Average SegSNR Improvement",
    "L_ILD_noisy"   : "Average ILD Error Noisy (dB)",
    "L_ILD"         : "Average ILD Error (dB)",
    # Existing 08 paper_style CSVs use degrees in `L_IPD_noisy` and `L_IPD`,
    # so we read the "deg" line from average_results.txt.
    "L_IPD_noisy"   : "Average IPD Error Noisy (deg)",
    "L_IPD"         : "Average IPD Error (deg)",
}
PAPER_COLS = ["SNR"] + list(COLUMN_MAP.keys())


def parse_average_txt(path: Path) -> dict:
    """Parse `average_results.txt` -> {label: value}.
    Each line looks like:  "Average MBSTOI Noisy: 0.5366"
    """
    out = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if ":" not in line:
                continue
            key, _, val = line.partition(":")
            try:
                out[key.strip()] = float(val.strip())
            except ValueError:
                continue
    return out


def snr_from_dirname(name: str) -> int | None:
    m = re.match(r"snr_(-?\d+)dB$", name)
    return int(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_snr_root", required=True, type=Path,
                    help="Folder containing snr_*dB/ subfolders (e.g. results/scratch/vctk)")
    ap.add_argument("--output", required=True, type=Path,
                    help="Output paper_style CSV path")
    args = ap.parse_args()

    if not args.per_snr_root.is_dir():
        sys.exit(f"Not a directory: {args.per_snr_root}")

    rows = []
    for snr_dir in sorted(args.per_snr_root.iterdir(), key=lambda p: snr_from_dirname(p.name) or 999):
        snr = snr_from_dirname(snr_dir.name)
        if snr is None:
            continue
        avg_txt = snr_dir / "average_results.txt"
        if not avg_txt.is_file():
            print(f"  [skip] {snr_dir}: no average_results.txt")
            continue
        parsed = parse_average_txt(avg_txt)
        row = {"SNR": snr}
        missing = []
        for paper_col, source_label in COLUMN_MAP.items():
            if source_label in parsed:
                row[paper_col] = parsed[source_label]
            else:
                missing.append(source_label)
                row[paper_col] = ""
        if missing:
            print(f"  [warn] {snr_dir.name}: missing {missing}")
        rows.append(row)

    if not rows:
        sys.exit("No SNR rows assembled.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PAPER_COLS)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"Wrote {len(rows)} SNR rows -> {args.output}")

    # ── Also write the *_averages.txt sibling file (matches eval.py format). ──
    def col_mean(col):
        vals = [r[col] for r in rows if isinstance(r.get(col), (int, float))]
        return sum(vals) / len(vals) if vals else float("nan")

    avgs = {k: col_mean(k) for k in COLUMN_MAP.keys()}

    avg_path = args.output.with_name(args.output.stem + "_averages.txt")
    dataset_name = args.output.stem.split("_")[0].upper()  # "vctk_paper_style_results" -> "VCTK"
    with avg_path.open("w") as f:
        f.write(f"===== {dataset_name} PAPER STYLE AVERAGES =====\n")
        f.write(f"Average MBSTOI Noisy: {avgs['MBSTOI_noisy']:.4f}\n")
        f.write(f"Average MBSTOI Enhanced: {avgs['MBSTOI']:.4f}\n")
        f.write(f"Average SegSNR Noisy: {avgs['SegSNR_noisy']:.2f}\n")
        f.write(f"Average SegSNR Enhanced: {avgs['SegSNR']:.2f}\n")
        f.write(f"Average SegSNR Improvement: {avgs['SegSNR_impr']:.2f}\n")
        f.write(f"Average ILD Error Noisy: {avgs['L_ILD_noisy']:.4f} dB\n")
        f.write(f"Average ILD Error Enhanced: {avgs['L_ILD']:.4f} dB\n")
        f.write(f"Average IPD Error Noisy: {avgs['L_IPD_noisy']:.2f} deg\n")
        f.write(f"Average IPD Error Enhanced: {avgs['L_IPD']:.2f} deg\n")
    print(f"Wrote averages -> {avg_path}")


if __name__ == "__main__":
    main()
