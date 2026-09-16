"""Collate every bilateral-eval run into the comparison-table rows.

Reads the `*_paper_style_results.csv` written by eval_bilateral.py / eval.py and
prints the markdown tables used by docs/RESULTS_*.md.

    python collate_rows.py            # summary + per-SNR MBSTOI/ILD
"""

import csv
from pathlib import Path

PKG = Path("/raid/R12K41024/21_mono_se_baselines")
DEPLOY = Path("/raid/R12K41024/18_w4a8_pow2_mixed_lmul_hw_handoff_20260530"
              "/results/apot_descale_eval/ft_pow2_ep3_full")

# label -> directory holding {vctk,timit}_paper_style_results.csv
RUNS = [
    ("Deploy (binaural W4A8)",            DEPLOY),
    ("FP32 binaural (09_tied_ft)",        PKG / "results/bilateral_eval/fp32_binaural"),
    ("NTU repro (mono, per-ear mask)",    PKG / "results/bilateral_eval/ntu"),
    ("NTU repro (mono, SHARED mask)",     PKG / "results/bilateral_eval/ntu_shared"),
    ("NTU (MSE loss, full data)",         PKG / "results/bilateral_eval/ntu_mse"),
    ("NTU (as-published regime)",         PKG / "results/bilateral_eval/ntu_aspub"),
    ("SNU repro (mono, per-ear mask)",    PKG / "results/bilateral_eval/snu"),
    ("SNU repro (mono, SHARED mask)",     PKG / "results/bilateral_eval/snu_shared"),
]

SNRS = [-6, -3, 0, 3, 6, 9, 12, 15]


def load(d, ds):
    p = d / f"{ds}_paper_style_results.csv"
    if not p.exists():
        return None
    with open(p) as f:
        return {int(float(r["SNR"])): r for r in csv.DictReader(f)}


def avg(rows, key):
    """Mean of `key` over the 8 SNR points; None when the run lacks that column."""
    if not rows:
        return None
    vals = [float(rows[s][key]) for s in SNRS if s in rows and rows[s].get(key)]
    return sum(vals) / len(vals) if vals else None


def cell(rows, key, fmt=".4f"):
    v = avg(rows, key)
    return "n/a" if v is None else format(v, fmt)


def main():
    have = [(lbl, {ds: load(d, ds) for ds in ("vctk", "timit")}) for lbl, d in RUNS]
    have = [(lbl, r) for lbl, r in have if r["vctk"]]

    ref = have[0][1]
    cols = [("MBSTOI", "vctk", "MBSTOI"), ("MBSTOI", "timit", "MBSTOI"),
            ("ILD", "vctk", "L_ILD"), ("ILD", "timit", "L_ILD"),
            ("PESQ", "vctk", "PESQ"), ("STOI", "vctk", "STOI")]

    print("## Summary (750 pairs x 8 SNR)\n")
    print("| Variant | VCTK MBSTOI | TIMIT MBSTOI | VCTK ILD err (dB) | TIMIT ILD err (dB) "
          "| VCTK PESQ (wb) | VCTK STOI |")
    print("|---|---|---|---|---|---|---|")
    print("| Noisy (unprocessed) | "
          + " | ".join(cell(ref[ds], k + "_noisy") for _, ds, k in cols) + " |")
    for lbl, r in have:
        print(f"| {lbl} | " + " | ".join(cell(r[ds], k) for _, ds, k in cols) + " |")

    for ds in ("vctk", "timit"):
        for metric, key, fmt in (("MBSTOI", "MBSTOI", ".4f"), ("ILD error (dB)", "L_ILD", ".4f")):
            rows = [(lbl, r[ds]) for lbl, r in have if r[ds]]
            if not rows:
                continue
            print(f"\n### {ds.upper()} per-SNR {metric}\n")
            print("| SNR | Noisy | " + " | ".join(lbl for lbl, _ in rows) + " |")
            print("|---" * (len(rows) + 2) + "|")
            for s in SNRS:
                nz = rows[0][1].get(s, {}).get(f"{key}_noisy")
                cells = [format(float(r[s][key]), fmt) if s in r else "-" for _, r in rows]
                print(f"| {s} | {format(float(nz), fmt) if nz else 'n/a'} | "
                      + " | ".join(cells) + " |")
            print(f"| **AVG** | **{cell(rows[0][1], key + '_noisy', fmt)}** | "
                  + " | ".join(f"**{cell(r, key, fmt)}**" for _, r in rows) + " |")


if __name__ == "__main__":
    main()
