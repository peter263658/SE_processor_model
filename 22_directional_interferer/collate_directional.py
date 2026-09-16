"""Collate the isotropic and directional conditions into one comparison.

Only within-condition comparisons are meaningful: a lateralised interferer
raises the noisy baseline for everyone, so the improvement over noisy (delta)
is the column to read across conditions, not the absolute score.
"""

import csv
from pathlib import Path

DEP = Path("/raid/R12K41024/18_w4a8_pow2_mixed_lmul_hw_handoff_20260530"
           "/results/apot_descale_eval")
MONO = Path("/raid/R12K41024/21_mono_se_baselines/results/bilateral_eval")

CONDITIONS = [
    ("isotropic (deploy考卷)", {
        "Deploy (binaural W4A8)":  DEP / "ft_pow2_ep3_full",
        "NTU mono, per-ear mask":  MONO / "ntu",
        "NTU mono, shared mask":   MONO / "ntu_shared",
    }),
    ("S0N±60: frontal target, two talkers ±60° (Westhausen-matched)", {
        "Deploy (binaural W4A8)":  DEP / "ft_dirspeech2x60_s0_deploy",
        "NTU mono, per-ear mask":  MONO / "dirspeech2x60_s0_ntu_independent",
        "NTU mono, shared mask":   MONO / "dirspeech2x60_s0_ntu_shared",
    }),
    ("two noise interferers, ±60°", {
        "Deploy (binaural W4A8)":  DEP / "ft_dirnoise2x60_deploy",
        "NTU mono, per-ear mask":  MONO / "dirnoise2x60_ntu_independent",
        "NTU mono, shared mask":   MONO / "dirnoise2x60_ntu_shared",
    }),
    ("two competing talkers, ±60° (Westhausen S0N±60)", {
        "Deploy (binaural W4A8)":  DEP / "ft_dirspeech2x60_deploy",
        "NTU mono, per-ear mask":  MONO / "dirspeech2x60_ntu_independent",
        "NTU mono, shared mask":   MONO / "dirspeech2x60_ntu_shared",
    }),
    ("directional noise, 90°", {
        "Deploy (binaural W4A8)":  DEP / "ft_dirnoise90_deploy",
        "NTU mono, per-ear mask":  MONO / "dirnoise90_ntu_independent",
        "NTU mono, shared mask":   MONO / "dirnoise90_ntu_shared",
    }),
    ("competing talker, 90°", {
        "Deploy (binaural W4A8)":  DEP / "ft_dirspeech90_deploy",
        "NTU mono, per-ear mask":  MONO / "dirspeech90_ntu_independent",
        "NTU mono, shared mask":   MONO / "dirspeech90_ntu_shared",
    }),
]
SNRS = [-6, -3, 0, 3, 6, 9, 12, 15]


def load(d):
    p = d / "vctk_paper_style_results.csv"
    if not p.exists():
        return None
    with open(p) as f:
        return {int(float(r["SNR"])): r for r in csv.DictReader(f)}


def avg(rows, key):
    vals = [float(rows[s][key]) for s in SNRS if s in rows and rows[s].get(key)]
    return sum(vals) / len(vals) if vals else None


def main():
    for cond, runs in CONDITIONS:
        loaded = [(lbl, load(d)) for lbl, d in runs.items()]
        loaded = [(lbl, r) for lbl, r in loaded if r]
        if not loaded:
            print(f"\n## {cond}\n\n_(not yet evaluated)_")
            continue
        nz_m = avg(loaded[0][1], "MBSTOI_noisy")
        nz_i = avg(loaded[0][1], "L_ILD_noisy")
        print(f"\n## {cond}\n")
        print("| model | MBSTOI | Δ vs noisy | ILD err (dB) | Δ vs noisy |")
        print("|---|---|---|---|---|")
        print(f"| _noisy_ | {nz_m:.4f} | — | {nz_i:.4f} | — |")
        for lbl, r in loaded:
            m, i = avg(r, "MBSTOI"), avg(r, "L_ILD")
            print(f"| {lbl} | {m:.4f} | {m - nz_m:+.4f} | {i:.4f} | {i - nz_i:+.4f} |")

    # per-SNR MBSTOI improvement, the shape that shows where binaural pays off
    for cond, runs in CONDITIONS[1:]:
        loaded = [(lbl, load(d)) for lbl, d in runs.items()]
        loaded = [(lbl, r) for lbl, r in loaded if r]
        if not loaded:
            continue
        print(f"\n### {cond} — per-SNR MBSTOI improvement over noisy\n")
        print("| SNR | noisy | " + " | ".join(l for l, _ in loaded) + " |")
        print("|---" * (len(loaded) + 2) + "|")
        for s in SNRS:
            nz = float(loaded[0][1][s]["MBSTOI_noisy"])
            cells = [f"{float(r[s]['MBSTOI']) - nz:+.4f}" if s in r else "-" for _, r in loaded]
            print(f"| {s} | {nz:.4f} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
