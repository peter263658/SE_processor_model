#!/usr/bin/env python3
"""Slide figure: parameter count (x, log) vs MBSTOI (y), one point per simplification step.

Replaces the bar chart on the "Network Restructuring" slide, as the advisor asked: start at the
first design, add a point after every simplification. Points follow the ORDER THE MODELS WERE
ACTUALLY TRAINED (the slide's FFT -> attention -> depthwise order has no trained intermediates):

  B   baseline               fft 512, E 512, full conv (5,1)          3,973,288  24_.../results/P0p_*
  1   + depthwise-sep, fft 256                                        2,798,940  24_.../results/P2_*
  2   + fft 128 (5-ms frame)                                          2,536,540  24_.../results/P3_*
  3   + 32-frame causal window (no param change)                      2,536,540  24_.../results/P4_*
  4   + attention 256-d / 16 heads (+ FP8 QAT)  = hardware-sized        830,300  main_model/results_May21
  5   + tied MHA (09)                                                   567,132  RESULTS_ARCH_PROGRESSION
  6   + grouped decoder + skip subset (12)                              485,332  RESULTS_ARCH_PROGRESSION
  7   + grouped QKV = production (14)                                   337,876  RESULTS_ARCH_PROGRESSION
  8   zero-multiplier W4A8 deploy (18)                                  337,876  RESULTS_ARCH_PROGRESSION

VCTK and TIMIT are two series (slots 1/2 of the reference palette, certified as a pair in
make_fp32_vs_deploy_fig.py). Output: SVG (text stays text for Visio), EMF, PNG.
Run:  ~/miniforge3/envs/bcctn/bin/python make_param_vs_mbstoi_fig.py
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import numpy as np

from make_paper_figures import BASE, C1_BLUE, GRID, INK, INK2, INKSCAPE, MUTED, OUT_EMF, OUT_PNG, OUT_SVG, ROOT, plt

C2_ORANGE = "#eb6834"
R24 = ROOT / "24_param_mbstoi_progression" / "results"


def avg(path: Path) -> float | None:
    if not path.exists():
        return None
    m = re.search(r"Average MBSTOI Enhanced:\s*([\d.]+)", path.read_text())
    return float(m.group(1)) if m else None


def new_point(tag: str) -> tuple[float | None, float | None]:
    return (avg(R24 / tag / "vctk_paper_style_results_averages.txt"),
            avg(R24 / tag / "timit_paper_style_results_averages.txt"))


POINTS = [  # (label, params, vctk, timit)
    ("Baseline\n(fft 512, E 512)", 3_973_288, *new_point("P0p_fft512_E512_full51_3.97M")),
    ("+ depthwise-sep,\nfft 256", 2_798_940, *new_point("P2_fft256_E512_dw_2.80M")),
    ("+ fft 128", 2_536_540, *new_point("P3_fft128_E512_dw_2.54M")),
    ("+ 32-frame\nwindow", 2_536_540, *new_point("P4_fft128_E512_dw_win31_2.54M")),
    ("+ attention 256-d /\n16 heads (HW-sized)", 830_300, 0.8399, 0.7966),
    ("+ tied MHA", 567_132, 0.8411, 0.7966),
    ("+ grouped decoder", 485_332, 0.8352, 0.7919),
    ("+ grouped QKV", 337_876, 0.8354, 0.7880),
    ("zero-mult W4A8\ndeploy", 337_876, 0.8357, 0.7898),
]


def main() -> None:
    for d in (OUT_SVG, OUT_EMF, OUT_PNG):
        d.mkdir(parents=True, exist_ok=True)
    pts = [p for p in POINTS if p[2] is not None]
    missing = [p[0].replace("\n", " ") for p in POINTS if p[2] is None]
    if missing:
        print("WARNING: no result yet for:", missing)
    x = np.array([p[1] for p in pts], float)
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    for key, color, name, idx in (("VCTK", C1_BLUE, "VCTK (matched)", 2), ("TIMIT", C2_ORANGE, "TIMIT (unmatched)", 3)):
        y = np.array([p[idx] for p in pts], float)
        ax.plot(x, y, color=color, linewidth=1.6, marker="o", markersize=5.5, markerfacecolor="white",
                markeredgecolor=color, markeredgewidth=1.6, zorder=3, label=f"{name}")
    # direct labels on the VCTK series (one per step; alternate above/below to avoid collisions)
    for i, p in enumerate(pts):
        dy = 9 if i % 2 == 0 else -13
        ax.annotate(p[0], (p[1], p[2]), textcoords="offset points", xytext=(0, dy), ha="center",
                    fontsize=5.6, color=INK2, zorder=4)
    ax.set_xscale("log")
    ax.invert_xaxis()  # simplification proceeds left -> right
    ax.set_xlabel("Parameters", color=INK)
    ax.set_ylabel("MBSTOI", color=INK)
    ax.set_xticks([4e6, 2e6, 1e6, 5e5, 3e5], ["4 M", "2 M", "1 M", "500 K", "300 K"])
    ax.grid(True, which="major", color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASE)
    ax.tick_params(colors=MUTED)
    ax.legend(frameon=False, loc="lower left")
    plt.tight_layout()
    name = "slide_param_vs_mbstoi"
    plt.savefig(OUT_SVG / f"{name}.svg")
    plt.savefig(OUT_PNG / f"{name}.png", dpi=300)
    plt.close()
    subprocess.run([str(INKSCAPE), str(OUT_SVG / f"{name}.svg"), f"--export-filename={OUT_EMF / f'{name}.emf'}"],
                   check=True, capture_output=True)
    print("points:", " | ".join(f"{p[0].replace(chr(10), ' ')}: {p[1]:,} -> {p[2]:.4f}/{p[3]:.4f}" for p in pts))
    print(f"wrote {name}.svg / .png / .emf")


if __name__ == "__main__":
    main()
