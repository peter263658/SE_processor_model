#!/usr/bin/env python3
"""Slide version of the MBSTOI-drop panel ("Performance Loss vs. FP32 Model").

Same VCTK deploy-vs-FP32 numbers as fig13's bottom-left panel, restyled to match
the slide (bold black frame, red dashed baseline, teal open circles) and with the
y-axis floor pulled down so the curve reads as hugging the FP32 baseline — the
change the advisor asked for ("adjust the y minimum so the two lines sit closer").

Run:  ~/miniforge3/envs/bcctn/bin/python make_perf_loss_slide_fig.py
"""

from __future__ import annotations

import subprocess

import numpy as np

from make_paper_figures import INKSCAPE, OUT_EMF, OUT_PNG, OUT_SVG, plt
from make_fp32_vs_deploy_fig import load

TEAL = "#3d9aae"
RED = "#e02020"

YMIN_DEFAULT = -0.30
# Headroom above the zero line is a fixed fraction of the axis span, so the red
# baseline label keeps the same clearance whatever floor is chosen.
HEADROOM = 0.13


def style(ax, snr, drop, ymin: float, tick: float, title: bool = True) -> None:
    ax.axhline(0, color=RED, linewidth=1.5, linestyle=(0, (4, 3)), zorder=2)
    ax.plot(snr, drop, color=TEAL, linewidth=1.7, marker="o", markersize=6,
            markerfacecolor="white", markeredgecolor=TEAL, markeredgewidth=1.7,
            zorder=3)
    # left-aligned: once the floor is pulled down the curve tail rises to ~0 and
    # would sit under a right-aligned label.
    ax.annotate("FP32 Baseline Model", (0.025, 0.99), xycoords="axes fraction",
                ha="left", va="top", fontsize=8.5, color=RED, weight="bold")

    ax.set_ylim(ymin, HEADROOM * abs(ymin))
    ax.set_yticks(np.arange(0.0, ymin - 1e-9, -tick))
    ax.set_xlim(snr[0] - 1.2, snr[-1] + 1.2)
    ax.set_xticks(snr, [f"{int(s)}dB" for s in snr], rotation=45, ha="right",
                  fontsize=9, weight="bold")
    ax.tick_params(labelsize=9)
    for lab in ax.get_yticklabels():
        lab.set_weight("bold")

    ax.grid(True, which="major", color="#bfbfbf", linewidth=0.7,
            linestyle=(0, (1, 2)), zorder=0)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color("black")
        sp.set_linewidth(1.8)
    if title:
        ax.set_title("Performance Loss vs. FP32 Model", fontsize=11.5,
                     weight="bold", color="black", pad=8)
    ax.set_ylabel("MBSTOI Drop", fontsize=10, weight="bold", color="black")


def main() -> None:
    for d in (OUT_SVG, OUT_EMF, OUT_PNG):
        d.mkdir(parents=True, exist_ok=True)
    data = load()
    snr = data[("fp32", "VCTK")].index.to_numpy()
    drop = (data[("deploy", "VCTK")]["MBSTOI"]
            - data[("fp32", "VCTK")]["MBSTOI"]).to_numpy()
    print("VCTK deploy - FP32 drop:",
          " ".join(f"{s:+.0f}dB {d:+.4f}" for s, d in zip(snr, drop)))

    # ── primary deliverable ──
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    style(ax, snr, drop, YMIN_DEFAULT, 0.10)
    plt.tight_layout()
    name = "slide_perf_loss_vs_fp32"
    plt.savefig(OUT_SVG / f"{name}.svg")
    plt.savefig(OUT_PNG / f"{name}.png", dpi=300)
    plt.close()
    subprocess.run([str(INKSCAPE), str(OUT_SVG / f"{name}.svg"),
                    f"--export-filename={OUT_EMF / f'{name}.emf'}"],
                   check=True, capture_output=True)
    print(f"  {name}.svg / .png / .emf   (y min {YMIN_DEFAULT})")

    # ── choose-your-floor comparison sheet (PNG only) ──
    opts = [(-0.08, 0.02), (-0.15, 0.05), (-0.20, 0.05), (-0.30, 0.10)]
    fig, axes = plt.subplots(1, len(opts), figsize=(13.0, 3.2))
    for ax, (ymin, tick) in zip(axes, opts):
        style(ax, snr, drop, ymin, tick, title=False)
        span = abs(drop.min()) / (abs(ymin) * (1 + HEADROOM))
        tag = "  ← current" if ymin == -0.08 else ""
        ax.set_title(f"y min = {ymin:g}   (curve = {span:.0%} of height){tag}",
                     fontsize=9.5, weight="bold", color="black", pad=8)
    plt.tight_layout()
    plt.savefig(OUT_PNG / "slide_perf_loss_ymin_options.png", dpi=200)
    plt.close()
    print("  slide_perf_loss_ymin_options.png")


if __name__ == "__main__":
    main()
