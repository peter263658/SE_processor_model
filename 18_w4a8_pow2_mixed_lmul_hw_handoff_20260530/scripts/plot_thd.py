#!/usr/bin/env python
"""Figure for the THD probe: harmonic-only distortion, total THD+N, and the
level-dependent tone gain that explains the THD+N number."""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RES = "results/thd"
OUT = os.path.join(RES, "thd_summary.png")

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#dedcd5"
# categorical slots 1-3 of the validated reference palette (all-pairs safe)
C = {"roundtrip": "#2a78d6", "fp32": "#eb6834", "hw": "#1baf7a"}
LABEL = {"roundtrip": "STFT/iSTFT round-trip", "fp32": "FP32 model", "hw": "W4A8 HW path"}
SEQ = ["#a9c9ef", "#7aa9e3", "#4d86d3", "#2a63b4", "#123f7d"]   # single-hue ramp

data = {m: json.load(open(f"{RES}/{m}.json"))["results"]
        for m in ("roundtrip", "fp32", "hw")}
freqs = sorted({r["f0"] for r in data["fp32"]})
levels = sorted({r["level_dbfs"] for r in data["fp32"]})
OPER = [l for l in levels if l >= -20]     # operating range for the summary panels


def pick(mode, f0, lv):
    return next(r for r in data[mode] if r["f0"] == f0 and r["level_dbfs"] == lv)


def worst(mode, f0, key):
    return max(pick(mode, f0, l)[key] for l in OPER)


fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.3), facecolor=SURFACE)
for ax in axes:
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=3)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)

x = np.arange(len(freqs))
offs = {"roundtrip": -0.22, "fp32": 0.0, "hw": 0.22}

for ax, key, title, sub in (
        (axes[0], "thd_pct", "Harmonic distortion (THD)",
         "worst case over inputs $\\geq$ -20 dBFS"),
        (axes[1], "thdn_pct", "Total distortion + noise (THD+N)",
         "worst case over inputs $\\geq$ -20 dBFS")):
    for mode in ("roundtrip", "fp32", "hw"):
        raw = [worst(mode, f, key) for f in freqs]
        y = [max(v, 1.2e-3) for v in raw]
        ax.plot(x + offs[mode], y, "o", ms=9, color=C[mode],
                mec=SURFACE, mew=2, label=LABEL[mode], zorder=3)
        dy = -13 if mode == "roundtrip" else 9
        for xi, yi, v in zip(x + offs[mode], y, raw):
            txt = "<0.001" if v < 1e-3 else (f"{v:.3f}" if v < 1 else f"{v:.2f}")
            ax.annotate(txt, (xi, yi), textcoords="offset points", xytext=(0, dy),
                        ha="center", fontsize=7.2, color=INK2)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(f)}" for f in freqs])
    ax.set_xlabel("tone frequency (Hz)", color=INK2, fontsize=9.5)
    ax.set_ylabel("distortion (%)", color=INK2, fontsize=9.5)
    ax.set_title(title, color=INK, fontsize=11.5, weight="bold", loc="left", pad=18)
    ax.text(0, 1.035, sub, transform=ax.transAxes, color=INK2, fontsize=9)
    ax.axhline(2.0, color="#e34948", lw=1.2, ls="--", zorder=1)
    if key == "thd_pct":
        ax.text(-0.42, 5.0, "2 % — typical hearing-aid THD limit\n(ANSI S3.22 / IEC 60118-7)",
                fontsize=8, color="#e34948")

axes[0].set_ylim(8e-4, 3e1)
axes[1].set_ylim(8e-4, 3e1)
h, la = axes[0].get_legend_handles_labels()
fig.legend(h, la, frameon=False, fontsize=9.5, ncol=3, labelcolor=INK2,
           loc="upper center", bbox_to_anchor=(0.34, 1.005),
           handletextpad=0.3, columnspacing=2.0)

ax = axes[2]
for i, f in enumerate(freqs):
    y = [pick("hw", f, l)["gain_db"] for l in levels]
    ax.plot(levels, y, "-o", ms=6, lw=2, color=SEQ[i], mec=SURFACE, mew=1.5,
            zorder=3, label=f"{int(f)} Hz")
ax.axhline(0, color=GRID, lw=1)
ax.set_xticks(levels)
ax.set_xlim(levels[0] - 2, levels[-1] + 2)
ax.legend(frameon=False, fontsize=8.5, loc="lower right", ncol=2,
          labelcolor=INK2, handletextpad=0.4, columnspacing=1.0,
          title="tone frequency", title_fontsize=8.5)
ax.set_xlabel("tone input level (dBFS)", color=INK2, fontsize=9.5)
ax.set_ylabel("fundamental gain (dB)", color=INK2, fontsize=9.5)
ax.set_title("Tone gain, W4A8 HW path", color=INK, fontsize=11.5, weight="bold",
             loc="left", pad=18)
ax.text(0, 1.035, "a steady tone is 'noise' to a speech enhancer: it is gated at low level",
        transform=ax.transAxes, color=INK2, fontsize=9)

fig.tight_layout(rect=[0, 0, 1, 0.90])
fig.savefig(OUT, dpi=170, facecolor=SURFACE)
print(f"wrote {OUT}")
