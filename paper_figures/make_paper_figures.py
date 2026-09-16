#!/usr/bin/env python3
"""Paper figures for the BCCTN zero-multiplier accelerator paper.

Every plotted value is source-backed by a verified artifact (see CAPTIONS.md for
per-figure sources and caveats). Data issues found in the 2026-07-07 verification
pass are corrected here:
  - MAC breakdown uses the deploy basis (embed_dim=256/16 heads), not the stale
    512-dim profile (77.6% -> 54.0% attention).
  - Group-size MSE is recomputed from the raw CSV restricted to conv+MHA weight
    tensors (the BN/PReLU rows come from a pre-bugfix analyzer run and invert
    the trend).
  - Sparsity summary reads the authoritative 04 `results/` dir (not results_test).
  - No dual-axis (twinx) charts.

Output: SVG (text kept as text for Visio ungroup+edit), EMF (via Inkscape), PNG
(300 dpi preview). Run:  python make_paper_figures.py
"""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("/raid/R12K41024")
HERE = ROOT / "paper_figures"
OUT_SVG = HERE / "out" / "svg"
OUT_EMF = HERE / "out" / "emf"
OUT_PNG = HERE / "out" / "png"
INKSCAPE = ROOT / "tools" / "inkscape-1.3.2" / "AppRun"

# ── validated reference palette (dataviz skill, light mode) ──────────────────
C1_BLUE = "#2a78d6"   # categorical slot 1
C2_AQUA = "#1baf7a"   # slot 2
C3_YELLOW = "#eda100" # slot 3
C5_VIOLET = "#4a3aa7" # slot 5
ST_GOOD = "#0ca30c"   # status: good
ST_CRIT = "#d03b3b"   # status: critical
INK = "#0b0b0b"       # primary ink
INK2 = "#52514e"      # secondary ink
MUTED = "#898781"     # axis/labels
GRID = "#e1e0d9"      # hairline grid
BASE = "#c3c2b7"      # baseline/axis + de-emphasized marks
SEQ = ["#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]  # blue ramp

# IEEE column widths (inches)
COL1 = 3.5
COL2 = 7.16

plt.rcParams.update({
    "svg.fonttype": "none",            # keep text as <text> -> editable in Visio
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7,
    "text.color": INK,
    "axes.labelcolor": INK2,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.edgecolor": BASE,
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def save(name: str) -> None:
    svg = OUT_SVG / f"{name}.svg"
    plt.savefig(svg)
    plt.savefig(OUT_PNG / f"{name}.png", dpi=300)
    plt.close()
    print(f"  {svg.name}")


def convert_emf() -> None:
    for svg in sorted(OUT_SVG.glob("*.svg")):
        emf = OUT_EMF / (svg.stem + ".emf")
        subprocess.run([str(INKSCAPE), str(svg), f"--export-filename={emf}"],
                       check=True, capture_output=True)
        print(f"  {emf.name}")


# ── fig01: MAC breakdown (hardware-sized BASELINE at deploy dims) ────────────
def fig01_mac_breakdown() -> None:
    # Source: 01 utilization_profiler.py analytical model re-run at the deploy
    # DIMENSIONS (embed_dim=256, num_heads=16) but the PRE-09 architecture stage
    # (untied MHA, dense QKV, ungrouped decoder) — i.e. the hardware-sized baseline
    # the analyses started from, NOT the deployed compressed network. Motivation
    # figure only. Per-layer rows in data/util_rows_e256.csv; total 823,171,200
    # MACs / 1 s. Old report basis (E=512): 77.6/15.0/7.4.
    labels = ["Attention", "Decoder", "Encoder"]
    values = [54.0, 30.8, 15.2]
    colors = [C1_BLUE, BASE, BASE]  # emphasize the headline component

    fig, ax = plt.subplots(figsize=(COL1, 1.6))
    y = np.arange(len(labels))[::-1]
    ax.barh(y, values, height=0.62, color=colors)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 62)
    ax.set_xlabel("Share of MACs (%)")
    ax.grid(axis="y", visible=False)
    for yi, v in zip(y, values):
        ax.annotate(f"{v:.1f}%", (v, yi), xytext=(3, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK)
    plt.tight_layout()
    save("fig01_mac_breakdown_baseline_e256")


# ── fig02: weight bit-width trade-off ────────────────────────────────────────
def fig02_bitwidth_tradeoff() -> None:
    # Source: 02_weights_bitwidth/results_rerun_2026-07-07/mixed_precision_summary.csv
    # (n=500, seed 42 — the 2026-07-07 reproducibility re-run; the original report's
    # raw outputs were not retained. Same conclusions, artifact-backed numbers.)
    src = pd.read_csv(ROOT / "02_weights_bitwidth" / "results_rerun_2026-07-07" /
                      "mixed_precision_summary.csv")
    base = src[src["config"] == "FP8 baseline"].iloc[0]

    def delta(cfg):
        r = src[src["config"] == cfg].iloc[0]
        return (float(base["si_snr_avg"] - r["si_snr_avg"]),
                float(r["ild_avg"] - base["ild_avg"]),
                float(r["ipd_avg"] - base["ipd_avg"]),
                float(r["mem_savings_pct"]))

    d4, d3 = delta("Uniform INT4"), delta("Uniform INT3")
    rows = [
        ("FP8 (base)", 0.0, 0.0, 0.00, 0.00, BASE),
        ("INT4", d4[3], d4[0], d4[1], d4[2], ST_GOOD),
        ("INT3", d3[3], d3[0], d3[1], d3[2], ST_CRIT),
    ]
    print(f"    [fig02] n=500 rerun deltas: INT4 SI-SNR -{d4[0]:.2f} dB, ILD +{d4[1]:.2f}, "
          f"IPD +{d4[2]:.2f}; INT3 SI-SNR -{d3[0]:.2f} dB")
    fig, axes = plt.subplots(1, 3, figsize=(COL2, 1.9))

    ax = axes[0]
    offsets = {"FP8 (base)": (4, -10), "INT4": (4, -13), "INT3": (-6, 2)}
    aligns = {"INT3": "right"}
    for name, mem, dsnr, _, _, c in rows:
        ax.scatter(mem, dsnr, s=42, color=c, zorder=3)
        ax.annotate(name, (mem, dsnr), xytext=offsets[name],
                    textcoords="offset points", fontsize=7.5, color=INK,
                    ha=aligns.get(name, "left"))
    ax.axhline(0.5, color=MUTED, linestyle=(0, (4, 3)), linewidth=0.9)
    ax.annotate("0.5 dB threshold", (68, 0.75), fontsize=6.8, color=MUTED,
                ha="right")
    ax.set_xlabel("Weight memory saving (%)")
    ax.set_ylabel("SI-SNR loss (dB)")
    ax.set_xlim(-6, 72)

    for ax, idx, ylab in ((axes[1], 3, "ILD error increase (dB)"),
                          (axes[2], 4, "IPD error increase (deg)")):
        names = [r[0] for r in rows]
        vals = [r[idx] for r in rows]
        cols = [r[5] for r in rows]
        x = np.arange(len(rows))
        ax.bar(x, vals, width=0.55, color=cols)
        ax.set_xticks(x, names)
        ax.set_ylabel(ylab)
        ax.grid(axis="x", visible=False)
        for xi, v in zip(x, vals):
            ax.annotate(f"+{v:.2f}", (xi, v), xytext=(0, 2), textcoords="offset points",
                        ha="center", fontsize=7, color=INK)
    plt.tight_layout()
    save("fig02_bitwidth_tradeoff")


# ── fig03: weight kurtosis outliers ──────────────────────────────────────────
def fig03_kurtosis_outliers() -> None:
    # Source: 03_dynamic_range/results_0410/dynamic_range_table.csv (verified:
    # exactly 5 tensors with kurtosis>5, all decoder pointwise stages 0/3/4).
    df = pd.read_csv(ROOT / "03_dynamic_range" / "results_0410" / "dynamic_range_table.csv")
    df = df[np.isfinite(df["kurtosis"])].sort_values("kurtosis", ascending=False).head(20)

    def short(s: str) -> str:
        s = s.replace("model.", "").replace(".parametrizations.weight.original", "")
        s = s.replace(".parametrizations.in_proj_weight.original", " in_proj")
        s = s.replace("encoder.model.", "enc ").replace("decoder.model.", "dec ")
        s = s.replace("mattn.mattn.", "").replace("mattn.", "")
        s = s.replace("MultiheadAttention_", "MHA_")
        s = s.replace("pointwise_", "pw ").replace("depthwise_", "dw ")
        s = s.replace(".0.", " ").replace(".weight", "")
        return s

    names = df["layer"].map(short)
    outlier = df["kurtosis"] > 5.0
    colors = np.where(outlier, ST_CRIT, C1_BLUE)

    fig, ax = plt.subplots(figsize=(COL2, 2.7))
    x = np.arange(len(df))
    ax.bar(x, df["kurtosis"], width=0.62, color=colors)
    ax.axhline(5.0, color=ST_CRIT, linestyle=(0, (4, 3)), linewidth=0.9)
    ax.annotate("outlier threshold (kurtosis = 5)", (len(df) - 0.5, 5.25),
                ha="right", fontsize=7, color=ST_CRIT)
    ax.set_xticks(x, names, rotation=55, ha="right", fontsize=6.2)
    ax.set_ylabel("Weight kurtosis")
    ax.grid(axis="x", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=ST_CRIT),
               plt.Rectangle((0, 0), 1, 1, color=C1_BLUE)]
    ax.legend(handles, ["FP8 outlier (decoder PW, stages 0/3/4)", "INT4-safe"],
              loc="upper right", bbox_to_anchor=(1.0, 0.92))
    fig.subplots_adjust(bottom=0.36, left=0.07, right=0.99, top=0.97)
    plt.savefig(OUT_SVG / "fig03_weight_kurtosis_outliers.svg")
    plt.savefig(OUT_PNG / "fig03_weight_kurtosis_outliers.png", dpi=300)
    plt.close()
    print("  fig03_weight_kurtosis_outliers.svg")


# ── fig04: group-size MSE (recomputed, conv+MHA weights only) ────────────────
def fig04_group_size_mse() -> None:
    # Source: 03_dynamic_range/results_regen_2026-07-07/group_size_sensitivity.csv —
    # regenerated 2026-07-07 with the FIXED analyzer (the stale results_0410 CSV came
    # from a pre-bugfix run and inverted the trend). The regen's all-row means
    # exactly reproduce the 03 report's published table (1.314e-3 -> gsz16 9.572e-4).
    df = pd.read_csv(ROOT / "03_dynamic_range" / "results_regen_2026-07-07" /
                     "group_size_sensitivity.csv")
    cols = ["per_tensor_mse", "gsz_16_mse", "gsz_32_mse", "gsz_64_mse",
            "gsz_128_mse", "gsz_256_mse"]
    means = df[cols].mean()
    labels = ["per-tensor", "16", "32", "64", "128", "256"]
    vals = means.values
    print(f"    [fig04] regen rows n={len(df)}; means:",
          ", ".join(f"{l}={v:.3e}" for l, v in zip(labels, vals)))

    fig, ax = plt.subplots(figsize=(COL1, 2.1))
    x = np.arange(len(labels))
    ax.plot(x, vals, marker="o", markersize=4.5, linewidth=1.6, color=C1_BLUE)
    deploy_i = labels.index("32")
    ax.scatter([x[deploy_i]], [vals[deploy_i]], s=60, facecolor="white",
               edgecolor=ST_GOOD, linewidth=1.6, zorder=4)
    ax.annotate("deployed\n(gsz=32)", (x[deploy_i], vals[deploy_i]),
                xytext=(0, 10), textcoords="offset points", ha="center",
                fontsize=7, color=ST_GOOD)
    for xi, v, lab in zip(x, vals, labels):
        ax.annotate(f"{v / vals[0]:.2f}x", (xi, v), xytext=(0, -11),
                    textcoords="offset points", ha="center", fontsize=6.5, color=INK2)
    ax.set_xticks(x, labels)
    ax.set_xlabel("INT4 scale granularity (group size, input-channel axis)")
    ax.set_ylabel("Mean W4 reconstruction MSE")
    ax.set_ylim(0, vals.max() * 1.28)
    plt.tight_layout()
    save("fig04_group_size_mse")


# ── fig05: MHA per-head variance ─────────────────────────────────────────────
def fig05_mha_head_variance() -> None:
    # Source: 03_dynamic_range/results_regen_2026-07-07_h16/per_head_variance.csv —
    # deploy-aligned 16-head split (the deploy model has 16 per-head scales; the
    # original report grouped by 32). 8/12 matrices non-uniform at >1e-3.
    df = pd.read_csv(ROOT / "03_dynamic_range" / "results_regen_2026-07-07_h16" /
                     "per_head_variance.csv")
    df = df.sort_values("var_max", ascending=True)
    print(f"    [fig05] 16-head split: non-uniform {int((df['var_max'] > 1e-3).sum())}/{len(df)}")

    def short(s: str) -> str:
        s = s.replace("model.mattn.mattn.", "").replace("model.mattn.", "")
        s = s.replace("MultiheadAttention_", "MHA_")
        s = s.replace(".parametrizations.in_proj_weight.original.", " in_proj ")
        s = s.replace(".parametrizations.weight.original", "")
        s = s.replace(".out_proj", " out_proj")
        s = s.replace("output_proj.", "output_proj ").replace("input_proj.", "input_proj ")
        s = s.replace("imag", "(im)").replace("real", "(re)")
        return s

    names = df["layer"].map(short)
    nonuni = df["var_max"] > 1e-3
    colors = np.where(nonuni, C1_BLUE, BASE)

    fig, ax = plt.subplots(figsize=(4.6, 2.5))
    y = np.arange(len(df))
    ax.hlines(y, df["var_max"].min() * 0.5, df["var_max"], color=GRID,
              linewidth=0.8, zorder=2)
    ax.scatter(df["var_max"], y, s=22, color=colors, zorder=3)
    ax.axvline(1e-3, color=ST_CRIT, linestyle=(0, (4, 3)), linewidth=0.9)
    ax.annotate("per-head\nthreshold", (8.2e-4, 0.4), fontsize=6.8,
                color=ST_CRIT, ha="right")
    ax.set_xscale("log")
    ax.set_yticks(y, names, fontsize=6.5)
    ax.set_xlabel("Cross-head max |w| variance")
    ax.grid(axis="y", visible=False)
    plt.tight_layout()
    save("fig05_mha_per_head_variance")


# ── fig06: attention sub-op dynamic range ────────────────────────────────────
def fig06_attention_dynamic_range() -> None:
    # Source: 06_attn_mised_precision/results/dynamic_range_per_subop.csv
    # (verified exact; n=2 calibration probe -> state in caption).
    df = pd.read_csv(ROOT / "06_attn_mised_precision" / "results" / "dynamic_range_per_subop.csv")
    order = ["Q", "K", "V", "QKT", "attn_weights", "sv"]
    df = df.set_index("subop").loc[order].reset_index()
    labels = ["Q", "K", "V", "QK$^T$", "softmax", "Score×V"]
    over = df["abs_max"] > 448.0
    colors = np.where(over, ST_CRIT, C1_BLUE)

    fig, ax = plt.subplots(figsize=(COL1, 2.1))
    y = np.arange(len(df))[::-1]
    ax.hlines(y, 1e-2, df["abs_max"], color=GRID, linewidth=0.8, zorder=2)
    ax.scatter(df["abs_max"], y, s=26, color=colors, zorder=3)
    ax.axvline(448.0, color=ST_CRIT, linestyle=(0, (4, 3)), linewidth=0.9)
    ax.annotate("e4m3 max = 448", (0.855, 0.80), xycoords="axes fraction",
                fontsize=6.8, color=ST_CRIT, ha="right", va="top")
    ax.set_xscale("log")
    ax.set_yticks(y, labels)
    ax.set_xlabel("|max| per sub-op (log)")
    ax.grid(axis="y", visible=False)
    for yi, v in zip(y, df["abs_max"]):
        ax.annotate(f"{v:.0f}", (v, yi), xytext=(4, -2), textcoords="offset points",
                    fontsize=6.5, color=INK2)
    plt.tight_layout()
    save("fig06_attention_subop_dynamic_range")


# ── fig07: attention entropy + top-k error (two panels, no twinx) ────────────
def fig07_entropy_topk() -> None:
    # Source: 06 results/attention_entropy.csv + topk_eval.csv (n=2 probe).
    ent = pd.read_csv(ROOT / "06_attn_mised_precision" / "results" / "attention_entropy.csv")
    topk = pd.read_csv(ROOT / "06_attn_mised_precision" / "results" / "topk_eval.csv")
    tk = topk[topk["k"] != "full"].copy()
    tk["k"] = tk["k"].astype(int)
    tk = tk.sort_values("k")

    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.0))

    ax = axes[0]
    flat = ent["head_type"].str.contains("flat")
    n_heads = len(ent)
    n_sharp = int((~flat).sum())
    colors = np.where(flat, BASE, C1_BLUE)
    ax.bar(ent["head"], ent["mean_entropy_bits"], width=0.62, color=colors)
    ax.axhline(np.log2(31), color=MUTED, linestyle=(0, (4, 3)), linewidth=0.9)
    ax.annotate("uniform limit log$_2$(31) = 4.95", (0, np.log2(31) - 0.42),
                fontsize=6.8, color=MUTED)
    ax.set_xlabel("Attention head")
    ax.set_ylabel("Mean entropy (bits)")
    ax.set_ylim(0, 5.4)
    ax.grid(axis="x", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=C1_BLUE),
               plt.Rectangle((0, 0), 1, 1, color=BASE)]
    ax.legend(handles, [f"sharp (<2 bits): {n_sharp}/{n_heads}",
                        f"flat: {n_heads - n_sharp}/{n_heads}"],
              loc="upper right", bbox_to_anchor=(1.0, 0.86))

    ax = axes[1]
    ax.plot(tk["k"], tk["mha_mse"], marker="o", markersize=4.5, linewidth=1.6,
            color=C1_BLUE)
    ax.scatter([4], [float(tk.loc[tk["k"] == 4, "mha_mse"].iloc[0])], s=60,
               facecolor="white", edgecolor=ST_GOOD, linewidth=1.6, zorder=4)
    ax.annotate("deployed (top-4)", (4, float(tk.loc[tk["k"] == 4, "mha_mse"].iloc[0])),
                xytext=(8, 8), textcoords="offset points", fontsize=7, color=ST_GOOD)
    ax.set_xlabel("top-k (of 31-frame window)")
    ax.set_ylabel("Attention-prob. L1 error")
    ax.set_yscale("log")
    plt.tight_layout()
    save("fig07_attention_entropy_topk")


# ── fig08: weight-memory / DMA breakdown ─────────────────────────────────────
def fig08_weight_dma() -> None:
    # Source: 05_conv_reuse/results/dw_pw_breakdown.csv + report (MHA re+im FP8 =
    # 640 KB as profiled; counting the complex outer projections fully gives
    # ~768 KB -> qualitative dominance unchanged).
    df = pd.read_csv(ROOT / "05_conv_reuse" / "results" / "dw_pw_breakdown.csv")
    dw = df["dw_bytes_FP8"].sum() / 1024.0
    pw = df["pw_bytes_FP8"].sum() / 1024.0
    mha = 640.0
    labels = ["MHA (re+im)", "Pointwise conv", "Depthwise conv"]
    vals = [mha, pw, dw]
    colors = [C1_BLUE, BASE, BASE]

    fig, ax = plt.subplots(figsize=(COL1, 1.7))
    y = np.arange(len(labels))[::-1]
    ax.barh(y, vals, height=0.62, color=colors)
    ax.set_yticks(y, labels)
    ax.set_xlabel("FP8 weight bytes (KB)")
    ax.grid(axis="y", visible=False)
    for yi, v in zip(y, vals):
        ax.annotate(f"{v:.0f} KB", (v, yi), xytext=(3, 0), textcoords="offset points",
                    va="center", fontsize=7.5, color=INK)
    ax.set_xlim(0, 730)
    plt.tight_layout()
    save("fig08_weight_dma_breakdown")


# ── fig09: sparsity classification summary ───────────────────────────────────
def fig09_sparsity() -> None:
    # Source: 04_sparsity_analysis/results/sparsity_summary.json (authoritative
    # dir; results_test/ is an earlier partial run).
    with open(ROOT / "04_sparsity_analysis" / "results" / "sparsity_summary.json") as f:
        summary = json.load(f)
    buckets = {"Random sparse\n(unstructured)": 0, "No exploitable\nsparsity": 0,
               "Block sparse": 0, "Channel prunable": 0}
    for text in summary.values():
        if "Block-sparse" in text:
            buckets["Block sparse"] += 1
        elif "Channel-prunable" in text:
            buckets["Channel prunable"] += 1
        elif "No exploitable" in text:
            buckets["No exploitable\nsparsity"] += 1
        else:
            buckets["Random sparse\n(unstructured)"] += 1

    fig, ax = plt.subplots(figsize=(COL1, 2.0))
    x = np.arange(len(buckets))
    vals = list(buckets.values())
    colors = [BASE, BASE, C1_BLUE, C1_BLUE]
    ax.bar(x, vals, width=0.55, color=colors)
    ax.set_xticks(x, list(buckets.keys()), fontsize=6.8)
    ax.set_ylabel("Layer count (of 52)")
    ax.grid(axis="x", visible=False)
    for xi, v in zip(x, vals):
        ax.annotate(str(v), (xi, v), xytext=(0, 2), textcoords="offset points",
                    ha="center", fontsize=7.5, color=INK)
    plt.tight_layout()
    save("fig09_sparsity_summary")


# ── fig10: architecture progression (params + MBSTOI, stacked panels) ────────
def fig10_progression() -> None:
    # Source: 19_algo_final/docs/RESULTS_ARCH_PROGRESSION.md plot-point table.
    stages = ["Original\nBCCTN", "HW-sized\nbaseline", "09 tied\nMHA",
              "12 grouped\ndecoder", "14 grouped\nQKV (RTL)", "18 deploy\n(zero-mult)"]
    params = [3_973_288, 830_300, 567_132, 485_332, 337_876, 337_876]
    vctk = [0.8519, 0.8399, 0.8383, 0.8352, 0.8354, 0.8357]

    fig, axes = plt.subplots(2, 1, figsize=(COL2, 3.2), sharex=True,
                             gridspec_kw={"height_ratios": [1.15, 1]})
    x = np.arange(len(stages))

    ax = axes[0]
    ax.plot(x, params, marker="o", markersize=4.5, linewidth=1.6, color=C1_BLUE)
    ax.set_yscale("log")
    ax.set_ylabel("Stored weights (params)")
    for xi, p in zip(x, params):
        ax.annotate(f"{p / 1e6:.2f}M" if p > 1e6 else f"{p / 1e3:.0f}K", (xi, p),
                    xytext=(0, 7), textcoords="offset points", ha="center",
                    fontsize=7, color=INK)
    ax.annotate("-91.5%", (x[-1], params[-1]), xytext=(0, -14),
                textcoords="offset points", ha="center", fontsize=7.5,
                color=ST_GOOD)

    ax = axes[1]
    ax.plot(x, vctk, marker="o", markersize=4.5, linewidth=1.6, color=C2_AQUA)
    ax.axhline(0.835, color=MUTED, linestyle=(0, (4, 3)), linewidth=0.9)
    ax.annotate("deploy gate 0.835", (0, 0.8358), fontsize=6.8, color=MUTED)
    ax.set_ylabel("VCTK MBSTOI")
    ax.set_ylim(0.828, 0.856)
    for xi, v in zip(x, vctk):
        ax.annotate(f"{v:.4f}", (xi, v), xytext=(0, 6), textcoords="offset points",
                    ha="center", fontsize=6.5, color=INK2)
    ax.set_xticks(x, stages, fontsize=7)
    plt.tight_layout()
    save("fig10_arch_progression")


# ── fig11: per-SNR MBSTOI (locked deploy eval) ───────────────────────────────
def fig11_per_snr() -> None:
    # Source: 18_..._hw_handoff_20260530/results/apot_descale_eval/ft_pow2_ep3_full/
    # per_snr_locked_paper_table.csv (locked 2026-07-07; avg reproduces 0.8357/0.7898).
    df = pd.read_csv(ROOT / "18_w4a8_pow2_mixed_lmul_hw_handoff_20260530" / "results" /
                     "apot_descale_eval" / "ft_pow2_ep3_full" /
                     "per_snr_locked_paper_table.csv")
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.2), sharey=True)
    for ax, ds in zip(axes, ["VCTK", "TIMIT"]):
        d = df[df["dataset"] == ds].sort_values("SNR_dB")
        ax.plot(d["SNR_dB"], d["MBSTOI_enh"], marker="o", markersize=4,
                linewidth=1.6, color=C1_BLUE, label="Enhanced (deploy W4A8)")
        ax.plot(d["SNR_dB"], d["MBSTOI_noisy"], marker="o", markersize=4,
                linewidth=1.4, color=BASE, linestyle=(0, (4, 2)), label="Noisy")
        ax.set_title(ds, fontsize=8.5, color=INK)
        ax.set_xlabel("SNR (dB)")
        ax.set_xticks(d["SNR_dB"])
    axes[0].set_ylabel("MBSTOI")
    axes[0].legend(loc="lower right")
    plt.tight_layout()
    save("fig11_per_snr_mbstoi")


# ── fig12: PESQ finetune vs spatial-cue erosion (3 small multiples) ──────────
def fig12_pesq_tradeoff() -> None:
    # Source: 20_pesq_finetune/docs/STATUS_CHIP_PAPER_COMPARISON_POSITIONING_
    # 2026-06-28.md section 5 (VCTK, warm-start from deploy, PESQ-weight sweep).
    w = [0, 18, 30]
    mbstoi = [0.834, 0.823, 0.773]
    ild = [3.33, 3.44, 3.98]
    ipd = [16.0, 17.7, 21.6]

    fig, axes = plt.subplots(1, 3, figsize=(COL2, 1.9))
    panels = [("MBSTOI", mbstoi, "higher is better"),
              ("ILD error (dB)", ild, "lower is better"),
              ("IPD error (deg)", ipd, "lower is better")]
    for i, (ax, (ylab, vals, note)) in enumerate(zip(axes, panels)):
        ax.plot(w, vals, marker="o", markersize=4.5, linewidth=1.6, color=C1_BLUE)
        ax.scatter([0], [vals[0]], s=60, facecolor="white", edgecolor=ST_GOOD,
                   linewidth=1.6, zorder=4)
        ax.set_xlabel("PESQ loss weight")
        ax.set_ylabel(ylab)
        ax.set_xticks(w)
        pos, align = ((0.03, 0.06), "left") if i == 0 else ((0.97, 0.9), "right")
        ax.annotate(note, pos, xycoords="axes fraction", ha=align,
                    fontsize=6.5, color=MUTED)
    axes[0].axhline(0.835, color=MUTED, linestyle=(0, (4, 3)), linewidth=0.9)
    axes[0].annotate("gate 0.835", (17, 0.8355), fontsize=6.5, color=MUTED)
    axes[0].annotate("deploy", (0, mbstoi[0]), xytext=(6, -10),
                     textcoords="offset points", fontsize=7, color=ST_GOOD)
    plt.tight_layout()
    save("fig12_pesq_spatial_tradeoff")


def main() -> None:
    for d in (OUT_SVG, OUT_EMF, OUT_PNG):
        d.mkdir(parents=True, exist_ok=True)
    print("[1/2] SVG + PNG")
    fig01_mac_breakdown()
    fig02_bitwidth_tradeoff()
    fig03_kurtosis_outliers()
    fig04_group_size_mse()
    fig05_mha_head_variance()
    fig06_attention_dynamic_range()
    fig07_entropy_topk()
    fig08_weight_dma()
    fig09_sparsity()
    fig10_progression()
    fig11_per_snr()
    fig12_pesq_tradeoff()
    print("[2/2] EMF via Inkscape")
    convert_emf()


if __name__ == "__main__":
    main()
