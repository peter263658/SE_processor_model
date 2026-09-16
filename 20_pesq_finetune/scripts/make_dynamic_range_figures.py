#!/usr/bin/env python3
"""Generate architecture-decision figures from the historical 01-06 analyses.

The figures are intentionally source-backed: every plotted value comes from the
CSV/JSON/markdown summaries in /raid/R12K41024/{02..06}_*.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("/raid/R12K41024")
PKG = ROOT / "18_w4a8_pow2_mixed_lmul_hw_handoff_20260530"
OUT = PKG / "docs" / "analysis_figures"


def savefig(name: str) -> None:
    path = OUT / name
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(path)


def short_layer(name: str) -> str:
    s = name
    s = s.replace("model.", "")
    s = s.replace(".parametrizations.weight.original", "")
    s = s.replace("mattn.mattn.", "")
    s = s.replace("MultiheadAttention_", "MHA_")
    s = s.replace("encoder.model.", "enc")
    s = s.replace("decoder.model.", "dec")
    s = s.replace(".0.", ".")
    s = s.replace("pointwise_", "pw_")
    s = s.replace("depthwise_", "dw_")
    return s


def fig_bitwidth_tradeoff() -> None:
    # Source: 02_weights_bitwidth/02_weights_bitwidth_analysis_summary.md
    rows = [
        ("FP8 baseline", 0.0, 10.492, 4.006, 23.97, "baseline"),
        ("Uniform INT4", 50.0, 9.929, 4.216, 24.61, "accepted"),
        ("Uniform INT3", 62.5, 4.701, 4.916, 25.94, "failed"),
    ]
    df = pd.DataFrame(rows, columns=["scheme", "memory_saving", "si_snr", "ild", "ipd", "status"])
    baseline = df.loc[df.scheme == "FP8 baseline"].iloc[0]
    df["si_snr_loss"] = baseline.si_snr - df.si_snr
    df["ild_delta"] = df.ild - baseline.ild
    df["ipd_delta"] = df.ipd - baseline.ipd

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = {"baseline": "#666666", "accepted": "#1f77b4", "failed": "#d62728"}
    for _, r in df.iterrows():
        axes[0].scatter(r.memory_saving, r.si_snr_loss, s=130, color=colors[r.status])
        axes[0].annotate(r.scheme, (r.memory_saving, r.si_snr_loss), xytext=(5, 7),
                         textcoords="offset points", fontsize=9)
    axes[0].axhline(0.5, color="#d62728", linestyle="--", linewidth=1, label="0.5 dB local threshold")
    axes[0].set_xlabel("Weight memory saving (%)")
    axes[0].set_ylabel("SI-SNR loss vs FP8 baseline (dB)")
    axes[0].set_title("Weight-only bitwidth sweep")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)

    x = np.arange(len(df))
    axes[1].bar(x - 0.18, df["ild_delta"], width=0.36, label="ILD delta (dB)", color="#ff7f0e")
    axes[1].bar(x + 0.18, df["ipd_delta"], width=0.36, label="IPD delta (deg)", color="#2ca02c")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(df.scheme, rotation=20, ha="right")
    axes[1].set_title("Spatial cue degradation")
    axes[1].grid(True, axis="y", alpha=0.25)
    axes[1].legend(fontsize=8)
    savefig("fig01_bitwidth_tradeoff.png")


def fig_utilization_breakdown() -> None:
    # Source: 01_utilization_analysis/01_utilization_analysis_report.md
    labels = ["Encoder", "Decoder", "Attention"]
    values = [7.4, 15.0, 77.6]
    colors = ["#2ca02c", "#ff7f0e", "#d62728"]
    plt.figure(figsize=(7.5, 4.5))
    bars = plt.bar(labels, values, color=colors)
    plt.ylabel("Share of MACs (%)")
    plt.title("01 utilization: attention dominates compute")
    plt.ylim(0, 90)
    plt.grid(True, axis="y", alpha=0.25)
    for b, v in zip(bars, values):
        plt.annotate(f"{v:.1f}%", (b.get_x() + b.get_width() / 2, v),
                     xytext=(0, 8), textcoords="offset points", ha="center")
    savefig("fig00_utilization_breakdown.png")


def fig_weight_outliers() -> None:
    src = ROOT / "03_dynamic_range" / "results_0410" / "dynamic_range_table.csv"
    df = pd.read_csv(src)
    df = df[np.isfinite(df["kurtosis"])].copy()
    df = df.sort_values("kurtosis", ascending=False).reset_index(drop=True)
    df["short"] = df["layer"].map(short_layer)
    df["is_decoder_pw"] = df["layer"].str.contains("decoder.model") & df["layer"].str.contains("pointwise")
    df["is_outlier"] = df["kurtosis"] > 5.0
    top = df.head(24).copy()

    plt.figure(figsize=(12, 5.2))
    colors = np.where(top["is_outlier"], "#d62728", np.where(top["is_decoder_pw"], "#ff9896", "#1f77b4"))
    plt.bar(np.arange(len(top)), top["kurtosis"], color=colors)
    plt.axhline(5.0, color="#d62728", linestyle="--", linewidth=1.2, label="FP/outlier threshold: kurtosis > 5")
    plt.xticks(np.arange(len(top)), top["short"], rotation=65, ha="right", fontsize=8)
    plt.ylabel("Kurtosis")
    plt.title("Weight distribution tails: decoder pointwise outliers")
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend(fontsize=8)
    savefig("fig02_weight_kurtosis_outliers.png")


def fig_group_size_mse() -> None:
    src = ROOT / "03_dynamic_range" / "results_0410" / "group_size_sensitivity.csv"
    df = pd.read_csv(src)
    cols = ["per_tensor_mse", "gsz_16_mse", "gsz_32_mse", "gsz_64_mse", "gsz_128_mse", "gsz_256_mse"]
    means = df[cols].mean()
    labels = ["tensor", "16", "32", "64", "128", "256"]
    values = means.values
    rel = values / values[0]

    fig, ax1 = plt.subplots(figsize=(8.5, 4.5))
    ax1.plot(labels, values, marker="o", linewidth=2, color="#1f77b4")
    ax1.set_xlabel("INT4 scale granularity / group size")
    ax1.set_ylabel("Average W4 reconstruction MSE")
    ax1.grid(True, alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(labels, rel, marker="s", linestyle="--", color="#ff7f0e")
    ax2.set_ylabel("MSE relative to per-tensor")
    ax1.set_title("Per-group scaling reduces W4 reconstruction error")
    for x, y in zip(labels, values):
        ax1.annotate(f"{y:.1e}", (x, y), xytext=(0, 8), textcoords="offset points",
                     ha="center", fontsize=8)
    savefig("fig03_group_size_mse_trend.png")


def fig_mha_head_variance() -> None:
    src = ROOT / "03_dynamic_range" / "results_0410" / "per_head_variance.csv"
    df = pd.read_csv(src).sort_values("var_max", ascending=False)
    df["short"] = df["layer"].map(short_layer)

    plt.figure(figsize=(11, 4.8))
    colors = np.where(df["var_max"] > 1e-3, "#d62728", "#1f77b4")
    plt.bar(np.arange(len(df)), df["var_max"], color=colors)
    plt.axhline(1e-3, color="#d62728", linestyle="--", linewidth=1.2,
                label="non-uniform head threshold: 1e-3")
    plt.yscale("log")
    plt.xticks(np.arange(len(df)), df["short"], rotation=65, ha="right", fontsize=8)
    plt.ylabel("Cross-head var_max (log scale)")
    plt.title("MHA head magnitude non-uniformity")
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend(fontsize=8)
    savefig("fig04_mha_per_head_variance.png")


def fig_attention_dynamic_range() -> None:
    src = ROOT / "06_attn_mised_precision" / "results" / "dynamic_range_per_subop.csv"
    df = pd.read_csv(src)
    order = ["Q", "K", "V", "QKT", "attn_weights", "sv"]
    df = df.set_index("subop").loc[order].reset_index()
    labels = ["Q", "K", "V", "QK^T", "softmax", "Score x V"]

    plt.figure(figsize=(9.5, 4.8))
    colors = ["#1f77b4"] * len(df)
    colors[3] = "#d62728"
    plt.bar(labels, df["abs_max"], color=colors)
    plt.axhline(448.0, color="#d62728", linestyle="--", linewidth=1.2, label="e4m3 max = 448")
    plt.yscale("log")
    plt.ylabel("abs max (log scale)")
    plt.title("Attention sub-op dynamic range: QK^T exceeds e4m3")
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend(fontsize=8)
    for i, r in df.iterrows():
        plt.annotate(f"log2={r.log2_range:.1f}", (i, r.abs_max), xytext=(0, 8),
                     textcoords="offset points", ha="center", fontsize=8)
    savefig("fig05_attention_subop_dynamic_range.png")


def fig_attention_entropy_topk() -> None:
    ent_src = ROOT / "06_attn_mised_precision" / "results" / "attention_entropy.csv"
    topk_src = ROOT / "06_attn_mised_precision" / "results" / "topk_eval.csv"
    ent = pd.read_csv(ent_src)
    topk = pd.read_csv(topk_src)
    topk_plot = topk[topk["k"] != "full"].copy()
    topk_plot["k"] = topk_plot["k"].astype(int)
    full = topk[topk["k"] == "full"].iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = np.where(ent["head_type"].str.contains("flat"), "#d62728", "#1f77b4")
    axes[0].bar(ent["head"], ent["mean_entropy_bits"], color=colors)
    axes[0].axhline(2.0, color="#d62728", linestyle="--", linewidth=1, label="flat threshold")
    axes[0].set_xlabel("Head")
    axes[0].set_ylabel("Entropy (bits)")
    axes[0].set_title("Attention entropy by head")
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[0].legend(fontsize=8)

    axes[1].plot(topk_plot["k"], topk_plot["mha_mse"], marker="o", color="#1f77b4", label="MHA MSE")
    axes[1].axhline(float(full["mha_mse"]), color="#1f77b4", linestyle="--", alpha=0.5, label="full MSE")
    axes[1].set_xlabel("top-k")
    axes[1].set_ylabel("MHA MSE")
    ax2 = axes[1].twinx()
    axes[1].set_title("Top-k=4 keeps waveform metrics close")
    ax2.plot(topk_plot["k"], topk_plot["ild_err_dB"], marker="s", color="#ff7f0e", label="ILD err")
    ax2.axhline(float(full["ild_err_dB"]), color="#ff7f0e", linestyle="--", alpha=0.5)
    ax2.set_ylabel("ILD error (dB)")
    axes[1].grid(True, alpha=0.25)
    lines, labels1 = axes[1].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes[1].legend(lines + lines2, labels1 + labels2, fontsize=8)
    savefig("fig06_attention_entropy_topk.png")


def fig_weight_dma_breakdown() -> None:
    src = ROOT / "05_conv_reuse" / "results" / "dw_pw_breakdown.csv"
    df = pd.read_csv(src)
    totals = {
        "DW conv": df["dw_bytes_FP8"].sum() / 1024.0,
        "PW conv": df["pw_bytes_FP8"].sum() / 1024.0,
        "MHA": 640.0,  # Source: 05_conv_reuse_analysis_report.md, re+im FP8 MHA.
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].bar(totals.keys(), totals.values(), color=["#8c564b", "#1f77b4", "#d62728"])
    axes[0].set_ylabel("FP8 weight bytes (KB)")
    axes[0].set_title("Weight DMA bottleneck")
    axes[0].grid(True, axis="y", alpha=0.25)
    for i, (k, v) in enumerate(totals.items()):
        axes[0].annotate(f"{v:.1f} KB", (i, v), xytext=(0, 8), textcoords="offset points",
                         ha="center", fontsize=9)

    axes[1].plot(df["layer"], df["pw_fraction"] * 100.0, marker="o", color="#1f77b4")
    axes[1].set_ylabel("PW fraction of conv weight DMA (%)")
    axes[1].set_title("Conv DMA is pointwise-dominated")
    axes[1].set_xticks(np.arange(len(df)))
    axes[1].set_xticklabels(df["layer"], rotation=45, ha="right")
    axes[1].grid(True, alpha=0.25)
    savefig("fig07_weight_dma_breakdown.png")


def fig_sparsity_summary() -> None:
    src = ROOT / "04_sparsity_analysis" / "results_test" / "sparsity_summary.json"
    with open(src) as f:
        summary = json.load(f)
    buckets = {
        "Random sparse\n(no structure)": 0,
        "No exploitable\nsparsity": 0,
        "Block sparse": 0,
        "Channel\nprunable": 0,
    }
    for text in summary.values():
        if "Block-sparse" in text:
            buckets["Block sparse"] += 1
        elif "Channel-prunable" in text:
            buckets["Channel\nprunable"] += 1
        elif "No exploitable" in text:
            buckets["No exploitable\nsparsity"] += 1
        else:
            buckets["Random sparse\n(no structure)"] += 1
    plt.figure(figsize=(8.5, 4.5))
    colors = ["#1f77b4", "#7f7f7f", "#ff7f0e", "#d62728"]
    bars = plt.bar(list(buckets.keys()), list(buckets.values()), color=colors)
    plt.ylabel("Layer count")
    plt.title("04 sparsity: little structured pruning opportunity")
    plt.grid(True, axis="y", alpha=0.25)
    for b, v in zip(bars, buckets.values()):
        plt.annotate(str(v), (b.get_x() + b.get_width() / 2, v),
                     xytext=(0, 8), textcoords="offset points", ha="center")
    savefig("fig09_sparsity_summary.png")


def fig_decision_map() -> None:
    decisions = [
        ("02 bitwidth", "INT4 wins\nINT3 fails cumulative", 0),
        ("03 dyn range", "decoder PW outliers\nstage 0/3/4 -> FP8/L-Mul", 1),
        ("03 head variance", "MHA per-head scale", 2),
        ("05 reuse", "MHA/PW dominate DMA\nfocus QKV + W4", 3),
        ("06 attention", "QK wide range\nfp16 acc + top4 packet", 4),
        ("18 deploy", "pow2 block-float act\nzero runtime act scaling", 5),
    ]
    plt.figure(figsize=(12, 3.6))
    y = np.zeros(len(decisions))
    x = np.arange(len(decisions))
    plt.plot(x, y, color="#444444", linewidth=2)
    plt.scatter(x, y, s=220, color="#1f77b4", zorder=3)
    for label, text, i in decisions:
        dy = 0.35 if i % 2 == 0 else -0.55
        va = "bottom" if dy > 0 else "top"
        plt.annotate(f"{label}\n{text}", (i, 0), xytext=(0, dy * 100),
                     textcoords="offset points", ha="center", va=va,
                     arrowprops=dict(arrowstyle="-", color="#777777"),
                     fontsize=9)
    plt.ylim(-1.2, 1.2)
    plt.axis("off")
    plt.title("Analysis-to-architecture decision map")
    savefig("fig08_analysis_decision_map.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })
    fig_utilization_breakdown()
    fig_bitwidth_tradeoff()
    fig_weight_outliers()
    fig_group_size_mse()
    fig_mha_head_variance()
    fig_attention_dynamic_range()
    fig_attention_entropy_topk()
    fig_weight_dma_breakdown()
    fig_sparsity_summary()
    fig_decision_map()


if __name__ == "__main__":
    main()
