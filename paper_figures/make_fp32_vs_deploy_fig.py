#!/usr/bin/env python3
"""FP32 original BCCTN vs zero-multiplier W4A8 deploy — per-SNR quality drop.

Three checkpoints, one identical paper-style eval subset (the `*_noisy` columns
agree to 4 decimals across all three runs, which is the evidence the subsets
match):

  FP32-ORIG   original BCCTN, fft_len=512 / embed_dim=512 / 32 heads, 3.97 M params
              BCCTN/results_May21/vctk_paper_style/          VCTK 0.8519 / TIMIT 0.8114
  HW-SIZED    hardware-sized FP8/QAT working point, fft_len=128 / embed_dim=256 /
              16 heads, 830 K params (ckpt epoch=134, *_quant_256dim)
              main_model/results_May21/vctk_paper_style/     VCTK 0.8399 / TIMIT 0.7966
  DEPLOY      W4A8 + pow2 block-float zero-multiplier core, 338 K RTL weights
              18_..._hw_handoff_20260530/results/apot_descale_eval/ft_pow2_ep3_full/
                                                             VCTK 0.8357 / TIMIT 0.7898

HW-SIZED is plotted because it splits the FP32->deploy drop into its two causes:
downsizing (FP32-ORIG -> HW-SIZED, which also carries the FP8/QAT step — that
checkpoint is already FP8-quantized, so the split is not arch-pure) and the W4A8
zero-multiplier core (HW-SIZED -> DEPLOY). Without it the whole gap reads as
"quantization cost", which overstates it by roughly 3x.

Colors are categorical slots 1/2/3 of the dataviz reference palette, the subset
that file certifies for all-pairs comparison in both modes. Aqua sits below 3:1
on a white surface, so the relief rule applies: every series carries its average
score as a visible label in the legend.

Output: SVG (text stays text for Visio ungroup), EMF (Inkscape), PNG (300 dpi).
Run:  ~/miniforge3/envs/bcctn/bin/python make_fp32_vs_deploy_fig.py
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from make_paper_figures import (
    BASE,
    COL2,
    GRID,
    INK,
    INK2,
    MUTED,
    OUT_EMF,
    OUT_PNG,
    OUT_SVG,
    ROOT,
    C1_BLUE,
    plt,
)

C2_ORANGE = "#eb6834"  # categorical slot 2
C3_AQUA = "#1baf7a"    # categorical slot 3

FP32_DIR = ROOT / "BCCTN" / "results_May21" / "vctk_paper_style"
HW_DIR = ROOT / "main_model" / "results_May21" / "vctk_paper_style"
DEPLOY_CSV = (ROOT / "18_w4a8_pow2_mixed_lmul_hw_handoff_20260530" / "results" /
              "apot_descale_eval" / "ft_pow2_ep3_full" / "per_snr_locked_paper_table.csv")

# Canonical averages come from the `*_averages.txt` files, not from an equal-weight
# mean of the per-SNR rows (RESPONSE_OPEN_ITEMS_PAPER_NUMBERS_2026-07-07 §3).
AVG_MBSTOI = {
    ("fp32", "VCTK"): 0.8519, ("fp32", "TIMIT"): 0.8114,
    ("hw", "VCTK"): 0.8399,   ("hw", "TIMIT"): 0.7966,
    ("deploy", "VCTK"): 0.8357, ("deploy", "TIMIT"): 0.7898,
}

MODELS = [
    ("fp32", "Original BCCTN FP32 (3.97 M)", C3_AQUA, "o", 1.8),
    ("hw", "HW-sized FP8/QAT (830 K)", C2_ORANGE, "s", 1.5),
    ("deploy", "W4A8 zero-mult deploy (338 K)", C1_BLUE, "o", 1.8),
]
METRICS = [
    ("MBSTOI", "MBSTOI", "higher is better"),
    ("SegSNR", "SegSNR (dB)", "higher is better"),
    ("ILD", "ILD error (dB)", "lower is better"),
    ("IPD", "IPD error (deg)", "lower is better"),
]


def load() -> dict:
    """Return {(model, dataset): DataFrame indexed by SNR} on a common schema."""
    data = {}
    legacy = {"MBSTOI": "MBSTOI", "SegSNR": "SegSNR", "ILD": "L_ILD", "IPD": "L_IPD"}
    noisy = {"MBSTOI": "MBSTOI_noisy", "SegSNR": "SegSNR_noisy",
             "ILD": "L_ILD_noisy", "IPD": "L_IPD_noisy"}

    for tag, base in (("fp32", FP32_DIR), ("hw", HW_DIR)):
        for ds in ("vctk", "timit"):
            raw = pd.read_csv(base / f"{ds}_paper_style_results.csv").sort_values("SNR")
            frame = pd.DataFrame({m: raw[legacy[m]].to_numpy(float) for m in legacy},
                                 index=raw["SNR"].to_numpy(float))
            data[(tag, ds.upper())] = frame
            if tag == "fp32":
                data[("noisy", ds.upper())] = pd.DataFrame(
                    {m: raw[noisy[m]].to_numpy(float) for m in noisy},
                    index=raw["SNR"].to_numpy(float))

    dep = pd.read_csv(DEPLOY_CSV)
    enh = {"MBSTOI": "MBSTOI_enh", "SegSNR": "SegSNR_enh",
           "ILD": "ILD_err_dB_enh", "IPD": "IPD_err_deg_enh"}
    for ds in ("VCTK", "TIMIT"):
        d = dep[dep["dataset"] == ds].sort_values("SNR_dB")
        data[("deploy", ds)] = pd.DataFrame(
            {m: d[enh[m]].to_numpy(float) for m in enh},
            index=d["SNR_dB"].to_numpy(float))

    # The three runs must share an eval subset for the deltas to mean anything.
    for ds in ("VCTK", "TIMIT"):
        ref = data[("noisy", ds)].index.to_numpy()
        for tag in ("fp32", "hw", "deploy"):
            assert np.array_equal(data[(tag, ds)].index.to_numpy(), ref), \
                f"SNR grid mismatch for {tag}/{ds}"
    fp32_noisy = pd.read_csv(FP32_DIR / "vctk_paper_style_results.csv")["MBSTOI_noisy"]
    hw_noisy = pd.read_csv(HW_DIR / "vctk_paper_style_results.csv")["MBSTOI_noisy"]
    assert np.allclose(fp32_noisy, hw_noisy, atol=1e-4), "VCTK noisy baselines differ"
    return data


def save(name: str) -> None:
    plt.savefig(OUT_SVG / f"{name}.svg")
    plt.savefig(OUT_PNG / f"{name}.png", dpi=300)
    plt.close()
    emf = OUT_EMF / f"{name}.emf"
    subprocess.run([str(ROOT / "tools" / "inkscape-1.3.2" / "AppRun"),
                    str(OUT_SVG / f"{name}.svg"), f"--export-filename={emf}"],
                   check=True, capture_output=True)
    print(f"  {name}.svg / .png / .emf")


# ── fig13: MBSTOI — absolute curves + drop relative to the FP32 original ─────
def fig13_mbstoi_drop(data: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(COL2, 4.1),
                             gridspec_kw={"height_ratios": [1.35, 1]})

    for col, ds in enumerate(("VCTK", "TIMIT")):
        ax = axes[0][col]
        snr = data[("fp32", ds)].index.to_numpy()

        ax.plot(snr, data[("noisy", ds)]["MBSTOI"], color=BASE, linewidth=1.3,
                linestyle=(0, (4, 2)), marker="o", markersize=3.2,
                label="Noisy input (unprocessed)")
        ax.fill_between(snr, data[("deploy", ds)]["MBSTOI"],
                        data[("fp32", ds)]["MBSTOI"], color=INK, alpha=0.07,
                        linewidth=0, zorder=1)
        for tag, name, color, marker, lw in MODELS:
            ax.plot(snr, data[(tag, ds)]["MBSTOI"], color=color, linewidth=lw,
                    marker=marker, markersize=4, markeredgecolor="white",
                    markeredgewidth=0.6, zorder=3,
                    label=f"{name} — avg {AVG_MBSTOI[(tag, ds)]:.4f}")

        title = "VCTK (matched)" if ds == "VCTK" else "TIMIT (unmatched)"
        ax.set_title(title, fontsize=8.5, color=INK)
        ax.set_ylim(0.44, 0.98)
        ax.set_xticks(snr)
        ax.tick_params(labelbottom=False)
        ax.legend(loc="lower right", handlelength=1.6, labelspacing=0.32,
                  borderpad=0.2)
        if col == 0:
            ax.set_ylabel("MBSTOI")

        # ── drop panel ──
        ax = axes[1][col]
        ax.axhline(0, color=C3_AQUA, linewidth=1.4, zorder=2)
        ax.annotate("Original BCCTN FP32 = reference", (0.985, 0.915),
                    xycoords="axes fraction", ha="right", va="bottom",
                    fontsize=6.5, color=C3_AQUA)

        d_hw = (data[("hw", ds)]["MBSTOI"] - data[("fp32", ds)]["MBSTOI"]).to_numpy()
        d_dep = (data[("deploy", ds)]["MBSTOI"] - data[("fp32", ds)]["MBSTOI"]).to_numpy()
        ax.fill_between(snr, d_dep, d_hw, color=C1_BLUE, alpha=0.10, linewidth=0)
        ax.plot(snr, d_hw, color=C2_ORANGE, linewidth=1.5, marker="s",
                markersize=3.6, markeredgecolor="white", markeredgewidth=0.6,
                zorder=3, label="after downsizing + FP8/QAT")
        ax.plot(snr, d_dep, color=C1_BLUE, linewidth=1.8, marker="o",
                markersize=4, markeredgecolor="white", markeredgewidth=0.6,
                zorder=3, label="+ W4A8 zero-multiplier core")

        avg_hw = AVG_MBSTOI[("hw", ds)] - AVG_MBSTOI[("fp32", ds)]
        avg_dep = AVG_MBSTOI[("deploy", ds)] - AVG_MBSTOI[("fp32", ds)]
        ax.annotate(f"avg drop {avg_dep:+.4f}\n"
                    f"{avg_hw:+.4f} downsizing + FP8/QAT\n"
                    f"{avg_dep - avg_hw:+.4f} W4A8 zero-mult core",
                    (0.97, 0.06), xycoords="axes fraction", ha="right",
                    fontsize=6.8, color=INK2, linespacing=1.45)

        ax.set_ylim(-0.104, 0.013)
        ax.set_yticks(np.arange(0.0, -0.101, -0.02))
        ax.set_xticks(snr)
        ax.set_xlabel("Input SNR (dB)")
        if col == 0:
            ax.set_ylabel("MBSTOI drop vs FP32")
            ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.02),
                      handlelength=1.6, labelspacing=0.28, borderpad=0.2,
                      fontsize=6.5)

    fig.align_ylabels(axes[:, 0])
    plt.tight_layout(h_pad=0.6, w_pad=1.2)
    save("fig13_fp32_vs_deploy_mbstoi")


# ── fig14: the other three paper metrics, same three checkpoints ─────────────
def fig14_other_metrics(data: dict) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(COL2, 3.9), sharex=True)

    for row, ds in enumerate(("VCTK", "TIMIT")):
        snr = data[("fp32", ds)].index.to_numpy()
        for col, (key, ylab, note) in enumerate(METRICS[1:]):
            ax = axes[row][col]
            ax.plot(snr, data[("noisy", ds)][key], color=BASE, linewidth=1.3,
                    linestyle=(0, (4, 2)), marker="o", markersize=3,
                    label="Noisy input")
            for tag, name, color, marker, lw in MODELS:
                ax.plot(snr, data[(tag, ds)][key], color=color, linewidth=lw,
                        marker=marker, markersize=3.6, markeredgecolor="white",
                        markeredgewidth=0.5, zorder=3, label=name)
            ax.set_ylabel(f"{ds}\n{ylab}" if col == 0 else ylab)
            if row == 0:
                ax.set_title(note, fontsize=7, color=MUTED)
            if row == 1:
                ax.set_xlabel("Input SNR (dB)")
                ax.set_xticks(snr)
                ax.tick_params(axis="x", labelsize=6.2)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.005), handlelength=1.6, columnspacing=1.4)
    plt.tight_layout(rect=(0, 0.055, 1, 1), h_pad=0.7, w_pad=1.0)
    save("fig14_fp32_vs_deploy_segsnr_ild_ipd")


def summary(data: dict) -> None:
    """Print the table behind the figures (the accessible text view)."""
    for ds in ("VCTK", "TIMIT"):
        print(f"\n{ds}  MBSTOI  (drop vs FP32 original)")
        print(f"{'SNR':>5} {'FP32':>8} {'HW-sized':>10} {'deploy':>8} "
              f"{'d_dnsz':>8} {'d_w4a8':>8} {'d_total':>8}")
        f = data[("fp32", ds)]["MBSTOI"]
        h = data[("hw", ds)]["MBSTOI"]
        d = data[("deploy", ds)]["MBSTOI"]
        for snr in f.index:
            print(f"{snr:>5.0f} {f[snr]:>8.4f} {h[snr]:>10.4f} {d[snr]:>8.4f} "
                  f"{h[snr] - f[snr]:>+8.4f} {d[snr] - h[snr]:>+8.4f} "
                  f"{d[snr] - f[snr]:>+8.4f}")
        af, ah, ad = (AVG_MBSTOI[(t, ds)] for t in ("fp32", "hw", "deploy"))
        print(f"{'avg':>5} {af:>8.4f} {ah:>10.4f} {ad:>8.4f} "
              f"{ah - af:>+8.4f} {ad - ah:>+8.4f} {ad - af:>+8.4f}")


def main() -> None:
    for d in (OUT_SVG, OUT_EMF, OUT_PNG):
        d.mkdir(parents=True, exist_ok=True)
    data = load()
    fig13_mbstoi_drop(data)
    fig14_other_metrics(data)
    summary(data)


if __name__ == "__main__":
    main()
