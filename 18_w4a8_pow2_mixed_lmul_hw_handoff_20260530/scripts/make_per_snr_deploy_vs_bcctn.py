#!/usr/bin/env python3
"""Per-SNR MBSTOI / SegSNR / ILD / IPD: original FP32 BCCTN vs HW-sized FP8 vs W4A8 pow2 deploy.

Rows = VCTK/TIMIT x SNR {-6..15} + an Avg row taken from the canonical *_averages.txt
(not an equal-weight mean of the per-SNR rows). Asserts the three runs share the same
noisy baseline, i.e. the same paper-style eval subset.

Run:  ~/miniforge3/envs/bcctn/bin/python scripts/make_per_snr_deploy_vs_bcctn.py [out.csv]
"""
import re
import sys
import pandas as pd
R = "/raid/R12K41024"
OUT = sys.argv[1] if len(sys.argv) > 1 else "per_snr_deploy_vs_bcctn.csv"
SRC = {
    "fp32": f"{R}/BCCTN/results_May21/vctk_paper_style",
    "hw": f"{R}/main_model/results_May21/vctk_paper_style",
    "dep": f"{R}/18_w4a8_pow2_mixed_lmul_hw_handoff_20260530/results/apot_descale_eval/ft_pow2_ep3_full",
}
COLS = {"MBSTOI": "MBSTOI", "SegSNR": "SegSNR", "ILD": "L_ILD", "IPD": "L_IPD"}
AVG_KEYS = {"MBSTOI": "Average MBSTOI Enhanced", "SegSNR": "Average SegSNR Enhanced",
            "ILD": "Average ILD Error Enhanced", "IPD": "Average IPD Error Enhanced"}
AVG_NOISY = "Average MBSTOI Noisy"

def avgs(path):
    out = {}
    for line in open(path):
        m = re.match(r"\s*(Average [^:]+):\s*([-\d.]+)", line)
        if m:
            out[m.group(1).strip()] = float(m.group(2))
    return out

rows = []
for ds in ("vctk", "timit"):
    d = {k: pd.read_csv(f"{v}/{ds}_paper_style_results.csv").set_index("SNR").sort_index() for k, v in SRC.items()}
    for k in ("hw", "dep"):
        assert (d[k]["MBSTOI_noisy"] - d["fp32"]["MBSTOI_noisy"]).abs().max() < 1e-4, (ds, k)
    for snr in d["fp32"].index:
        r = {"dataset": ds.upper(), "SNR": int(snr), "noisy": d["fp32"].loc[snr, "MBSTOI_noisy"]}
        for m, c in COLS.items():
            for k in SRC:
                r[f"{m}_{k}"] = d[k].loc[snr, c]
        rows.append(r)
    a = {k: avgs(f"{v}/{ds}_paper_style_results_averages.txt") for k, v in SRC.items()}
    r = {"dataset": ds.upper(), "SNR": "Avg", "noisy": a["fp32"][AVG_NOISY]}
    for m in COLS:
        for k in SRC:
            key = AVG_KEYS[m]
            val = next((v for kk, v in a[k].items() if kk.startswith(key) and "Noisy" not in kk and "Improvement" not in kk), None)
            r[f"{m}_{k}"] = val
    rows.append(r)

df = pd.DataFrame(rows)
for m in COLS:
    df[f"{m}_dep_minus_fp32"] = df[f"{m}_dep"] - df[f"{m}_fp32"]
    df[f"{m}_dep_minus_hw"] = df[f"{m}_dep"] - df[f"{m}_hw"]
df["MBSTOI_rel_drop_pct"] = 100 * df["MBSTOI_dep_minus_fp32"] / df["MBSTOI_fp32"]
df["MBSTOI_gain_kept_pct"] = 100 * (df["MBSTOI_dep"] - df["noisy"]) / (df["MBSTOI_fp32"] - df["noisy"])
df.to_csv(OUT, index=False, float_format="%.4f")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50)
print(df[["dataset","SNR","noisy","MBSTOI_fp32","MBSTOI_hw","MBSTOI_dep","MBSTOI_dep_minus_fp32","MBSTOI_dep_minus_hw","MBSTOI_rel_drop_pct","MBSTOI_gain_kept_pct"]].round(4).to_string(index=False))
print()
print(df[["dataset","SNR","SegSNR_fp32","SegSNR_dep","SegSNR_dep_minus_fp32","ILD_fp32","ILD_dep","ILD_dep_minus_fp32","IPD_fp32","IPD_dep","IPD_dep_minus_fp32"]].round(3).to_string(index=False))
