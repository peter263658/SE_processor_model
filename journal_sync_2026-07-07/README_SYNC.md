# Journal-repo sync bundle — FINAL (2026-07-07, v2)

**This directory is the complete, self-contained handover to the HW consultant.**
It answers the HW review's 🔴2 ("audit artifacts not synced") and carries the canonical
v2.2 documents. Suggested placement in the Journal repo: this whole directory →
`Journal/algo_feature/`; `FEATURES_en.md` here is the **canonical v2.2** — replace
`Journal/FEATURES_en.md` (and retire `Journal/BCCTN_features/FEATURES_en.md` v2) with it,
then sync `FEATURES_zh.md` from v2.2.

## Top-level documents

| File | What it is |
|---|---|
| **`FEATURES_en.md`** (v2.2) | THE consolidated feature inventory. All 13 HW-review findings resolved: 🔴1 competitor P-numbering abolished → venue-institution labels (JSSC'20-NTU / CICC'26-PKU / A-SSCC'25-Tsinghua / JSSC'24-SNU; MIT VLSID'25 dropped); 🔴3 third-MAC-basis warning (P1, §9-23); 🟡4–10 doc sync; 🟠 P3 reworded, "this repo" disambiguated; **E6 = paper-ready PESQ narrative** |
| `PAPER_FEATURE_SUMMARY_2026-07-07.md` / `.zh.md` | Algorithm consultant's summary (EN/ZH), synced to v2.2 state |
| `RESULTS_ANALYSIS_VERIFICATION_01_06_2026-07-07.md` | dirs-01–06 numbers audit + §7 addendum (all three re-runs closed) |
| `RESPONSE_OPEN_ITEMS_PAPER_NUMBERS_2026-07-07.md` | answers to the 5 open items (−62/−74.7 basis, MBSTOI pair, per-SNR lock, E-series location, citations) |
| `per_snr_locked_paper_table.csv` | locked per-SNR table (VCTK+TIMIT × 8 SNR × noisy/enh/Δ × 5 metrics; means reproduce 0.8357/0.7898) |
| `E_SERIES_MANIFEST.md` | locator for all 107 E-series runs (20_pesq_finetune) |

## `paper_figures/` (fig01–fig12)

SVG + **EMF (Visio-ungroupable)** + 300-dpi PNG; `CAPTIONS.md` v2 has a **model/stage
column** (BASELINE-PROFILE / FP8-QAT-BASELINE / DEPLOY) stating which model each figure is
computed on — fig10/11/12 are the DEPLOY model itself; fig01 is the pre-compression
baseline profile (renamed `_baseline_e256`); fig02–09 are the FP8-QAT analysis stage.
`data/util_rows_e256.csv` = deploy-dims per-layer profile. `make_paper_figures.py`
regenerates everything from the artifacts below.

## `rerun_artifacts_2026-07-07/` (reproducibility re-runs, one GPU, 2026-07-07)

| Dir | Contents | Key result |
|---|---|---|
| `02_bitwidth/` | sensitivity_matrix, mixed_precision_{config,summary,eval}, fp8_baseline (n=500/seed 42) | tier 29C+1B & savings 61.5/50/62.5% reproduce the report exactly; INT4 −0.38 dB / INT3 −4.36 dB / mixed −2.31 dB |
| `03_dynamic_range/` | fixed-analyzer regen (num_heads=32, as the report) | all-row group-size means reproduce the report's table digit-for-digit; `per_channel_variance.csv` restores the 42/42 claim |
| `03_dynamic_range_h16/` | per_head_variance at the **deploy-aligned 16-head split** | 8/12 matrices non-uniform (fig05's source) |
| `06_attn_n20/` | attention analysis re-run at n=20 (n=2 originals kept in the algo repo) | QK^T \|max\| 1049 > 448; 12/16 sharp heads confirmed |

Algorithm-repo originals (canonical, full trees incl. logs/plots): `/raid/R12K41024/`
under `02_weights_bitwidth/results_rerun_2026-07-07/`, `03_dynamic_range/
results_regen_2026-07-07{,_h16}/`, `06_attn_mised_precision/results/` (+
`results_n2_2026-04-16/` snapshot), per-SNR source at `18_w4a8_pow2_mixed_lmul_hw_handoff_
20260530/results/apot_descale_eval/ft_pow2_ep3_full/`.

Remaining open on the HW side: synthesis/measured area & power (comparison-table gap);
`FEATURES_zh.md` sync to v2.2; converging the two English FEATURES copies onto this one.
