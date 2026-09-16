# Paper figures — captions, model stage, sources, and mandatory caveats (2026-07-07, v2)

Formats: `out/svg/` (text kept as text), `out/emf/` (Visio: Insert → Pictures, then
right-click → Group → Ungroup to edit), `out/png/` (300 dpi previews).
Regenerate: `python make_paper_figures.py` (fig01–fig12) and
`python make_fp32_vs_deploy_fig.py` (fig13–fig14) — bcctn env
(`~/miniforge3/envs/bcctn/bin/python`); EMF needs
`/raid/R12K41024/tools/inkscape-1.3.2/`.

**Model / stage legend** — which model each figure is computed on:
- **BASELINE-PROFILE** = analytical profiler at deploy *dimensions* (E=256/16) but the
  pre-09 *architecture* (untied MHA, dense QKV, ungrouped decoder). Motivation only —
  NOT the deployed network's distribution.
- **FP8-QAT-BASELINE** = the epoch-134 FP8/QAT precursor checkpoint the 01–06 analyses
  target (the model the quantization decisions were derived FROM).
- **DEPLOY** = the deployed zero-multiplier W4A8 model
  (`strict_core_pow2_blockfloat_deploy_ep3_vctk0.8357_timit0.7898.ckpt`) or evals
  warm-started from it.

| Fig | Stage | Suggested caption core | Source | MUST state in caption / text |
|---|---|---|---|---|
| fig01 (`_baseline_e256`) | **BASELINE-PROFILE** | MAC distribution of the hardware-sized baseline **prior to structural compression**: attention is the largest block (54.0%) — the motivation for the tied/grouped attention work | 01 profiler re-run at E=256/16; per-layer rows in `data/util_rows_e256.csv`; total 0.823 G MAC/s | NOT the deploy network's split (deploy compresses QKV −75%, dec-PW −74.7%); the historical 77.6% is the E=512 variant. Never divide 0.823 G against the 1.10 M m-eq/frame deploy count |
| fig02 | FP8-QAT-BASELINE | Weight-only PTQ: INT4 halves weight memory at −0.38 dB SI-SNR (below the 0.5 dB threshold); INT3 collapses (−4.36 dB) | `02_weights_bitwidth/results_rerun_2026-07-07/mixed_precision_summary.csv` (**n=500/seed 42 re-run**, artifacts retained) | The recipe this motivated (INT4 + FP8 outliers + per-group scale) is exactly what DEPLOY implements. Original report's absolute values = different eval subset, historical only |
| fig03 | FP8-QAT-BASELINE | Weight kurtosis: exactly 5 decoder-pointwise tensors (stages 0/3/4) exceed the outlier threshold → kept FP8+L-Mul | `03_dynamic_range/results_0410/dynamic_range_table.csv` (verified; regen reproduces) | These 5 become DEPLOY's 12 FP8 outlier tensors after the decoder split (r/i × x/skip) |
| fig04 | FP8-QAT-BASELINE | Per-group INT4 scaling: finer groups reduce reconstruction MSE; deployed gsz=32 marked | `03_dynamic_range/results_regen_2026-07-07/group_size_sensitivity.csv` (**fixed-analyzer regen**; all-row means exactly reproduce the 03 report's table) | The stale `results_0410` CSV/`summary.md` are pre-bugfix — never cite "gsz256 = 0.28×" |
| fig05 | FP8-QAT-BASELINE | MHA cross-head magnitude variance at the **deploy-aligned 16-head split**: 8/12 matrices non-uniform → per-head scales | `03_dynamic_range/results_regen_2026-07-07_h16/per_head_variance.csv` | The original report grouped by 32 heads (10/12 non-uniform); DEPLOY has 16 per-head scales — this figure matches that granularity |
| fig06 | FP8-QAT-BASELINE | Attention sub-op dynamic range: QK^T (\|max\| 1049) exceeds e4m3 max 448 → fp16 accumulation | `06 .../dynamic_range_per_subop.csv` (**n=20 re-run**; n=2 probe gave ±911, conclusion unchanged) | State n=20. DEPLOY's answer = FP16 accumulator + QK descale pre-softmax |
| fig07 | FP8-QAT-BASELINE | Attention concentration: 12/16 sharp heads (n=20); top-4 keeps prob-row L1 error low → top-4 packet | `attention_entropy.csv`, `topk_eval.csv` (**n=20 re-run**; n=2 originals in `results_n2_2026-04-16/`) | Top-4 survived into DEPLOY; top-4 *quality* cited from at-scale evals (07: 0.8340 / 14: 0.8360) |
| fig08 | FP8-QAT-BASELINE | FP8 weight-memory breakdown: MHA (640 KB) ≈ 4× all pointwise convs → the DMA bottleneck motivating W4 + grouped QKV + full WBUF residency | `05 .../dw_pw_breakdown.csv` + 05 report | 640 KB counts outer complex projections real-only (~768 KB fully complex); DEPLOY eliminates this transport entirely (all weights resident, zero weight DMA) |
| fig09 | FP8-QAT-BASELINE | Sparsity classification of 52 layers: no hardware-exploitable structure → no zero-skip hardware | `04_sparsity_analysis/results/sparsity_summary.json` (authoritative dir) | "6 dense" layers are 5 conv + 1 MHA (NOT "Q/K/V/O" as the report says) |
| fig10 | **DEPLOY** (endpoint) | Architecture progression: stored weights 3.97 M → 338 K (−91.5%) while VCTK MBSTOI stays above the 0.835 gate; endpoint = deploy 0.8357 | `RESULTS_ARCH_PROGRESSION.md` plot points | Param basis switches PyTorch→RTL at stage 14 (footnote) |
| fig11 | **DEPLOY** | Per-SNR MBSTOI, deploy model (noisy vs enhanced), VCTK + TIMIT | `per_snr_locked_paper_table.csv` (locked; averages reproduce 0.8357/0.7898) | Canonical averages = the `*_averages.txt` files |
| fig12 | **DEPLOY** (w=0 point) | PESQ-weight sweep erodes spatial cues dose-dependently → deploy reports the w=0 point | STATUS_CHIP_PAPER_COMPARISON_POSITIONING_2026-06-28.md §5 | The design-decision ablation ("we chose to preserve spatial cues"); pairs with FEATURES E6 narrative |
| fig13 | FP32-ORIG + FP8-QAT-BASELINE + **DEPLOY** | Per-SNR MBSTOI cost of the whole FP32→chip path: −0.0162 VCTK / −0.0216 TIMIT, concentrated at low SNR and ≤0.006 above 9 dB | `BCCTN/results_May21/` + `main_model/results_May21/` + `per_snr_locked_paper_table.csv` | The FP32 reference is the **3.97 M original** (fft 512 / embed 512), so the gap is downsizing **+** arithmetic, not quantization alone. The middle curve splits it: downsizing+FP8/QAT −0.0120/−0.0148, the W4A8 zero-multiplier core only −0.0042/−0.0068. Never cite the total as "quantization loss" — and note the middle checkpoint is already FP8/QAT (epoch=134, `*_quant_256dim`), not a downsized FP32 model |
| fig14 | FP32-ORIG + FP8-QAT-BASELINE + **DEPLOY** | Same three checkpoints on SegSNR / ILD / IPD: SegSNR pays ~1.2 dB, spatial cues stay at FP32 parity | same as fig13 | ILD/IPD deltas are ±0.1 dB / ±1.7° — inside the band where the deploy sometimes **beats** FP32 (TIMIT ILD −0.084 dB). Supports the "spatial cues are the USP" claim; SegSNR is the metric that actually pays |

Not regenerated: the old `fig08_analysis_decision_map` (conceptual flow diagram) — draw
natively in Visio instead; all numeric content for it is in the table above.

Style: validated colorblind-safe palette (dataviz reference instance); no dual axes;
Arial-family text kept editable in SVG/EMF.
