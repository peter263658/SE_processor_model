# Project map

Directory names are retained because experiment scripts and configurations reference them. Numbering reflects research history; some numbers have multiple experiments.

| Stage | Directories / entry points |
|---|---|
| Original and architectural baselines | `BCCTN/`, `BCCTN_sep_conv/`, `BCCTN_sep_conv_causal/`, `BCCTN_sep_conv_causal_slide_win/`, `BCCTN_sep_conv_causal_slide_win_quant_256dim/`, `BCCTN_hardware/`, `main_model/` |
| Utilization, bitwidth, range, sparsity, reuse | `01_utilization_analysis/`, `01_utilization_analysis_8khz/`, `02_weights_bitwidth/`, `03_dynamic_range/`, `04_sparsity_analysis/`, `05_conv_reuse/` |
| Mixed precision and FP8 | `06_attn_mised_precision/`, `07_mixed_precision_qat/`, `08_mixed_precision_qat_from_scratch/`, `08_all_fp8_ptq/` |
| Weight tying and hardware export | [09_weight_tied_mha](../09_weight_tied_mha/README.md), `09_software_to_hardware/` |
| STFT and architecture optimization | [10_stft_fixpoint_ptq](../10_stft_fixpoint_ptq/README.md), [11_weight_tied_mha_fast_mbstoi](../11_weight_tied_mha_fast_mbstoi/README.md), [12_decoder_grouped_pointwise](../12_decoder_grouped_pointwise/README.md) |
| Quantization and quality recovery | [13_aggressive_algo_qat](../13_aggressive_algo_qat/README.md), [14_low_power_arch_qat](../14_low_power_arch_qat/README.md), [15_all_lmul_mixed_mult_eval](../15_all_lmul_mixed_mult_eval/README.md), [16_kd_heatmap_recovery](../16_kd_heatmap_recovery/README.md), [17_soft_lmul_ste_recovery](../17_soft_lmul_ste_recovery/README.md) |
| Deploy model and handoff | [18 model](../18_w4a8_pow2_mixed_lmul/README.md), [18 export package](../18_w4a8_pow2_mixed_lmul_hw_handoff_20260530/README.md), [19 final hardware package](../19_algo_final/README.md) |
| Quality and baseline comparisons | [20 PESQ](../20_pesq_finetune/README.md), [21 LBCCN](../21_lbccn_baseline/README.md), [21 monaural baselines](../21_mono_se_baselines/README.md), [22 directional interferer](../22_directional_interferer/README.md), [23 official checkpoint](../23_official_bcctn_ckpt_eval/README.md) |
| Parameter progression and retraining | [24 parameter progression](../24_param_mbstoi_progression/README.md), [25 official retraining](../25_official_bcctn_retrain/README.md) |
| Paper material | `paper_figures/`, `journal_sync_2026-07-07/`, root feature summaries and verification notes |

## Files supplied separately

- Data: local `Dataset/` trees and experiment-specific dataset paths.
- Models: local checkpoints (`.ckpt`, `.pt`, `.pth`) and training output.
- Tooling: Python/CUDA environments and commercial RTL simulation/lint tools.

Selected result summaries under experiment `docs/` directories are versioned. Full `results/`, `outputs/`, and logs are generally ignored. A path mentioned in an older experiment README may describe a local run artifact rather than a file available in a fresh clone.
