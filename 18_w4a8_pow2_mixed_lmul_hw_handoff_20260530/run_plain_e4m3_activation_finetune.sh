#!/usr/bin/env bash
# Recovery finetune for the hardware-preferred activation policy:
#   A8 activation = plain saturating e4m3, no activation amax/scale/descale.
#
# This wrapper changes only BCCTN_ACTIVATION_FP8_MODE and the warm-start ckpt.
# Weight quantization, grouped decoder, MHA projection grouping, BN/PReLU APoT,
# and decoder outlier delayed APoT weight descale stay identical to the current
# deploy path.
set -euo pipefail

PKG=/raid/R12K41024/18_w4a8_pow2_mixed_lmul_hw_handoff_20260530

export START_CKPT="${START_CKPT:-${PKG}/ckpt/strict_core_apot_descale_deploy_ep2_vctk0.8365_timit0.7912.ckpt}"
export ACT_FP8_MODE="${ACT_FP8_MODE:-plain_e4m3}"
export BCCTN_ACTIVATION_FP8_MODE="$ACT_FP8_MODE"
export RUN_TAG="${RUN_TAG:-plain_e4m3_activation_from_ep2_lr${LR:-1e-5}_$(date +%Y%m%d_%H%M%S)}"

exec bash "${PKG}/run_apot_descale_finetune.sh"
