#!/usr/bin/env bash
# Recovery finetune for method B:
#   A8 activation = e4m3 with offline calibrated fixed per-boundary scales.
set -euo pipefail

PKG=/raid/R12K41024/18_w4a8_pow2_mixed_lmul_hw_handoff_20260530

export START_CKPT="${START_CKPT:-${PKG}/ckpt/strict_core_apot_descale_deploy_ep2_vctk0.8365_timit0.7912.ckpt}"
ACTIVATION_CALIB_PATH="${ACTIVATION_CALIB_PATH:-${BCCTN_ACTIVATION_CALIB_PATH:-}}"
[ -n "$ACTIVATION_CALIB_PATH" ] || {
  echo "[calib-ft] FATAL: set ACTIVATION_CALIB_PATH=<activation_calibration.json>" >&2
  exit 1
}
[ -f "$ACTIVATION_CALIB_PATH" ] || {
  echo "[calib-ft] FATAL: missing calibration json $ACTIVATION_CALIB_PATH" >&2
  exit 1
}

export ACT_FP8_MODE="${ACT_FP8_MODE:-calibrated_e4m3}"
export BCCTN_ACTIVATION_FP8_MODE="$ACT_FP8_MODE"
export BCCTN_ACTIVATION_CALIB_PATH="$ACTIVATION_CALIB_PATH"
export RUN_TAG="${RUN_TAG:-calibrated_e4m3_activation_from_ep2_lr${LR:-1e-5}_$(date +%Y%m%d_%H%M%S)}"

echo "[calib-ft] activation_calib_path=$BCCTN_ACTIVATION_CALIB_PATH"
exec bash "${PKG}/run_apot_descale_finetune.sh"
