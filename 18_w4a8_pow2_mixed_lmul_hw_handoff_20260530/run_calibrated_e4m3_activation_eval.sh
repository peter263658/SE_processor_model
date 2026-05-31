#!/usr/bin/env bash
# Eval wrapper for method B calibrated_e4m3 activation.
set -euo pipefail

PKG=/raid/R12K41024/18_w4a8_pow2_mixed_lmul_hw_handoff_20260530

ACTIVATION_CALIB_PATH="${ACTIVATION_CALIB_PATH:-${BCCTN_ACTIVATION_CALIB_PATH:-}}"
[ -n "$ACTIVATION_CALIB_PATH" ] || {
  echo "[calib-eval] FATAL: set ACTIVATION_CALIB_PATH=<activation_calibration.json>" >&2
  exit 1
}
[ -f "$ACTIVATION_CALIB_PATH" ] || {
  echo "[calib-eval] FATAL: missing calibration json $ACTIVATION_CALIB_PATH" >&2
  exit 1
}

export ACT_FP8_MODE="${ACT_FP8_MODE:-calibrated_e4m3}"
export BCCTN_ACTIVATION_FP8_MODE="$ACT_FP8_MODE"
export BCCTN_ACTIVATION_CALIB_PATH="$ACTIVATION_CALIB_PATH"

echo "[calib-eval] activation_calib_path=$BCCTN_ACTIVATION_CALIB_PATH"
exec bash "${PKG}/run_apot_descale_eval.sh"
