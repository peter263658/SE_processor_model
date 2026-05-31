#!/usr/bin/env bash
# Eval wrapper for plain saturating e4m3 activation. Only activation FP8 policy
# differs from the current deploy path.
set -euo pipefail

PKG=/raid/R12K41024/18_w4a8_pow2_mixed_lmul_hw_handoff_20260530

export ACT_FP8_MODE="${ACT_FP8_MODE:-plain_e4m3}"
export BCCTN_ACTIVATION_FP8_MODE="$ACT_FP8_MODE"
export TAG="${TAG:-plain_e4m3_$(basename "${CKPT:?set CKPT=<path>}" .ckpt)}"
export OUT="${OUT:-${PKG}/results/plain_e4m3_activation_eval/${TAG}}"

exec bash "${PKG}/run_apot_descale_eval.sh"
