#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

CONDA_ENV="${CONDA_ENV:-bcctn}"
PACKAGE_ROOT="$(cd .. && pwd)"
OUTDIR="${1:-${PACKAGE_ROOT}/export_out_strict18}"

# Override the source ckpt/config. Empty => eh.DEFAULT_CKPT, which is the current
# deploy = pow2 block-float ep3 (strict_core_pow2_blockfloat_deploy_ep3_...).
# Pass CKPT=... to dump a different ckpt.
CKPT="${CKPT:-}"
CONFIG="${CONFIG:-}"
ckpt_args=()
[ -n "$CKPT" ] && ckpt_args+=(--ckpt "$CKPT")
[ -n "$CONFIG" ] && ckpt_args+=(--config "$CONFIG")
if [ -n "$CKPT" ]; then echo "[run_output] ckpt override: $CKPT"; else echo "[run_output] using eh.DEFAULT_CKPT (pow2 block-float deploy ep3)"; fi

export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl_cache}"
export PYTHONPATH="${PYTHONPATH:-}:${PACKAGE_ROOT}/code"
export BCCTN_APOT_A_MIN="${BCCTN_APOT_A_MIN:--16}"
export BCCTN_APOT_A_MAX="${BCCTN_APOT_A_MAX:--1}"
export BCCTN_APOT_B_MIN="${BCCTN_APOT_B_MIN:--16}"
export BCCTN_APOT_B_MAX="${BCCTN_APOT_B_MAX:--1}"
# golden_dump runs the model forward -> force the TRUE deployed path:
#  (a) pow2 BLOCK-FLOAT activations (raw e4m3 code + per-tensor pow2 shared exponent;
#      no runtime amax/scale) -- eager==fused, so golden == hardware; and
#  (b) the delayed §3.4 FP8-outlier APoT weight descale, range [-16,-6].
# (ckpt2hwdat/dump_hw_config read weights only, but inherit these for consistency.)
export BCCTN_ACTIVATION_FP8_MODE="${BCCTN_ACTIVATION_FP8_MODE:-calibrated_pow2_e4m3}"
export BCCTN_ACTIVATION_CALIB_PATH="${BCCTN_ACTIVATION_CALIB_PATH:-${PACKAGE_ROOT}/ckpt/activation_calibration_p999_b64.json}"
export BCCTN_ACTIVATION_POW2_ROUND="${BCCTN_ACTIVATION_POW2_ROUND:-floor}"
export BCCTN_OUTLIER_DELAYED_DESCALE="${BCCTN_OUTLIER_DELAYED_DESCALE:-1}"
export BCCTN_OUTLIER_DESCALE_APOT_MIN="${BCCTN_OUTLIER_DESCALE_APOT_MIN:--16}"
export BCCTN_OUTLIER_DESCALE_APOT_MAX="${BCCTN_OUTLIER_DESCALE_APOT_MAX:--6}"
echo "[run_output] activation=${BCCTN_ACTIVATION_FP8_MODE} (pow2_round=${BCCTN_ACTIVATION_POW2_ROUND}); golden delayed APoT descale=${BCCTN_OUTLIER_DELAYED_DESCALE} range [${BCCTN_OUTLIER_DESCALE_APOT_MIN},${BCCTN_OUTLIER_DESCALE_APOT_MAX}]"

conda run --no-capture-output -n "${CONDA_ENV}" python ckpt2hwdat.py \
  --outdir "${OUTDIR}" \
  --overwrite "${ckpt_args[@]}"

conda run --no-capture-output -n "${CONDA_ENV}" python dump_hw_config.py \
  --out "${OUTDIR}/layer_config.json" "${ckpt_args[@]}"

conda run --no-capture-output -n "${CONDA_ENV}" python golden_dump.py \
  --outdir "${OUTDIR}/golden" "${ckpt_args[@]}"

# Regenerate the block-float TB patterns (raw e4m3 code + fused weight descales)
# from the freshly-dumped export, unless SKIP_TB=1.
if [ "${SKIP_TB:-0}" != "1" ]; then
  tb_ckpt_args=()
  [ -n "$CKPT" ] && tb_ckpt_args+=(--ckpt "$CKPT")
  [ -n "$CONFIG" ] && tb_ckpt_args+=(--config "$CONFIG")
  conda run --no-capture-output -n "${CONDA_ENV}" python prepare_tb_patterns.py \
    --export-root "${OUTDIR}" "${tb_ckpt_args[@]}"
fi

echo "[OK] wrote ${OUTDIR}"
