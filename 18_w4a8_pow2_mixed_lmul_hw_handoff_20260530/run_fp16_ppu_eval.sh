#!/usr/bin/env bash
# fp16 conv-PPU MBSTOI eval (REQUEST_FP16_PPU_EVAL_2026-06-10, deliverable #1/#2).
#
# Runs the deploy config (pow2 block-float, VCTK 0.8357 / TIMIT 0.7898) with the
# conv post-MAC PPU forced into fp16 via BCCTN_FP16_PPU. Segments:
#   tree  : 32-way fp16 saturating MAC tree + 4->2 complex OACC
#   bn    : BN dealer (APoT alpha) + fp16 beta, fp16 saturating add
#   prelu : PReLU pow2 slope, fp16  (numerically inert: pow2 shift is exact)
# Off (BCCTN_FP16_PPU="" or "off") reproduces the fp32 deploy baseline bit-for-bit.
#
# NOTE: the decoder FP8-outlier stages (S0/S3/S4) L-Mul multiply stays fp32 here
# (no overflow) — their fp16 overflow is the SEPARATE preshift re-bake blocker
# (REQUEST 追加 section), not this conv-PPU question.
#
#   SEG=all|off|tree,bn,prelu  GPU_ID=<free> VCTK_ONLY=0 bash run_fp16_ppu_eval.sh
set -euo pipefail
PKG=/raid/R12K41024/18_w4a8_pow2_mixed_lmul_hw_handoff_20260530

SEG="${SEG:-all}"
case "$SEG" in
  all)  PPU="tree,bn,prelu" ;;
  off|none|"") PPU="" ;;
  *)    PPU="$SEG" ;;
esac
export BCCTN_FP16_PPU="$PPU"
export BCCTN_ACTIVATION_CALIB_PATH="${BCCTN_ACTIVATION_CALIB_PATH:-${PKG}/ckpt/activation_calibration_p999_b64.json}"

SEGTAG=$(echo "${PPU:-off}" | tr ',' '+')
export CKPT="${CKPT:-${PKG}/ckpt/strict_core_pow2_blockfloat_deploy_ep3_vctk0.8357_timit0.7898.ckpt}"
export ACT_FP8_MODE=calibrated_pow2_e4m3
export VCTK_ONLY="${VCTK_ONLY:-0}"
export GPU_ID="${GPU_ID:-1}"
export METRIC_WORKERS="${METRIC_WORKERS:-32}"
export OUT="${PKG}/results/fp16_ppu_eval/seg_${SEGTAG}"
export LOG="${OUT}/eval.log"
export TAG="fp16ppu_${SEGTAG}"
mkdir -p "$OUT"

echo "[fp16-ppu-eval] BCCTN_FP16_PPU='${PPU}' (seg=${SEGTAG})  VCTK_ONLY=${VCTK_ONLY}  GPU=${GPU_ID}"
echo "[fp16-ppu-eval] baseline to beat: VCTK 0.8357 / TIMIT 0.7898 (= SEG=off)"
echo "[fp16-ppu-eval] out=$OUT"
bash "${PKG}/run_apot_descale_eval.sh"
