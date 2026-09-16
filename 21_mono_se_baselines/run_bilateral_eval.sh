#!/bin/bash
# Bilateral MBSTOI/ILD/IPD eval of a mono-SE baseline on the deploy考卷
# (same datasets / limit / batch as 18_.../run_apot_descale_eval.sh).
#
#   CKPT=<best_*.pt> TAG=<snu|ntu> GPU_ID=<free> bash run_bilateral_eval.sh
#
# MASK_MODE=shared applies the mean of the two ears' masks to both ears, so the
# mono network stops altering the inter-ear level ratio (the reviewer's "just
# share one mask" counter-proposal).  Default independent = per-ear masks.
set -euo pipefail

PKG=/raid/R12K41024/21_mono_se_baselines
cd "$PKG/code"

CKPT="${CKPT:?set CKPT=<mono baseline .pt>}"
TAG="${TAG:?set TAG=<snu|ntu>}"
GPU_ID="${GPU_ID:-1}"; BATCH_SIZE="${BATCH_SIZE:-8}"; LIMIT="${LIMIT:-750}"
METRIC_WORKERS="${METRIC_WORKERS:-32}"; VCTK_ONLY="${VCTK_ONLY:-0}"
MASK_MODE="${MASK_MODE:-independent}"

OUT="$PKG/results/bilateral_eval/${TAG}"
mkdir -p "$OUT" "$PKG/logs"
LOG="$PKG/logs/bilateral_eval_${TAG}.log"

# Override to score a different acoustic condition (e.g. the directional-
# interferer set in 22_directional_interferer) on the same clean targets.
VCTK_TEST_DIR="${VCTK_TEST_DIR:-/raid/R12K41024/BCCTN/Dataset/noisy_testset}"
VCTK_CLEAN_DIR="${VCTK_CLEAN_DIR:-/raid/R12K41024/BCCTN/Dataset/clean_testset}"

timit_args=()
if [ "$VCTK_ONLY" != "1" ]; then
  timit_args=( --timit_test_dir /raid/R12K41024/BCCTN/Dataset/noisy_testset_timit
               --timit_clean_dir /raid/R12K41024/BCCTN/Dataset/clean_testset_timit )
fi

CUDA_VISIBLE_DEVICES="$GPU_ID" BCCTN_MONO_MASK_MODE="$MASK_MODE" \
  conda run --no-capture-output -n bcctn python eval_bilateral.py \
  --model_checkpoint "$CKPT" \
  --vctk_test_dir "$VCTK_TEST_DIR" \
  --vctk_clean_dir "$VCTK_CLEAN_DIR" \
  "${timit_args[@]}" \
  --output_dir "$OUT" \
  --paper_style_eval --batch_size "$BATCH_SIZE" --limit_pairs "$LIMIT" \
  --metric_workers "$METRIC_WORKERS" \
  2>&1 | tee "$LOG"

echo ""; echo "[bilateral-eval:${TAG}] VCTK:"
grep -i 'MBSTOI Enhanced' "${OUT}/vctk_paper_style_results_averages.txt" 2>/dev/null || true
echo "[bilateral-eval:${TAG}] TIMIT:"
grep -i 'MBSTOI Enhanced' "${OUT}/timit_paper_style_results_averages.txt" 2>/dev/null || true
