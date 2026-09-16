#!/usr/bin/env bash
# Paper-style eval of the official BCCTN checkpoint with the SAME settings as the deploy evals
# (750 pairs x 8 SNR, VCTK + TIMIT, 1 GPU, 32 CPU metric workers, dir-18 metric code).
#   GPU_ID=<free> [VCTK_ONLY=0] [LIMIT=750] [METRIC_WORKERS=32] [TAG=official_ckpt_full] bash run_eval.sh
set -euo pipefail
D=/raid/R12K41024/23_official_bcctn_ckpt_eval
GPU_ID="${GPU_ID:-1}"; LIMIT="${LIMIT:-750}"; METRIC_WORKERS="${METRIC_WORKERS:-32}"; VCTK_ONLY="${VCTK_ONLY:-0}"
TAG="${TAG:-official_ckpt_full}"; OUT="${OUT:-$D/results/$TAG}"; mkdir -p "$OUT"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache}" MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl_cache}"
timit_args=()
[ "$VCTK_ONLY" = "1" ] || timit_args=( --timit_test_dir /raid/R12K41024/BCCTN/Dataset/noisy_testset_timit
                                        --timit_clean_dir /raid/R12K41024/BCCTN/Dataset/clean_testset_timit )
echo "[official-eval] gpu=$GPU_ID limit=$LIMIT workers=$METRIC_WORKERS vctk_only=$VCTK_ONLY out=$OUT"
CUDA_VISIBLE_DEVICES="$GPU_ID" conda run --no-capture-output -n bcctn python "$D/code/eval_official.py" \
  --vctk_test_dir /raid/R12K41024/BCCTN/Dataset/noisy_testset \
  --vctk_clean_dir /raid/R12K41024/BCCTN/Dataset/clean_testset \
  "${timit_args[@]}" \
  --output_dir "$OUT" --limit_pairs "$LIMIT" --metric_workers "$METRIC_WORKERS" \
  2>&1 | tee "$OUT/eval.log"
echo; echo "[official-eval] VCTK:"; grep -i 'MBSTOI Enhanced' "$OUT/vctk_paper_style_results_averages.txt" || true
echo "[official-eval] TIMIT:"; grep -i 'MBSTOI Enhanced' "$OUT/timit_paper_style_results_averages.txt" 2>/dev/null || true
