#!/usr/bin/env bash
# One restructuring point: TAG=<name> CODE_DIR=<variant dir> RUN_DIR=<hydra run dir> CKPT=<ckpt> GPU_ID=<free> [LIMIT=750] [METRIC_WORKERS=32] [VCTK_ONLY=0] bash run_point.sh
set -euo pipefail
D=/raid/R12K41024/24_param_mbstoi_progression
: "${TAG:?}" "${CODE_DIR:?}" "${RUN_DIR:?}" "${CKPT:?}"
GPU_ID="${GPU_ID:-1}"; LIMIT="${LIMIT:-750}"; METRIC_WORKERS="${METRIC_WORKERS:-32}"; VCTK_ONLY="${VCTK_ONLY:-0}"
OUT="${OUT:-$D/results/$TAG}"; mkdir -p "$OUT"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache}" MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl_cache}"
timit_args=()
[ "$VCTK_ONLY" = "1" ] || timit_args=( --timit_test_dir /raid/R12K41024/BCCTN/Dataset/noisy_testset_timit --timit_clean_dir /raid/R12K41024/BCCTN/Dataset/clean_testset_timit )
echo "[point] $TAG gpu=$GPU_ID limit=$LIMIT workers=$METRIC_WORKERS out=$OUT"
CUDA_VISIBLE_DEVICES="$GPU_ID" conda run --no-capture-output -n bcctn python "$D/code/eval_legacy_ckpt.py" \
  --code_dir "$CODE_DIR" --run_dir "$RUN_DIR" --ckpt "$CKPT" \
  --vctk_test_dir /raid/R12K41024/BCCTN/Dataset/noisy_testset --vctk_clean_dir /raid/R12K41024/BCCTN/Dataset/clean_testset \
  "${timit_args[@]}" --output_dir "$OUT" --limit_pairs "$LIMIT" --metric_workers "$METRIC_WORKERS" --ignore_missing_prefixes "${IGNORE_MISSING:-}" 2>&1 | tee "$OUT/eval.log"
grep -i 'MBSTOI Enhanced' "$OUT"/*_paper_style_results_averages.txt || true
