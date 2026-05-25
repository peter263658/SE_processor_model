#!/bin/bash
# Eval a tied-MHA checkpoint with paper-style metrics.
# Defaults to VCTK-only. Pass `--with-timit` (or env SKIP_TIMIT=0) to also
# evaluate on TIMIT.
#
# Usage:
#   ./run_eval.sh <checkpoint_path> <output_dir> [gpu_id] [--with-timit]
set -e
cd "$(dirname "$0")"

CHECKPOINT="$1"
OUTPUT_DIR="$2"
GPU_ID="${3:-5}"
WITH_TIMIT="${4:-}"

if [ -z "$CHECKPOINT" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: $0 <checkpoint_path> <output_dir> [gpu_id] [--with-timit]"
    exit 1
fi

VCTK_NOISY_DIR="/raid/R12K41024/BCCTN/Dataset/noisy_testset"
VCTK_CLEAN_DIR="/raid/R12K41024/BCCTN/Dataset/clean_testset"

TIMIT_ARGS=()
if [ "$WITH_TIMIT" = "--with-timit" ] || [ "$SKIP_TIMIT" = "0" ]; then
    TIMIT_ARGS=(--timit_test_dir /raid/R12K41024/BCCTN/Dataset/noisy_testset_timit
                --timit_clean_dir /raid/R12K41024/BCCTN/Dataset/clean_testset_timit)
    echo "  Datasets    : VCTK + TIMIT"
else
    echo "  Datasets    : VCTK only (TIMIT skipped)"
fi

mkdir -p "$OUTPUT_DIR"

echo "Eval tied-MHA checkpoint"
echo "  Checkpoint  : $CHECKPOINT"
echo "  Output dir  : $OUTPUT_DIR"
echo "  GPU         : $GPU_ID"

CUDA_VISIBLE_DEVICES=$GPU_ID conda run --no-capture-output -n bcctn python eval.py \
    --model_checkpoint "$CHECKPOINT" \
    --vctk_test_dir "$VCTK_NOISY_DIR" \
    --vctk_clean_dir "$VCTK_CLEAN_DIR" \
    "${TIMIT_ARGS[@]}" \
    --output_dir "$OUTPUT_DIR" \
    --paper_style_eval \
    --batch_size 8 \
    --limit_pairs 750
