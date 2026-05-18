#!/bin/bash
# Eval the latest 09 strict checkpoint with Taylor2 or ReLU2 attention.
#
# Usage:
#   bash softmax_rtl_taylor_relu/run_eval_exp_free_strict.sh [taylor2|relu2] [checkpoint] [output_dir] [gpu_id]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_DIR"

MODE="${1:-taylor2}"
if [ "$MODE" != "taylor2" ] && [ "$MODE" != "relu2" ]; then
    echo "Mode must be taylor2 or relu2"
    exit 1
fi

DEFAULT_CKPT="${REPO_DIR}/outputs/finetune_strict/2026-05-10_02-40-16/logs/lightning_logs/version_0/checkpoints/epoch=45-step=44528.ckpt"
CHECKPOINT="${2:-$DEFAULT_CKPT}"
OUTPUT_DIR="${3:-${REPO_DIR}/softmax_rtl_taylor_relu/results/${MODE}_strict}"
GPU_ID="${4:-5}"
LIMIT_PAIRS="${LIMIT_PAIRS:-750}"
BATCH_SIZE="${BATCH_SIZE:-8}"
METRIC_WORKERS="${METRIC_WORKERS:-1}"

mkdir -p "$OUTPUT_DIR"

echo "Eval 09 strict + ${MODE} RTL attention"
echo "  Checkpoint  : $CHECKPOINT"
echo "  Output dir  : $OUTPUT_DIR"
echo "  GPU         : $GPU_ID"
echo "  Limit pairs : $LIMIT_PAIRS"
echo "  Metric workers: $METRIC_WORKERS"

CUDA_VISIBLE_DEVICES=$GPU_ID conda run --no-capture-output -n bcctn python eval.py \
    --model_checkpoint "$CHECKPOINT" \
    --vctk_test_dir /raid/R12K41024/BCCTN/Dataset/noisy_testset \
    --vctk_clean_dir /raid/R12K41024/BCCTN/Dataset/clean_testset \
    --output_dir "$OUTPUT_DIR" \
    --paper_style_eval \
    --batch_size "$BATCH_SIZE" \
    --limit_pairs "$LIMIT_PAIRS" \
    --metric_workers "$METRIC_WORKERS" \
    --hydra_overrides \
        training.qat_scope=full \
        training.top_k_attention=4 \
        model.attention_softmax.mode="$MODE"
