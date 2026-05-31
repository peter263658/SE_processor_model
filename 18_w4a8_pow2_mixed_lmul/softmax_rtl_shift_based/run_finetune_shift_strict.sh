#!/bin/bash
# Fine-tune from the latest 09 strict checkpoint with shift-based RTL softmax.
#
# Usage:
#   bash softmax_rtl_shift_based/run_finetune_shift_strict.sh [checkpoint] [gpu_id]
#
# Optional env:
#   MODE=shift | shift_topk_pow2 | shift_topk_pow2_floor
#        | shift_topk_pow2_ceil | shift_topk_pow2_clamp8
#        | shift_topk_pow2_nomantissa
#        | shift_topk_pow2_clamp8_nomantissa
#        | floor | ceil | clamp8 | mantissa_free | clamp8_nomantissa

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_DIR"

DEFAULT_CKPT="${REPO_DIR}/outputs/finetune_strict/2026-05-10_02-40-16/logs/lightning_logs/version_0/checkpoints/epoch=45-step=44528.ckpt"
MODE="${MODE:-shift}"
CHECKPOINT="${1:-$DEFAULT_CKPT}"
GPU_ID="${2:-0}"
EPOCHS="${EPOCHS:-30}"
LR="${LR:-1e-4}"
RUN_NAME="${MODE}_strict_${EPOCHS}ep"
LOGFILE="${REPO_DIR}/softmax_rtl_shift_based/${RUN_NAME}.log"

export PYTHONPATH="$PYTHONPATH:$REPO_DIR"
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES="$GPU_ID"

mkdir -p "${REPO_DIR}/softmax_rtl_shift_based"

echo "Fine-tune 09 strict + ${MODE} RTL softmax"
echo "  Checkpoint  : $CHECKPOINT"
echo "  GPU         : $GPU_ID"
echo "  Epochs      : $EPOCHS"
echo "  LR          : $LR"
echo "  Log         : $LOGFILE"

conda run --no-capture-output -n bcctn python -u train.py \
    training.train_checkpoint_path="$CHECKPOINT" \
    training.qat_scope=full \
    training.top_k_attention=4 \
    training.learning_rate="$LR" \
    training.n_epochs="$EPOCHS" \
    model.attention_softmax.mode="$MODE" \
    hydra.run.dir="outputs/softmax_rtl_${MODE}/\${now:%Y-%m-%d_%H-%M-%S}" \
    2>&1 | tee "$LOGFILE"
