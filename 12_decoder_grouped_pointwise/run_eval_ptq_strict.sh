#!/bin/bash
# Apply STRICT W4A8 PTQ at eval-time on a checkpoint trained with the lighter
# (mha_only / no top-k) QAT spec. Hydra overrides re-build the model with:
#   - qat_scope=full     (conv layers also W4A8, not just MHA)
#   - top_k_attention=4  (top-4 sparse softmax)
# Tied MHA is already baked in via the ckpt.
#
# Usage:
#   ./run_eval_ptq_strict.sh <checkpoint_path> <output_dir> [gpu_id]
set -e
cd "$(dirname "$0")"

CHECKPOINT="$1"
OUTPUT_DIR="$2"
GPU_ID="${3:-5}"

if [ -z "$CHECKPOINT" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: $0 <checkpoint_path> <output_dir> [gpu_id]"
    exit 1
fi

VCTK_NOISY_DIR="/raid/R12K41024/BCCTN/Dataset/noisy_testset"
VCTK_CLEAN_DIR="/raid/R12K41024/BCCTN/Dataset/clean_testset"

mkdir -p "$OUTPUT_DIR"

echo "PTQ eval (strict): full-scope W4A8 + top-k=4 + tied MHA"
echo "  Checkpoint  : $CHECKPOINT"
echo "  Output dir  : $OUTPUT_DIR"
echo "  GPU         : $GPU_ID"
echo "  Datasets    : VCTK only"

CUDA_VISIBLE_DEVICES=$GPU_ID conda run --no-capture-output -n bcctn python eval.py \
    --model_checkpoint "$CHECKPOINT" \
    --vctk_test_dir "$VCTK_NOISY_DIR" \
    --vctk_clean_dir "$VCTK_CLEAN_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --paper_style_eval \
    --batch_size 8 \
    --limit_pairs 750 \
    --hydra_overrides training.qat_scope=full training.top_k_attention=4
