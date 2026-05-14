#!/bin/bash
# Eval strict W4A8 checkpoint with STFT/iSTFT floating-format PTQ.
#
# Usage:
#   ./run_eval_stft_float_precision.sh [fp32|fp16xfp16|fp8xfp8] [checkpoint] [output_dir] [gpu_id] [hydra overrides...]
#
# Optional env:
#   LIMIT_PAIRS=750 BATCH_SIZE=8 ./run_eval_stft_float_precision.sh fp16xfp16
set -e
cd "$(dirname "$0")"

MODE="${1:-fp16xfp16}"
DEFAULT_CHECKPOINT="/raid/R12K41024/09_weight_tied_mha/outputs/finetune_strict/2026-05-10_02-40-16/logs/lightning_logs/version_0/checkpoints/epoch=45-step=44528.ckpt"
CHECKPOINT="${2:-$DEFAULT_CHECKPOINT}"
GPU_ID="${4:-5}"
LIMIT_PAIRS="${LIMIT_PAIRS:-750}"
BATCH_SIZE="${BATCH_SIZE:-8}"

case "$MODE" in
    fp32|float32)
        FORMAT="fp32"
        ENABLE_FLOAT="false"
        ;;
    fp16|float16|fp16xfp16)
        FORMAT="fp16"
        ENABLE_FLOAT="true"
        ;;
    fp8|fp8_e4m3|e4m3|fp8xfp8)
        FORMAT="fp8_e4m3"
        ENABLE_FLOAT="true"
        ;;
    fp8_e5m2|e5m2)
        FORMAT="fp8_e5m2"
        ENABLE_FLOAT="true"
        ;;
    *)
        FORMAT="$MODE"
        ENABLE_FLOAT="true"
        ;;
esac

OUTPUT_DIR="${3:-results/stft_${MODE}}"
EXTRA_OVERRIDES=("${@:5}")

VCTK_NOISY_DIR="/raid/R12K41024/BCCTN/Dataset/noisy_testset"
VCTK_CLEAN_DIR="/raid/R12K41024/BCCTN/Dataset/clean_testset"

mkdir -p "$OUTPUT_DIR" /tmp/numba_cache /tmp/mpl_cache

echo "STFT/iSTFT floating-format PTQ eval"
echo "  Mode        : $MODE ($FORMAT)"
echo "  Checkpoint  : $CHECKPOINT"
echo "  Output dir  : $OUTPUT_DIR"
echo "  GPU         : $GPU_ID"
echo "  Limit pairs : $LIMIT_PAIRS per SNR"

export PYTHONPATH="$PYTHONPATH:$(pwd)"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl_cache}"

CUDA_VISIBLE_DEVICES=$GPU_ID conda run --no-capture-output -n bcctn python eval.py \
    --model_checkpoint "$CHECKPOINT" \
    --vctk_test_dir "$VCTK_NOISY_DIR" \
    --vctk_clean_dir "$VCTK_CLEAN_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --paper_style_eval \
    --batch_size "$BATCH_SIZE" \
    --limit_pairs "$LIMIT_PAIRS" \
    --hydra_overrides \
        training.qat_scope=full \
        training.top_k_attention=4 \
        model.stft_fixed_point.enabled=false \
        model.stft_float_precision.enabled="$ENABLE_FLOAT" \
        model.stft_float_precision.format="$FORMAT" \
        "${EXTRA_OVERRIDES[@]}"
