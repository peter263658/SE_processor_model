#!/bin/bash

# Sparsity analysis for BCCTN (dim=256, num_heads=16)

MODEL_CHECKPOINT="/raid/R12K41024/BCCTN_sep_conv_causal_slide_win_quant_256dim/outputs/2026-03-30/15-18-17/logs/lightning_logs/version_0/checkpoints/epoch=134-step=130680.ckpt"
TEST_DIR="/raid/R12K41024/BCCTN/Dataset/noisy_testset"
NUM_SAMPLES=20
RESULTS_DIR="./results_sparsity"

echo "=== BCCTN Sparsity Analysis ==="
echo "Checkpoint: ${MODEL_CHECKPOINT}"
echo "Test dir:   ${TEST_DIR}"
echo "Samples:    ${NUM_SAMPLES}"
echo "Output:     ${RESULTS_DIR}"
echo ""

cd /raid/R12K41024/04_sparsity_analysis

CUDA_VISIBLE_DEVICES=0 python sparsity_analyzer.py \
    --checkpoint "$MODEL_CHECKPOINT" \
    --test_dir "$TEST_DIR" \
    --num_samples "$NUM_SAMPLES" \
    --results_dir "$RESULTS_DIR"

echo ""
echo "=== Done. Results in ${RESULTS_DIR} ==="
