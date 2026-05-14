#!/bin/bash
# PTQ sweep — all-FP8 attention variant
# ----------------------------------------
# * conv layers : FP8 (qat_scope=mha_only)
# * MHA weights : INT4 per-head (W4A8)
# * INT4 dequant scale : FP8 (E4M3)  ← new vs fp16-accum run
# * QK^T / softmax / Score×V operands : FP8 (E4M3)  ← softmax output re-quantized
# * matmul accumulators : FP16 (unchanged vs fp16-accum run)
#
# Baseline to compare against: ptq_full_scope_topk4_gsz256_fp16accum_eval / mha_only variants.

MODEL_CHECKPOINT="/raid/R12K41024/08_mixed_precision_qat_from_scratch/outputs/2026-04-19/00-35-08/logs/lightning_logs/version_0/checkpoints/epoch=7-step=7744.ckpt"

VCTK_NOISY_DIR="/raid/R12K41024/BCCTN/Dataset/noisy_testset"
VCTK_CLEAN_DIR="/raid/R12K41024/BCCTN/Dataset/clean_testset"
TIMIT_NOISY_DIR="/raid/R12K41024/BCCTN/Dataset/noisy_testset_timit"
TIMIT_CLEAN_DIR="/raid/R12K41024/BCCTN/Dataset/clean_testset_timit"
OUTPUT_DIR="./ptq_all_fp8_attention_eval"

mkdir -p "$OUTPUT_DIR"

echo "Running paper-style eval with fp8_attention=true + fp8_scale=true ..."
CUDA_VISIBLE_DEVICES=5 python eval.py \
    --model_checkpoint "$MODEL_CHECKPOINT" \
    --vctk_test_dir    "$VCTK_NOISY_DIR" \
    --vctk_clean_dir   "$VCTK_CLEAN_DIR" \
    --timit_test_dir   "$TIMIT_NOISY_DIR" \
    --timit_clean_dir  "$TIMIT_CLEAN_DIR" \
    --output_dir       "${OUTPUT_DIR}/paper_style" \
    --paper_style_eval \
    --batch_size 8 \
    --limit_pairs 750 \
    --hydra_overrides \
        training.qat_scope=mha_only \
        +training.fp8_attention=true \
        +training.fp8_scale=true
