#!/bin/bash

# MODEL_CHECKPOINT="/raid/R12K41024/BCCTN_sep_conv_causal/outputs/2025-06-06/03-45-36/logs/lightning_logs/version_0/checkpoints/epoch=42-step=41624.ckpt"  # causal 
# MODEL_CHECKPOINT="/raid/R12K41024/BCCTN_sep_conv_causal/outputs/2025-06-12/16-10-22/logs/lightning_logs/version_0/checkpoints/epoch=159-step=154880.ckpt" # causal with kernel_size=(self.kernel_size, 1)
# MODEL_CHECKPOINT="/raid/R12K41024/BCCTN_sep_conv_causal/outputs/2025-06-12/23-58-57/logs/lightning_logs/version_0/checkpoints/epoch=198-step=192632.ckpt" # causal with kernel_size=(self.kernel_size, 1) and fft = 128
# MODEL_CHECKPOINT="/raid/R12K41024/BCCTN_sep_conv_causal/outputs/2025-06-15/16-49-13/logs/lightning_logs/version_0/checkpoints/epoch=60-step=59048.ckpt"
MODEL_CHECKPOINT="/raid/R12K41024/BCCTN_sep_conv_causal/outputs/2025-06-16/00-32-24/logs/lightning_logs/version_0/checkpoints/epoch=142-step=138424.ckpt"

VCTK_NOISY_DIR="/raid/R12K41024/BCCTN/Dataset/noisy_testset"
VCTK_CLEAN_DIR="/raid/R12K41024/BCCTN/Dataset/clean_testset"
# VCTK_NOISY_DIR="/raid/R12K41024/LBCCN/Dataset/noisy_testset"
# VCTK_CLEAN_DIR="/raid/R12K41024/LBCCN/Dataset/clean_testset"
TIMIT_NOISY_DIR="/raid/R12K41024/BCCTN/Dataset/noisy_testset_timit"
TIMIT_CLEAN_DIR="/raid/R12K41024/BCCTN/Dataset/clean_testset_timit"
OUTPUT_DIR="./results_May21"  # Define output directory

# Create results directory
mkdir -p "$OUTPUT_DIR"

# Step 1: Run paper-style evaluation with fixed SNR levels on VCTK dataset (matched condition)
echo "Running paper-style evaluation on VCTK dataset (matched condition)..."
CUDA_VISIBLE_DEVICES=5 python eval.py \
    --model_checkpoint "$MODEL_CHECKPOINT" \
    --vctk_test_dir "$VCTK_NOISY_DIR" \
    --vctk_clean_dir "$VCTK_CLEAN_DIR" \
    --timit_test_dir "$TIMIT_NOISY_DIR" \
    --timit_clean_dir "$TIMIT_CLEAN_DIR" \
    --output_dir "${OUTPUT_DIR}/vctk_paper_style" \
    --paper_style_eval \
    --batch_size 8 \
    --limit_pairs 750

# CUDA_VISIBLE_DEVICES=5 python eval.py \
#     --model_checkpoint "$MODEL_CHECKPOINT" \
#     --vctk_test_dir "$VCTK_NOISY_DIR" \
#     --vctk_clean_dir "$VCTK_CLEAN_DIR" \
#     --output_dir "${OUTPUT_DIR}/vctk_paper_style" \
#     --paper_style_eval \
#     --batch_size 8 \
#     --limit_pairs 750

# # Step 3: Generate comparison table between our results and paper results
# echo "Generating comparison table..."
# CUDA_VISIBLE_DEVICES=5 python compare_results.py \
#     --vctk_results "${OUTPUT_DIR}/vctk_paper_style/vctk_paper_style_results.csv" \
#     --timit_results "${OUTPUT_DIR}/timit_paper_style/timit_paper_style_results.csv" \
#     --output_file "${OUTPUT_DIR}/comparison_with_paper.csv"

# echo "Evaluation complete! Results are saved in ${OUTPUT_DIR}"