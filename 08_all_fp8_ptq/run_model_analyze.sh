
VCTK_NOISY_DIR="/raid/R12K41024/BCCTN/Dataset/noisy_testset"
MODEL_CHECKPOINT="/raid/R12K41024/BCCTN_sep_conv_causal_slide_win/epoch=89-step=87120.ckpt"

python model_analysis.py \
    --model_checkpoint  "$MODEL_CHECKPOINT"  \
    --data_path "$VCTK_NOISY_DIR"  \
    --output_dir ./analysis_results