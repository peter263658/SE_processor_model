MODEL_CHECKPOINT="/raid/R12K41024/BCCTN_sep_conv/outputs/2025-06-02/17-24-32/logs/lightning_logs/version_0/checkpoints/epoch=128-step=124872.ckpt" 

python analyze_model_weights.py --model_path "$MODEL_CHECKPOINT" --output_dir ./analysis \
  --save_plots --save_json --save_report