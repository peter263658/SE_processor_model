#!/bin/bash
# Re-score the original full-size BCCTN (win 400 / fft 512) with the CURRENT eval
# code. The stored July results are still valid for MBSTOI/STOI/PESQ/IPD -- their
# noisy baselines match ours to 4 decimals -- but their ILD was produced by an
# older version of compute_ild_error_with_mask (noisy ILD 3.3072 against the
# current 3.7141), so that column cannot be placed next to the mono-baseline
# numbers without this re-run.
#
#   GPU_ID=3 bash run_fp32_512_reeval.sh
set -uo pipefail
S=/raid/R12K41024/20_pesq_finetune/fp32_512_pesq_ft/code
PY=/home/R12K41024/miniforge3/envs/bcctn/bin/python
GPU="${GPU_ID:-3}"
OUT=/raid/R12K41024/20_pesq_finetune/results/pesq_eval/fp32_512_ref_recheck_2026-08-27
cd "$S"
CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 "$PY" eval.py \
  --model_checkpoint /raid/R12K41024/BCCTN/outputs/2025-05-10/19-39-07/logs/lightning_logs/version_0/checkpoints/epoch=98-step=95832.ckpt \
  --vctk_test_dir /raid/R12K41024/BCCTN/Dataset/noisy_testset \
  --vctk_clean_dir /raid/R12K41024/BCCTN/Dataset/clean_testset \
  --timit_test_dir /raid/R12K41024/BCCTN/Dataset/noisy_testset_timit \
  --timit_clean_dir /raid/R12K41024/BCCTN/Dataset/clean_testset_timit \
  --output_dir "$OUT" \
  --paper_style_eval --batch_size 8 --limit_pairs 750
echo "FP32_512_REEVAL_EXIT=$?"
echo "FP32_512_REEVAL_DONE"
