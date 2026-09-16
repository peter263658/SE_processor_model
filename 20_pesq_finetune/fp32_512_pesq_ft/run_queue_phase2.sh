#!/usr/bin/env bash
# Phase 2: wait for the C/D queue (run_queue_eval.sh) to finish, then re-run
# Run B (FP32-512 reference full eval) which was SIGTERMed by the 01:05 VPN
# teardown sweep. Runs on the same GPU so eval stays at <=1 GPU.
set -uo pipefail
GPU="${GPU:-6}"
MARK=/raid/R12K41024/20_pesq_finetune/fp32_512_pesq_ft/QUEUE_DONE
LOG=/raid/R12K41024/20_pesq_finetune/fp32_512_pesq_ft/queue_phase2.log
exec >>"$LOG" 2>&1
export PATH=/home/R12K41024/miniforge3/bin:$PATH
export NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/mpl_cache

echo "=== phase2 waiting for QUEUE_DONE $(date) ==="
until [ -f "$MARK" ]; do sleep 60; done
echo "=== phase2 start Run B $(date) ==="

cd /raid/R12K41024/20_pesq_finetune/fp32_512_pesq_ft/code
CUDA_VISIBLE_DEVICES=$GPU conda run --no-capture-output -n bcctn python eval.py \
  --model_checkpoint /raid/R12K41024/BCCTN/outputs/2025-05-10/19-39-07/logs/lightning_logs/version_0/checkpoints/epoch=98-step=95832.ckpt \
  --vctk_test_dir /raid/R12K41024/BCCTN/Dataset/noisy_testset \
  --vctk_clean_dir /raid/R12K41024/BCCTN/Dataset/clean_testset \
  --timit_test_dir /raid/R12K41024/BCCTN/Dataset/noisy_testset_timit \
  --timit_clean_dir /raid/R12K41024/BCCTN/Dataset/clean_testset_timit \
  --output_dir /raid/R12K41024/20_pesq_finetune/results/pesq_eval/fp32_512_ref_full \
  --paper_style_eval --batch_size 8 --limit_pairs 750
echo "=== phase2 Run B exit=$? $(date) ==="
touch /raid/R12K41024/20_pesq_finetune/fp32_512_pesq_ft/PHASE2_DONE
