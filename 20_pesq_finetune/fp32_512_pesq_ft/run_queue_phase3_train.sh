#!/usr/bin/env bash
# Phase 3: wait for the GPU-6 eval queue (C -> D -> B) to finish, then run the
# FP32-512 PESQ finetune (w=12) on the SAME GPU 6 — the only card outside the
# labmate launcher's kill set {1,3,5,7}. Keeps us at 1 GPU total.
set -uo pipefail
GPU="${GPU:-6}"
MARK=/raid/R12K41024/20_pesq_finetune/fp32_512_pesq_ft/QUEUE_DONE
LOG=/raid/R12K41024/20_pesq_finetune/fp32_512_pesq_ft/queue_phase3.log
exec >>"$LOG" 2>&1
export PATH=/home/R12K41024/miniforge3/bin:$PATH
export NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/mpl_cache

echo "=== phase3 waiting for QUEUE_DONE $(date) ==="
until [ -f "$MARK" ]; do sleep 60; done
echo "=== phase3 start training w=12 on GPU $GPU $(date) ==="

cd /raid/R12K41024/20_pesq_finetune/fp32_512_pesq_ft/code
CUDA_VISIBLE_DEVICES=$GPU conda run --no-capture-output -n bcctn python train.py \
  +model.pesq_weight=12 \
  training.learning_rate=2e-5 \
  training.n_epochs=20 \
  training.early_stopping.patience_in_epochs=6 \
  training.train_checkpoint_path=/raid/R12K41024/20_pesq_finetune/fp32_512_pesq_ft/warmstart_fp32_512_ep98.ckpt \
  hydra.run.dir=/raid/R12K41024/20_pesq_finetune/fp32_512_pesq_ft/outputs/pesqft_w12_lr2e-5_20ep \
  > /raid/R12K41024/20_pesq_finetune/fp32_512_pesq_ft/train_w12.log 2>&1
echo "=== phase3 training exit=$? $(date) ==="
touch /raid/R12K41024/20_pesq_finetune/fp32_512_pesq_ft/PHASE3_DONE
