#!/usr/bin/env bash
set -euo pipefail

cd /raid/R12K41024/BCCTN_hardware

# 設定 PYTHONPATH 確保 package 可以被正確匯入
export PYTHONPATH="/raid/R12K41024/BCCTN_hardware:${PYTHONPATH:-}"

# 確保必要的 __init__.py 檔案存在
echo "🔧 確保 package 結構完整..."
mkdir -p DCNN/models
touch DCNN/__init__.py
touch DCNN/models/__init__.py

echo "🚀 啟動 PTQ Export (硬編碼版本)..."
CUDA_VISIBLE_DEVICES=4 python ptq_export.py \
  --model-import DCNN/models/binaural_attention_model.py \
  --model-class BinauralAttentionDCNN \
  --weights  /raid/R12K41024/BCCTN_hardware/model.ckpt \
  --calib-dir /raid/R12K41024/BCCTN/Dataset/noisy_valset \
  --eval-dir /raid/R12K41024/BCCTN/Dataset \
  --eval-clean-dir /raid/R12K41024/BCCTN/Dataset/clean_testset \
  --eval-noisy-dir /raid/R12K41024/BCCTN/Dataset/noisy_testset \
  --output-root artifacts \
  --run-id ptq_$(date +%Y%m%d_%H%M%S) \
  --stream-dump per-frame \
  --attn-window 32 \
  --device cuda \
  --eval-entry eval:compute_mbstoi \
  --dump-complex-split true

echo "✅ PTQ Export 完成！"