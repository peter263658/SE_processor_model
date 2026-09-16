#!/bin/bash
# Train both mono-SE baselines sequentially on one GPU (repo GPU rule).
set -e
cd "$(dirname "$0")/code"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

mamba run -n bcctn python train_mono.py --arch snu --output_dir ../results/snu \
    2>&1 | tee ../logs/train_snu.log
mamba run -n bcctn python train_mono.py --arch ntu --output_dir ../results/ntu \
    2>&1 | tee ../logs/train_ntu.log
echo "ALL TRAINING DONE"
