#!/bin/bash

# Mixed-Precision W4A8 QAT fine-tune from FP8 pre-trained checkpoint.
# See 07_mixed_precision_qat_handoff.md §Step 4.

export PYTHONPATH="$PYTHONPATH:$(pwd)"
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=5

mkdir -p ./Checkpoints

echo "Starting Mixed-Precision W4A8 QAT fine-tune..."
echo "  Resume from : $(grep train_checkpoint_path config/config.yaml)"
echo "  LR          : $(grep '^  learning_rate:' config/config.yaml)"
echo "  Epochs      : $(grep '^  n_epochs:' config/config.yaml)"
echo "  top_k       : $(grep '^  top_k_attention:' config/config.yaml)"

python train.py

echo "Training complete!"