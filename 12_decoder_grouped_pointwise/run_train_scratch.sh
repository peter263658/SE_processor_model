#!/bin/bash
# 09_weight_tied_mha — train from scratch (random init)
# No checkpoint, full 150-epoch budget at lr=1e-3.
# Runs on GPU 2.

set -e
cd "$(dirname "$0")"

export PYTHONPATH="$PYTHONPATH:$(pwd)"
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=2

mkdir -p logs Checkpoints
LOGFILE=logs/train_scratch_$(date +%Y%m%d_%H%M%S).log

echo "Tied ComplexMHA train-from-scratch"
echo "  GPU         : ${CUDA_VISIBLE_DEVICES}"
echo "  Init        : random (config defaults)"
echo "  Log         : ${LOGFILE}"
echo

conda run --no-capture-output -n bcctn python -u train.py \
    training.train_checkpoint_path=null \
    training.learning_rate=1e-3 \
    training.n_epochs=150 \
    hydra.run.dir=outputs/scratch/\${now:%Y-%m-%d_%H-%M-%S} \
    2>&1 | tee "${LOGFILE}"
