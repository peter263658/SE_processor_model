#!/bin/bash
# 09_weight_tied_mha — STRICT W4A8 QAT from-scratch
#   - tied_mha=true            (already in config)
#   - qat_scope=full           (override)
#   - top_k_attention=4        (override)
#   - random init (no checkpoint)
#
# Shares GPU 0 with run_train_finetune_strict.sh — H100 80GB has headroom for both.

set -e
cd "$(dirname "$0")"

export PYTHONPATH="$PYTHONPATH:$(pwd)"
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0

mkdir -p logs Checkpoints
LOGFILE=logs/train_scratch_strict_$(date +%Y%m%d_%H%M%S).log

echo "Tied ComplexMHA STRICT scratch (W4A8 full-scope + top-k=4, random init)"
echo "  GPU         : ${CUDA_VISIBLE_DEVICES} (shared with finetune_strict)"
echo "  Init        : random"
echo "  Log         : ${LOGFILE}"
echo

conda run --no-capture-output -n bcctn python -u train.py \
    training.train_checkpoint_path=null \
    training.qat_scope=full \
    training.top_k_attention=4 \
    training.learning_rate=1e-3 \
    training.n_epochs=150 \
    hydra.run.dir=outputs/scratch_strict/\${now:%Y-%m-%d_%H-%M-%S} \
    2>&1 | tee "${LOGFILE}"
