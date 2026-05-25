#!/bin/bash
# 09_weight_tied_mha — STRICT W4A8 QAT fine-tune
#   - tied_mha=true            (already in config)
#   - qat_scope=full           (override: conv ALSO gets W4A8, not just MHA)
#   - top_k_attention=4        (override: top-4 sparse softmax)
#   - int4_group_size=32       (default)
#
# Init from baseline_tied_init.ckpt.
# Runs on GPU 0.

set -e
cd "$(dirname "$0")"

export PYTHONPATH="$PYTHONPATH:$(pwd)"
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0

mkdir -p logs Checkpoints
LOGFILE=logs/train_finetune_strict_$(date +%Y%m%d_%H%M%S).log

echo "Tied ComplexMHA STRICT fine-tune (W4A8 full-scope + top-k=4)"
echo "  GPU         : ${CUDA_VISIBLE_DEVICES}"
echo "  Init from   : baseline_tied_init.ckpt"
echo "  Log         : ${LOGFILE}"
echo

conda run --no-capture-output -n bcctn python -u train.py \
    training.train_checkpoint_path=/raid/R12K41024/09_weight_tied_mha/baseline_tied_init.ckpt \
    training.qat_scope=full \
    training.top_k_attention=4 \
    training.learning_rate=5e-4 \
    training.n_epochs=80 \
    hydra.run.dir=outputs/finetune_strict/\${now:%Y-%m-%d_%H-%M-%S} \
    2>&1 | tee "${LOGFILE}"
