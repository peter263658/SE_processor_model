#!/bin/bash
# 09_weight_tied_mha — fine-tune from baseline (tied-init checkpoint)
# Loads baseline_tied_init.ckpt (= avg(MHA_re, MHA_im) of epoch=134-step=130680.ckpt)
# Runs on GPU 0.

set -e
cd "$(dirname "$0")"

export PYTHONPATH="$PYTHONPATH:$(pwd)"
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0

mkdir -p logs Checkpoints
LOGFILE=logs/train_finetune_$(date +%Y%m%d_%H%M%S).log

echo "Tied ComplexMHA fine-tune"
echo "  GPU         : ${CUDA_VISIBLE_DEVICES}"
echo "  Init from   : baseline_tied_init.ckpt"
echo "  Log         : ${LOGFILE}"
echo

conda run --no-capture-output -n bcctn python -u train.py \
    training.train_checkpoint_path=/raid/R12K41024/09_weight_tied_mha/baseline_tied_init.ckpt \
    training.learning_rate=5e-4 \
    training.n_epochs=80 \
    hydra.run.dir=outputs/finetune/\${now:%Y-%m-%d_%H-%M-%S} \
    2>&1 | tee "${LOGFILE}"
