#!/bin/bash
# Train the ORIGINAL-resolution BCCTN (win 400 / fft 512) from scratch, keeping
# the CAUSAL conv kernel (5,1)/padding(2,0).
#
# Not a fine-tune of the author's checkpoint: that checkpoint was trained with a
# (5,2)/padding(2,1) time kernel, which sees one future frame. Adopting it to
# match shapes would have silently discarded the causality the chip claims
# (~8 ms against NTU's ~32 ms), so the weights are discarded and only the
# resolution and capacity of the original are kept.
#
#   COND=dirnoise90 GPU_ID=3 [EPOCHS=60] bash run_512_scratch.sh
#   COND=isotropic  ...                             # matched control arm
set -uo pipefail
PKG=/raid/R12K41024/22_directional_interferer
PY=/home/R12K41024/miniforge3/envs/bcctn/bin/python
DATA=/raid/R12K41024/BCCTN/Dataset
GPU="${GPU_ID:-3}"; EPOCHS="${EPOCHS:-60}"; COND="${COND:-dirnoise90}"
if [ "$COND" = isotropic ]; then
  TRAIN="$DATA/noisy_trainset"; VAL="$DATA/noisy_valset"
else
  TRAIN="$DATA/noisy_trainset_$COND"; VAL="$DATA/noisy_valset_$COND"
fi
OUTD="$PKG/outputs/scratch512_${COND}_$(date +%Y-%m-%d_%H-%M-%S)"
cd "$PKG/code512"
echo "=== scratch-512 (causal) on $COND: $(find "$TRAIN" -name '*.wav' | wc -l) mixes, $EPOCHS epochs ==="
CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="${PYTHONPATH:-}:$PKG/code512" HYDRA_FULL_ERROR=1 \
  "$PY" -u train.py \
    training.n_epochs="$EPOCHS" \
    dataset.noisy_training_dataset_dir="$TRAIN" \
    dataset.noisy_validation_dataset_dir="$VAL" \
    hydra.run.dir="$OUTD"
echo "SCRATCH512_${COND}_EXIT=$?"
echo "$OUTD" > "$PKG/results/scratch512_${COND}_outdir.txt"
echo "SCRATCH512_${COND}_DONE"
