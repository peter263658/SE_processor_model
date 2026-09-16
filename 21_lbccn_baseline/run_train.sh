#!/usr/bin/env bash
# Train LBCCN (official model/loss) on the BCCTN binaural dataset.
# Box rule: run exactly ONE compute job (train OR eval) at a time.
set -u
cd "$(dirname "$0")/code"
GPU=${GPU:-5}
export CUDA_VISIBLE_DEVICES=$GPU
PY=/home/R12K41024/miniforge3/envs/bcctn/bin/python
echo "=== LBCCN train start $(date) on GPU $GPU ==="
$PY train.py "$@"
rc=$?
echo "=== LBCCN train exit=$rc $(date) ==="
exit $rc
