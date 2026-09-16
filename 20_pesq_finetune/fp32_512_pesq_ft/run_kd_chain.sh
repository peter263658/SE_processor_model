#!/usr/bin/env bash
# KD chain (single GPU, sequential):
#   1) teacher (saturated FP32-512 w12) generates enhanced targets for train/val
#   2) deploy-arch QAT finetune with teacher outputs as targets (+ pesq_weight 6)
#   3) eval best ckpt vs REAL clean (gates + PESQ), dir-20 canonical chain
set -uo pipefail
GPU="${GPU:-5}"
ROOT=/raid/R12K41024/20_pesq_finetune/fp32_512_pesq_ft
LOG=$ROOT/queue_kd.log
exec >>"$LOG" 2>&1
export PATH=/home/R12K41024/miniforge3/bin:$PATH
export NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/mpl_cache
PYENV=/home/R12K41024/miniforge3/envs/bcctn/bin/python

echo "=== [KD-1] teacher target generation $(date) ==="
cd $ROOT/code
CUDA_VISIBLE_DEVICES=$GPU $PYENV kd_gen_teacher_targets.py
echo "=== [KD-1] exit=$? $(date) ==="
N=$(ls /raid/R12K41024/20_pesq_finetune/data/kd_teacher_w12sat/trainset/*.wav 2>/dev/null | wc -l)
[ "$N" -lt 30000 ] && { echo "[KD-1] FATAL: only $N teacher targets"; exit 1; }

echo "=== [KD-2] student QAT finetune on teacher targets $(date) ==="
PESQ_WEIGHT=6 LR=2e-5 N_EPOCHS=20 GPU_ID=$GPU RUN_TAG=kd_w12sat_pesq6 \
  CLEAN_TRAIN=/raid/R12K41024/20_pesq_finetune/data/kd_teacher_w12sat/trainset \
  CLEAN_VAL=/raid/R12K41024/20_pesq_finetune/data/kd_teacher_w12sat/valset \
  bash /raid/R12K41024/20_pesq_finetune/code/run_pesq_finetune.sh
echo "=== [KD-2] exit=$? $(date) ==="

echo "=== [KD-3] eval best ckpt vs real clean $(date) ==="
CKDIR=$(for d in $(ls -d /raid/R12K41024/20_pesq_finetune/outputs/pesq_finetune/kd_w12sat_pesq6/ft/logs/lightning_logs/version_*/checkpoints 2>/dev/null | sort -Vr); do
  ls "$d"/*epoch=*.ckpt >/dev/null 2>&1 && { echo "$d"; break; }
done)
BEST=$($PYENV - "$CKDIR" <<'PY'
import sys, torch, glob, os
ckdir = sys.argv[1]
best = ""
last = os.path.join(ckdir, "last.ckpt")
if os.path.exists(last):
    try:
        st = torch.load(last, map_location="cpu")
        for k, v in (st.get("callbacks") or {}).items():
            if "ModelCheckpoint" in str(k) and isinstance(v, dict):
                best = v.get("best_model_path") or ""
    except Exception:
        best = ""
if not best or not os.path.exists(best):
    cands = [p for p in glob.glob(os.path.join(ckdir, "*epoch=*.ckpt"))
             if "all-" not in os.path.basename(p)]
    best = max(cands, key=os.path.getmtime) if cands else ""
print(best)
PY
)
[ -z "$BEST" ] && { echo "[KD-3] FATAL: no ckpt"; exit 1; }
echo "[KD-3] best=$BEST"
CKPT="$BEST" TAG=kdverify_w12sat GPU_ID=$GPU LIMIT=750 \
  bash /raid/R12K41024/20_pesq_finetune/code/run_pesq_eval.sh
echo "=== [KD-3] exit=$? $(date) ==="
echo "=== KD chain COMPLETE $(date) ==="
touch $ROOT/KD_DONE
