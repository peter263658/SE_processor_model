#!/usr/bin/env bash
# Combined KD + spatial/PESQ objective chain (single GPU):
#   1) teacher (saturated FP32-512 w12) targets for train/val (skips existing)
#   2) deploy-arch QAT finetune: targets = teacher outputs,
#      weights ILD 10 / IPD 10 / PESQ 12 / STOI 2 / SNR 1
#   3) eval best ckpt vs REAL clean (gates + PESQ + ILD/IPD, dir-20 chain)
set -uo pipefail
GPU="${GPU:-5}"
ROOT=/raid/R12K41024/20_pesq_finetune/fp32_512_pesq_ft
LOG=$ROOT/queue_kdspatial.log
exec >>"$LOG" 2>&1
export PATH=/home/R12K41024/miniforge3/bin:$PATH
export NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/mpl_cache
PYENV=/home/R12K41024/miniforge3/envs/bcctn/bin/python

echo "=== [KS-1] teacher target generation (skip existing) $(date) ==="
cd $ROOT/code
CUDA_VISIBLE_DEVICES=$GPU $PYENV kd_gen_teacher_targets.py
echo "=== [KS-1] exit=$? $(date) ==="
N=$(find /raid/R12K41024/20_pesq_finetune/data/kd_teacher_w12sat/trainset -name "*.wav" 2>/dev/null | wc -l)
[ "$N" -lt 30000 ] && { echo "[KS-1] FATAL: only $N teacher targets"; exit 1; }

echo "=== [KS-2] student QAT ft: KD targets + ILD10/IPD10/PESQ12/STOI2 $(date) ==="
# Supervisor loop: the process cleaner only SIGTERMs pythons; this bash loop
# survives and resumes training from the newest last.ckpt of THIS run tag.
CKGLOB=/raid/R12K41024/20_pesq_finetune/outputs/pesq_finetune/kdspatial_pesq12_ild10/ft/logs/lightning_logs
for attempt in 1 2 3 4 5 6; do
  RESUME=$(ls -t "$CKGLOB"/version_*/checkpoints/last.ckpt 2>/dev/null | head -1)
  if [ -n "$RESUME" ]; then export START_CKPT="$RESUME"; echo "[KS-2] attempt $attempt resume=$RESUME"; else echo "[KS-2] attempt $attempt fresh start"; fi
  PESQ_WEIGHT=12 ILD_W=10 IPD_W=10 STOI_W=2 SNR_W=1 \
    LR=2e-5 N_EPOCHS=20 GPU_ID=$GPU RUN_TAG=kdspatial_pesq12_ild10 \
    CLEAN_TRAIN=/raid/R12K41024/20_pesq_finetune/data/kd_teacher_w12sat/trainset \
    CLEAN_VAL=/raid/R12K41024/20_pesq_finetune/data/kd_teacher_w12sat/valset \
    bash /raid/R12K41024/20_pesq_finetune/code/run_pesq_finetune.sh
  rc=$?
  echo "[KS-2] attempt $attempt exit=$rc $(date)"
  [ $rc -eq 0 ] && break
  sleep 20
done
echo "=== [KS-2] done $(date) ==="

echo "=== [KS-3] eval best ckpt vs real clean $(date) ==="
CKDIR=$(for d in $(ls -d /raid/R12K41024/20_pesq_finetune/outputs/pesq_finetune/kdspatial_pesq12_ild10/ft/logs/lightning_logs/version_*/checkpoints 2>/dev/null | sort -Vr); do
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
[ -z "$BEST" ] && { echo "[KS-3] FATAL: no ckpt"; exit 1; }
echo "[KS-3] best=$BEST"
CKPT="$BEST" TAG=kdspatialverify GPU_ID=$GPU LIMIT=750 \
  bash /raid/R12K41024/20_pesq_finetune/code/run_pesq_eval.sh
echo "=== [KS-3] exit=$? $(date) ==="
echo "=== KD+spatial chain COMPLETE $(date) ==="
touch $ROOT/KDSPATIAL_DONE
