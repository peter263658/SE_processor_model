#!/usr/bin/env bash
# Spatial-rebalance continuation of plain-QAT w12 (single GPU, self-healing):
# warm-start from qatverify ep10 (wb 1.866 / ILD 3.49), raise ILD 1->4 and
# IPD 10->12 with STOI kept at 10, PESQ kept at 12, low lr refinement.
# Goal: claw back spatial damage while holding most of the PESQ gain.
set -uo pipefail
GPU="${GPU:-5}"
ROOT=/raid/R12K41024/20_pesq_finetune/fp32_512_pesq_ft
LOG=$ROOT/queue_rebal.log
exec >>"$LOG" 2>&1
export PATH=/home/R12K41024/miniforge3/bin:$PATH
export NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/mpl_cache
PYENV=/home/R12K41024/miniforge3/envs/bcctn/bin/python
TAG=qatw12_rebal_ild4_ipd12

WARM=/raid/R12K41024/20_pesq_finetune/outputs/pesq_finetune/qatverify_w12_lr2e-5_ep20/ft/logs/lightning_logs/version_0/checkpoints/epoch=10-step=5324-val=21.6478.ckpt

echo "=== [RB-1] rebalance ft: ILD4/IPD12/STOI10/PESQ12 lr1e-5 $(date) ==="
CKGLOB=/raid/R12K41024/20_pesq_finetune/outputs/pesq_finetune/$TAG/ft/logs/lightning_logs
for attempt in 1 2 3 4 5 6; do
  RESUME=$(ls -t "$CKGLOB"/version_*/checkpoints/last.ckpt 2>/dev/null | head -1)
  if [ -n "$RESUME" ]; then export START_CKPT="$RESUME"; echo "[RB-1] attempt $attempt resume=$RESUME"
  else export START_CKPT="$WARM"; echo "[RB-1] attempt $attempt warm=$WARM"; fi
  PESQ_WEIGHT=12 ILD_W=4 IPD_W=12 STOI_W=10 SNR_W=1 \
    LR=1e-5 N_EPOCHS=15 GPU_ID=$GPU RUN_TAG=$TAG \
    bash /raid/R12K41024/20_pesq_finetune/code/run_pesq_finetune.sh
  rc=$?
  echo "[RB-1] attempt $attempt exit=$rc $(date)"
  [ $rc -eq 0 ] && break
  sleep 20
done

echo "=== [RB-2] eval best ckpt $(date) ==="
CKDIR=$(for d in $(ls -d "$CKGLOB"/version_*/checkpoints 2>/dev/null | sort -Vr); do
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
[ -z "$BEST" ] && { echo "[RB-2] FATAL: no ckpt"; exit 1; }
echo "[RB-2] best=$BEST"
for i in 1 2 3; do
  CKPT="$BEST" TAG=qatrebal_verify GPU_ID=$GPU LIMIT=750 \
    bash /raid/R12K41024/20_pesq_finetune/code/run_pesq_eval.sh
  [ -s /raid/R12K41024/20_pesq_finetune/results/pesq_eval/qatrebal_verify/vctk_paper_style_results_averages.txt ] && break
  sleep 15
done
echo "=== rebalance chain COMPLETE $(date) ==="
touch $ROOT/REBAL_DONE
