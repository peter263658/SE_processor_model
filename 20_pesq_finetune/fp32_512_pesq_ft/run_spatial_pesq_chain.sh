#!/usr/bin/env bash
# Spatial+PESQ objective chain (single GPU): deploy-arch QAT finetune with the
# table-winning axes as the objective — ILD 10 / IPD 10 / PESQ 12, STOI anchor
# reduced to 2 (kept nonzero to avoid the w30-style collapse), SNR 1 — then
# eval best ckpt vs real clean (canonical dir-20 chain).
set -uo pipefail
GPU="${GPU:-5}"
ROOT=/raid/R12K41024/20_pesq_finetune/fp32_512_pesq_ft
LOG=$ROOT/queue_spatial.log
exec >>"$LOG" 2>&1
export PATH=/home/R12K41024/miniforge3/bin:$PATH
export NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/mpl_cache
PYENV=/home/R12K41024/miniforge3/envs/bcctn/bin/python

echo "=== [SP-1] deploy QAT ft, ILD10/IPD10/PESQ12/STOI2/SNR1 $(date) ==="
PESQ_WEIGHT=12 ILD_W=10 IPD_W=10 STOI_W=2 SNR_W=1 \
  LR=2e-5 N_EPOCHS=20 GPU_ID=$GPU RUN_TAG=spatial_pesq12_ild10_stoi2 \
  bash /raid/R12K41024/20_pesq_finetune/code/run_pesq_finetune.sh
echo "=== [SP-1] exit=$? $(date) ==="

echo "=== [SP-2] eval best ckpt $(date) ==="
CKDIR=$(for d in $(ls -d /raid/R12K41024/20_pesq_finetune/outputs/pesq_finetune/spatial_pesq12_ild10_stoi2/ft/logs/lightning_logs/version_*/checkpoints 2>/dev/null | sort -Vr); do
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
[ -z "$BEST" ] && { echo "[SP-2] FATAL: no ckpt"; exit 1; }
echo "[SP-2] best=$BEST"
CKPT="$BEST" TAG=spatialverify_pesq12 GPU_ID=$GPU LIMIT=750 \
  bash /raid/R12K41024/20_pesq_finetune/code/run_pesq_eval.sh
echo "=== [SP-2] exit=$? $(date) ==="
echo "=== spatial chain COMPLETE $(date) ==="
touch $ROOT/SPATIAL_DONE
