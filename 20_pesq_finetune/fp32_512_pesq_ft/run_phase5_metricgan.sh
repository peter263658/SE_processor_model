#!/usr/bin/env bash
# Phase 5: when the w=12 continuation (bait_run) finishes, launch MetricGAN
# PESQ finetuning on GPU 7, warm-starting from the continuation's best ckpt.
set -uo pipefail
GPU="${GPU:-7}"
LOG=/raid/R12K41024/20_pesq_finetune/fp32_512_pesq_ft/queue_phase5.log
exec >>"$LOG" 2>&1
export PATH=/home/R12K41024/miniforge3/bin:$PATH
export NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/mpl_cache

echo "=== phase5 waiting for bait_run (w12 continuation) to finish $(date) ==="
while pgrep -f "hydra.run.dir=.*bait_run" >/dev/null; do sleep 120; done
echo "=== phase5 picking warm-start ckpt $(date) ==="

CKDIR=$(ls -d /raid/R12K41024/20_pesq_finetune/fp32_512_pesq_ft/outputs/bait_run/logs/lightning_logs/version_*/checkpoints 2>/dev/null | sort -V | tail -1)
BEST=$(conda run -n bcctn python - "$CKDIR" <<'PY'
import sys, torch, glob, os
ckdir = sys.argv[1]
best = ""
last = os.path.join(ckdir, "last.ckpt")
if os.path.exists(last):
    st = torch.load(last, map_location="cpu")
    for k, v in (st.get("callbacks") or {}).items():
        if "ModelCheckpoint" in str(k) and isinstance(v, dict):
            best = v.get("best_model_path") or ""
if not best or not os.path.exists(best):
    cands = glob.glob(os.path.join(ckdir, "epoch=*.ckpt"))
    best = max(cands, key=os.path.getmtime) if cands else ""
print(best)
PY
)
[ -z "$BEST" ] && { echo "phase5: no ckpt found"; exit 1; }
LINK=/raid/R12K41024/20_pesq_finetune/fp32_512_pesq_ft/warmstart_metricgan.ckpt
ln -sf "$BEST" "$LINK"
echo "=== phase5 launching MetricGAN from $BEST $(date) ==="

cd /raid/R12K41024/20_pesq_finetune/fp32_512_pesq_ft/code
CUDA_VISIBLE_DEVICES=$GPU conda run --no-capture-output -n bcctn python train_metricgan.py \
  +model.gan_weight=12 \
  training.learning_rate=2e-5 \
  training.n_epochs=15 \
  training.train_checkpoint_path="$LINK" \
  hydra.run.dir=/raid/R12K41024/20_pesq_finetune/fp32_512_pesq_ft/outputs/metricgan_w12 \
  > /raid/R12K41024/20_pesq_finetune/fp32_512_pesq_ft/train_metricgan.log 2>&1
echo "=== phase5 MetricGAN exit=$? $(date) ==="
touch /raid/R12K41024/20_pesq_finetune/fp32_512_pesq_ft/PHASE5_DONE
