#!/usr/bin/env bash
# Phase 6: when MetricGAN training ends, evaluate BOTH its best-val and last
# checkpoints on VCTK (all 8 SNRs, parallel metrics). GPU 6 (after queue+ftq).
set -uo pipefail
GPU="${GPU:-6}"
LOG=/raid/R12K41024/20_pesq_finetune/fp32_512_pesq_ft/queue_phase6.log
exec >>"$LOG" 2>&1
export PATH=/home/R12K41024/miniforge3/bin:$PATH
export NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/mpl_cache

echo "=== phase6 waiting for MetricGAN training to end $(date) ==="
sleep 5
while pgrep -f "hydra.run.dir=.*mgan_cont" >/dev/null; do sleep 180; done
# also wait for the eval GPU to be free (queue + ftq finished)
echo "=== phase6 waiting for ftq/queue to release GPU $GPU $(date) ==="
until [ -f /raid/R12K41024/20_pesq_finetune/fp32_512_pesq_ft/PHASE4_DONE ]; do sleep 120; done

CKDIR=$(ls -d /raid/R12K41024/20_pesq_finetune/fp32_512_pesq_ft/outputs/mgan_cont1/logs/lightning_logs/version_*/checkpoints 2>/dev/null | sort -V | tail -1)
[ -z "$CKDIR" ] && { echo "phase6: no ckpt dir"; exit 1; }
BEST=$(/home/R12K41024/miniforge3/envs/bcctn/bin/python - "$CKDIR" <<'PY'
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
cd /raid/R12K41024/20_pesq_finetune/fp32_512_pesq_ft/code
for TAG in bestval last; do
  if [ "$TAG" = bestval ]; then SRC="$BEST"; else SRC="$CKDIR/last.ckpt"; fi
  [ -z "$SRC" ] && continue
  LINK=/raid/R12K41024/20_pesq_finetune/fp32_512_pesq_ft/mgan_${TAG}.ckpt
  ln -sf "$SRC" "$LINK"
  echo "=== phase6 evaluating $TAG = $SRC $(date) ==="
  CUDA_VISIBLE_DEVICES=$GPU conda run --no-capture-output -n bcctn python eval.py \
    --model_checkpoint "$LINK" \
    --vctk_test_dir /raid/R12K41024/BCCTN/Dataset/noisy_testset \
    --vctk_clean_dir /raid/R12K41024/BCCTN/Dataset/clean_testset \
    --output_dir /raid/R12K41024/20_pesq_finetune/results/pesq_eval/fp32_512_metricgan_${TAG} \
    --paper_style_eval --batch_size 8 --limit_pairs 750
  echo "=== phase6 $TAG exit=$? $(date) ==="
done
touch /raid/R12K41024/20_pesq_finetune/fp32_512_pesq_ft/PHASE6_DONE
