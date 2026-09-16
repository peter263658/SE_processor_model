#!/usr/bin/env bash
# Phase 4: after the w=12 finetune (PHASE3_DONE), evaluate the best-val
# checkpoint on VCTK (all 8 SNRs, new metrics) with the FP32-512 eval code.
# This yields the "FP32-512 PESQ-ft ceiling" row for the comparison table.
set -uo pipefail
GPU="${GPU:-6}"
MARK=/raid/R12K41024/20_pesq_finetune/fp32_512_pesq_ft/PHASE3_DONE
MARK2=/raid/R12K41024/20_pesq_finetune/fp32_512_pesq_ft/QUEUE_DONE
LOG=/raid/R12K41024/20_pesq_finetune/fp32_512_pesq_ft/queue_phase4.log
exec >>"$LOG" 2>&1
export PATH=/home/R12K41024/miniforge3/bin:$PATH
export NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/mpl_cache

echo "=== phase4 waiting for PHASE3_DONE $(date) ==="
until [ -f "$MARK" ] && [ -f "$MARK2" ]; do sleep 120; done

CKDIR=$(for d in $(ls -d /raid/R12K41024/20_pesq_finetune/fp32_512_pesq_ft/outputs/pesqft_w12_lr2e-5_20ep/logs/lightning_logs/version_*/checkpoints | sort -Vr); do
  ls "$d"/epoch=*.ckpt >/dev/null 2>&1 && { echo "$d"; break; }
done)
# best-val ckpt path is recorded in the ModelCheckpoint state inside last.ckpt
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
    cands = [p for p in glob.glob(os.path.join(ckdir, "epoch=*.ckpt"))]
    best = max(cands, key=os.path.getmtime) if cands else ""
print(best)
PY
)
if [ -z "$BEST" ]; then echo "phase4: no checkpoints found in $CKDIR"; exit 1; fi
# symlink without '=' so nothing downstream chokes on the path
LINK=/raid/R12K41024/20_pesq_finetune/fp32_512_pesq_ft/best_w12.ckpt
ln -sf "$BEST" "$LINK"
echo "=== phase4 evaluating $BEST $(date) ==="

cd /raid/R12K41024/20_pesq_finetune/fp32_512_pesq_ft/code
CUDA_VISIBLE_DEVICES=$GPU conda run --no-capture-output -n bcctn python eval.py \
  --model_checkpoint "$LINK" \
  --vctk_test_dir /raid/R12K41024/BCCTN/Dataset/noisy_testset \
  --vctk_clean_dir /raid/R12K41024/BCCTN/Dataset/clean_testset \
  --output_dir /raid/R12K41024/20_pesq_finetune/results/pesq_eval/fp32_512_pesqft_w12_best \
  --paper_style_eval --batch_size 8 --limit_pairs 750
echo "=== phase4 exit=$? $(date) ==="
touch /raid/R12K41024/20_pesq_finetune/fp32_512_pesq_ft/PHASE4_DONE
