#!/usr/bin/env bash
# Saturation + QAT-verification chain (single GPU, sequential):
#   A) FP32-512 w12 continuation to saturation (warm from ep11, +20 ep, ES p6)
#   B) eval best-of-A on VCTK (parallel-metric eval)
#   C) deploy-arch QAT PESQ finetune w=12 (full HW stack, 20 ep, lr 2e-5)
#   D) eval best-of-C via dir-18 deploy eval chain (nb@8k + STOI)
# Verifies the scaling-chain estimate: deploy-achievable ~= FP32-ft x 0.867 (wb).
set -uo pipefail
GPU="${GPU:-6}"
ROOT=/raid/R12K41024/20_pesq_finetune/fp32_512_pesq_ft
PKG=/raid/R12K41024/18_w4a8_pow2_mixed_lmul_hw_handoff_20260530
LOG=$ROOT/queue_satqat.log
exec >>"$LOG" 2>&1
export PATH=/home/R12K41024/miniforge3/bin:$PATH
export NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/mpl_cache
PYENV=/home/R12K41024/miniforge3/envs/bcctn/bin/python

pick_best() {  # pick_best <ckpt-root-with-version-dirs>
  local root="$1"
  local ckdir
  ckdir=$(for d in $(ls -d "$root"/version_*/checkpoints 2>/dev/null | sort -Vr); do
    ls "$d"/epoch=*.ckpt >/dev/null 2>&1 && { echo "$d"; break; }
  done)
  [ -z "$ckdir" ] && return 1
  $PYENV - "$ckdir" <<'PY'
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
}

echo "=== [A] w12 continuation to saturation $(date) ==="
cd $ROOT/code
CUDA_VISIBLE_DEVICES=$GPU conda run --no-capture-output -n bcctn python train.py \
  +model.pesq_weight=12 \
  training.learning_rate=2e-5 \
  training.n_epochs=20 \
  training.early_stopping.patience_in_epochs=6 \
  training.train_checkpoint_path=$ROOT/outputs/pesqft_w12_lr2e-5_20ep/logs/lightning_logs/version_8/checkpoints/last.ckpt \
  hydra.run.dir=$ROOT/outputs/pesqft_w12_sat \
  > $ROOT/train_w12_sat.log 2>&1
echo "=== [A] exit=$? $(date) ==="

echo "=== [B] eval best of saturated w12 $(date) ==="
BEST_A=$(pick_best "$ROOT/outputs/pesqft_w12_sat/logs/lightning_logs")
if [ -n "$BEST_A" ]; then
  ln -sf "$BEST_A" $ROOT/w12sat_best.ckpt
  echo "[B] best=$BEST_A"
  CUDA_VISIBLE_DEVICES=$GPU conda run --no-capture-output -n bcctn python eval.py \
    --model_checkpoint $ROOT/w12sat_best.ckpt \
    --vctk_test_dir /raid/R12K41024/BCCTN/Dataset/noisy_testset \
    --vctk_clean_dir /raid/R12K41024/BCCTN/Dataset/clean_testset \
    --output_dir /raid/R12K41024/20_pesq_finetune/results/pesq_eval/fp32_512_pesqft_w12_sat \
    --paper_style_eval --batch_size 8 --limit_pairs 750
  echo "=== [B] exit=$? $(date) ==="
else
  echo "[B] SKIP: no ckpt from phase A"
fi

echo "=== [C] deploy-arch QAT PESQ finetune w=12 $(date) ==="
PESQ_WEIGHT=12 LR=2e-5 N_EPOCHS=20 GPU_ID=$GPU RUN_TAG=qatverify_w12_lr2e-5_ep20 \
  bash /raid/R12K41024/20_pesq_finetune/code/run_pesq_finetune.sh
echo "=== [C] exit=$? $(date) ==="

echo "=== [D] eval best deploy-arch ft ckpt $(date) ==="
BEST_C=$(pick_best "/raid/R12K41024/20_pesq_finetune/outputs/pesq_finetune/qatverify_w12_lr2e-5_ep20/ft/logs/lightning_logs")
if [ -n "$BEST_C" ]; then
  ln -sf "$BEST_C" $ROOT/qatverify_best.ckpt
  echo "[D] best=$BEST_C"
  CKPT=$ROOT/qatverify_best.ckpt TAG="qatverify_w12_vctk" VCTK_ONLY=1 GPU_ID=$GPU \
    ACT_FP8_MODE=calibrated_pow2_e4m3 \
    BCCTN_ACTIVATION_CALIB_PATH=$PKG/ckpt/activation_calibration_p999_b64.json \
    bash $PKG/run_apot_descale_eval.sh
  echo "=== [D] exit=$? $(date) ==="
else
  echo "[D] SKIP: no ckpt from phase C"
fi

echo "=== satqat chain COMPLETE $(date) ==="
touch $ROOT/SATQAT_DONE
