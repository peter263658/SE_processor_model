#!/usr/bin/env bash
# ITD + IC (interaural coherence) backfill eval chain (single GPU, sequential).
# Re-evaluates the SAME checkpoints already in the comparison table, now with the
# newly-added ITD (GCC-PHAT) and IC (MSC) metrics, plus the existing
# MBSTOI/STOI/PESQ/ILD/IPD/(SISDR where present). New _itdic tags; nothing prior
# is overwritten. Respects the GPU rule: one eval at a time on one GPU.
set -uo pipefail
ROOT=/raid/R12K41024/20_pesq_finetune
PKG18=/raid/R12K41024/18_w4a8_pow2_mixed_lmul_hw_handoff_20260530
MONO=/raid/R12K41024/21_mono_se_baselines
LOGDIR=$ROOT/logs/itdic_chain
mkdir -p "$LOGDIR"
LOG=$LOGDIR/chain.log
exec >>"$LOG" 2>&1
export NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/mpl_cache

# --- pick a free GPU from the known-free set (want <2000 MiB used) ---
pick_gpu() {
  for g in 5 7 3 1; do
    used=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    [ -n "$used" ] && [ "$used" -lt 2000 ] && { echo "$g"; return 0; }
  done
  echo ""; return 1
}
GPU=$(pick_gpu)
[ -z "$GPU" ] && { echo "=== FATAL: no free GPU $(date) ==="; exit 1; }
echo "############ ITD/IC chain start $(date) — GPU $GPU ############"

REB=$ROOT/outputs/pesq_finetune/qatw12_rebal_ild4_ipd12/ft/logs/lightning_logs/version_0/checkpoints/epoch=14-step=7260-val=33.7963.ckpt
PQAT=$ROOT/outputs/pesq_finetune/qatverify_w12_lr2e-5_ep20/ft/logs/lightning_logs/version_0/checkpoints/epoch=10-step=5324-val=21.6478.ckpt
DEPLOY=$PKG18/ckpt/strict_core_pow2_blockfloat_deploy_ep3_vctk0.8357_timit0.7898.ckpt
CALIB=$PKG18/ckpt/activation_calibration_p999_b64.json

run_step() {  # $1 = human label; $2.. = command
  local label="$1"; shift
  echo; echo "=== [$label] START $(date) ==="
  "$@"; local rc=$?
  echo "=== [$label] EXIT=$rc $(date) ==="
}

# 1) SNU mono baseline (bilateral)  ------------------------------------------
run_step SNU env CKPT="$MONO/results/snu/best_snu.pt" TAG=snu_itdic \
  GPU_ID="$GPU" VCTK_ONLY=0 LIMIT=750 METRIC_WORKERS=32 \
  bash "$MONO/run_bilateral_eval.sh"

# 2) NTU mono baseline (bilateral)  ------------------------------------------
run_step NTU env CKPT="$MONO/results/ntu/best_ntu.pt" TAG=ntu_itdic \
  GPU_ID="$GPU" VCTK_ONLY=0 LIMIT=750 METRIC_WORKERS=32 \
  bash "$MONO/run_bilateral_eval.sh"

# 3) Deploy W4A8 (dir-18 APoT delayed-descale, calibrated pow2 e4m3)  --------
run_step DEPLOY env CKPT="$DEPLOY" TAG=deploy_ep3_itdic VCTK_ONLY=0 \
  ACT_FP8_MODE=calibrated_pow2_e4m3 BCCTN_ACTIVATION_CALIB_PATH="$CALIB" \
  GPU_ID="$GPU" LIMIT=750 METRIC_WORKERS=32 \
  bash "$PKG18/run_apot_descale_eval.sh"

# 4) Rebalance (PESQ variant, dir-20)  ---------------------------------------
run_step REBAL env CKPT="$REB" TAG=rebal_itdic GPU_ID="$GPU" LIMIT=750 \
  METRIC_WORKERS=32 bash "$ROOT/code/run_pesq_eval.sh"

# 5) Plain-QAT w12 (dir-20)  -------------------------------------------------
run_step PLAINQAT env CKPT="$PQAT" TAG=plainqat_itdic GPU_ID="$GPU" LIMIT=750 \
  METRIC_WORKERS=32 bash "$ROOT/code/run_pesq_eval.sh"

echo; echo "############ ITD/IC chain COMPLETE $(date) ############"
touch "$ROOT/ITDIC_DONE"
