#!/usr/bin/env bash
# MetricGAN-on-deploy: train (self-healing) + subset epoch-sweep eval.
# Goal: hold MBSTOI >= 0.835 while lifting wb PESQ past reproduced-SNU (+0.562
# @ -6..9), ILD positive. Warm from deploy ep3 (the gate model).
set -uo pipefail
ROOT=/raid/R12K41024/20_pesq_finetune
DEPS=/raid/R12K41024/18_w4a8_pow2_mixed_lmul
LOGD=$ROOT/logs/itdic_chain; mkdir -p "$LOGD"
exec >>"$LOGD/mgan_chain.log" 2>&1
export NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/mpl_cache
PYENV=/home/R12K41024/miniforge3/envs/bcctn/bin/python

pick(){ for g in 5 7 3 1; do u=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null|tr -d ' '); [ -n "$u" ]&&[ "$u" -lt 2000 ]&&{ echo $g; return; }; done; }
GPU=$(pick); [ -z "$GPU" ] && { echo "=== FATAL no free GPU $(date) ==="; exit 1; }

TAG=metricgan_deploy_gw16
DEPLOY=$DEPS/outputs/d1c3_pow2_finetune/d1c3_apotclip16_attnm8_outlierm4_15ep/ft/logs/lightning_logs/version_0/checkpoints/epoch=2-step=726.ckpt
CKGLOB=$ROOT/outputs/pesq_finetune/$TAG/ft/logs/lightning_logs
N_TARGET=20

echo "############ MetricGAN-deploy chain start $(date) GPU $GPU ############"

# newest epoch ckpt across all version dirs; epoch numbering restarts per
# version after a warm resume, so REMAINING is computed from the TOTAL ckpt
# count across versions, not the newest ckpt's local epoch index.
newest_ckpt(){ ls -t "$CKGLOB"/version_*/checkpoints/epoch=*.ckpt 2>/dev/null | grep -v 'all-' | head -1; }
epoch_of(){ basename "$1" | sed -n 's/^epoch=\([0-9]*\)-.*/\1/p'; }
done_epochs(){ ls "$CKGLOB"/version_*/checkpoints/epoch=*.ckpt 2>/dev/null | grep -v 'all-' | wc -l; }

for attempt in 1 2 3 4 5; do
  NC=$(newest_ckpt)
  if [ -n "$NC" ]; then
    DN=$(done_epochs); REMAIN=$(( N_TARGET - DN ))
    [ "$REMAIN" -le 0 ] && { echo "[train] target epochs reached ($DN done)"; break; }
    echo "[train] attempt $attempt resume-warm from $NC ($DN done, $REMAIN left)"
    START="$NC"; NE="$REMAIN"
  else
    echo "[train] attempt $attempt fresh warm from deploy ep3"; START="$DEPLOY"; NE="$N_TARGET"
  fi
  START_CKPT="$START" BCCTN_D_INIT_CKPT="$NC" GAN_W=16 N_EPOCHS="$NE" BATCH_SIZE=32 GPU_ID="$GPU" \
    STOI_W=10 IPD_W=10 ILD_W=4 SNR_W=1 RUN_TAG="$TAG" \
    bash "$ROOT/code/run_metricgan_deploy.sh"
  rc=$?; echo "[train] attempt $attempt exit=$rc $(date)"
  [ $rc -eq 0 ] && break
  sleep 20
done

echo "=== [sweep] eval subset epochs (VCTK-only, limit 750) $(date) ==="
# gather all epoch ckpts across version dirs, pick a spread to map the frontier
mapfile -t ALL < <(ls "$CKGLOB"/version_*/checkpoints/epoch=*.ckpt 2>/dev/null | grep -v 'all-' | sort -t= -k2 -n -u)
echo "[sweep] found ${#ALL[@]} epoch ckpts"
# choose epochs ~4,8,12,16 and the last two
declare -A SEEN
pick_list=()
for want in 4 8 12 16 18 19; do
  for c in "${ALL[@]}"; do
    e=$(epoch_of "$c")
    if [ "$e" = "$want" ] && [ -z "${SEEN[$e]:-}" ]; then pick_list+=("$c"); SEEN[$e]=1; fi
  done
done
# always include the newest
NC=$(newest_ckpt); [ -n "$NC" ] && [ -z "${SEEN[$(epoch_of "$NC")]:-}" ] && pick_list+=("$NC")

for c in "${pick_list[@]}"; do
  e=$(epoch_of "$c"); tag="mgan_gw16_ep${e}"
  echo "=== [sweep] eval $tag ($c) $(date) ==="
  CKPT="$c" TAG="$tag" GPU_ID="$GPU" LIMIT=750 METRIC_WORKERS=32 \
    bash "$ROOT/code/run_pesq_eval.sh"
done

echo "=== [sweep] SUMMARY (VCTK -6..15 avg) $(date) ==="
for c in "${pick_list[@]}"; do
  e=$(epoch_of "$c"); f=$ROOT/results/pesq_eval/mgan_gw16_ep${e}/vctk_paper_style_results_averages.txt
  [ -f "$f" ] && printf "ep%-3s MBSTOI=%s wb=%s ILD=%s\n" "$e" \
    "$(grep 'MBSTOI Enhanced' "$f"|awk '{print $NF}')" \
    "$(grep 'PESQ Enhanced (wb)' "$f"|awk '{print $NF}')" \
    "$(grep 'ILD Error Enhanced' "$f"|awk '{print $(NF-1)}')"
done
echo "############ MetricGAN-deploy chain COMPLETE $(date) ############"
touch "$ROOT/MGAN_DONE"
