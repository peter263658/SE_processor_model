#!/usr/bin/env bash
# Gate-holding PESQ fine-sweep — TRAINER (Codex-review priority 1).
# Four sequential short finetunes from gate-adjacent starts, small LR, every
# epoch saved (all- series). A separate screener process evals each epoch on
# another GPU. Self-healing: resumes each run from its last.ckpt.
#
#   R1 w6cont_s12 : continue w6 ep4 (VCTK MBSTOI 0.8337 rising, TIMIT gate ok)
#                   PESQ_W=6 STOI_W=12 LR=5e-6 x6ep  -- prime gate-recovery
#   R2 w4_lr5e-6  : deploy ep3, PESQ_W=4 STOI_W=10 LR=5e-6 x6ep
#   R3 w2_lr1e-5  : deploy ep3, PESQ_W=2 STOI_W=10 LR=1e-5 x5ep
#   R4 w4_sisdr   : deploy ep3, PESQ_W=4 + snr_loss_mode=sisdr_tied LR=5e-6 x6ep
set -uo pipefail
ROOT=/raid/R12K41024/20_pesq_finetune
DEPS=/raid/R12K41024/18_w4a8_pow2_mixed_lmul
LOGD=$ROOT/logs/gate_sweep; mkdir -p "$LOGD"
exec >>"$LOGD/train_chain.log" 2>&1
export NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/mpl_cache

pick_free() { for g in $1; do u=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null|tr -d ' '); [ -n "$u" ] && [ "$u" -lt 2000 ] && { echo "$g"; return; }; done; }
TRAIN_GPU="${TRAIN_GPU:-$(pick_free "5 3 1 7")}"
[ -z "$TRAIN_GPU" ] && { echo "FATAL: no free train GPU"; exit 1; }
echo "$TRAIN_GPU" > "$LOGD/train_gpu"
DEPLOY=$DEPS/outputs/d1c3_pow2_finetune/d1c3_apotclip16_attnm8_outlierm4_15ep/ft/logs/lightning_logs/version_0/checkpoints/epoch=2-step=726.ckpt
W6EP4=$ROOT/outputs/pesq_finetune/pesq_w6_lr1e-5_ep5/ft/logs/lightning_logs/version_0/checkpoints/all-epoch=4-step=2420-val=5.1325.ckpt

echo "############ gate-sweep TRAINER start $(date) GPU $TRAIN_GPU ############"
for f in "$DEPLOY" "$W6EP4"; do [ -f "$f" ] || { echo "FATAL missing $f"; exit 1; }; done

# run_one <tag> <start_ckpt> <n_epochs> <extra env assignments...>
run_one() {
  local tag="$1" start="$2" nep="$3"; shift 3
  local ckglob=$ROOT/outputs/pesq_finetune/$tag/ft/logs/lightning_logs
  for attempt in 1 2 3 4; do
    local done_n resume
    done_n=$(ls "$ckglob"/version_*/checkpoints/all-epoch=*.ckpt 2>/dev/null | wc -l)
    if [ "$done_n" -ge "$nep" ]; then echo "[$tag] already has $done_n epochs (target $nep) — skip"; return 0; fi
    resume=$(ls -t "$ckglob"/version_*/checkpoints/last.ckpt 2>/dev/null | head -1)
    local st="$start" ne="$nep"
    if [ -n "$resume" ]; then st="$resume"; ne=$(( nep - done_n )); echo "[$tag] attempt $attempt resume from last.ckpt ($done_n done, $ne left)"; else echo "[$tag] attempt $attempt fresh from $start"; fi
    env START_CKPT="$st" N_EPOCHS="$ne" GPU_ID="$TRAIN_GPU" RUN_TAG="$tag" "$@" \
      bash "$ROOT/code/run_pesq_finetune.sh"
    local rc=$?
    # Success is judged by BANKED EPOCH COUNT, never by exit code: Lightning
    # exits 0 on a graceful SIGTERM without completing the requested epochs
    # (observed 2026-07-10: R0 "exit=0" with 3/4 epochs banked).
    done_n=$(ls "$ckglob"/version_*/checkpoints/all-epoch=*.ckpt 2>/dev/null | wc -l)
    echo "[$tag] attempt $attempt exit=$rc banked=$done_n/$nep $(date)"
    [ "$done_n" -ge "$nep" ] && return 0
    sleep 20
  done
  done_n=$(ls "$ckglob"/version_*/checkpoints/all-epoch=*.ckpt 2>/dev/null | wc -l)
  [ "$done_n" -ge "$nep" ] && return 0
  echo "[$tag] FAILED: banked $done_n/$nep after retries"; return 1
}

FAILED=""

echo "=== R0 w6cont_exact (true continuation: original weights, original lr) $(date) ==="
run_one sw_w6cont_exact "$W6EP4" 4 PESQ_WEIGHT=6 STOI_W=10 IPD_W=10 ILD_W=1 SNR_W=1 LR=1e-5 || FAILED="$FAILED sw_w6cont_exact"

echo "=== R1 w6cont_s12 (recovery ablation: STOI 12, lr 5e-6) $(date) ==="
run_one sw_w6cont_s12 "$W6EP4" 6 PESQ_WEIGHT=6 STOI_W=12 IPD_W=10 ILD_W=1 SNR_W=1 LR=5e-6 || FAILED="$FAILED sw_w6cont_s12"

echo "=== R2 w4_lr5e-6 $(date) ==="
run_one sw_w4_lr5e6 "$DEPLOY" 6 PESQ_WEIGHT=4 STOI_W=10 IPD_W=10 ILD_W=1 SNR_W=1 LR=5e-6 || FAILED="$FAILED sw_w4_lr5e6"

echo "=== R3 w2_lr1e-5 $(date) ==="
run_one sw_w2_lr1e5 "$DEPLOY" 5 PESQ_WEIGHT=2 STOI_W=10 IPD_W=10 ILD_W=1 SNR_W=1 LR=1e-5 || FAILED="$FAILED sw_w2_lr1e5"

echo "=== R4 w4_sisdr_tied $(date) ==="
run_one sw_w4_sisdr "$DEPLOY" 6 PESQ_WEIGHT=4 STOI_W=10 IPD_W=10 ILD_W=1 SNR_W=1 SNR_MODE=sisdr_tied LR=5e-6 || FAILED="$FAILED sw_w4_sisdr"

if [ -n "$FAILED" ]; then
  echo "############ TRAINER FAILURES:$FAILED $(date) ############"
  touch "$ROOT/SWEEPTRAIN_FAILED"
fi
echo "############ gate-sweep TRAINER COMPLETE $(date) ############"
# DONE = trainer finished trying (screener's terminator); FAILED marks shortfalls.
touch "$ROOT/SWEEPTRAIN_DONE"
