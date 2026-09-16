#!/usr/bin/env bash
# Gate-rescue (§4(a) of HANDOFF_GATE_SWEEP_CONTINUATION_2026-07-10): the sweep
# ended NO_GATE_PASS; best candidate sw_w4_sisdr_v2e5 FAILed VCTK by 0.0006
# (canonical 0.8344/0.7910, TIMIT passing with margin). Continue it 3 epochs
# with STOI_W=12 (rest of the R4 recipe unchanged: PESQ 4, sisdr_tied, lr 5e-6),
# then canonical-eval EVERY new epoch directly — no scaled screen; the sweep
# proved scaled reads ~0.002-0.003 HIGH vs canonical on finetuned lineages.
#
# Single compute job at a time throughout (train, then evals, all serial).
# Launch: user tmux foreground —  bash run_gate_rescue.sh
set -uo pipefail
ROOT=/raid/R12K41024/20_pesq_finetune
PKG18=/raid/R12K41024/18_w4a8_pow2_mixed_lmul_hw_handoff_20260530
LOGD=$ROOT/logs/gate_sweep; mkdir -p "$LOGD"
exec 9>"$LOGD/rescue.lock"
flock -n 9 || { echo "another rescue holds the lock; exiting"; exit 0; }
exec >>"$LOGD/rescue_chain.log" 2>&1
export NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/mpl_cache
rm -f "$LOGD"/RESCUE_GATE_PASS "$LOGD"/RESCUE_NO_PASS "$ROOT"/RESCUE_DONE 2>/dev/null

SEED=$ROOT/outputs/pesq_finetune/sw_w4_sisdr/ft/logs/lightning_logs/version_2/checkpoints/all-epoch=5-step=2904-val=-0.8682.ckpt
CALIB=$PKG18/ckpt/activation_calibration_p999_b64.json
TAG=sw_sisdr_s12_rescue
NEP=3
[ -f "$SEED" ] || { echo "FATAL missing seed $SEED"; exit 1; }

pick_free() { for g in 5 3 1 7; do u=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null|tr -d ' '); [ -n "$u" ] && [ "$u" -lt 2000 ] && { echo "$g"; return; }; done; }
GPU="${GPU:-$(pick_free)}"
[ -z "$GPU" ] && { echo "FATAL: no free GPU"; exit 1; }
echo "############ gate-rescue start $(date) GPU $GPU ############"

# ---- 1) train: seed + 3 epochs, STOI_W=12 (success = banked count, not exit code)
CKGLOB=$ROOT/outputs/pesq_finetune/$TAG/ft/logs/lightning_logs
for attempt in 1 2 3 4; do
  done_n=$(ls "$CKGLOB"/version_*/checkpoints/all-epoch=*.ckpt 2>/dev/null | wc -l)
  [ "$done_n" -ge "$NEP" ] && break
  resume=$(ls -t "$CKGLOB"/version_*/checkpoints/last.ckpt 2>/dev/null | head -1)
  st="$SEED"; ne="$NEP"
  if [ -n "$resume" ]; then st="$resume"; ne=$(( NEP - done_n )); echo "[$TAG] attempt $attempt resume ($done_n done, $ne left)"; else echo "[$TAG] attempt $attempt fresh from seed"; fi
  env START_CKPT="$st" N_EPOCHS="$ne" GPU_ID="$GPU" RUN_TAG="$TAG" \
    PESQ_WEIGHT=4 STOI_W=12 IPD_W=10 ILD_W=1 SNR_W=1 SNR_MODE=sisdr_tied LR=5e-6 \
    bash "$ROOT/code/run_pesq_finetune.sh" 9>&-
  rc=$?
  done_n=$(ls "$CKGLOB"/version_*/checkpoints/all-epoch=*.ckpt 2>/dev/null | wc -l)
  echo "[$TAG] attempt $attempt exit=$rc banked=$done_n/$NEP $(date)"
  [ "$done_n" -ge "$NEP" ] || sleep 20
done
done_n=$(ls "$CKGLOB"/version_*/checkpoints/all-epoch=*.ckpt 2>/dev/null | wc -l)
[ "$done_n" -lt 1 ] && { echo "FATAL: nothing banked"; touch "$ROOT/RESCUE_DONE"; exit 1; }
[ "$done_n" -lt "$NEP" ] && echo "WARN: only $done_n/$NEP banked — evaluating what exists"

# ---- 2) canonical-eval every banked epoch (serial, 2 attempts each)
CSUM=$LOGD/rescue_summary.tsv
printf "stag\tvMBSTOI\ttMBSTOI\tv_wb\tt_wb\tv_ILD\tverdict\tckpt\n" > "$CSUM"
BEST=""
for ck in $(ls "$CKGLOB"/version_*/checkpoints/all-epoch=*.ckpt 2>/dev/null | sort); do
  ver=$(echo "$ck" | grep -oE 'version_[0-9]+' | sed 's/version_//')
  ep=$(echo "$ck" | grep -oE 'epoch=[0-9]+' | head -1 | cut -d= -f2)
  stag="${TAG}_v${ver}e${ep}"
  out=$PKG18/results/apot_descale_eval/ft_canon_$stag/timit_paper_style_results_averages.txt
  for attempt in 1 2; do
    [ -f "$out" ] && break
    timeout -k 30 5400 env CKPT="$ck" TAG="canon_$stag" VCTK_ONLY=0 LIMIT=750 METRIC_WORKERS=32 \
      ACT_FP8_MODE=calibrated_pow2_e4m3 BCCTN_ACTIVATION_CALIB_PATH="$CALIB" \
      GPU_ID="$GPU" bash "$PKG18/run_apot_descale_eval.sh" 9>&- || true
    [ -f "$out" ] || { echo "!! canonical $stag attempt $attempt failed"; sleep 20; }
  done
  d=$PKG18/results/apot_descale_eval/ft_canon_$stag
  vf=$d/vctk_paper_style_results_averages.txt; tf=$d/timit_paper_style_results_averages.txt
  [ -f "$vf" ] && [ -f "$tf" ] || { echo "!! no results for $stag — skipping row"; continue; }
  vmb=$(grep 'MBSTOI Enhanced' "$vf"|awk '{print $NF}'); : "${vmb:=nan}"
  tmb=$(grep 'MBSTOI Enhanced' "$tf"|awk '{print $NF}'); : "${tmb:=nan}"
  vwb=$(grep 'PESQ Enhanced (wb)' "$vf"|awk '{print $NF}'); : "${vwb:=nan}"
  twb=$(grep 'PESQ Enhanced (wb)' "$tf"|awk '{print $NF}'); : "${twb:=nan}"
  vild=$(grep 'ILD Error Enhanced' "$vf"|awk '{print $(NF-1)}'); : "${vild:=nan}"
  verdict=$(awk -v a="$vmb" -v b="$tmb" 'BEGIN{print (a>=0.835 && b>=0.788) ? "PASS" : "FAIL"}')
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$stag" "$vmb" "$tmb" "$vwb" "$twb" "$vild" "$verdict" "$ck" >> "$CSUM"
  echo "[$stag] canonical v=$vmb t=$tmb wb=$vwb/$twb -> $verdict"
  # among passers keep the highest v_wb as BEST
  if [ "$verdict" = "PASS" ]; then
    if [ -z "$BEST" ] || awk -v new="$vwb" -v old="$(echo "$BEST"|cut -f4)" 'BEGIN{exit !(new>old)}'; then
      BEST=$(printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s" "$stag" "$vmb" "$tmb" "$vwb" "$twb" "$vild" "PASS" "$ck")
    fi
  fi
done

if [ -n "$BEST" ]; then echo "$BEST" > "$LOGD/RESCUE_GATE_PASS"; echo "*** RESCUE GATE PASS: $BEST"
else touch "$LOGD/RESCUE_NO_PASS"; echo "*** rescue: no gate passer"; fi
touch "$ROOT/RESCUE_DONE"
echo "############ gate-rescue COMPLETE $(date) ############"
