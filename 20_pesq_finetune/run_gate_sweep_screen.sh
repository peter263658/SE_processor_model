#!/usr/bin/env bash
# Gate-holding PESQ fine-sweep — SCREENER + canonical gate judge.
#
# 1) FIRST canonical-evals the existing w6 ep4 anchor (may already pass both
#    gates: scaled 0.8337/0.7888 + measured screen offset ≈ canonical
#    0.8353/0.7907 — Codex F1).
# 2) Polls the five sweep tags for new all-epoch ckpts; screens each with the
#    full VCTK+TIMIT 750-pair scaled eval (dir-20; screens ONLY — the screen
#    reads ~0.0016/0.0019 LOW vs canonical, measured on deploy ep3).
# 3) When the trainer is done and every epoch is screened (or gave up after 3
#    attempts), selects a DIVERSE candidate set: global top-3 by v_wb among
#    threshold passers + best passer per run + highest-vMBSTOI fallback.
# 4) Runs candidates through the CANONICAL dir-18 eval (calibrated_pow2_e4m3),
#    PARSES the gates (VCTK >= 0.835 && TIMIT >= 0.788), writes
#    canonical_summary.tsv and BEST_GATE_PASS / NO_GATE_PASS (Codex F3).
# Sentinels: SWEEP_DONE = screener finished; SWEEP_FAILED = something needed
# retrying past its cap; BEST_GATE_PASS/NO_GATE_PASS = the actual verdict.
set -uo pipefail
ROOT=/raid/R12K41024/20_pesq_finetune
PKG18=/raid/R12K41024/18_w4a8_pow2_mixed_lmul_hw_handoff_20260530
LOGD=$ROOT/logs/gate_sweep; mkdir -p "$LOGD"
# single-instance guard (nanny may relaunch; avoid double screeners)
exec 9>"$LOGD/screen.lock"
flock -n 9 || { echo "another screener holds the lock; exiting"; exit 0; }
exec >>"$LOGD/screen_chain.log" 2>&1
export NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/mpl_cache
# re-entry: clear stale verdicts/failcounts so a rerun can't leave
# contradictory sentinels (BEST_GATE_PASS + SWEEP_FAILED from different runs)
rm -f "$LOGD"/BEST_GATE_PASS "$LOGD"/NO_GATE_PASS "$LOGD"/failcount_* \
      "$ROOT"/SWEEP_FAILED "$ROOT"/SWEEP_DONE 2>/dev/null

# Wait for the trainer to record its GPU (launcher starts trainer first).
for i in $(seq 1 60); do [ -f "$LOGD/train_gpu" ] && break; sleep 10; done
TGPU=$(cat "$LOGD/train_gpu" 2>/dev/null || echo "5")
pick_free() { for g in $1; do [ "$g" = "$TGPU" ] && continue; u=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null|tr -d ' '); [ -n "$u" ] && [ "$u" -lt 2000 ] && { echo "$g"; return; }; done; }
SCREEN_GPU="${SCREEN_GPU:-$(pick_free "3 1 7 5")}"
[ -z "$SCREEN_GPU" ] && { echo "FATAL: no free screen GPU"; exit 1; }

TAGS=(sw_w6cont_exact sw_w6cont_s12 sw_w4_lr5e6 sw_w2_lr1e5 sw_w4_sisdr)
SUMMARY=$LOGD/sweep_summary.txt
CSUM=$LOGD/canonical_summary.tsv
CALIB=$PKG18/ckpt/activation_calibration_p999_b64.json
ANCHOR_CK=$ROOT/outputs/pesq_finetune/pesq_w6_lr1e-5_ep5/ft/logs/lightning_logs/version_0/checkpoints/all-epoch=4-step=2420-val=5.1325.ckpt
FAILURES=""

echo "############ gate-sweep SCREENER start $(date) GPU $SCREEN_GPU (train GPU $TGPU) ############"

# ---- canonical eval helper (2 attempts) ------------------------------------
canon_eval() { # $1=stag $2=ckpt
  local out=$PKG18/results/apot_descale_eval/ft_canon_$1/timit_paper_style_results_averages.txt
  [ -f "$out" ] && return 0
  for attempt in 1 2 3; do
    # timeout guards against the orphaned-tee pipeline hang: if the eval's
    # python is killed externally, surviving metric workers hold the tee pipe
    # open and the child bash never returns (observed 2026-07-10 16:59).
    timeout -k 30 5400 env CKPT="$2" TAG="canon_$1" VCTK_ONLY=0 LIMIT=750 METRIC_WORKERS=32 \
      ACT_FP8_MODE=calibrated_pow2_e4m3 BCCTN_ACTIVATION_CALIB_PATH="$CALIB" \
      GPU_ID="$SCREEN_GPU" bash "$PKG18/run_apot_descale_eval.sh" 9>&- || true
    [ -f "$out" ] && return 0
    echo "!! canonical $1 attempt $attempt failed"; sleep 20
  done
  FAILURES="$FAILURES canon_$1"
  return 1
}

# ---- F1: anchor candidate first — w6 ep4 may already pass both gates -------
echo "=== [anchor] canonical eval of existing w6 ep4 $(date) ==="
canon_eval w6ep4_anchor "$ANCHOR_CK" || true

# ---- screening --------------------------------------------------------------
screen_tag() { # $1=run_tag $2=ckpt_path
  local v e
  v=$(echo "$2" | sed -n 's/.*version_\([0-9]*\).*/\1/p')
  e=$(basename "$2" | sed -n 's/^all-epoch=\([0-9]*\)-.*/\1/p')
  echo "${1}_v${v}e${e}"
}

screen_one() { # $1=run_tag $2=ckpt_path
  local stag; stag=$(screen_tag "$1" "$2")
  local avg=$ROOT/results/pesq_eval/$stag/timit_paper_style_results_averages.txt
  [ -f "$avg" ] && return 0
  local fc=$LOGD/failcount_$stag
  local n; n=$(cat "$fc" 2>/dev/null || echo 0)
  [ "$n" -ge 3 ] && return 0
  local age=$(( $(date +%s) - $(stat -c %Y "$2") ))
  [ "$age" -lt 90 ] && return 1
  echo "--- screening $stag ($2) $(date) ---"
  timeout -k 30 3600 env CKPT="$2" TAG="$stag" GPU_ID="$SCREEN_GPU" LIMIT=750 METRIC_WORKERS=32 \
    bash "$ROOT/code/run_pesq_eval.sh" 9>&- || true
  if [ -f "$avg" ]; then rm -f "$fc"; else echo $((n+1)) > "$fc"; echo "--- screen $stag failed (attempt $((n+1))/3) ---"; fi
  return 0
}
gave_up() { local n; n=$(cat "$LOGD/failcount_$1" 2>/dev/null || echo 0); [ "$n" -ge 3 ]; }

all_ckpts() {
  for t in "${TAGS[@]}"; do
    ls "$ROOT"/outputs/pesq_finetune/"$t"/ft/logs/lightning_logs/version_*/checkpoints/all-epoch=*.ckpt 2>/dev/null | while read -r c; do echo "$t|$c"; done
  done
}

while :; do
  pending=0
  while IFS='|' read -r t c; do
    [ -z "$t" ] && continue
    stag=$(screen_tag "$t" "$c")
    if [ ! -f "$ROOT/results/pesq_eval/$stag/timit_paper_style_results_averages.txt" ] && ! gave_up "$stag"; then
      pending=1
      screen_one "$t" "$c" || true
    fi
  done < <(all_ckpts)
  if [ -f "$ROOT/SWEEPTRAIN_DONE" ] && [ "$pending" -eq 0 ]; then break; fi
  sleep 60
done
# only exhausted (>=3) failcounts are real failures; transient ones were
# cleared on success inside screen_one
for f in "$LOGD"/failcount_*; do
  [ -e "$f" ] || continue
  [ "$(cat "$f" 2>/dev/null || echo 0)" -ge 3 ] && FAILURES="$FAILURES screen_$(basename "$f")"
done

# ---- summary -----------------------------------------------------------------
echo "=== all epochs screened; building summary $(date) ==="
: > "$SUMMARY"
printf "%-26s %-8s %-8s %-8s %-8s %-8s %-8s %s\n" "tag" "vMBSTOI" "v_wb" "v_ILD" "tMBSTOI" "t_wb" "d69wb" "ild69" >> "$SUMMARY"
declare -A CKOF
while IFS='|' read -r t c; do
  [ -z "$t" ] && continue
  stag=$(screen_tag "$t" "$c"); CKOF[$stag]="$c"
  vf=$ROOT/results/pesq_eval/$stag/vctk_paper_style_results_averages.txt
  tf=$ROOT/results/pesq_eval/$stag/timit_paper_style_results_averages.txt
  cf=$ROOT/results/pesq_eval/$stag/vctk_paper_style_results.csv
  [ -f "$vf" ] && [ -f "$tf" ] || continue
  vmb=$(grep 'MBSTOI Enhanced' "$vf"|awk '{print $NF}'); : "${vmb:=nan}"
  vwb=$(grep 'PESQ Enhanced (wb)' "$vf"|awk '{print $NF}'); : "${vwb:=nan}"
  vild=$(grep 'ILD Error Enhanced' "$vf"|awk '{print $(NF-1)}'); : "${vild:=nan}"
  tmb=$(grep 'MBSTOI Enhanced' "$tf"|awk '{print $NF}'); : "${tmb:=nan}"
  twb=$(grep 'PESQ Enhanced (wb)' "$tf"|awk '{print $NF}'); : "${twb:=nan}"
  read -r d69 ild69 < <(awk -F',' 'NR==1{for(i=1;i<=NF;i++){if($i=="PESQ")pc=i;if($i=="PESQ_noisy")pn=i;if($i=="L_ILD")il=i;if($i=="SNR")sc=i}}
    NR>1{s=$sc+0; if(s>=-6&&s<=9){se+=$pc;sn+=$pn;n++;ii+=$il}}
    END{if(n)printf "%+.3f %.3f", (se-sn)/n, ii/n; else printf "nan nan"}' "$cf")
  printf "%-26s %-8s %-8s %-8s %-8s %-8s %-8s %s\n" "$stag" "$vmb" "$vwb" "$vild" "$tmb" "$twb" "$d69" "$ild69" >> "$SUMMARY"
done < <(all_ckpts)
cat "$SUMMARY"

# ---- candidate selection (Codex F4: diverse set) ----------------------------
# Screen thresholds sit BELOW the canonical gates by the measured offset:
#   canonical 0.835 -> expected scaled 0.8334; admit >= 0.8325
#   canonical 0.788 -> expected scaled 0.7861; admit >= 0.7855
echo "=== candidate selection $(date) ==="
CAND=()
add_cand() { local s="$1"; [ -z "$s" ] && return; printf '%s\n' "${CAND[@]:-}" | grep -qx "$s" || CAND+=("$s"); }
# global top-3 by v_wb among threshold passers
while read -r s; do add_cand "$s"; done < <(tail -n +2 "$SUMMARY" | awk '$2>=0.8325 && $5>=0.7855' | sort -k3 -rn | head -3 | awk '{print $1}')
# best passer per run (calibration reorder safety)
for t in "${TAGS[@]}"; do
  add_cand "$(tail -n +2 "$SUMMARY" | awk -v p="${t}_v" 'index($1,p)==1 && $2>=0.8325 && $5>=0.7855' | sort -k3 -rn | head -1 | awk '{print $1}')"
done
# global top-3 by vMBSTOI with meaningful PESQ (2026-07-10 anchor canonical showed the
# scaled->canonical offset FLIPS SIGN on w6-lineage weights: canonical read LOW by
# ~0.0024/0.0013 vs scaled — so MBSTOI-ranked points are the likely gate passers)
while read -r s; do add_cand "$s"; done < <(tail -n +2 "$SUMMARY" | awk '$3>1.65' | sort -k2 -rn | head -3 | awk '{print $1}')
echo "candidates: ${CAND[*]:-NONE}"

for stag in "${CAND[@]:-}"; do
  [ -z "$stag" ] && continue
  ck="${CKOF[$stag]:-}"
  [ -z "$ck" ] && { echo "!! no ckpt for $stag"; FAILURES="$FAILURES map_$stag"; continue; }
  echo "=== CANONICAL gate eval: $stag $(date) ==="
  canon_eval "$stag" "$ck" || true
done

# ---- F3: parse gates, write canonical_summary.tsv + verdict ----------------
echo "=== canonical gate judgment $(date) ==="
printf "stag\tvMBSTOI\ttMBSTOI\tv_wb\tt_wb\tv_ILD\tverdict\tckpt\n" > "$CSUM"
CKOF[w6ep4_anchor]="$ANCHOR_CK"
for stag in w6ep4_anchor "${CAND[@]:-}"; do
  [ -z "$stag" ] && continue
  d=$PKG18/results/apot_descale_eval/ft_canon_$stag
  vf=$d/vctk_paper_style_results_averages.txt; tf=$d/timit_paper_style_results_averages.txt
  [ -f "$vf" ] && [ -f "$tf" ] || { printf "%s\tnan\tnan\tnan\tnan\tnan\tEVAL_FAILED\t%s\n" "$stag" "${CKOF[$stag]:-?}" >> "$CSUM"; continue; }
  vmb=$(grep 'MBSTOI Enhanced' "$vf"|awk '{print $NF}'); : "${vmb:=nan}"
  tmb=$(grep 'MBSTOI Enhanced' "$tf"|awk '{print $NF}'); : "${tmb:=nan}"
  vwb=$(grep 'PESQ Enhanced (wb)' "$vf"|awk '{print $NF}'); : "${vwb:=nan}"
  twb=$(grep 'PESQ Enhanced (wb)' "$tf"|awk '{print $NF}'); : "${twb:=nan}"
  vild=$(grep 'ILD Error Enhanced' "$vf"|awk '{print $(NF-1)}'); : "${vild:=nan}"
  verdict=$(awk -v a="$vmb" -v b="$tmb" 'BEGIN{print (a>=0.835 && b>=0.788) ? "PASS" : "FAIL"}')
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$stag" "$vmb" "$tmb" "$vwb" "$twb" "$vild" "$verdict" "${CKOF[$stag]:-?}" >> "$CSUM"
done
column -t "$CSUM"

best=$(tail -n +2 "$CSUM" | awk -F'\t' '$7=="PASS"' | sort -t$'\t' -k4 -rn | head -1)
if [ -n "$best" ]; then
  echo "$best" > "$LOGD/BEST_GATE_PASS"
  echo "*** BEST_GATE_PASS: $best"
else
  touch "$LOGD/NO_GATE_PASS"
  echo "*** NO canonical gate passer among candidates"
fi
[ -n "$FAILURES" ] && { echo "FAILURES:$FAILURES"; touch "$ROOT/SWEEP_FAILED"; }
echo "############ gate-sweep SCREENER COMPLETE $(date) ############"
touch "$ROOT/SWEEP_DONE"
