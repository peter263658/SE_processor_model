#!/usr/bin/env bash
# Nanny v2 for the gate-sweep chains. Handles the three observed failure
# modes of the box's external kill sweeps (16:59 and 17:10 on 2026-07-10):
#   1. chain bash killed              -> relaunch (screener has flock)
#   2. eval python killed, orphan metric workers (ppid 1) keep the tee pipe
#      open                           -> reap orphans every cycle
#   3. eval killed but child bash/tee stuck waiting on the pipe (screener
#      frozen inside its timeout)     -> if screener log is stale >8 min with
#      no eval python alive, kill the stuck child subtree so the in-script
#      retry fires immediately instead of waiting out the full timeout
# NEVER touches train.py or the trainer's python.
ROOT=/raid/R12K41024/20_pesq_finetune
LOGD=$ROOT/logs/gate_sweep; mkdir -p "$LOGD"
exec >>"$LOGD/nanny.log" 2>&1
echo "nanny v2 start $(date)"
while :; do
  if [ -f "$ROOT/SWEEP_DONE" ]; then echo "nanny: SWEEP_DONE, exiting $(date)"; exit 0; fi

  # --- 2: reap orphan eval metric workers (ppid 1, never train.py) ---------
  ps -eo pid,ppid,cmd | awk '$2==1 && /eval\.py|spawn_main|resource_tracker/ && !/train\.py/ && !/awk/ {print $1}' | while read -r p; do
    kill -9 "$p" 2>/dev/null && echo "nanny: reaped orphan $p $(date)"
  done

  # --- 1: relaunch dead chains ---------------------------------------------
  if [ ! -f "$ROOT/SWEEPTRAIN_DONE" ] && ! ps -eo cmd | grep -q "[b]ash run_gate_sweep_train.sh"; then
    echo "nanny: trainer dead, relaunching $(date)"
    ( cd "$ROOT" && setsid nohup bash run_gate_sweep_train.sh </dev/null >/dev/null 2>&1 & )
  fi
  if ! ps -eo cmd | grep -q "[b]ash run_gate_sweep_screen.sh"; then
    echo "nanny: screener dead, relaunching $(date)"
    ( cd "$ROOT" && setsid nohup bash run_gate_sweep_screen.sh </dev/null >/dev/null 2>&1 & )
  fi

  # --- 3: unstick a frozen screener (stale log + no eval python) -----------
  SPID=$(ps -eo pid,cmd | grep "[b]ash run_gate_sweep_screen.sh" | awk '{print $1}' | head -1)
  if [ -n "$SPID" ] && [ -f "$LOGD/screen_chain.log" ]; then
    age=$(( $(date +%s) - $(stat -c %Y "$LOGD/screen_chain.log") ))
    has_eval=$(ps -eo cmd | grep -c "[e]val.py --model_checkpoint" || true)
    if [ "$age" -gt 480 ] && [ "$has_eval" -eq 0 ]; then
      # screener silent >8 min with no eval running: it is either in its
      # normal 60 s poll loop (harmless to leave) or stuck on a dead child.
      # Killing child subtrees is safe in both cases: the poll loop has no
      # children; a stuck canon/screen call returns and retries.
      for c in $(pgrep -P "$SPID" 2>/dev/null); do
        cmd=$(ps -o cmd= -p "$c" 2>/dev/null)
        case "$cmd" in
          *sleep*) : ;;  # poll loop sleep — leave it
          *) echo "nanny: unsticking screener child $c ($cmd) $(date)"
             pkill -9 -P "$c" 2>/dev/null; kill -9 "$c" 2>/dev/null ;;
        esac
      done
    fi
  fi
  sleep 120
done
