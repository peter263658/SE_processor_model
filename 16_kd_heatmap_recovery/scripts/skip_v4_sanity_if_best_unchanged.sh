#!/usr/bin/env bash
set -u

V4_PID="${V4_PID:?set V4_PID to the v4 wrapper parent PID}"
V4_LOG="${V4_LOG:?set V4_LOG}"
EXPECTED_BEST="${EXPECTED_BEST:?set EXPECTED_BEST to the already-evaluated best ckpt}"
WATCH_LOG="${WATCH_LOG:?set WATCH_LOG}"
POLL_SECONDS="${POLL_SECONDS:-20}"

log() {
  echo "[skip-sanity] $(date '+%F %T %Z') $*" >> "$WATCH_LOG"
}

log "start; watching for [best] in ${V4_LOG}; expected=${EXPECTED_BEST}"

while ps -p "$V4_PID" >/dev/null 2>&1; do
  best_line="$(grep -F '[best]' "$V4_LOG" 2>/dev/null | tail -n 1 || true)"
  if [ -n "$best_line" ]; then
    best_path="${best_line#*] }"
    log "detected ${best_line}"
    if [ "$best_path" = "$EXPECTED_BEST" ]; then
      pgid="$(ps -o pgid= -p "$V4_PID" 2>/dev/null | tr -d ' ')"
      if [ -n "$pgid" ]; then
        log "best unchanged; terminating v4 process group ${pgid} to skip redundant sanity eval"
        kill -TERM "-$pgid" 2>/dev/null || true
      else
        log "best unchanged but could not resolve process group for pid ${V4_PID}"
      fi
    else
      log "best changed; leaving v4 wrapper running for sanity eval"
    fi
    exit 0
  fi
  log "waiting; v4 still alive and no [best] line yet"
  sleep "$POLL_SECONDS"
done

log "v4 pid ${V4_PID} exited before [best] was observed"
