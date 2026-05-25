#!/usr/bin/env bash
set -u

V4_PID="${V4_PID:?set V4_PID to the v4 parent or train PID}"
GPU_ID="${GPU_ID:-6}"
GPU_MEM_THRESHOLD_MB="${GPU_MEM_THRESHOLD_MB:-20000}"
POLL_SECONDS="${POLL_SECONDS:-60}"
WATCH_LOG="${WATCH_LOG:?set WATCH_LOG}"
KD_LOG="${KD_LOG:?set KD_LOG}"
TEACHER_OUTPUT_DIR="${TEACHER_OUTPUT_DIR:?set TEACHER_OUTPUT_DIR}"
PROJECT_DIR="${PROJECT_DIR:-/raid/R12K41024/16_kd_heatmap_recovery}"
V4_CKPT_DIR="${V4_CKPT_DIR:-}"

log() {
  echo "[watcher] $(date '+%F %T %Z') $*" >> "$WATCH_LOG"
}

gpu_mem_used() {
  nvidia-smi --id="$GPU_ID" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -dc '0-9'
}

log "start; waiting for v4 pid=${V4_PID}; gpu=${GPU_ID}; threshold=${GPU_MEM_THRESHOLD_MB}MiB"

while ps -p "$V4_PID" >/dev/null 2>&1; do
  mem="$(gpu_mem_used)"
  log "v4_alive=1 gpu${GPU_ID}_mem=${mem:-unknown}MiB"
  sleep "$POLL_SECONDS"
done

log "v4 pid ${V4_PID} is gone; waiting for gpu${GPU_ID} memory release"

while true; do
  mem="$(gpu_mem_used)"
  log "v4_alive=0 gpu${GPU_ID}_mem=${mem:-unknown}MiB"
  if [ -n "$mem" ] && [ "$mem" -lt "$GPU_MEM_THRESHOLD_MB" ]; then
    break
  fi
  sleep "$POLL_SECONDS"
done

cd "$PROJECT_DIR" || {
  log "ERROR: cannot cd to ${PROJECT_DIR}"
  exit 1
}

selected_base_ckpt="${BASE_CKPT:-}"
if [ -n "$V4_CKPT_DIR" ]; then
  latest_base_ckpt="$(find "$V4_CKPT_DIR" -maxdepth 1 -type f -name '*.ckpt' ! -name 'last.ckpt' -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2-)"
  if [ -n "$latest_base_ckpt" ]; then
    selected_base_ckpt="$latest_base_ckpt"
  fi
fi

kd_env=(
  GPU_ID="$GPU_ID"
  TEACHER_OUTPUT_DIR="$TEACHER_OUTPUT_DIR"
  LMUL_LINEAR_CHUNK_SIZE="${LMUL_LINEAR_CHUNK_SIZE:-2048}"
  LMUL_CHECKPOINT_CHUNKS="${LMUL_CHECKPOINT_CHUNKS:-false}"
)
if [ -n "$selected_base_ckpt" ]; then
  kd_env+=(BASE_CKPT="$selected_base_ckpt")
  log "KD base checkpoint=${selected_base_ckpt}"
fi

log "launching KD smoke in a detached session; log=${KD_LOG}"
setsid env "${kd_env[@]}" bash scripts/run_kd_output_qat_smoke.sh > "$KD_LOG" 2>&1 < /dev/null &

kd_pid=$!
log "KD smoke pid=${kd_pid}"

# Early-failure monitor. Most launch/config/OOM failures happen before this window ends.
for _ in $(seq 1 20); do
  sleep "$POLL_SECONDS"
  if ! ps -p "$kd_pid" >/dev/null 2>&1; then
    log "KD smoke exited early with status unknown; last log follows"
    tail -80 "$KD_LOG" >> "$WATCH_LOG" 2>/dev/null
    exit 1
  fi
  if grep -Eqi 'OutOfMemoryError|CUDA out of memory|Traceback|failed' "$KD_LOG" 2>/dev/null; then
    log "KD smoke failure signature detected; stopping monitor; last log follows"
    tail -80 "$KD_LOG" >> "$WATCH_LOG" 2>/dev/null
    exit 1
  fi
  log "KD smoke alive pid=${kd_pid}"
done

log "KD smoke survived first $((20 * POLL_SECONDS / 60)) min; monitor complete"
