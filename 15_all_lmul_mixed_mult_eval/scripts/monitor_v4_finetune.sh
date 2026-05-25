#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/raid/R12K41024/15_all_lmul_mixed_mult_eval}"
RUN_DIR="${RUN_DIR:-${REPO}/outputs/all_lmul_mixed_qat/all_lmul_m4_bitexact_qat_2026-05-24_02-49-17}"
TRAIN_LOG="${TRAIN_LOG:-${REPO}/logs/all_lmul_mixed_qat_finetune_ep3_lr1e-5_b128chk2048_gpu6_setsid_retry.txt}"
MONITOR_LOG="${MONITOR_LOG:-${REPO}/logs/monitor_v4_finetune_$(date +%Y%m%d_%H%M%S).log}"
LATEST_SUMMARY="${LATEST_SUMMARY:-${REPO}/logs/monitor_v4_finetune_latest_summary.txt}"
POLL_SECONDS="${POLL_SECONDS:-300}"
GPU_ID="${GPU_ID:-6}"
WATCH_PIDS="${WATCH_PIDS:-}"

mkdir -p "$(dirname "$MONITOR_LOG")"
is_alive() {
  if [ -n "$WATCH_PIDS" ]; then
    for pid in $WATCH_PIDS; do
      if kill -0 "$pid" >/dev/null 2>&1; then
        return 0
      fi
    done
  fi
  pgrep -f "hydra.run.dir=${RUN_DIR}" >/dev/null 2>&1
}

write_status() {
  {
    echo "===== $(date '+%F %T') ====="
    echo "run_dir: ${RUN_DIR}"
    echo "train_log: ${TRAIN_LOG}"
    echo "watch_pids: ${WATCH_PIDS:-none}"
    echo "gpu:"
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader -i "${GPU_ID}" 2>/dev/null || true
    echo "process:"
    if [ -n "$WATCH_PIDS" ]; then
      for pid in $WATCH_PIDS; do
        ps -fp "$pid" 2>/dev/null || true
      done
    fi
    pgrep -af "hydra.run.dir=${RUN_DIR}" || true
    python3 - "$TRAIN_LOG" <<'PY'
import re, sys
p = sys.argv[1]
try:
    text = open(p, errors="ignore").read().replace("\r", "\n")
except FileNotFoundError:
    print("log not found yet")
    raise SystemExit

last = {}
for m in re.finditer(r"Epoch (\d+): 100%[^\n]*validation_loss=([-0-9.]+), validation_std=([-0-9.]+)", text):
    last[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))

if last:
    print("completed validation:")
    for e in sorted(last):
        v, s = last[e]
        print(f"  Ep{e}: val={v:.3f} std={s:.3f}")
    best_e, (best_v, best_s) = min(last.items(), key=lambda kv: kv[1][0])
    print(f"best completed: Ep{best_e} val={best_v:.3f} std={best_s:.3f}")
else:
    print("completed validation: none yet")

progress = None
for m in re.finditer(r"Epoch (\d+):\s+(\d+)%[^\n]*?(\d+)/(\d+) \[([^\]]+)\].*?validation_loss=([-0-9.]+)", text):
    progress = m.groups()
if progress:
    e, pct, done, total, timing, shown = progress
    print(f"latest progress: Ep{e} {pct}% {done}/{total} [{timing}] shown_val={shown}")

if "[all-lmul-qat] done" in text:
    print("wrapper_status: done")
elif "Traceback" in text or "failed." in text:
    print("wrapper_status: possible error seen in log")
else:
    print("wrapper_status: running or pending")
PY
    echo
  } >> "$MONITOR_LOG"
}

write_final_summary() {
  python3 - "$TRAIN_LOG" "$RUN_DIR" "$LATEST_SUMMARY" <<'PY'
import glob, os, re, sys
log_path, run_dir, out_path = sys.argv[1:4]
text = open(log_path, errors="ignore").read().replace("\r", "\n") if os.path.exists(log_path) else ""

last = {}
for m in re.finditer(r"Epoch (\d+): 100%[^\n]*validation_loss=([-0-9.]+), validation_std=([-0-9.]+)", text):
    last[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))

ckpts = sorted(glob.glob(os.path.join(run_dir, "logs/lightning_logs/version_0/checkpoints/*.ckpt")))
non_last = [p for p in ckpts if not p.endswith("/last.ckpt")]
best_ckpt = non_last[-1] if non_last else (ckpts[-1] if ckpts else "NONE")

lines = []
lines.append(f"finished_at: {__import__('datetime').datetime.now().strftime('%F %T')}")
lines.append(f"run_dir: {run_dir}")
lines.append(f"train_log: {log_path}")
lines.append("validation_trajectory:")
if last:
    for e in sorted(last):
        v, s = last[e]
        lines.append(f"  Ep{e}: val={v:.3f} std={s:.3f}")
    best_e, (best_v, best_s) = min(last.items(), key=lambda kv: kv[1][0])
    lines.append(f"best_completed_val: Ep{best_e} val={best_v:.3f} std={best_s:.3f}")
else:
    lines.append("  none")
lines.append(f"best_ckpt_by_filename: {best_ckpt}")

for dataset in ("vctk", "timit"):
    avg = os.path.join(os.path.dirname(log_path), "..", "results", "all_lmul_mixed_qat", "all_lmul_m4_bitexact_qat_w200", f"{dataset}_paper_style_results_averages.txt")
    avg = os.path.abspath(avg)
    if os.path.exists(avg):
        lines.append(f"{dataset}_w200_summary: {avg}")
        for line in open(avg, errors="ignore"):
            if "Average MBSTOI" in line or "Average PESQ" in line or "Average SegSNR Improvement" in line:
                lines.append("  " + line.strip())

open(out_path, "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
PY
}

echo "[monitor] started at $(date '+%F %T')" >> "$MONITOR_LOG"
echo "[monitor] log=${MONITOR_LOG}" >> "$MONITOR_LOG"
echo "[monitor] latest_summary=${LATEST_SUMMARY}" >> "$MONITOR_LOG"

while is_alive; do
  write_status
  sleep "$POLL_SECONDS"
done

write_status
{
  echo "===== final summary ====="
  write_final_summary
} >> "$MONITOR_LOG" 2>&1

cp "$MONITOR_LOG" "${REPO}/logs/monitor_v4_finetune_last.log"
