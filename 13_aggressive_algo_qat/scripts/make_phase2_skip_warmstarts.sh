#!/bin/bash
# Build P2 skip-subset index files and warm-start checkpoints from the P1 winner.
#
# Usage:
#   bash scripts/make_phase2_skip_warmstarts.sh
#   bash scripts/make_phase2_skip_warmstarts.sh s075_topk s050_topk

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_DIR"

export PYTHONPATH="${PYTHONPATH:-}:${REPO_DIR}"
export HYDRA_FULL_ERROR=1
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl_cache}"

GPU_ID="${GPU_ID:-0}"
P1_VARIANT="${P1_VARIANT:-g42}"
GX="${GX:-4}"
GS="${GS:-2}"

# Auto-detect the latest non-last ckpt for $P1_VARIANT unless P1_CKPT is set
# explicitly. Avoids the script breaking when a new g42 finetune run is added.
auto_p1_ckpt() {
  local variant="$1"
  local run_dir ckpt_dir ckpt
  run_dir="$(ls -d "${REPO_DIR}/outputs/p1_grouped/${variant}"_* 2>/dev/null | sort | tail -n 1)"
  if [ -z "$run_dir" ]; then return 1; fi
  ckpt_dir="${run_dir}/logs/lightning_logs/version_0/checkpoints"
  ckpt="$(find "$ckpt_dir" -maxdepth 1 -type f -name '*.ckpt' ! -name 'last.ckpt' 2>/dev/null | sort | tail -n 1)"
  if [ -z "$ckpt" ]; then ckpt="${ckpt_dir}/last.ckpt"; fi
  if [ ! -f "$ckpt" ]; then return 1; fi
  echo "$ckpt"
}

if [ -z "${P1_CKPT:-}" ]; then
  P1_CKPT="$(auto_p1_ckpt "$P1_VARIANT")" || {
    echo "auto-detect P1 ckpt failed for variant=$P1_VARIANT" >&2
    echo "set P1_CKPT explicitly to override" >&2
    exit 1
  }
  echo "[autoprobe] using P1_CKPT=$P1_CKPT"
fi
INDEX_DIR="${INDEX_DIR:-${REPO_DIR}/Checkpoints}"
WARMSTART_DIR="${WARMSTART_DIR:-${REPO_DIR}/Checkpoints}"
TOPK_BATCH_SIZE="${TOPK_BATCH_SIZE:-8}"
TOPK_N_WORKERS="${TOPK_N_WORKERS:-4}"
LIMIT_BATCHES="${LIMIT_BATCHES:-0}"
VARIANTS=("$@")
if [ "${#VARIANTS[@]}" -eq 0 ]; then
  VARIANTS=(s075_first s075_topk s050_first s050_topk)
fi

variant_mode_ratio() {
  case "$1" in
    s075_first) echo "first_n 0.75 075" ;;
    s075_topk) echo "topk_l1 0.75 075" ;;
    s050_first) echo "first_n 0.50 050" ;;
    s050_topk) echo "topk_l1 0.50 050" ;;
    *)
      echo "unknown P2 variant: $1" >&2
      exit 2
      ;;
  esac
}

if [ ! -f "$P1_CKPT" ]; then
  echo "missing P1 checkpoint: $P1_CKPT" >&2
  exit 1
fi

mkdir -p "$INDEX_DIR" "$WARMSTART_DIR"

for variant in "${VARIANTS[@]}"; do
  read -r mode ratio tag < <(variant_mode_ratio "$variant")
  index_path="${INDEX_DIR}/p2_skip_indices_${P1_VARIANT}_${mode}_r${tag}.pt"
  warmstart_path="${WARMSTART_DIR}/p2_warmstart_${P1_VARIANT}_${mode}_r${tag}.ckpt"

  if [ ! -f "$index_path" ]; then
    echo "[indices] ${variant}: mode=${mode}, ratio=${ratio}, out=${index_path}"
    if [ "$mode" = "topk_l1" ]; then
      CUDA_VISIBLE_DEVICES="$GPU_ID" conda run --no-capture-output -n bcctn python scripts/precompute_skip_indices.py \
        --checkpoint "$P1_CKPT" \
        --out "$index_path" \
        --mode "$mode" \
        --ratio "$ratio" \
        --groups_x "$GX" \
        --groups_skip "$GS" \
        --batch_size "$TOPK_BATCH_SIZE" \
        --n_workers "$TOPK_N_WORKERS" \
        --limit_batches "$LIMIT_BATCHES"
    else
      conda run --no-capture-output -n bcctn python scripts/precompute_skip_indices.py \
        --out "$index_path" \
        --mode "$mode" \
        --ratio "$ratio" \
        --groups_x "$GX" \
        --groups_skip "$GS"
    fi
  else
    echo "[indices] ${variant}: reuse ${index_path}"
  fi

  echo "[convert] ${variant}: ${warmstart_path}"
  conda run --no-capture-output -n bcctn python scripts/convert_ckpt_to_skip_subset.py \
    --in "$P1_CKPT" \
    --out "$warmstart_path" \
    --skip-indices "$index_path" \
    --ratio "$ratio" \
    --groups_x "$GX" \
    --groups_skip "$GS"
done
