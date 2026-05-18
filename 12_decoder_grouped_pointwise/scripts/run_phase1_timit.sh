#!/bin/bash
# Run VCTK+TIMIT cross-domain eval on P1 best candidates.
#
# Usage:
#   bash scripts/run_phase1_timit.sh g42 g44
#   GPU_ID=1 bash scripts/run_phase1_timit.sh g42

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_DIR"

export PYTHONPATH="${PYTHONPATH:-}:${REPO_DIR}"
export HYDRA_FULL_ERROR=1
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl_cache}"

GPU_ID="${GPU_ID:-0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LIMIT_PAIRS="${LIMIT_PAIRS:-750}"
METRIC_WORKERS="${METRIC_WORKERS:-64}"
MODE="${MODE:-shift_topk_pow2_clamp8_nomantissa}"
OUT_ROOT="${OUT_ROOT:-${REPO_DIR}/outputs/p1_grouped}"
RESULT_ROOT="${RESULT_ROOT:-${REPO_DIR}/results/p1_grouped_timit}"
VARIANTS=("$@")
if [ "${#VARIANTS[@]}" -eq 0 ]; then
  VARIANTS=(g42 g44)
fi

variant_groups() {
  case "$1" in
    g11) echo "1 1" ;;
    g21) echo "2 1" ;;
    g12) echo "1 2" ;;
    g22) echo "2 2" ;;
    g42) echo "4 2" ;;
    g44) echo "4 4" ;;
    *)
      echo "unknown P1 variant: $1" >&2
      exit 2
      ;;
  esac
}

latest_non_last_ckpt() {
  ckpt_dir="$1/logs/lightning_logs/version_0/checkpoints"
  ckpt="$(find "$ckpt_dir" -maxdepth 1 -type f -name '*.ckpt' ! -name 'last.ckpt' | sort | tail -n 1)"
  if [ -z "$ckpt" ]; then
    ckpt="${ckpt_dir}/last.ckpt"
  fi
  if [ ! -f "$ckpt" ]; then
    echo "no checkpoint found under $ckpt_dir" >&2
    exit 1
  fi
  echo "$ckpt"
}

latest_run_dir() {
  variant="$1"
  run_dir="$(ls -d "${OUT_ROOT}/${variant}"_* 2>/dev/null | sort | tail -n 1)"
  if [ -z "$run_dir" ]; then
    echo "no run dir found for variant $variant under $OUT_ROOT" >&2
    exit 1
  fi
  echo "$run_dir"
}

mkdir -p "$RESULT_ROOT" logs

for variant in "${VARIANTS[@]}"; do
  read -r gx gs < <(variant_groups "$variant")
  run_dir="$(latest_run_dir "$variant")"
  best_ckpt="$(latest_non_last_ckpt "$run_dir")"
  out_dir="${RESULT_ROOT}/${variant}_vctk_timit_w${LIMIT_PAIRS}"
  log_file="logs/p1_timit_${variant}_$(date +%Y%m%d_%H%M%S).log"
  echo "[timit] ${variant}: gx=${gx}, gs=${gs}, ckpt=${best_ckpt}, output=${out_dir}"

  CUDA_VISIBLE_DEVICES="$GPU_ID" conda run --no-capture-output -n bcctn python eval.py \
    --model_checkpoint "$best_ckpt" \
    --vctk_test_dir /raid/R12K41024/BCCTN/Dataset/noisy_testset \
    --vctk_clean_dir /raid/R12K41024/BCCTN/Dataset/clean_testset \
    --timit_test_dir /raid/R12K41024/BCCTN/Dataset/noisy_testset_timit \
    --timit_clean_dir /raid/R12K41024/BCCTN/Dataset/clean_testset_timit \
    --output_dir "$out_dir" \
    --paper_style_eval \
    --batch_size "$BATCH_SIZE" \
    --limit_pairs "$LIMIT_PAIRS" \
    --metric_workers "$METRIC_WORKERS" \
    --hydra_overrides \
      training.qat_scope=full \
      training.top_k_attention=4 \
      model.attention_softmax.mode="$MODE" \
      model.decoder_grouped_pointwise.mode=grouped \
      model.decoder_grouped_pointwise.groups_x="$gx" \
      model.decoder_grouped_pointwise.groups_skip="$gs" \
    2>&1 | tee "$log_file"
done

echo "[timit] done. Results under ${RESULT_ROOT}"
