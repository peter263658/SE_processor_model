#!/bin/bash
# Train P1 decoder grouped-pointwise variants, then run 20-pair sanity eval.
#
# Usage:
#   bash scripts/run_phase1_grouped_sweep.sh g11
#   bash scripts/run_phase1_grouped_sweep.sh g22 g21 g12 g42 g44

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_DIR"

export PYTHONPATH="${PYTHONPATH:-}:${REPO_DIR}"
export HYDRA_FULL_ERROR=1
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl_cache}"

GPU_ID="${GPU_ID:-0}"
EPOCHS="${EPOCHS:-30}"
LR="${LR:-5e-4}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LIMIT_PAIRS="${LIMIT_PAIRS:-20}"
METRIC_WORKERS="${METRIC_WORKERS:-64}"
MODE="${MODE:-shift_topk_pow2_clamp8_nomantissa}"
WARMSTART_DIR="${WARMSTART_DIR:-${REPO_DIR}/Checkpoints}"
OUT_ROOT="${OUT_ROOT:-${REPO_DIR}/outputs/p1_grouped}"
RESULT_ROOT="${RESULT_ROOT:-${REPO_DIR}/results/p1_grouped_sanity}"
VARIANTS=("$@")
if [ "${#VARIANTS[@]}" -eq 0 ]; then
  VARIANTS=(g11 g22 g21 g12 g42 g44)
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

mkdir -p "$OUT_ROOT" "$RESULT_ROOT" logs

for variant in "${VARIANTS[@]}"; do
  read -r gx gs < <(variant_groups "$variant")
  warmstart="${WARMSTART_DIR}/p1_warmstart_${variant}.ckpt"
  if [ ! -f "$warmstart" ]; then
    echo "missing warm-start checkpoint: $warmstart" >&2
    echo "run: bash scripts/make_phase1_warmstarts.sh $variant" >&2
    exit 1
  fi

  run_dir="${OUT_ROOT}/${variant}_$(date +%F_%H-%M-%S)"
  log_file="logs/p1_grouped_${variant}_$(date +%Y%m%d_%H%M%S).log"
  echo "[train] ${variant}: groups_x=${gx}, groups_skip=${gs}, run_dir=${run_dir}"

  CUDA_VISIBLE_DEVICES="$GPU_ID" conda run --no-capture-output -n bcctn python -u train.py \
    training.train_checkpoint_path="$warmstart" \
    training.n_epochs="$EPOCHS" \
    training.learning_rate="$LR" \
    training.qat_scope=full \
    training.top_k_attention=4 \
    model.attention_softmax.mode="$MODE" \
    model.decoder_grouped_pointwise.mode=grouped \
    model.decoder_grouped_pointwise.groups_x="$gx" \
    model.decoder_grouped_pointwise.groups_skip="$gs" \
    hydra.run.dir="$run_dir" \
    2>&1 | tee "$log_file"

  best_ckpt="$(latest_non_last_ckpt "$run_dir")"
  sanity_dir="${RESULT_ROOT}/${variant}_w${LIMIT_PAIRS}"
  echo "[sanity] ${variant}: checkpoint=${best_ckpt}, output=${sanity_dir}"
  CUDA_VISIBLE_DEVICES="$GPU_ID" conda run --no-capture-output -n bcctn python eval.py \
    --model_checkpoint "$best_ckpt" \
    --vctk_test_dir /raid/R12K41024/BCCTN/Dataset/noisy_testset \
    --vctk_clean_dir /raid/R12K41024/BCCTN/Dataset/clean_testset \
    --output_dir "$sanity_dir" \
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
      model.decoder_grouped_pointwise.groups_skip="$gs"
done
