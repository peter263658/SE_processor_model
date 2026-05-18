#!/bin/bash
# Train P2 skip-subset variants, then run 20-pair sanity eval.
#
# Usage:
#   bash scripts/run_phase2_skip_sweep.sh s075_topk
#   bash scripts/run_phase2_skip_sweep.sh s075_first s075_topk s050_first s050_topk

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
EPOCHS="${EPOCHS:-30}"
LR="${LR:-5e-4}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LIMIT_PAIRS="${LIMIT_PAIRS:-20}"
METRIC_WORKERS="${METRIC_WORKERS:-64}"
MODE="${MODE:-shift_topk_pow2_clamp8_nomantissa}"
WARMSTART_DIR="${WARMSTART_DIR:-${REPO_DIR}/Checkpoints}"
INDEX_DIR="${INDEX_DIR:-${REPO_DIR}/Checkpoints}"
OUT_ROOT="${OUT_ROOT:-${REPO_DIR}/outputs/p2_skip}"
RESULT_ROOT="${RESULT_ROOT:-${REPO_DIR}/results/p2_skip_sanity}"
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
  read -r skip_mode ratio tag < <(variant_mode_ratio "$variant")
  warmstart="${WARMSTART_DIR}/p2_warmstart_${P1_VARIANT}_${skip_mode}_r${tag}.ckpt"
  index_path="${INDEX_DIR}/p2_skip_indices_${P1_VARIANT}_${skip_mode}_r${tag}.pt"
  if [ ! -f "$warmstart" ] || [ ! -f "$index_path" ]; then
    echo "missing P2 warm-start or skip index for ${variant}" >&2
    echo "run: bash scripts/make_phase2_skip_warmstarts.sh ${variant}" >&2
    exit 1
  fi

  run_dir="${OUT_ROOT}/${variant}_$(date +%F_%H-%M-%S)"
  log_file="logs/p2_skip_${variant}_$(date +%Y%m%d_%H%M%S).log"
  echo "[train] ${variant}: ratio=${ratio}, skip_mode=${skip_mode}, run_dir=${run_dir}"

  CUDA_VISIBLE_DEVICES="$GPU_ID" conda run --no-capture-output -n bcctn python -u train.py \
    training.train_checkpoint_path="$warmstart" \
    training.n_epochs="$EPOCHS" \
    training.learning_rate="$LR" \
    training.qat_scope=full \
    training.top_k_attention=4 \
    model.attention_softmax.mode="$MODE" \
    model.decoder_grouped_pointwise.mode=grouped \
    model.decoder_grouped_pointwise.groups_x="$GX" \
    model.decoder_grouped_pointwise.groups_skip="$GS" \
    model.decoder_grouped_pointwise.skip_select_ratio="$ratio" \
    model.decoder_grouped_pointwise.skip_select_mode="$skip_mode" \
    model.decoder_grouped_pointwise.skip_indices_path="$index_path" \
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
      model.decoder_grouped_pointwise.groups_x="$GX" \
      model.decoder_grouped_pointwise.groups_skip="$GS" \
      model.decoder_grouped_pointwise.skip_select_ratio="$ratio" \
      model.decoder_grouped_pointwise.skip_select_mode="$skip_mode" \
      model.decoder_grouped_pointwise.skip_indices_path="$index_path"
done
