#!/bin/bash
# Phase D: grouped / block-diagonal MHA projection experiments.
#
# Usage:
#   TRAIN=0 bash scripts/run_low_power_phase_d_mha_grouping.sh d0_qkv_g2
#   bash scripts/run_low_power_phase_d_mha_grouping.sh d1_qkv_g4 d2_qkv_out_g4

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_DIR"

export PYTHONPATH="${PYTHONPATH:-}:${REPO_DIR}"
export HYDRA_FULL_ERROR=1
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl_cache}"

GPU_ID="${GPU_ID:-0}"
TRAIN="${TRAIN:-1}"
EPOCHS="${EPOCHS:-10}"
LR="${LR:-2e-4}"
BATCH_SIZE="${BATCH_SIZE:-8}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
LIMIT_PAIRS="${LIMIT_PAIRS:-20}"
METRIC_WORKERS="${METRIC_WORKERS:-64}"
MODE="${MODE:-shift_topk_pow2_clamp8_nomantissa}"
BASE_CKPT="${BASE_CKPT:-${REPO_DIR}/Checkpoints/p4_qk_m4_qat_best.ckpt}"
P2_INDEX="${P2_INDEX:-${REPO_DIR}/Checkpoints/p2_skip_indices_g42_topk_l1_r050.pt}"
OUT_ROOT="${OUT_ROOT:-${REPO_DIR}/outputs/low_power_phase_d}"
RESULT_ROOT="${RESULT_ROOT:-${REPO_DIR}/results/low_power_phase_d_sanity}"
VARIANTS=("$@")
if [ "${#VARIANTS[@]}" -eq 0 ]; then
  VARIANTS=(d0_qkv_g2 d1_qkv_g4 d2_qkv_out_g4 d3_io_g4 d4_all_g4)
fi

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

variant_overrides() {
  case "$1" in
    d0_qkv_g2)
      echo "model.low_power.mha_projection.qkv_groups=2"
      ;;
    d1_qkv_g4)
      echo "model.low_power.mha_projection.qkv_groups=4"
      ;;
    d2_qkv_out_g4)
      echo "model.low_power.mha_projection.qkv_groups=4 model.low_power.mha_projection.out_proj_groups=4"
      ;;
    d3_io_g4)
      echo "model.low_power.mha_projection.input_proj_groups=4 model.low_power.mha_projection.output_proj_groups=4"
      ;;
    d4_all_g4)
      echo "model.low_power.mha_projection.input_proj_groups=4 model.low_power.mha_projection.qkv_groups=4 model.low_power.mha_projection.out_proj_groups=4 model.low_power.mha_projection.output_proj_groups=4"
      ;;
    d5_qkv_g8)
      echo "model.low_power.mha_projection.qkv_groups=8"
      ;;
    *)
      echo "unknown Phase D variant: $1" >&2
      exit 2
      ;;
  esac
}

analysis_groups() {
  case "$1" in
    d0_qkv_g2) echo "--qkv_groups 2" ;;
    d1_qkv_g4) echo "--qkv_groups 4" ;;
    d2_qkv_out_g4) echo "--qkv_groups 4 --out_groups 4" ;;
    d3_io_g4) echo "--input_groups 4 --output_groups 4" ;;
    d4_all_g4) echo "--input_groups 4 --qkv_groups 4 --out_groups 4 --output_groups 4" ;;
    d5_qkv_g8) echo "--qkv_groups 8" ;;
    *) echo "" ;;
  esac
}

common_overrides=(
  training.qat_scope=full
  training.top_k_attention=4
  training.fp16_accumulator=true
  training.int4_scale_format=fp32
  model.attention_softmax.mode="$MODE"
  model.attention_datapath.enabled=true
  model.attention_datapath.score_quant=none
  model.attention_datapath.prob_quant=none
  model.attention_datapath.lmul_enabled=true
  model.attention_datapath.lmul_targets=[qk,score_v]
  model.attention_datapath.lmul_mantissa_bits=4
  model.low_power.mha_projection.enabled=true
  model.low_power.mha_projection.input_proj_groups=1
  model.low_power.mha_projection.qkv_groups=1
  model.low_power.mha_projection.out_proj_groups=1
  model.low_power.mha_projection.output_proj_groups=1
  model.decoder_grouped_pointwise.mode=grouped
  model.decoder_grouped_pointwise.groups_x=4
  model.decoder_grouped_pointwise.groups_skip=2
  model.decoder_grouped_pointwise.skip_select_ratio=0.5
  model.decoder_grouped_pointwise.skip_select_mode=topk_l1
  model.decoder_grouped_pointwise.skip_indices_path="$P2_INDEX"
)

if [ ! -f "$BASE_CKPT" ] || [ ! -f "$P2_INDEX" ]; then
  echo "missing base checkpoint or skip index" >&2
  echo "BASE_CKPT=$BASE_CKPT" >&2
  echo "P2_INDEX=$P2_INDEX" >&2
  exit 1
fi

mkdir -p "$OUT_ROOT" "$RESULT_ROOT" logs results/low_power_analysis

for variant in "${VARIANTS[@]}"; do
  read -r -a phase_overrides <<< "$(variant_overrides "$variant")"
  read -r -a analysis_args <<< "$(analysis_groups "$variant")"
  python scripts/analyze_low_power_arch.py \
    "${analysis_args[@]}" \
    --output_json "${RESULT_ROOT}/${variant}.json"

  log_file="logs/low_power_phase_d_${variant}_$(date +%Y%m%d_%H%M%S).log"
  base_sanity_dir="${RESULT_ROOT}/${variant}_base_w${LIMIT_PAIRS}"
  echo "[base-sanity] ${variant}: checkpoint=${BASE_CKPT}, output=${base_sanity_dir}"
  CUDA_VISIBLE_DEVICES="$GPU_ID" conda run --no-capture-output -n bcctn python eval.py \
    --model_checkpoint "$BASE_CKPT" \
    --vctk_test_dir /raid/R12K41024/BCCTN/Dataset/noisy_testset \
    --vctk_clean_dir /raid/R12K41024/BCCTN/Dataset/clean_testset \
    --output_dir "$base_sanity_dir" \
    --paper_style_eval \
    --batch_size "$BATCH_SIZE" \
    --limit_pairs "$LIMIT_PAIRS" \
    --metric_workers "$METRIC_WORKERS" \
    --hydra_overrides \
      "${common_overrides[@]}" \
      "${phase_overrides[@]}" \
    2>&1 | tee "$log_file"

  if [ "$TRAIN" = "0" ]; then
    continue
  fi

  run_dir="${OUT_ROOT}/${variant}_$(date +%F_%H-%M-%S)"
  echo "[train] ${variant}: epochs=${EPOCHS}, lr=${LR}, run_dir=${run_dir}"
  CUDA_VISIBLE_DEVICES="$GPU_ID" conda run --no-capture-output -n bcctn python -u train.py \
    training.train_checkpoint_path="$BASE_CKPT" \
    training.n_epochs="$EPOCHS" \
    training.learning_rate="$LR" \
    training.batch_size="$TRAIN_BATCH_SIZE" \
    "${common_overrides[@]}" \
    "${phase_overrides[@]}" \
    hydra.run.dir="$run_dir" \
    2>&1 | tee -a "$log_file"

  best_ckpt="$(latest_non_last_ckpt "$run_dir")"
  sanity_dir="${RESULT_ROOT}/${variant}_finetuned_w${LIMIT_PAIRS}"
  echo "[finetuned-sanity] ${variant}: checkpoint=${best_ckpt}, output=${sanity_dir}"
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
      "${common_overrides[@]}" \
      "${phase_overrides[@]}" \
    2>&1 | tee -a "$log_file"
done

echo "[phase-d] done. Results under ${RESULT_ROOT}; analysis under results/low_power_analysis"
