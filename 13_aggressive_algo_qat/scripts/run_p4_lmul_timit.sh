#!/bin/bash
# Full 750-pair VCTK eval for P4 L-Mul variants (post-QAT).
#
# Mirrors scripts/run_phase3_attention_full_eval.sh but for L-Mul targets.
# Auto-detects the latest finetuned ckpt under outputs/p4_lmul/<variant>_<TS>/.
#
# Usage:
#   bash scripts/run_p4_lmul_full_eval.sh lmul_qk_m4 lmul_scorev_m4
#   GPU_ID=1 LIMIT_PAIRS=750 bash scripts/run_p4_lmul_full_eval.sh

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
P2_INDEX="${P2_INDEX:-${REPO_DIR}/Checkpoints/p2_skip_indices_g42_topk_l1_r050.pt}"
OUT_ROOT="${OUT_ROOT:-${REPO_DIR}/outputs/p4_lmul}"
RESULT_ROOT="${RESULT_ROOT:-${REPO_DIR}/results/p4_lmul_timit}"
VARIANTS=("$@")
if [ "${#VARIANTS[@]}" -eq 0 ]; then
  VARIANTS=(lmul_qk_m4 lmul_scorev_m4)
fi

variant_overrides() {
  case "$1" in
    lmul_qk_m4)
      echo "training.fp16_accumulator=true model.attention_datapath.enabled=true model.attention_datapath.lmul_enabled=true model.attention_datapath.lmul_targets=[qk] model.attention_datapath.lmul_mantissa_bits=4 training.int4_scale_format=fp32"
      ;;
    lmul_scorev_m4)
      echo "training.fp16_accumulator=true model.attention_datapath.enabled=true model.attention_datapath.lmul_enabled=true model.attention_datapath.lmul_targets=[score_v] model.attention_datapath.lmul_mantissa_bits=4 training.int4_scale_format=fp32"
      ;;
    lmul_attn_m4)
      echo "training.fp16_accumulator=true model.attention_datapath.enabled=true model.attention_datapath.lmul_enabled=true model.attention_datapath.lmul_targets=[qk,score_v] model.attention_datapath.lmul_mantissa_bits=4 training.int4_scale_format=fp32"
      ;;
    *)
      echo "unknown P4 L-Mul variant: $1" >&2
      exit 2
      ;;
  esac
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

common_overrides=(
  training.qat_scope=full
  training.top_k_attention=4
  model.attention_softmax.mode="$MODE"
  model.decoder_grouped_pointwise.mode=grouped
  model.decoder_grouped_pointwise.groups_x=4
  model.decoder_grouped_pointwise.groups_skip=2
  model.decoder_grouped_pointwise.skip_select_ratio=0.5
  model.decoder_grouped_pointwise.skip_select_mode=topk_l1
  model.decoder_grouped_pointwise.skip_indices_path="$P2_INDEX"
)

mkdir -p "$RESULT_ROOT" logs

for variant in "${VARIANTS[@]}"; do
  read -r -a p4_overrides <<< "$(variant_overrides "$variant")"
  run_dir="$(latest_run_dir "$variant")"
  best_ckpt="$(latest_non_last_ckpt "$run_dir")"
  out_dir="${RESULT_ROOT}/${variant}_vctk_timit_w${LIMIT_PAIRS}"
  log_file="logs/p4_lmul_timit_${variant}_$(date +%Y%m%d_%H%M%S).log"
  echo "[full-eval] ${variant}: checkpoint=${best_ckpt}, output=${out_dir}"

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
      "${common_overrides[@]}" \
      "${p4_overrides[@]}" \
    2>&1 | tee "$log_file"
done

echo "[p4-lmul-timit] done. Results under ${RESULT_ROOT}"
