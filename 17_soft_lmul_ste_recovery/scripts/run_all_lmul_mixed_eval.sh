#!/bin/bash
# Evaluate the final mixed W4A8 stack with all MAC-bearing Conv/Linear ops
# switched to the L-Mul product approximation.
#
# Usage:
#   LIMIT_PAIRS=20 BATCH_SIZE=1 bash scripts/run_all_lmul_mixed_eval.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_DIR"

export PYTHONPATH="${PYTHONPATH:-}:${REPO_DIR}"
export HYDRA_FULL_ERROR=1
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl_cache}"

GPU_ID="${GPU_ID:-0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LIMIT_PAIRS="${LIMIT_PAIRS:-20}"
METRIC_WORKERS="${METRIC_WORKERS:-16}"
MODE="${MODE:-shift_topk_pow2_clamp8_nomantissa}"
BASE_CKPT="${BASE_CKPT:-${REPO_DIR}/Checkpoints/p4_qk_m4_qat_best.ckpt}"
P2_INDEX="${P2_INDEX:-${REPO_DIR}/Checkpoints/p2_skip_indices_g42_topk_l1_r050.pt}"
RESULT_ROOT="${RESULT_ROOT:-${REPO_DIR}/results/all_lmul_mixed_eval}"
RUN_TIMIT="${RUN_TIMIT:-0}"
TIMIT_TEST_DIR="${TIMIT_TEST_DIR:-/raid/R12K41024/BCCTN/Dataset/noisy_testset_timit}"
TIMIT_CLEAN_DIR="${TIMIT_CLEAN_DIR:-/raid/R12K41024/BCCTN/Dataset/clean_testset_timit}"
LMUL_LINEAR_CHUNK_SIZE="${LMUL_LINEAR_CHUNK_SIZE:-0}"
LMUL_CHECKPOINT_CHUNKS="${LMUL_CHECKPOINT_CHUNKS:-false}"

timit_args=()
if [ "$RUN_TIMIT" = "1" ]; then
  timit_args=(
    --timit_test_dir "$TIMIT_TEST_DIR"
    --timit_clean_dir "$TIMIT_CLEAN_DIR"
  )
fi

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
  model.attention_datapath.lmul_targets=[qk]
  model.attention_datapath.lmul_mantissa_bits=4
  model.attention_datapath.lmul_bit_exact=true
  model.attention_datapath.lmul_linear_chunk_size="$LMUL_LINEAR_CHUNK_SIZE"
  model.attention_datapath.lmul_checkpoint_chunks="$LMUL_CHECKPOINT_CHUNKS"
  model.attention_datapath.prob_packet.enabled=true
  model.attention_datapath.prob_packet.top_k=4
  model.attention_datapath.prob_packet.local_idx_bits=5
  model.attention_datapath.prob_packet.score_v_mode=lmul
  model.attention_datapath.prob_packet.validate_local_idx=true
  model.attention_datapath.prob_packet.prob_bits=8
  model.attention_datapath.prob_packet.prob_scale=256
  model.attention_datapath.prob_packet.prob_max_code=255
  model.attention_datapath.prob_packet.causal_window=null
  model.attention_datapath.prob_packet.renormalize=false
  model.decoder_grouped_pointwise.mode=grouped
  model.decoder_grouped_pointwise.groups_x=4
  model.decoder_grouped_pointwise.groups_skip=2
  model.decoder_grouped_pointwise.skip_select_ratio=0.5
  model.decoder_grouped_pointwise.skip_select_mode=topk_l1
  model.decoder_grouped_pointwise.skip_indices_path="$P2_INDEX"
  model.low_power.mha_projection.enabled=true
  model.low_power.mha_projection.qkv_groups=4
)

variant_overrides() {
  case "$1" in
    baseline_final)
      echo "model.low_power.all_lmul_multiply.enabled=false model.attention_datapath.lmul_all_projection_ops=false"
      ;;
    all_lmul_m4_bitexact)
      echo "model.low_power.all_lmul_multiply.enabled=true model.low_power.all_lmul_multiply.mantissa_bits=4 model.low_power.all_lmul_multiply.bit_exact=true model.low_power.all_lmul_multiply.linear_chunk_size=${LMUL_LINEAR_CHUNK_SIZE} model.low_power.all_lmul_multiply.checkpoint_chunks=${LMUL_CHECKPOINT_CHUNKS} model.attention_datapath.lmul_all_projection_ops=true"
      ;;
    *)
      echo "unknown variant: $1" >&2
      exit 2
      ;;
  esac
}

if [ ! -f "$BASE_CKPT" ] || [ ! -f "$P2_INDEX" ]; then
  echo "missing base checkpoint or skip index" >&2
  echo "BASE_CKPT=$BASE_CKPT" >&2
  echo "P2_INDEX=$P2_INDEX" >&2
  exit 1
fi

mkdir -p "$RESULT_ROOT" logs

for variant in baseline_final all_lmul_m4_bitexact; do
  read -r -a extra_overrides <<< "$(variant_overrides "$variant")"
  out_dir="${RESULT_ROOT}/${variant}_w${LIMIT_PAIRS}"
  log_file="logs/all_lmul_mixed_${variant}_$(date +%Y%m%d_%H%M%S).log"
  echo "[eval] ${variant}: checkpoint=${BASE_CKPT}, output=${out_dir}"
  CUDA_VISIBLE_DEVICES="$GPU_ID" conda run --no-capture-output -n bcctn python eval.py \
    --model_checkpoint "$BASE_CKPT" \
    --vctk_test_dir /raid/R12K41024/BCCTN/Dataset/noisy_testset \
    --vctk_clean_dir /raid/R12K41024/BCCTN/Dataset/clean_testset \
    "${timit_args[@]}" \
    --output_dir "$out_dir" \
    --paper_style_eval \
    --batch_size "$BATCH_SIZE" \
    --limit_pairs "$LIMIT_PAIRS" \
    --metric_workers "$METRIC_WORKERS" \
    --hydra_overrides \
      "${common_overrides[@]}" \
      "${extra_overrides[@]}" \
    2>&1 | tee "$log_file"
done
