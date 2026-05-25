#!/bin/bash
# QAT recovery run for the all-MAC L-Mul mixed W4A8 backend.
#
# Usage:
#   GPU_ID=4 EPOCHS=10 LR=1e-4 LIMIT_PAIRS=20 BATCH_SIZE=1 TRAIN_BATCH_SIZE=4 \
#     bash scripts/run_all_lmul_mixed_qat_rescue.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_DIR"

export PYTHONPATH="${PYTHONPATH:-}:${REPO_DIR}"
export HYDRA_FULL_ERROR=1
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl_cache}"

GPU_ID="${GPU_ID:-0}"
EPOCHS="${EPOCHS:-10}"
LR="${LR:-1e-4}"
BATCH_SIZE="${BATCH_SIZE:-1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-1}"
NUM_SANITY_VAL_STEPS="${NUM_SANITY_VAL_STEPS:-0}"
LIMIT_PAIRS="${LIMIT_PAIRS:-20}"
METRIC_WORKERS="${METRIC_WORKERS:-64}"
MODE="${MODE:-shift_topk_pow2_clamp8_nomantissa}"
BASE_CKPT="${BASE_CKPT:-${REPO_DIR}/Checkpoints/p4_qk_m4_qat_best.ckpt}"
# RESUME_CKPT: set this to enable true Lightning resume (restores optimizer/scheduler/epoch).
# When set, BASE_CKPT is ignored and EPOCHS becomes the total epoch cap (not additional epochs).
# The ckpt must have been saved with ModelCheckpoint(save_weights_only=False).
RESUME_CKPT="${RESUME_CKPT:-}"
P2_INDEX="${P2_INDEX:-${REPO_DIR}/Checkpoints/p2_skip_indices_g42_topk_l1_r050.pt}"
OUT_ROOT="${OUT_ROOT:-${REPO_DIR}/outputs/all_lmul_mixed_qat}"
RESULT_ROOT="${RESULT_ROOT:-${REPO_DIR}/results/all_lmul_mixed_qat}"
LMUL_LINEAR_CHUNK_SIZE="${LMUL_LINEAR_CHUNK_SIZE:-64}"
LMUL_CHECKPOINT_CHUNKS="${LMUL_CHECKPOINT_CHUNKS:-true}"

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

overrides=(
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
  model.attention_datapath.lmul_all_projection_ops=true
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
  model.low_power.all_lmul_multiply.enabled=true
  model.low_power.all_lmul_multiply.mantissa_bits=4
  model.low_power.all_lmul_multiply.bit_exact=true
  model.low_power.all_lmul_multiply.linear_chunk_size="$LMUL_LINEAR_CHUNK_SIZE"
  model.low_power.all_lmul_multiply.checkpoint_chunks="$LMUL_CHECKPOINT_CHUNKS"
)

if [ -n "$RESUME_CKPT" ]; then
  ACTIVE_CKPT="$RESUME_CKPT"
  CKPT_OVERRIDE="training.resume_ckpt_path='${RESUME_CKPT}'"
  CKPT_MODE_LABEL="resume (full state; EPOCHS=${EPOCHS} is TOTAL cap)"
else
  ACTIVE_CKPT="$BASE_CKPT"
  CKPT_OVERRIDE="training.train_checkpoint_path='${BASE_CKPT}'"
  CKPT_MODE_LABEL="warm-start (weights only; fresh Adam)"
fi

if [ ! -f "$ACTIVE_CKPT" ] || [ ! -f "$P2_INDEX" ]; then
  echo "missing checkpoint or skip index" >&2
  echo "ACTIVE_CKPT=$ACTIVE_CKPT (mode=${CKPT_MODE_LABEL})" >&2
  echo "P2_INDEX=$P2_INDEX" >&2
  exit 1
fi

mkdir -p "$OUT_ROOT" "$RESULT_ROOT" logs

run_dir="${OUT_ROOT}/all_lmul_m4_bitexact_qat_$(date +%F_%H-%M-%S)"
log_file="logs/all_lmul_mixed_qat_$(date +%Y%m%d_%H%M%S).log"

echo "[train] all_lmul_m4_bitexact_qat: mode=${CKPT_MODE_LABEL}, ckpt=${ACTIVE_CKPT}, epochs=${EPOCHS}, lr=${LR}, batch=${TRAIN_BATCH_SIZE}, accum=${ACCUMULATE_GRAD_BATCHES}, sanity=${NUM_SANITY_VAL_STEPS}, chunk=${LMUL_LINEAR_CHUNK_SIZE}, checkpoint=${LMUL_CHECKPOINT_CHUNKS}, run_dir=${run_dir}"
CUDA_VISIBLE_DEVICES="$GPU_ID" conda run --no-capture-output -n bcctn python -u train.py \
  "$CKPT_OVERRIDE" \
  training.n_epochs="$EPOCHS" \
  training.learning_rate="$LR" \
  training.batch_size="$TRAIN_BATCH_SIZE" \
  training.accumulate_grad_batches="$ACCUMULATE_GRAD_BATCHES" \
  training.num_sanity_val_steps="$NUM_SANITY_VAL_STEPS" \
  "${overrides[@]}" \
  hydra.run.dir="$run_dir" \
  2>&1 | tee "$log_file"

best_ckpt="$(latest_non_last_ckpt "$run_dir")"
echo "[best] ${best_ckpt}" | tee -a "$log_file"

sanity_dir="${RESULT_ROOT}/all_lmul_m4_bitexact_qat_w${LIMIT_PAIRS}"
echo "[sanity] checkpoint=${best_ckpt}, output=${sanity_dir}"
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
    "${overrides[@]}" \
  2>&1 | tee -a "$log_file"

echo "[all-lmul-qat] done. Results under ${RESULT_ROOT}"
