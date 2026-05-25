#!/bin/bash
# Output-level KD fine-tune for the all-MAC L-Mul student.
#
# This is intentionally a smoke/staging wrapper:
#   - requires a complete precomputed teacher-output cache
#   - warm-starts student weights with fresh Adam
#   - runs a short KD training window by default
#
# Usage:
#   GPU_ID=3 TEACHER_OUTPUT_DIR=/raid/.../teacher_outputs/d1c3_epoch25_step100672_fp16 \
#     bash scripts/run_kd_output_qat_smoke.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_DIR"

export PYTHONPATH="${PYTHONPATH:-}:${REPO_DIR}"
export HYDRA_FULL_ERROR=1
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl_cache}"

GPU_ID="${GPU_ID:-3}"
EPOCHS="${EPOCHS:-5}"
LR="${LR:-5e-6}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-1}"
NUM_SANITY_VAL_STEPS="${NUM_SANITY_VAL_STEPS:-0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LIMIT_PAIRS="${LIMIT_PAIRS:-200}"
METRIC_WORKERS="${METRIC_WORKERS:-64}"
KD_LAMBDA="${KD_LAMBDA:-0.1}"
KD_MODE="${KD_MODE:-waveform_mse}"
MODE="${MODE:-shift_topk_pow2_clamp8_nomantissa}"
BASE_CKPT="${BASE_CKPT:-/raid/R12K41024/15_all_lmul_mixed_mult_eval/outputs/all_lmul_mixed_qat/all_lmul_m4_bitexact_qat_2026-05-24_02-49-17/logs/lightning_logs/version_0/checkpoints/epoch=5-step=1452.ckpt}"
RESUME_CKPT="${RESUME_CKPT:-}"
TEACHER_OUTPUT_DIR="${TEACHER_OUTPUT_DIR:?TEACHER_OUTPUT_DIR is required}"
NOISY_TRAIN_DIR="${NOISY_TRAIN_DIR:-/raid/R12K41024/BCCTN/Dataset/noisy_trainset}"
P2_INDEX="${P2_INDEX:-${REPO_DIR}/Checkpoints/p2_skip_indices_g42_topk_l1_r050.pt}"
OUT_ROOT="${OUT_ROOT:-${REPO_DIR}/outputs/kd_output_qat}"
RESULT_ROOT="${RESULT_ROOT:-${REPO_DIR}/results/kd_output_qat}"
# Match v4 throughput defaults. Override to 64/true only if memory becomes tight.
LMUL_LINEAR_CHUNK_SIZE="${LMUL_LINEAR_CHUNK_SIZE:-2048}"
LMUL_CHECKPOINT_CHUNKS="${LMUL_CHECKPOINT_CHUNKS:-false}"

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

for path in "$BASE_CKPT" "$P2_INDEX"; do
  if [ ! -f "$path" ]; then
    echo "missing required file: $path" >&2
    exit 1
  fi
done
if [ -n "$RESUME_CKPT" ] && [ ! -f "$RESUME_CKPT" ]; then
  echo "missing RESUME_CKPT: $RESUME_CKPT" >&2
  exit 1
fi
if [ ! -d "$TEACHER_OUTPUT_DIR" ] || [ ! -d "$NOISY_TRAIN_DIR" ]; then
  echo "missing TEACHER_OUTPUT_DIR or NOISY_TRAIN_DIR" >&2
  echo "TEACHER_OUTPUT_DIR=$TEACHER_OUTPUT_DIR" >&2
  echo "NOISY_TRAIN_DIR=$NOISY_TRAIN_DIR" >&2
  exit 1
fi

cache_count="$(find "$TEACHER_OUTPUT_DIR" -type f -name '*.pt' ! -name '*.tmp.pt' | wc -l)"
train_count="$(find "$NOISY_TRAIN_DIR" -type f -name '*.wav' | wc -l)"
if [ "$cache_count" -lt "$train_count" ]; then
  echo "teacher cache is incomplete: ${cache_count}/${train_count} .pt files" >&2
  exit 1
fi

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
  model.kd.enabled=true
  model.kd.teacher_output_dir="$TEACHER_OUTPUT_DIR"
  model.kd.teacher_noisy_root="$NOISY_TRAIN_DIR"
  model.kd.lambda_kd="$KD_LAMBDA"
  model.kd.kd_mode="$KD_MODE"
)

mkdir -p "$OUT_ROOT" "$RESULT_ROOT" logs

run_dir="${OUT_ROOT}/all_lmul_m4_output_kd_l${KD_LAMBDA}_${KD_MODE}_$(date +%F_%H-%M-%S)"
log_file="logs/kd_output_qat_l${KD_LAMBDA}_${KD_MODE}_$(date +%Y%m%d_%H%M%S).log"

echo "[kd-train] base=${BASE_CKPT}"
if [ -n "$RESUME_CKPT" ]; then
  echo "[kd-train] resume=${RESUME_CKPT}"
fi
echo "[kd-train] teacher_cache=${TEACHER_OUTPUT_DIR} (${cache_count}/${train_count})"
echo "[kd-train] epochs=${EPOCHS}, lr=${LR}, batch=${TRAIN_BATCH_SIZE}, lambda=${KD_LAMBDA}, mode=${KD_MODE}, run_dir=${run_dir}"
if [ -n "$RESUME_CKPT" ]; then
  checkpoint_args=(
    training.train_checkpoint_path=null
    "training.resume_ckpt_path='${RESUME_CKPT}'"
  )
else
  checkpoint_args=(
    "training.train_checkpoint_path='${BASE_CKPT}'"
    training.resume_ckpt_path=null
  )
fi
CUDA_VISIBLE_DEVICES="$GPU_ID" conda run --no-capture-output -n bcctn python -u train.py \
  "${checkpoint_args[@]}" \
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

sanity_dir="${RESULT_ROOT}/all_lmul_m4_output_kd_l${KD_LAMBDA}_${KD_MODE}_w${LIMIT_PAIRS}"
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

echo "[kd-output-qat] done. Results under ${RESULT_ROOT}"
