#!/bin/bash
# Run the KD loss-scale probe. Required env:
#   CKPT          student checkpoint to probe against (e.g. KD smoke Ep1, or v4 Ep5)
# Optional env:
#   GPU_ID        default 3
#   NUM_BATCHES   default 20
#   BATCH_SIZE    default 8
#   TEACHER_DIR   default precomputed d1+c3 fp16
#   OUT_JSON      default results/loss_scale_probe/<ckpt_tag>.json
#
# Defaults the architecture overrides to v3/v4/KD-smoke setup (all-LMul m=4,
# packet, grouped MHA, depthwise sep). Pass HYDRA_OVERRIDES env to replace.
set -euo pipefail

REPO=/raid/R12K41024/16_kd_heatmap_recovery
CKPT="${CKPT:?CKPT is required; set to KD smoke best or v4 Ep5 ckpt}"
P2_INDEX="${P2_INDEX:-${REPO}/Checkpoints/p2_skip_indices_g42_topk_l1_r050.pt}"
TEACHER_DIR="${TEACHER_DIR:-${REPO}/teacher_outputs/d1c3_epoch25_step100672_fp16}"
NOISY_DIR="${NOISY_DIR:-/raid/R12K41024/BCCTN/Dataset/noisy_trainset}"
CLEAN_DIR="${CLEAN_DIR:-/raid/R12K41024/BCCTN/Dataset/clean_trainset}"
GPU_ID="${GPU_ID:-3}"
NUM_BATCHES="${NUM_BATCHES:-20}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-2}"
SEED="${SEED:-12345}"

CKPT_TAG=$(basename "$CKPT" .ckpt | tr '=.' '__')
OUT_JSON="${OUT_JSON:-${REPO}/results/loss_scale_probe/${CKPT_TAG}.json}"

cd "$REPO"
export PYTHONPATH="${PYTHONPATH:-}:${REPO}"
export HYDRA_FULL_ERROR=1
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl_cache}"

if [ ! -f "$CKPT" ] || [ ! -d "$TEACHER_DIR" ] || [ ! -d "$NOISY_DIR" ] || [ ! -d "$CLEAN_DIR" ]; then
  echo "missing input(s)" >&2
  echo "  CKPT=$CKPT"; echo "  TEACHER_DIR=$TEACHER_DIR"
  echo "  NOISY_DIR=$NOISY_DIR"; echo "  CLEAN_DIR=$CLEAN_DIR"
  exit 1
fi

# Student architecture overrides — matches v3/v4/KD smoke setup
overrides=(
  training.qat_scope=full
  training.top_k_attention=4
  training.fp16_accumulator=true
  training.int4_scale_format=fp32
  model.attention_softmax.mode=shift_topk_pow2_clamp8_nomantissa
  model.attention_datapath.enabled=true
  model.attention_datapath.score_quant=none
  model.attention_datapath.prob_quant=none
  model.attention_datapath.lmul_enabled=true
  model.attention_datapath.lmul_targets=[qk]
  model.attention_datapath.lmul_mantissa_bits=4
  model.attention_datapath.lmul_bit_exact=true
  model.attention_datapath.lmul_all_projection_ops=true
  model.attention_datapath.lmul_linear_chunk_size=2048
  model.attention_datapath.lmul_checkpoint_chunks=false
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
  "model.decoder_grouped_pointwise.skip_indices_path=$P2_INDEX"
  model.low_power.mha_projection.enabled=true
  model.low_power.mha_projection.qkv_groups=4
  model.low_power.all_lmul_multiply.enabled=true
  model.low_power.all_lmul_multiply.mantissa_bits=4
  model.low_power.all_lmul_multiply.attention_mantissa_bits=4
  model.low_power.all_lmul_multiply.bit_exact=true
  model.low_power.all_lmul_multiply.linear_chunk_size=2048
  model.low_power.all_lmul_multiply.checkpoint_chunks=false
)

mkdir -p "$(dirname "$OUT_JSON")"

echo "[probe-wrapper] CKPT       = $CKPT"
echo "[probe-wrapper] TEACHER_DIR= $TEACHER_DIR"
echo "[probe-wrapper] OUT_JSON   = $OUT_JSON"
echo "[probe-wrapper] GPU_ID     = $GPU_ID"
echo "[probe-wrapper] NUM_BATCHES=$NUM_BATCHES  BATCH_SIZE=$BATCH_SIZE  SEED=$SEED"

CUDA_VISIBLE_DEVICES="$GPU_ID" conda run --no-capture-output -n bcctn python scripts/probe_kd_loss_scale.py \
  --ckpt "$CKPT" \
  --teacher_dir "$TEACHER_DIR" \
  --noisy_dir "$NOISY_DIR" \
  --clean_dir "$CLEAN_DIR" \
  --num_batches "$NUM_BATCHES" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --seed "$SEED" \
  --output_json "$OUT_JSON" \
  --hydra_overrides "${overrides[@]}"

echo ""
echo "[probe-wrapper] done. JSON: $OUT_JSON"
