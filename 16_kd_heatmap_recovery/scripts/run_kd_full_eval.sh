#!/usr/bin/env bash
# Full paper-style eval for a KD/all-LMul checkpoint.
#
# Usage:
#   CKPT=/path/to/best.ckpt GPU_ID=6 bash scripts/run_kd_full_eval.sh
#
# Notes:
#   - LIMIT is a cap: VCTK uses up to 750 pairs, TIMIT uses its available 160.
#   - This uses the production all-LMul student datapath: packet top-k=4,
#     QK L-Mul, Score x V packet mode, all conv/linear L-Mul m=4.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_DIR"

export PYTHONPATH="${PYTHONPATH:-}:${REPO_DIR}"
export HYDRA_FULL_ERROR=1
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl_cache}"

CKPT="${CKPT:?CKPT is required}"
GPU_ID="${GPU_ID:-6}"
LIMIT="${LIMIT:-750}"
BATCH_SIZE="${BATCH_SIZE:-1}"
METRIC_WORKERS="${METRIC_WORKERS:-32}"
P2_INDEX="${P2_INDEX:-${REPO_DIR}/Checkpoints/p2_skip_indices_g42_topk_l1_r050.pt}"
OUT_ROOT="${OUT_ROOT:-${REPO_DIR}/results/kd_output_qat_full}"
TAG="${TAG:-}"

if [ ! -f "$CKPT" ]; then
  echo "missing CKPT: $CKPT" >&2
  exit 1
fi
if [ ! -f "$P2_INDEX" ]; then
  echo "missing P2_INDEX: $P2_INDEX" >&2
  exit 1
fi

ckpt_tag="$(basename "$CKPT" .ckpt | tr '= ' '__' | tr -c 'A-Za-z0-9_.-' '_')"
if [ -n "$TAG" ]; then
  run_tag="${TAG}__${ckpt_tag}__w${LIMIT}"
else
  parent_tag="$(basename "$(dirname "$(dirname "$(dirname "$CKPT")")")" | tr '= ' '__' | tr -c 'A-Za-z0-9_.-' '_')"
  run_tag="${parent_tag}__${ckpt_tag}__w${LIMIT}"
fi
OUT="${OUT:-${OUT_ROOT}/${run_tag}}"
LOG="${LOG:-${REPO_DIR}/logs/kd_full_eval_${run_tag}_gpu${GPU_ID}.log}"

mkdir -p "$OUT_ROOT" "$(dirname "$LOG")"

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
  model.decoder_grouped_pointwise.skip_indices_path="$P2_INDEX"
  model.low_power.mha_projection.enabled=true
  model.low_power.mha_projection.qkv_groups=4
  model.low_power.all_lmul_multiply.enabled=true
  model.low_power.all_lmul_multiply.mantissa_bits=4
  model.low_power.all_lmul_multiply.bit_exact=true
  model.low_power.all_lmul_multiply.linear_chunk_size=2048
  model.low_power.all_lmul_multiply.checkpoint_chunks=false
)

echo "[kd-full-eval] ckpt=${CKPT}"
echo "[kd-full-eval] out=${OUT}"
echo "[kd-full-eval] log=${LOG}"
echo "[kd-full-eval] gpu=${GPU_ID}, limit=${LIMIT}, metric_workers=${METRIC_WORKERS}"

CUDA_VISIBLE_DEVICES="$GPU_ID" conda run --no-capture-output -n bcctn python eval.py \
  --model_checkpoint "$CKPT" \
  --vctk_test_dir /raid/R12K41024/BCCTN/Dataset/noisy_testset \
  --vctk_clean_dir /raid/R12K41024/BCCTN/Dataset/clean_testset \
  --timit_test_dir /raid/R12K41024/BCCTN/Dataset/noisy_testset_timit \
  --timit_clean_dir /raid/R12K41024/BCCTN/Dataset/clean_testset_timit \
  --output_dir "$OUT" \
  --paper_style_eval \
  --batch_size "$BATCH_SIZE" \
  --limit_pairs "$LIMIT" \
  --metric_workers "$METRIC_WORKERS" \
  --hydra_overrides "${overrides[@]}" \
  2>&1 | tee "$LOG"

echo "[kd-full-eval] done. Results under ${OUT}"
