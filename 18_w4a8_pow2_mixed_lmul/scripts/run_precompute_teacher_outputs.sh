#!/bin/bash
# Pre-compute the d1+c3 teacher's enhanced outputs on the training set so KD
# training does NOT need a live teacher forward (saves ~50% wall-clock).
#
# IMPORTANT: this script writes outputs but does NOT auto-launch a full run.
# Default LIMIT=32 (smoke test). To run full set, pass LIMIT=0 explicitly.
#
# Required env vars (no defaults — fails fast if unset):
#   TEACHER_CKPT  path to teacher .ckpt (production d1+c3 best, see FUTURE_SPEC)
#   OUT_DIR       where to write <basename>.pt files
#
# Optional env vars:
#   GPU_ID        default 3
#   LIMIT         default 32 (smoke test). Set to 0 for full ~30k-file run (~3h).
#   DTYPE         default fp16 (saves ~50% disk vs fp32)
#   NOISY_DIR     default /raid/R12K41024/BCCTN/Dataset/noisy_trainset
#   CLEAN_DIR     default /raid/R12K41024/BCCTN/Dataset/clean_trainset (meta only)
#
# Recommended FIRST RUN:
#   TEACHER_CKPT=/raid/R12K41024/14_low_power_arch_qat/.../epoch=25-step=100672.ckpt \
#   OUT_DIR=/raid/R12K41024/16_kd_heatmap_recovery/teacher_outputs/d1c3 \
#   bash scripts/run_precompute_teacher_outputs.sh
#   # → processes 32 samples, writes _meta.json, prints first sample's shape
#   # Verify: ls OUT_DIR has 32 .pt + 1 _meta.json; shapes look right.
#
# THEN full run:
#   TEACHER_CKPT=... OUT_DIR=... LIMIT=0 bash scripts/run_precompute_teacher_outputs.sh
set -euo pipefail

REPO=/raid/R12K41024/16_kd_heatmap_recovery
TEACHER_CKPT="${TEACHER_CKPT:?TEACHER_CKPT is required (e.g. d1+c3 epoch=25-step=100672.ckpt)}"
OUT_DIR="${OUT_DIR:?OUT_DIR is required (e.g. teacher_outputs/d1c3)}"
P2_INDEX="${P2_INDEX:-${REPO}/Checkpoints/p2_skip_indices_g42_topk_l1_r050.pt}"
NOISY_DIR="${NOISY_DIR:-/raid/R12K41024/BCCTN/Dataset/noisy_trainset}"
CLEAN_DIR="${CLEAN_DIR:-/raid/R12K41024/BCCTN/Dataset/clean_trainset}"
GPU_ID="${GPU_ID:-3}"
LIMIT="${LIMIT:-32}"          # smoke test by default
DTYPE="${DTYPE:-fp16}"

cd "$REPO"
export PYTHONPATH="${PYTHONPATH:-}:${REPO}"
export HYDRA_FULL_ERROR=1
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl_cache}"

if [ ! -f "$TEACHER_CKPT" ]; then
  echo "TEACHER_CKPT does not exist: $TEACHER_CKPT" >&2; exit 1
fi
if [ ! -f "$P2_INDEX" ]; then
  echo "P2_INDEX does not exist: $P2_INDEX" >&2; exit 1
fi
if [ ! -d "$NOISY_DIR" ]; then
  echo "NOISY_DIR does not exist: $NOISY_DIR" >&2; exit 1
fi

echo "[precompute-wrapper] TEACHER_CKPT  = $TEACHER_CKPT"
echo "[precompute-wrapper] OUT_DIR       = $OUT_DIR"
echo "[precompute-wrapper] NOISY_DIR     = $NOISY_DIR"
echo "[precompute-wrapper] CLEAN_DIR     = $CLEAN_DIR (metadata only)"
echo "[precompute-wrapper] GPU_ID        = $GPU_ID"
echo "[precompute-wrapper] LIMIT         = $LIMIT  (0 = all, 32 = smoke test)"
echo "[precompute-wrapper] DTYPE         = $DTYPE"

# Teacher = d1_qkv_g4 + c3_packet_lmul_bitexact (production gate, 0.8354/0.7880).
# Architecture: real-mult conv, packet+L-Mul attention only.
# CRITICAL: lmul_all_projection_ops=false and all_lmul_multiply.enabled=false,
# otherwise we'd be loading a teacher into an all-LMul student-style graph.
teacher_overrides=(
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
  model.attention_datapath.lmul_all_projection_ops=false       # KEY: NOT all proj (c3 only on qk)
  model.attention_datapath.lmul_linear_chunk_size=0
  model.attention_datapath.lmul_checkpoint_chunks=false
  model.attention_datapath.prob_packet.enabled=true            # packet ON (c3)
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
  model.low_power.all_lmul_multiply.enabled=false              # KEY: conv is real-mult
)

CUDA_VISIBLE_DEVICES="$GPU_ID" conda run --no-capture-output -n bcctn python scripts/precompute_teacher_outputs.py \
  --teacher_ckpt "$TEACHER_CKPT" \
  --noisy_dir "$NOISY_DIR" \
  --clean_dir "$CLEAN_DIR" \
  --output_dir "$OUT_DIR" \
  --dtype "$DTYPE" \
  --limit "$LIMIT" \
  --hydra_overrides "${teacher_overrides[@]}"

echo ""
echo "[precompute-wrapper] done."
if [ "$LIMIT" -gt 0 ]; then
  echo "  Smoke test only (LIMIT=$LIMIT)."
  echo "  Verify: ls -la $OUT_DIR | head ; cat $OUT_DIR/_meta.json"
  echo "  To run the full set: LIMIT=0 TEACHER_CKPT=... OUT_DIR=... bash $0"
fi
