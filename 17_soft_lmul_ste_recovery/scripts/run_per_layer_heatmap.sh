#!/bin/bash
# Phase 2: per-layer L-Mul *local operator error* heatmap.
# For each Conv2d/ConvTranspose2d/Linear: compare lmul output vs real-mult output
# on the SAME captured input → isolates per-layer error from upstream divergence.
#
# IMPORTANT COVERAGE NOTE (codex review #4):
# This heatmap covers Conv2d, ConvTranspose2d, and Linear modules. It does NOT
# cover MHA QK / Score×V L-Mul (those happen inside custom matmul functions in
# complexLayers.py, not module forward hooks). Call this a "conv/linear heatmap"
# rather than a "full datapath heatmap".
#
# After running this, top 5-10 "hot" layers should be verified with
# scripts/run_single_layer_swap.sh which sweeps each one's swap effect on MBSTOI.
#
# CKPT is REQUIRED — no default fallback.
set -euo pipefail

REPO=/raid/R12K41024/16_kd_heatmap_recovery
CKPT="${CKPT:?CKPT is required; set CKPT=/path/to/v4_best.ckpt or other target ckpt}"
P2_INDEX="${P2_INDEX:-${REPO}/Checkpoints/p2_skip_indices_g42_topk_l1_r050.pt}"
GPU_ID="${GPU_ID:-3}"
NUM_SAMPLES="${NUM_SAMPLES:-24}"

# Tag output CSV with ckpt identity so heatmaps from different ckpts don't collide.
CKPT_TAG=$(basename "$CKPT" .ckpt | tr '=.' '__')
OUT_CSV="${OUT_CSV:-${REPO}/results/heatmap/${CKPT_TAG}__N${NUM_SAMPLES}.csv}"
echo "[heatmap] CKPT_TAG=$CKPT_TAG  OUT_CSV=$OUT_CSV"

cd "$REPO"
export PYTHONPATH="${PYTHONPATH:-}:${REPO}"
export HYDRA_FULL_ERROR=1
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl_cache}"

if [ ! -f "$CKPT" ]; then
  echo "missing CKPT: $CKPT" >&2; exit 1
fi

mkdir -p "$(dirname "$OUT_CSV")"

# Same all-LMul m=4 setup as v3/v4 — so the heatmap measures the production
# all-LMul student's per-layer error vs reference real-mult.
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

CUDA_VISIBLE_DEVICES="$GPU_ID" conda run --no-capture-output -n bcctn python scripts/per_layer_error_heatmap.py \
  --ckpt "$CKPT" \
  --num_samples "$NUM_SAMPLES" \
  --output_csv "$OUT_CSV" \
  --hydra_overrides "${overrides[@]}"

echo ""
echo "[heatmap] CSV: $OUT_CSV"
echo "[heatmap] top 10 hot layers also printed above (stdout)."
