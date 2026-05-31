#!/bin/bash
# Phase 2 verification: single-layer swap eval (VCTK-only quick probe, w=200).
# For each layer in LAYERS, run a quick w=200 VCTK eval with that layer forced
# back to stock real-multiply (BCCTN_FORCE_REAL_LAYERS) while every other layer
# stays all-LMul m=4. Compare MBSTOI delta vs the all-LMul control.
#
# Promising results MUST be re-confirmed at full VCTK w=750 + TIMIT.
#
# Safety features (per codex review):
#   - CKPT is REQUIRED (no default fallback).
#   - HEATMAP_CSV defaults to a ckpt-tagged path; pass explicit HEATMAP_CSV
#     when re-using LAYERS from a different ckpt's heatmap.
#   - all_lmul_multiply.py raises ValueError if any layer name in
#     BCCTN_FORCE_REAL_LAYERS doesn't match a Conv2d/ConvTranspose2d/Linear
#     module, so a typo or stale CSV name fails loudly instead of silently
#     producing a "swap" eval identical to control.
#
# Usage:
#   CKPT=/path/to/best.ckpt LAYERS="decoder.model.0.0.conv_r ..." \
#     bash scripts/run_single_layer_swap.sh
#   # Or auto-pick top-N from heatmap CSV:
#   CKPT=... HEATMAP_CSV=/path/to/heatmap.csv TOP_N=5 \
#     bash scripts/run_single_layer_swap.sh
set -euo pipefail

REPO=/raid/R12K41024/16_kd_heatmap_recovery
CKPT="${CKPT:?CKPT is required; set CKPT=/path/to/v4_best.ckpt or other target ckpt}"
P2_INDEX="${P2_INDEX:-${REPO}/Checkpoints/p2_skip_indices_g42_topk_l1_r050.pt}"
# Default to ckpt-tagged heatmap CSV (matches run_per_layer_heatmap.sh naming).
CKPT_TAG=$(basename "$CKPT" .ckpt | tr '=.' '__')
HEATMAP_CSV="${HEATMAP_CSV:-${REPO}/results/heatmap/${CKPT_TAG}__N24.csv}"
OUT_ROOT="${OUT_ROOT:-${REPO}/results/single_layer_swap}"
GPU_ID="${GPU_ID:-3}"
LIMIT="${LIMIT:-200}"
METRIC_WORKERS="${METRIC_WORKERS:-16}"
TOP_N="${TOP_N:-5}"
LAYERS="${LAYERS:-}"

RUN_DIR="${OUT_ROOT}/${CKPT_TAG}__w${LIMIT}"
mkdir -p "$RUN_DIR"
echo "[swap] CKPT_TAG=$CKPT_TAG  RUN_DIR=$RUN_DIR  HEATMAP_CSV=$HEATMAP_CSV"

cd "$REPO"
export PYTHONPATH="${PYTHONPATH:-}:${REPO}"
export HYDRA_FULL_ERROR=1
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl_cache}"

if [ ! -f "$CKPT" ]; then
  echo "missing CKPT: $CKPT" >&2; exit 1
fi

if [ -z "$LAYERS" ]; then
  if [ ! -f "$HEATMAP_CSV" ]; then
    echo "LAYERS is empty AND no heatmap CSV at $HEATMAP_CSV" >&2
    echo "Run run_per_layer_heatmap.sh first, or pass LAYERS=..." >&2
    exit 1
  fi
  echo "[swap] reading top-${TOP_N} layers from $HEATMAP_CSV"
  # CSV columns: rank,name,type,rel_error,rel_error_dB,mse,ref_pwr,cumulative_pct_by_mse,out_shape,n_samples
  LAYERS=$(awk -F, 'NR>1 && NR<='$((TOP_N + 1))' {print $2}' "$HEATMAP_CSV" | tr '\n' ' ')
  echo "[swap] selected: $LAYERS"
fi

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

run_swap() {
  local tag="$1" layer_name="$2"
  local out="$RUN_DIR/$tag"
  echo ""
  echo "==============================================="
  echo "[swap] $tag: force-real layer = ${layer_name:-<none, pure control>}"
  echo "  out: $out"
  echo "==============================================="
  # all_lmul_multiply.py raises ValueError if layer_name doesn't match a
  # patchable module, so typos / stale CSV will fail loudly.
  BCCTN_FORCE_REAL_LAYERS="$layer_name" \
  CUDA_VISIBLE_DEVICES="$GPU_ID" conda run --no-capture-output -n bcctn python eval.py \
    --model_checkpoint "$CKPT" \
    --vctk_test_dir /raid/R12K41024/BCCTN/Dataset/noisy_testset \
    --vctk_clean_dir /raid/R12K41024/BCCTN/Dataset/clean_testset \
    --output_dir "$out" \
    --paper_style_eval \
    --batch_size 1 \
    --limit_pairs "$LIMIT" \
    --metric_workers "$METRIC_WORKERS" \
    --hydra_overrides "${overrides[@]}"
}

# 0. control (no force-real, pure all-LMul) — sanity
echo ""
echo "[swap] running control (no swap)..."
run_swap "control_no_swap" ""

# 1. per-layer swap
i=0
for layer in $LAYERS; do
  i=$((i + 1))
  # Sanitize tag: replace . with _
  tag="swap_$(printf 'L%02d_' "$i")$(echo "$layer" | tr '.' '_')"
  run_swap "$tag" "$layer"
done

echo ""
echo "==============================================="
echo "[swap] summary  (VCTK-only w=$LIMIT; full eval required for confirmation)"
echo "==============================================="
for d in "$RUN_DIR"/*/; do
  tag=$(basename "$d")
  f="$d/vctk_paper_style_results_averages.txt"
  if [ -f "$f" ]; then
    m=$(grep "Average MBSTOI Enhanced:" "$f" | awk '{print $NF}')
    p=$(grep "Average PESQ Enhanced:" "$f" | awk '{print $NF}')
    printf "  %-50s MBSTOI=%s  PESQ=%s\n" "$tag" "$m" "$p"
  fi
done
