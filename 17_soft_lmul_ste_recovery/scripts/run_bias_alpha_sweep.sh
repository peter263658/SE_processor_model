#!/bin/bash
# VCTK-only quick sweep for scaled post-hoc L-Mul bias correction.
#
# This tests whether the offline local correction δ is over-correcting a QAT
# checkpoint. It intentionally runs VCTK only (w=200 by default); promising
# signals must be confirmed with full VCTK w=750 + TIMIT.
#
# Required env:
#   CKPT                 student checkpoint to evaluate
#   BIAS_CORRECTION_PT   .pt file from calibrate_lmul_bias_correction.py
#
# Optional env:
#   GPU_ID               default 3
#   LIMIT                default 200
#   ALPHAS               default "0 0.05 0.1 0.25 0.5 1.0"
#   METRIC_WORKERS       default 16
set -euo pipefail

REPO=/raid/R12K41024/16_kd_heatmap_recovery
CKPT="${CKPT:?CKPT is required}"
BIAS_CORRECTION_PT="${BIAS_CORRECTION_PT:?BIAS_CORRECTION_PT is required}"
P2_INDEX="${P2_INDEX:-${REPO}/Checkpoints/p2_skip_indices_g42_topk_l1_r050.pt}"
GPU_ID="${GPU_ID:-3}"
LIMIT="${LIMIT:-200}"
METRIC_WORKERS="${METRIC_WORKERS:-16}"
ALPHAS="${ALPHAS:-0 0.05 0.1 0.25 0.5 1.0}"
OUT_ROOT="${OUT_ROOT:-${REPO}/results/bias_alpha_sweep}"

cd "$REPO"
export PYTHONPATH="${PYTHONPATH:-}:${REPO}"
export HYDRA_FULL_ERROR=1
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl_cache}"

if [ ! -f "$CKPT" ]; then
  echo "missing CKPT: $CKPT" >&2; exit 1
fi
if [ ! -f "$BIAS_CORRECTION_PT" ]; then
  echo "missing BIAS_CORRECTION_PT: $BIAS_CORRECTION_PT" >&2; exit 1
fi
if [ ! -f "$P2_INDEX" ]; then
  echo "missing P2_INDEX: $P2_INDEX" >&2; exit 1
fi

ckpt_tag="$(basename "$CKPT" .ckpt | tr '= ' '__' | tr -c 'A-Za-z0-9_.-' '_')"
run_dir="${OUT_ROOT}/${ckpt_tag}__w${LIMIT}"
mkdir -p "$run_dir" "$REPO/logs"

base_overrides=(
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
  model.low_power.all_lmul_multiply.bit_exact=true
  model.low_power.all_lmul_multiply.linear_chunk_size=2048
  model.low_power.all_lmul_multiply.checkpoint_chunks=false
)

echo "[bias-alpha] CKPT=$CKPT"
echo "[bias-alpha] BIAS_CORRECTION_PT=$BIAS_CORRECTION_PT"
echo "[bias-alpha] RUN_DIR=$run_dir"
echo "[bias-alpha] ALPHAS=$ALPHAS"

for alpha in $ALPHAS; do
  alpha_tag="$(printf '%s' "$alpha" | tr '.' 'p' | tr '-' 'm')"
  out="${run_dir}/alpha_${alpha_tag}"
  log="${REPO}/logs/bias_alpha_${alpha_tag}_${ckpt_tag}_w${LIMIT}_gpu${GPU_ID}.log"
  echo ""
  echo "==============================================="
  echo "[bias-alpha] alpha=$alpha  (VCTK-only quick probe, w=$LIMIT)"
  echo "  out: $out"
  echo "  log: $log"
  echo "==============================================="
  BCCTN_BIAS_CORRECTION_PATH="$BIAS_CORRECTION_PT" \
  BCCTN_BIAS_CORRECTION_ALPHA="$alpha" \
  CUDA_VISIBLE_DEVICES="$GPU_ID" \
  conda run --no-capture-output -n bcctn python eval.py \
    --model_checkpoint "$CKPT" \
    --vctk_test_dir /raid/R12K41024/BCCTN/Dataset/noisy_testset \
    --vctk_clean_dir /raid/R12K41024/BCCTN/Dataset/clean_testset \
    --output_dir "$out" \
    --paper_style_eval \
    --batch_size 1 \
    --limit_pairs "$LIMIT" \
    --metric_workers "$METRIC_WORKERS" \
    --hydra_overrides "${base_overrides[@]}" \
    2>&1 | tee "$log"
done

echo ""
echo "==============================================="
echo "[bias-alpha] summary"
echo "==============================================="
for d in "$run_dir"/alpha_*/; do
  tag="$(basename "$d")"
  f="$d/vctk_paper_style_results_averages.txt"
  if [ -f "$f" ]; then
    m="$(grep "Average MBSTOI Enhanced:" "$f" | awk '{print $NF}')"
    p="$(grep "Average PESQ Enhanced:" "$f" | awk '{print $NF}')"
    s="$(grep "Average SegSNR Improvement:" "$f" | awk '{print $NF}')"
    printf "  %-12s MBSTOI=%s  PESQ=%s  SegSNR_imp=%s\n" "$tag" "$m" "$p" "$s"
  fi
done
