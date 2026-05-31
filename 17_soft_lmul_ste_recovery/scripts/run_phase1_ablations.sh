#!/bin/bash
# Phase 1: cheap config-level ablations (VCTK-only quick probe, w=200).
# Each variant overrides one (group of) config key(s) from the v3/v4 baseline.
# Output: per variant under  results/phase1_ablations/<CKPT_TAG>__w<LIMIT>/<tag>/
#
# This is a QUICK PROBE. Any promising signal must be re-confirmed with full
# VCTK w=750 + TIMIT eval (see FUTURE_SPEC eval protocol).
#
# Decision rule per FUTURE_SPEC:
#   >= +0.005 MBSTOI : candidate for QAT / longer training
#   +0.002 to +0.004 : keep as secondary, combine only after attribution
#   <= +0.001        : noise, drop unless strong hardware value
#
# Usage:
#   CKPT=/path/to/best.ckpt GPU_ID=3 bash scripts/run_phase1_ablations.sh
#   GPU_ID=3 ONLY="topk8 fp32" CKPT=... bash scripts/run_phase1_ablations.sh
#
# CKPT is REQUIRED — no default fallback to avoid silently testing the wrong model.
set -euo pipefail

REPO=/raid/R12K41024/16_kd_heatmap_recovery
CKPT="${CKPT:?CKPT is required; set CKPT=/path/to/v4_best.ckpt or other target ckpt}"
P2_INDEX="${P2_INDEX:-${REPO}/Checkpoints/p2_skip_indices_g42_topk_l1_r050.pt}"
OUT_ROOT="${OUT_ROOT:-${REPO}/results/phase1_ablations}"
GPU_ID="${GPU_ID:-3}"
LIMIT="${LIMIT:-200}"
METRIC_WORKERS="${METRIC_WORKERS:-16}"
ONLY="${ONLY:-}"          # space-separated tags to run; empty = all

# Tag the output directory with the ckpt identity + width so reruns from
# different ckpts don't overwrite each other.
CKPT_TAG=$(basename "$CKPT" .ckpt | tr '=.' '__')
RUN_DIR="${OUT_ROOT}/${CKPT_TAG}__w${LIMIT}"
mkdir -p "$RUN_DIR"
echo "[phase1] CKPT_TAG=$CKPT_TAG  RUN_DIR=$RUN_DIR"

cd "$REPO"
export PYTHONPATH="${PYTHONPATH:-}:${REPO}"
export HYDRA_FULL_ERROR=1
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl_cache}"

if [ ! -f "$CKPT" ]; then
  echo "missing CKPT: $CKPT" >&2; exit 1
fi
if [ ! -f "$P2_INDEX" ]; then
  echo "missing P2_INDEX: $P2_INDEX" >&2; exit 1
fi

# Baseline (v3/v4) overrides. Each ablation will start from this and patch.
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
  model.low_power.all_lmul_multiply.attention_mantissa_bits=4
  model.low_power.all_lmul_multiply.bit_exact=true
  model.low_power.all_lmul_multiply.linear_chunk_size=2048
  model.low_power.all_lmul_multiply.checkpoint_chunks=false
)

# Each ablation: tag → array of OVERRIDING keys (replace any base key with same prefix).
# We rely on Hydra "last override wins" semantics; just append the new key=value.

want() {
  local t="$1"
  [ -z "$ONLY" ] && return 0
  for w in $ONLY; do
    [ "$w" = "$t" ] && return 0
  done
  return 1
}

run_one() {
  local tag="$1"; shift
  local out="$RUN_DIR/$tag"
  echo ""
  echo "==============================================="
  echo "[phase1] $tag  (VCTK-only quick probe, w=$LIMIT)"
  echo "  ckpt    : $CKPT"
  echo "  out     : $out"
  echo "  overrides on top of baseline: $*"
  echo "==============================================="
  # VCTK-only by design — TIMIT is run separately at full w=750/160 only after
  # a promising signal here.
  CUDA_VISIBLE_DEVICES="$GPU_ID" conda run --no-capture-output -n bcctn python eval.py \
    --model_checkpoint "$CKPT" \
    --vctk_test_dir /raid/R12K41024/BCCTN/Dataset/noisy_testset \
    --vctk_clean_dir /raid/R12K41024/BCCTN/Dataset/clean_testset \
    --output_dir "$out" \
    --paper_style_eval \
    --batch_size 1 \
    --limit_pairs "$LIMIT" \
    --metric_workers "$METRIC_WORKERS" \
    --hydra_overrides "${base_overrides[@]}" "$@"
}

# 0. control: pure baseline (should reproduce v3 ~0.8240 at w=200, sanity check)
if want control; then
  run_one "control_baseline"
fi

# 1. top_k=8 packet
if want topk8; then
  run_one "topk8_packet" \
    training.top_k_attention=8 \
    model.attention_datapath.prob_packet.top_k=8
fi

# 2. packet renormalize
if want renorm; then
  run_one "packet_renormalize_true" \
    model.attention_datapath.prob_packet.renormalize=true
fi

# 3. packet probability 10-bit
if want prob10; then
  run_one "packet_prob_bits_10" \
    model.attention_datapath.prob_packet.prob_bits=10 \
    model.attention_datapath.prob_packet.prob_scale=1024 \
    model.attention_datapath.prob_packet.prob_max_code=1023
fi

# 4. fp32 accumulator diagnostic
if want fp32; then
  run_one "fp32_accumulator" \
    training.fp16_accumulator=false
fi

# 5. no-packet fair (per FUTURE_SPEC: must add score_v to lmul_targets, otherwise
#    score x V silently falls back to torch.matmul which isn't all-LMul)
if want no_packet; then
  run_one "no_packet_dense_lmul_qk_scorev" \
    model.attention_datapath.prob_packet.enabled=false \
    model.attention_datapath.lmul_targets=[qk,score_v]
fi

# 6. (deferred) sparse top-k no-q0.8 — per FUTURE_SPEC this needs a code path,
#    not pure hydra override. Placeholder so we don't forget.
if want sparse_no_q08; then
  echo ""
  echo "[phase1] sparse_no_q08 SKIPPED: needs code path (not pure config)."
  echo "  TODO: implement sparse top-k indices + non-quantized prob -> L-Mul score_v"
fi

echo ""
echo "==============================================="
echo "[phase1] all done. Results under $RUN_DIR/"
echo "==============================================="
for d in "$RUN_DIR"/*/; do
  tag=$(basename "$d")
  f="$d/vctk_paper_style_results_averages.txt"
  if [ -f "$f" ]; then
    m=$(grep "Average MBSTOI Enhanced:" "$f" | awk '{print $NF}')
    p=$(grep "Average PESQ Enhanced:" "$f" | awk '{print $NF}')
    s=$(grep "Average SegSNR Improvement:" "$f" | awk '{print $NF}')
    printf "  %-32s MBSTOI=%s  PESQ=%s  SegSNR_imp=%s\n" "$tag" "$m" "$p" "$s"
  fi
done
