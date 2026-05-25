#!/bin/bash
# Calibrate per-channel L-Mul bias-correction (Conv2d + ConvTranspose2d only by
# default) on a given student ckpt, then optionally bake δ into a copy of the
# ckpt.
#
# Required env:
#   CKPT             student checkpoint to calibrate against (e.g. v4 Ep5)
#   OUT_PT           where to save the {layer: tensor} dict + _meta
#
# Optional env:
#   GPU_ID           default 6
#   NUM_SAMPLES      default 128 (codex guardrail: 64-256 is enough)
#   NOISY_DIR        calibration noisy audio root; default trainset, not testset
#   SAMPLE_SEED      deterministic sample shuffle seed; default 42
#   BAKE_CKPT_OUT    if set, also write a baked ckpt with δ added into bias param
#   INCLUDE_LINEAR   set to 1 to also calibrate Linear (default off, conv-only per codex #3)
#
# Notes:
#   - δ = local L-Mul vs real-mult arithmetic offset (NOT teacher distill residual)
#   - the .pt file can be loaded at eval/train time via BCCTN_BIAS_CORRECTION_PATH
#   - the baked ckpt has δ folded into module.bias — DO NOT also enable
#     BCCTN_BIAS_CORRECTION_PATH on the baked ckpt (would double-apply)
set -euo pipefail

REPO=/raid/R12K41024/16_kd_heatmap_recovery
CKPT="${CKPT:?CKPT is required; set to student ckpt to calibrate}"
OUT_PT="${OUT_PT:?OUT_PT is required; .pt path for bias-correction dict}"
P2_INDEX="${P2_INDEX:-${REPO}/Checkpoints/p2_skip_indices_g42_topk_l1_r050.pt}"
GPU_ID="${GPU_ID:-6}"
NUM_SAMPLES="${NUM_SAMPLES:-128}"
NOISY_DIR="${NOISY_DIR:-/raid/R12K41024/BCCTN/Dataset/noisy_trainset}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"
BAKE_CKPT_OUT="${BAKE_CKPT_OUT:-}"
INCLUDE_LINEAR="${INCLUDE_LINEAR:-0}"

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
if [ ! -d "$NOISY_DIR" ]; then
  echo "missing NOISY_DIR: $NOISY_DIR" >&2; exit 1
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

extra_args=()
[ "$INCLUDE_LINEAR" = "1" ] && extra_args+=("--include_linear")
if [ -n "$BAKE_CKPT_OUT" ]; then
  extra_args+=("--bake_ckpt_out" "$BAKE_CKPT_OUT")
fi

mkdir -p "$(dirname "$OUT_PT")"
echo "[calib-wrapper] CKPT          = $CKPT"
echo "[calib-wrapper] OUT_PT        = $OUT_PT"
echo "[calib-wrapper] NUM_SAMPLES   = $NUM_SAMPLES"
echo "[calib-wrapper] NOISY_DIR     = $NOISY_DIR"
echo "[calib-wrapper] SAMPLE_SEED   = $SAMPLE_SEED"
echo "[calib-wrapper] INCLUDE_LINEAR= $INCLUDE_LINEAR"
echo "[calib-wrapper] BAKE_CKPT_OUT = ${BAKE_CKPT_OUT:-<none>}"
echo "[calib-wrapper] GPU_ID        = $GPU_ID"

CUDA_VISIBLE_DEVICES="$GPU_ID" conda run --no-capture-output -n bcctn python scripts/calibrate_lmul_bias_correction.py \
  --ckpt "$CKPT" \
  --noisy_dir "$NOISY_DIR" \
  --num_samples "$NUM_SAMPLES" \
  --sample_seed "$SAMPLE_SEED" \
  --output_pt "$OUT_PT" \
  "${extra_args[@]}" \
  --hydra_overrides "${overrides[@]}"

echo ""
echo "[calib-wrapper] done."
echo "Runtime apply (toggleable):"
echo "  BCCTN_BIAS_CORRECTION_PATH=$OUT_PT  <your eval/train command>"
if [ -n "$BAKE_CKPT_OUT" ]; then
  echo ""
  echo "Baked deploy (ckpt + delta folded into bias):"
  echo "  <eval/train command>  with model_checkpoint=$BAKE_CKPT_OUT"
  echo "  ⚠ DO NOT also set BCCTN_BIAS_CORRECTION_PATH on the baked ckpt."
fi
