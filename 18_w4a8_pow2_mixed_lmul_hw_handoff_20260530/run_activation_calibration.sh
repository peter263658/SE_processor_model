#!/usr/bin/env bash
# Offline activation calibration for method B:
#   collect fixed per-boundary activation scales, then use calibrated_e4m3.
# Hardware never computes running amax; the emitted JSON is a static constant
# table consumed by finetune/eval/export/TB generation.
set -euo pipefail

PKG=/raid/R12K41024/18_w4a8_pow2_mixed_lmul_hw_handoff_20260530
cd "$PKG/code"

CKPT="${CKPT:-${PKG}/ckpt/strict_core_apot_descale_deploy_ep2_vctk0.8365_timit0.7912.ckpt}"
P2_INDEX="${P2_INDEX:-${PKG}/ckpt/p2_skip_indices_g42_topk_l1_r050.pt}"
CALIB_BATCHES="${CALIB_BATCHES:-64}"
CALIB_QUANTILE="${CALIB_QUANTILE:-0.999}"
CALIB_SPLIT="${CALIB_SPLIT:-train}"
GPU_ID="${GPU_ID:-6}"
BATCH_SIZE="${BATCH_SIZE:-64}"
RUN_TAG="${RUN_TAG:-calibrated_e4m3_p999_b${CALIB_BATCHES}_$(date +%Y%m%d_%H%M%S)}"
OUT="${OUT:-${PKG}/outputs/activation_calibration/${RUN_TAG}/activation_calibration.json}"
LOG="${LOG:-${PKG}/logs/activation_calibration/${RUN_TAG}.log}"

ATTN_M="${ATTN_M:-8}"; OUTLIER_M="${OUTLIER_M:-4}"; SCALE_FMT="${SCALE_FMT:-apot_clip}"
APOT_A_MIN="${APOT_A_MIN:--16}"; APOT_A_MAX="${APOT_A_MAX:--1}"
APOT_B_MIN="${APOT_B_MIN:--16}"; APOT_B_MAX="${APOT_B_MAX:--1}"
DELAYED_DESCALE="${DELAYED_DESCALE:-1}"; DD_MIN="${DD_MIN:--16}"; DD_MAX="${DD_MAX:--6}"
BN_MODE="${BN_MODE:-fold_apot}"; BN_MIN_EXP="${BN_MIN_EXP:--9}"; BN_MAX_EXP="${BN_MAX_EXP:-4}"; BN_TIE="${BN_TIE:-false}"
PRELU_POW2="${PRELU_POW2:-true}"; PRELU_MIN_EXP="${PRELU_MIN_EXP:--8}"; PRELU_MAX_EXP="${PRELU_MAX_EXP:-0}"

for f in "$CKPT" "$P2_INDEX"; do
  [ -e "$f" ] || { echo "[act-calib] FATAL: missing $f" >&2; exit 1; }
done

mkdir -p "$(dirname "$OUT")" "$(dirname "$LOG")"
export PYTHONPATH="${PKG}/code"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl_cache}"
export BCCTN_APOT_A_MIN="$APOT_A_MIN"; export BCCTN_APOT_A_MAX="$APOT_A_MAX"
export BCCTN_APOT_B_MIN="$APOT_B_MIN"; export BCCTN_APOT_B_MAX="$APOT_B_MAX"
export BCCTN_OUTLIER_DELAYED_DESCALE="$DELAYED_DESCALE"
export BCCTN_OUTLIER_DESCALE_APOT_MIN="$DD_MIN"; export BCCTN_OUTLIER_DESCALE_APOT_MAX="$DD_MAX"

echo "[act-calib] ckpt=$CKPT"
echo "[act-calib] out=$OUT"
echo "[act-calib] split=$CALIB_SPLIT batches=$CALIB_BATCHES q=$CALIB_QUANTILE batch_size=$BATCH_SIZE gpu=$GPU_ID"

overrides=(
  training.qat_scope=full
  training.top_k_attention=4
  training.fp16_accumulator=true
  training.int4_scale_format="$SCALE_FMT"
  training.batch_size="$BATCH_SIZE"
  model.attention_softmax.mode=shift_topk_pow2_clamp8_nomantissa
  model.attention_datapath.enabled=true
  model.attention_datapath.score_quant=none
  model.attention_datapath.prob_quant=none
  model.attention_datapath.lmul_enabled=true
  model.attention_datapath.lmul_targets=[qk]
  model.attention_datapath.lmul_bit_exact=true
  model.attention_datapath.lmul_mantissa_bits="$ATTN_M"
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
  model.low_power.all_lmul_multiply.enabled=false
  model.low_power.outlier_fp8_lmul.enabled=true
  model.low_power.outlier_fp8_lmul.mantissa_bits="$OUTLIER_M"
  model.low_power.bn_affine.scale_mode="$BN_MODE"
  model.low_power.bn_affine.pow2_min_exp="$BN_MIN_EXP"
  model.low_power.bn_affine.pow2_max_exp="$BN_MAX_EXP"
  model.low_power.bn_affine.tie_real_imag_scale="$BN_TIE"
  model.low_power.prelu_pow2.enabled="$PRELU_POW2"
  model.low_power.prelu_pow2.min_exp="$PRELU_MIN_EXP"
  model.low_power.prelu_pow2.max_exp="$PRELU_MAX_EXP"
  model.kd.enabled=false
)

CUDA_VISIBLE_DEVICES="$GPU_ID" conda run --no-capture-output -n bcctn \
  python -u calibrate_activation_fp8.py \
    --ckpt "$CKPT" \
    --out "$OUT" \
    --split "$CALIB_SPLIT" \
    --batches "$CALIB_BATCHES" \
    --quantile "$CALIB_QUANTILE" \
    --hydra_overrides "${overrides[@]}" \
    2>&1 | tee "$LOG"

echo "[act-calib] done: $OUT"
