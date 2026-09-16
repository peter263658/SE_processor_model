#!/usr/bin/env bash
# Single-SNR-dir eval of the deploy model on the synthetic -8/-10 dB test
# points (20_pesq_finetune/data/lowsnr_ext). Same strict-core deploy overrides
# as run_apot_descale_eval.sh, but uses eval.py's single-dir mode
# (--test_dataset_path) because the extra SNR points live outside the
# paper_style snr_* loop.
#
#   TEST_DIR=<noisy dir> CLEAN_DIR=<clean dir> OUT=<output dir> GPU_ID=<id> \
#     bash run_lowsnr_ext_eval.sh
set -euo pipefail

PKG=/raid/R12K41024/18_w4a8_pow2_mixed_lmul_hw_handoff_20260530
cd "$PKG/code"

CKPT="${CKPT:-$PKG/ckpt/strict_core_pow2_blockfloat_deploy_ep3_vctk0.8357_timit0.7898.ckpt}"
P2_INDEX="${P2_INDEX:-${PKG}/ckpt/p2_skip_indices_g42_topk_l1_r050.pt}"
TEST_DIR="${TEST_DIR:?set TEST_DIR=<noisy snr dir>}"
CLEAN_DIR="${CLEAN_DIR:?set CLEAN_DIR=<clean dir>}"
OUT="${OUT:?set OUT=<output dir>}"
GPU_ID="${GPU_ID:-6}"; BATCH_SIZE="${BATCH_SIZE:-8}"; LIMIT="${LIMIT:-750}"
METRIC_WORKERS="${METRIC_WORKERS:-32}"

ATTN_M="${ATTN_M:-8}"; OUTLIER_M="${OUTLIER_M:-4}"; SCALE_FMT="${SCALE_FMT:-apot_clip}"
ACT_FP8_MODE="${ACT_FP8_MODE:-calibrated_pow2_e4m3}"
APOT_A_MIN="${APOT_A_MIN:--16}"; APOT_A_MAX="${APOT_A_MAX:--1}"
APOT_B_MIN="${APOT_B_MIN:--16}"; APOT_B_MAX="${APOT_B_MAX:--1}"
DELAYED_DESCALE="${DELAYED_DESCALE:-1}"; DD_MIN="${DD_MIN:--16}"; DD_MAX="${DD_MAX:--6}"
BN_MODE="${BN_MODE:-fold_apot}"; BN_MIN_EXP="${BN_MIN_EXP:--9}"; BN_MAX_EXP="${BN_MAX_EXP:-4}"; BN_TIE="${BN_TIE:-false}"
PRELU_POW2="${PRELU_POW2:-true}"; PRELU_MIN_EXP="${PRELU_MIN_EXP:--8}"; PRELU_MAX_EXP="${PRELU_MAX_EXP:-0}"
BCCTN_ACTIVATION_CALIB_PATH="${BCCTN_ACTIVATION_CALIB_PATH:-${PKG}/ckpt/activation_calibration_p999_b64.json}"

mkdir -p "$OUT"
export PYTHONPATH="${PKG}/code"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl_cache}"
export BCCTN_APOT_A_MIN="$APOT_A_MIN"; export BCCTN_APOT_A_MAX="$APOT_A_MAX"
export BCCTN_APOT_B_MIN="$APOT_B_MIN"; export BCCTN_APOT_B_MAX="$APOT_B_MAX"
export BCCTN_ACTIVATION_FP8_MODE="$ACT_FP8_MODE"
export BCCTN_ACTIVATION_CALIB_PATH
export BCCTN_OUTLIER_DELAYED_DESCALE="$DELAYED_DESCALE"
export BCCTN_OUTLIER_DESCALE_APOT_MIN="$DD_MIN"; export BCCTN_OUTLIER_DESCALE_APOT_MAX="$DD_MAX"

overrides=(
  training.qat_scope=full
  training.top_k_attention=4
  training.fp16_accumulator=true
  training.int4_scale_format="$SCALE_FMT"
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
  model.low_power.mha_projection.enabled=true
  model.low_power.mha_projection.qkv_groups=4
  model.decoder_grouped_pointwise.mode=grouped
  model.decoder_grouped_pointwise.groups_x=4
  model.decoder_grouped_pointwise.groups_skip=2
  model.decoder_grouped_pointwise.skip_select_ratio=0.5
  model.decoder_grouped_pointwise.skip_select_mode=topk_l1
  "model.decoder_grouped_pointwise.skip_indices_path=$P2_INDEX"
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
)

CUDA_VISIBLE_DEVICES="$GPU_ID" conda run --no-capture-output -n bcctn python eval.py \
  --model_checkpoint "$CKPT" \
  --test_dataset_path "$TEST_DIR" \
  --clean_dataset_path "$CLEAN_DIR" \
  --output_dir "$OUT" \
  --batch_size "$BATCH_SIZE" --limit_pairs "$LIMIT" \
  --metric_workers "$METRIC_WORKERS" \
  --hydra_overrides "${overrides[@]}" \
  2>&1 | tee "$OUT/eval.log"

echo ""; echo "[lowsnr-eval] done: $OUT"
grep -E "MBSTOI|STOI|PESQ" "$OUT/average_results.txt" 2>/dev/null || true
