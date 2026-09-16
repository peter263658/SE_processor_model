#!/usr/bin/env bash
# APoT-descale EVAL — runs ENTIRELY inside the new handoff package, on the TRUE
# §3.4 hardware path (delayed post-acc APoT descale, BCCTN_OUTLIER_DELAYED_DESCALE=1).
# Use this (NOT code/run_strict_core_eval.sh, which cd's to the old repo and does
# NOT set the delayed-descale flag) to score APoT-descale finetune ckpts.
#
#   CKPT=<ckpt> GPU_ID=<free> [VCTK_ONLY=1] bash run_apot_descale_eval.sh
set -euo pipefail

PKG=/raid/R12K41024/18_w4a8_pow2_mixed_lmul_hw_handoff_20260530
cd "$PKG/code"

CKPT="${CKPT:?set CKPT=<path to ckpt>}"
P2_INDEX="${P2_INDEX:-${PKG}/ckpt/p2_skip_indices_g42_topk_l1_r050.pt}"
GPU_ID="${GPU_ID:-1}"; BATCH_SIZE="${BATCH_SIZE:-8}"; LIMIT="${LIMIT:-750}"
METRIC_WORKERS="${METRIC_WORKERS:-32}"; VCTK_ONLY="${VCTK_ONLY:-1}"

ATTN_M="${ATTN_M:-8}"; OUTLIER_M="${OUTLIER_M:-4}"; SCALE_FMT="${SCALE_FMT:-apot_clip}"
ACT_FP8_MODE="${ACT_FP8_MODE:-${BCCTN_ACTIVATION_FP8_MODE:-scaled}}"
APOT_A_MIN="${APOT_A_MIN:--16}"; APOT_A_MAX="${APOT_A_MAX:--1}"
APOT_B_MIN="${APOT_B_MIN:--16}"; APOT_B_MAX="${APOT_B_MAX:--1}"
# delayed APoT descale = the TRUE deployed path (must match finetune/export)
DELAYED_DESCALE="${DELAYED_DESCALE:-1}"; DD_MIN="${DD_MIN:--16}"; DD_MAX="${DD_MAX:--6}"
BN_MODE="${BN_MODE:-fold_apot}"; BN_MIN_EXP="${BN_MIN_EXP:--9}"; BN_MAX_EXP="${BN_MAX_EXP:-4}"; BN_TIE="${BN_TIE:-false}"
PRELU_POW2="${PRELU_POW2:-true}"; PRELU_MIN_EXP="${PRELU_MIN_EXP:--8}"; PRELU_MAX_EXP="${PRELU_MAX_EXP:-0}"

[ -f "$CKPT" ] || { echo "[apot-eval] FATAL: missing CKPT $CKPT" >&2; exit 1; }

TAG="${TAG:-$(basename "$CKPT" .ckpt)}"
OUT="${OUT:-${PKG}/results/apot_descale_eval/ft_${TAG}}"
LOG="${LOG:-${OUT}/eval.log}"
mkdir -p "$OUT"
export PYTHONPATH="${PKG}/code"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl_cache}"
export BCCTN_APOT_A_MIN="$APOT_A_MIN"; export BCCTN_APOT_A_MAX="$APOT_A_MAX"
export BCCTN_APOT_B_MIN="$APOT_B_MIN"; export BCCTN_APOT_B_MAX="$APOT_B_MAX"
export BCCTN_ACTIVATION_FP8_MODE="$ACT_FP8_MODE"
export BCCTN_OUTLIER_DELAYED_DESCALE="$DELAYED_DESCALE"
export BCCTN_OUTLIER_DESCALE_APOT_MIN="$DD_MIN"; export BCCTN_OUTLIER_DESCALE_APOT_MAX="$DD_MAX"

echo "[apot-eval] ckpt=$CKPT"
echo "[apot-eval] activation_fp8_mode=$ACT_FP8_MODE"
echo "[apot-eval] DELAYED descale=$DELAYED_DESCALE exp[$DD_MIN,$DD_MAX]  VCTK_ONLY=$VCTK_ONLY gpu=$GPU_ID out=$OUT"

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

# Override to score a different acoustic condition (e.g. the directional-
# interferer set in 22_directional_interferer) on the same clean targets.
VCTK_TEST_DIR="${VCTK_TEST_DIR:-/raid/R12K41024/BCCTN/Dataset/noisy_testset}"
VCTK_CLEAN_DIR="${VCTK_CLEAN_DIR:-/raid/R12K41024/BCCTN/Dataset/clean_testset}"

timit_args=()
if [ "$VCTK_ONLY" != "1" ]; then
  timit_args=( --timit_test_dir /raid/R12K41024/BCCTN/Dataset/noisy_testset_timit
               --timit_clean_dir /raid/R12K41024/BCCTN/Dataset/clean_testset_timit )
fi

CUDA_VISIBLE_DEVICES="$GPU_ID" conda run --no-capture-output -n bcctn python eval.py \
  --model_checkpoint "$CKPT" \
  --vctk_test_dir "$VCTK_TEST_DIR" \
  --vctk_clean_dir "$VCTK_CLEAN_DIR" \
  "${timit_args[@]}" \
  --output_dir "$OUT" \
  --paper_style_eval --batch_size "$BATCH_SIZE" --limit_pairs "$LIMIT" \
  --metric_workers "$METRIC_WORKERS" \
  --hydra_overrides "${overrides[@]}" \
  2>&1 | tee "$LOG"

echo ""; echo "[apot-eval] VCTK:"
grep -i 'MBSTOI Enhanced' "${OUT}/vctk_paper_style_results_averages.txt" 2>/dev/null || true
