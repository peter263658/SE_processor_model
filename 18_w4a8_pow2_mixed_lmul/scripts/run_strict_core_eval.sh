#!/usr/bin/env bash
# STRICT network-inference-core zero-multiplier EVAL-ONLY.
#
# Closes the multiplier-free claim over the WHOLE inference core by adding the
# two ops the B1b stack still left as real multiplies:
#   - BN    : fold to per-channel affine a=gamma/sqrt(running_var+eps), then
#             quantize the *effective* a to APoT (NOT gamma — see complexLayers).
#             y = a_q*x + b, b = beta - a_q*mean.  HW = shift(s)+add.
#   - PReLU : slope -> pow2 (negative half = 1 shift).  slope IS the multiply.
#
# Everything else is the bounded-APoT m8 KD best (VCTK 0.835059 / TIMIT 0.7892):
#   conv/proj  : INT4xFP8 shift-add proxy + APoT-clip[-16,-1] delayed per-group scale
#   QK / PV    : L-Mul m8
#   PW outlier : FP8xFP8 L-Mul m4 (12 layers)
#
# BN exponent range is DATA-DRIVEN from the ckpt (analyze_bn_fold_range):
# folded a in [0.0037, 14.8], log2 in [-8.07, +3.89], 34% of channels |a|>=1.
# => use [-9, 4]  (NOT the conv weight-scale [-16,-1]; that would crush a>=1).
# PReLU slopes in [0.006, 0.40] (all <1) => [-8, 0].
#
# eval-only, NO training. This answers: does the *fully* mul-free core still
# pass the gate zero-shot, or do we need a short BN/PReLU recovery FT?
#
#   GPU_ID=<free> bash scripts/run_strict_core_eval.sh
set -euo pipefail

REPO=/raid/R12K41024/18_w4a8_pow2_mixed_lmul
cd "$REPO"

# Bounded-APoT m8 KD best ckpt (VCTK 0.835059 / TIMIT 0.789213).
CKPT="${CKPT:-${REPO}/outputs/d1c3_pow2_finetune/d1c3_apotclip16_attnm8_outlierm4_15ep/ft/logs/lightning_logs/version_0/checkpoints/epoch=2-step=726.ckpt}"
P2_INDEX="${P2_INDEX:-/raid/R12K41024/14_low_power_arch_qat/Checkpoints/p2_skip_indices_g42_topk_l1_r050.pt}"
GPU_ID="${GPU_ID:-3}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LIMIT="${LIMIT:-750}"
METRIC_WORKERS="${METRIC_WORKERS:-32}"

ATTN_M="${ATTN_M:-8}"               # qk + score_v L-Mul mantissa
OUTLIER_M="${OUTLIER_M:-4}"         # decoder_pw_outlier FP8xFP8 L-Mul mantissa
SCALE_FMT="${SCALE_FMT:-apot_clip}" # conv int4 scale (bounded APoT)
APOT_A_MIN="${APOT_A_MIN:--16}"; APOT_A_MAX="${APOT_A_MAX:--1}"   # conv weight scale < 1
APOT_B_MIN="${APOT_B_MIN:--16}"; APOT_B_MAX="${APOT_B_MAX:--1}"

# strict-core knobs
BN_MODE="${BN_MODE:-fold_apot}"     # fold_apot | fold_pow2 (fold_* = mul-free); none = real mult
BN_MIN_EXP="${BN_MIN_EXP:--9}"      # data-driven from ckpt (a in [-8.07,+3.89])
BN_MAX_EXP="${BN_MAX_EXP:-4}"
BN_TIE="${BN_TIE:-false}"
PRELU_POW2="${PRELU_POW2:-true}"
PRELU_MIN_EXP="${PRELU_MIN_EXP:--8}"
PRELU_MAX_EXP="${PRELU_MAX_EXP:-0}"

for f in "$CKPT" "$P2_INDEX"; do
  [ -f "$f" ] || { echo "[strict-core] FATAL: missing $f" >&2; exit 1; }
done

TAG="${TAG:-strict_core_bnfold_prelu_pow2_eval}"
OUT="${OUT:-${REPO}/results/mixed_lmul_eval/${TAG}}"
LOG="${LOG:-${REPO}/logs/mixed_lmul_eval/${TAG}_gpu${GPU_ID}.log}"
mkdir -p "$OUT" "$(dirname "$LOG")"
export PYTHONPATH="${PYTHONPATH:-}:${REPO}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl_cache}"
export BCCTN_APOT_A_MIN="$APOT_A_MIN"; export BCCTN_APOT_A_MAX="$APOT_A_MAX"
export BCCTN_APOT_B_MIN="$APOT_B_MIN"; export BCCTN_APOT_B_MAX="$APOT_B_MAX"

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
  # --- strict-core: make BN + PReLU multiplier-free ---
  model.low_power.bn_affine.scale_mode="$BN_MODE"
  model.low_power.bn_affine.pow2_min_exp="$BN_MIN_EXP"
  model.low_power.bn_affine.pow2_max_exp="$BN_MAX_EXP"
  model.low_power.bn_affine.tie_real_imag_scale="$BN_TIE"
  model.low_power.prelu_pow2.enabled="$PRELU_POW2"
  model.low_power.prelu_pow2.min_exp="$PRELU_MIN_EXP"
  model.low_power.prelu_pow2.max_exp="$PRELU_MAX_EXP"
)

echo "[strict-core] ckpt=$CKPT"
echo "[strict-core] conv=$SCALE_FMT a/b=[$APOT_A_MIN,$APOT_A_MAX]  attn_m=$ATTN_M  outlier_m=$OUTLIER_M"
echo "[strict-core] BN=$BN_MODE exp[$BN_MIN_EXP,$BN_MAX_EXP] tie=$BN_TIE   PReLU pow2=$PRELU_POW2 exp[$PRELU_MIN_EXP,$PRELU_MAX_EXP]"
echo "[strict-core] out=$OUT gpu=$GPU_ID limit=$LIMIT"
echo "[strict-core] gate VCTK 0.835 / TIMIT 0.7880; bounded-APoT m8 (BN/PReLU still mult) = 0.835059 / 0.789213"
case "$BN_MODE" in
  fold_apot|fold_pow2) echo "[strict-core] BN fold = multiplier-free (shift(s)+add)" ;;
  none|off) echo "[strict-core] WARN: BN_MODE=$BN_MODE -> BN is a REAL multiply (NOT strict-core)" ;;
esac

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

echo ""
echo "[strict-core] done. VCTK/TIMIT MBSTOI:"
for t in vctk timit; do
  f="${OUT}/${t}_paper_style_results_averages.txt"
  [ -f "$f" ] && echo "  ${t}: $(grep -i 'MBSTOI Enhanced' "$f")"
done
echo "[strict-core] >=0.835 VCTK => strict-core closed zero-shot; else short FT 3-5ep from bounded best."
