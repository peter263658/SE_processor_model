#!/usr/bin/env bash
# Eval a (PESQ-finetuned or baseline) ckpt on VCTK+TIMIT with the SAME deploy
# W4A8 hardware stack as run_pesq_finetune.sh, reporting MBSTOI + PESQ.
# Mirrors run_strict_core_eval.sh but runs from THIS folder and takes CKPT/TAG.
#
#   CKPT=<ckpt> TAG=<name> GPU_ID=<free> bash run_pesq_eval.sh
set -euo pipefail

CODE_DIR=/raid/R12K41024/20_pesq_finetune/code
DEPS=/raid/R12K41024/18_w4a8_pow2_mixed_lmul
cd "$CODE_DIR"

# Default ckpt = deploy baseline (for the apples-to-apples PESQ/MBSTOI baseline).
CKPT="${CKPT:-${DEPS}/outputs/d1c3_pow2_finetune/d1c3_apotclip16_attnm8_outlierm4_15ep/ft/logs/lightning_logs/version_0/checkpoints/epoch=2-step=726.ckpt}"
P2_INDEX="${P2_INDEX:-/raid/R12K41024/14_low_power_arch_qat/Checkpoints/p2_skip_indices_g42_topk_l1_r050.pt}"
TAG="${TAG:-baseline}"
GPU_ID="${GPU_ID:-3}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LIMIT="${LIMIT:-750}"
METRIC_WORKERS="${METRIC_WORKERS:-32}"
CONDA_BIN="${CONDA_BIN:-/home/R12K41024/miniforge3/bin/conda}"

ATTN_M="${ATTN_M:-8}"; OUTLIER_M="${OUTLIER_M:-4}"; SCALE_FMT="${SCALE_FMT:-apot_clip}"
export BCCTN_APOT_A_MIN=-16; export BCCTN_APOT_A_MAX=-1
export BCCTN_APOT_B_MIN=-16; export BCCTN_APOT_B_MAX=-1
BN_MODE="${BN_MODE:-fold_apot}"; BN_MIN_EXP=-9; BN_MAX_EXP=4
PRELU_POW2="${PRELU_POW2:-true}"; PRELU_MIN_EXP=-8; PRELU_MAX_EXP=0

[ -f "$CKPT" ] || { echo "[pesq-eval] FATAL: missing $CKPT" >&2; exit 1; }

OUT="/raid/R12K41024/20_pesq_finetune/results/pesq_eval/${TAG}"
LOG="/raid/R12K41024/20_pesq_finetune/logs/pesq_eval/${TAG}_gpu${GPU_ID}.log"
mkdir -p "$OUT" "$(dirname "$LOG")"
export PYTHONPATH="${CODE_DIR}:${PYTHONPATH:-}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl_cache}"

overrides=(
  # eval never uses the loss; skip building PesqLoss (loss.* keys are filtered
  # out on load in eval.py regardless).
  model.pesq_weight=0
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
  model.low_power.bn_affine.tie_real_imag_scale=false
  model.low_power.prelu_pow2.enabled="$PRELU_POW2"
  model.low_power.prelu_pow2.min_exp="$PRELU_MIN_EXP"
  model.low_power.prelu_pow2.max_exp="$PRELU_MAX_EXP"
)

echo "[pesq-eval] TAG=$TAG ckpt=$CKPT gpu=$GPU_ID limit=$LIMIT"
CUDA_VISIBLE_DEVICES="$GPU_ID" "$CONDA_BIN" run --no-capture-output -n bcctn python eval.py \
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
echo "[pesq-eval] $TAG  VCTK/TIMIT MBSTOI + PESQ:"
for t in vctk timit; do
  f="${OUT}/${t}_paper_style_results_averages.txt"
  [ -f "$f" ] && { echo "  --- $t ---"; grep -iE "MBSTOI|PESQ" "$f"; }
done
