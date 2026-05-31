#!/usr/bin/env bash
# APoT-descale recovery finetune — runs ENTIRELY inside the new handoff package.
# Old folder (/raid/R12K41024/18_w4a8_pow2_mixed_lmul) is READ-ONLY here: we only
# READ its KD teacher outputs (precomputed data), never write to it.
#
# Goal: recover VCTK MBSTOI >= 0.835 under the TRUE §3.4 hardware path
#   out = APoT(amax/448) * Σ LMul(act, code)        (delayed post-acc descale)
# warm-started from the exact-descale deploy ckpt (0.8355). The delayed descale
# is active during TRAINING (BCCTN_OUTLIER_DELAYED_DESCALE=1) so the model adapts
# to the deployed arithmetic. save_top_k=3 keeps the 3 best-by-val epoch ckpts.
#
#   GPU_ID=<free> bash run_apot_descale_finetune.sh
set -euo pipefail

PKG=/raid/R12K41024/18_w4a8_pow2_mixed_lmul_hw_handoff_20260530
OLD_RO=/raid/R12K41024/18_w4a8_pow2_mixed_lmul        # READ-ONLY: KD teacher only
cd "$PKG/code"

START_CKPT="${START_CKPT:-${PKG}/ckpt/strict_core_epoch4_step1210_val13.6105.ckpt}"
TEACHER_DIR="${TEACHER_DIR:-${OLD_RO}/teacher_outputs/d1c3_epoch25_step100672_fp16}"
TEACHER_ROOT="${TEACHER_ROOT:-/raid/R12K41024/BCCTN/Dataset/noisy_trainset}"
P2_INDEX="${P2_INDEX:-${PKG}/ckpt/p2_skip_indices_g42_topk_l1_r050.pt}"

ATTN_M="${ATTN_M:-8}"; OUTLIER_M="${OUTLIER_M:-4}"; SCALE_FMT="${SCALE_FMT:-apot_clip}"
ACT_FP8_MODE="${ACT_FP8_MODE:-${BCCTN_ACTIVATION_FP8_MODE:-scaled}}"
APOT_A_MIN="${APOT_A_MIN:--16}"; APOT_A_MAX="${APOT_A_MAX:--1}"
APOT_B_MIN="${APOT_B_MIN:--16}"; APOT_B_MAX="${APOT_B_MAX:--1}"
STE_MODE="${STE_MODE:-lmul_exact}"
KD_LAMBDA="${KD_LAMBDA:-0.0528721747}"
LR="${LR:-1e-5}"; N_EPOCHS="${N_EPOCHS:-5}"
LIMIT_TRAIN="${LIMIT_TRAIN:-1.0}"; LIMIT_VAL="${LIMIT_VAL:-1.0}"
BATCH_SIZE="${BATCH_SIZE:-128}"; SAVE_TOP_K="${SAVE_TOP_K:-3}"; GPU_ID="${GPU_ID:-6}"
SAVE_EVERY_EPOCH="${SAVE_EVERY_EPOCH:-false}"

# delayed APoT descale (true §3.4 HW path) — the whole point of this finetune
DELAYED_DESCALE="${DELAYED_DESCALE:-1}"
DD_MIN="${DD_MIN:--16}"; DD_MAX="${DD_MAX:--6}"

# strict-core BN / PReLU (unchanged from deploy)
BN_MODE="${BN_MODE:-fold_apot}"; BN_MIN_EXP="${BN_MIN_EXP:--9}"; BN_MAX_EXP="${BN_MAX_EXP:-4}"; BN_TIE="${BN_TIE:-false}"
PRELU_POW2="${PRELU_POW2:-true}"; PRELU_MIN_EXP="${PRELU_MIN_EXP:--8}"; PRELU_MAX_EXP="${PRELU_MAX_EXP:-0}"

RUN_TAG="${RUN_TAG:-apot_descale_delayed_dd${DD_MIN}_${DD_MAX}_lr${LR}_$(date +%Y%m%d_%H%M%S)}"

for f in "$START_CKPT" "$P2_INDEX" "$TEACHER_DIR" "$TEACHER_ROOT"; do
  [ -e "$f" ] || { echo "[apot-ft] FATAL: missing $f" >&2; exit 1; }
done

OUT_DIR="${PKG}/outputs/apot_descale_finetune/${RUN_TAG}"
LOG_DIR="${PKG}/logs/apot_descale_finetune/${RUN_TAG}"
mkdir -p "$OUT_DIR" "$LOG_DIR"
export PYTHONPATH="${PKG}/code"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl_cache}"
export BCCTN_STE_MODE="$STE_MODE"
export BCCTN_APOT_A_MIN="$APOT_A_MIN"; export BCCTN_APOT_A_MAX="$APOT_A_MAX"
export BCCTN_APOT_B_MIN="$APOT_B_MIN"; export BCCTN_APOT_B_MAX="$APOT_B_MAX"
export BCCTN_ACTIVATION_FP8_MODE="$ACT_FP8_MODE"
export BCCTN_OUTLIER_DELAYED_DESCALE="$DELAYED_DESCALE"
export BCCTN_OUTLIER_DESCALE_APOT_MIN="$DD_MIN"; export BCCTN_OUTLIER_DESCALE_APOT_MAX="$DD_MAX"

echo "[apot-ft] RUN_TAG=$RUN_TAG"
echo "[apot-ft] warm-start=$START_CKPT"
echo "[apot-ft] activation_fp8_mode=$ACT_FP8_MODE"
echo "[apot-ft] DELAYED descale=$DELAYED_DESCALE exp[$DD_MIN,$DD_MAX]  (true §3.4 HW path)"
echo "[apot-ft] LR=$LR epochs=$N_EPOCHS save_top_k=$SAVE_TOP_K save_every_epoch=$SAVE_EVERY_EPOCH GPU=$GPU_ID out=$OUT_DIR"

overrides=(
  training.qat_scope=full
  training.top_k_attention=4
  training.fp16_accumulator=true
  training.int4_scale_format="$SCALE_FMT"
  training.batch_size="$BATCH_SIZE"
  training.accumulate_grad_batches=1
  training.num_sanity_val_steps=0
  training.learning_rate="$LR"
  training.limit_train_batches="$LIMIT_TRAIN"
  training.limit_val_batches="$LIMIT_VAL"
  training.checkpoint_save_top_k="$SAVE_TOP_K"
  training.checkpoint_save_every_epoch="$SAVE_EVERY_EPOCH"
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
  model.kd.enabled=true
  model.kd.kd_mode=binaural
  model.kd.lambda_kd="$KD_LAMBDA"
  "model.kd.teacher_output_dir=$TEACHER_DIR"
  "model.kd.teacher_noisy_root=$TEACHER_ROOT"
)

stage_dir="${OUT_DIR}/ft"
ckpt_args=( "training.train_checkpoint_path='${START_CKPT}'" "training.resume_ckpt_path=null" )

CUDA_VISIBLE_DEVICES="$GPU_ID" conda run --no-capture-output -n bcctn \
  python -u train.py \
    "${ckpt_args[@]}" \
    training.n_epochs="$N_EPOCHS" \
    "${overrides[@]}" \
    "hydra.run.dir=${stage_dir}" \
    2>&1 | tee "${LOG_DIR}/train.log"

echo ""
echo "[apot-ft] done. ckpts (top-$SAVE_TOP_K by val):"
ls -1 "${stage_dir}/logs/lightning_logs/version_0/checkpoints/" 2>/dev/null || true
