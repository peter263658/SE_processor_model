#!/usr/bin/env bash
# STRICT network-inference-core zero-multiplier FINE-TUNE.
#
# Warm-start from the bounded-APoT m8 KD best (VCTK 0.835059 / TIMIT 0.789213)
# and fine-tune with the TWO remaining ops now made multiplier-free:
#   BN    : fold a=gamma/sqrt(running_var+eps) -> APoT[-9,4] shift(s)+add + bias
#   PReLU : slope -> pow2[-8,0] shift
# plus the existing B1b stack (conv INT4xFP8 shift-add + APoT-clip scale,
# QK/PV L-Mul m8, PW outlier FP8 L-Mul m4, binaural KD).
#
# The BN fold path reads RUNNING stats directly (bypasses F.batch_norm), so BN
# stats are FROZEN during FT — only gamma/beta (+ everything else) train through
# the STE. That is the intended "BN freeze + fold" recovery.
#
# This is a fresh-optimizer WARM-START (new arithmetic the model hasn't seen),
# NOT a resume. Short (5 ep), LR 1e-5. save_top_k=3 keeps the 3 best-by-val
# epoch ckpts for post-hoc MBSTOI screening (val_loss is a poor MBSTOI proxy).
#
# DRAFT — review + smoke (scripts/smoke_strict_core.py) BEFORE launching:
#   GPU_ID=<free> bash scripts/run_strict_core_finetune.sh
set -euo pipefail

REPO=/raid/R12K41024/18_w4a8_pow2_mixed_lmul
cd "$REPO"

# Bounded-APoT m8 KD best (= strict-FT base).
START_CKPT="${START_CKPT:-${REPO}/outputs/d1c3_pow2_finetune/d1c3_apotclip16_attnm8_outlierm4_15ep/ft/logs/lightning_logs/version_0/checkpoints/epoch=2-step=726.ckpt}"
TEACHER_DIR="${TEACHER_DIR:-${REPO}/teacher_outputs/d1c3_epoch25_step100672_fp16}"
TEACHER_ROOT="${TEACHER_ROOT:-/raid/R12K41024/BCCTN/Dataset/noisy_trainset}"
P2_INDEX="${P2_INDEX:-/raid/R12K41024/14_low_power_arch_qat/Checkpoints/p2_skip_indices_g42_topk_l1_r050.pt}"

ATTN_M="${ATTN_M:-8}"
OUTLIER_M="${OUTLIER_M:-4}"
SCALE_FMT="${SCALE_FMT:-apot_clip}"
APOT_A_MIN="${APOT_A_MIN:--16}"; APOT_A_MAX="${APOT_A_MAX:--1}"
APOT_B_MIN="${APOT_B_MIN:--16}"; APOT_B_MAX="${APOT_B_MAX:--1}"
STE_MODE="${STE_MODE:-lmul_exact}"
KD_LAMBDA="${KD_LAMBDA:-0.0528721747}"
LR="${LR:-1e-5}"
N_EPOCHS="${N_EPOCHS:-5}"
LIMIT_TRAIN="${LIMIT_TRAIN:-1.0}"
LIMIT_VAL="${LIMIT_VAL:-1.0}"
BATCH_SIZE="${BATCH_SIZE:-128}"
SAVE_TOP_K="${SAVE_TOP_K:-3}"
GPU_ID="${GPU_ID:-1}"

# strict-core BN / PReLU
BN_MODE="${BN_MODE:-fold_apot}"
BN_MIN_EXP="${BN_MIN_EXP:--9}"; BN_MAX_EXP="${BN_MAX_EXP:-4}"; BN_TIE="${BN_TIE:-false}"
PRELU_POW2="${PRELU_POW2:-true}"; PRELU_MIN_EXP="${PRELU_MIN_EXP:--8}"; PRELU_MAX_EXP="${PRELU_MAX_EXP:-0}"

RUN_TAG="${RUN_TAG:-strict_core_bnfold_prelu_pow2_lr${LR}_$(date +%Y%m%d_%H%M%S)}"

for f in "$START_CKPT" "$P2_INDEX" "$TEACHER_DIR" "$TEACHER_ROOT"; do
  [ -e "$f" ] || { echo "[strict-ft] FATAL: missing $f" >&2; exit 1; }
done

OUT_DIR="${REPO}/outputs/strict_core_finetune/${RUN_TAG}"
LOG_DIR="${REPO}/logs/strict_core_finetune/${RUN_TAG}"
mkdir -p "$OUT_DIR" "$LOG_DIR"
export PYTHONPATH="${PYTHONPATH:-}:${REPO}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl_cache}"
export BCCTN_STE_MODE="$STE_MODE"
export BCCTN_APOT_A_MIN="$APOT_A_MIN"; export BCCTN_APOT_A_MAX="$APOT_A_MAX"
export BCCTN_APOT_B_MIN="$APOT_B_MIN"; export BCCTN_APOT_B_MAX="$APOT_B_MAX"

echo "[strict-ft] RUN_TAG=$RUN_TAG"
echo "[strict-ft] base(warm-start)=$START_CKPT"
echo "[strict-ft] conv=$SCALE_FMT a/b=[$APOT_A_MIN,$APOT_A_MAX]  attn_m=$ATTN_M  outlier_m=$OUTLIER_M"
echo "[strict-ft] BN=$BN_MODE exp[$BN_MIN_EXP,$BN_MAX_EXP] tie=$BN_TIE  PReLU pow2=$PRELU_POW2 exp[$PRELU_MIN_EXP,$PRELU_MAX_EXP]"
echo "[strict-ft] LR=$LR epochs=$N_EPOCHS save_top_k=$SAVE_TOP_K GPU=$GPU_ID"
case "$BN_MODE" in
  fold_apot|fold_pow2) : ;;
  *) echo "[strict-ft] WARN: BN_MODE=$BN_MODE is NOT a fold mode -> BN keeps a real multiply (NOT strict-core)" ;;
esac

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
  # --- strict-core: BN + PReLU multiplier-free ---
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
# WARM-START (fresh optimizer): new BN/PReLU arithmetic the model hasn't seen.
ckpt_args=( "training.train_checkpoint_path='${START_CKPT}'" "training.resume_ckpt_path=null" )

CUDA_VISIBLE_DEVICES="$GPU_ID" conda run --no-capture-output -n bcctn \
  python -u train.py \
    "${ckpt_args[@]}" \
    training.n_epochs="$N_EPOCHS" \
    "${overrides[@]}" \
    "hydra.run.dir=${stage_dir}" \
    2>&1 | tee "${LOG_DIR}/train.log"

echo ""
echo "[strict-ft] done. ckpts (top-$SAVE_TOP_K by val + last):"
ls -1 "${stage_dir}/logs/lightning_logs/version_0/checkpoints/" 2>/dev/null || true
echo "[strict-ft] next: full eval EACH epoch=*.ckpt, pick MBSTOI-best (NOT val-best):"
echo "  CKPT=<epoch ckpt> GPU_ID=<free> bash scripts/run_strict_core_eval.sh"
