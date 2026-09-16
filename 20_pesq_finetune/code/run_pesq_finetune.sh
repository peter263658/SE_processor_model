#!/usr/bin/env bash
# PESQ-augmented FINE-TUNE (20_pesq_finetune).
#
# Goal: probe how high wideband PESQ can be pushed on the *existing* deploy
# W4A8 architecture WITHOUT giving up MBSTOI / spatial cues. First time PESQ
# enters the loss: BinauralLoss now adds a differentiable torch_pesq term
# (model.pesq_weight). STOI(10)+IPD(10)+ILD(1)+SNR(1) are kept so MBSTOI and
# the binaural cues are held while PESQ is optimised.
#
# Warm-start from the SAME deploy base 19_algo_final used (d1c3 bounded-APoT m8,
# VCTK 0.835 / TIMIT 0.789) and keep the full multiplier-free hardware stack so
# the checkpoint loads and the number reflects the real chip model.
#
# KEY DIFFERENCE vs run_strict_core_finetune.sh:
#   * runs THIS folder's train.py / DCNN (the PESQ-augmented loss), NOT 18's
#   * model.pesq_weight=$PESQ_WEIGHT  (new term)
#   * KD disabled (removes teacher dep + halves per-step PESQ compute)
#   * batch 64 (head-room for the extra PESQ activation memory)
#
# Usage (sweep PESQ strength across free GPUs):
#   PESQ_WEIGHT=6 GPU_ID=0 bash run_pesq_finetune.sh
set -euo pipefail

# THIS folder's code (modified PESQ loss lives here) — train.py runs from here.
CODE_DIR=/raid/R12K41024/20_pesq_finetune/code
# Deploy-base deps (checkpoint + grouped-pointwise skip index) live in 18.
DEPS=/raid/R12K41024/18_w4a8_pow2_mixed_lmul
cd "$CODE_DIR"

START_CKPT="${START_CKPT:-${DEPS}/outputs/d1c3_pow2_finetune/d1c3_apotclip16_attnm8_outlierm4_15ep/ft/logs/lightning_logs/version_0/checkpoints/epoch=2-step=726.ckpt}"
P2_INDEX="${P2_INDEX:-/raid/R12K41024/14_low_power_arch_qat/Checkpoints/p2_skip_indices_g42_topk_l1_r050.pt}"

# --- experiment knobs ---
PESQ_WEIGHT="${PESQ_WEIGHT:-6}"     # NEW: differentiable PESQ loss weight
LR="${LR:-1e-5}"
N_EPOCHS="${N_EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-64}"
SAVE_TOP_K="${SAVE_TOP_K:-3}"
GPU_ID="${GPU_ID:-0}"

# --- optional dataset override (e.g. VB-finetune). Empty = use config (VCTK). ---
NOISY_TRAIN="${NOISY_TRAIN:-}"; CLEAN_TRAIN="${CLEAN_TRAIN:-}"
NOISY_VAL="${NOISY_VAL:-}";     CLEAN_VAL="${CLEAN_VAL:-}"
NOISY_TRAIN2="${NOISY_TRAIN2:-}"; CLEAN_TRAIN2="${CLEAN_TRAIN2:-}"  # 2nd corpus -> balanced mix
CROP_LEN="${CROP_LEN:-}"        # frames; set 32000 for variable-length VB

# --- deploy hardware-stack knobs (match the strict-core base so ckpt loads) ---
ATTN_M="${ATTN_M:-8}"
OUTLIER_M="${OUTLIER_M:-4}"
SCALE_FMT="${SCALE_FMT:-apot_clip}"
APOT_A_MIN="${APOT_A_MIN:--16}"; APOT_A_MAX="${APOT_A_MAX:--1}"
APOT_B_MIN="${APOT_B_MIN:--16}"; APOT_B_MAX="${APOT_B_MAX:--1}"
STE_MODE="${STE_MODE:-lmul_exact}"
BN_MODE="${BN_MODE:-fold_apot}"
BN_MIN_EXP="${BN_MIN_EXP:--9}"; BN_MAX_EXP="${BN_MAX_EXP:-4}"; BN_TIE="${BN_TIE:-false}"
PRELU_POW2="${PRELU_POW2:-true}"; PRELU_MIN_EXP="${PRELU_MIN_EXP:--8}"; PRELU_MAX_EXP="${PRELU_MAX_EXP:-0}"

RUN_TAG="${RUN_TAG:-pesq_w${PESQ_WEIGHT}_lr${LR}_ep${N_EPOCHS}}"

for f in "$START_CKPT" "$P2_INDEX"; do
  [ -e "$f" ] || { echo "[pesq-ft] FATAL: missing $f" >&2; exit 1; }
done

OUT_DIR="/raid/R12K41024/20_pesq_finetune/outputs/pesq_finetune/${RUN_TAG}"
LOG_DIR="/raid/R12K41024/20_pesq_finetune/logs/pesq_finetune/${RUN_TAG}"
mkdir -p "$OUT_DIR" "$LOG_DIR"
export PYTHONPATH="${CODE_DIR}:${PYTHONPATH:-}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl_cache}"
export BCCTN_STE_MODE="$STE_MODE"
export BCCTN_APOT_A_MIN="$APOT_A_MIN"; export BCCTN_APOT_A_MAX="$APOT_A_MAX"
export BCCTN_APOT_B_MIN="$APOT_B_MIN"; export BCCTN_APOT_B_MAX="$APOT_B_MAX"

echo "[pesq-ft] RUN_TAG=$RUN_TAG  PESQ_WEIGHT=$PESQ_WEIGHT  LR=$LR  epochs=$N_EPOCHS  batch=$BATCH_SIZE  GPU=$GPU_ID"
echo "[pesq-ft] warm-start=$START_CKPT"

overrides=(
  # --- new PESQ loss term ---
  model.pesq_weight="$PESQ_WEIGHT"
  # --- keep MBSTOI + spatial terms (unchanged from deploy) ---
  model.stoi_weight=10
  model.ipd_weight=10
  model.ild_weight=1
  model.snr_loss_weight=1
  # --- deploy W4A8 hardware stack (verbatim from strict-core, so ckpt loads) ---
  training.qat_scope=full
  training.top_k_attention=4
  training.fp16_accumulator=true
  training.int4_scale_format="$SCALE_FMT"
  training.batch_size="$BATCH_SIZE"
  training.accumulate_grad_batches=1
  training.num_sanity_val_steps=0
  training.learning_rate="$LR"
  training.checkpoint_save_top_k="$SAVE_TOP_K"
  training.checkpoint_save_every_epoch=true
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
  # --- KD off: removes teacher dep + halves per-step PESQ compute ---
  model.kd.enabled=false
)

# Optional dataset override (VB-finetune): point train/val at rendered VB dirs.
[ -n "$NOISY_TRAIN" ] && overrides+=( "dataset.noisy_training_dataset_dir=$NOISY_TRAIN" )
[ -n "$CLEAN_TRAIN" ] && overrides+=( "dataset.target_training_dataset_dir=$CLEAN_TRAIN" )
[ -n "$NOISY_VAL" ]   && overrides+=( "dataset.noisy_validation_dataset_dir=$NOISY_VAL" "dataset.noisy_test_dataset_dir=$NOISY_VAL" )
[ -n "$CLEAN_VAL" ]   && overrides+=( "dataset.target_validation_dataset_dir=$CLEAN_VAL" "dataset.target_test_dataset_dir=$CLEAN_VAL" )
[ -n "$CROP_LEN" ]    && overrides+=( "dataset.crop_len=$CROP_LEN" )
[ -n "$NOISY_TRAIN2" ] && overrides+=( "dataset.noisy_training_dataset_dir2=$NOISY_TRAIN2" )
[ -n "$CLEAN_TRAIN2" ] && overrides+=( "dataset.target_training_dataset_dir2=$CLEAN_TRAIN2" )

stage_dir="${OUT_DIR}/ft"
ckpt_args=( "training.train_checkpoint_path='${START_CKPT}'" "training.resume_ckpt_path=null" )

CONDA_BIN="${CONDA_BIN:-/home/R12K41024/miniforge3/bin/conda}"
CUDA_VISIBLE_DEVICES="$GPU_ID" "$CONDA_BIN" run --no-capture-output -n bcctn \
  python -u train.py \
    "${ckpt_args[@]}" \
    training.n_epochs="$N_EPOCHS" \
    "${overrides[@]}" \
    "hydra.run.dir=${stage_dir}" \
    2>&1 | tee "${LOG_DIR}/train.log"

echo ""
echo "[pesq-ft] done. ckpts:"
ls -1 "${stage_dir}/logs/lightning_logs/version_0/checkpoints/" 2>/dev/null || true
