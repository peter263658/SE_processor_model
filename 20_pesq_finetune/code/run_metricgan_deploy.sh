#!/usr/bin/env bash
# MetricGAN PESQ finetune on the W4A8 DEPLOY architecture.
#
# Target: beat reproduced-SNU wideband PESQ (Δ +0.562 @ −6..9) WHILE holding
# the MBSTOI ≥ 0.835 gate and keeping ILD positive. MetricGAN's learned PESQ
# critic pushes PESQ with a gentler gradient than torch_pesq weight-cranking
# (which drops MBSTOI to ~0.82). Warm-starts from the deploy ep3 gate model.
#
#   GAN_W=16 GPU_ID=<free> bash run_metricgan_deploy.sh
set -euo pipefail

CODE_DIR=/raid/R12K41024/20_pesq_finetune/code
DEPS=/raid/R12K41024/18_w4a8_pow2_mixed_lmul
cd "$CODE_DIR"

# Warm-start = deploy ep3 (VCTK MBSTOI 0.8357 / TIMIT 0.7898 gate model).
START_CKPT="${START_CKPT:-${DEPS}/outputs/d1c3_pow2_finetune/d1c3_apotclip16_attnm8_outlierm4_15ep/ft/logs/lightning_logs/version_0/checkpoints/epoch=2-step=726.ckpt}"
P2_INDEX="${P2_INDEX:-/raid/R12K41024/14_low_power_arch_qat/Checkpoints/p2_skip_indices_g42_topk_l1_r050.pt}"

# --- MetricGAN knobs ---
GAN_W="${GAN_W:-16}"                 # PESQ-critic pressure on the generator
LR="${LR:-2e-5}"                     # generator lr
LR_D="${LR_D:-5e-4}"                 # discriminator lr
N_EPOCHS="${N_EPOCHS:-25}"
D_WARMUP="${D_WARMUP:-2}"            # D-only epochs before G updates start
BATCH_SIZE="${BATCH_SIZE:-32}"       # smaller than finetune (extra D memory)
GPU_ID="${GPU_ID:-0}"
# BCCTN_D_INIT_CKPT (env, optional): warm-init D from a prior MetricGAN ckpt.
# gate-holding anchors: STOI high (holds MBSTOI), ILD moderate (holds ILD)
STOI_W="${STOI_W:-10}"; IPD_W="${IPD_W:-10}"; ILD_W="${ILD_W:-4}"; SNR_W="${SNR_W:-1}"

# --- deploy hardware-stack knobs (verbatim from run_pesq_finetune.sh) ---
ATTN_M="${ATTN_M:-8}"; OUTLIER_M="${OUTLIER_M:-4}"; SCALE_FMT="${SCALE_FMT:-apot_clip}"
APOT_A_MIN="${APOT_A_MIN:--16}"; APOT_A_MAX="${APOT_A_MAX:--1}"
APOT_B_MIN="${APOT_B_MIN:--16}"; APOT_B_MAX="${APOT_B_MAX:--1}"
STE_MODE="${STE_MODE:-lmul_exact}"
BN_MODE="${BN_MODE:-fold_apot}"; BN_MIN_EXP="${BN_MIN_EXP:--9}"; BN_MAX_EXP="${BN_MAX_EXP:-4}"; BN_TIE="${BN_TIE:-false}"
PRELU_POW2="${PRELU_POW2:-true}"; PRELU_MIN_EXP="${PRELU_MIN_EXP:--8}"; PRELU_MAX_EXP="${PRELU_MAX_EXP:-0}"

RUN_TAG="${RUN_TAG:-metricgan_deploy_gw${GAN_W}}"

for f in "$START_CKPT" "$P2_INDEX"; do
  [ -e "$f" ] || { echo "[mgan-deploy] FATAL: missing $f" >&2; exit 1; }
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

echo "[mgan-deploy] RUN_TAG=$RUN_TAG GAN_W=$GAN_W STOI_W=$STOI_W ILD_W=$ILD_W LR=$LR/$LR_D ep=$N_EPOCHS batch=$BATCH_SIZE GPU=$GPU_ID"
echo "[mgan-deploy] warm-start=$START_CKPT"

overrides=(
  # PESQ comes from the GAN critic, NOT torch_pesq:
  model.pesq_weight=0
  +model.gan_weight="$GAN_W"
  +model.d_warmup_epochs="$D_WARMUP"
  +training.lr_d="$LR_D"
  +model.snr_loss_mode="${SNR_MODE:-snr}"
  # gate-holding + spatial anchors:
  model.stoi_weight="$STOI_W"
  model.ipd_weight="$IPD_W"
  model.ild_weight="$ILD_W"
  model.snr_loss_weight="$SNR_W"
  # deploy W4A8 hardware stack (verbatim so ckpt loads / number is the chip):
  training.qat_scope=full
  training.top_k_attention=4
  training.fp16_accumulator=true
  training.int4_scale_format="$SCALE_FMT"
  training.batch_size="$BATCH_SIZE"
  training.accumulate_grad_batches=1
  training.num_sanity_val_steps=0
  training.learning_rate="$LR"
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

# SMOKE=1 -> tiny run to validate build/warm-start/GAN-step/checkpoint plumbing.
[ -n "${SMOKE:-}" ] && overrides+=( ++training.limit_train_batches=3 ++training.limit_val_batches=2 )

stage_dir="${OUT_DIR}/ft"
ckpt_args=( "training.train_checkpoint_path='${START_CKPT}'" "training.resume_ckpt_path=null" )

CONDA_BIN="${CONDA_BIN:-/home/R12K41024/miniforge3/bin/conda}"
CUDA_VISIBLE_DEVICES="$GPU_ID" "$CONDA_BIN" run --no-capture-output -n bcctn \
  python -u train_metricgan_deploy.py \
    "${ckpt_args[@]}" \
    training.n_epochs="$N_EPOCHS" \
    "${overrides[@]}" \
    "hydra.run.dir=${stage_dir}" \
    2>&1 | tee "${LOG_DIR}/train.log"

echo ""; echo "[mgan-deploy] done. ckpts:"
ls -1 "${stage_dir}/logs/lightning_logs/version_0/checkpoints/" 2>/dev/null || true
