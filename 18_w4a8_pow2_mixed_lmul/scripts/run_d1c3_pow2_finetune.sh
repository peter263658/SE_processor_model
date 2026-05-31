#!/usr/bin/env bash
# Multiplier-free fine-tune from the d1+c3 production base (0.8354).
#
# A1 (2026-05-26) showed attention m4 ≈ m10 (0.8356 vs 0.8354) → attention
# precision is NOT the bottleneck; the full-L-Mul drop is CONV. And B1b-proxy
# showed L-Mul-on-dequant-conv only HURTS — the exact shift-add (= real-mult)
# is the right multiplier-free conv. So fine-tune the production model to
# absorb the full multiplier-free arithmetic below.
#
# Target arithmetic (every multiply removed; this is what THIS run trains):
#   conv / ConvT / linear / MHA-proj : INT4 shift-add (= real-mult EXACT proxy), all_lmul OFF
#   qk / score_v                     : L-Mul m4  (c3 packet datapath, STE lmul_exact)
#   decoder_pw_outlier (12, FP8)     : FP8×FP8 L-Mul m4 (outlier_fp8_lmul, multiplier-free)
#   INT4 weight scale                : pow2 (1 shift) | apot=2^a±2^b (2 shift+1 add) — both multiplier-free
#   accumulation                     : fp16 (attention) / fp32 (conv) tensors
# => fully multiplier-free; per-group scale is amortized shift(s)+add, no multiply.
#    (fp32/fp16/fp8 scale would be a real multiply — NOT this target.)
#
# vs the old 15/16/17 runs: (1) base = d1+c3 (real-conv, gate-passing), not the
# 15/ all-LMul plateau; (2) int4_scale_format=pow2 — the 15/16/17 runs used
# fp32 scale, i.e. they were never actually multiplier-free.
#
# DRAFT — not auto-launched. Review, then:
#   GPU_ID=<free> bash scripts/run_d1c3_pow2_finetune.sh
set -euo pipefail

REPO=/raid/R12K41024/18_w4a8_pow2_mixed_lmul
cd "$REPO"

# d1+c3 production ckpt (= KD teacher base, conv REAL, VCTK 0.8354 / TIMIT 0.7880)
START_CKPT="${START_CKPT:-/raid/R12K41024/14_low_power_arch_qat/outputs/low_power_phase_d/d1_qkv_g4_2026-05-20_02-56-03/logs/lightning_logs/version_0/checkpoints/epoch=25-step=100672.ckpt}"
TEACHER_DIR="${TEACHER_DIR:-${REPO}/teacher_outputs/d1c3_epoch25_step100672_fp16}"
TEACHER_ROOT="${TEACHER_ROOT:-/raid/R12K41024/BCCTN/Dataset/noisy_trainset}"
P2_INDEX="${P2_INDEX:-/raid/R12K41024/14_low_power_arch_qat/Checkpoints/p2_skip_indices_g42_topk_l1_r050.pt}"

CONV_M="${CONV_M:-4}"            # conv/linear/proj L-Mul mantissa (try 5/6 if m4 too lossy)
ATTN_M="${ATTN_M:-4}"            # qk/pv L-Mul mantissa (A1: precision doesn't matter, keep cheap)
SCALE_FMT="${SCALE_FMT:-pow2}"   # pow2 | apot | apot_clip = multiplier-free; fp32/fp16/fp8 = real multiply
APOT_A_MIN="${APOT_A_MIN:--16}"
APOT_A_MAX="${APOT_A_MAX:--1}"
APOT_B_MIN="${APOT_B_MIN:--16}"
APOT_B_MAX="${APOT_B_MAX:--1}"
STE_MODE="${STE_MODE:-lmul_exact}"
OUTLIER_M="${OUTLIER_M:-4}"          # FP8×FP8 L-Mul mantissa for decoder_pw_outlier (FP8 is 3-bit → m4 lossless)
KD_LAMBDA="${KD_LAMBDA:-0.0528721747}"
KD_ENABLED="${KD_ENABLED:-true}"
LR="${LR:-1e-5}"
N_EPOCHS="${N_EPOCHS:-5}"
LIMIT_TRAIN="${LIMIT_TRAIN:-0.5}"   # fast first val
LIMIT_VAL="${LIMIT_VAL:-1.0}"
BATCH_SIZE="${BATCH_SIZE:-128}"
GPU_ID="${GPU_ID:-1}"
RUN_TAG="${RUN_TAG:-d1c3_${SCALE_FMT}_outlierm4_lr${LR}_$(date +%Y%m%d_%H%M%S)}"

for f in "$START_CKPT" "$P2_INDEX" "$TEACHER_DIR" "$TEACHER_ROOT"; do
  [ -e "$f" ] || { echo "[pow2-ft] FATAL: missing $f" >&2; exit 1; }
done

OUT_DIR="${REPO}/outputs/d1c3_pow2_finetune/${RUN_TAG}"
LOG_DIR="${REPO}/logs/d1c3_pow2_finetune/${RUN_TAG}"
mkdir -p "$OUT_DIR" "$LOG_DIR"
export PYTHONPATH="${PYTHONPATH:-}:${REPO}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl_cache}"
export BCCTN_APOT_A_MIN="$APOT_A_MIN"
export BCCTN_APOT_A_MAX="$APOT_A_MAX"
export BCCTN_APOT_B_MIN="$APOT_B_MIN"
export BCCTN_APOT_B_MAX="$APOT_B_MAX"
# all_lmul is OFF (conv=real proxy), so enable_all_lmul_multiply() won't publish
# BCCTN_STE_MODE. Set it here so the qk/score_v attention-datapath L-Mul backward
# uses the intended STE (else it silently defaults to "real").
export BCCTN_STE_MODE="$STE_MODE"

echo "[pow2-ft] RUN_TAG=$RUN_TAG"
echo "[pow2-ft] base=$START_CKPT"
echo "[pow2-ft] conv_m=$CONV_M attn_m=$ATTN_M scale=$SCALE_FMT ste=$STE_MODE"
echo "[pow2-ft] LR=$LR epochs=$N_EPOCHS limit_train=$LIMIT_TRAIN KD=$KD_ENABLED GPU=$GPU_ID"
case "$SCALE_FMT" in
  pow2|power_of_two)            echo "[pow2-ft] scale=pow2 (1 shift) — multiplier-free" ;;
  apot|apot2|apot_2|two_pow2|two_term_pow2) echo "[pow2-ft] scale=apot 2^a±2^b (2 shift+1 add) — multiplier-free, finer than pow2" ;;
  apot_clip|apot_clipped)       echo "[pow2-ft] scale=apot_clip 2^a±2^b, a=[${APOT_A_MIN},${APOT_A_MAX}], b=[${APOT_B_MIN},${APOT_B_MAX}] — multiplier-free bounded exponents" ;;
  *)                           echo "[pow2-ft] WARN: scale=$SCALE_FMT is a REAL multiply (NOT multiplier-free; diagnostic only)" ;;
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
  # B1b: conv/ConvT/Linear = REAL multiply = shift-add EXACT proxy (NOT L-Mul).
  # Multiplier-free comes from int4_scale_format=pow2 above + INT4 W4A8 (qat_scope=full).
  # all_lmul OFF because A1/B1b-proxy showed L-Mul on dequant-weight only HURTS
  # (0.8287/0.8288) vs the exact shift-add proxy = A0 0.8354.
  model.low_power.all_lmul_multiply.enabled=false
  # decoder_pw_outlier (12 FP8 layers) → FP8×FP8 L-Mul m4 (multiplier-free).
  # Selective: only model._fp8_preserved_names, never MHA / INT4 conv.
  model.low_power.outlier_fp8_lmul.enabled=true
  model.low_power.outlier_fp8_lmul.mantissa_bits="$OUTLIER_M"
)

case "$KD_ENABLED" in
  true|True|TRUE|1|yes|YES)
    overrides+=(
      model.kd.enabled=true
      model.kd.kd_mode=binaural
      model.kd.lambda_kd="$KD_LAMBDA"
      "model.kd.teacher_output_dir=$TEACHER_DIR"
      "model.kd.teacher_noisy_root=$TEACHER_ROOT"
    )
    ;;
  false|False|FALSE|0|no|NO)
    overrides+=( model.kd.enabled=false )
    ;;
  *)
    echo "[pow2-ft] FATAL: KD_ENABLED must be true/false, got $KD_ENABLED" >&2
    exit 1
    ;;
esac

stage_dir="${OUT_DIR}/ft"
# RESUME_CKPT set => TRUE resume (Lightning ckpt_path: keep optimizer/scheduler/
# epoch) — the right way to CONTINUE an improving run. n_epochs is the TOTAL
# (resume picks up from the saved epoch). Else warm-start (fresh optimizer).
if [ -n "${RESUME_CKPT:-}" ]; then
  [ -f "$RESUME_CKPT" ] || { echo "[pow2-ft] FATAL: missing RESUME_CKPT $RESUME_CKPT" >&2; exit 1; }
  ckpt_args=( "training.resume_ckpt_path='${RESUME_CKPT}'" "training.train_checkpoint_path=null" )
  echo "[pow2-ft] RESUME mode (keep optimizer) from: $RESUME_CKPT  (n_epochs total=$N_EPOCHS)"
else
  ckpt_args=( "training.train_checkpoint_path='${START_CKPT}'" "training.resume_ckpt_path=null" )
fi
CUDA_VISIBLE_DEVICES="$GPU_ID" conda run --no-capture-output -n bcctn \
  python -u train.py \
    "${ckpt_args[@]}" \
    training.n_epochs="$N_EPOCHS" \
    "${overrides[@]}" \
    "hydra.run.dir=${stage_dir}" \
    2>&1 | tee "${LOG_DIR}/train.log"

echo ""
echo "[pow2-ft] done. ckpt: ${stage_dir}/logs/lightning_logs/version_0/checkpoints/last.ckpt"
echo "[pow2-ft] read signal: grep 'New best score' ${LOG_DIR}/train.log"
echo "[pow2-ft] conv/proj = real-mult shift-add proxy + pow2 scale; qk/pv L-Mul m4; outlier FP8 L-Mul m4."
echo "[pow2-ft] first val = AFTER limit_train=${LIMIT_TRAIN} epoch (NOT zero-shot); trend up over epochs = recovery."
echo "[pow2-ft] full MBSTOI eval (slow ~hrs, conv real=fast) only after val_loss trends right."
