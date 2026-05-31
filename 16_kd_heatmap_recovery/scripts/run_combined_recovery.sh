#!/bin/bash
# Combined recovery run — curriculum L-Mul (m=6 → m=5 → m=4) +
# learnable per-channel bias-correction + binaural KD.
#
# Goal: from v4 Ep5 baseline (VCTK MBSTOI ~0.8281, w=200 ~0.8255), recover
# above the production gate (0.8354 VCTK / 0.7880 TIMIT) WITHOUT giving up
# the "L-Mul replaces all multiplies" hardware story.
#
# What this script does NOT do:
#   - post-hoc bias-correction (already ruled out by α-sweep)
#   - output-scale calibration
#   - magnitude-only KD (stft_l1, ruled out by Scout A)
#   - architecture changes that break L-Mul-everywhere
#
# Three sequential train.py invocations, each warm-starts weights from the
# prior stage's last.ckpt with a fresh optimizer and a reset epoch counter.
# Mantissa rises in difficulty: 6 → 5 → 4. The learnable δ
# (nn.Parameter[C_out] per Conv/ConvT/Linear) persists across stages via
# state_dict; stage 1 initializes missing δ parameters to zero when loading
# the v4 Ep5 checkpoint.
#
# Required env:
#   KD_LAMBDA          binaural-KD λ from probe (DO NOT GUESS; run
#                       scripts/probe_kd_loss_scale.py first)
#
# Optional env:
#   START_CKPT         starting ckpt (default v4 Ep5)
#   TEACHER_DIR        precomputed teacher dir (default d1c3 epoch25)
#   TEACHER_ROOT       teacher_noisy_root (default noisy_trainset)
#   GPU_ID             default 6
#   EPOCHS_M6 / M5 / M4   default 5 / 5 / 10
#   LR                 default 5e-6
#   BATCH_SIZE         default 128
#   RUN_TAG            timestamp tag for output dir (default: date +%Y%m%d_%H%M%S)
#   SKIP_EVAL          set to 1 to skip post-stage-3 full eval
#
# Outputs:
#   outputs/recovery_combined/<RUN_TAG>/m6/...
#   outputs/recovery_combined/<RUN_TAG>/m5/...
#   outputs/recovery_combined/<RUN_TAG>/m4/...
#   results/recovery_combined/<RUN_TAG>/eval/full_eval.log
#   logs/recovery_combined/<RUN_TAG>/{stage_m6,stage_m5,stage_m4,eval}.log
set -euo pipefail

REPO=/raid/R12K41024/16_kd_heatmap_recovery
cd "$REPO"

KD_LAMBDA="${KD_LAMBDA:?KD_LAMBDA is required (calibrated from probe_kd_loss_scale.py for binaural mode)}"
START_CKPT="${START_CKPT:-/raid/R12K41024/15_all_lmul_mixed_mult_eval/outputs/all_lmul_mixed_qat/all_lmul_m4_bitexact_qat_2026-05-24_02-49-17/logs/lightning_logs/version_0/checkpoints/epoch=5-step=1452.ckpt}"
TEACHER_DIR="${TEACHER_DIR:-${REPO}/teacher_outputs/d1c3_epoch25_step100672_fp16}"
TEACHER_ROOT="${TEACHER_ROOT:-/raid/R12K41024/BCCTN/Dataset/noisy_trainset}"
P2_INDEX="${P2_INDEX:-${REPO}/Checkpoints/p2_skip_indices_g42_topk_l1_r050.pt}"
GPU_ID="${GPU_ID:-6}"
EPOCHS_M6="${EPOCHS_M6:-5}"
EPOCHS_M5="${EPOCHS_M5:-5}"
EPOCHS_M4="${EPOCHS_M4:-10}"
LR="${LR:-5e-6}"
BATCH_SIZE="${BATCH_SIZE:-128}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
SKIP_EVAL="${SKIP_EVAL:-0}"

# Sanity-check inputs
for f in "$START_CKPT" "$P2_INDEX"; do
  [ -f "$f" ] || { echo "missing file: $f" >&2; exit 1; }
done
[ -d "$TEACHER_DIR" ] || { echo "missing TEACHER_DIR: $TEACHER_DIR" >&2; exit 1; }
[ -d "$TEACHER_ROOT" ] || { echo "missing TEACHER_ROOT: $TEACHER_ROOT" >&2; exit 1; }

RUN_DIR="${REPO}/outputs/recovery_combined/${RUN_TAG}"
LOG_DIR="${REPO}/logs/recovery_combined/${RUN_TAG}"
EVAL_DIR="${REPO}/results/recovery_combined/${RUN_TAG}/eval"
mkdir -p "$LOG_DIR" "$EVAL_DIR"

echo "[recovery] RUN_TAG       = $RUN_TAG"
echo "[recovery] START_CKPT    = $START_CKPT"
echo "[recovery] TEACHER_DIR   = $TEACHER_DIR"
echo "[recovery] KD_LAMBDA     = $KD_LAMBDA (binaural mode)"
echo "[recovery] EPOCHS        = m6:$EPOCHS_M6 → m5:$EPOCHS_M5 → m4:$EPOCHS_M4"
echo "[recovery] LR / BS       = $LR / $BATCH_SIZE"
echo "[recovery] GPU_ID        = $GPU_ID"
echo "[recovery] RUN_DIR       = $RUN_DIR"

export PYTHONPATH="${PYTHONPATH:-}:${REPO}"
export HYDRA_FULL_ERROR=1
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl_cache}"

# Common architecture overrides — d1_qkv_g4 + c3 packet+L-Mul stack.
# Mantissa fields are overridden per stage; everything else is constant.
common_overrides=(
  training.qat_scope=full
  training.top_k_attention=4
  training.fp16_accumulator=true
  training.int4_scale_format=fp32
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
  model.attention_datapath.lmul_all_projection_ops=true
  model.attention_datapath.lmul_linear_chunk_size=2048
  model.attention_datapath.lmul_checkpoint_chunks=false
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
  model.low_power.all_lmul_multiply.enabled=true
  model.low_power.all_lmul_multiply.bit_exact=true
  model.low_power.all_lmul_multiply.linear_chunk_size=2048
  model.low_power.all_lmul_multiply.checkpoint_chunks=false
  model.low_power.all_lmul_multiply.learnable_bias_correction.enabled=true
  model.low_power.all_lmul_multiply.learnable_bias_correction.init_value=0.0
  model.kd.enabled=true
  model.kd.kd_mode=binaural
  model.kd.lambda_kd="$KD_LAMBDA"
  "model.kd.teacher_output_dir=$TEACHER_DIR"
  "model.kd.teacher_noisy_root=$TEACHER_ROOT"
)

run_stage() {
  local stage_name="$1"        # m6 | m5 | m4
  local mantissa="$2"          # 6 | 5 | 4
  local n_epochs="$3"
  local resume_ckpt="$4"
  local stage_dir="${RUN_DIR}/${stage_name}"
  local stage_log="${LOG_DIR}/stage_${stage_name}.log"

  echo ""
  echo "==================================================================="
  echo "[recovery] stage=${stage_name}  mantissa=${mantissa}  epochs=${n_epochs}"
  echo "[recovery] resume from: ${resume_ckpt}"
  echo "[recovery] stage_dir  : ${stage_dir}"
  echo "[recovery] log        : ${stage_log}"
  echo "==================================================================="

  local mantissa_overrides=(
    model.attention_datapath.lmul_mantissa_bits="$mantissa"
    model.low_power.all_lmul_multiply.mantissa_bits="$mantissa"
    model.low_power.all_lmul_multiply.attention_mantissa_bits="$mantissa"
  )

  CUDA_VISIBLE_DEVICES="$GPU_ID" conda run --no-capture-output -n bcctn \
    python -u train.py \
      "training.train_checkpoint_path='${resume_ckpt}'" \
      training.resume_ckpt_path=null \
      training.n_epochs="$n_epochs" \
      "${common_overrides[@]}" \
      "${mantissa_overrides[@]}" \
      "hydra.run.dir=${stage_dir}" \
      > "$stage_log" 2>&1

  local last_ckpt="${stage_dir}/logs/lightning_logs/version_0/checkpoints/last.ckpt"
  if [ ! -f "$last_ckpt" ]; then
    echo "[recovery] FATAL: stage ${stage_name} did not produce last.ckpt at ${last_ckpt}" >&2
    exit 2
  fi
  echo "[recovery] stage=${stage_name} OK → ${last_ckpt}"
  echo "$last_ckpt"
}

# Stage 1 — m=6 warm-up from v4 Ep5
m6_ckpt=$(run_stage m6 6 "$EPOCHS_M6" "$START_CKPT" | tail -1)

# Stage 2 — m=5 from m6 last.ckpt
m5_ckpt=$(run_stage m5 5 "$EPOCHS_M5" "$m6_ckpt" | tail -1)

# Stage 3 — m=4 final from m5 last.ckpt
m4_ckpt=$(run_stage m4 4 "$EPOCHS_M4" "$m5_ckpt" | tail -1)

echo ""
echo "[recovery] curriculum complete. final ckpt = $m4_ckpt"

if [ "$SKIP_EVAL" = "1" ]; then
  echo "[recovery] SKIP_EVAL=1 set, skipping post-stage-3 full eval."
  exit 0
fi

# Full eval (VCTK 750 + TIMIT 160) on final m=4 ckpt
echo ""
echo "==================================================================="
echo "[recovery] full eval — VCTK 750 + TIMIT 160 on m=4 final ckpt"
echo "==================================================================="
eval_log="${LOG_DIR}/eval.log"
CKPT="$m4_ckpt" \
OUT="$EVAL_DIR" \
GPU_ID="$GPU_ID" \
LEARNABLE_BC=1 \
bash "${REPO}/scripts/run_kd_full_eval.sh" > "$eval_log" 2>&1 || {
  echo "[recovery] eval wrapper failed; see $eval_log" >&2
  exit 3
}

echo ""
echo "[recovery] DONE."
echo "  Final ckpt:    $m4_ckpt"
echo "  Eval results:  $EVAL_DIR"
echo "  Eval log:      $eval_log"
echo ""
echo "Compare to gate: VCTK 0.8354 / TIMIT 0.7880"
grep -E "PAPER STYLE|Average MBSTOI Enhanced|Average PESQ Enhanced" "$eval_log" | head -20 || true
