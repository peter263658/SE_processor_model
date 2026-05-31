#!/usr/bin/env bash
# m4-direct lmul_exact probe — the CLEAN backward diagnosis.
#
# Question this answers (and ONLY this): can the *correct* L-Mul backward
# (STE_MODE=lmul_exact) move validation_loss off the ~-13.1 plateau that the
# naive real-mult STE (16/ curriculum scout) got stuck on?
#
# Design (everything except the two deltas matches the 16/ scout exactly):
#   - delta 1: STE_MODE = lmul_exact   (vs scout's "real")
#   - delta 2: m4-direct               (vs scout's m6->m5->m4 curriculum)
#   forward path, c3 packet+L-Mul stack, d1 qkv_g4, grouped decoder, binaural
#   KD (lambda=0.0528721747), learnable_bc — all identical to the scout, so a
#   result is attributable to the STE change, not a forward/objective change.
#   (delayed-scale / scale-LMul forward changes are a SEPARATE later experiment.)
#
# Fast first signal: defaults to limit_train_batches=0.5 (half-epoch train ~1h)
# + limit_val_batches=1.0 (full, trustworthy val ~0.5h) => ~1.5h to first val,
# vs ~2.5h for a full epoch. Needs the BaseTrainer limit_* plumbing (prep A).
#
# DRAFT — not auto-launched. Review, then run manually:
#   STE_MODE=lmul_exact GPU_ID=<free> bash scripts/run_m4_direct_probe.sh
# Ablation (prove "direction > magnitude"): STE_MODE=grad_scale_0.67 ...
set -euo pipefail

REPO=/raid/R12K41024/17_soft_lmul_ste_recovery
cd "$REPO"

# --- knobs (env-overridable) ---------------------------------------------
# Default start = the SAME ckpt the 16/ scout started from (15/ all-L-Mul m4
# bit-exact QAT, epoch=5), so probe vs scout share an identical start and
# differ only by the STE + curriculum-skip. Point START_CKPT at the scout's
# stuck m4 last.ckpt instead if you prefer "continue from the plateau".
START_CKPT="${START_CKPT:-/raid/R12K41024/15_all_lmul_mixed_mult_eval/outputs/all_lmul_mixed_qat/all_lmul_m4_bitexact_qat_2026-05-24_02-49-17/logs/lightning_logs/version_0/checkpoints/epoch=5-step=1452.ckpt}"
TEACHER_DIR="${TEACHER_DIR:-${REPO}/teacher_outputs/d1c3_epoch25_step100672_fp16}"
TEACHER_ROOT="${TEACHER_ROOT:-/raid/R12K41024/BCCTN/Dataset/noisy_trainset}"
P2_INDEX="${P2_INDEX:-${REPO}/Checkpoints/p2_skip_indices_g42_topk_l1_r050.pt}"

STE_MODE="${STE_MODE:-lmul_exact}"       # the whole point; "real"/"grad_scale_0.67" for ablation
MANTISSA="${MANTISSA:-4}"                 # m4-direct, no curriculum
N_EPOCHS="${N_EPOCHS:-1}"
LIMIT_TRAIN="${LIMIT_TRAIN:-0.5}"         # half-epoch train for a fast first signal
LIMIT_VAL="${LIMIT_VAL:-1.0}"            # full val => trustworthy plateau read
KD_LAMBDA="${KD_LAMBDA:-0.0528721747}"    # same binaural-KD weight as the scout
LR="${LR:-5e-6}"
BATCH_SIZE="${BATCH_SIZE:-128}"           # compute-bound: bigger buys nothing (smoke_throughput)
GPU_ID="${GPU_ID:-3}"
RUN_TAG="${RUN_TAG:-m4direct_${STE_MODE}_$(date +%Y%m%d_%H%M%S)}"

# --- preflight ------------------------------------------------------------
for f in "$START_CKPT" "$P2_INDEX" "$TEACHER_DIR" "$TEACHER_ROOT"; do
  [ -e "$f" ] || { echo "[m4-direct] FATAL: missing $f" >&2; exit 1; }
done

OUT_DIR="${REPO}/outputs/m4_direct_probe/${RUN_TAG}"
LOG_DIR="${REPO}/logs/m4_direct_probe/${RUN_TAG}"
mkdir -p "$OUT_DIR" "$LOG_DIR"

export PYTHONPATH="${PYTHONPATH:-}:${REPO}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl_cache}"

echo "[m4-direct] RUN_TAG    = $RUN_TAG"
echo "[m4-direct] START_CKPT = $START_CKPT"
echo "[m4-direct] STE_MODE   = $STE_MODE  (delta vs scout)   mantissa=$MANTISSA (m4-direct)"
echo "[m4-direct] epochs=$N_EPOCHS  limit_train=$LIMIT_TRAIN  limit_val=$LIMIT_VAL"
echo "[m4-direct] KD=binaural lambda=$KD_LAMBDA  LR=$LR  BS=$BATCH_SIZE  GPU=$GPU_ID"
echo "[m4-direct] OUT_DIR    = $OUT_DIR"

# --- overrides: identical to run_combined_recovery.sh common_overrides ----
common_overrides=(
  training.qat_scope=full
  training.top_k_attention=4
  training.fp16_accumulator=true
  training.int4_scale_format=fp32
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
  model.attention_datapath.lmul_mantissa_bits="$MANTISSA"
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
  model.low_power.all_lmul_multiply.ste_mode="$STE_MODE"
  model.low_power.all_lmul_multiply.mantissa_bits="$MANTISSA"
  model.low_power.all_lmul_multiply.attention_mantissa_bits="$MANTISSA"
  model.kd.enabled=true
  model.kd.kd_mode=binaural
  model.kd.lambda_kd="$KD_LAMBDA"
  "model.kd.teacher_output_dir=$TEACHER_DIR"
  "model.kd.teacher_noisy_root=$TEACHER_ROOT"
)

stage_dir="${OUT_DIR}/m4"
CUDA_VISIBLE_DEVICES="$GPU_ID" conda run --no-capture-output -n bcctn \
  python -u train.py \
    "training.train_checkpoint_path='${START_CKPT}'" \
    training.resume_ckpt_path=null \
    training.n_epochs="$N_EPOCHS" \
    "${common_overrides[@]}" \
    "hydra.run.dir=${stage_dir}" \
    2>&1 | tee "${LOG_DIR}/stage_m4.log"

last_ckpt="${stage_dir}/logs/lightning_logs/version_0/checkpoints/last.ckpt"
[ -f "$last_ckpt" ] || { echo "[m4-direct] FATAL: no last.ckpt at $last_ckpt" >&2; exit 2; }

echo ""
echo "[m4-direct] done -> $last_ckpt"
echo "[m4-direct] READ THE SIGNAL:  grep 'New best score' ${LOG_DIR}/stage_m4.log"
echo "[m4-direct]   val_loss clearly < -13.1  -> lmul_exact escapes the plateau; extend (raise N_EPOCHS/LIMIT_TRAIN)."
echo "[m4-direct]   val_loss still ~-13.0      -> plateau is a real L-Mul local min; curriculum won't help either."
echo "[m4-direct] For MBSTOI, run the existing full-eval script on last.ckpt only if val_loss moved."
