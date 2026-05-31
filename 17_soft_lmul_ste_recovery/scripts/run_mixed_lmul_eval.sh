#!/usr/bin/env bash
# Mixed-precision L-Mul EVAL-ONLY sweep on the 14/ d1_qkv_g4 + c3 baseline.
#
# Question: was "uniform m=4 everywhere" (15/16/17) artificially inflating the
# full-L-Mul drop by crushing the attention path? Test format-faithful mixed
# precision (attention m=10, conv m=4) ZERO-SHOT (no fine-tune) on the
# gate-passing production ckpt, and compare to the known points.
#
# Reference points (already measured):
#   A0  d1c3 baseline (attn qk/score_v L-Mul m4, conv REAL):  VCTK 0.8354 / TIMIT 0.7880
#   17/ full-L-Mul UNIFORM m4 (+fine-tune):                   <- the eval running now
#
# Experiments here (eval-only, warm ckpt = d1_qkv_g4 epoch=25):
#   A1  attn m=10, conv REAL          -> control: is attention precision the issue?
#   A2  attn m=10, conv L-Mul m=4     -> your test: faithful mixed, full multiplierless
#
# A2 == "qk/pv m=10, conv m=4" (qk & score_v share lmul_mantissa_bits, so =10 sets
# both; conv/linear use all_lmul mantissa_bits=4). No new code needed for this.
# (A true qk!=pv split, and B1 INT4 shift-add + delayed scale, need code — separate.)
#
# DRAFT — not auto-launched. Review, then:
#   EXP=A1 GPU_ID=<free> bash scripts/run_mixed_lmul_eval.sh
#   EXP=A2 GPU_ID=<free> bash scripts/run_mixed_lmul_eval.sh
set -euo pipefail

REPO=/raid/R12K41024/17_soft_lmul_ste_recovery
cd "$REPO"

EXP="${EXP:?EXP is required (A1 | A2)}"
# Gate-passing production ckpt = the KD teacher's base = "before full L-Mul".
CKPT="${CKPT:-/raid/R12K41024/14_low_power_arch_qat/outputs/low_power_phase_d/d1_qkv_g4_2026-05-20_02-56-03/logs/lightning_logs/version_0/checkpoints/epoch=25-step=100672.ckpt}"
# Use the SAME skip-indices file the d1 ckpt trained with (14/ original).
P2_INDEX="${P2_INDEX:-/raid/R12K41024/14_low_power_arch_qat/Checkpoints/p2_skip_indices_g42_topk_l1_r050.pt}"
GPU_ID="${GPU_ID:-3}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LIMIT="${LIMIT:-750}"
METRIC_WORKERS="${METRIC_WORKERS:-32}"
ATTN_M="${ATTN_M:-10}"          # attention (qk + score_v) L-Mul mantissa
CONV_M="${CONV_M:-4}"           # conv/linear L-Mul mantissa (A2 only)

for f in "$CKPT" "$P2_INDEX"; do
  [ -f "$f" ] || { echo "[mixed-eval] FATAL: missing $f" >&2; exit 1; }
done

OUT="${OUT:-${REPO}/results/mixed_lmul_eval/${EXP}_attn${ATTN_M}_conv${CONV_M}}"
LOG="${LOG:-${REPO}/logs/mixed_lmul_eval/${EXP}_attn${ATTN_M}_conv${CONV_M}_gpu${GPU_ID}.log}"
mkdir -p "$OUT" "$(dirname "$LOG")"
export PYTHONPATH="${PYTHONPATH:-}:${REPO}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl_cache}"

# --- d1 architecture + c3 datapath (authoritative, from 14/ overrides.yaml) ---
overrides=(
  training.qat_scope=full
  training.top_k_attention=4
  training.fp16_accumulator=true
  training.int4_scale_format=fp32
  model.attention_softmax.mode=shift_topk_pow2_clamp8_nomantissa
  model.attention_datapath.enabled=true
  model.attention_datapath.score_quant=none
  model.attention_datapath.prob_quant=none
  model.attention_datapath.lmul_enabled=true
  model.attention_datapath.lmul_targets=[qk]
  model.attention_datapath.lmul_bit_exact=true
  model.attention_datapath.lmul_mantissa_bits="$ATTN_M"   # qk + score_v share this
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
)

case "$EXP" in
  A1)  # attention m=ATTN_M, conv REAL multiply (all_lmul off)
    overrides+=( model.low_power.all_lmul_multiply.enabled=false ) ;;
  A2)  # attention m=ATTN_M, conv/linear L-Mul m=CONV_M (full multiplierless, mixed precision)
    overrides+=(
      model.low_power.all_lmul_multiply.enabled=true
      model.low_power.all_lmul_multiply.mantissa_bits="$CONV_M"
      model.low_power.all_lmul_multiply.attention_mantissa_bits="$ATTN_M"
      model.low_power.all_lmul_multiply.bit_exact=true
      model.low_power.all_lmul_multiply.linear_chunk_size=2048
      model.low_power.all_lmul_multiply.checkpoint_chunks=false
      model.low_power.all_lmul_multiply.learnable_bias_correction.enabled=false
    ) ;;
  *) echo "[mixed-eval] FATAL: EXP must be A1 or A2 (got $EXP)" >&2; exit 1 ;;
esac

echo "[mixed-eval] EXP=$EXP  attn_m=$ATTN_M  conv_m=$CONV_M (A2 only)"
echo "[mixed-eval] ckpt=$CKPT"
echo "[mixed-eval] out=$OUT   gpu=$GPU_ID  limit=$LIMIT"
echo "[mixed-eval] baseline to beat: A0 d1c3 = VCTK 0.8354 / TIMIT 0.7880"

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
echo "[mixed-eval] done. Compare 'Average MBSTOI Enhanced' to A0 0.8354 / TIMIT 0.7880:"
grep -E "Average MBSTOI Enhanced" "$OUT"/*_averages.txt 2>/dev/null || true
