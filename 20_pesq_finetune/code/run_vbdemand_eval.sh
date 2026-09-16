#!/usr/bin/env bash
# VoiceBank+DEMAND eval for the 20_pesq_finetune (PESQ-finetuned) model.
# Binaural-renders the EXISTING standard mono VB+DEMAND clean+noisy through an
# HRIR (no re-mix) -> 2ch, runs the deploy W4A8 datapath, reports per-ear
# nb+wb PESQ / STOI / ESTOI / SI-SDR. Same deploy overrides as run_pesq_eval.sh
# so numbers are consistent with the VCTK paper-style eval.
#
# input_mode:
#   binaural_az0 : HRIR-render mono->2ch (in-distribution; HRIR-colored, NOT
#                  standard-mono-comparable). Azimuth = whichever --hrir file.
#   mono_dup     : L=R true mono (directly comparable to mono SE papers / P2).
#
#   CKPT=<ckpt> [INPUT_MODE=binaural_az0] [HRIR=<az wav>] GPU_ID=<one> bash run_vbdemand_eval.sh
set -euo pipefail

CODE_DIR=/raid/R12K41024/20_pesq_finetune/code
DEPS=/raid/R12K41024/18_w4a8_pow2_mixed_lmul
cd "$CODE_DIR"

CKPT="${CKPT:?set CKPT}"
P2_INDEX="${P2_INDEX:-/raid/R12K41024/14_low_power_arch_qat/Checkpoints/p2_skip_indices_g42_topk_l1_r050.pt}"
VB=/raid/R12K41024/Dataset/VoiceBank_DEMAND
CLEAN_DIR="${CLEAN_DIR:-$VB/clean_testset_wav}"
NOISY_DIR="${NOISY_DIR:-$VB/noisy_testset_wav}"
# anechoic, front (az 0), ear-level, 80cm — matches the binaural training render
# (BCCTN dataset used hrir/anechoic) and is anechoic like the P2 protocol.
HRIR="${HRIR:-/raid/R12K41024/Dataset/HRIR_database_wav/hrir/anechoic/anechoic_distcm_80_el_0_az_0.wav}"
INPUT_MODE="${INPUT_MODE:-binaural_az0}"
TAG="${TAG:-vbdemand}"
GPU_ID="${GPU_ID:-0}"
LIMIT="${LIMIT:-0}"          # 0 = all 824
# CPU metric workers. Shared box: keep <=32 per eval, <=80 total across parallel
# evals (so 4-way parallel -> set ~20 each).
METRIC_WORKERS="${METRIC_WORKERS:-24}"
CONDA_BIN="${CONDA_BIN:-/home/R12K41024/miniforge3/bin/conda}"

ATTN_M="${ATTN_M:-8}"; OUTLIER_M="${OUTLIER_M:-4}"; SCALE_FMT="${SCALE_FMT:-apot_clip}"
export BCCTN_APOT_A_MIN=-16; export BCCTN_APOT_A_MAX=-1
export BCCTN_APOT_B_MIN=-16; export BCCTN_APOT_B_MAX=-1

for f in "$CKPT" "$P2_INDEX" "$HRIR" "$CLEAN_DIR" "$NOISY_DIR"; do
  [ -e "$f" ] || { echo "[vbd] FATAL: missing $f" >&2; exit 1; }
done

OUT="/raid/R12K41024/20_pesq_finetune/results/voicebank_demand/${TAG}"
LOG="/raid/R12K41024/20_pesq_finetune/logs/voicebank_demand/${TAG}_gpu${GPU_ID}.log"
mkdir -p "$OUT" "$(dirname "$LOG")"
export PYTHONPATH="${CODE_DIR}:${PYTHONPATH:-}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl_cache}"

overrides=(
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
  model.low_power.bn_affine.scale_mode=fold_apot
  model.low_power.bn_affine.pow2_min_exp=-9
  model.low_power.bn_affine.pow2_max_exp=4
  model.low_power.bn_affine.tie_real_imag_scale=false
  model.low_power.prelu_pow2.enabled=true
  model.low_power.prelu_pow2.min_exp=-8
  model.low_power.prelu_pow2.max_exp=0
)

echo "[vbd] TAG=$TAG ckpt=$CKPT"
echo "[vbd] mode=$INPUT_MODE hrir=$HRIR gpu=$GPU_ID limit=$LIMIT"
CUDA_VISIBLE_DEVICES="$GPU_ID" "$CONDA_BIN" run --no-capture-output -n bcctn python eval_voicebank_demand.py \
  --clean_dir "$CLEAN_DIR" \
  --noisy_dir "$NOISY_DIR" \
  --hrir "$HRIR" \
  --ckpt "$CKPT" \
  --out "$OUT" \
  --limit "$LIMIT" \
  --metric_workers "$METRIC_WORKERS" \
  --input_mode "$INPUT_MODE" \
  --hydra_overrides "${overrides[@]}" \
  2>&1 | tee "$LOG"

echo ""; echo "[vbd] $TAG averages:"; cat "$OUT/vbdemand_averages.txt" 2>/dev/null
