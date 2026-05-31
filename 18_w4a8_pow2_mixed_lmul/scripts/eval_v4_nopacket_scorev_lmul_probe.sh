#!/usr/bin/env bash
# Cold eval: remove prob_packet, keep Score x V on dense L-Mul.
# This tests the hardware point "skip packetization if L-Mul PE is cheap enough".
set -euo pipefail

REPO="${REPO:-/raid/R12K41024/15_all_lmul_mixed_mult_eval}"
CKPT="${CKPT:-${REPO}/outputs/all_lmul_mixed_qat/all_lmul_m4_bitexact_qat_2026-05-24_02-49-17/logs/lightning_logs/version_0/checkpoints/epoch=5-step=1452.ckpt}"
P2_INDEX="${P2_INDEX:-${REPO}/Checkpoints/p2_skip_indices_g42_topk_l1_r050.pt}"
OUT="${OUT:-${REPO}/results/v4_nopacket_scorev_lmul_probe/epoch5_step1452_w750}"
GPU_ID="${GPU_ID:-0}"
LIMIT="${LIMIT:-750}"
METRIC_WORKERS="${METRIC_WORKERS:-32}"

cd "$REPO"
export PYTHONPATH="${PYTHONPATH:-}:${REPO}"
export HYDRA_FULL_ERROR=1
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl_cache}"

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
  model.attention_datapath.lmul_targets=[qk,score_v]
  model.attention_datapath.lmul_mantissa_bits=4
  model.attention_datapath.lmul_bit_exact=true
  model.attention_datapath.lmul_all_projection_ops=true
  model.attention_datapath.lmul_linear_chunk_size=2048
  model.attention_datapath.lmul_checkpoint_chunks=false
  model.attention_datapath.prob_packet.enabled=false
  model.decoder_grouped_pointwise.mode=grouped
  model.decoder_grouped_pointwise.groups_x=4
  model.decoder_grouped_pointwise.groups_skip=2
  model.decoder_grouped_pointwise.skip_select_ratio=0.5
  model.decoder_grouped_pointwise.skip_select_mode=topk_l1
  model.decoder_grouped_pointwise.skip_indices_path="$P2_INDEX"
  model.low_power.mha_projection.enabled=true
  model.low_power.mha_projection.qkv_groups=4
  model.low_power.all_lmul_multiply.enabled=true
  model.low_power.all_lmul_multiply.mantissa_bits=4
  model.low_power.all_lmul_multiply.attention_mantissa_bits=4
  model.low_power.all_lmul_multiply.bit_exact=true
  model.low_power.all_lmul_multiply.linear_chunk_size=2048
  model.low_power.all_lmul_multiply.checkpoint_chunks=false
)

echo "[eval] ckpt=${CKPT}"
echo "[eval] out=${OUT}"
echo "[eval] GPU_ID=${GPU_ID}, limit=${LIMIT}, metric_workers=${METRIC_WORKERS}"

CUDA_VISIBLE_DEVICES="$GPU_ID" conda run --no-capture-output -n bcctn python eval.py \
  --model_checkpoint "$CKPT" \
  --vctk_test_dir /raid/R12K41024/BCCTN/Dataset/noisy_testset \
  --vctk_clean_dir /raid/R12K41024/BCCTN/Dataset/clean_testset \
  --timit_test_dir /raid/R12K41024/BCCTN/Dataset/noisy_testset_timit \
  --timit_clean_dir /raid/R12K41024/BCCTN/Dataset/clean_testset_timit \
  --output_dir "$OUT" \
  --paper_style_eval \
  --batch_size 1 \
  --limit_pairs "$LIMIT" \
  --metric_workers "$METRIC_WORKERS" \
  --hydra_overrides "${overrides[@]}"
