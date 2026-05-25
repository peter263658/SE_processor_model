#!/bin/bash
# Baseline eval: re-run the 14_low_power_arch_qat production stack
# (d1_qkv_g4 + c3_packet_lmul_bitexact) under THIS repo's eval pipeline.
# Purpose: confirm that the 0.8354 / 0.7880 gate is reproducible under
# 15_all_lmul_mixed_mult_eval's wrapper / pair selection / SNR averaging.
# Conv stays as real multiplier (NOT all-L-Mul).
set -euo pipefail

REPO=/raid/R12K41024/15_all_lmul_mixed_mult_eval
CKPT=/raid/R12K41024/14_low_power_arch_qat/outputs/low_power_phase_d/d1_qkv_g4_2026-05-20_02-56-03/logs/lightning_logs/version_0/checkpoints/epoch=25-step=100672.ckpt
P2_INDEX="${REPO}/Checkpoints/p2_skip_indices_g42_topk_l1_r050.pt"
OUT="${REPO}/results/all_lmul_mixed_qat_full/baseline_d1c3_under_15pipeline_w750"
GPU_ID="${GPU_ID:-1}"
LIMIT="${LIMIT:-200}"
METRIC_WORKERS="${METRIC_WORKERS:-32}"

cd "$REPO"
export PYTHONPATH="${PYTHONPATH:-}:${REPO}"
export HYDRA_FULL_ERROR=1
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl_cache}"

# 14 d1+c3 overrides: MHA-only L-Mul (qk + packet score_v), grouped MHA, packet encoding.
# Conv stays a real multiplier (all_lmul_multiply DISABLED, lmul_all_projection_ops DISABLED).
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
  model.attention_datapath.lmul_mantissa_bits=4
  model.attention_datapath.lmul_bit_exact=true
  model.attention_datapath.lmul_all_projection_ops=false
  model.attention_datapath.lmul_linear_chunk_size=0
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
  model.decoder_grouped_pointwise.skip_indices_path="$P2_INDEX"
  model.low_power.mha_projection.enabled=true
  model.low_power.mha_projection.qkv_groups=4
  model.low_power.all_lmul_multiply.enabled=false
)

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
