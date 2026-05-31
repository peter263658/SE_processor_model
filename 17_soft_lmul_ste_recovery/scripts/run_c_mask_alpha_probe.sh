#!/bin/bash
# Probe C: mask-level per-frequency-bin alpha calibration on Ep3 ckpt.
# Step 1: calibrate alpha from N=30 mid-SNR VCTK samples
# Step 2: eval on VCTK w=200 with calibrated alpha applied at output STFT
set -euo pipefail

REPO=/raid/R12K41024/15_all_lmul_mixed_mult_eval
CKPT="${REPO}/outputs/all_lmul_mixed_qat/all_lmul_m4_bitexact_qat_2026-05-22_14-13-49/logs/lightning_logs/version_0/checkpoints/epoch=3-step=968.ckpt"
P2_INDEX="${REPO}/Checkpoints/p2_skip_indices_g42_topk_l1_r050.pt"
OUT_DIR="${REPO}/results/c_mask_alpha_probe"
ALPHA_PATH="${OUT_DIR}/alpha.npy"
GPU_ID="${GPU_ID:-3}"
LIMIT="${LIMIT:-200}"
METRIC_WORKERS="${METRIC_WORKERS:-16}"

cd "$REPO"
export PYTHONPATH="${PYTHONPATH:-}:${REPO}"
export HYDRA_FULL_ERROR=1
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl_cache}"

mkdir -p "$OUT_DIR"

# v3-style overrides (Ep3 ckpt's training setup)
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
  model.decoder_grouped_pointwise.skip_indices_path="$P2_INDEX"
  model.low_power.mha_projection.enabled=true
  model.low_power.mha_projection.qkv_groups=4
  model.low_power.all_lmul_multiply.enabled=true
  model.low_power.all_lmul_multiply.mantissa_bits=4
  model.low_power.all_lmul_multiply.bit_exact=true
  model.low_power.all_lmul_multiply.linear_chunk_size=2048
  model.low_power.all_lmul_multiply.checkpoint_chunks=false
)

echo ""
echo "==============================================="
echo "[C probe] Step 1: calibrate alpha (N=30 samples)"
echo "==============================================="
CUDA_VISIBLE_DEVICES="$GPU_ID" conda run --no-capture-output -n bcctn python scripts/calibrate_mask_alpha.py \
  --ckpt "$CKPT" \
  --noisy_dir /raid/R12K41024/BCCTN/Dataset/noisy_testset \
  --clean_dir /raid/R12K41024/BCCTN/Dataset/clean_testset \
  --num_samples 30 \
  --snr_filter 0 3 \
  --n_fft 128 \
  --hop 80 \
  --output "$ALPHA_PATH" \
  --hydra_overrides "${overrides[@]}"

echo ""
echo "==============================================="
echo "[C probe] Step 2: eval with calibrated alpha"
echo "==============================================="
BCCTN_MASK_ALPHA_PATH="$ALPHA_PATH" \
CUDA_VISIBLE_DEVICES="$GPU_ID" conda run --no-capture-output -n bcctn python eval.py \
  --model_checkpoint "$CKPT" \
  --vctk_test_dir /raid/R12K41024/BCCTN/Dataset/noisy_testset \
  --vctk_clean_dir /raid/R12K41024/BCCTN/Dataset/clean_testset \
  --output_dir "${OUT_DIR}/eval_with_alpha" \
  --paper_style_eval \
  --batch_size 1 \
  --limit_pairs "$LIMIT" \
  --metric_workers "$METRIC_WORKERS" \
  --hydra_overrides "${overrides[@]}"

echo ""
echo "[C probe] done. results under ${OUT_DIR}/"
