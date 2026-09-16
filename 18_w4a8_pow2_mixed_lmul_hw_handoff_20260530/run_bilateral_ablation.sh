#!/usr/bin/env bash
# Experiment A (weight-shared bilateral ablation) + C (ITD-based localization)
# on the deploy model, using the TRUE §3.4 hardware path (same env + overrides
# as run_apot_descale_eval.sh).
#
#   GPU_ID=<free> [LIMIT=600] [WORKERS=48] bash run_bilateral_ablation.sh
set -euo pipefail

PKG=/raid/R12K41024/18_w4a8_pow2_mixed_lmul_hw_handoff_20260530
cd "$PKG/code"

CKPT="${CKPT:-${PKG}/ckpt/strict_core_pow2_blockfloat_deploy_ep3_vctk0.8357_timit0.7898.ckpt}"
P2_INDEX="${P2_INDEX:-${PKG}/ckpt/p2_skip_indices_g42_topk_l1_r050.pt}"
GPU_ID="${GPU_ID:-4}"
LIMIT="${LIMIT:-600}"
WORKERS="${WORKERS:-48}"
TOL="${TOL:-10}"
TAG="${TAG:-bilateral_ablation}"
# Point at a different acoustic condition (22_directional_interferer builds them);
# "isotropic" is the original考卷.
COND="${COND:-isotropic}"
if [ "$COND" = isotropic ]; then
  TEST_DIR="/raid/R12K41024/BCCTN/Dataset/noisy_testset"
else
  TEST_DIR="/raid/R12K41024/BCCTN/Dataset/noisy_testset_${COND}"
fi
OUT="${OUT:-${PKG}/results/${TAG}}"
LOG="${LOG:-${OUT}/run.log}"
mkdir -p "$OUT"

export PYTHONPATH="${PKG}/code"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl_cache}"
export BCCTN_APOT_A_MIN=-16; export BCCTN_APOT_A_MAX=-1
export BCCTN_APOT_B_MIN=-16; export BCCTN_APOT_B_MAX=-1
export BCCTN_ACTIVATION_FP8_MODE=calibrated_pow2_e4m3
export BCCTN_ACTIVATION_CALIB_PATH="${PKG}/ckpt/activation_calibration_p999_b64.json"
export BCCTN_OUTLIER_DELAYED_DESCALE=1
export BCCTN_OUTLIER_DESCALE_APOT_MIN=-16; export BCCTN_OUTLIER_DESCALE_APOT_MAX=-6

overrides=(
  training.qat_scope=full
  training.top_k_attention=4
  training.fp16_accumulator=true
  training.int4_scale_format=apot_clip
  model.attention_softmax.mode=shift_topk_pow2_clamp8_nomantissa
  model.attention_datapath.enabled=true
  model.attention_datapath.score_quant=none
  model.attention_datapath.prob_quant=none
  model.attention_datapath.lmul_enabled=true
  model.attention_datapath.lmul_targets=[qk]
  model.attention_datapath.lmul_bit_exact=true
  model.attention_datapath.lmul_mantissa_bits=8
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
  model.low_power.outlier_fp8_lmul.mantissa_bits=4
  model.low_power.bn_affine.scale_mode=fold_apot
  model.low_power.bn_affine.pow2_min_exp=-9
  model.low_power.bn_affine.pow2_max_exp=4
  model.low_power.bn_affine.tie_real_imag_scale=false
  model.low_power.prelu_pow2.enabled=true
  model.low_power.prelu_pow2.min_exp=-8
  model.low_power.prelu_pow2.max_exp=0
)

echo "[ablation] ckpt=$CKPT gpu=$GPU_ID limit=$LIMIT out=$OUT"

CUDA_VISIBLE_DEVICES="$GPU_ID" \
"${PYBIN:-/home/R12K41024/miniforge3/envs/bcctn/bin/python}" eval_bilateral_ablation.py \
  --model_checkpoint "$CKPT" \
  --test_dir "$TEST_DIR" \
  --clean_dir /raid/R12K41024/BCCTN/Dataset/clean_testset \
  --output_dir "$OUT" \
  --limit_pairs "$LIMIT" \
  --metric_workers "$WORKERS" \
  --tol_deg "$TOL" \
  --hydra_overrides "${overrides[@]}" \
  2>&1 | tee "$LOG"
