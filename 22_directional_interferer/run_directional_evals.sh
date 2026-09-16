#!/bin/bash
# Score the binaural deploy model and the mono baselines on the directional-
# interferer condition, on the same clean targets and SNR grid as the isotropic
#考卷 -- so the only variable is whether the interferer is diffuse or lateral.
#
# Strictly sequential: concurrent GPU jobs get SIGTERMed on this box
# (see 21_mono_se_baselines/run_full_pipeline.sh).
#
#   GPU_ID=3 bash run_directional_evals.sh            # both conditions
#   COND=dirnoise90 GPU_ID=3 bash run_directional_evals.sh
set -uo pipefail

PKG=/raid/R12K41024/22_directional_interferer
DEP=/raid/R12K41024/18_w4a8_pow2_mixed_lmul_hw_handoff_20260530
MONO=/raid/R12K41024/21_mono_se_baselines
DATA=/raid/R12K41024/BCCTN/Dataset
GPU="${GPU_ID:-3}"
# dirspeech2x60 first: it mirrors Westhausen et al. (TASLP 2024) S0N+-60,
# the configuration where better-ear listening cannot carry the score.
CONDS="${COND:-dirspeech2x60 dirnoise2x60 dirnoise90 dirspeech90}"

for cond in $CONDS; do
  TEST_DIR="$DATA/noisy_testset_$cond"
  if [ ! -d "$TEST_DIR" ]; then echo "SKIP $cond (no $TEST_DIR)"; continue; fi

  echo "=== [$cond] deploy binaural W4A8 ==="
  CKPT="$DEP/ckpt/strict_core_pow2_blockfloat_deploy_ep3_vctk0.8357_timit0.7898.ckpt" \
  TAG="${cond}_deploy" VCTK_ONLY=1 ACT_FP8_MODE=calibrated_pow2_e4m3 \
  BCCTN_ACTIVATION_CALIB_PATH="$DEP/ckpt/activation_calibration_p999_b64.json" \
  VCTK_TEST_DIR="$TEST_DIR" GPU_ID="$GPU" \
    bash "$DEP/run_apot_descale_eval.sh"
  echo "DEPLOY_${cond}_EXIT=$?"

  for variant in shared independent; do
    mm=independent; [ "$variant" = shared ] && mm=shared
    echo "=== [$cond] NTU mono, $variant mask ==="
    CKPT="$MONO/results/ntu/best_ntu.pt" TAG="${cond}_ntu_${variant}" \
    VCTK_ONLY=1 MASK_MODE="$mm" VCTK_TEST_DIR="$TEST_DIR" GPU_ID="$GPU" \
      bash "$MONO/run_bilateral_eval.sh"
    echo "NTU_${variant}_${cond}_EXIT=$?"
  done
done
echo "DIRECTIONAL_EVALS_DONE"
