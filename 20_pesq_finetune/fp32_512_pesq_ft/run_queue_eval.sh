#!/usr/bin/env bash
# Sequential eval queue on ONE GPU (default 6), designed to run inside tmux so
# it survives Claude session teardown. Order:
#   B) FP32-512 original-reference full eval (VCTK+TIMIT, new metrics)
#   C) pow2 recovery-FT ep2 / ep4 VCTK evals (checkpoint re-selection close-out)
#   D) deploy zero-shot at synthetic -8/-10 dB (VCTK + TIMIT)
set -uo pipefail

GPU="${GPU:-6}"
STAMP="$(date +%H%M%S)"
QLOG=/raid/R12K41024/20_pesq_finetune/fp32_512_pesq_ft/queue_${STAMP}.log
exec >>"$QLOG" 2>&1
echo "=== eval queue start $(date) on GPU $GPU ==="

PKG=/raid/R12K41024/18_w4a8_pow2_mixed_lmul_hw_handoff_20260530
export PATH=/home/R12K41024/miniforge3/bin:$PATH
export NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/mpl_cache

# ---- C: ep2 / ep4 re-selection close-out (VCTK, new metrics) --------------
CKROOT=$PKG/outputs/apot_descale_finetune/pow2_act_from_calibep2_5ep_lr1e-5_20260531/ft/logs/lightning_logs/version_0/checkpoints
for EP in 2 4; do
  CK=$(ls $CKROOT/epoch=${EP}-step=*.ckpt | head -1)
  echo "--- [C] pow2 ep${EP} VCTK eval: $CK $(date) ---"
  CKPT="$CK" TAG="pow2_ep${EP}_vctk_stoi" VCTK_ONLY=1 GPU_ID=$GPU \
    ACT_FP8_MODE=calibrated_pow2_e4m3 \
    BCCTN_ACTIVATION_CALIB_PATH=$PKG/ckpt/activation_calibration_p999_b64.json \
    bash $PKG/run_apot_descale_eval.sh
  echo "--- [C] ep${EP} exit=$? $(date) ---"
done

# ---- D: deploy zero-shot at -8 / -10 dB ------------------------------------
EXT=/raid/R12K41024/20_pesq_finetune/data/lowsnr_ext
DOUT=/raid/R12K41024/20_pesq_finetune/results/pesq_eval/deploy_lowsnr_ext
for DS in vctk timit; do
  if [ "$DS" = vctk ]; then CLEAN=/raid/R12K41024/BCCTN/Dataset/clean_testset
  else CLEAN=/raid/R12K41024/BCCTN/Dataset/clean_testset_timit; fi
  for SNR in -8 -10; do
    echo "--- [D] deploy $DS snr ${SNR}dB $(date) ---"
    TEST_DIR=$EXT/$DS/snr_${SNR}dB CLEAN_DIR=$CLEAN \
      OUT=$DOUT/${DS}_snr${SNR}dB GPU_ID=$GPU \
      bash $PKG/run_lowsnr_ext_eval.sh
    echo "--- [D] $DS ${SNR}dB exit=$? $(date) ---"
  done
done

# ---- B: FP32-512 reference full eval -------------------------------------
S=/raid/R12K41024/20_pesq_finetune/fp32_512_pesq_ft/code
cd "$S"
echo "--- [B] FP32-512 reference full eval $(date) ---"
CUDA_VISIBLE_DEVICES=$GPU conda run --no-capture-output -n bcctn python eval.py \
  --model_checkpoint /raid/R12K41024/BCCTN/outputs/2025-05-10/19-39-07/logs/lightning_logs/version_0/checkpoints/epoch=98-step=95832.ckpt \
  --vctk_test_dir /raid/R12K41024/BCCTN/Dataset/noisy_testset \
  --vctk_clean_dir /raid/R12K41024/BCCTN/Dataset/clean_testset \
  --timit_test_dir /raid/R12K41024/BCCTN/Dataset/noisy_testset_timit \
  --timit_clean_dir /raid/R12K41024/BCCTN/Dataset/clean_testset_timit \
  --output_dir /raid/R12K41024/20_pesq_finetune/results/pesq_eval/fp32_512_ref_full \
  --paper_style_eval --batch_size 8 --limit_pairs 750
echo "--- [B] exit=$? $(date) ---"

echo "=== eval queue COMPLETE $(date) ==="
touch /raid/R12K41024/20_pesq_finetune/fp32_512_pesq_ft/QUEUE_DONE
