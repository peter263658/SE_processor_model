#!/usr/bin/env bash
# Robust eval queue v2 (GPU 6): each stage runs under `timeout`, checks its
# own success artifact, and retries once — survives the silent main-process
# sweeps that killed v1 mid-stage. Order: C(ep2,ep4) -> D(4 dirs) -> B.
set -uo pipefail
GPU="${GPU:-6}"
QLOG=/raid/R12K41024/20_pesq_finetune/fp32_512_pesq_ft/queue2_$(date +%H%M%S).log
exec >>"$QLOG" 2>&1
echo "=== eval queue v2 start $(date) on GPU $GPU ==="

PKG=/raid/R12K41024/18_w4a8_pow2_mixed_lmul_hw_handoff_20260530
export PATH=/home/R12K41024/miniforge3/bin:$PATH
export NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/mpl_cache

run_with_retry() {
  # run_with_retry <timeout> <success_file> <label> <cmd...>
  local T="$1" CHECK="$2" LABEL="$3"; shift 3
  for attempt in 1 2; do
    if [ -s "$CHECK" ]; then echo "[$LABEL] already done ($CHECK)"; return 0; fi
    echo "--- [$LABEL] attempt $attempt $(date) ---"
    timeout -k 60 "$T" "$@"
    local rc=$?
    # clear any orphaned metric workers before judging/retrying
    pkill -9 -f "multiprocessing.spawn" 2>/dev/null
    if [ -s "$CHECK" ]; then echo "--- [$LABEL] OK rc=$rc $(date) ---"; return 0; fi
    echo "--- [$LABEL] attempt $attempt FAILED rc=$rc $(date), retrying ---"
    sleep 30
  done
  echo "--- [$LABEL] FAILED after 2 attempts ---"
  return 1
}

# ---- C: ep2 / ep4 re-selection close-out (VCTK, new metrics) --------------
CKROOT=$PKG/outputs/apot_descale_finetune/pow2_act_from_calibep2_5ep_lr1e-5_20260531/ft/logs/lightning_logs/version_0/checkpoints
for EP in 2 4; do
  CK=$(ls $CKROOT/epoch=${EP}-step=*.ckpt | head -1)
  OUTA=$PKG/results/apot_descale_eval/ft_pow2_ep${EP}_vctk_stoi/vctk_paper_style_results_averages.txt
  CKPT="$CK" TAG="pow2_ep${EP}_vctk_stoi" VCTK_ONLY=1 GPU_ID=$GPU \
    ACT_FP8_MODE=calibrated_pow2_e4m3 \
    BCCTN_ACTIVATION_CALIB_PATH=$PKG/ckpt/activation_calibration_p999_b64.json \
    run_with_retry 100m "$OUTA" "C-ep${EP}" bash $PKG/run_apot_descale_eval.sh
done

# ---- D: deploy zero-shot at -8 / -10 dB ------------------------------------
EXT=/raid/R12K41024/20_pesq_finetune/data/lowsnr_ext
DOUT=/raid/R12K41024/20_pesq_finetune/results/pesq_eval/deploy_lowsnr_ext
for DS in vctk timit; do
  if [ "$DS" = vctk ]; then CLEAN=/raid/R12K41024/BCCTN/Dataset/clean_testset
  else CLEAN=/raid/R12K41024/BCCTN/Dataset/clean_testset_timit; fi
  for SNR in -8 -10; do
    OUT=$DOUT/${DS}_snr${SNR}dB
    TEST_DIR=$EXT/$DS/snr_${SNR}dB CLEAN_DIR=$CLEAN OUT=$OUT GPU_ID=$GPU \
      run_with_retry 45m "$OUT/average_results.txt" "D-$DS$SNR" \
      bash $PKG/run_lowsnr_ext_eval.sh
  done
done

# ---- B: FP32-512 reference full eval (parallel metrics) --------------------
cd /raid/R12K41024/20_pesq_finetune/fp32_512_pesq_ft/code
BOUT=/raid/R12K41024/20_pesq_finetune/results/pesq_eval/fp32_512_ref_full
run_with_retry 150m "$BOUT/timit_paper_style_results_averages.txt" "B" \
  env CUDA_VISIBLE_DEVICES=$GPU conda run --no-capture-output -n bcctn python eval.py \
  --model_checkpoint /raid/R12K41024/BCCTN/outputs/2025-05-10/19-39-07/logs/lightning_logs/version_0/checkpoints/epoch=98-step=95832.ckpt \
  --vctk_test_dir /raid/R12K41024/BCCTN/Dataset/noisy_testset \
  --vctk_clean_dir /raid/R12K41024/BCCTN/Dataset/clean_testset \
  --timit_test_dir /raid/R12K41024/BCCTN/Dataset/noisy_testset_timit \
  --timit_clean_dir /raid/R12K41024/BCCTN/Dataset/clean_testset_timit \
  --output_dir "$BOUT" \
  --paper_style_eval --batch_size 8 --limit_pairs 750

echo "=== eval queue v2 COMPLETE $(date) ==="
touch /raid/R12K41024/20_pesq_finetune/fp32_512_pesq_ft/QUEUE_DONE
