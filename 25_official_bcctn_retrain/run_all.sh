#!/usr/bin/env bash
# Train the official 11.09 M BCCTN on our dataset (official recipe, resumable), then evaluate with the
# deploy-model protocol and push. Re-run the SAME command after a reboot: training resumes from
# outputs/<TAG>/checkpoints/last.ckpt (full state: weights, loss buffers, Adam, LR scheduler, epoch).
#   GPU_ID=<one free gpu> [TAG=official_recipe] bash run_all.sh        # ONE process, ONE GPU at a time
set -uo pipefail
D=/raid/R12K41024/25_official_bcctn_retrain; R=/raid/R12K41024
GPU_ID="${GPU_ID:-1}"; TAG="${TAG:-official_recipe}"; OUT=$D/outputs/$TAG; mkdir -p $OUT $D/docs
export CUDA_VISIBLE_DEVICES="$GPU_ID" NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/mpl_cache
PY="conda run --no-capture-output -n bcctn python"
echo "[all] train start $(date '+%F %T') gpu=$GPU_ID out=$OUT"
$PY $D/code/train_official.py --out $OUT 2>&1 | grep -av "it/s\]" >> $OUT/train.log
grep -aq "\[train\] finished" $OUT/train.log || { echo "[all] TRAINING DID NOT FINISH (killed / reboot?) - rerun this script to resume"; exit 1; }
echo "[all] train done $(date '+%F %T')"
# evaluate the best-validation epoch and the final epoch (each ~15 min), sequentially on the same GPU
BEST=$(ls $OUT/checkpoints/all-epoch=*.ckpt | sed -E 's/.*validation_loss=([-0-9.]+)\.ckpt/\1 &/' | sort -g | head -1 | cut -d' ' -f2-)
LAST=$(ls $OUT/checkpoints/all-epoch=*.ckpt | sort | tail -1)
for pair in "bestval:$BEST" "final:$LAST"; do
  name=${pair%%:*}; ck=${pair#*:}; E=$D/results/${TAG}_$name
  [ -f $E/timit_paper_style_results_averages.txt ] && { echo "[all] eval $name already done"; continue; }
  echo "[all] eval $name <- $ck  $(date '+%F %T')"; mkdir -p $E; echo "$ck" > $E/ckpt.txt
  $PY $R/23_official_bcctn_ckpt_eval/code/eval_official.py --model_checkpoint "$ck" \
    --vctk_test_dir $R/BCCTN/Dataset/noisy_testset --vctk_clean_dir $R/BCCTN/Dataset/clean_testset \
    --timit_test_dir $R/BCCTN/Dataset/noisy_testset_timit --timit_clean_dir $R/BCCTN/Dataset/clean_testset_timit \
    --output_dir $E --limit_pairs 750 --metric_workers 32 > $E/eval.log 2>&1
  echo "[all] eval $name: $(grep -h 'MBSTOI Enhanced' $E/*_averages.txt | awk '{print $NF}' | tr '\n' ' ')"
  for f in vctk_paper_style_results.csv vctk_paper_style_results_averages.txt timit_paper_style_results.csv timit_paper_style_results_averages.txt ckpt.txt; do cp $E/$f $D/docs/${TAG}_${name}_$f 2>/dev/null; done
  [ "$BEST" = "$LAST" ] && break
done
cp $(ls -t $OUT/csv/version_*/metrics.csv | head -1) $D/docs/${TAG}_training_metrics.csv 2>/dev/null
cd $R && git add 25_official_bcctn_retrain && git commit -q -m "25_official_bcctn_retrain: official 11.09 M BCCTN trained on our dataset (official recipe) + eval

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_018BJohsHRjWmNAJPsPUxbS8" && git push -q origin add-single-channel-stoi-eval && git branch -f master add-single-channel-stoi-eval && git push -q origin master && echo "[all] pushed $(git rev-parse --short HEAD)"
echo "[all] DONE $(date '+%F %T')"
