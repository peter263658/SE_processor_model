#!/bin/bash
set -euo pipefail

# Sanity check: 8 kHz vs 16 kHz on the same 5 source files (15 pairs each).
# Both runs use identical W4A8 PTQ config from 07/08 handoff (recommended PTQ).
#
# Q1: How much MBSTOI degrades when input is bandlimited to 4 kHz before being
#     upsampled back to 16 kHz for the 16-kHz-trained checkpoint to consume?
#
# Same model, same PTQ overrides, same source speakers/azimuths/seed → the
# only experimental delta is the 4-8 kHz band being zeroed out.

REPO=/raid/R12K41024/08_mixed_precision_qat_from_scratch
ROOT=/raid/R12K41024/01_utilization_analysis_8khz
PYTHON=/home/R12K41024/miniforge3/envs/bcctn/bin/python

# Use the FP8 QAT baseline ckpt — PTQ overrides apply W4A8 at eval time
CKPT="${REPO}/epoch=134-step=130680.ckpt"

OVERRIDES=(
    training.qat_scope=full
    training.top_k_attention=4
    training.int4_group_size=256
    +training.fp16_accumulator=true
)

cd "$REPO"

run_one () {
    local label=$1
    local clean=$2
    local noisy=$3
    local outdir=$4

    echo "========================================"
    echo "[$label]  noisy=$noisy"
    echo "         clean=$clean"
    echo "========================================"
    mkdir -p "$outdir"

    CUDA_VISIBLE_DEVICES=0 "$PYTHON" eval.py \
        --model_checkpoint "$CKPT" \
        --vctk_test_dir   "$noisy" \
        --vctk_clean_dir  "$clean" \
        --timit_test_dir  "$noisy" \
        --timit_clean_dir "$clean" \
        --output_dir      "$outdir" \
        --paper_style_eval \
        --batch_size 4 \
        --limit_pairs 50 \
        --hydra_overrides "${OVERRIDES[@]}" \
        2>&1 | tee "$outdir/run.log" | tail -50
}

run_one "16kHz baseline" \
    "$ROOT/Dataset_16khz_baseline/clean_testset" \
    "$ROOT/Dataset_16khz_baseline/noisy_testset" \
    "$ROOT/eval_16khz_baseline"

run_one "8kHz->16kHz upsampled" \
    "$ROOT/Dataset_8khz_upsampled_16k/clean_testset" \
    "$ROOT/Dataset_8khz_upsampled_16k/noisy_testset" \
    "$ROOT/eval_8khz_upsampled"

echo ""
echo "========== SUMMARY =========="
for d in "$ROOT/eval_16khz_baseline" "$ROOT/eval_8khz_upsampled"; do
    echo ""
    echo "--- $d ---"
    cat "$d/vctk_paper_style_results_averages.txt" 2>/dev/null \
        || echo "(averages file not produced — see run.log)"
done
