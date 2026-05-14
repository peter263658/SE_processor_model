#!/bin/bash
# QAT fine-tune + eval for the "all-FP8 attention" hardware variant.
#
# Config:
#   * Baseline ckpt : FP8 pre-trained epoch=134-step=130680.ckpt (07 handoff §Step 4)
#   * Scope         : full  (encoder/decoder DW+PW W4A8, MHA W4A8, 6 outlier PWs FP8)
#   * Attention     : FP8 operands end-to-end (Q/K/V, softmax out, V)  + FP16 accumulator
#   * INT4 scale    : FP8 (E4M3)
#   * LR / epochs   : 1e-4 / 30  (per 07 handoff §Step 4 fine-tune recipe)
#
# After training completes this script automatically runs paper-style eval on
# the best checkpoint with the matching precision overrides.

set -e

REPO=/raid/R12K41024/08_all_fp8_ptq
OUTDIR="$REPO/qat_all_fp8_attention"
BASELINE_CKPT="$REPO/epoch=134-step=130680.ckpt"   # FP8 baseline from 07_mixed_precision_qat
PY=/home/R12K41024/miniforge3/envs/bcctn/bin/python

mkdir -p "$OUTDIR"
cd "$REPO"

export PYTHONPATH="$REPO:$PYTHONPATH"
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=1

echo "[$(date)] Starting QAT fine-tune (all-FP8 attention) on GPU 1..."
echo "  baseline ckpt : $BASELINE_CKPT"
echo "  output dir    : $OUTDIR"

# NOTE: the checkpoint filename contains '=' characters, which Hydra's override
# parser treats as a separator. Wrap the value in single quotes (as a literal,
# via the outer "" shell quoting) so Hydra sees it as a quoted string.
"$PY" train.py \
    hydra.run.dir="$OUTDIR" \
    training.qat_scope=full \
    training.int4_group_size=256 \
    training.learning_rate=1e-4 \
    training.n_epochs=30 \
    "training.train_checkpoint_path='$BASELINE_CKPT'" \
    +training.fp8_attention=true \
    +training.fp8_scale=true \
    > "$OUTDIR/train.log" 2>&1

echo "[$(date)] QAT training complete."

# Best-checkpoint locator — ModelCheckpoint monitors validation_loss and
# saves the best as "epoch=X-step=Y.ckpt" (also last.ckpt). Pick the
# non-"last" file (there should be exactly one).
CKPT_DIR="$OUTDIR/logs/lightning_logs/version_0/checkpoints"
BEST_CKPT=$(ls "$CKPT_DIR"/epoch=*.ckpt 2>/dev/null | head -1)
if [[ -z "$BEST_CKPT" ]]; then
    echo "ERROR: best checkpoint not found under $CKPT_DIR"
    ls -la "$CKPT_DIR" || true
    exit 1
fi
echo "  best ckpt     : $BEST_CKPT"

echo "[$(date)] Starting paper-style evaluation..."
"$PY" eval.py \
    --model_checkpoint "$BEST_CKPT" \
    --vctk_test_dir    "/raid/R12K41024/BCCTN/Dataset/noisy_testset" \
    --vctk_clean_dir   "/raid/R12K41024/BCCTN/Dataset/clean_testset" \
    --output_dir       "$OUTDIR/eval/paper_style" \
    --paper_style_eval \
    --batch_size 8 \
    --limit_pairs 750 \
    --hydra_overrides \
        training.qat_scope=full \
        training.int4_group_size=256 \
        +training.fp8_attention=true \
        +training.fp8_scale=true \
    > "$OUTDIR/eval.log" 2>&1

echo "[$(date)] Eval complete. Results in $OUTDIR/eval/paper_style"
