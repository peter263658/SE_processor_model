#!/bin/bash

# ============================================================
# Export FP8 Weights from QAT-trained Checkpoint
# ============================================================
# This script converts a QAT-trained PyTorch Lightning checkpoint
# into per-layer FP8 weight files (hex + decimal format).
#
# Output goes to ./fp8_weights/ by default.
# Each layer produces:
#   - .txt file (human readable: index | hex | decimal)
#   - .csv file (machine parseable: index,hex,decimal)
# ============================================================

# Set environment
export PYTHONPATH="$PYTHONPATH:$(pwd)"

# ── Configuration ──────────────────────────────────────────
# Point this to your QAT-trained checkpoint
# Example: after running run_train.sh, the best checkpoint is saved by
# PyTorch Lightning under outputs/<date>/<time>/logs/lightning_logs/...
MODEL_CHECKPOINT="${1:-./model.ckpt}"

# Output directory for the exported weight files
OUTPUT_DIR="${2:-./fp8_weights}"

# FP8 format: "e4m3" (more precision) or "e5m2" (more range)
FP8_DTYPE="${3:-e4m3}"
# ───────────────────────────────────────────────────────────

echo "============================================"
echo " FP8 Weight Export"
echo "============================================"
echo " Checkpoint : $MODEL_CHECKPOINT"
echo " Output Dir : $OUTPUT_DIR"
echo " FP8 Format : $FP8_DTYPE"
echo "============================================"
echo ""

# Check checkpoint exists
if [ ! -f "$MODEL_CHECKPOINT" ]; then
    echo "❌ ERROR: Checkpoint not found: $MODEL_CHECKPOINT"
    echo ""
    echo "Usage: bash run_export_fp8_weights.sh <checkpoint_path> [output_dir] [fp8_dtype]"
    echo ""
    echo "Examples:"
    echo "  bash run_export_fp8_weights.sh ./model.ckpt"
    echo "  bash run_export_fp8_weights.sh ./model.ckpt ./fp8_weights e4m3"
    echo "  bash run_export_fp8_weights.sh outputs/2026-01-08/.../epoch=125-step=121968.ckpt ./fp8_e5m2 e5m2"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run the export
python export_fp8_weights.py \
    --checkpoint "$MODEL_CHECKPOINT" \
    --output_dir "$OUTPUT_DIR" \
    --fp8_dtype "$FP8_DTYPE"

echo ""
echo "============================================"
echo " Export complete!"
echo " Files saved to: $OUTPUT_DIR"
echo "============================================"
echo ""
echo "Each .txt file format:"
echo "  Index      Hex        Decimal"
echo "  0          0x3C       1.00000000e+00"
echo "  1          0xBC       -1.00000000e+00"
echo "  ..."
