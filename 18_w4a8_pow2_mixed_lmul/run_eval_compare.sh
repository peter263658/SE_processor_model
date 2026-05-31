#!/bin/bash
# After training completes, eval both 09 variants (finetune + scratch) and
# merge into a comparison table that includes the existing 08 baseline.
#
# Usage:
#   ./run_eval_compare.sh
#
# Auto-discovers the latest run under outputs/finetune/ and outputs/scratch/,
# picks `last.ckpt` (or the highest-epoch checkpoint), evals each, then merges
# the resulting CSVs alongside 08's baseline.

set -e
cd "$(dirname "$0")"

ROOT_08=/raid/R12K41024/08_mixed_precision_qat_from_scratch
ROOT_09=/raid/R12K41024/09_weight_tied_mha

find_best_ckpt() {
    # $1 = dir like outputs/finetune
    # Pick the latest `epoch=N-step=M.ckpt` (= best val_loss so far per
    # Lightning's ModelCheckpoint(monitor=val_loss)). Skip `last.ckpt` —
    # that is "most recent epoch" not "best val_loss".
    local base="$1"
    local ckpt
    ckpt=$(ls -t "$base"/*/logs/lightning_logs/version_0/checkpoints/epoch=*.ckpt 2>/dev/null | head -n 1 || true)
    if [ -z "$ckpt" ]; then
        echo ""
        return
    fi
    echo "$ckpt"
}

FT_CKPT=$(find_best_ckpt outputs/finetune)
SC_CKPT=$(find_best_ckpt outputs/scratch)

echo "Discovered checkpoints:"
echo "  finetune : ${FT_CKPT:-<none>}"
echo "  scratch  : ${SC_CKPT:-<none>}"

if [ -z "$FT_CKPT" ] && [ -z "$SC_CKPT" ]; then
    echo "No checkpoints found. Did training start?"
    exit 1
fi

mkdir -p results

if [ -n "$FT_CKPT" ]; then
    echo
    echo "[1/3] Eval fine-tune ..."
    bash run_eval.sh "$FT_CKPT" results/finetune 5
fi

if [ -n "$SC_CKPT" ]; then
    echo
    echo "[2/3] Eval scratch ..."
    bash run_eval.sh "$SC_CKPT" results/scratch 5
fi

echo
echo "[3/3] Merge comparison ..."

# Build variant args: 08 baseline + every 09 variant present + every 08 PTQ folder.
VARIANTS=()
B08="$ROOT_08/results_May21/vctk_paper_style/vctk_paper_style_results.csv"
[ -f "$B08" ] && VARIANTS+=("08_baseline=$B08")

for d in "$ROOT_08"/ptq_*_eval/paper_style; do
    csv="$d/vctk_paper_style_results.csv"
    if [ -f "$csv" ]; then
        label=$(basename "$(dirname "$d")")
        VARIANTS+=("${label}=$csv")
    fi
done

# 09 variants in priority order: regular FT/SC (mha_only QAT), strict FT/SC (full+topk4 QAT), PTQ-strict on regular FT.
[ -f results/finetune/vctk_paper_style_results.csv        ] && VARIANTS+=("09_tied_ft=results/finetune/vctk_paper_style_results.csv")
[ -f results/scratch/vctk_paper_style_results.csv         ] && VARIANTS+=("09_tied_sc=results/scratch/vctk_paper_style_results.csv")
[ -f results/finetune_strict/vctk_paper_style_results.csv ] && VARIANTS+=("09_tied_ft_strict=results/finetune_strict/vctk_paper_style_results.csv")
[ -f results/scratch_strict/vctk_paper_style_results.csv  ] && VARIANTS+=("09_tied_sc_strict=results/scratch_strict/vctk_paper_style_results.csv")
[ -f results/ptq_strict/vctk_paper_style_results.csv      ] && VARIANTS+=("09_tied_ft_ptq_strict=results/ptq_strict/vctk_paper_style_results.csv")

conda run --no-capture-output -n bcctn python merge_eval_comparison.py \
    --dataset vctk \
    --outfile results/compare_table_vctk.csv \
    "${VARIANTS[@]}"

# TIMIT
TVARIANTS=()
TB08="$ROOT_08/results_May21/timit_paper_style/timit_paper_style_results.csv"
[ -f "$TB08" ] && TVARIANTS+=("08_baseline=$TB08")
[ -f results/finetune/timit_paper_style_results.csv         ] && TVARIANTS+=("09_tied_ft=results/finetune/timit_paper_style_results.csv")
[ -f results/scratch/timit_paper_style_results.csv          ] && TVARIANTS+=("09_tied_sc=results/scratch/timit_paper_style_results.csv")
[ -f results/finetune_strict/timit_paper_style_results.csv  ] && TVARIANTS+=("09_tied_ft_strict=results/finetune_strict/timit_paper_style_results.csv")
[ -f results/scratch_strict/timit_paper_style_results.csv   ] && TVARIANTS+=("09_tied_sc_strict=results/scratch_strict/timit_paper_style_results.csv")
[ -f results/ptq_strict/timit_paper_style_results.csv       ] && TVARIANTS+=("09_tied_ft_ptq_strict=results/ptq_strict/timit_paper_style_results.csv")

if [ ${#TVARIANTS[@]} -ge 1 ]; then
    conda run --no-capture-output -n bcctn python merge_eval_comparison.py \
        --dataset timit \
        --outfile results/compare_table_timit.csv \
        "${TVARIANTS[@]}"
fi

echo
echo "Done. Comparison tables saved under results/."
