#!/bin/bash
# Build P1 split/grouped warm-start checkpoints.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_DIR"

BASE_CKPT="${BASE_CKPT:-${REPO_DIR}/outputs/finetune_strict/2026-05-10_02-40-16/logs/lightning_logs/version_0/checkpoints/epoch=45-step=44528.ckpt}"
OUT_DIR="${OUT_DIR:-${REPO_DIR}/Checkpoints}"
VARIANTS=("$@")
if [ "${#VARIANTS[@]}" -eq 0 ]; then
  VARIANTS=(g11 g21 g12 g22 g42 g44)
fi

mkdir -p "$OUT_DIR"

for variant in "${VARIANTS[@]}"; do
  case "$variant" in
    g11) gx=1; gs=1 ;;
    g21) gx=2; gs=1 ;;
    g12) gx=1; gs=2 ;;
    g22) gx=2; gs=2 ;;
    g42) gx=4; gs=2 ;;
    g44) gx=4; gs=4 ;;
    *)
      echo "unknown P1 variant: $variant" >&2
      exit 2
      ;;
  esac

  out_ckpt="${OUT_DIR}/p1_warmstart_${variant}.ckpt"
  echo "[warmstart] ${variant}: groups_x=${gx}, groups_skip=${gs}"
  conda run --no-capture-output -n bcctn python scripts/convert_ckpt_to_grouped.py \
    --in "$BASE_CKPT" \
    --out "$out_ckpt" \
    --groups_x "$gx" \
    --groups_skip "$gs"
done

ls -lh "${OUT_DIR}"/p1_warmstart_g*.ckpt
