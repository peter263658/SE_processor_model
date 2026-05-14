#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

CONDA_ENV="${CONDA_ENV:-bcctn}"
OUTDIR="${1:-./export_out}"

conda run --no-capture-output -n "${CONDA_ENV}" python ckpt2hwdat.py \
  --outdir "${OUTDIR}"

conda run --no-capture-output -n "${CONDA_ENV}" python golden_dump.py \
  --outdir "${OUTDIR}/golden"
