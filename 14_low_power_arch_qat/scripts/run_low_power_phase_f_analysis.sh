#!/bin/bash
# Phase E/F: weight-encoding notes + active-lane operating-point sweep.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_DIR"

mkdir -p results/low_power_analysis

python scripts/analyze_low_power_arch.py \
  --output_json results/low_power_analysis/f0_dense_64_128_256.json

python scripts/analyze_low_power_arch.py \
  --qkv_groups 4 \
  --output_json results/low_power_analysis/f3_qkv_g4_64_128_256.json

python scripts/analyze_low_power_arch.py \
  --input_groups 4 \
  --qkv_groups 4 \
  --out_groups 4 \
  --output_groups 4 \
  --output_json results/low_power_analysis/f3_all_attn_g4_64_128_256.json

echo "[phase-f-analysis] wrote results/low_power_analysis/*.json"
