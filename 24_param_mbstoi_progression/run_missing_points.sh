#!/usr/bin/env bash
# Sequential full evals (VCTK+TIMIT, 750 pairs, 32 workers, ONE GPU) of the restructuring points that
# have a checkpoint but no score. ~15 min each. GPU_ID=<free> bash run_missing_points.sh
set -uo pipefail
D=/raid/R12K41024/24_param_mbstoi_progression; R=/raid/R12K41024; export GPU_ID="${GPU_ID:-1}" LIMIT=750 METRIC_WORKERS=32 VCTK_ONLY=0
run() { echo "[chain] $1 start $(date '+%F %T')"; TAG=$1 CODE_DIR=$2 RUN_DIR=$3 CKPT=$3/logs/lightning_logs/version_0/checkpoints/$4 timeout 2h bash $D/run_point.sh > $D/results/$1.out 2>&1; echo "[chain] $1 exit=$? $(date '+%F %T')"; }
run P0p_fft512_E512_full51_3.97M  $R/BCCTN                             $R/BCCTN/outputs/2025-05-11/16-34-17                 epoch=94-step=91960.ckpt
run P4_fft128_E512_dw_win31_2.54M $R/BCCTN_sep_conv_causal_slide_win   $R/BCCTN_sep_conv_causal_slide_win/outputs/2026-01-06/17-03-19 epoch=142-step=138424.ckpt
run P3_fft128_E512_dw_2.54M       $R/BCCTN_sep_conv_causal             $R/BCCTN_sep_conv_causal/outputs/2025-06-16/00-32-24 epoch=142-step=138424.ckpt
run P2_fft256_E512_dw_2.80M       $R/BCCTN_sep_conv_causal             $R/BCCTN_sep_conv_causal/outputs/2025-06-12/16-10-22 epoch=159-step=154880.ckpt
echo "[chain] DONE"; for t in P0p_fft512_E512_full51_3.97M P4_fft128_E512_dw_win31_2.54M P3_fft128_E512_dw_2.54M P2_fft256_E512_dw_2.80M; do echo "$t: $(grep -h 'MBSTOI Enhanced' $D/results/$t/*_averages.txt 2>/dev/null | awk '{print $NF}' | tr '\n' ' ')"; done
