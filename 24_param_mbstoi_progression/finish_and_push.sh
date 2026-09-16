#!/usr/bin/env bash
# Waits for run_missing_points.sh to finish, then draws the slide figure, archives the result
# tables into docs/, commits and pushes — so the results reach GitHub even if the session is gone.
D=/raid/R12K41024/24_param_mbstoi_progression; cd /raid/R12K41024
until grep -q "\[chain\] DONE" $D/results/chain.log; do sleep 30; done
echo "[finish] chain done $(date '+%F %T')"
mkdir -p $D/docs
for t in P0p_fft512_E512_full51_3.97M P4_fft128_E512_dw_win31_2.54M P3_fft128_E512_dw_2.54M P2_fft256_E512_dw_2.80M; do
  for f in vctk_paper_style_results.csv vctk_paper_style_results_averages.txt timit_paper_style_results.csv timit_paper_style_results_averages.txt model_info.txt; do
    [ -f $D/results/$t/$f ] && cp $D/results/$t/$f $D/docs/${t}_$f; done; done
cd /raid/R12K41024/paper_figures && CUDA_VISIBLE_DEVICES="" ~/miniforge3/envs/bcctn/bin/python make_param_vs_mbstoi_fig.py > $D/results/fig.log 2>&1; echo "[finish] fig exit=$?"; cat $D/results/fig.log
cd /raid/R12K41024
{ echo "# Restructuring points — auto-generated $(date '+%F %T')"; echo; echo "| point | params | VCTK | TIMIT |"; echo "|---|---:|---:|---:|"
  for t in P0p_fft512_E512_full51_3.97M P2_fft256_E512_dw_2.80M P3_fft128_E512_dw_2.54M P4_fft128_E512_dw_win31_2.54M; do
    v=$(grep -h 'MBSTOI Enhanced' $D/results/$t/vctk_paper_style_results_averages.txt 2>/dev/null | awk '{print $NF}'); ti=$(grep -h 'MBSTOI Enhanced' $D/results/$t/timit_paper_style_results_averages.txt 2>/dev/null | awk '{print $NF}'); p=$(grep params $D/results/$t/model_info.txt 2>/dev/null | awk '{print $2}')
    echo "| $t | $p | ${v:-FAILED} | ${ti:-FAILED} |"; done; } > $D/docs/RESULTS_POINTS_AUTO.md
git add $D paper_figures/make_param_vs_mbstoi_fig.py paper_figures/out/svg/slide_param_vs_mbstoi.svg paper_figures/out/png/slide_param_vs_mbstoi.png paper_figures/out/emf/slide_param_vs_mbstoi.emf 2>/dev/null
git commit -q -F - <<'MSG'
Add 24_param_mbstoi_progression: restructuring points (params vs MBSTOI) + slide figure

Auto-committed by finish_and_push.sh after the 4-point eval chain (baseline 3.97 M,
+depthwise/fft256, +fft128, +window) evaluated with dir-18's eval pipeline.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_018BJohsHRjWmNAJPsPUxbS8
MSG
git push -q origin add-single-channel-stoi-eval && git branch -f master add-single-channel-stoi-eval && git push -q origin master && echo "[finish] pushed $(git rev-parse --short HEAD)" || echo "[finish] PUSH FAILED"
