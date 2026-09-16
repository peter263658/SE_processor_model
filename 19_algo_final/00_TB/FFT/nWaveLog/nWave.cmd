wvSetPosition -win $_nWave1 {("G1" 0)}
wvOpenFile -win $_nWave1 {/home/r12k41024/Research/00_TB/FFT/tb_stft.fsdb}
verdiSetActWin -win $_nWave1
wvResizeWindow -win $_nWave1 1080 392 1920 1057
wvResizeWindow -win $_nWave1 1080 392 1920 1057
wvResizeWindow -win $_nWave1 1080 392 1920 1009
wvResizeWindow -win $_nWave1 1080 392 1920 1057
wvResizeWindow -win $_nWave1 1080 392 1920 1009
wvGetSignalOpen -win $_nWave1
wvGetSignalSetScope -win $_nWave1 "/Mem_def"
wvGetSignalSetScope -win $_nWave1 "/tb_stft"
wvGetSignalSetScope -win $_nWave1 "/tb_stft/u_stft"
wvSetPosition -win $_nWave1 {("G1" 3)}
wvSetPosition -win $_nWave1 {("G1" 3)}
wvAddSignal -win $_nWave1 -clear
wvAddSignal -win $_nWave1 -group {"G1" \
{/tb_stft/u_stft/i_BR_dat_en} \
{/tb_stft/u_stft/i_BR_dat_im\[15:0\]} \
{/tb_stft/u_stft/i_BR_dat_re\[15:0\]} \
}
wvAddSignal -win $_nWave1 -group {"G2" \
}
wvSelectSignal -win $_nWave1 {( "G1" 1 2 3 )} 
wvSetPosition -win $_nWave1 {("G1" 3)}
wvSetPosition -win $_nWave1 {("G1" 4)}
wvSetPosition -win $_nWave1 {("G1" 4)}
wvAddSignal -win $_nWave1 -clear
wvAddSignal -win $_nWave1 -group {"G1" \
{/tb_stft/u_stft/i_BR_dat_en} \
{/tb_stft/u_stft/i_BR_dat_im\[15:0\]} \
{/tb_stft/u_stft/i_BR_dat_re\[15:0\]} \
{/tb_stft/u_stft/i_rst_n} \
}
wvAddSignal -win $_nWave1 -group {"G2" \
}
wvSelectSignal -win $_nWave1 {( "G1" 4 )} 
wvSetPosition -win $_nWave1 {("G1" 4)}
wvSetPosition -win $_nWave1 {("G1" 5)}
wvSetPosition -win $_nWave1 {("G1" 5)}
wvAddSignal -win $_nWave1 -clear
wvAddSignal -win $_nWave1 -group {"G1" \
{/tb_stft/u_stft/i_BR_dat_en} \
{/tb_stft/u_stft/i_BR_dat_im\[15:0\]} \
{/tb_stft/u_stft/i_BR_dat_re\[15:0\]} \
{/tb_stft/u_stft/i_rst_n} \
{/tb_stft/u_stft/i_clk} \
}
wvAddSignal -win $_nWave1 -group {"G2" \
}
wvSelectSignal -win $_nWave1 {( "G1" 5 )} 
wvSetPosition -win $_nWave1 {("G1" 5)}
wvGetSignalClose -win $_nWave1
wvSelectSignal -win $_nWave1 {( "G1" 4 5 )} 
wvSetPosition -win $_nWave1 {("G1" 4)}
wvSetPosition -win $_nWave1 {("G1" 2)}
wvSetPosition -win $_nWave1 {("G1" 0)}
wvMoveSelected -win $_nWave1
wvSetPosition -win $_nWave1 {("G1" 0)}
wvSetPosition -win $_nWave1 {("G1" 2)}
wvGetSignalOpen -win $_nWave1
wvGetSignalSetScope -win $_nWave1 "/Mem_def"
wvGetSignalSetScope -win $_nWave1 "/tb_stft"
wvGetSignalSetScope -win $_nWave1 "/tb_stft/u_stft"
wvGetSignalSetScope -win $_nWave1 "/tb_stft/u_stft"
wvSetPosition -win $_nWave1 {("G1" 4)}
wvSetPosition -win $_nWave1 {("G1" 4)}
wvAddSignal -win $_nWave1 -clear
wvAddSignal -win $_nWave1 -group {"G1" \
{/tb_stft/u_stft/i_rst_n} \
{/tb_stft/u_stft/i_clk} \
{/tb_stft/u_stft/i_FFT_dat_en} \
{/tb_stft/u_stft/i_hamm_dat_en} \
{/tb_stft/u_stft/i_BR_dat_en} \
{/tb_stft/u_stft/i_BR_dat_im\[15:0\]} \
{/tb_stft/u_stft/i_BR_dat_re\[15:0\]} \
}
wvAddSignal -win $_nWave1 -group {"G2" \
}
wvSelectSignal -win $_nWave1 {( "G1" 3 4 )} 
wvSetPosition -win $_nWave1 {("G1" 4)}
wvGetSignalClose -win $_nWave1
wvSelectSignal -win $_nWave1 {( "G1" 4 )} 
wvSetPosition -win $_nWave1 {("G1" 3)}
wvSetPosition -win $_nWave1 {("G1" 2)}
wvMoveSelected -win $_nWave1
wvSetPosition -win $_nWave1 {("G1" 2)}
wvSetPosition -win $_nWave1 {("G1" 3)}
wvSelectSignal -win $_nWave1 {( "G1" 4 )} 
wvGetSignalOpen -win $_nWave1
wvGetSignalSetScope -win $_nWave1 "/Mem_def"
wvGetSignalSetScope -win $_nWave1 "/tb_stft"
wvGetSignalSetScope -win $_nWave1 "/tb_stft/u_stft"
wvGetSignalSetScope -win $_nWave1 "/tb_stft/u_stft"
wvSetPosition -win $_nWave1 {("G1" 6)}
wvSetPosition -win $_nWave1 {("G1" 6)}
wvAddSignal -win $_nWave1 -clear
wvAddSignal -win $_nWave1 -group {"G1" \
{/tb_stft/u_stft/i_rst_n} \
{/tb_stft/u_stft/i_clk} \
{/tb_stft/u_stft/i_hamm_dat_en} \
{/tb_stft/u_stft/o_FFT_dat_en} \
{/tb_stft/u_stft/o_FFT_dat_im\[15:0\]} \
{/tb_stft/u_stft/o_FFT_dat_re\[15:0\]} \
{/tb_stft/u_stft/i_FFT_dat_en} \
{/tb_stft/u_stft/i_BR_dat_en} \
{/tb_stft/u_stft/i_BR_dat_im\[15:0\]} \
{/tb_stft/u_stft/i_BR_dat_re\[15:0\]} \
}
wvAddSignal -win $_nWave1 -group {"G2" \
}
wvSelectSignal -win $_nWave1 {( "G1" 4 5 6 )} 
wvSetPosition -win $_nWave1 {("G1" 6)}
wvGetSignalClose -win $_nWave1
wvSetPosition -win $_nWave1 {("G1" 4)}
wvSetPosition -win $_nWave1 {("G1" 7)}
wvMoveSelected -win $_nWave1
wvSetPosition -win $_nWave1 {("G1" 7)}
wvSetCursor -win $_nWave1 1373003.014572 -snap {("G1" 6)}
wvSelectSignal -win $_nWave1 {( "G1" 2 )} 
wvSetPosition -win $_nWave1 {("G1" 2)}
wvSetPosition -win $_nWave1 {("G1" 1)}
wvMoveSelected -win $_nWave1
wvSetPosition -win $_nWave1 {("G1" 1)}
wvSetPosition -win $_nWave1 {("G1" 2)}
wvSelectSignal -win $_nWave1 {( "G1" 1 )} 
wvSelectSignal -win $_nWave1 {( "G1" 1 )} 
wvSelectSignal -win $_nWave1 {( "G1" 1 )} 
wvSelectSignal -win $_nWave1 {( "G1" 2 )} 
wvSetPosition -win $_nWave1 {("G1" 1)}
wvSetPosition -win $_nWave1 {("G1" 0)}
wvMoveSelected -win $_nWave1
wvSetPosition -win $_nWave1 {("G1" 0)}
wvSetPosition -win $_nWave1 {("G1" 1)}
wvZoomIn -win $_nWave1
wvZoomIn -win $_nWave1
wvGetSignalOpen -win $_nWave1
wvGetSignalSetScope -win $_nWave1 "/Mem_def"
wvGetSignalSetScope -win $_nWave1 "/tb_stft"
wvGetSignalSetScope -win $_nWave1 "/tb_stft/u_stft"
wvGetSignalSetScope -win $_nWave1 "/tb_stft/u_stft"
wvSetPosition -win $_nWave1 {("G1" 3)}
wvSetPosition -win $_nWave1 {("G1" 3)}
wvAddSignal -win $_nWave1 -clear
wvAddSignal -win $_nWave1 -group {"G1" \
{/tb_stft/u_stft/i_clk} \
{/tb_stft/u_stft/o_hamm_dat_en} \
{/tb_stft/u_stft/o_hamm_dat_im\[15:0\]} \
{/tb_stft/u_stft/i_rst_n} \
{/tb_stft/u_stft/i_hamm_dat_en} \
{/tb_stft/u_stft/i_FFT_dat_en} \
{/tb_stft/u_stft/o_FFT_dat_en} \
{/tb_stft/u_stft/o_FFT_dat_im\[15:0\]} \
{/tb_stft/u_stft/o_FFT_dat_re\[15:0\]} \
{/tb_stft/u_stft/i_BR_dat_en} \
{/tb_stft/u_stft/i_BR_dat_im\[15:0\]} \
{/tb_stft/u_stft/i_BR_dat_re\[15:0\]} \
}
wvAddSignal -win $_nWave1 -group {"G2" \
}
wvSelectSignal -win $_nWave1 {( "G1" 2 3 )} 
wvSetPosition -win $_nWave1 {("G1" 3)}
wvGetSignalClose -win $_nWave1
wvSelectSignal -win $_nWave1 {( "G1" 4 )} 
wvSetPosition -win $_nWave1 {("G1" 4)}
wvSetPosition -win $_nWave1 {("G1" 2)}
wvSetPosition -win $_nWave1 {("G1" 1)}
wvMoveSelected -win $_nWave1
wvSetPosition -win $_nWave1 {("G1" 1)}
wvSetPosition -win $_nWave1 {("G1" 2)}
wvSelectSignal -win $_nWave1 {( "G1" 4 )} 
wvSelectSignal -win $_nWave1 {( "G1" 5 )} 
wvSetPosition -win $_nWave1 {("G1" 5)}
wvSetPosition -win $_nWave1 {("G1" 4)}
wvSetPosition -win $_nWave1 {("G1" 3)}
wvSetPosition -win $_nWave1 {("G1" 2)}
wvSetPosition -win $_nWave1 {("G1" 3)}
wvSetPosition -win $_nWave1 {("G1" 4)}
wvSetPosition -win $_nWave1 {("G1" 3)}
wvSetPosition -win $_nWave1 {("G1" 2)}
wvMoveSelected -win $_nWave1
wvSetPosition -win $_nWave1 {("G1" 2)}
wvSetPosition -win $_nWave1 {("G1" 3)}
wvSelectSignal -win $_nWave1 {( "G1" 6 )} 
wvZoomOut -win $_nWave1
wvZoomOut -win $_nWave1
wvZoomOut -win $_nWave1
wvZoomIn -win $_nWave1
wvZoomIn -win $_nWave1
wvZoomIn -win $_nWave1
wvSetCursor -win $_nWave1 1454516.337350 -snap {("G1" 7)}
wvGetSignalOpen -win $_nWave1
wvGetSignalSetScope -win $_nWave1 "/Mem_def"
wvGetSignalSetScope -win $_nWave1 "/tb_stft"
wvGetSignalSetScope -win $_nWave1 "/tb_stft/u_stft"
wvGetSignalSetScope -win $_nWave1 "/tb_stft/u_stft"
wvGetSignalClose -win $_nWave1
wvGetSignalOpen -win $_nWave1
wvGetSignalSetScope -win $_nWave1 "/Mem_def"
wvGetSignalSetScope -win $_nWave1 "/tb_stft"
wvGetSignalSetScope -win $_nWave1 "/tb_stft/u_stft"
wvGetSignalSetScope -win $_nWave1 "/tb_stft/u_stft"
wvSetPosition -win $_nWave1 {("G1" 4)}
wvSetPosition -win $_nWave1 {("G1" 4)}
wvAddSignal -win $_nWave1 -clear
wvAddSignal -win $_nWave1 -group {"G1" \
{/tb_stft/u_stft/i_clk} \
{/tb_stft/u_stft/i_rst_n} \
{/tb_stft/u_stft/i_hamm_dat_en} \
{/tb_stft/u_stft/o_dat_en} \
{/tb_stft/u_stft/o_hamm_dat_en} \
{/tb_stft/u_stft/o_hamm_dat_im\[15:0\]} \
{/tb_stft/u_stft/i_FFT_dat_en} \
{/tb_stft/u_stft/o_FFT_dat_en} \
{/tb_stft/u_stft/o_FFT_dat_im\[15:0\]} \
{/tb_stft/u_stft/o_FFT_dat_re\[15:0\]} \
{/tb_stft/u_stft/i_BR_dat_en} \
{/tb_stft/u_stft/i_BR_dat_im\[15:0\]} \
{/tb_stft/u_stft/i_BR_dat_re\[15:0\]} \
}
wvAddSignal -win $_nWave1 -group {"G2" \
}
wvSelectSignal -win $_nWave1 {( "G1" 4 )} 
wvSetPosition -win $_nWave1 {("G1" 4)}
wvGetSignalClose -win $_nWave1
wvDisplayGridCount -win $_nWave1 -off
wvGetSignalClose -win $_nWave1
wvReloadFile -win $_nWave1
wvSelectSignal -win $_nWave1 {( "G1" 4 5 6 7 8 9 10 11 12 13 )} 
wvSelectSignal -win $_nWave1 {( "G1" 3 )} 
wvSelectSignal -win $_nWave1 {( "G1" 4 )} 
wvSetPosition -win $_nWave1 {("G1" 7)}
wvSetPosition -win $_nWave1 {("G1" 10)}
wvSetPosition -win $_nWave1 {("G1" 11)}
wvSetPosition -win $_nWave1 {("G1" 12)}
wvSetPosition -win $_nWave1 {("G1" 13)}
wvSetPosition -win $_nWave1 {("G2" 0)}
wvSetPosition -win $_nWave1 {("G1" 4)}
wvSetPosition -win $_nWave1 {("G1" 7)}
wvSetPosition -win $_nWave1 {("G1" 11)}
wvSetPosition -win $_nWave1 {("G1" 12)}
wvSetPosition -win $_nWave1 {("G1" 13)}
wvSetPosition -win $_nWave1 {("G2" 0)}
wvMoveSelected -win $_nWave1
wvSetPosition -win $_nWave1 {("G2" 1)}
wvSetPosition -win $_nWave1 {("G2" 1)}
wvZoomOut -win $_nWave1
wvZoomOut -win $_nWave1
wvZoomOut -win $_nWave1
wvZoomOut -win $_nWave1
wvZoomOut -win $_nWave1
wvZoomOut -win $_nWave1
wvZoomOut -win $_nWave1
wvZoomOut -win $_nWave1
wvZoomOut -win $_nWave1
wvZoomOut -win $_nWave1
wvZoomOut -win $_nWave1
wvZoomOut -win $_nWave1
wvZoomOut -win $_nWave1
wvZoomOut -win $_nWave1
wvZoomOut -win $_nWave1
wvGetSignalOpen -win $_nWave1
wvSetPosition -win $_nWave1 {("G2" 4)}
wvSetPosition -win $_nWave1 {("G2" 4)}
wvAddSignal -win $_nWave1 -clear
wvAddSignal -win $_nWave1 -group {"G1" \
{/tb_stft/u_stft/i_clk} \
{/tb_stft/u_stft/i_rst_n} \
{/tb_stft/u_stft/i_hamm_dat_en} \
{/tb_stft/u_stft/o_hamm_dat_en} \
{/tb_stft/u_stft/o_hamm_dat_im\[15:0\]} \
{/tb_stft/u_stft/i_FFT_dat_en} \
{/tb_stft/u_stft/o_FFT_dat_en} \
{/tb_stft/u_stft/o_FFT_dat_im\[15:0\]} \
{/tb_stft/u_stft/o_FFT_dat_re\[15:0\]} \
{/tb_stft/u_stft/i_BR_dat_en} \
{/tb_stft/u_stft/i_BR_dat_im\[15:0\]} \
{/tb_stft/u_stft/i_BR_dat_re\[15:0\]} \
}
wvAddSignal -win $_nWave1 -group {"G2" \
{/tb_stft/u_stft/o_dat_en} \
{/tb_stft/u_stft/o_BR_dat_en} \
{/tb_stft/u_stft/o_BR_dat_im\[63:0\]} \
{/tb_stft/u_stft/o_BR_dat_re\[63:0\]} \
}
wvAddSignal -win $_nWave1 -group {"G3" \
}
wvSelectSignal -win $_nWave1 {( "G2" 2 3 4 )} 
wvSetPosition -win $_nWave1 {("G2" 4)}
wvGetSignalClose -win $_nWave1
wvZoomOut -win $_nWave1
wvZoomOut -win $_nWave1
wvResizeWindow -win $_nWave1 1080 392 1920 1009
wvSetCursor -win $_nWave1 8524490.563513 -snap {("G3" 0)}
wvSetCursor -win $_nWave1 9282566.856820 -snap {("G3" 0)}
wvDisplayGridCount -win $_nWave1 -off
wvGetSignalClose -win $_nWave1
wvReloadFile -win $_nWave1
wvGetSignalOpen -win $_nWave1
wvSetPosition -win $_nWave1 {("G2" 5)}
wvSetPosition -win $_nWave1 {("G2" 5)}
wvAddSignal -win $_nWave1 -clear
wvAddSignal -win $_nWave1 -group {"G1" \
{/tb_stft/u_stft/i_clk} \
{/tb_stft/u_stft/i_rst_n} \
{/tb_stft/u_stft/i_hamm_dat_en} \
{/tb_stft/u_stft/o_hamm_dat_en} \
{/tb_stft/u_stft/o_hamm_dat_im\[15:0\]} \
{/tb_stft/u_stft/i_FFT_dat_en} \
{/tb_stft/u_stft/o_FFT_dat_en} \
{/tb_stft/u_stft/o_FFT_dat_im\[15:0\]} \
{/tb_stft/u_stft/o_FFT_dat_re\[15:0\]} \
{/tb_stft/u_stft/i_BR_dat_en} \
{/tb_stft/u_stft/i_BR_dat_im\[15:0\]} \
{/tb_stft/u_stft/i_BR_dat_re\[15:0\]} \
}
wvAddSignal -win $_nWave1 -group {"G2" \
{/tb_stft/u_stft/o_dat_en} \
{/tb_stft/u_stft/o_BR_dat_en} \
{/tb_stft/u_stft/o_BR_dat_im\[63:0\]} \
{/tb_stft/u_stft/o_BR_dat_re\[63:0\]} \
{/tb_stft/u_stft/i_BR_new_frame} \
}
wvAddSignal -win $_nWave1 -group {"G3" \
}
wvSelectSignal -win $_nWave1 {( "G2" 5 )} 
wvSetPosition -win $_nWave1 {("G2" 5)}
wvGetSignalClose -win $_nWave1
wvSetPosition -win $_nWave1 {("G2" 2)}
wvSetPosition -win $_nWave1 {("G2" 1)}
wvSetPosition -win $_nWave1 {("G2" 0)}
wvMoveSelected -win $_nWave1
wvSetPosition -win $_nWave1 {("G2" 0)}
wvSetPosition -win $_nWave1 {("G2" 1)}
wvZoomIn -win $_nWave1
wvZoomIn -win $_nWave1
wvZoomIn -win $_nWave1
wvSetCursor -win $_nWave1 1458138.684526 -snap {("G1" 9)}
wvZoomIn -win $_nWave1
wvZoomIn -win $_nWave1
wvZoomIn -win $_nWave1
wvGetSignalOpen -win $_nWave1
wvGetSignalSetScope -win $_nWave1 "/Mem_def"
wvGetSignalSetScope -win $_nWave1 "/tb_stft"
wvGetSignalSetScope -win $_nWave1 "/tb_stft/u_stft"
wvGetSignalSetScope -win $_nWave1 "/tb_stft/u_stft"
wvGetSignalSetScope -win $_nWave1 "/tb_stft/u_stft/u_hamming"
wvGetSignalSetScope -win $_nWave1 "/tb_stft/u_stft/u_bit_reorder"
wvSetPosition -win $_nWave1 {("G2" 9)}
wvSetPosition -win $_nWave1 {("G2" 9)}
wvAddSignal -win $_nWave1 -clear
wvAddSignal -win $_nWave1 -group {"G1" \
{/tb_stft/u_stft/i_clk} \
{/tb_stft/u_stft/i_rst_n} \
{/tb_stft/u_stft/i_hamm_dat_en} \
{/tb_stft/u_stft/o_hamm_dat_en} \
{/tb_stft/u_stft/o_hamm_dat_im\[15:0\]} \
{/tb_stft/u_stft/i_FFT_dat_en} \
{/tb_stft/u_stft/o_FFT_dat_en} \
{/tb_stft/u_stft/o_FFT_dat_im\[15:0\]} \
{/tb_stft/u_stft/o_FFT_dat_re\[15:0\]} \
{/tb_stft/u_stft/i_BR_dat_en} \
{/tb_stft/u_stft/i_BR_dat_im\[15:0\]} \
{/tb_stft/u_stft/i_BR_dat_re\[15:0\]} \
}
wvAddSignal -win $_nWave1 -group {"G2" \
{/tb_stft/u_stft/i_BR_new_frame} \
{/tb_stft/u_stft/u_bit_reorder/bit_counter\[6:0\]} \
{/tb_stft/u_stft/u_bit_reorder/bit_reverse_counter\[6:0\]} \
{/tb_stft/u_stft/u_bit_reorder/fft_bank_im_r\[63:0\]} \
{/tb_stft/u_stft/u_bit_reorder/fft_bank_re_r\[63:0\]} \
{/tb_stft/u_stft/u_bit_reorder/i_dat_en} \
{/tb_stft/u_stft/u_bit_reorder/i_dat_im\[15:0\]} \
{/tb_stft/u_stft/u_bit_reorder/i_dat_re\[15:0\]} \
{/tb_stft/u_stft/u_bit_reorder/i_new_frame} \
{/tb_stft/u_stft/o_dat_en} \
{/tb_stft/u_stft/o_BR_dat_en} \
{/tb_stft/u_stft/o_BR_dat_im\[63:0\]} \
{/tb_stft/u_stft/o_BR_dat_re\[63:0\]} \
}
wvAddSignal -win $_nWave1 -group {"G3" \
}
wvSelectSignal -win $_nWave1 {( "G2" 2 3 4 5 6 7 8 9 )} 
wvSetPosition -win $_nWave1 {("G2" 9)}
wvGetSignalClose -win $_nWave1
wvSetPosition -win $_nWave1 {("G2" 7)}
wvSetPosition -win $_nWave1 {("G2" 11)}
wvSetPosition -win $_nWave1 {("G3" 0)}
wvMoveSelected -win $_nWave1
wvSetPosition -win $_nWave1 {("G3" 8)}
wvSetPosition -win $_nWave1 {("G3" 8)}
wvSelectSignal -win $_nWave1 {( "G3" 1 )} 
wvSelectSignal -win $_nWave1 {( "G3" 1 )} 
wvSelectSignal -win $_nWave1 {( "G3" 1 2 )} 
wvSelectSignal -win $_nWave1 {( "G3" 1 2 )} 
wvSetRadix -win $_nWave1 -format UDec
wvSelectSignal -win $_nWave1 {( "G3" 5 )} 
wvSelectSignal -win $_nWave1 {( "G3" 7 )} 
wvSelectSignal -win $_nWave1 {( "G3" 8 )} 
wvSelectSignal -win $_nWave1 {( "G3" 4 )} 
wvSelectSignal -win $_nWave1 {( "G3" 3 4 )} 
wvSelectSignal -win $_nWave1 {( "G3" 3 )} 
wvSelectSignal -win $_nWave1 {( "G3" 4 )} 
wvSelectSignal -win $_nWave1 {( "G3" 2 )} 
wvSelectSignal -win $_nWave1 {( "G3" 3 )} 
wvSetCursor -win $_nWave1 1604073.645503 -snap {("G4" 0)}
wvDisplayGridCount -win $_nWave1 -off
wvGetSignalClose -win $_nWave1
wvReloadFile -win $_nWave1
wvSetCursor -win $_nWave1 142139.304995 -snap {("G4" 0)}
wvSetCursor -win $_nWave1 95968.204223 -snap {("G4" 0)}
wvDisplayGridCount -win $_nWave1 -off
wvGetSignalClose -win $_nWave1
wvReloadFile -win $_nWave1
wvZoomOut -win $_nWave1
wvZoomOut -win $_nWave1
wvSetCursor -win $_nWave1 817261.117333 -snap {("G4" 0)}
wvDisplayGridCount -win $_nWave1 -off
wvGetSignalClose -win $_nWave1
wvReloadFile -win $_nWave1
wvSetCursor -win $_nWave1 6593642.037159 -snap {("G2" 2)}
wvSelectSignal -win $_nWave1 {( "G2" 3 )} 
wvSetCursor -win $_nWave1 2753875.458361 -snap {("G2" 3)}
wvSetCursor -win $_nWave1 6581455.809126 -snap {("G2" 3)}
wvSetCursor -win $_nWave1 6437382.636036 -snap {("G4" 0)}
wvDisplayGridCount -win $_nWave1 -off
wvGetSignalClose -win $_nWave1
wvReloadFile -win $_nWave1
wvDisplayGridCount -win $_nWave1 -off
wvGetSignalClose -win $_nWave1
wvReloadFile -win $_nWave1
wvDisplayGridCount -win $_nWave1 -off
wvGetSignalClose -win $_nWave1
wvReloadFile -win $_nWave1
wvDisplayGridCount -win $_nWave1 -off
wvGetSignalClose -win $_nWave1
wvReloadFile -win $_nWave1
wvDisplayGridCount -win $_nWave1 -off
wvGetSignalClose -win $_nWave1
wvReloadFile -win $_nWave1
wvSetCursor -win $_nWave1 155240.325698 -snap {("G4" 0)}
wvDisplayGridCount -win $_nWave1 -off
wvGetSignalClose -win $_nWave1
wvReloadFile -win $_nWave1
wvSetCursor -win $_nWave1 141548.571940 -snap {("G4" 0)}
wvDisplayGridCount -win $_nWave1 -off
wvGetSignalClose -win $_nWave1
wvReloadFile -win $_nWave1
wvSetCursor -win $_nWave1 1911655.479428 -snap {("G4" 0)}
wvSetCursor -win $_nWave1 2745271.866507 -snap {("G2" 3)}
wvSelectSignal -win $_nWave1 {( "G3" 8 )} 
wvSetPosition -win $_nWave1 {("G3" 1)}
wvSetPosition -win $_nWave1 {("G2" 2)}
wvSetPosition -win $_nWave1 {("G2" 1)}
wvSetPosition -win $_nWave1 {("G2" 0)}
wvSetPosition -win $_nWave1 {("G1" 12)}
wvSetPosition -win $_nWave1 {("G1" 11)}
wvSetPosition -win $_nWave1 {("G1" 9)}
wvSetPosition -win $_nWave1 {("G1" 8)}
wvSetPosition -win $_nWave1 {("G1" 7)}
wvSetPosition -win $_nWave1 {("G1" 6)}
wvSetPosition -win $_nWave1 {("G1" 5)}
wvSetPosition -win $_nWave1 {("G1" 4)}
wvSetPosition -win $_nWave1 {("G1" 3)}
wvSetPosition -win $_nWave1 {("G1" 2)}
wvMoveSelected -win $_nWave1
wvSetPosition -win $_nWave1 {("G1" 2)}
wvSetPosition -win $_nWave1 {("G1" 3)}
wvSelectSignal -win $_nWave1 {( "G1" 3 )} 
wvSelectSignal -win $_nWave1 {( "G1" 3 )} 
wvSelectSignal -win $_nWave1 {( "G1" 3 )} 
wvSelectSignal -win $_nWave1 {( "G1" 3 )} 
wvSelectSignal -win $_nWave1 {( "G1" 4 )} 
wvSelectSignal -win $_nWave1 {( "G1" 3 )} 
wvSetCursor -win $_nWave1 7743470.714702 -snap {("G4" 0)}
wvDisplayGridCount -win $_nWave1 -off
wvGetSignalClose -win $_nWave1
wvReloadFile -win $_nWave1
wvSetCursor -win $_nWave1 39133441.911997 -snap {("G4" 0)}
wvSetCursor -win $_nWave1 39109884.041560 -snap {("G4" 0)}
wvSetCursor -win $_nWave1 39146328.268475 -snap {("G4" 0)}
wvSelectSignal -win $_nWave1 {( "G1" 3 )} 
wvSelectSignal -win $_nWave1 {( "G1" 3 )} 
wvResizeWindow -win $_nWave1 1080 392 1920 1009
wvDisplayGridCount -win $_nWave1 -off
wvGetSignalClose -win $_nWave1
wvReloadFile -win $_nWave1
wvSelectSignal -win $_nWave1 {( "G2" 3 )} 
wvSelectSignal -win $_nWave1 {( "G2" 2 )} 
wvSelectSignal -win $_nWave1 {( "G2" 3 )} 
wvSelectSignal -win $_nWave1 {( "G1" 2 )} 
wvSelectSignal -win $_nWave1 {( "G2" 2 )} 
wvSelectSignal -win $_nWave1 {( "G2" 3 )} 
wvSetCursor -win $_nWave1 10454665.212364 -snap {("G1" 5)}
wvSelectSignal -win $_nWave1 {( "G2" 2 )} 
wvSetPosition -win $_nWave1 {("G2" 2)}
wvSetPosition -win $_nWave1 {("G3" 2)}
wvSetPosition -win $_nWave1 {("G3" 6)}
wvSetPosition -win $_nWave1 {("G3" 7)}
wvSetPosition -win $_nWave1 {("G4" 0)}
wvMoveSelected -win $_nWave1
wvSetPosition -win $_nWave1 {("G4" 1)}
wvSetPosition -win $_nWave1 {("G4" 1)}
wvSelectSignal -win $_nWave1 {( "G2" 2 )} 
wvSelectSignal -win $_nWave1 {( "G2" 1 )} 
wvSelectSignal -win $_nWave1 {( "G2" 2 )} 
wvSearchNext -win $_nWave1
wvSearchNext -win $_nWave1
wvSearchPrev -win $_nWave1
wvSearchPrev -win $_nWave1
wvSetCursor -win $_nWave1 74567.806130 -snap {("G2" 1)}
wvDisplayGridCount -win $_nWave1 -off
wvGetSignalClose -win $_nWave1
wvReloadFile -win $_nWave1
wvSetCursor -win $_nWave1 2744682.015396 -snap {("G2" 2)}
wvSelectSignal -win $_nWave1 {( "G2" 1 )} 
wvSelectSignal -win $_nWave1 {( "G2" 2 )} 
wvSelectSignal -win $_nWave1 {( "G4" 1 )} 
wvSetCursor -win $_nWave1 231205.075387 -snap {("G5" 0)}
wvDisplayGridCount -win $_nWave1 -off
wvGetSignalClose -win $_nWave1
wvReloadFile -win $_nWave1
wvResizeWindow -win $_nWave1 0 23 1080 1849
wvResizeWindow -win $_nWave1 0 23 1920 1009
wvResizeWindow -win $_nWave1 0 23 1512 877
wvResizeWindow -win $_nWave1 0 23 1512 925
wvResizeWindow -win $_nWave1 0 23 1512 877
wvResizeWindow -win $_nWave1 0 23 1920 1009
wvSetCursor -win $_nWave1 5992462.143946 -snap {("G5" 0)}
wvExit
