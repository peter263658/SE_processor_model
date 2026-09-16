wvSetPosition -win $_nWave1 {("G1" 0)}
wvOpenFile -win $_nWave1 {/home/r12k41024/Research/00_TB/WBUF/tb_wbuf.fsdb}
verdiSetActWin -win $_nWave1
wvGetSignalOpen -win $_nWave1
wvGetSignalSetScope -win $_nWave1 "/tb_wbuf"
wvGetSignalClose -win $_nWave1
wvSetCursor -win $_nWave1 108602150.537634
wvGetSignalOpen -win $_nWave1
wvGetSignalSetScope -win $_nWave1 "/tb_wbuf"
wvGetSignalSetScope -win $_nWave1 "/tb_wbuf"
wvGetSignalSetScope -win $_nWave1 "/tb_wbuf/u_wbuf_top"
wvGetSignalSetScope -win $_nWave1 "/tb_wbuf/Unnamed_\$tb_wbuf_sv_160"
wvGetSignalSetScope -win $_nWave1 "/tb_wbuf/Unnamed_\$tb_wbuf_sv_105"
wvGetSignalSetScope -win $_nWave1 "/tb_wbuf/u_wbuf_top"
wvGetSignalSetScope -win $_nWave1 "/tb_wbuf"
wvGetSignalSetScope -win $_nWave1 "/tb_wbuf/u_wbuf_top/u_wbuf_ctrl_inst"
wvGetSignalSetScope -win $_nWave1 "/tb_wbuf/u_wbuf_top"
wvGetSignalSetScope -win $_nWave1 "/tb_wbuf/u_wbuf_top/u_wbuf_sram_inst"
wvGetSignalSetScope -win $_nWave1 "/tb_wbuf/u_wbuf_top"
wvGetSignalSetScope -win $_nWave1 "/tb_wbuf/Unnamed_\$tb_wbuf_sv_105"
wvGetSignalSetScope -win $_nWave1 "/tb_wbuf"
wvGetSignalSetScope -win $_nWave1 "/tb_wbuf/Unnamed_\$tb_wbuf_sv_160"
wvGetSignalSetScope -win $_nWave1 "/tb_wbuf/Unnamed_\$tb_wbuf_sv_105"
wvGetSignalSetScope -win $_nWave1 "/tb_wbuf/Unnamed_\$tb_wbuf_sv_160"
wvGetSignalSetScope -win $_nWave1 "/tb_wbuf/u_wbuf_top"
wvGetSignalSetScope -win $_nWave1 "/tb_wbuf/Unnamed_\$tb_wbuf_sv_160"
wvGetSignalSetScope -win $_nWave1 "/tb_wbuf/u_wbuf_top"
wvDisplayGridCount -win $_nWave1 -off
wvCloseGetStreamsDialog -win $_nWave1
wvAttrOrderConfigDlg -win $_nWave1 -close
wvCloseDetailsViewDlg -win $_nWave1
wvCloseDetailsViewDlg -win $_nWave1 -streamLevel
wvCloseFilterColorizeDlg -win $_nWave1
wvGetSignalClose -win $_nWave1
wvReloadFile -win $_nWave1
wvGetSignalSetScope -win $_nWave1 "/tb_wbuf/u_wbuf_top/u_wbuf_ctrl_inst"
wvGetSignalSetScope -win $_nWave1 "/tb_wbuf/u_wbuf_top"
wvGetSignalSetScope -win $_nWave1 "/tb_wbuf/u_wbuf_top/u_wbuf_sram_inst"
wvGetSignalSetScope -win $_nWave1 "/tb_wbuf/u_wbuf_top"
wvSetPosition -win $_nWave1 {("G1" 1)}
wvSetPosition -win $_nWave1 {("G1" 1)}
wvAddSignal -win $_nWave1 -clear
wvAddSignal -win $_nWave1 -group {"G1" \
{/tb_wbuf/u_wbuf_top/i_clk} \
}
wvAddSignal -win $_nWave1 -group {"G2" \
}
wvSelectSignal -win $_nWave1 {( "G1" 1 )} 
wvSetPosition -win $_nWave1 {("G1" 1)}
wvSetPosition -win $_nWave1 {("G1" 2)}
wvSetPosition -win $_nWave1 {("G1" 2)}
wvAddSignal -win $_nWave1 -clear
wvAddSignal -win $_nWave1 -group {"G1" \
{/tb_wbuf/u_wbuf_top/i_clk} \
{/tb_wbuf/u_wbuf_top/i_rst_n} \
}
wvAddSignal -win $_nWave1 -group {"G2" \
}
wvSelectSignal -win $_nWave1 {( "G1" 2 )} 
wvSetPosition -win $_nWave1 {("G1" 2)}
wvSetPosition -win $_nWave1 {("G1" 5)}
wvSetPosition -win $_nWave1 {("G1" 5)}
wvAddSignal -win $_nWave1 -clear
wvAddSignal -win $_nWave1 -group {"G1" \
{/tb_wbuf/u_wbuf_top/i_clk} \
{/tb_wbuf/u_wbuf_top/i_rst_n} \
{/tb_wbuf/u_wbuf_top/i_ae_mode\[2:0\]} \
{/tb_wbuf/u_wbuf_top/i_ce_layer\[2:0\]} \
{/tb_wbuf/u_wbuf_top/i_ce_mode\[1:0\]} \
}
wvAddSignal -win $_nWave1 -group {"G2" \
}
wvSelectSignal -win $_nWave1 {( "G1" 3 4 5 )} 
wvSetPosition -win $_nWave1 {("G1" 5)}
wvSetPosition -win $_nWave1 {("G1" 6)}
wvSetPosition -win $_nWave1 {("G1" 6)}
wvAddSignal -win $_nWave1 -clear
wvAddSignal -win $_nWave1 -group {"G1" \
{/tb_wbuf/u_wbuf_top/i_clk} \
{/tb_wbuf/u_wbuf_top/i_rst_n} \
{/tb_wbuf/u_wbuf_top/i_ae_mode\[2:0\]} \
{/tb_wbuf/u_wbuf_top/i_ce_layer\[2:0\]} \
{/tb_wbuf/u_wbuf_top/i_ce_mode\[1:0\]} \
{/tb_wbuf/u_wbuf_top/i_exec_mode\[2:0\]} \
}
wvAddSignal -win $_nWave1 -group {"G2" \
}
wvSelectSignal -win $_nWave1 {( "G1" 6 )} 
wvSetPosition -win $_nWave1 {("G1" 6)}
wvSetPosition -win $_nWave1 {("G1" 9)}
wvSetPosition -win $_nWave1 {("G1" 9)}
wvAddSignal -win $_nWave1 -clear
wvAddSignal -win $_nWave1 -group {"G1" \
{/tb_wbuf/u_wbuf_top/i_clk} \
{/tb_wbuf/u_wbuf_top/i_rst_n} \
{/tb_wbuf/u_wbuf_top/i_ae_mode\[2:0\]} \
{/tb_wbuf/u_wbuf_top/i_ce_layer\[2:0\]} \
{/tb_wbuf/u_wbuf_top/i_ce_mode\[1:0\]} \
{/tb_wbuf/u_wbuf_top/i_exec_mode\[2:0\]} \
{/tb_wbuf/u_wbuf_top/i_top_data\[127:0\]} \
{/tb_wbuf/u_wbuf_top/i_top_valid} \
{/tb_wbuf/u_wbuf_top/o_top_ready} \
}
wvAddSignal -win $_nWave1 -group {"G2" \
}
wvSelectSignal -win $_nWave1 {( "G1" 7 8 9 )} 
wvSetPosition -win $_nWave1 {("G1" 9)}
wvSetPosition -win $_nWave1 {("G1" 9)}
wvSetPosition -win $_nWave1 {("G1" 9)}
wvAddSignal -win $_nWave1 -clear
wvAddSignal -win $_nWave1 -group {"G1" \
{/tb_wbuf/u_wbuf_top/i_clk} \
{/tb_wbuf/u_wbuf_top/i_rst_n} \
{/tb_wbuf/u_wbuf_top/i_ae_mode\[2:0\]} \
{/tb_wbuf/u_wbuf_top/i_ce_layer\[2:0\]} \
{/tb_wbuf/u_wbuf_top/i_ce_mode\[1:0\]} \
{/tb_wbuf/u_wbuf_top/i_exec_mode\[2:0\]} \
{/tb_wbuf/u_wbuf_top/i_top_data\[127:0\]} \
{/tb_wbuf/u_wbuf_top/i_top_valid} \
{/tb_wbuf/u_wbuf_top/o_top_ready} \
}
wvAddSignal -win $_nWave1 -group {"G2" \
}
wvSelectSignal -win $_nWave1 {( "G1" 7 8 9 )} 
wvSetPosition -win $_nWave1 {("G1" 9)}
wvGetSignalClose -win $_nWave1
wvZoomIn -win $_nWave1
wvZoomIn -win $_nWave1
wvZoomIn -win $_nWave1
wvZoomIn -win $_nWave1
wvZoomIn -win $_nWave1
wvZoomIn -win $_nWave1
wvSelectNet -win $_nWave1
wvSetCursor -win $_nWave1 466649.865591 -snap {("G2" 0)}
wvSelectGroup -win $_nWave1 {G2}
wvSaveSignal -win $_nWave1 "/home/r12k41024/Research/00_TB/WBUF/tb_wbuf.rc"
wvResizeWindow -win $_nWave1 8 31 1902 991
wvSetCursor -win $_nWave1 1940871.113546 -snap {("G2" 0)}
wvDisplayGridCount -win $_nWave1 -off
wvCloseGetStreamsDialog -win $_nWave1
wvAttrOrderConfigDlg -win $_nWave1 -close
wvCloseDetailsViewDlg -win $_nWave1
wvCloseDetailsViewDlg -win $_nWave1 -streamLevel
wvCloseFilterColorizeDlg -win $_nWave1
wvGetSignalClose -win $_nWave1
wvReloadFile -win $_nWave1
wvSetCursor -win $_nWave1 2078747.203579 -snap {("G2" 0)}
wvDisplayGridCount -win $_nWave1 -off
wvCloseGetStreamsDialog -win $_nWave1
wvAttrOrderConfigDlg -win $_nWave1 -close
wvCloseDetailsViewDlg -win $_nWave1
wvCloseDetailsViewDlg -win $_nWave1 -streamLevel
wvCloseFilterColorizeDlg -win $_nWave1
wvGetSignalClose -win $_nWave1
wvReloadFile -win $_nWave1
wvDisplayGridCount -win $_nWave1 -off
wvCloseGetStreamsDialog -win $_nWave1
wvAttrOrderConfigDlg -win $_nWave1 -close
wvCloseDetailsViewDlg -win $_nWave1
wvCloseDetailsViewDlg -win $_nWave1 -streamLevel
wvCloseFilterColorizeDlg -win $_nWave1
wvGetSignalClose -win $_nWave1
wvReloadFile -win $_nWave1
wvDisplayGridCount -win $_nWave1 -off
wvCloseGetStreamsDialog -win $_nWave1
wvAttrOrderConfigDlg -win $_nWave1 -close
wvCloseDetailsViewDlg -win $_nWave1
wvCloseDetailsViewDlg -win $_nWave1 -streamLevel
wvCloseFilterColorizeDlg -win $_nWave1
wvGetSignalClose -win $_nWave1
wvReloadFile -win $_nWave1
wvDisplayGridCount -win $_nWave1 -off
wvCloseGetStreamsDialog -win $_nWave1
wvAttrOrderConfigDlg -win $_nWave1 -close
wvCloseDetailsViewDlg -win $_nWave1
wvCloseDetailsViewDlg -win $_nWave1 -streamLevel
wvCloseFilterColorizeDlg -win $_nWave1
wvGetSignalClose -win $_nWave1
wvReloadFile -win $_nWave1
wvDisplayGridCount -win $_nWave1 -off
wvCloseGetStreamsDialog -win $_nWave1
wvAttrOrderConfigDlg -win $_nWave1 -close
wvCloseDetailsViewDlg -win $_nWave1
wvCloseDetailsViewDlg -win $_nWave1 -streamLevel
wvCloseFilterColorizeDlg -win $_nWave1
wvGetSignalClose -win $_nWave1
wvReloadFile -win $_nWave1
wvResizeWindow -win $_nWave1 0 0 1494 884
wvResizeWindow -win $_nWave1 9 32 1494 860
wvDisplayGridCount -win $_nWave1 -off
wvCloseGetStreamsDialog -win $_nWave1
wvAttrOrderConfigDlg -win $_nWave1 -close
wvCloseDetailsViewDlg -win $_nWave1
wvCloseDetailsViewDlg -win $_nWave1 -streamLevel
wvCloseFilterColorizeDlg -win $_nWave1
wvGetSignalClose -win $_nWave1
wvReloadFile -win $_nWave1
wvDisplayGridCount -win $_nWave1 -off
wvCloseGetStreamsDialog -win $_nWave1
wvAttrOrderConfigDlg -win $_nWave1 -close
wvCloseDetailsViewDlg -win $_nWave1
wvCloseDetailsViewDlg -win $_nWave1 -streamLevel
wvCloseFilterColorizeDlg -win $_nWave1
wvGetSignalClose -win $_nWave1
wvReloadFile -win $_nWave1
wvDisplayGridCount -win $_nWave1 -off
wvCloseGetStreamsDialog -win $_nWave1
wvAttrOrderConfigDlg -win $_nWave1 -close
wvCloseDetailsViewDlg -win $_nWave1
wvCloseDetailsViewDlg -win $_nWave1 -streamLevel
wvCloseFilterColorizeDlg -win $_nWave1
wvGetSignalClose -win $_nWave1
wvReloadFile -win $_nWave1
wvDisplayGridCount -win $_nWave1 -off
wvCloseGetStreamsDialog -win $_nWave1
wvAttrOrderConfigDlg -win $_nWave1 -close
wvCloseDetailsViewDlg -win $_nWave1
wvCloseDetailsViewDlg -win $_nWave1 -streamLevel
wvCloseFilterColorizeDlg -win $_nWave1
wvGetSignalClose -win $_nWave1
wvReloadFile -win $_nWave1
wvGetSignalOpen -win $_nWave1
wvSetCursor -win $_nWave1 1286630.937259 -snap {("G2" 0)}
wvSetCursor -win $_nWave1 1467024.553204 -snap {("G2" 0)}
