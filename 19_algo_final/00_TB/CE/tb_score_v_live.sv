/*
    tb_score_v_live.sv — LIVE Score×V through ce_engine (sub-op 33).
    oc=256=context[head*16+d] (head=oc[7:4], d=oc[3:0]). Preloads the per-head 4 packet V-frames
    (vbuf_all) + packet prob (sv_prob) [TB-fed now; live gather-from-KVBUF + ce_attn_sfu packet is next].
    ce_engine assembles 4 products prob[s]·V[idx[s]][d] on PEG lanes 0,2,4,6 → MODE_LMUL_ATTN → PPU_PASS
    → o_re = context fp16. Samples o_re per oc vs sv_ctx golden. (score_v datapath already 256/256
    standalone; this proves the live operand routing through ce_engine.)  +GDIR=<golden qk dir>.
*/
`timescale 1ns/10ps
import Param_def::*;

module tb_score_v_live;
    localparam int WBUF_BYTES=262144, N=256, NH=16, SAMPLE_LAT=4, SV_SUBOP=33;  // o_re = PPU out (+RESULT_LAT=4)
    logic i_clk=0, i_rst_n; always #5 i_clk=~i_clk;
    logic i_start, o_busy, o_done, o_rv; logic [15:0] o_re, o_im;
    ce_engine u_eng(.i_clk(i_clk),.i_rst_n(i_rst_n),.i_start(i_start),.o_busy(o_busy),.o_done(o_done),
                    .o_re(o_re),.o_im(o_im),.o_result_valid(o_rv));

    localparam string GD="../../10_algo/hw_patterns_final/sram_images/golden/qk_real/";
    localparam string IMG="../../10_algo/hw_patterns_final/sram_images/wbuf_unified_init.bin";
    logic [7:0] img [0:WBUF_BYTES-1];
    logic [15:0] vbuf [0:NH*4*16-1], prob [0:NH*4-1], gctx [0:N-1], got [0:N-1];
    integer fd,nrd,b;

    initial begin
        i_rst_n=1; i_start=0;
        for(int o=0;o<N;o++) got[o]=16'hx;
        fd=$fopen(IMG,"rb"); nrd=$fread(img,fd); $fclose(fd);
        for(b=0;b<WBUF_BYTES;b++) u_eng.wbuf_mem[b]=img[b];
        $readmemh({GD,"sv_vbuf.hex"}, vbuf); $readmemh({GD,"sv_prob.hex"}, prob); $readmemh({GD,"sv_ctx.hex"}, gctx);
        for(b=0;b<NH*4*16;b++) u_eng.vbuf_all[b]=vbuf[b];
        for(b=0;b<NH*4;b++)    u_eng.packet_prob[b]=prob[b];
        $display("[TB] score_v_live: image + vbuf(%0d) + prob(%0d) loaded, golden %0d ctx",NH*4*16,NH*4,N);
        repeat(2)@(negedge i_clk); i_rst_n=0; repeat(2)@(negedge i_clk); i_rst_n=1; repeat(2)@(negedge i_clk);
        u_eng.u_kv.filled = 6'd32;  // steady-state: full-window mask (TB preloads full history)
        @(negedge i_clk); i_start=1; @(negedge i_clk); i_start=0;
    end

    wire [6:0] dsub=u_eng.u_id.subop_idx;
    wire [9:0] doc =u_eng.u_id.oc_idx;
    wire       docc=u_eng.u_id.o_oc_complete;
    logic [15:0] occ_sr; logic [9:0] oc_sr[0:15]; logic [6:0] sub_sr[0:15];
    always @(posedge i_clk) begin
        occ_sr<={occ_sr[14:0],(i_rst_n&docc)}; oc_sr[0]<=doc; sub_sr[0]<=dsub;
        for(int k=1;k<16;k++) begin oc_sr[k]<=oc_sr[k-1]; sub_sr[k]<=sub_sr[k-1]; end
        if(occ_sr[SAMPLE_LAT-1] && sub_sr[SAMPLE_LAT-1]==SV_SUBOP && oc_sr[SAMPLE_LAT-1]<N)
            got[oc_sr[SAMPLE_LAT-1]]<=o_re;
    end

    task automatic report; int er; begin
        er=0;
        for(int oc=0;oc<N;oc++) if(got[oc]!==gctx[oc]) begin er++; if(er<=10) $display("[TB] oc%0d (h%0d d%0d) got %04h exp %04h",oc,oc>>4,oc&15,got[oc],gctx[oc]); end
        $display("[TB] LIVE Score×V: %0d/%0d context match (prob × V[idx] -> PEG -> PPU_PASS)",N-er,N);
        if(er==0) $display("[TB] ALL PASS (live Score×V operand routing bit-exact)");
        else $display("[TB] FAIL er=%0d",er);
        $finish; end
    endtask

    initial begin
        @(posedge i_rst_n); wait(i_start===1'b1);
        fork
            begin wait((u_eng.u_id.subop_idx>SV_SUBOP)||(o_done===1'b1)); repeat(8)@(posedge i_clk); report(); end
            begin repeat(400000)@(posedge i_clk); $display("[TB] TIMEOUT subop=%0d",u_eng.u_id.subop_idx); report(); end
        join_any
        $finish;
    end
endmodule
