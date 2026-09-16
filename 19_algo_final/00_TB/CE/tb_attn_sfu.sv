/*
    tb_attn_sfu.sv — verify ce_attn_sfu (softmax + top-4 packet glue) on the live QK postscale scores.
    Writes the 16-head × 32-frame QK postscale golden into score_buf, runs the FSM (16× Softmax_v3),
    and checks each head's extracted top-4 packet {idx, prob} (as a SET, order-independent) vs the golden
    packet (idx + dense fp16 prob). Softmax_v3 itself is already verified 512/512; this proves the
    score-collect + per-head FSM handshake + top-4 extraction.
    +GDIR=<golden qk dir>.
*/
`timescale 1ns/10ps
import Param_def::*;

module tb_attn_sfu;
    localparam int NH=NUM_HEAD, NK=SEQ_LEN, HB=$clog2(NUM_HEAD), FB=$clog2(SEQ_LEN);

    logic i_clk=0, i_rst_n; always #5 i_clk=~i_clk;
    logic i_wr_en, i_start, o_done;
    logic [HB-1:0] i_wr_head;
    logic [FB-1:0] i_wr_frame;
    logic [ATTN_FP_WIDTH-1:0] i_wr_score;
    logic [NH-1:0][3:0][FB-1:0] o_pidx;
    logic [NH-1:0][3:0][ATTN_FP_WIDTH-1:0] o_pprob;

    ce_attn_sfu u_sfu (.i_clk(i_clk), .i_rst_n(i_rst_n),
        .i_wr_en(i_wr_en), .i_wr_head(i_wr_head), .i_wr_frame(i_wr_frame), .i_wr_score(i_wr_score),
        .i_start(i_start), .o_done(o_done), .i_valid_mask({32{1'b1}}), .o_pidx(o_pidx), .o_pprob(o_pprob));

    logic [15:0] post [0:NH*NK-1], dense [0:NH*NK-1];
    logic [7:0]  pidx [0:NH*4-1], pcode [0:NH*4-1];
    string GDIR; integer h,s,f, err;

    // does HW slot (idx,prob) exist in the golden set for head h?
    function automatic bit in_gold(input int h, input logic [FB-1:0] idx, input logic [15:0] prob);
        in_gold=0;
        for (int g=0; g<4; g++)
            if (pcode[h*4+g]!=0 && pidx[h*4+g]==idx && dense[h*NK+idx]==prob) in_gold=1;
    endfunction

    initial begin
        if(!$value$plusargs("GDIR=%s",GDIR)) begin $display("[TB] need +GDIR"); $finish; end
        $readmemh({GDIR,"qk_postscale.hex"}, post);
        $readmemh({GDIR,"sm_dense.hex"},     dense);
        $readmemh({GDIR,"packet_idx.hex"},   pidx);
        $readmemh({GDIR,"packet_code.hex"},  pcode);
        i_wr_en=0; i_start=0; i_rst_n=1;
        repeat(2)@(negedge i_clk); i_rst_n=0; repeat(2)@(negedge i_clk); i_rst_n=1; repeat(2)@(negedge i_clk);
        // fill score_buf
        for(h=0;h<NH;h++) for(f=0;f<NK;f++) begin
            i_wr_en=1; i_wr_head=h[HB-1:0]; i_wr_frame=f[FB-1:0]; i_wr_score=post[h*NK+f]; @(negedge i_clk);
        end
        i_wr_en=0; @(negedge i_clk);
        i_start=1; @(negedge i_clk); i_start=0;
        wait(o_done===1'b1); @(negedge i_clk);

        err=0;
        for(h=0;h<NH;h++) begin
            int nhw, ngold;
            nhw=0; ngold=0;
            for(s=0;s<4;s++) begin
                if(o_pprob[h][s]!=0) begin nhw++; if(!in_gold(h,o_pidx[h][s],o_pprob[h][s])) begin err++; if(err<=8) $display("[TB] h%0d slot%0d idx%0d prob %04h NOT in golden",h,s,o_pidx[h][s],o_pprob[h][s]); end end
                if(pcode[h*4+s]!=0) ngold++;
            end
            if(nhw!=ngold) begin err++; if(err<=8) $display("[TB] h%0d packet count HW %0d != gold %0d",h,nhw,ngold); end
        end
        $display("[TB] attn_sfu: %0d heads, packet errors=%0d",NH,err);
        if(err==0) $display("[TB] ALL PASS (softmax + top-4 packet glue: live QK scores -> golden packet)");
        else $display("[TB] FAIL err=%0d",err);
        $finish;
    end
    initial begin repeat(20000)@(posedge i_clk); $display("[TB] TIMEOUT done=%b",o_done); $finish; end
endmodule
