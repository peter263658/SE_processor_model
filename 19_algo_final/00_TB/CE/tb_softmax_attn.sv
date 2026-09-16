/*
    tb_softmax_attn.sv — verify Softmax_v3 (the deployed shift_topk_pow2_clamp8 softmax) on the REAL
    attention scores: feed the QK postscale scores (fp16-tree golden) for each of the 16 heads, mask =
    full window (all valid), and check the dense-32 FP16 output bit-exact vs the deploy golden softmax
    (mha_module_io/03_softmax). This is the deploy-softmax == Softmax_v3 cross-check on real data (the
    existing 00_TB/SoftMax/01_run checks vs tb_SM.py; this checks vs the deploy MHA dump).

    +GDIR=<golden qk dir>  (qk_postscale.hex 16×32, sm_dense.hex 16×32)
*/
`timescale 1ns/10ps
import Param_def::*;

module tb_softmax_attn;
    localparam int NHEAD=16, NKEY=32;

    logic i_clk=0, i_rst_n; always #5 i_clk=~i_clk;
    logic i_SM_valid, o_SM_ready, o_SM_valid;
    logic [SEQ_LEN-1:0][ATTN_FP_WIDTH-1:0] i_SM_data, o_SM_data;
    logic [SEQ_LEN-1:0] i_SM_mask;

    Softmax u_sm (
        .i_clk(i_clk), .i_rst_n(i_rst_n),
        .i_SM_valid(i_SM_valid), .i_SM_data(i_SM_data), .i_SM_mask(i_SM_mask),
        .o_SM_ready(o_SM_ready), .o_SM_valid(o_SM_valid), .o_SM_data(o_SM_data)
    );

    logic [15:0] post [0:NHEAD*NKEY-1], dense [0:NHEAD*NKEY-1];
    string GDIR;
    integer err, herr, h, j;

    initial begin
        if (!$value$plusargs("GDIR=%s", GDIR)) begin $display("[TB] need +GDIR"); $finish; end
        $readmemh({GDIR,"qk_postscale.hex"}, post);
        $readmemh({GDIR,"sm_dense.hex"},     dense);
        i_SM_valid=0; i_SM_data='0; i_SM_mask='0; i_rst_n=1;
        repeat(2)@(negedge i_clk); i_rst_n=0; repeat(2)@(negedge i_clk); i_rst_n=1; repeat(2)@(negedge i_clk);

        err=0;
        for (h=0; h<NHEAD; h++) begin
            // wait until ready, then present this head's 32 postscale scores (full-window mask)
            while (!o_SM_ready) @(negedge i_clk);
            for (j=0;j<SEQ_LEN;j++) i_SM_data[j] = post[h*NKEY+j];
            i_SM_mask = {SEQ_LEN{1'b1}};
            i_SM_valid = 1'b1; @(negedge i_clk); i_SM_valid = 1'b0;
            while (!o_SM_valid) @(negedge i_clk);
            herr=0;
            for (j=0;j<SEQ_LEN;j++)
                if (o_SM_data[j] !== dense[h*NKEY+j]) begin
                    herr++; if (err+herr<=10) $display("[TB] h%0d j%0d got %04h exp %04h", h, j, o_SM_data[j], dense[h*NKEY+j]);
                end
            err += herr;
            @(negedge i_clk);   // let FSM return to IDLE
        end
        $display("[TB] Softmax_v3 vs deploy dense softmax: %0d/%0d entries match over %0d heads",
                 NHEAD*NKEY-err, NHEAD*NKEY, NHEAD);
        if (err==0) $display("[TB] ALL PASS (Softmax_v3 == deploy softmax on real QK scores)");
        else $display("[TB] FAIL err=%0d", err);
        $finish;
    end

    initial begin repeat(20000)@(posedge i_clk); $display("[TB] TIMEOUT"); $finish; end
endmodule
