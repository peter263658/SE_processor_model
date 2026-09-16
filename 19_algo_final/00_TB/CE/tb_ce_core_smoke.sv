/*
    tb_ce_core_smoke.sv — compile + basic sanity for the CE_core datapath spine.
    NOT a golden test (golden/bit-exact accumulation comes later with the controller).

    Drives one W4A8 accumulate: PEG0 = 32 × (q=1 × fp8 1.0) = 32.0; PEG1..3 act=0.
    dealer = 2^0 (×1); OACC clr+en for one cycle (re=P0−P1=32, im=P2+P3=0);
    PPU mode = PASS (fp16). Expect o_re == fp16(32.0)=0x5000, o_im == 0.
*/
`timescale 1ns/10ps
module tb_ce_core_smoke;
    import Param_def::*;

    logic i_clk, i_rst_n;
    CE_LANE_mode i_pe_mode;
    logic [NUM_PEG-1:0][NUM_PE_PG-1:0][15:0] i_x, i_y;
    logic [NUM_PEG-1:0][11:0] i_dealer_apot;
    logic [NUM_PEG-1:0][7:0]  i_preshift;
    logic i_oacc_en, i_oacc_clr;
    logic [1:0]  i_ppu_mode;
    logic [11:0] i_bn_alpha_re, i_bn_alpha_im, i_prelu_re, i_prelu_im, i_scale_re, i_scale_im;
    logic [15:0] i_bn_beta_re, i_bn_beta_im;
    logic signed [7:0] i_cast_k;
    logic [15:0] o_re, o_im;

    CE_core u_dut (
        .i_clk(i_clk), .i_rst_n(i_rst_n),
        .i_pe_mode(i_pe_mode), .i_x(i_x), .i_y(i_y),
        .i_dealer_apot(i_dealer_apot), .i_preshift(i_preshift),
        .i_oacc_en(i_oacc_en), .i_oacc_clr(i_oacc_clr),
        .i_ppu_mode(i_ppu_mode),
        .i_bn_alpha_re(i_bn_alpha_re), .i_bn_alpha_im(i_bn_alpha_im),
        .i_bn_beta_re(i_bn_beta_re), .i_bn_beta_im(i_bn_beta_im),
        .i_prelu_re(i_prelu_re), .i_prelu_im(i_prelu_im),
        .i_cast_k(i_cast_k), .i_scale_re(i_scale_re), .i_scale_im(i_scale_im),
        .o_re(o_re), .o_im(o_im)
    );

    initial i_clk = 0;
    always #5 i_clk = ~i_clk;

    initial begin
        i_rst_n = 0;
        i_pe_mode = MODE_W4A8;
        for (int p = 0; p < NUM_PEG; p++)
            for (int l = 0; l < NUM_PE_PG; l++) begin
                i_x[p][l] = 16'h0001;                               // INT4 weight = +1
                i_y[p][l] = (p == 0) ? 16'h0038 : 16'h0000;         // fp8 1.0 on PEG0 only
            end
        for (int p = 0; p < NUM_PEG; p++) i_dealer_apot[p] = 12'hC00;  // 2^0 (a=0,no 2nd)
        i_preshift = '0;                                              // bit-exact: no pre-tree shift
        i_oacc_en = 0; i_oacc_clr = 0;
        i_ppu_mode = 2'd1;                                          // PPU_PASS
        i_bn_alpha_re=0; i_bn_alpha_im=0; i_bn_beta_re=0; i_bn_beta_im=0;
        i_prelu_re=0; i_prelu_im=0; i_cast_k=0; i_scale_re=0; i_scale_im=0;

        @(posedge i_clk); #1 i_rst_n = 1;
        repeat (2) @(posedge i_clk);                          // settle PEG output reg (stage 1)
        // one accumulate cycle: load (clr) + advance (en)
        @(posedge i_clk); #1 i_oacc_en = 1; i_oacc_clr = 1;
        @(posedge i_clk); #1 i_oacc_en = 0; i_oacc_clr = 0;   // OACC loads this edge
        repeat (3) @(posedge i_clk);                          // drain OACC→PPU→o_re reg (stage 3)
        #1;

        $display("[smoke] o_re=%h o_im=%h (expect re=5000 im=0000)", o_re, o_im);
        if (o_re === 16'h5000 && o_im === 16'h0000)
            $display("[TB] ALL PASS");
        else
            $fatal(1, "[TB] *** FAIL: o_re=%h o_im=%h", o_re, o_im);
        $finish;
    end
endmodule
