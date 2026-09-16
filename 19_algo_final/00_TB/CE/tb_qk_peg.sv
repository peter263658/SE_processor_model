/*
    tb_qk_peg.sv — bit-exact unit test of the CE QK^T attention datapath (MODE_LMUL_ATTN).

    For query t*=351, per (head, key): score = L-Mul m8 dot over head_dim=16, reduced by the PEG's
    FP16 paired saturating tree (= the HARDWARE QK accumulator, user-chosen fp16-tree), then × 2^-2
    (1/sqrt(16)) via PPU SCALE. Drives the golden Q + K-window operands (mha_module_io) and checks:
      PEG.o_accum            vs qk_prescale  (fp16-tree, pre ×2^-2)
      ppu_fma(PPU_SCALE,2^-2) vs qk_postscale (× 2^-2)
    The model (attn_lmul_m8_ref.qk_score acc="tree") already == staged fp16tree golden 512/512; this
    proves RTL == model. Operand map: head_dim d -> PEG lane 2*d (the pe_pair LO; HI forced 0 in attn).

    +GDIR=<golden qk dir>   (e.g. .../golden/qk_real/ or qk_imag/)
*/
`timescale 1ns/10ps
import Param_def::*;

module tb_qk_peg;
    localparam int NHEAD=16, NKEY=32, HD=16, N=NHEAD*NKEY;   // 512
    localparam logic [1:0] PPU_SCALE = 2'd2;
    localparam logic [11:0] SCALE_2M2 = 12'hB00;             // apot6a a=-2,h2=0 -> ×2^-2 (1/sqrt(16))

    // ---- PEG (MODE_LMUL_ATTN) ----
    CE_LANE_mode mode;
    logic [NUM_PE_PG-1:0][15:0] i_x, i_y;
    logic signed [7:0] preshift;
    logic [15:0] peg_accum;
    PEG u_peg (.i_mode(mode), .i_x(i_x), .i_y(i_y), .i_preshift(preshift), .o_accum(peg_accum));

    // ---- PPU SCALE (×2^-2) on the PEG prescale ----
    logic [15:0] ppu_re, ppu_im;
    ppu_fma u_ppu (
        .i_mode(PPU_SCALE), .i_re(peg_accum), .i_im(16'h0000),
        .i_bn_alpha_re(12'hC00), .i_bn_alpha_im(12'hC00), .i_bn_beta_re(16'h0), .i_bn_beta_im(16'h0),
        .i_prelu_re(12'hC00), .i_prelu_im(12'hC00), .i_cast_k(8'sd0),
        .i_scale_re(SCALE_2M2), .i_scale_im(SCALE_2M2), .o_re(ppu_re), .o_im(ppu_im)
    );

    logic [15:0] Q [0:NHEAD*HD-1], Kwin [0:NHEAD*NKEY*HD-1], gpre [0:N-1], gpost [0:N-1];
    string GDIR;
    integer epre, epost, h, k, d, idx;

    initial begin
        if (!$value$plusargs("GDIR=%s", GDIR)) begin $display("[TB] need +GDIR"); $finish; end
        $readmemh({GDIR,"qk_Q.hex"},        Q);
        $readmemh({GDIR,"qk_Kwin.hex"},     Kwin);
        $readmemh({GDIR,"qk_prescale.hex"}, gpre);
        $readmemh({GDIR,"qk_postscale.hex"},gpost);
        mode = MODE_LMUL_ATTN; preshift = 8'sd0;
        $display("[TB] qk_peg: GDIR=%s  MODE_LMUL_ATTN, preshift=0, PPU_SCALE 0xB00 (×2^-2)", GDIR);

        epre=0; epost=0;
        for (h=0; h<NHEAD; h++) begin
            for (k=0; k<NKEY; k++) begin
                idx = h*NKEY + k;
                for (d=0; d<NUM_PE_PG; d++) begin i_x[d]=16'h0; i_y[d]=16'h0; end
                for (d=0; d<HD; d++) begin                       // head_dim d -> lane 2*d (LO); HI=0
                    i_x[2*d] = Q[h*HD + d];
                    i_y[2*d] = Kwin[(h*NKEY+k)*HD + d];
                end
                #1;
                if (peg_accum !== gpre[idx]) begin
                    epre++; if (epre<=8) $display("[TB] PRE  h%0d k%0d got %04h exp %04h", h, k, peg_accum, gpre[idx]);
                end
                if (ppu_re !== gpost[idx]) begin
                    epost++; if (epost<=8) $display("[TB] POST h%0d k%0d got %04h exp %04h", h, k, ppu_re, gpost[idx]);
                end
                #1;
            end
        end
        $display("[TB] QK datapath: prescale %0d/%0d, postscale %0d/%0d match (L-Mul m8 + fp16-tree + ×2^-2)",
                 N-epre, N, N-epost, N);
        if (epre==0 && epost==0) $display("[TB] ALL PASS (QK datapath bit-exact vs fp16-tree golden)");
        else $display("[TB] FAIL pre=%0d post=%0d", epre, epost);
        $finish;
    end
endmodule
