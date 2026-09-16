/*
    CE_core.sv — W4A8 zero-multiplier Compute Engine datapath SPINE.

    Wires the post-PE compute chain from CE_DATAPATH_DRAFT.md:

        4× PEG (pe_pair + 32-way FP16 saturating tree)
          → 4× dealer  (per-PEG APoT 2^a±2^b fused descale, 雷②)
          → 1× oacc    (re/im 2-bank: re+=P0−P1, im+=P2+P3; PEG-per-stream §2)
          → 1× ppu_fma (mode-select: CONV BN→PReLU→e4m3 / PASS / SCALE, 雷③)

    SCOPE = DATAPATH only. Operands, per-PEG descale codes, OACC accumulate control,
    and PPU constants are PORTS — driven by the (separate) memory-control / layer-
    control module + WBUF/UFBUF/KVBUF/SKIPBUF instances (the NEXT step, user-deferred).
    The controller routes i_x/i_y per PEG to realise PEG-per-stream
    (PEG0=fr·xr  PEG1=fi·xi  PEG2=fr·xi  PEG3=fi·xr) and the 64-lane attn modes.

    NOTE: legacy `CE.sv` (FP16 engine, module `CE`) is left intact; this is the new
    W4A8 spine (module `CE_core`). Pure-comb PE+dealer+ppu; only OACC holds state.
*/
`timescale 1ns/10ps
module CE_core
    import Param_def::*;
(
    input  logic                                         i_clk,
    input  logic                                         i_rst_n,

    // ---- PE operands (per PEG; controller routes weight/act per stream) ----
    input  CE_LANE_mode                                  i_pe_mode,
    input  logic [NUM_PEG-1:0][NUM_PE_PG-1:0][15:0]      i_x,     // weight / x / Q
    input  logic [NUM_PEG-1:0][NUM_PE_PG-1:0][15:0]      i_y,     // act / K

    // ---- per-PEG per-tensor pre-tree pow2 descale (overflow fix; 0 → bit-exact) ----
    input  logic [NUM_PEG-1:0][7:0]                      i_preshift,

    // ---- per-PEG fused APoT descale code (dealer; RESIDUAL = full − preshift) ----
    input  logic [NUM_PEG-1:0][11:0]                     i_dealer_apot,

    // ---- OACC accumulate control ----
    input  logic                                         i_oacc_en,   // advance
    input  logic                                         i_oacc_clr,  // first chunk/tap → load

    // ---- PPU-FMA constants + mode ----
    input  logic [1:0]                                   i_ppu_mode,  // PPU_CONV/PASS/SCALE
    input  logic [11:0]                                  i_bn_alpha_re,
    input  logic [11:0]                                  i_bn_alpha_im,
    input  logic [15:0]                                  i_bn_beta_re,
    input  logic [15:0]                                  i_bn_beta_im,
    input  logic [11:0]                                  i_prelu_re,
    input  logic [11:0]                                  i_prelu_im,
    input  logic signed [7:0]                            i_cast_k,
    input  logic [11:0]                                  i_scale_re,
    input  logic [11:0]                                  i_scale_im,

    // ---- complex output (CONV: e4m3 in [7:0]; PASS/SCALE: fp16) ----
    output logic [15:0]                                  o_re,
    output logic [15:0]                                  o_im,
    // pre-cast fp16 (CONV: post BN+PReLU, before e4m3) — same pipe stage as o_re; the skip-write
    // dual-cast (a 2nd consumer at a different block-float k) casts from this (CONSULT B1(a)).
    output logic [15:0]                                  o_pre_re,
    output logic [15:0]                                  o_pre_im
);
    // ---- 4 PEG → 4 pre-descale group partials ----
    logic [NUM_PEG-1:0][15:0] peg_accum;
    genvar g;
    generate
        for (g = 0; g < NUM_PEG; g++) begin : g_peg
            PEG u_peg (
                .i_mode     (i_pe_mode),
                .i_x        (i_x[g]),
                .i_y        (i_y[g]),
                .i_preshift ($signed(i_preshift[g])),
                .o_accum    (peg_accum[g])
            );
        end
    endgenerate

    // ---- PIPE STAGE 1: PEG output register (isolates the 32-way tree) ----
    // free-running; controller aligns i_oacc_clr/en to this +1-cycle latency.
    logic [NUM_PEG-1:0][15:0] peg_accum_r;
    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) peg_accum_r <= '0;
        else          peg_accum_r <= peg_accum;
    end

    // ---- 4 dealer (per-PEG APoT fused descale) ----
    logic [NUM_PEG-1:0][15:0] dealt;
    generate
        for (g = 0; g < NUM_PEG; g++) begin : g_dealer
            dealer u_dealer (
                .i_sum      (peg_accum_r[g]),
                .i_apot     (i_dealer_apot[g]),
                .o_descaled (dealt[g])
            );
        end
    endgenerate

    // ---- OACC (re/im 2-bank; P0..P3 = dealt[0..3]) ----
    logic [15:0] oacc_re, oacc_im;
    oacc u_oacc (
        .i_clk   (i_clk),
        .i_rst_n (i_rst_n),
        .i_en    (i_oacc_en),
        .i_clr   (i_oacc_clr),
        .i_p0    (dealt[0]),
        .i_p1    (dealt[1]),
        .i_p2    (dealt[2]),
        .i_p3    (dealt[3]),
        .o_re    (oacc_re),
        .o_im    (oacc_im)
    );

    // ---- PPU-FMA (mode-select post path) ----
    logic [15:0] ppu_re, ppu_im, ppu_pre_re, ppu_pre_im;
    ppu_fma u_ppu (
        .i_mode        (i_ppu_mode),
        .i_re          (oacc_re),
        .i_im          (oacc_im),
        .i_bn_alpha_re (i_bn_alpha_re),
        .i_bn_alpha_im (i_bn_alpha_im),
        .i_bn_beta_re  (i_bn_beta_re),
        .i_bn_beta_im  (i_bn_beta_im),
        .i_prelu_re    (i_prelu_re),
        .i_prelu_im    (i_prelu_im),
        .i_cast_k      (i_cast_k),
        .i_scale_re    (i_scale_re),
        .i_scale_im    (i_scale_im),
        .o_re          (ppu_re),
        .o_im          (ppu_im),
        .o_pre_re      (ppu_pre_re),
        .o_pre_im      (ppu_pre_im)
    );

    // ---- PIPE STAGE 3: PPU output register (isolates the CONV post-chain) ----
    // Latency operand→o_re = PEG-reg(1) + oacc(1) + ppu-reg(1) = 3 (+SRAM read 1).
    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin o_re <= 16'h0000; o_im <= 16'h0000; o_pre_re <= 16'h0000; o_pre_im <= 16'h0000; end
        else          begin o_re <= ppu_re;   o_im <= ppu_im;   o_pre_re <= ppu_pre_re; o_pre_im <= ppu_pre_im; end
    end
endmodule
