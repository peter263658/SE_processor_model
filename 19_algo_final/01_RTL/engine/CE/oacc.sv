/*
    oacc.sv — complex output accumulator, 2 banks (re/im).

    PEG-per-stream mapping (§2): the 4 dealer outputs of one 32-ic chunk are
        P0 = fr·xr   P1 = fi·xi   P2 = fr·xi   P3 = fi·xr
    Combine (fixed wiring) and accumulate across ic-chunks (and DW 5-taps):
        re += (P0 − P1)        im += (P2 + P3)
    i_clr starts a fresh output channel (load instead of add); i_en advances.
    All FP16 (saturating add). The cross-chunk / 5-tap accumulation lives here —
    no dedicated DW pre-accumulator (§5).
*/
`timescale 1ns/10ps
module oacc (
    input  logic        i_clk,
    input  logic        i_rst_n,
    input  logic        i_en,          // advance accumulation this cycle
    input  logic        i_clr,         // first chunk/tap of an output channel: load (not add)
    input  logic [15:0] i_p0,          // fr·xr  (descaled)
    input  logic [15:0] i_p1,          // fi·xi
    input  logic [15:0] i_p2,          // fr·xi
    input  logic [15:0] i_p3,          // fi·xr
    output logic [15:0] o_re,          // accumulated real  (valid after last chunk)
    output logic [15:0] o_im           // accumulated imag
);
    // per-chunk partials: re_partial = P0 − P1 , im_partial = P2 + P3
    wire   [15:0] p1_neg = {~i_p1[15], i_p1[14:0]};
    logic  [15:0] dre, dim;
    fp16_add_sat u_dre (.a(i_p0), .b(p1_neg), .z(dre));
    fp16_add_sat u_dim (.a(i_p2), .b(i_p3),   .z(dim));

    // accumulate into banks
    logic [15:0] re_r, im_r;
    logic [15:0] re_acc, im_acc;
    fp16_add_sat u_re (.a(re_r), .b(dre), .z(re_acc));
    fp16_add_sat u_im (.a(im_r), .b(dim), .z(im_acc));

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            re_r <= 16'h0000;
            im_r <= 16'h0000;
        end else if (i_en) begin
            re_r <= i_clr ? dre : re_acc;
            im_r <= i_clr ? dim : im_acc;
        end
    end

    assign o_re = re_r;
    assign o_im = im_r;
endmodule
