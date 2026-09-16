/*
    stft_fp8_cast.sv — STFT (64-bin parallel, Q1.15) -> block-float e4m3 for UFBUF.

    Sits between Bit_Reorder (o_dat_re/o_dat_im, 64 bins 1..64, Q1.15) and UFBUF.
    Casts every bin (real & imag) to e4m3 = e4m3(model_value * 2^k) via fxp_to_e4m3,
    which also folds the HW(÷128) vs model(÷sqrt128) normalization (see fxp_to_e4m3).

    i_k = consuming boundary block-float exponent (encoder.model.0.0.depthwise = 9).
    Output registered, 1-cycle latency; o_dat_en mirrors the input enable delayed 1.
    128 cast cells (64 re + 64 im); frame-rate, so area is not throughput-critical.
*/
`timescale 1ns/10ps
module stft_fp8_cast
    import Param_def::*;
(
    input  logic                                   i_clk,
    input  logic                                   i_rst_n,
    input  logic                                   i_dat_en,
    input  logic signed [7:0]                      i_k,        // block-float exp (=9 encoder in)
    input  logic [63:0][FX_FULL_WIDTH-1:0]         i_dat_re,   // Q1.15 (Bit_Reorder out)
    input  logic [63:0][FX_FULL_WIDTH-1:0]         i_dat_im,
    output logic                                   o_dat_en,
    output logic [63:0][7:0]                        o_e4m3_re,  // block-float e4m3
    output logic [63:0][7:0]                        o_e4m3_im
);
    logic [63:0][7:0] cast_re, cast_im;

    genvar b;
    generate
        for (b = 0; b < 64; b++) begin : g_cast
            fxp_to_e4m3 u_re (.i_fxp($signed(i_dat_re[b])), .i_k(i_k), .o_e4m3(cast_re[b]));
            fxp_to_e4m3 u_im (.i_fxp($signed(i_dat_im[b])), .i_k(i_k), .o_e4m3(cast_im[b]));
        end
    endgenerate

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            o_dat_en  <= 1'b0;
            o_e4m3_re <= '0;
            o_e4m3_im <= '0;
        end else begin
            o_dat_en  <= i_dat_en;
            o_e4m3_re <= cast_re;
            o_e4m3_im <= cast_im;
        end
    end
endmodule
