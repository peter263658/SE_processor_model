/*
    istft_reorder.sv — ISTFT back: FFT128 (used as IFFT) serial output -> natural-order
    real time samples (per frame, serial).

    FFT128 emits 128 samples in BIT-REVERSED order; for a real output we take the REAL
    part only (imag ~0, see conj_sym_feeder). Collect into a 128-deep buffer at the
    bit-reversed address, then replay 0..127 in natural order.

    NO synthesis window here: the synthesis Hamming, hop=80 overlap-add, and the istft
    normalization (÷ Σ window²) are all done OFF-CHIP / in the TB (they live in the same
    OLA/normalize domain — see RTL_HANDOFF_FFT_STFT_INTEGRATION §8). Keeping the window
    off-chip also avoids a Q-format mismatch (the analysis Hamming expects Q2.14, the IFFT
    output is Q1.15) and keeps the ISTFT add-on to exactly two small modules
    (conj_sym_feeder + istft_reorder) on top of the shared FFT128. Single-frame.
*/
`timescale 1ns/10ps
module istft_reorder
    import Param_def::*;
(
    input  logic                              i_clk,
    input  logic                              i_rst_n,
    input  logic                              i_dat_en,   // FFT128 do_en (bit-reversed time)
    input  logic signed [FX_FULL_WIDTH-1:0]   i_dat_re,   // take REAL part (imag discarded)
    output logic                              o_en,       // natural-order time sample valid
    output logic signed [FX_FULL_WIDTH-1:0]   o_sample    // Q1.15 (pre-window, pre-OLA)
);
    logic signed [FX_FULL_WIDTH-1:0] tbuf [0:127];
    logic [7:0] in_cnt, rd_cnt;
    logic       replaying;

    wire [6:0] in_brev = {in_cnt[0],in_cnt[1],in_cnt[2],in_cnt[3],in_cnt[4],in_cnt[5],in_cnt[6]};

    // collect: write FFT128 serial output at the bit-reversed address -> natural order
    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n)        in_cnt <= 8'd0;
        else if (i_dat_en) begin
            tbuf[in_brev] <= i_dat_re;
            in_cnt <= (in_cnt == 8'd127) ? 8'd0 : in_cnt + 8'd1;
        end
    end

    // replay: after the 128th sample collected, stream tbuf[0..127] in order
    wire start_replay = i_dat_en && (in_cnt == 8'd127);
    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n)            begin replaying <= 1'b0; rd_cnt <= 8'd0; end
        else if (start_replay)   begin replaying <= 1'b1; rd_cnt <= 8'd0; end
        else if (replaying) begin
            if (rd_cnt == 8'd127) begin replaying <= 1'b0; rd_cnt <= 8'd0; end
            else                       rd_cnt <= rd_cnt + 8'd1;
        end
    end

    // registered output (1-cycle): natural-order real time sample
    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin o_en <= 1'b0; o_sample <= '0; end
        else begin
            o_en     <= replaying;
            o_sample <= tbuf[rd_cnt[6:0]];
        end
    end
endmodule
