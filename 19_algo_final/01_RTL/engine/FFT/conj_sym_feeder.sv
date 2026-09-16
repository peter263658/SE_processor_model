/*
    conj_sym_feeder.sv — ISTFT front: masked 64-bin onesided spectrum -> 128-point
    conjugate-symmetric serial stream for FFT128 (used as IFFT).

    IFFT(X) = conj( FFT128( conj(X) ) ), and FFT128's built-in ÷128 IS the IFFT 1/N.
    For a real output signal the result is real, so the downstream takes only the real
    part (no output conj needed). This feeder therefore emits conj(X) of the full
    128-bin conj-symmetric spectrum, built from the 64 masked bins (1..64, DC dropped):

        Xfull[0]      = 0                         (DC was removed by the model; pad 0)
        Xfull[1..64]  = masked bins 1..64         (bank[0..63] = bin1..64)
        Xfull[65..127]= conj(Xfull[128-k])        (conjugate symmetry; bin64 = Nyquist, real)

    emit[k] = conj(Xfull[k]):
        k=0      -> 0
        1..64    -> conj(bin k)    = bank[k-1],   imag NEGATED
        65..127  -> bin(128-k)     = bank[127-k], imag as-is
    (verified bit-exact vs np.fft in gen/STFT handoff: real(FFT128(emit)) == x - DC.)

    Load the 64 masked bins (parallel) with i_load, then pulse i_start to stream 128
    samples (o_en high for 128 cycles) into FFT128.di_*.
*/
`timescale 1ns/10ps
module conj_sym_feeder
    import Param_def::*;
(
    input  logic                              i_clk,
    input  logic                              i_rst_n,
    input  logic                              i_load,      // latch the 64 masked bins
    input  logic [63:0][FX_FULL_WIDTH-1:0]    i_spec_re,   // masked bins 1..64 at [0..63]
    input  logic [63:0][FX_FULL_WIDTH-1:0]    i_spec_im,
    input  logic                              i_start,     // start the 128-sample stream
    output logic                              o_en,        // -> FFT128.di_en
    output logic [FX_FULL_WIDTH-1:0]          o_re,        // -> FFT128.di_re
    output logic [FX_FULL_WIDTH-1:0]          o_im         // -> FFT128.di_im
);
    logic [63:0][FX_FULL_WIDTH-1:0] bank_re, bank_im;
    logic [7:0] cnt;          // 0..127
    logic       running;

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            running <= 1'b0;  cnt <= 8'd0;
        end else if (i_load) begin
            bank_re <= i_spec_re;  bank_im <= i_spec_im;  running <= 1'b0;  cnt <= 8'd0;
        end else if (i_start) begin
            running <= 1'b1;  cnt <= 8'd0;
        end else if (running) begin
            if (cnt == 8'd127) begin running <= 1'b0;  cnt <= 8'd0; end
            else                     cnt <= cnt + 8'd1;
        end
    end

    // address + conj sign for sample index k = cnt
    logic [6:0] addr;
    logic       neg_im, is_dc;
    always_comb begin
        is_dc = (cnt == 8'd0);
        if (cnt <= 8'd64) begin addr = cnt[6:0] - 7'd1;   neg_im = 1'b1; end   // conj(bin k)
        else              begin addr = 7'd127 - cnt[6:0]; neg_im = 1'b0; end   // bin(128-k)
    end

    logic signed [FX_FULL_WIDTH-1:0] re_sel, im_sel;
    assign re_sel = bank_re[addr];
    assign im_sel = bank_im[addr];

    assign o_en = running;
    assign o_re = is_dc ? '0 : re_sel;
    assign o_im = is_dc ? '0 : (neg_im ? (-im_sel) : im_sel);   // two's-complement negate
endmodule
