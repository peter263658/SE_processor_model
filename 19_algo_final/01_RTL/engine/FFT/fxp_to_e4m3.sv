/*
    fxp_to_e4m3.sv — STFT fixed-point (Q1.15) -> block-float e4m3 activation.
    For the STFT -> UFBUF boundary (encoder input). Combinational; frame-rate.

    Reconciles TWO things in one cast:

    (a) HW vs model STFT scale gap.
        HW FFT128 (radix-2^2 SDF) divides by N=128 (each of the 7 butterfly stages
        does >>1).  The model torch.stft(normalized=True) divides by sqrt(N).
        => model_value = HW_value * sqrt(N).   sqrt(128) = 2^3.5 = 2^3 * sqrt(2).
        The 2^3 folds into the pow2 shift (b); the sqrt(2) is the only real multiply.

    (b) Consuming boundary's block-float exponent k.
        Stored activation code = e4m3(model_value * 2^k).  For encoder.model.0.0
        .depthwise the calibrated k = 9 (drive i_k=9). i_k is a PORT so the same
        cell serves other boundaries.

    Q1.15 code c -> v_hw = c / 2^15.  Target:
        e4m3( v_hw * sqrt(N) * 2^k ) = e4m3( c * 2^(k-15) * 2^3.5 )
                                     = e4m3( c * sqrt(2) * 2^(k-12) )

    Chain (one fp16 mult is fine here — this is NOT in the MAC loop, runs once/frame):
        int16 c --i2flt--> fp16(c) --*sqrt2--> fp16 --scale2(k-12)--> fp16 --cast--> e4m3

    NOTE: sqrt(N)=sqrt(128) is BAKED (the "12" = 15 - floor(log2 sqrt 128) = 15-3, and
    the sqrt(2) constant). For a different n_fft, change SQRT2_FP16 / the shift.
*/
`timescale 1ns/10ps
module fxp_to_e4m3
    import Param_def::*;
(
    input  logic signed [FX_FULL_WIDTH-1:0] i_fxp,   // Q1.15 HW STFT sample (re or im)
    input  logic signed [7:0]               i_k,     // consuming block-float exp (encoder = 9)
    output logic        [7:0]               o_e4m3
);
    localparam logic [15:0] SQRT2_FP16 = 16'h3DA8;   // sqrt(2) in fp16 (~1.41406)

    // 1. int16 -> fp16 (RNE).  |c| <= ~1454 for the STFT amax -> exact (< 2^11).
    logic [15:0] f_int;
    DW_fp_i2flt_inst #(.sig_width(10), .exp_width(5), .isize(16), .isign(1)) u_i2f (
        .inst_a      (i_fxp),
        .inst_rnd    (3'b000),
        .z_inst      (f_int),
        .status_inst ()
    );

    // 2. x sqrt(2)  — the 0.5-octave residual of the sqrt(N) normalization.
    logic [15:0] f_rt2;
    DW_fp_mult_inst #(.sig_width(10), .exp_width(5), .ieee_compliance(1)) u_mul (
        .inst_a      (f_int),
        .inst_b      (SQRT2_FP16),
        .inst_rnd    (3'b000),
        .z_inst      (f_rt2),
        .status_inst ()
    );

    // 3. x 2^(k-12) — pure exponent add: -15 (Q1.15 frac) + 3 (floor log2 sqrt128) + k.
    logic signed [7:0] kshift;
    logic [15:0]       f_scaled;
    assign kshift = i_k - 8'sd12;
    fp16_scale2 u_sc (.i_x(f_rt2), .i_d(kshift), .o_y(f_scaled));

    // 4. fp16 -> e4m3 (RNE, saturate to +-448).
    fp16_to_fp8 u_cast (.i_fp16(f_scaled), .o_fp8(o_e4m3));
endmodule
