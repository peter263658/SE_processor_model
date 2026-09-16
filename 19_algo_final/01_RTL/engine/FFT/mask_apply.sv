/*
    mask_apply.sv — output-stage complex mask multiply (mode C), per (bin, frame).

    masked = spec * mask  (complex):  re = sr*mr - si*mi ,  im = sr*mi + si*mr
      spec (sr,si) : Q2.14 int16  (STFT bus; |spec|<2)
      mask (mr,mi) : fp16         (decoder fp16-PASS output; NO e4m3 cast)
      masked       : Q2.14 int16  -> iSTFT input (caller pads DC bin0=0; bins 0..63 -> masked 1..64)

    Bit-exact to mask_apply_golden: the products + the one sub/add are evaluated in fp32 and a
    SINGLE round-half-to-even to Q2.14 (DW_fp_flt2i rnd=0) + int16 saturate. VERIFIED equivalent to
    the fp64 reference on both sides (L/R 25536/25536). KEY: the Q2.14 step (2^-14) is NOT applied to
    the operands — fp32 rounding is significand-relative (scale-invariant), so int16 spec codes feed
    the mults directly and flt2i RNE on the raw fp32 product == round(spec_code*mask - ...). No 2^±14
    scaling unit. Pure combinational (loose timing; caller may register I/O).

    House style: arithmetic = instantiated DesignWare/util modules. 4 fp32 mult + 2 fp32 addsub +
    2 int16->fp32 (i2flt) + 2 fp16->fp32 + 2 fp32->int (flt2i) + int16 saturate.
*/
`timescale 1ns/10ps
module mask_apply (
    input  logic [15:0] i_spec_re, i_spec_im,    // Q2.14 int16
    input  logic [15:0] i_mask_re, i_mask_im,    // fp16
    output logic [15:0] o_masked_re, o_masked_im // Q2.14 int16
);
    localparam logic [2:0] RNE = 3'b000;          // DW round-to-nearest-even

    // ---- operands -> fp32 ----
    wire [31:0] sr_f, si_f, mr_f, mi_f;
    DW_fp_i2flt_inst #(.sig_width(23), .exp_width(8), .isize(16), .isign(1))
        u_sr (.inst_a(i_spec_re), .inst_rnd(RNE), .z_inst(sr_f), .status_inst());
    DW_fp_i2flt_inst #(.sig_width(23), .exp_width(8), .isize(16), .isign(1))
        u_si (.inst_a(i_spec_im), .inst_rnd(RNE), .z_inst(si_f), .status_inst());
    fp16_to_fp32 u_mr (.i_h(i_mask_re), .o_f(mr_f));
    fp16_to_fp32 u_mi (.i_h(i_mask_im), .o_f(mi_f));

    // ---- 4 fp32 products: P0=sr*mr, P1=si*mi, P2=sr*mi, P3=si*mr ----
    wire [31:0] p0, p1, p2, p3;
    DW_fp_mult_inst #(.sig_width(23), .exp_width(8), .ieee_compliance(1))
        u_p0 (.inst_a(sr_f), .inst_b(mr_f), .inst_rnd(RNE), .z_inst(p0), .status_inst());
    DW_fp_mult_inst #(.sig_width(23), .exp_width(8), .ieee_compliance(1))
        u_p1 (.inst_a(si_f), .inst_b(mi_f), .inst_rnd(RNE), .z_inst(p1), .status_inst());
    DW_fp_mult_inst #(.sig_width(23), .exp_width(8), .ieee_compliance(1))
        u_p2 (.inst_a(sr_f), .inst_b(mi_f), .inst_rnd(RNE), .z_inst(p2), .status_inst());
    DW_fp_mult_inst #(.sig_width(23), .exp_width(8), .ieee_compliance(1))
        u_p3 (.inst_a(si_f), .inst_b(mr_f), .inst_rnd(RNE), .z_inst(p3), .status_inst());

    // ---- re = P0 - P1 (op=1), im = P2 + P3 (op=0) ----
    wire [31:0] re_f, im_f;
    DW_fp_addsub_inst #(.sig_width(23), .exp_width(8), .ieee_compliance(1))
        u_re (.inst_a(p0), .inst_b(p1), .inst_rnd(RNE), .inst_op(1'b1), .z_inst(re_f), .status_inst());
    DW_fp_addsub_inst #(.sig_width(23), .exp_width(8), .ieee_compliance(1))
        u_im (.inst_a(p2), .inst_b(p3), .inst_rnd(RNE), .inst_op(1'b0), .z_inst(im_f), .status_inst());

    // ---- fp32 -> int (single RNE) ; saturate to int16 Q2.14 code ----
    wire [31:0] re_i, im_i;
    DW_fp_flt2i_inst #(.inst_sig_width(23), .inst_exp_width(8), .inst_isize(32), .inst_ieee_compliance(1))
        u_re2i (.inst_a(re_f), .inst_rnd(RNE), .z_inst(re_i), .status_inst());
    DW_fp_flt2i_inst #(.inst_sig_width(23), .inst_exp_width(8), .inst_isize(32), .inst_ieee_compliance(1))
        u_im2i (.inst_a(im_f), .inst_rnd(RNE), .z_inst(im_i), .status_inst());

    wire signed [31:0] re_s = re_i;
    wire signed [31:0] im_s = im_i;
    assign o_masked_re = (re_s > 32'sd32767)  ? 16'h7FFF
                       : (re_s < -32'sd32768) ? 16'h8000 : re_s[15:0];
    assign o_masked_im = (im_s > 32'sd32767)  ? 16'h7FFF
                       : (im_s < -32'sd32768) ? 16'h8000 : im_s[15:0];
endmodule
