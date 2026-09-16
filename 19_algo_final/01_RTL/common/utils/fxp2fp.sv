/************************************
* Filename: fxp2fp.sv
* Author: I-Ting Lin
* Description:
*   fixed-point to floating-point module
*   Currently only works for FXP20 to FP16 (Intermediate exponent issue)
* Review history: 2022/02/12
*************************************/
module fxp2fp #(    
    FP_WIDTH            = 16, 
    FP_EXP_WIDTH        = 5 ,   
    FP_MAN_WIDTH        = 10,
    FP_BIAS             = 5'd15,
    FXP_WIDTH           = 20,   
    FXP_INT             = 9,
    FXP_FRAC            = 10,
    FP17_EXP_BIAS       = 31,
    FP16_EXP_BIAS       = 15,
    CORRECTION_TERM     = -1 * FP17_EXP_BIAS - FXP_FRAC + FP16_EXP_BIAS
)(
    input logic [FXP_WIDTH - 1 : 0]    fxp_idata  ,
    output logic [FP_WIDTH - 1 : 0]    fp_odata     
);

    logic [7 : 0]                   i2flt_status_w;
    logic [FP_WIDTH : 0]            i2flt_odata_w;

    logic [FP_EXP_WIDTH + 1 : 0]    exp_correct_term;

    assign exp_correct_term = CORRECTION_TERM;

    DW_fp_i2flt #(
        .sig_width(FP_MAN_WIDTH),
        .exp_width(FP_EXP_WIDTH + 1),
        .isize(FXP_WIDTH),
        .isign(1)
    ) fxp2fp_i2flt(
        .a(fxp_idata),
        .rnd(3'b000),
        .z(i2flt_odata_w),
        .status(i2flt_status_w)
    );

    always_comb begin
        fp_odata[FP_WIDTH - 1] = i2flt_odata_w[FP_WIDTH];
        fp_odata[FP_MAN_WIDTH - 1 : 0] = i2flt_odata_w[FP_MAN_WIDTH - 1 : 0];
        fp_odata[FP_WIDTH - 2 -: FP_EXP_WIDTH] = $signed({1'b0, i2flt_odata_w[FP_WIDTH - 1 -: (FP_EXP_WIDTH + 1)]}) + $signed(exp_correct_term);
    end
 
endmodule