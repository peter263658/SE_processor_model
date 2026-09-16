/************************************
* Filename: fp2fxp.sv
* Author: I-Ting Lin
* Description:
*   floating-point to fixed point module
* Review history: 2022/02/12
*************************************/
module fp2fxp #(    
    FP_WIDTH     = 16, 
    FP_EXP_WIDTH = 5 ,   
    FP_MAN_WIDTH = 10,
    FP_BIAS      = 5'd15,   
    FXP_WIDTH    = 20,   
    FXP_INT      = 9,
    FXP_FRAC     = 10
) (
    input  logic [FP_WIDTH - 1 : 0]   fp_idata,
    output logic [FXP_WIDTH - 1 : 0]  fxp_odata     
);

    logic [FP_EXP_WIDTH - 1 : 0]   fp_exp_w;
    logic [FP_MAN_WIDTH - 1 : 0]   fp_man_w;
    logic [FP_EXP_WIDTH : 0]       fp_unbiased_exp_w;

    assign fp_unbiased_exp_w = $signed({1'b0, fp_exp_w}) - $signed({1'b0, FP_BIAS});
    always_comb begin
        fp_man_w = fp_idata[FP_MAN_WIDTH - 1 : 0];
        fp_exp_w = fp_idata[FP_MAN_WIDTH +: FP_EXP_WIDTH];
        fxp_odata = 0;
        fxp_odata[FXP_FRAC] = 1'b1;
        //if(FP_MAN_WIDTH >= FXP_FRAC) fxp_odata[FXP_FRAC - 1 : 0] = fp_man_w[FP_MAN_WIDTH - 1 -: FXP_FRAC];
        //else                      fxp_odata[FXP_FRAC - 1 : 0] = {fp_man_w[FP_MAN_WIDTH - 1 : 0], {FXP_FRAC - FP_MAN_WIDTH{1'b0}}};
        fxp_odata[FXP_FRAC - 1 : 0] = {fp_man_w[FP_MAN_WIDTH - 1 : 0], {FXP_FRAC - FP_MAN_WIDTH{1'b0}}};
        if(fp_unbiased_exp_w[FP_EXP_WIDTH]) begin
            // Right shift
            fxp_odata = fxp_odata >> (~fp_unbiased_exp_w[FP_EXP_WIDTH - 1 : 0] + 1'b1);
        end
        else begin
            // Left shift
            fxp_odata = fxp_odata << fp_unbiased_exp_w[FP_EXP_WIDTH - 1 : 0];
        end
        if(fp_idata[FP_WIDTH - 1]) begin
            fxp_odata = ~fxp_odata + 1'b1;
        end
    end
    
endmodule