/*
    Module:     INT4 (W) x FP8 E4M3 (A) -> FP16 (IEEE) lane multiplier
    Project:    BCCTN W4A8 PEG (Mapping B), CE / AE 共用之 lane building block
    Note:
        - Combinational. 32 個本 lane + 32-way FP16 adder tree = 1 group_unit。
        - 結果 bit-exact：mag_prod 7-bit 完全裝進 FP16 mantissa(10-bit)，
          且所有 valid 輸入組合的 FP16 unbiased exp 都落在 [-9, +11]，
          biased [6, 26] 全在 normal range，不會 over/underflow。
        - FP8 subnormal -> FP16 normal（IEEE-strict，不 FTZ）
        - FP8 NaN (E4M3 only encoding: e=1111, m=111) -> sign-preserving FP16 qNaN {sign_prod, 15'h7E00}
        - INT4 software guarantees [-7, +7]；4'b1000 (-8) 是 don't-care
*/

module mult_int4_fp8 (
    input  logic        [3 : 0]   i_w_int4,    // 2's complement
    input  logic        [7 : 0]   i_a_fp8,     // FP8 E4M3 (1+4+3, bias=7)
    output logic        [15 : 0]  o_prod_fp16  // IEEE FP16 (1+5+10, bias=15)
);

    // ------------------------------------------------------------
    // 1. Unpack FP8 E4M3
    // ------------------------------------------------------------
    logic        s_a;
    logic [3:0]  e_a;
    logic [2:0]  m_a;
    logic        is_zero_a, is_nan_a, is_subn_a, is_norm_a;

    assign s_a       = i_a_fp8[7];
    assign e_a       = i_a_fp8[6:3];
    assign m_a       = i_a_fp8[2:0];
    assign is_zero_a = (e_a == 4'h0) && (m_a == 3'h0);
    assign is_nan_a  = (e_a == 4'hF) && (m_a == 3'h7);
    assign is_subn_a = (e_a == 4'h0) && (m_a != 3'h0);
    assign is_norm_a = (e_a != 4'h0) && !is_nan_a;

    // ------------------------------------------------------------
    // 2. INT4 (2's complement) -> sign-magnitude
    // ------------------------------------------------------------
    logic        s_w;
    logic [2:0]  mag_w;
    logic        is_zero_w;

    assign s_w       = i_w_int4[3];
    assign mag_w     = s_w ? (~i_w_int4[2:0] + 3'b001) : i_w_int4[2:0];
    assign is_zero_w = (mag_w == 3'h0);

    // ------------------------------------------------------------
    // 3. Magnitude product (3b x 4b unsigned -> 7b)
    //    significand = 1.mmm (normal) or 0.mmm (subnormal)
    // ------------------------------------------------------------
    logic [3:0]  significand;
    logic [6:0]  mag_prod;
    logic        sign_prod;

    assign significand = is_norm_a ? {1'b1, m_a} : {1'b0, m_a};
    assign mag_prod    = mag_w * significand;
    assign sign_prod   = s_w ^ s_a;

    // ------------------------------------------------------------
    // 4. Effective biased exponent
    //    subnormal: e_a=0 在算術上等效 e=1，所以 e_eff = e_a | (e_a==0)
    // ------------------------------------------------------------
    logic [3:0] e_eff;
    assign e_eff = is_subn_a ? 4'h1 : e_a;

    // ------------------------------------------------------------
    // 5. Leading-1 detect on mag_prod[6:0]
    // ------------------------------------------------------------
    logic [2:0] lz;
    always_comb begin
        casez (mag_prod)
            7'b1??????: lz = 3'd6;
            7'b01?????: lz = 3'd5;
            7'b001????: lz = 3'd4;
            7'b0001???: lz = 3'd3;
            7'b00001??: lz = 3'd2;
            7'b000001?: lz = 3'd1;
            7'b0000001: lz = 3'd0;
            default   : lz = 3'd0; // all-zero, 由 is_zero_out 路徑覆蓋
        endcase
    end

    // ------------------------------------------------------------
    // 6. FP16 packing
    //    exp_field = e_eff + lz + 5   (推導: e_eff - 10 + lz + 15)
    //    frac      = (mag_prod << (10 - lz))[9:0]   (drop implicit 1 at bit 10)
    // ------------------------------------------------------------
    logic [4:0]  exp_field;
    logic [9:0]  frac_field;
    logic [16:0] frac_shifted;
    logic [3:0]  shift_amt;
    logic        is_zero_out;

    assign exp_field    = {1'b0, e_eff} + {2'b0, lz} + 5'd5;
    assign shift_amt    = 4'd10 - {1'b0, lz};
    assign frac_shifted = {10'b0, mag_prod} << shift_amt;
    assign frac_field   = frac_shifted[9:0];
    assign is_zero_out  = is_zero_w | is_zero_a;

    always_comb begin
        if (is_nan_a)         o_prod_fp16 = {sign_prod, 15'h7E00};      // sign-preserving qNaN
        else if (is_zero_out) o_prod_fp16 = {sign_prod, 15'h0000};      // signed zero
        else                  o_prod_fp16 = {sign_prod, exp_field, frac_field};
    end

endmodule
