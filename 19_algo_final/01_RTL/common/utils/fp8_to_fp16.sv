/*
    Module:     FP8 E4M3 -> IEEE FP16 converter (combinational)
    Project:    BCCTN W4A8 PEG, FP cast utility
    Note:
        - Bit-exact, 不需要 rounding（FP16 mantissa 10b 完全包住 FP8 mantissa 3b，
          且所有 FP8 normal/subnormal 的 unbiased exp 都落在 FP16 normal range 內）
        - FP8 normal:    2^(-6) .. 2^8 × 1.875 = 1/64 .. 448
        - FP8 subnormal: 2^(-9) .. 2^(-7) × 1.75，IEEE-strict 歸一化（不 FTZ）
        - FP8 NaN（唯一編碼 e=1111, m=111）-> sign-preserving FP16 qNaN {s, 15'h7E00}
        - FP8 沒有 Inf
*/
module fp8_to_fp16 (
    input  logic        [7  : 0]  i_fp8,
    output logic        [15 : 0]  o_fp16
);

    // ------------------------------------------------------------
    // 1. Unpack FP8 E4M3
    // ------------------------------------------------------------
    logic        s;
    logic [3:0]  e8;
    logic [2:0]  m8;
    logic        is_zero, is_nan, is_subn;

    assign s       = i_fp8[7];
    assign e8      = i_fp8[6:3];
    assign m8      = i_fp8[2:0];
    assign is_zero = (e8 == 4'h0) && (m8 == 3'h0);
    assign is_nan  = (e8 == 4'hF) && (m8 == 3'h7);
    assign is_subn = (e8 == 4'h0) && (m8 != 3'h0);

    // ------------------------------------------------------------
    // 2. Normal-input path
    //    biased_fp16 = (e8 - 7) + 15 = e8 + 8
    //    mantissa_fp16 = {m8, 7'b0}
    // ------------------------------------------------------------
    logic [4:0] e16_norm;
    logic [9:0] m16_norm;
    assign e16_norm = {1'b0, e8} + 5'd8;
    assign m16_norm = {m8, 7'b0};

    // ------------------------------------------------------------
    // 3. Subnormal-input path：對 m8 做 leading-1 detect 並歸一化
    //    m8 = 1xx -> leading-1 at bit 2, shift = 1, biased_fp16 = 8
    //    m8 = 01x -> leading-1 at bit 1, shift = 2, biased_fp16 = 7
    //    m8 = 001 -> leading-1 at bit 0, shift = 3, biased_fp16 = 6
    // ------------------------------------------------------------
    logic [1:0] lz_m;
    logic [4:0] e16_subn;
    logic [9:0] m16_subn;

    always_comb begin
        casez (m8)
            3'b1??:  lz_m = 2'd0;
            3'b01?:  lz_m = 2'd1;
            3'b001:  lz_m = 2'd2;
            default: lz_m = 2'd0;
        endcase
    end

    // biased_fp16 = -6 - (lz_m + 1) + 15 = 8 - lz_m
    assign e16_subn = 5'd8 - {3'b0, lz_m};

    // 把 implicit leading 1 移除（左移 lz_m+1）後，下方補 0 湊成 10-bit FP16 mantissa
    always_comb begin
        case (lz_m)
            2'd0:    m16_subn = {m8[1:0], 8'b0};   // m8 = 1xx, 移掉 bit2
            2'd1:    m16_subn = {m8[0],   9'b0};   // m8 = 01x, 移掉 bit1
            2'd2:    m16_subn = 10'b0;             // m8 = 001, 移掉 bit0, 無餘量
            default: m16_subn = 10'b0;
        endcase
    end

    // ------------------------------------------------------------
    // 4. Output mux
    // ------------------------------------------------------------
    always_comb begin
        if      (is_nan)  o_fp16 = {s, 5'h1F, 10'h200};       // sign-preserving qNaN
        else if (is_zero) o_fp16 = {s, 15'h0000};             // signed zero
        else if (is_subn) o_fp16 = {s, e16_subn, m16_subn};
        else              o_fp16 = {s, e16_norm, m16_norm};
    end

endmodule
