/*
    Module:     IEEE FP16 -> FP8 E4M3 converter (combinational)
    Project:    BCCTN W4A8 PEG, FP cast utility
    Note:
        - Round-to-Nearest-Even (RNE), IEEE-strict
        - FP16 NaN  -> FP8 NaN preserving sign:           {s, 4'hF, 3'h7}
        - FP16 ±Inf -> FP8 saturate to ±max-finite (±448): {s, 4'hF, 3'h6}
        - FP16 ±0   -> FP8 ±0
        - FP16 subnormal (e=0, m≠0) -> FP8 ±0
            理由：max FP16 subnormal ≈ 2^(-14)，遠小於 2^(-10)
            （= 2^(-9)/2，FP8 最小 subnormal 的 RNE tie point），所以一律 round to 0
        - FP16 normal -> FP8 normal/subnormal/zero，含 RNE 與 saturation
            saturation 也涵蓋 (e=15, m=7) 這個 NaN encoding，避免 finite 輸入產生 NaN 輸出
*/
module fp16_to_fp8 (
    input  logic        [15 : 0]  i_fp16,
    output logic        [7  : 0]  o_fp8
);

    // ------------------------------------------------------------
    // 1. Unpack FP16
    // ------------------------------------------------------------
    logic        s;
    logic [4:0]  e16;
    logic [9:0]  m16;
    logic        is_zero, is_inf, is_nan, is_subn;

    assign s       = i_fp16[15];
    assign e16     = i_fp16[14:10];
    assign m16     = i_fp16[9:0];
    assign is_zero = (e16 == 5'h00) && (m16 == 10'h000);
    assign is_inf  = (e16 == 5'h1F) && (m16 == 10'h000);
    assign is_nan  = (e16 == 5'h1F) && (m16 != 10'h000);
    assign is_subn = (e16 == 5'h00) && (m16 != 10'h000);

    // ------------------------------------------------------------
    // 2. Normal-form 11-bit significand 與目標 FP8 biased exp
    //    e8_biased_pre = (e16 - 15) + 7 = e16 - 8
    // ------------------------------------------------------------
    logic [10:0]       sig11;
    logic signed [6:0] exp_un;
    logic signed [6:0] e8_biased_pre;

    assign sig11         = {1'b1, m16};
    assign exp_un        = $signed({2'b00, e16}) - 7'sd15;
    assign e8_biased_pre = exp_un + 7'sd7;

    // ------------------------------------------------------------
    // 3. 為 subnormal/underflow 對齊所需的右移量
    //    e8_biased_pre >= 1 (FP8 normal target)   : rshift = 0
    //    e8_biased_pre <= 0 (FP8 subnormal target): rshift = 1 - e8_biased_pre
    //    對 FP16 normal 輸入 (e16 ∈ [1,30])，e8_biased_pre ∈ [-7, 22]，
    //    所以實際 rshift ≤ 8，4-bit 足夠。
    // ------------------------------------------------------------
    logic signed [6:0] rshift_s;
    logic [3:0]        rshift;

    assign rshift_s = (e8_biased_pre >= 7'sd1) ? 7'sd0 : (7'sd1 - e8_biased_pre);
    assign rshift   = rshift_s[3:0];

    // ------------------------------------------------------------
    // 4. 用加寬版 significand 做右移以保留正確 sticky
    //    sig_wide = {sig11, 11'b0} → rshift 上限 11，位元不會掉到 sticky 區之外
    // ------------------------------------------------------------
    logic [21:0] sig_wide, sig_shifted;
    assign sig_wide    = {sig11, 11'b0};
    assign sig_shifted = sig_wide >> rshift;

    // ------------------------------------------------------------
    // 5. 抽出 4-bit 候選 mantissa + L/G/S，套 RNE
    //    mant4 = bits[21:18]  (頂位元為 normal target 的 implicit 1，或 subnormal target
    //                          shift 後的 0/1)
    //    L     = bit[18]      (kept LSB，用於 tie-to-even)
    //    G     = bit[17]      (round bit)
    //    S     = |bits[16:0]  (sticky)
    // ------------------------------------------------------------
    logic [3:0] mant4;
    logic       g_bit, s_bit, l_bit, round_up;
    logic [4:0] mant4_rounded;

    assign mant4         = sig_shifted[21:18];
    assign l_bit         = sig_shifted[18];
    assign g_bit         = sig_shifted[17];
    assign s_bit         = |sig_shifted[16:0];
    assign round_up      = g_bit & (l_bit | s_bit);
    assign mant4_rounded = {1'b0, mant4} + {4'b0, round_up};

    // ------------------------------------------------------------
    // 6. 組 final FP8 欄位，含 overflow / NaN-encoding saturation
    // ------------------------------------------------------------
    logic [3:0] e8_out;
    logic [2:0] m8_out;

    always_comb begin
        if (e8_biased_pre >= 7'sd16) begin
            // 還沒 round 就確定 overflow：飽和到 ±448
            e8_out = 4'hF;
            m8_out = 3'h6;
        end
        else if (e8_biased_pre >= 7'sd1) begin
            // FP8 normal target
            if (mant4_rounded == 5'b10000) begin
                // mantissa rounding 進位變成 1.000 × 2^(exp+1)
                if ((e8_biased_pre + 7'sd1) >= 7'sd16) begin
                    e8_out = 4'hF;
                    m8_out = 3'h6;
                end else begin
                    e8_out = e8_biased_pre[3:0] + 4'd1;
                    m8_out = 3'b000;
                end
            end
            else if ((e8_biased_pre == 7'sd15) && (mant4_rounded == 5'b01111)) begin
                // (e=15, m=7) 是 FP8 NaN encoding；finite 輸入要飽和而非變 NaN
                e8_out = 4'hF;
                m8_out = 3'h6;
            end
            else begin
                e8_out = e8_biased_pre[3:0];
                m8_out = mant4_rounded[2:0];
            end
        end
        else begin
            // FP8 subnormal target (e8_biased_pre <= 0)
            if (mant4_rounded[3] == 1'b1) begin
                // RNE 跨過 subnormal/normal 邊界 → 變成 FP8 最小 normal
                e8_out = 4'd1;
                m8_out = 3'b000;
            end else begin
                e8_out = 4'd0;
                m8_out = mant4_rounded[2:0];
            end
        end
    end

    // ------------------------------------------------------------
    // 7. Output mux
    // ------------------------------------------------------------
    always_comb begin
        if      (is_nan)  o_fp8 = {s, 4'hF, 3'h7};       // sign-preserving NaN
        else if (is_inf)  o_fp8 = {s, 4'hF, 3'h6};       // ±Inf -> ±448
        else if (is_zero) o_fp8 = {s, 7'h00};
        else if (is_subn) o_fp8 = {s, 7'h00};            // FP16 subnormal -> ±0
        else              o_fp8 = {s, e8_out, m8_out};
    end

endmodule
