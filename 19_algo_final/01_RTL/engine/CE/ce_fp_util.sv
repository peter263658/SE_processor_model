/*
    ce_fp_util.sv — shared combinational FP helpers for the CE post-PE datapath
    (dealer / OACC / PPU-FMA). Zero-multiplier: pow2/APoT scaling is exponent-add,
    not a real multiply. FP16 = 1+5+10 bias 15. e4m3 = 1+4+3 bias 7 (OCP, max 448).

    Reuses fp16_add_sat (defined in PEG.sv) for all FP16 add/sub.
*/
`timescale 1ns/10ps

// ---------------------------------------------------------------------------
// fp16_scale2 — multiply an FP16 by 2^i_d (signed exponent delta). Pure
// exponent-add: mantissa unchanged. Underflow -> FTZ(±0); overflow -> ±max-normal.
// Input assumed finite (the saturating PE tree never emits Inf); subnormal in -> 0.
// ---------------------------------------------------------------------------
module fp16_scale2 (
    input  logic [15:0]      i_x,
    input  logic signed [7:0] i_d,
    output logic [15:0]      o_y
);
    wire        s = i_x[15];
    wire [4:0]  e = i_x[14:10];
    wire [9:0]  f = i_x[9:0];
    wire        is_zero = (e == 5'd0);          // FTZ subnormal/zero input
    logic signed [9:0] ne;
    always_comb begin
        ne = $signed({5'b0, e}) + i_d;
        if      (is_zero)    o_y = {s, 15'h0000};
        else if (ne <= 0)    o_y = {s, 15'h0000};        // underflow -> signed 0
        else if (ne >= 31)   o_y = {s, 15'h7BFF};        // overflow  -> ±max-normal
        else                 o_y = {s, ne[4:0], f};
    end
endmodule

// ---------------------------------------------------------------------------
// apot6a_decode — unpack a 12-bit APoT "6a" code (weight/BN scale) into the two
// signed pow2 exponents a,b (biased +24), has_second, and residual sign.
//   value applied = 2^a  [+/- 2^b]   (residual_sign: 1 => minus 2^b)
// ---------------------------------------------------------------------------
module apot6a_decode (
    input  logic [11:0]       i_code,
    output logic signed [7:0] o_a,
    output logic signed [7:0] o_b,
    output logic              o_has_second,
    output logic              o_res_sign
);
    assign o_a          = $signed({3'b0, i_code[11:7]}) - 8'sd24;
    assign o_b          = $signed({3'b0, i_code[6:2]})  - 8'sd24;
    assign o_has_second = i_code[1];
    assign o_res_sign   = i_code[0];
endmodule

// ---------------------------------------------------------------------------
// apot6b_decode — unpack a 12-bit PReLU "6b" pow2 code: single term ±2^k.
//   k biased +24 in [11:7]; bit[0] = overall slope sign (1 => negative).
// ---------------------------------------------------------------------------
module apot6b_decode (
    input  logic [11:0]       i_code,
    output logic signed [7:0] o_k,
    output logic              o_sign
);
    assign o_k    = $signed({3'b0, i_code[11:7]}) - 8'sd24;
    assign o_sign = i_code[0];
endmodule

// ---------------------------------------------------------------------------
// fp16_to_e4m3 — round-to-nearest-even FP16 -> FP8 e4m3 (OCP: bias 7, max 448,
// NaN only at 0x7F/0xFF). Overflow saturates to ±448 (0x7E/0xFE); underflow ->
// subnormal then ±0. NOTE [雷④]: rounding must be bit-exact vs DCNN fp8_qat —
// this is RNE; confirm against golden before trusting attention-adjacent casts.
// ---------------------------------------------------------------------------
module fp16_to_e4m3 (
    input  logic [15:0] i_h,
    output logic [7:0]  o_e4m3
);
    wire        s = i_h[15];
    wire [4:0]  e = i_h[14:10];
    wire [9:0]  f = i_h[9:0];

    always_comb begin
        logic signed [7:0] ue;          // unbiased fp16 exp
        logic signed [7:0] te;          // target e4m3 biased exp
        logic [10:0]       mant;        // 1.f (11 bits)
        logic [3:0]        m_round;     // {carry,3 mantissa bits} after RNE
        logic              g, r, st;
        logic [2:0]        m3;
        logic signed [7:0] sh;          // right shift for subnormal
        logic [10:0]       sm;
        logic              gg, ss;
        logic [3:0]        m4;          // subnormal mantissa + round (carry at [3])

        o_e4m3 = {s, 7'h00};

        if (e == 5'd0) begin
            o_e4m3 = {s, 7'h00};                       // fp16 zero/subnormal -> 0
        end else begin
            ue   = $signed({3'b0, e}) - 8'sd15;        // value = 1.f * 2^ue
            te   = ue + 8'sd7;                         // e4m3 biased exponent
            mant = {1'b1, f};

            if (te >= 8'sd16) begin
                // overflow: clamp to max-normal 448 (e=15,m=6)=0x7E  (BUG-1 fix: was >=15)
                o_e4m3 = {s, 7'h7E};
            end else if (te >= 8'sd1) begin
                // normal e4m3: keep f[9:7], RNE on f[6:0]
                g  = f[6];
                r  = f[5];
                st = |f[4:0];
                m_round = {1'b0, f[9:7]} + ((g & (r | st | f[7])) ? 4'd1 : 4'd0);
                if (m_round[3]) begin                  // mantissa carry -> exp+1
                    te = te + 8'sd1;
                    m3 = 3'd0;
                end else begin
                    m3 = m_round[2:0];
                end
                if      (te >= 8'sd16)                o_e4m3 = {s, 7'h7E};   // carried into overflow
                else if (te == 8'sd15 && m3 == 3'd7) o_e4m3 = {s, 7'h7E};   // (15,7)=NaN -> 448
                else                                 o_e4m3 = {s, te[3:0], m3};
            end else begin
                // subnormal e4m3 (te <= 0): m = round(mant >> (8-te))  (BUG-2 fix: was 1-te)
                sh = 8'sd8 - te;                       // te<=0 -> sh in [8, ...]
                if (sh >= 8'sd11) begin
                    o_e4m3 = {s, 7'h00};               // too small -> 0
                end else begin
                    sm = mant >> sh[3:0];
                    gg = mant[sh[3:0] - 4'd1];
                    ss = |(mant & ((11'd1 << (sh[3:0] - 4'd1)) - 11'd1));
                    m4 = {1'b0, sm[2:0]} + ((gg & (ss | sm[0])) ? 4'd1 : 4'd0);
                    if (m4[3]) o_e4m3 = {s, 4'd1, 3'd0};      // rounded up to smallest normal
                    else       o_e4m3 = {s, 4'd0, m4[2:0]};
                end
            end
        end
    end
endmodule
