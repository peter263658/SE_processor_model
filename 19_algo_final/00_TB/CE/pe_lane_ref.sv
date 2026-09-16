/*
    pe_lane_ref.sv — TB-ONLY golden reference (moved out of PEG.sv 2026-06-07).

    `pe_lane` is the validated single-lane v1 product unit; the DUT (pe_pair, fused
    2×8b slices) is checked bit-exact against PEG_ref(pe_lane×32) in tb_PEG_w4a8.sv.
    It is NOT used by the synthesizable datapath (PEG/CE_core use pe_pair). The helper
    modules pe_fp_unpack / pe_int4_csd stay in PEG.sv (shared with pe_pair); only
    pe_lane + its private pe_slice_addsub live here.
*/
`timescale 1ns/10ps

module pe_lane
    import Param_def::*;
    import Mem_def::*;
(
    input  CE_LANE_mode  i_mode,
    input  logic [15:0]  i_op_x,        // weight / x operand (field map per mode, see header)
    input  logic [15:0]  i_op_y,        // activation / y operand
    output logic [15:0]  o_prod_fp16    // IEEE FP16 (1+5+10, bias 15)
);

    wire is_int4   = (i_mode == MODE_W4A8) || (i_mode == MODE_W4A16);
    wire is_lmul   = ~is_int4;
    wire y_is_fp16 = (i_mode == MODE_LMUL_ATTN) || (i_mode == MODE_W4A16);

    // ============================================================
    // 1. Float unpack (frexp convention: value = o_sig * 2^o_exp,
    //    o_sig integer width FRAC_W+1, leading 1 at bit FRAC_W for nonzero).
    //    Y unpacked as both fp8 and fp16; X (L-Mul only) likewise.
    // ============================================================
    logic        y8_s,  y8_z,  y8_sp;  logic signed [7:0] y8_e;  logic [3:0]  y8_sig;
    logic        y16_s, y16_z, y16_sp; logic signed [7:0] y16_e; logic [10:0] y16_sig;
    logic        x8_s,  x8_z,  x8_sp;  logic signed [7:0] x8_e;  logic [3:0]  x8_sig;
    logic        x16_s, x16_z, x16_sp; logic signed [7:0] x16_e; logic [10:0] x16_sig;

    pe_fp_unpack #(.EXP_W(4), .FRAC_W(3),  .HAS_INF(0)) u_y8
        (.i_fp(i_op_y[7:0]),  .o_sign(y8_s),  .o_exp(y8_e),  .o_sig(y8_sig),  .o_zero(y8_z),  .o_special(y8_sp));
    pe_fp_unpack #(.EXP_W(5), .FRAC_W(10), .HAS_INF(1)) u_y16
        (.i_fp(i_op_y[15:0]), .o_sign(y16_s), .o_exp(y16_e), .o_sig(y16_sig), .o_zero(y16_z), .o_special(y16_sp));
    pe_fp_unpack #(.EXP_W(4), .FRAC_W(3),  .HAS_INF(0)) u_x8
        (.i_fp(i_op_x[7:0]),  .o_sign(x8_s),  .o_exp(x8_e),  .o_sig(x8_sig),  .o_zero(x8_z),  .o_special(x8_sp));
    pe_fp_unpack #(.EXP_W(5), .FRAC_W(10), .HAS_INF(1)) u_x16
        (.i_fp(i_op_x[15:0]), .o_sign(x16_s), .o_exp(x16_e), .o_sig(x16_sig), .o_zero(x16_z), .o_special(x16_sp));

    // Selected activation/Y significand (11-bit, right-aligned) + exponent + flags
    logic [10:0]      sigY;
    logic signed [7:0] expY;
    logic             signY, zeroY, specialY;
    always_comb begin
        if (y_is_fp16) begin sigY = y16_sig;            expY = y16_e; signY = y16_s; zeroY = y16_z; specialY = y16_sp; end
        else           begin sigY = {7'd0, y8_sig};     expY = y8_e;  signY = y8_s;  zeroY = y8_z;  specialY = y8_sp;  end
    end

    // ============================================================
    // 2. INT4 weight decode -> sign + CSD (two shifters + one add/sub)
    // ============================================================
    logic       w_sign, w_zero, w_useB, w_sub;
    logic [1:0] shA, shB;
    pe_int4_csd u_csd (.i_w(i_op_x[3:0]), .o_sign(w_sign), .o_zero(w_zero),
                       .o_shA(shA), .o_shB(shB), .o_useB(w_useB), .o_sub(w_sub));

    // INT4 product operands: (sigY << shA) +/- (sigY << shB)
    logic [15:0] int4_A, int4_B;
    assign int4_A = {5'd0, sigY} << shA;
    assign int4_B = w_useB ? ({5'd0, sigY} << shB) : 16'd0;

    // ============================================================
    // 3. L-Mul mantissa codes (top-m fraction bits) + significand-scale exponents
    //    ex_lmul = o_exp + FRAC_W  (exponent at significand in [1,2))
    // ============================================================
    logic [7:0]       fx_code, fy_code;
    logic signed [7:0] exX, exY_l;
    logic             sXl, sYl, zXl, zYl, spXl, spYl;
    always_comb begin
        if (i_mode == MODE_LMUL_ATTN) begin       // fp16, m=8 : fx = frac10 >> 2
            fx_code = x16_sig[9:2];               fy_code = y16_sig[9:2];
            exX     = x16_e + 8'sd10;             exY_l   = y16_e + 8'sd10;
            sXl=x16_s; sYl=y16_s; zXl=x16_z; zYl=y16_z; spXl=x16_sp; spYl=y16_sp;
        end else begin                            // fp8,  m=4 : fx = m_a << 1
            fx_code = {4'd0, x8_sig[2:0], 1'b0};  fy_code = {4'd0, y8_sig[2:0], 1'b0};
            exX     = x8_e + 8'sd3;               exY_l   = y8_e + 8'sd3;
            sXl=x8_s;  sYl=y8_s;  zXl=x8_z;  zYl=y8_z;  spXl=x8_sp;  spYl=y8_sp;
        end
    end

    // ============================================================
    // 4. SHARED significand add/sub (two 8-bit slices). §4 resource.
    //    INT4 : (sigY<<shA) +/- (sigY<<shB).   L-Mul : fx_code + fy_code.
    //    i_split kept for future dual-issue; single product uses chained (0).
    // ============================================================
    logic [15:0] add_a, add_b, add_sum;
    logic        add_sub;
    always_comb begin
        if (is_int4) begin add_a = int4_A;          add_b = int4_B;          add_sub = w_sub; end
        else         begin add_a = {8'd0, fx_code}; add_b = {8'd0, fy_code}; add_sub = 1'b0;  end
    end
    pe_slice_addsub u_add (.i_a(add_a), .i_b(add_b), .i_sub(add_sub), .i_split(1'b0), .o_sum(add_sum));

    // ============================================================
    // 5a. INT4 pack: value = sign * add_sum * 2^expY ; LZD-normalise to FP16
    //     with round-to-nearest-even. W4A8 product fits 10 frac bits (exact, no
    //     rounding); W4A16 product can exceed 10 frac bits -> RNE on dropped bits.
    // ============================================================
    logic [4:0] L;                 // leading-one bit index of add_sum
    always_comb begin
        L = 5'd0;
        for (int i = 0; i < 16; i++) if (add_sum[i]) L = i[4:0];
    end

    logic [9:0] int4_frac;
    logic       int4_rnd_carry;    // mantissa rollover from rounding -> exp + 1
    always_comb begin
        logic [4:0]  sh;
        logic        g, st;
        logic [11:0] kept;
        logic [24:0] shr;
        logic [25:0] up;
        int4_rnd_carry = 1'b0;
        int4_frac      = 10'd0;
        if (L > 5'd10) begin                                   // drop (L-10) LSBs -> RNE
            sh   = L - 5'd10;
            shr  = {9'd0, add_sum} >> sh;
            kept = {1'b0, shr[10:0]};                           // {hidden,10 frac}
            g    = add_sum[sh - 5'd1];                          // guard bit
            st   = |(add_sum & (({15'd0, 1'b1} << (sh - 5'd1)) - 16'd1));  // sticky
            if (g & (st | kept[0])) begin
                kept = kept + 12'd1;
                if (kept[11]) begin kept = kept >> 1; int4_rnd_carry = 1'b1; end
            end
            int4_frac = kept[9:0];
        end else begin                                          // exact (leading 1 <= bit10)
            up        = {10'd0, add_sum} << (5'd10 - L);
            int4_frac = up[9:0];
        end
    end

    logic signed [31:0] int4_expf;
    assign int4_expf = $signed({{24{expY[7]}}, expY}) + $signed({27'd0, L}) + 32'sd15
                       + $signed({31'd0, int4_rnd_carry});
    wire        int4_sign  = w_sign ^ signY;

    // ============================================================
    // 5b. L-Mul pack: sig_code = scale + (fx+fy) + offset ; one carry renormalise.
    //     m=4 : scale=16  offset=2  (const 18)  2*scale=32   frac<<6
    //     m=8 : scale=256 offset=16 (const 272) 2*scale=512  frac<<2
    // ============================================================
    wire        attn       = (i_mode == MODE_LMUL_ATTN);
    wire [10:0] lmul_const = attn ? 11'd272 : 11'd18;
    wire [10:0] two_scale  = attn ? 11'd512 : 11'd32;
    wire [10:0] scale_v    = attn ? 11'd256 : 11'd16;
    wire [3:0]  fshl       = attn ? 4'd2    : 4'd6;

    logic [11:0] sig_pre;          // fx+fy + const  (<= 782+ -> 11 bit, give 12)
    assign sig_pre = {2'd0, add_sum[9:0]} + {1'd0, lmul_const};
    wire         carry   = (sig_pre >= {1'd0, two_scale});
    logic [11:0] sig_cd;
    assign sig_cd = carry ? (sig_pre >> 1) : sig_pre;            // floor(/2) on carry
    logic [9:0]  lmul_frac;
    assign lmul_frac = (sig_cd[10:0] - scale_v) << fshl;         // (sig_code-scale)<<(10-m), fits 10b
    logic signed [31:0] lmul_expf;
    assign lmul_expf = $signed({{24{exX[7]}}, exX}) + $signed({{24{exY_l[7]}}, exY_l})
                       + $signed({31'd0, carry}) + 32'sd15;
    wire         lmul_sign = sXl ^ sYl;

    // ============================================================
    // 6. Output select + special cases
    // ============================================================
    localparam logic [14:0] QNAN15 = 15'h7E00;     // {exp=1F, frac=10'h200}
    localparam logic [14:0] INF15  = 15'h7C00;     // {exp=1F, frac=0}

    always_comb begin
        if (is_int4) begin
            if      (specialY)             o_prod_fp16 = {int4_sign, QNAN15};      // NaN/Inf act -> qNaN
            else if (w_zero | zeroY)       o_prod_fp16 = {int4_sign, 15'h0000};    // signed zero
            else if (int4_expf <= 0)       o_prod_fp16 = {int4_sign, 15'h0000};    // FTZ
            else if (int4_expf >= 31)      o_prod_fp16 = {int4_sign, INF15};       // overflow
            else                           o_prod_fp16 = {int4_sign, int4_expf[4:0], int4_frac};
        end else begin // L-Mul : zero / non-finite operand -> +0 (matches SW zero-mask)
            if      (zXl | zYl | spXl | spYl) o_prod_fp16 = 16'h0000;
            else if (lmul_expf <= 0)          o_prod_fp16 = 16'h0000;              // FTZ
            else if (lmul_expf >= 31)         o_prod_fp16 = {lmul_sign, INF15};    // overflow
            else                              o_prod_fp16 = {lmul_sign, lmul_expf[4:0], lmul_frac};
        end
    end

endmodule


/* ---------------------------------------------------------------------------
   pe_slice_addsub — shared significand adder, physically TWO 8-bit slices.
     i_split=0 : chained 16-bit add/sub (carry/borrow crosses bit7->bit8).
     i_split=1 : two independent 8-bit ops (carry gated) — for future dual-issue.
   a +/- b via 2's complement: a + (b ^ {16{sub}}) + sub.
--------------------------------------------------------------------------- */
module pe_slice_addsub (
    input  logic [15:0] i_a,
    input  logic [15:0] i_b,
    input  logic        i_sub,
    input  logic        i_split,
    output logic [15:0] o_sum
);
    logic [15:0] b_x;
    logic [8:0]  s0;          // {carry-out, sum} of slice 0
    logic        cin1;
    logic [8:0]  s1;

    assign b_x  = i_b ^ {16{i_sub}};
    assign s0   = {1'b0, i_a[7:0]}  + {1'b0, b_x[7:0]}  + {8'd0, i_sub};
    assign cin1 = i_split ? i_sub : s0[8];                 // chained: carry/borrow from slice0
    assign s1   = {1'b0, i_a[15:8]} + {1'b0, b_x[15:8]} + {8'd0, cin1};
    assign o_sum = {s1[7:0], s0[7:0]};
endmodule
