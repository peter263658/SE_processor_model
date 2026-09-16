/*
    PEG — W4A8 zero-multiplier PE group (Design B, precision-scalable).

    One group = 32 lanes = 16 precision-scalable pe_pair cells, reduced by a
    32-way FP16 SATURATING adder tree to a single PRE-DESCALE partial sum.

      i_mode    8-bit operand modes (W4A8 / FP8 L-Mul m4)  -> 128 lanes (32/grp)
                16-bit operand modes (FP16 L-Mul m8 / W4A16) -> 64 lanes (16/grp)
      pair p owns lane 2p (lo) and lane 2p+1 (hi):
        8-bit  : lo & hi each emit one FP16 product            -> 32 products
        16-bit : lo emits the FP16 product from the FULL 16-bit
                 operand on the EVEN lane slot; hi forced 0     -> 16 products + 16 zeros

    Pure combinational (20 MHz / 28 nm: mult + 32-way tree fits one cycle, no FF).
    Output buffering and the post-acc chain (per-group APoT dealer -> OACC ->
    BN/PReLU/cast FMA) live in CE, NOT here. o_accum is the pre-descale group sum.

    See: project_pe_lane_impl (Design B), 10_algo/docs/RTL_HANDOFF_PE_LANE_DATAPATH_2026-06-06.md
*/
module PEG
    import Param_def::*;
(   input  CE_LANE_mode                     i_mode,
    input  logic [NUM_PE_PG-1:0][15:0]      i_x,        // per-lane x / weight / Q operand
    input  logic [NUM_PE_PG-1:0][15:0]      i_y,        // per-lane y / activation / K operand
    input  logic signed [7:0]               i_preshift, // per-tensor pre-tree pow2 (broadcast to all pairs); =0 INT4
    output logic [15:0]                     o_accum     // pre-descale 32-way FP16 partial sum
);
    localparam int NUM_PAIR = NUM_PE_PG/2;           // 16

    logic [NUM_PE_PG-1:0][15:0] prod;                // 32 FP16 per-element products

    genvar p;
    generate
        for (p = 0; p < NUM_PAIR; p++) begin : g_pair
            pe_pair u_pair (
                .i_mode    (i_mode),
                .i_x_lo    (i_x[2*p]),   .i_y_lo (i_y[2*p]),
                .i_x_hi    (i_x[2*p+1]), .i_y_hi (i_y[2*p+1]),
                .i_preshift(i_preshift),
                .o_prod_lo (prod[2*p]),
                .o_prod_hi (prod[2*p+1])
            );
        end
    endgenerate

    adder_tree_fp16 u_tree (.i_add_array(prod), .o_accum(o_accum));
endmodule


/* ---------------------------------------------------------------------------
   adder_tree_fp16 — pure combinational 32-way FP16 adder tree with SATURATING
   add (clamp +/-Inf -> +/- max-normal 0x7BFF/0xFBFF).

   Saturation (not Inf) is REQUIRED: descale is post-accumulation, so this tree
   sums PRE-descale products whose worst-case single-group magnitude
   (32*(7*448) = 100352) exceeds FP16 max 65504. A silent Inf would destroy the
   whole output and MBSTOI cannot see occasional Inf.
   See RTL_HANDOFF_PE_LANE_DATAPATH_2026-06-06.md (pitfall #1).

   Reduction order is a fixed paired binary tree, so a behavioural reference can
   reproduce it bit-exactly.
--------------------------------------------------------------------------- */
module adder_tree_fp16
    import Param_def::*;
(   input  logic [NUM_PE_PG-1:0][15:0] i_add_array,   // 32 FP16
    output logic [15:0]                o_accum
);
    logic [15:0] s16 [0:15];
    logic [15:0] s8  [0:7];
    logic [15:0] s4  [0:3];
    logic [15:0] s2  [0:1];

    genvar i;
    generate
        for (i = 0; i < 16; i++) begin : g_l0
            fp16_add_sat u (.a(i_add_array[2*i]), .b(i_add_array[2*i+1]), .z(s16[i]));
        end
        for (i = 0; i < 8; i++) begin : g_l1
            fp16_add_sat u (.a(s16[2*i]), .b(s16[2*i+1]), .z(s8[i]));
        end
        for (i = 0; i < 4; i++) begin : g_l2
            fp16_add_sat u (.a(s8[2*i]), .b(s8[2*i+1]), .z(s4[i]));
        end
        for (i = 0; i < 2; i++) begin : g_l3
            fp16_add_sat u (.a(s4[2*i]), .b(s4[2*i+1]), .z(s2[i]));
        end
    endgenerate
    fp16_add_sat u_l4 (.a(s2[0]), .b(s2[1]), .z(o_accum));
endmodule


/* ---------------------------------------------------------------------------
   fp16_add_sat — FP16 add (DesignWare, round-to-nearest-even) with overflow
   clamped to +/- max-normal (0x7BFF/0xFBFF) instead of Inf. Inputs are finite
   in normal operation; NaN propagates (only Inf-with-zero-frac is saturated).
--------------------------------------------------------------------------- */
module fp16_add_sat
    import Param_def::*;
(   input  logic [15:0] a,
    input  logic [15:0] b,
    output logic [15:0] z
);
    logic [15:0] z_raw;
    DW_fp_add_inst #(
        .sig_width(FP_HP_FRAC_WIDTH),   // 10
        .exp_width(FP_HP_EXPN_WIDTH),   // 5
        .ieee_compliance(1)
    ) u_add (
        .inst_a     (a),
        .inst_b     (b),
        .inst_rnd   (3'b000),           // round-to-nearest-even
        .z_inst     (z_raw),
        .status_inst()
    );

    wire is_inf = (z_raw[14:10] == 5'h1F) && (z_raw[9:0] == 10'h000);
    assign z = is_inf ? {z_raw[15], 15'h7BFF} : z_raw;   // +/- max-normal
endmodule


// pe_lane (golden reference) + its private pe_slice_addsub were MOVED to
// 00_TB/CE/pe_lane_ref.sv (TB-only; not synthesized). The DUT path (PEG/CE_core)
// uses pe_pair, which shares pe_fp_unpack / pe_int4_csd (kept below).


/* ---------------------------------------------------------------------------
   pe_fp_unpack — IEEE-style unpack to frexp convention.
     value = (-1)^o_sign * o_sig * 2^o_exp ,  o_sig integer width FRAC_W+1,
     leading 1 at bit FRAC_W for any nonzero (subnormals normalised via LZD).
     o_special = NaN (and, when HAS_INF, also Inf) — caller routes to qNaN / 0.
--------------------------------------------------------------------------- */
module pe_fp_unpack #(
    parameter int EXP_W   = 5,
    parameter int FRAC_W  = 10,
    parameter bit HAS_INF = 1
)(
    input  logic [EXP_W+FRAC_W:0] i_fp,
    output logic                  o_sign,
    output logic signed [7:0]     o_exp,
    output logic [FRAC_W:0]       o_sig,
    output logic                  o_zero,
    output logic                  o_special
);
    localparam int BIAS = (1 << (EXP_W-1)) - 1;
    localparam int W    = EXP_W + FRAC_W + 1;

    logic              s;
    logic [EXP_W-1:0]  e;
    logic [FRAC_W-1:0] f;
    assign s = i_fp[W-1];
    assign e = i_fp[W-2 -: EXP_W];
    assign f = i_fp[FRAC_W-1:0];

    wire e_all1 = (e == {EXP_W{1'b1}});
    wire e_zero = (e == {EXP_W{1'b0}});
    wire f_all1 = (f == {FRAC_W{1'b1}});
    wire f_zero = (f == {FRAC_W{1'b0}});

    assign o_sign    = s;
    assign o_zero    = e_zero & f_zero;
    // FP16: e=all1 -> Inf/NaN (special).  E4M3: only e=all1 & f=all1 is NaN.
    assign o_special = HAS_INF ? e_all1 : (e_all1 & f_all1);

    // leading-one index of f (for subnormal normalisation)
    logic [$clog2(FRAC_W+1)-1:0] p;
    always_comb begin
        p = '0;
        for (int i = 0; i < FRAC_W; i++) if (f[i]) p = i[$bits(p)-1:0];
    end

    int exp_i;
    always_comb begin
        if (e_zero) begin                 // subnormal (zero handled by o_zero downstream)
            o_sig = f << (FRAC_W - p);     // shift leading 1 up to bit FRAC_W
            exp_i = 1 - BIAS - 2*FRAC_W + int'(p);
        end else begin                    // normal
            o_sig = {1'b1, f};
            exp_i = int'(e) - BIAS - FRAC_W;
        end
        o_exp = exp_i[7:0];
    end
endmodule


/* ---------------------------------------------------------------------------
   pe_int4_csd — INT4 (2's complement, [-7,+7]) -> sign + canonical-signed-digit
   decode into <=2 terms: |q| = (1<<shA) [+/-] (1<<shB).  Two shifters + 1 add/sub.
   (raw 4'b1000 = -8 is don't-care; software clamps to [-7,+7].)
--------------------------------------------------------------------------- */
module pe_int4_csd (
    input  logic [3:0] i_w,
    output logic       o_sign,
    output logic       o_zero,
    output logic [1:0] o_shA,
    output logic [1:0] o_shB,
    output logic       o_useB,
    output logic       o_sub
);
    logic [2:0] mag;
    assign o_sign = i_w[3];
    assign mag    = i_w[3] ? (~i_w[2:0] + 3'd1) : i_w[2:0];
    assign o_zero = (mag == 3'd0);

    always_comb begin
        o_shA = 2'd0; o_shB = 2'd0; o_useB = 1'b0; o_sub = 1'b0;
        unique case (mag)
            3'd0: ;                                                          // 0
            3'd1: begin o_shA = 2'd0;                              end       // x
            3'd2: begin o_shA = 2'd1;                              end       // x<<1
            3'd3: begin o_shA = 2'd1; o_shB = 2'd0; o_useB = 1'b1; end       // x<<1 + x
            3'd4: begin o_shA = 2'd2;                              end       // x<<2
            3'd5: begin o_shA = 2'd2; o_shB = 2'd0; o_useB = 1'b1; end       // x<<2 + x
            3'd6: begin o_shA = 2'd2; o_shB = 2'd1; o_useB = 1'b1; end       // x<<2 + x<<1
            3'd7: begin o_shA = 2'd3; o_shB = 2'd0; o_useB = 1'b1; o_sub = 1'b1; end // x<<3 - x
            default: ;   // mag is 3-bit (0..7) -> all covered; silences the time-0 X 'no match' note
        endcase
    end
endmodule


// pe_slice_addsub MOVED to 00_TB/CE/pe_lane_ref.sv (only pe_lane used it).

// =======================================================================
// pe_pair — moved here from pe_pair.sv (file consolidation 2026-06-10). Shares
// pe_fp_unpack / pe_int4_csd defined in this file (PEG.sv).
// =======================================================================
/*
    pe_pair — Design B precision-scalable cell (TWO 8-bit slices, fused).

    Self-contained: pe_lane's datapath is folded in, so the pairing (carry chain /
    hierarchical LZD / peer significand) is INTERNAL wiring — no cross-module ports.

    The significand adder is physically TWO 8-bit slices (slice_lo, slice_hi):
      8-bit modes (W4A8 / FP8 L-Mul m4): slices run INDEPENDENT -> 2 products
          o_prod_lo = product(x_lo, y_lo)   o_prod_hi = product(x_hi, y_hi)
      16-bit modes (FP16 L-Mul m8 / W4A16): slice_lo (low byte) + slice_hi (high
          byte) CHAIN their carry into one 16-bit add -> 1 product on lo
          o_prod_lo = product(x_lo[15:0], y_lo[15:0])   o_prod_hi = 0

    => "two 8-bit adders combined into a 16-bit adder" (cin_hi = slice_lo carry-out
    in 16-bit mode; independent in 8-bit mode). lo carries the 16-bit normalize/RNE
    (needed for W4A16); hi is 8-bit-only.

    Bit-exact to pe_lane: the shared adder produces the SAME 16-bit `lo_add16` as
    pe_lane's pe_slice_addsub, and the lo pack mirrors pe_lane; the hi pack mirrors
    pe_lane for the 8-bit modes. Helpers pe_fp_unpack / pe_int4_csd shared from PEG.sv.
    Pure combinational.
*/
`timescale 1ns/10ps
module pe_pair
    import Param_def::*;
(
    input  CE_LANE_mode i_mode,
    input  logic [15:0] i_x_lo, i_y_lo,    // 8-bit modes: lo operand; 16-bit modes: full FP16 operand
    input  logic [15:0] i_x_hi, i_y_hi,    // 8-bit modes: hi operand; 16-bit modes: unused
    input  logic signed [7:0] i_preshift,  // per-tensor pre-tree pow2 (overflow fix); =0 → bit-exact
    output logic [15:0] o_prod_lo,
    output logic [15:0] o_prod_hi
);
    wire is_16bit    = (i_mode == MODE_LMUL_ATTN) || (i_mode == MODE_W4A16);
    wire lo_is_int4  = (i_mode == MODE_W4A8)       || (i_mode == MODE_W4A16);
    wire hi_is_int4  = (i_mode == MODE_W4A8);                 // hi only sees modes 0/1
    wire y_lo_fp16   = (i_mode == MODE_LMUL_ATTN)  || (i_mode == MODE_W4A16);
    wire attn        = (i_mode == MODE_LMUL_ATTN);

    // Per-tensor pre-tree pow2 descale, sign-extended to product-exponent width. Folded into
    // all 4 product exponents below = zero extra hardware (the exp adders already exist).
    // =0 → every term vanishes → bit-exact to the pre-fix design. Legal because the three
    // overflowing descales (FP8 weight, QK Q/K 2^-k, ScoreV prob/256 + V 2^-k) are per-tensor
    // constants along the contraction axis: descale·Σ(w·x) = Σ((descale)·w·x). 雷② (per-GROUP
    // INT4 pointwise descale) keeps preshift=0 and is untouched. Residual stays in the dealer.
    wire signed [31:0] preshift_ext = $signed({{24{i_preshift[7]}}, i_preshift});

    // ============================================================
    // Unpack — lo: fp8 + fp16 (x,y) ; hi: fp8 only (x,y)
    // ============================================================
    logic        yl8_s, yl8_z, yl8_sp;   logic signed [7:0] yl8_e;   logic [3:0]  yl8_sig;
    logic        yl16_s,yl16_z,yl16_sp;  logic signed [7:0] yl16_e;  logic [10:0] yl16_sig;
    logic        xl8_s, xl8_z, xl8_sp;   logic signed [7:0] xl8_e;   logic [3:0]  xl8_sig;
    logic        xl16_s,xl16_z,xl16_sp;  logic signed [7:0] xl16_e;  logic [10:0] xl16_sig;
    pe_fp_unpack #(.EXP_W(4), .FRAC_W(3),  .HAS_INF(0)) u_yl8  (.i_fp(i_y_lo[7:0]),  .o_sign(yl8_s),  .o_exp(yl8_e),  .o_sig(yl8_sig),  .o_zero(yl8_z),  .o_special(yl8_sp));
    pe_fp_unpack #(.EXP_W(5), .FRAC_W(10), .HAS_INF(1)) u_yl16 (.i_fp(i_y_lo[15:0]), .o_sign(yl16_s), .o_exp(yl16_e), .o_sig(yl16_sig), .o_zero(yl16_z), .o_special(yl16_sp));
    pe_fp_unpack #(.EXP_W(4), .FRAC_W(3),  .HAS_INF(0)) u_xl8  (.i_fp(i_x_lo[7:0]),  .o_sign(xl8_s),  .o_exp(xl8_e),  .o_sig(xl8_sig),  .o_zero(xl8_z),  .o_special(xl8_sp));
    pe_fp_unpack #(.EXP_W(5), .FRAC_W(10), .HAS_INF(1)) u_xl16 (.i_fp(i_x_lo[15:0]), .o_sign(xl16_s), .o_exp(xl16_e), .o_sig(xl16_sig), .o_zero(xl16_z), .o_special(xl16_sp));

    logic        yh8_s, yh8_z, yh8_sp;   logic signed [7:0] yh8_e;   logic [3:0]  yh8_sig;
    logic        xh8_s, xh8_z, xh8_sp;   logic signed [7:0] xh8_e;   logic [3:0]  xh8_sig;
    pe_fp_unpack #(.EXP_W(4), .FRAC_W(3),  .HAS_INF(0)) u_yh8 (.i_fp(i_y_hi[7:0]), .o_sign(yh8_s), .o_exp(yh8_e), .o_sig(yh8_sig), .o_zero(yh8_z), .o_special(yh8_sp));
    pe_fp_unpack #(.EXP_W(4), .FRAC_W(3),  .HAS_INF(0)) u_xh8 (.i_fp(i_x_hi[7:0]), .o_sign(xh8_s), .o_exp(xh8_e), .o_sig(xh8_sig), .o_zero(xh8_z), .o_special(xh8_sp));

    // ============================================================
    // INT4 CSD decode (lo, hi)
    // ============================================================
    logic       wl_sign, wl_zero, wl_useB, wl_sub;  logic [1:0] shA_l, shB_l;
    logic       wh_sign, wh_zero, wh_useB, wh_sub;  logic [1:0] shA_h, shB_h;
    pe_int4_csd u_csd_l (.i_w(i_x_lo[3:0]), .o_sign(wl_sign), .o_zero(wl_zero), .o_shA(shA_l), .o_shB(shB_l), .o_useB(wl_useB), .o_sub(wl_sub));
    pe_int4_csd u_csd_h (.i_w(i_x_hi[3:0]), .o_sign(wh_sign), .o_zero(wh_zero), .o_shA(shA_h), .o_shB(shB_h), .o_useB(wh_useB), .o_sub(wh_sub));

    // ============================================================
    // LO operand prep (mirror pe_lane)
    // ============================================================
    logic [10:0]       sigY_l;  logic signed [7:0] expY_l;  logic signY_l, zeroY_l, specialY_l;
    always_comb begin
        if (y_lo_fp16) begin sigY_l = yl16_sig;        expY_l = yl16_e; signY_l = yl16_s; zeroY_l = yl16_z; specialY_l = yl16_sp; end
        else           begin sigY_l = {7'd0, yl8_sig}; expY_l = yl8_e;  signY_l = yl8_s;  zeroY_l = yl8_z;  specialY_l = yl8_sp;  end
    end
    logic [15:0] int4A_l, int4B_l;
    assign int4A_l = {5'd0, sigY_l} << shA_l;
    assign int4B_l = wl_useB ? ({5'd0, sigY_l} << shB_l) : 16'd0;

    logic [7:0]        fx_l, fy_l;  logic signed [7:0] exX_l, exY_ll;
    logic              sXl_l, sYl_l, zXl_l, zYl_l, spXl_l, spYl_l;
    always_comb begin
        if (attn) begin                                   // fp16 m=8
            fx_l = xl16_sig[9:2];                fy_l = yl16_sig[9:2];
            exX_l = xl16_e + 8'sd10;             exY_ll = yl16_e + 8'sd10;
            sXl_l=xl16_s; sYl_l=yl16_s; zXl_l=xl16_z; zYl_l=yl16_z; spXl_l=xl16_sp; spYl_l=yl16_sp;
        end else begin                                    // fp8 m=4
            fx_l = {4'd0, xl8_sig[2:0], 1'b0};   fy_l = {4'd0, yl8_sig[2:0], 1'b0};
            exX_l = xl8_e + 8'sd3;               exY_ll = yl8_e + 8'sd3;
            sXl_l=xl8_s;  sYl_l=yl8_s;  zXl_l=xl8_z;  zYl_l=yl8_z;  spXl_l=xl8_sp;  spYl_l=yl8_sp;
        end
    end

    // ============================================================
    // HI operand prep (mirror pe_lane, fp8 / m4 only)
    // ============================================================
    logic [7:0] int4A_h, int4B_h;
    assign int4A_h = {4'd0, yh8_sig} << shA_h;            // <=15<<3=120 fits 8-bit
    assign int4B_h = wh_useB ? ({4'd0, yh8_sig} << shB_h) : 8'd0;
    logic [7:0]        fx_h, fy_h;  logic signed [7:0] exX_h, exY_lh;
    assign fx_h = {4'd0, xh8_sig[2:0], 1'b0};
    assign fy_h = {4'd0, yh8_sig[2:0], 1'b0};
    assign exX_h  = xh8_e + 8'sd3;
    assign exY_lh = yh8_e + 8'sd3;

    // ============================================================
    // SHARED 2x8-bit adder/sub : slice_lo + slice_hi.
    //   8-bit  : slice_lo=lo product, slice_hi=hi product (independent subs).
    //   16-bit : slice_lo=lo low byte, slice_hi=lo high byte, carry chained.
    // a +/- b via 2's complement: a + (b ^ {sub}) + sub.
    // ============================================================
    logic [15:0] a_lo16, b_lo16;  logic sub_lo;
    always_comb begin
        if (lo_is_int4) begin a_lo16 = int4A_l;      b_lo16 = int4B_l;      sub_lo = wl_sub; end
        else            begin a_lo16 = {8'd0, fx_l}; b_lo16 = {8'd0, fy_l}; sub_lo = 1'b0;   end
    end
    logic [7:0] a_hi8, b_hi8;  logic sub_hi;
    always_comb begin
        if (hi_is_int4) begin a_hi8 = int4A_h;  b_hi8 = int4B_h;  sub_hi = wh_sub; end
        else            begin a_hi8 = fx_h;     b_hi8 = fy_h;     sub_hi = 1'b0;   end
    end

    // slice_lo : always lo's low byte
    logic [7:0] bx_lo;  logic [8:0] s_lo;
    assign bx_lo = b_lo16[7:0] ^ {8{sub_lo}};
    assign s_lo  = {1'b0, a_lo16[7:0]} + {1'b0, bx_lo} + {8'd0, sub_lo};

    // slice_hi : 16-bit -> lo high byte (chained) ; 8-bit -> hi product (independent)
    logic [7:0] a_hi_in, b_hi_in;  logic sub_hi_eff, cin_hi;
    always_comb begin
        if (is_16bit) begin a_hi_in = a_lo16[15:8]; b_hi_in = b_lo16[15:8]; sub_hi_eff = sub_lo; cin_hi = s_lo[8]; end
        else          begin a_hi_in = a_hi8;        b_hi_in = b_hi8;        sub_hi_eff = sub_hi; cin_hi = sub_hi; end
    end
    logic [7:0] bx_hi;  logic [8:0] s_hi;
    assign bx_hi = b_hi_in ^ {8{sub_hi_eff}};
    assign s_hi  = {1'b0, a_hi_in} + {1'b0, bx_hi} + {8'd0, cin_hi};

    wire [15:0] lo_add16 = is_16bit ? {s_hi[7:0], s_lo[7:0]} : {8'd0, s_lo[7:0]};
    wire [7:0]  hi_add8  = s_hi[7:0];

    // ============================================================
    // LO pack (mirror pe_lane stages 5a/5b/6, add_sum = lo_add16)
    // ============================================================
    logic [4:0] Ll;
    always_comb begin
        Ll = 5'd0;
        for (int i = 0; i < 16; i++) if (lo_add16[i]) Ll = i[4:0];
    end
    logic [9:0] int4_frac_l;  logic int4_rc_l;
    always_comb begin
        logic [4:0]  sh;  logic g, st;  logic [11:0] kept;  logic [24:0] shr;  logic [25:0] up;
        int4_rc_l   = 1'b0;
        int4_frac_l = 10'd0;
        if (Ll > 5'd10) begin
            sh   = Ll - 5'd10;
            shr  = {9'd0, lo_add16} >> sh;
            kept = {1'b0, shr[10:0]};
            g    = lo_add16[sh - 5'd1];
            st   = |(lo_add16 & (({15'd0, 1'b1} << (sh - 5'd1)) - 16'd1));
            if (g & (st | kept[0])) begin
                kept = kept + 12'd1;
                if (kept[11]) begin kept = kept >> 1; int4_rc_l = 1'b1; end
            end
            int4_frac_l = kept[9:0];
        end else begin
            up          = {10'd0, lo_add16} << (5'd10 - Ll);
            int4_frac_l = up[9:0];
        end
    end
    logic signed [31:0] int4_expf_l;
    assign int4_expf_l = $signed({{24{expY_l[7]}}, expY_l}) + $signed({27'd0, Ll}) + 32'sd15
                         + $signed({31'd0, int4_rc_l}) + preshift_ext;
    wire int4_sign_l = wl_sign ^ signY_l;

    wire [10:0] lmul_const_l = attn ? 11'd272 : 11'd18;
    wire [10:0] two_scale_l  = attn ? 11'd512 : 11'd32;
    wire [10:0] scale_v_l    = attn ? 11'd256 : 11'd16;
    wire [3:0]  fshl_l       = attn ? 4'd2    : 4'd6;
    logic [11:0] sig_pre_l;
    assign sig_pre_l = {2'd0, lo_add16[9:0]} + {1'd0, lmul_const_l};
    wire         carry_l = (sig_pre_l >= {1'd0, two_scale_l});
    logic [11:0] sig_cd_l;
    assign sig_cd_l = carry_l ? (sig_pre_l >> 1) : sig_pre_l;
    logic [9:0]  lmul_frac_l;
    assign lmul_frac_l = (sig_cd_l[10:0] - scale_v_l) << fshl_l;
    logic signed [31:0] lmul_expf_l;
    assign lmul_expf_l = $signed({{24{exX_l[7]}}, exX_l}) + $signed({{24{exY_ll[7]}}, exY_ll})
                         + $signed({31'd0, carry_l}) + 32'sd15 + preshift_ext;
    wire lmul_sign_l = sXl_l ^ sYl_l;

    localparam logic [14:0] QNAN15 = 15'h7E00;
    localparam logic [14:0] INF15  = 15'h7C00;
    always_comb begin
        if (lo_is_int4) begin
            if      (specialY_l)            o_prod_lo = {int4_sign_l, QNAN15};
            else if (wl_zero | zeroY_l)     o_prod_lo = {int4_sign_l, 15'h0000};
            else if (int4_expf_l <= 0)      o_prod_lo = {int4_sign_l, 15'h0000};
            else if (int4_expf_l >= 31)     o_prod_lo = {int4_sign_l, INF15};
            else                            o_prod_lo = {int4_sign_l, int4_expf_l[4:0], int4_frac_l};
        end else begin
            if      (zXl_l | zYl_l | spXl_l | spYl_l) o_prod_lo = 16'h0000;
            else if (lmul_expf_l <= 0)                o_prod_lo = 16'h0000;
            else if (lmul_expf_l >= 31)               o_prod_lo = {lmul_sign_l, INF15};
            else                                      o_prod_lo = {lmul_sign_l, lmul_expf_l[4:0], lmul_frac_l};
        end
    end

    // ============================================================
    // HI pack (mirror pe_lane modes 0/1, 8-bit, add_sum = hi_add8). 16-bit -> 0.
    // ============================================================
    logic [3:0] Lh;
    always_comb begin
        Lh = 4'd0;
        for (int i = 0; i < 8; i++) if (hi_add8[i]) Lh = i[3:0];
    end
    logic [17:0] up_h;
    assign up_h = {10'd0, hi_add8} << (5'd10 - {1'b0, Lh});   // 8-bit product: Lh<=7<10, exact
    wire  [9:0]  int4_frac_h = up_h[9:0];
    logic signed [31:0] int4_expf_h;
    assign int4_expf_h = $signed({{24{yh8_e[7]}}, yh8_e}) + $signed({28'd0, Lh}) + 32'sd15 + preshift_ext;
    wire int4_sign_h = wh_sign ^ yh8_s;

    logic [7:0] sig_pre_h;
    assign sig_pre_h = hi_add8 + 8'd18;                       // m4: hi_add8 = fx+fy
    wire        carry_h = (sig_pre_h >= 8'd32);
    logic [7:0] sig_cd_h;
    assign sig_cd_h = carry_h ? (sig_pre_h >> 1) : sig_pre_h;
    logic [9:0] lmul_frac_h;
    assign lmul_frac_h = ({2'd0, (sig_cd_h - 8'd16)}) << 6;
    logic signed [31:0] lmul_expf_h;
    assign lmul_expf_h = $signed({{24{exX_h[7]}}, exX_h}) + $signed({{24{exY_lh[7]}}, exY_lh})
                         + $signed({31'd0, carry_h}) + 32'sd15 + preshift_ext;
    wire lmul_sign_h = xh8_s ^ yh8_s;

    logic [15:0] prod_hi8;
    always_comb begin
        if (hi_is_int4) begin
            if      (yh8_sp)               prod_hi8 = {int4_sign_h, QNAN15};
            else if (wh_zero | yh8_z)      prod_hi8 = {int4_sign_h, 15'h0000};
            else if (int4_expf_h <= 0)     prod_hi8 = {int4_sign_h, 15'h0000};
            else if (int4_expf_h >= 31)    prod_hi8 = {int4_sign_h, INF15};
            else                           prod_hi8 = {int4_sign_h, int4_expf_h[4:0], int4_frac_h};
        end else begin
            if      (xh8_z | yh8_z | xh8_sp | yh8_sp) prod_hi8 = 16'h0000;
            else if (lmul_expf_h <= 0)                prod_hi8 = 16'h0000;
            else if (lmul_expf_h >= 31)               prod_hi8 = {lmul_sign_h, INF15};
            else                                      prod_hi8 = {lmul_sign_h, lmul_expf_h[4:0], lmul_frac_h};
        end
    end

    assign o_prod_hi = is_16bit ? 16'h0000 : prod_hi8;
endmodule
