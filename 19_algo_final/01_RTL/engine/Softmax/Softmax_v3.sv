/*
    Module: Fused shift-based Softmax + Top-K approximation

    Primary mode from RTL_HANDOFF_SOFTMAX_2026-05-14.md:

        score-domain Top-K=4
        -> top-score centering
        -> 1.5x, clamp [-8, 0], floor
        -> pow2 Q0.8 weight
        -> nearest-pow2 denominator
        -> sparse probability scatter

    This module keeps the legacy dense 32-entry probability output for
    integration convenience. Only the selected Top-K entries are nonzero.

    Implementation note: all combinational helpers are instantiated as
    sub-modules (sm_*) rather than SystemVerilog functions, so their I/O is
    visible in the hierarchy / waveform. The per-lane Top-K insertion is wrapped
    in sm_topk_insert and chained SM_SCAN_LANES times to reproduce the original
    sequential insertion-sort within one SCAN cycle.
*/

module Softmax
    import Param_def::*;
(
    input   logic                                                   i_clk,
    input   logic                                                   i_rst_n,
    input   logic                                                   i_SM_valid,
    input   logic   [SEQ_LEN-1 : 0] [ATTN_FP_WIDTH-1 : 0]           i_SM_data,
    input   logic   [SEQ_LEN-1 : 0]                                 i_SM_mask,
    output  logic                                                   o_SM_ready,
    output  logic                                                   o_SM_valid,
    output  logic   [SEQ_LEN-1 : 0] [ATTN_FP_WIDTH-1 : 0]           o_SM_data
);
    localparam TOP_K           = 4;
    localparam N_GRP           = (SEQ_LEN + SM_SCAN_LANES - 1) / SM_SCAN_LANES;
    localparam GRP_BIT         = (N_GRP <= 1) ? 1 : $clog2(N_GRP);
    localparam PROB_FRAC       = 8;
    localparam WEIGHT_WIDTH     = PROB_FRAC + 1;  // Q0.8, 0..256
    localparam WEIGHT_EXP_WIDTH = 4;              // pow2 exponent, 0..8
    localparam DENOM_WIDTH     = PROB_FRAC + 3;  // 4 * 256 = 1024
    localparam SCORE_Q_FRAC    = 4;
    localparam SCORE_Q_WIDTH   = 16;

    localparam logic [GRP_BIT-1 : 0]           LAST_GRP     = N_GRP - 1;
    localparam logic [GRP_BIT-1 : 0]           GRP_ZERO     = {GRP_BIT{1'b0}};
    localparam logic [ATTN_FP_WIDTH-1 : 0]     FP_ZERO      = {ATTN_FP_WIDTH{1'b0}};
    localparam logic [SEQ_LEN_BIT-1 : 0]       SEQ_IDX_ZERO = {SEQ_LEN_BIT{1'b0}};
    localparam logic [WEIGHT_WIDTH-1 : 0]      WEIGHT_ZERO  = {WEIGHT_WIDTH{1'b0}};
    localparam logic [DENOM_WIDTH-1 : 0]       DENOM_ZERO   = {DENOM_WIDTH{1'b0}};

    typedef enum logic [1 : 0] {
        SM_IDLE,
        SM_SCAN,
        SM_CALC,
        SM_DONE
    } SM_state;

    SM_state state_w, state_r;

    logic [SEQ_LEN-1 : 0] [ATTN_FP_WIDTH-1 : 0] score_w, score_r;
    logic [SEQ_LEN-1 : 0]                       mask_w, mask_r;
    logic [SEQ_LEN-1 : 0] [ATTN_FP_WIDTH-1 : 0] prob_w, prob_r;

    logic [TOP_K-1 : 0] [ATTN_FP_WIDTH-1 : 0]   top_score_w, top_score_r;
    logic [TOP_K-1 : 0] [SEQ_LEN_BIT-1 : 0]     top_idx_w, top_idx_r;
    logic [TOP_K-1 : 0]                         top_valid_w, top_valid_r;

    logic [GRP_BIT-1 : 0]                       grp_idx_w, grp_idx_r;
    logic [SEQ_LEN_BIT : 0]                     grp_base;
    logic                                       grp_last;
    logic                                       accept;

    logic [TOP_K-1 : 0] [ATTN_FP_WIDTH-1 : 0]   scan_score;
    logic [TOP_K-1 : 0] [SEQ_LEN_BIT-1 : 0]     scan_idx;
    logic [TOP_K-1 : 0]                         scan_valid;

    logic [TOP_K-1 : 0] [WEIGHT_WIDTH-1 : 0]    weight_q;
    logic [TOP_K-1 : 0] [WEIGHT_EXP_WIDTH-1 : 0] weight_exp;
    logic [DENOM_WIDTH-1 : 0]                   denom_q;
    logic [3 : 0]                               denom_exp;

    // Top-K insertion chain wiring (SM_SCAN_LANES stages per SCAN cycle)
    logic [SM_SCAN_LANES : 0] [TOP_K-1 : 0] [ATTN_FP_WIDTH-1 : 0] chain_score;
    logic [SM_SCAN_LANES : 0] [TOP_K-1 : 0] [SEQ_LEN_BIT-1 : 0]   chain_idx;
    logic [SM_SCAN_LANES : 0] [TOP_K-1 : 0]                       chain_valid;

    // Weight / denominator datapath wiring — fp16-delta (BIT-EXACT to deploy _shift_topk_softmax:
    // x = score-max in fp16, scaled = 1.5*x = x + x/2 (shift-add, no multiplier), k = floor(scaled)
    // clamp[-8,0]). REPLACES the old sq4 (Q?.4) quantization which collapsed near-max ties — see
    // CONSULT_SOFTMAX_SCORE_DELTA_PRECISION_2026-06-09.md.
    logic [ATTN_FP_WIDTH-1 : 0]                 neg_max  [0 : TOP_K-1];
    logic [ATTN_FP_WIDTH-1 : 0]                 x16      [0 : TOP_K-1];   // top[gk]-top[0]  (fp16, <=0)
    logic [ATTN_FP_WIDTH-1 : 0]                 xh16     [0 : TOP_K-1];   // x/2  (exponent decrement)
    logic [ATTN_FP_WIDTH-1 : 0]                 scaled16 [0 : TOP_K-1];   // 1.5*x = x + x/2  (fp16)
    logic signed [4 : 0]                        k_i      [0 : TOP_K-1];
    logic [TOP_K-1 : 0] [WEIGHT_WIDTH-1 : 0]    weight_raw;

    // Sparse probability scatter wiring
    logic signed [5 : 0]                        prob_exp_q8 [0 : TOP_K-1];
    logic [ATTN_FP_WIDTH-1 : 0]                 prob_fp     [0 : TOP_K-1];

    genvar gl;
    genvar gk;

    assign accept     = i_SM_valid && o_SM_ready;
    assign o_SM_ready = (state_r == SM_IDLE);
    assign o_SM_valid = (state_r == SM_DONE);
    assign o_SM_data  = prob_r;

    assign grp_base = grp_idx_r * SM_SCAN_LANES;
    assign grp_last = (grp_idx_r == LAST_GRP);

    //--------------------------------------------------------------------------
    // Top-K score-domain selection (sequential insertion over SM_SCAN_LANES lanes,
    // unrolled into a combinational chain of sm_topk_insert stages)
    //--------------------------------------------------------------------------
    assign chain_score[0] = top_score_r;
    assign chain_idx[0]   = top_idx_r;
    assign chain_valid[0] = top_valid_r;

    generate
        for (gl = 0; gl < SM_SCAN_LANES; gl = gl + 1) begin : g_scan
            logic [SEQ_LEN_BIT : 0]     lane_idx;
            logic                       lane_valid;
            logic [ATTN_FP_WIDTH-1 : 0] cand_score;

            assign lane_idx   = grp_base + gl;
            assign lane_valid = (lane_idx < SEQ_LEN) &&
                                mask_r[lane_idx[SEQ_LEN_BIT-1 : 0]];
            assign cand_score = lane_valid ? score_r[lane_idx[SEQ_LEN_BIT-1 : 0]]
                                           : FP_ZERO;

            sm_topk_insert #(
                .TOP_K (TOP_K),
                .FPW   (ATTN_FP_WIDTH),
                .IDXW  (SEQ_LEN_BIT)
            ) u_insert (
                .i_score      (chain_score[gl]),
                .i_idx        (chain_idx[gl]),
                .i_valid      (chain_valid[gl]),
                .i_cand_score (cand_score),
                .i_cand_idx   (lane_idx[SEQ_LEN_BIT-1 : 0]),
                .i_cand_valid (lane_valid),
                .o_score      (chain_score[gl+1]),
                .o_idx        (chain_idx[gl+1]),
                .o_valid      (chain_valid[gl+1])
            );
        end
    endgenerate

    assign scan_score = chain_score[SM_SCAN_LANES];
    assign scan_idx   = chain_idx[SM_SCAN_LANES];
    assign scan_valid = chain_valid[SM_SCAN_LANES];

    //--------------------------------------------------------------------------
    // Weight + denominator: 1.5x centering -> clamp/floor -> pow2 Q0.8 weight
    //--------------------------------------------------------------------------
    generate
        for (gk = 0; gk < TOP_K; gk = gk + 1) begin : g_wcalc
            if (gk == 0) begin : g_k0
                assign neg_max[gk]    = '0;
                assign x16[gk]        = FP_ZERO;
                assign xh16[gk]       = FP_ZERO;
                assign scaled16[gk]   = FP_ZERO;
                assign k_i[gk]        = 5'sd0;
                assign weight_exp[gk] = 4'd8;          // top score (x=0) -> 2^0 weight
            end else begin : g_kn
                // x = top[gk] - top[0]  (fp16; <=0, exact for nearby scores by Sterbenz)
                assign neg_max[gk] = {~top_score_r[0][ATTN_FP_WIDTH-1], top_score_r[0][ATTN_FP_WIDTH-2:0]};
                DW_fp_add_inst #(.sig_width(ATTN_FP_FRAC_WIDTH), .exp_width(ATTN_FP_EXPN_WIDTH), .ieee_compliance(1))
                    u_sub (.inst_a(top_score_r[gk]), .inst_b(neg_max[gk]), .inst_rnd(3'b000),
                           .z_inst(x16[gk]), .status_inst());
                // scaled = 1.5*x = RNE_fp16( x + x/2 )  (x/2 = exponent decrement; multiplier-free).
                // MUST be the fp16 RNE add, NOT a wide-exact x+x>>1 (that differs at fp16-rounded
                // integer boundaries — consultant audit 18/50880; see CONSULT_SOFTMAX_...2026-06-09).
                sm_fp16_halve u_half (.i_h(x16[gk]), .o_h(xh16[gk]));
                DW_fp_add_inst #(.sig_width(ATTN_FP_FRAC_WIDTH), .exp_width(ATTN_FP_EXPN_WIDTH), .ieee_compliance(1))
                    u_mul15 (.inst_a(x16[gk]), .inst_b(xh16[gk]), .inst_rnd(3'b000),
                             .z_inst(scaled16[gk]), .status_inst());
                // k = floor( clamp(scaled, -8, 0) )  in [-8,0]
                sm_fp16_floor_clamp8 u_floor (.i_h(scaled16[gk]), .o_k(k_i[gk]));
                sm_weight_exp_from_k  u_wexp  (.i_k(k_i[gk]),     .o_e(weight_exp[gk]));
            end

            sm_weight_from_exp #(
                .WEIGHT_WIDTH (WEIGHT_WIDTH)
            ) u_wval (
                .i_e (weight_exp[gk]),
                .o_w (weight_raw[gk])
            );

            assign weight_q[gk] = top_valid_r[gk] ? weight_raw[gk] : WEIGHT_ZERO;
        end
    endgenerate

    always_comb begin
        denom_q = DENOM_ZERO;
        for (integer kk = 0; kk < TOP_K; kk = kk + 1) begin
            denom_q = denom_q + weight_q[kk];
        end
    end

    sm_round_log2_pow2 #(
        .DENOM_WIDTH (DENOM_WIDTH)
    ) u_denom_exp (
        .i_value (denom_q),
        .o_e     (denom_exp)
    );

    //--------------------------------------------------------------------------
    // Sparse probability: p = w >> round(log2(denom)), encoded back to FP16
    //--------------------------------------------------------------------------
    generate
        for (gk = 0; gk < TOP_K; gk = gk + 1) begin : g_prob
            assign prob_exp_q8[gk] = $signed({2'b00, weight_exp[gk]}) +
                                     6'sd8 -
                                     $signed({2'b00, denom_exp});

            sm_prob_exp_to_fp16 #(
                .PROB_FRAC (PROB_FRAC)
            ) u_prob (
                .i_q  (prob_exp_q8[gk]),
                .o_fp (prob_fp[gk])
            );
        end
    endgenerate

    //--------------------------------------------------------------------------
    // Control FSM
    //--------------------------------------------------------------------------
    always_comb begin
        state_w     = state_r;
        score_w     = score_r;
        mask_w      = mask_r;
        prob_w      = prob_r;
        top_score_w = top_score_r;
        top_idx_w   = top_idx_r;
        top_valid_w = top_valid_r;
        grp_idx_w   = grp_idx_r;

        case (state_r)
            SM_IDLE: begin
                grp_idx_w   = GRP_ZERO;
                for (integer k = 0; k < TOP_K; k = k + 1) begin
                    top_score_w[k] = FP_ZERO;
                    top_idx_w[k]   = SEQ_IDX_ZERO;
                    top_valid_w[k] = 1'b0;
                end
                for (integer i = 0; i < SEQ_LEN; i = i + 1) begin
                    prob_w[i] = FP_ZERO;
                end

                if (accept) begin
                    score_w = i_SM_data;
                    mask_w  = i_SM_mask;
                    state_w = SM_SCAN;
                end
            end

            SM_SCAN: begin
                top_score_w = scan_score;
                top_idx_w   = scan_idx;
                top_valid_w = scan_valid;

                if (grp_last) begin
                    grp_idx_w = GRP_ZERO;
                    state_w   = SM_CALC;
                end else begin
                    grp_idx_w = grp_idx_r + 1'b1;
                end
            end

            SM_CALC: begin
                for (integer i = 0; i < SEQ_LEN; i = i + 1) begin
                    prob_w[i] = FP_ZERO;
                end

                if (denom_q != DENOM_ZERO) begin
                    for (integer k = 0; k < TOP_K; k = k + 1) begin
                        if (top_valid_r[k]) begin
                            prob_w[top_idx_r[k]] = prob_fp[k];
                        end
                    end
                end

                state_w = SM_DONE;
            end

            SM_DONE: begin
                state_w = SM_IDLE;
            end

            default: begin
                state_w = SM_IDLE;
            end
        endcase
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state_r     <= SM_IDLE;
            mask_r      <= {SEQ_LEN{1'b0}};
            grp_idx_r   <= GRP_ZERO;
            for (integer i = 0; i < SEQ_LEN; i = i + 1) begin
                score_r[i] <= FP_ZERO;
                prob_r[i]  <= FP_ZERO;
            end
            for (integer k = 0; k < TOP_K; k = k + 1) begin
                top_score_r[k] <= FP_ZERO;
                top_idx_r[k]   <= SEQ_IDX_ZERO;
                top_valid_r[k] <= 1'b0;
            end
        end else begin
            state_r     <= state_w;
            score_r     <= score_w;
            mask_r      <= mask_w;
            prob_r      <= prob_w;
            top_score_r <= top_score_w;
            top_idx_r   <= top_idx_w;
            top_valid_r <= top_valid_w;
            grp_idx_r   <= grp_idx_w;
        end
    end

endmodule

//==============================================================================
// Combinational helper modules (replace the former SystemVerilog functions)
//==============================================================================

// FP16 sign-magnitude "greater than": o_gt = (i_a > i_b)
module sm_fp16_gt #(
    parameter FPW = 16
) (
    input  logic [FPW-1 : 0] i_a,
    input  logic [FPW-1 : 0] i_b,
    output logic             o_gt
);
    logic a_zero;
    logic b_zero;

    always_comb begin
        a_zero = (i_a[FPW-2 : 0] == '0);
        b_zero = (i_b[FPW-2 : 0] == '0);

        if (a_zero && b_zero) begin
            o_gt = 1'b0;
        end else if (i_a[FPW-1] != i_b[FPW-1]) begin
            o_gt = ~i_a[FPW-1];
        end else if (~i_a[FPW-1]) begin
            o_gt = (i_a[FPW-2 : 0] > i_b[FPW-2 : 0]);
        end else begin
            o_gt = (i_a[FPW-2 : 0] < i_b[FPW-2 : 0]);
        end
    end
endmodule

// Insert one candidate into a descending Top-K list (one scan lane)
module sm_topk_insert #(
    parameter TOP_K = 4,
    parameter FPW   = 16,
    parameter IDXW  = 5
) (
    input  logic [TOP_K-1 : 0] [FPW-1 : 0]  i_score,
    input  logic [TOP_K-1 : 0] [IDXW-1 : 0] i_idx,
    input  logic [TOP_K-1 : 0]              i_valid,
    input  logic [FPW-1 : 0]                i_cand_score,
    input  logic [IDXW-1 : 0]               i_cand_idx,
    input  logic                            i_cand_valid,
    output logic [TOP_K-1 : 0] [FPW-1 : 0]  o_score,
    output logic [TOP_K-1 : 0] [IDXW-1 : 0] o_idx,
    output logic [TOP_K-1 : 0]              o_valid
);
    logic [TOP_K-1 : 0] gt;

    genvar k;
    generate
        for (k = 0; k < TOP_K; k = k + 1) begin : g_cmp
            sm_fp16_gt #(.FPW(FPW)) u_gt (
                .i_a  (i_cand_score),
                .i_b  (i_score[k]),
                .o_gt (gt[k])
            );
        end
    endgenerate

    always_comb begin
        logic [2 : 0] insert_pos;

        o_score = i_score;
        o_idx   = i_idx;
        o_valid = i_valid;

        insert_pos = 3'd4;
        for (integer kk = 0; kk < TOP_K; kk = kk + 1) begin
            if (i_cand_valid &&
                (insert_pos == 3'd4) &&
                (!i_valid[kk] || gt[kk])) begin
                insert_pos = kk[2 : 0];
            end
        end

        if (insert_pos < 3'd4) begin
            for (integer kk = TOP_K - 1; kk > 0; kk = kk - 1) begin
                if (insert_pos < kk) begin
                    o_score[kk] = i_score[kk-1];
                    o_idx[kk]   = i_idx[kk-1];
                    o_valid[kk] = i_valid[kk-1];
                end
            end

            o_score[insert_pos] = i_cand_score;
            o_idx[insert_pos]   = i_cand_idx;
            o_valid[insert_pos] = 1'b1;
        end
    end
endmodule

// FP16 -> signed fixed-point (QFRAC fractional bits, QW total), saturating
module sm_fp16_to_sq4
    import Param_def::*;
#(
    parameter QW    = 16,
    parameter QFRAC = 4
) (
    input  logic [ATTN_FP_WIDTH-1 : 0] i_h,
    output logic signed [QW-1 : 0]     o_q
);
    localparam EXTW = QW + ATTN_FP_EXPN_WIDTH + 1;

    always_comb begin
        logic                                sign;
        logic [ATTN_FP_EXPN_WIDTH-1 : 0]     exp;
        logic [ATTN_FP_FRAC_WIDTH-1 : 0]     frac;
        logic [ATTN_FP_FRAC_WIDTH : 0]       mant;
        logic signed [7 : 0]                 shift_amt;
        logic signed [EXTW-1 : 0]            scaled;
        logic signed [EXTW-1 : 0]            max_val;
        logic signed [EXTW-1 : 0]            min_val;

        sign    = i_h[ATTN_FP_WIDTH-1];
        exp     = i_h[ATTN_FP_FRAC_WIDTH +: ATTN_FP_EXPN_WIDTH];
        frac    = i_h[ATTN_FP_FRAC_WIDTH-1 : 0];
        max_val = {{(EXTW-QW){1'b0}}, 1'b0, {(QW-1){1'b1}}};
        min_val = {{(EXTW-QW){1'b1}}, 1'b1, {(QW-1){1'b0}}};

        if (exp == 0) begin
            scaled = {EXTW{1'b0}};
        end else if (exp == ((1 << ATTN_FP_EXPN_WIDTH) - 1)) begin
            scaled = sign ? min_val : max_val;
        end else begin
            mant = (1 << ATTN_FP_FRAC_WIDTH) | frac;
            shift_amt = (exp - ((1 << (ATTN_FP_EXPN_WIDTH - 1)) - 1))
                        + QFRAC - ATTN_FP_FRAC_WIDTH;

            if (shift_amt >= 0) begin
                scaled = $signed({1'b0, mant}) <<< shift_amt;
            end else begin
                scaled = $signed({1'b0, mant}) >>> (-shift_amt);
            end

            if (sign) begin
                scaled = -scaled;
            end

            if (scaled > max_val) begin
                scaled = max_val;
            end else if (scaled < min_val) begin
                scaled = min_val;
            end
        end

        o_q = scaled[QW-1 : 0];
    end
endmodule

// floor(x) then clamp to [-8, 0]; input is Q(QFRAC) fixed-point
module sm_floor_clamp8 #(
    parameter QW    = 16,
    parameter QFRAC = 4
) (
    input  logic signed [QW : 0]  i_scaled,
    output logic signed [4 : 0]   o_k
);
    always_comb begin
        logic signed [QW : 0] k_ext;

        k_ext = i_scaled >>> QFRAC;

        if (k_ext < -8) begin
            k_ext = -8;
        end else if (k_ext > 0) begin
            k_ext = {(QW+1){1'b0}};
        end

        o_k = k_ext[4 : 0];
    end
endmodule

// fp16 halve (x/2 = fp16_scale2(x,-1)): exponent decrement, subnormal handling. Exact for normals.
module sm_fp16_halve
    import Param_def::*;
(
    input  logic [ATTN_FP_WIDTH-1 : 0] i_h,
    output logic [ATTN_FP_WIDTH-1 : 0] o_h
);
    localparam int EW = ATTN_FP_EXPN_WIDTH;   // 5
    localparam int FW = ATTN_FP_FRAC_WIDTH;   // 10
    always_comb begin
        logic                  s;
        logic [EW-1 : 0]       e;
        logic [FW-1 : 0]       f;
        s = i_h[ATTN_FP_WIDTH-1];
        e = i_h[FW +: EW];
        f = i_h[FW-1 : 0];
        if (e == {EW{1'b0}})        o_h = {s, {EW{1'b0}}, 1'b0, f[FW-1 : 1]};  // subnormal -> >>1
        else if (e == {{(EW-1){1'b0}}, 1'b1})
                                    o_h = {s, {EW{1'b0}}, 1'b1, f[FW-1 : 1]};  // smallest normal -> subnormal 0.1f
        else                        o_h = {s, (e - 1'b1), f};                   // exact: exponent-1
    end
endmodule

// floor(i_h) clamped to [-8,0], i_h = the fp16 1.5x value (<=0). floor toward -inf:
//  zero -> 0 ; tiny-negative (e==0,f!=0) -> -1 ; |m|<1 -> -1 ; |m|>=8 -> -8 ;
//  else (m in [1,8)) -> -(intpart + has_frac) = -ceil(|m|).
module sm_fp16_floor_clamp8
    import Param_def::*;
(
    input  logic [ATTN_FP_WIDTH-1 : 0] i_h,
    output logic signed [4 : 0]        o_k
);
    localparam int EW   = ATTN_FP_EXPN_WIDTH;            // 5
    localparam int FW   = ATTN_FP_FRAC_WIDTH;            // 10
    localparam int BIAS = (1 << (EW-1)) - 1;             // 15
    always_comb begin
        logic                   s;
        logic [EW-1 : 0]        e;
        logic [FW-1 : 0]        f;
        logic signed [7 : 0]    E;
        logic [FW : 0]          mant;       // 11 bits {1,f}
        logic [4 : 0]           sh;
        logic [FW : 0]          intpart;
        logic                   has_frac;
        logic signed [7 : 0]    kk;
        s    = i_h[ATTN_FP_WIDTH-1];
        e    = i_h[FW +: EW];
        f    = i_h[FW-1 : 0];
        mant = {1'b1, f};
        if (e == {EW{1'b0}})        kk = (s && (f != {FW{1'b0}})) ? -8'sd1 : 8'sd0; // tiny neg ->-1; 0 ->0
        else if (!s)                kk = 8'sd0;                                     // positive -> clamp 0
        else begin
            E = $signed({2'b00, e}) - 8'sd15;
            if (E < 0)              kk = -8'sd1;                                    // |m|<1 -> floor(-m)=-1
            else if (E >= 8'sd3)    kk = -8'sd8;                                    // |m|>=8 -> clamp -8
            else begin
                sh       = 5'd10 - E[4:0];                                          // 10-E, E in [0,2] -> [8,10]
                intpart  = mant >> sh;
                has_frac = |(mant & ((11'd1 << sh) - 11'd1));
                kk       = -($signed({4'b0, intpart[3:0]}) + (has_frac ? 8'sd1 : 8'sd0));
            end
        end
        o_k = kk[4:0];
    end
endmodule

// pow2 exponent select: k in [-8,0] -> 2^k weight exponent in [0,8]
module sm_weight_exp_from_k (
    input  logic signed [4 : 0] i_k,
    output logic [3 : 0]        o_e
);
    always_comb begin
        case (i_k)
            -5'sd8: o_e = 4'd0;
            -5'sd7: o_e = 4'd1;
            -5'sd6: o_e = 4'd2;
            -5'sd5: o_e = 4'd3;
            -5'sd4: o_e = 4'd4;
            -5'sd3: o_e = 4'd5;
            -5'sd2: o_e = 4'd6;
            -5'sd1: o_e = 4'd7;
             5'sd0: o_e = 4'd8;
            default: o_e = 4'd0;
        endcase
    end
endmodule

// Q0.8 weight from pow2 exponent: o_w = 1 << i_e  (i_e in [0,8])
module sm_weight_from_exp #(
    parameter WEIGHT_WIDTH = 9
) (
    input  logic [3 : 0]              i_e,
    output logic [WEIGHT_WIDTH-1 : 0] o_w
);
    always_comb begin
        case (i_e)
            4'd0: o_w = 9'd1;
            4'd1: o_w = 9'd2;
            4'd2: o_w = 9'd4;
            4'd3: o_w = 9'd8;
            4'd4: o_w = 9'd16;
            4'd5: o_w = 9'd32;
            4'd6: o_w = 9'd64;
            4'd7: o_w = 9'd128;
            4'd8: o_w = 9'd256;
            default: o_w = {WEIGHT_WIDTH{1'b0}};
        endcase
    end
endmodule

// round(log2(value)) -> nearest power-of-two exponent, in the LOG domain (deploy contract:
// 2^round(log2(sum))): round UP iff value >= 2^(msb+0.5) = 2^msb*sqrt2. The previous
// "next bit below the leading one" rule was the ARITHMETIC midpoint (1.5*2^msb) and diverges
// from the deploy for mantissa in [sqrt2, 1.5) — exposed by the t*=10 warm-up golden
// (h9 wsum q8=368: 368 >= 363 -> denom 512; the next-bit rule gave 256). Thresholds =
// ceil(2^(msb+0.5)) per msb; value is an integer (sum of Q0.8 pow2 weights) so never a tie.
module sm_round_log2_pow2 #(
    parameter DENOM_WIDTH = 11
) (
    input  logic [DENOM_WIDTH-1 : 0] i_value,
    output logic [3 : 0]             o_e
);
    always_comb begin
        logic [3 : 0]             msb;
        logic [3 : 0]             exp_i;
        logic [DENOM_WIDTH-1 : 0] thr;

        msb = 4'd0;
        for (integer i = 0; i < DENOM_WIDTH; i = i + 1) begin
            if (i_value[i]) begin
                msb = i[3 : 0];
            end
        end

        unique case (msb)                        // ceil(2^msb * sqrt2)
            4'd0:    thr = DENOM_WIDTH'(2);
            4'd1:    thr = DENOM_WIDTH'(3);
            4'd2:    thr = DENOM_WIDTH'(6);
            4'd3:    thr = DENOM_WIDTH'(12);
            4'd4:    thr = DENOM_WIDTH'(23);
            4'd5:    thr = DENOM_WIDTH'(46);
            4'd6:    thr = DENOM_WIDTH'(91);
            4'd7:    thr = DENOM_WIDTH'(182);
            4'd8:    thr = DENOM_WIDTH'(363);
            4'd9:    thr = DENOM_WIDTH'(725);
            default: thr = {DENOM_WIDTH{1'b1}};  // msb>=10: sum<=1024 here -> never rounds up
        endcase

        exp_i = msb;
        if ((i_value != {DENOM_WIDTH{1'b0}}) && (i_value >= thr)) begin
            exp_i = msb + 4'd1;
        end

        o_e = exp_i;
    end
endmodule

// Encode a (>= 0) pow2 probability exponent back into FP16; negative -> 0
module sm_prob_exp_to_fp16
    import Param_def::*;
#(
    parameter PROB_FRAC = 8
) (
    input  logic signed [5 : 0]        i_q,
    output logic [ATTN_FP_WIDTH-1 : 0] o_fp
);
    localparam logic [ATTN_FP_EXPN_WIDTH-1 : 0] Q8_EXP_OFFSET =
        (((1 << (ATTN_FP_EXPN_WIDTH - 1)) - 1) - PROB_FRAC);

    always_comb begin
        logic [ATTN_FP_EXPN_WIDTH-1 : 0] exp_i;

        if (i_q < 0) begin
            o_fp = {ATTN_FP_WIDTH{1'b0}};
        end else begin
            case (i_q)
                6'sd0: exp_i = Q8_EXP_OFFSET + 5'd0;
                6'sd1: exp_i = Q8_EXP_OFFSET + 5'd1;
                6'sd2: exp_i = Q8_EXP_OFFSET + 5'd2;
                6'sd3: exp_i = Q8_EXP_OFFSET + 5'd3;
                6'sd4: exp_i = Q8_EXP_OFFSET + 5'd4;
                6'sd5: exp_i = Q8_EXP_OFFSET + 5'd5;
                6'sd6: exp_i = Q8_EXP_OFFSET + 5'd6;
                6'sd7: exp_i = Q8_EXP_OFFSET + 5'd7;
                default: exp_i = Q8_EXP_OFFSET + 5'd8;
            endcase

            o_fp = {1'b0, exp_i, {ATTN_FP_FRAC_WIDTH{1'b0}}};
        end
    end
endmodule
