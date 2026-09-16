/*
    Module: Softmax function, streaming-row architecture

    This v2 implementation is intentionally not derived from the legacy
    ln(sum)+second-exp pipeline. It computes one attention row with three
    iterative passes over SEQ_LEN entries:

        1. max(score)
        2. exp(score - max), sum(exp), and store exp in a work buffer
        3. reciprocal(sum), then exp * reciprocal(sum)

    The datapath processes NUM_PEG entries per cycle. With the current
    configuration SEQ_LEN=32 and NUM_PEG=8, each row has four chunks.
*/

module Softmax
    import Param_def::*;
(
    input   logic                                                   i_clk,
    input   logic                                                   i_rst_n,
    input   logic                                                   i_SM_valid,
    input   logic   [SEQ_LEN-1 : 0] [ATTN_FP_WIDTH-1 : 0]           i_SM_data,
    output  logic                                                   o_SM_ready,
    output  logic                                                   o_SM_valid,
    output  logic   [SEQ_LEN-1 : 0] [ATTN_FP_WIDTH-1 : 0]           o_SM_data
);
    localparam int N_GRP = (SEQ_LEN + NUM_PEG - 1) / NUM_PEG;
    localparam int GRP_BIT = (N_GRP <= 1) ? 1 : $clog2(N_GRP);
    localparam int REDUCE_STAGES = $clog2(NUM_PEG);

    localparam logic [GRP_BIT-1 : 0] LAST_GRP = N_GRP - 1;
    localparam logic [ATTN_FP_EXPN_WIDTH-1 : 0] FP_BIAS = (1 << (ATTN_FP_EXPN_WIDTH - 1)) - 1;
    localparam logic [ATTN_FP_WIDTH-1 : 0] FP_ONE = {1'b0, FP_BIAS, {ATTN_FP_FRAC_WIDTH{1'b0}}};
    localparam logic [ATTN_FP_WIDTH-1 : 0] ATTN_FP_ZERO = {ATTN_FP_WIDTH{1'b0}};
    localparam logic [GRP_BIT-1 : 0] GRP_ZERO = {GRP_BIT{1'b0}};

    typedef enum logic [2 : 0] {
        SM_IDLE,
        SM_MAX,
        SM_EXP_SUM,
        SM_RECIP,
        SM_NORM,
        SM_DONE
    } SM_state;

    SM_state state_w, state_r;

    logic [SEQ_LEN-1 : 0] [ATTN_FP_WIDTH-1 : 0] score_w, score_r;
    logic [SEQ_LEN-1 : 0] [ATTN_FP_WIDTH-1 : 0] exp_w, exp_r;
    logic [SEQ_LEN-1 : 0] [ATTN_FP_WIDTH-1 : 0] prob_w, prob_r;

    logic [GRP_BIT-1 : 0]       grp_idx_w, grp_idx_r;
    logic [ATTN_FP_WIDTH-1 : 0] current_max_w, current_max_r;
    logic [ATTN_FP_WIDTH-1 : 0] partial_sum_w, partial_sum_r;
    logic [ATTN_FP_WIDTH-1 : 0] sum_exp_w, sum_exp_r;
    logic [ATTN_FP_WIDTH-1 : 0] inv_sum_w, inv_sum_r;

    logic                       accept;
    logic [SEQ_LEN_BIT : 0] grp_base;
    logic                       grp_last;

    logic [NUM_PEG-1 : 0] [SEQ_LEN_BIT : 0] lane_idx;
    logic [NUM_PEG-1 : 0]                         lane_valid;
    logic [NUM_PEG-1 : 0] [ATTN_FP_WIDTH-1 : 0]   lane_score;

    logic [NUM_PEG-1 : 0] [ATTN_FP_WIDTH-1 : 0] exp_sub_in;
    logic [NUM_PEG-1 : 0] [ATTN_FP_WIDTH-1 : 0] exp_val;
    logic [NUM_PEG-1 : 0] [ATTN_FP_WIDTH-1 : 0] norm_exp_in;
    logic [NUM_PEG-1 : 0] [ATTN_FP_WIDTH-1 : 0] norm_val;

    logic [REDUCE_STAGES : 0] [NUM_PEG-1 : 0] [ATTN_FP_WIDTH-1 : 0] max_tree;
    logic [REDUCE_STAGES : 0] [NUM_PEG-1 : 0] [ATTN_FP_WIDTH-1 : 0] sum_tree;
    logic [ATTN_FP_WIDTH-1 : 0] group_max;
    logic [ATTN_FP_WIDTH-1 : 0] max_next;
    logic                       current_gt_group;
    logic [ATTN_FP_WIDTH-1 : 0] group_sum;
    logic [ATTN_FP_WIDTH-1 : 0] sum_next;
    logic [ATTN_FP_WIDTH-1 : 0] recip_val;

    assign accept = i_SM_valid && o_SM_ready;
    assign o_SM_ready = (state_r == SM_IDLE);
    assign o_SM_valid = (state_r == SM_DONE);
    assign o_SM_data  = prob_r;

    assign grp_base = grp_idx_r * NUM_PEG;
    assign grp_last = (grp_idx_r == LAST_GRP);

    always_comb begin
        for (int i = 0; i < NUM_PEG; i++) begin
            lane_idx[i] = grp_base + i;
            lane_valid[i] = (lane_idx[i] < SEQ_LEN);

            lane_score[i] = ATTN_FP_ZERO;
            if (lane_valid[i]) begin
                lane_score[i] = score_r[lane_idx[i][SEQ_LEN_BIT-1 : 0]];
            end
        end
    end

    always_comb begin
        for (int i = 0; i < NUM_PEG; i++) begin
            max_tree[0][i] = lane_valid[i] ? lane_score[i] : current_max_r;
        end
    end

    generate
        for (genvar stage = 0; stage < REDUCE_STAGES; stage++) begin : gen_max_stage
            for (genvar node = 0; node < (NUM_PEG >> (stage + 1)); node++) begin : gen_max_node
                logic a_gt_b;

                DW_fp_cmp_inst #(
                    .sig_width       (ATTN_FP_FRAC_WIDTH),
                    .exp_width       (ATTN_FP_EXPN_WIDTH),
                    .ieee_compliance (1)
                ) u_cmp (
                    .inst_a         (max_tree[stage][2*node]),
                    .inst_b         (max_tree[stage][2*node + 1]),
                    .inst_zctr      (1'b0),
                    .aeqb_inst      (),
                    .altb_inst      (),
                    .agtb_inst      (a_gt_b),
                    .unordered_inst (),
                    .z0_inst        (),
                    .z1_inst        (),
                    .status0_inst   (),
                    .status1_inst   ()
                );

                assign max_tree[stage + 1][node] = a_gt_b ?
                                                   max_tree[stage][2*node] :
                                                   max_tree[stage][2*node + 1];
            end
        end
    endgenerate

    assign group_max = max_tree[REDUCE_STAGES][0];

    DW_fp_cmp_inst #(
        .sig_width       (ATTN_FP_FRAC_WIDTH),
        .exp_width       (ATTN_FP_EXPN_WIDTH),
        .ieee_compliance (1)
    ) u_cmp_running_max (
        .inst_a         (current_max_r),
        .inst_b         (group_max),
        .inst_zctr      (1'b0),
        .aeqb_inst      (),
        .altb_inst      (),
        .agtb_inst      (current_gt_group),
        .unordered_inst (),
        .z0_inst        (),
        .z1_inst        (),
        .status0_inst   (),
        .status1_inst   ()
    );

    assign max_next = current_gt_group ? current_max_r : group_max;

    generate
        for (genvar lane = 0; lane < NUM_PEG; lane++) begin : gen_exp_lane
            DW_fp_sub_inst #(
                .sig_width       (ATTN_FP_FRAC_WIDTH),
                .exp_width       (ATTN_FP_EXPN_WIDTH),
                .ieee_compliance (1)
            ) u_sub (
                .inst_a      (lane_score[lane]),
                .inst_b      (current_max_r),
                .inst_rnd    (3'b000),
                .z_inst      (exp_sub_in[lane]),
                .status_inst ()
            );

            DW_fp_exp_inst #(
                .inst_sig_width       (ATTN_FP_FRAC_WIDTH),
                .inst_exp_width       (ATTN_FP_EXPN_WIDTH),
                .inst_ieee_compliance (1)
            ) u_exp (
                .inst_a      (exp_sub_in[lane]),
                .z_inst      (exp_val[lane]),
                .status_inst ()
            );
        end
    endgenerate

    always_comb begin
        for (int i = 0; i < NUM_PEG; i++) begin
            sum_tree[0][i] = lane_valid[i] ? exp_val[i] : ATTN_FP_ZERO;
        end
    end

    generate
        for (genvar stage = 0; stage < REDUCE_STAGES; stage++) begin : gen_sum_stage
            for (genvar node = 0; node < (NUM_PEG >> (stage + 1)); node++) begin : gen_sum_node
                DW_fp_add_inst #(
                    .sig_width       (ATTN_FP_FRAC_WIDTH),
                    .exp_width       (ATTN_FP_EXPN_WIDTH),
                    .ieee_compliance (1)
                ) u_add (
                    .inst_a      (sum_tree[stage][2*node]),
                    .inst_b      (sum_tree[stage][2*node + 1]),
                    .inst_rnd    (3'b000),
                    .z_inst      (sum_tree[stage + 1][node]),
                    .status_inst ()
                );
            end
        end
    endgenerate

    assign group_sum = sum_tree[REDUCE_STAGES][0];

    DW_fp_add_inst #(
        .sig_width       (ATTN_FP_FRAC_WIDTH),
        .exp_width       (ATTN_FP_EXPN_WIDTH),
        .ieee_compliance (1)
    ) u_sum_acc (
        .inst_a      (partial_sum_r),
        .inst_b      (group_sum),
        .inst_rnd    (3'b000),
        .z_inst      (sum_next),
        .status_inst ()
    );

    DW_fp_div_inst #(
        .sig_width       (ATTN_FP_FRAC_WIDTH),
        .exp_width       (ATTN_FP_EXPN_WIDTH),
        .ieee_compliance (1)
    ) u_recip (
        .inst_a      (FP_ONE),
        .inst_b      (sum_exp_r),
        .inst_rnd    (3'b000),
        .z_inst      (recip_val),
        .status_inst ()
    );

    always_comb begin
        for (int i = 0; i < NUM_PEG; i++) begin
            norm_exp_in[i] = ATTN_FP_ZERO;
            if (lane_valid[i]) begin
                norm_exp_in[i] = exp_r[lane_idx[i][SEQ_LEN_BIT-1 : 0]];
            end
        end
    end

    generate
        for (genvar lane = 0; lane < NUM_PEG; lane++) begin : gen_norm_lane
            DW_fp_mult_inst #(
                .sig_width       (ATTN_FP_FRAC_WIDTH),
                .exp_width       (ATTN_FP_EXPN_WIDTH),
                .ieee_compliance (1)
            ) u_mult (
                .inst_a      (norm_exp_in[lane]),
                .inst_b      (inv_sum_r),
                .inst_rnd    (3'b000),
                .z_inst      (norm_val[lane]),
                .status_inst ()
            );
        end
    endgenerate

    always_comb begin
        state_w       = state_r;
        score_w       = score_r;
        exp_w         = exp_r;
        prob_w        = prob_r;
        grp_idx_w     = grp_idx_r;
        current_max_w = current_max_r;
        partial_sum_w = partial_sum_r;
        sum_exp_w     = sum_exp_r;
        inv_sum_w     = inv_sum_r;

        case (state_r)
            SM_IDLE: begin
                grp_idx_w     = GRP_ZERO;
                partial_sum_w = ATTN_FP_ZERO;
                sum_exp_w     = ATTN_FP_ZERO;
                inv_sum_w     = ATTN_FP_ZERO;

                if (accept) begin
                    score_w = i_SM_data;
                    for (int i = 0; i < SEQ_LEN; i++) begin
                        prob_w[i] = ATTN_FP_ZERO;
                        exp_w[i]  = ATTN_FP_ZERO;
                    end
                    current_max_w = i_SM_data[0];
                    state_w       = SM_MAX;
                end
            end

            SM_MAX: begin
                current_max_w = max_next;

                if (grp_last) begin
                    grp_idx_w     = GRP_ZERO;
                    partial_sum_w = ATTN_FP_ZERO;
                    state_w       = SM_EXP_SUM;
                end else begin
                    grp_idx_w = grp_idx_r + 1'b1;
                end
            end

            SM_EXP_SUM: begin
                partial_sum_w = sum_next;

                for (int i = 0; i < NUM_PEG; i++) begin
                    if (lane_valid[i]) begin
                        exp_w[lane_idx[i][SEQ_LEN_BIT-1 : 0]] = exp_val[i];
                    end
                end

                if (grp_last) begin
                    sum_exp_w = sum_next;
                    grp_idx_w = GRP_ZERO;
                    state_w   = SM_RECIP;
                end else begin
                    grp_idx_w = grp_idx_r + 1'b1;
                end
            end

            SM_RECIP: begin
                inv_sum_w = recip_val;
                grp_idx_w = GRP_ZERO;
                state_w   = SM_NORM;
            end

            SM_NORM: begin
                for (int i = 0; i < NUM_PEG; i++) begin
                    if (lane_valid[i]) begin
                        prob_w[lane_idx[i][SEQ_LEN_BIT-1 : 0]] = norm_val[i];
                    end
                end

                if (grp_last) begin
                    grp_idx_w = GRP_ZERO;
                    state_w   = SM_DONE;
                end else begin
                    grp_idx_w = grp_idx_r + 1'b1;
                end
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
            state_r       <= SM_IDLE;
            grp_idx_r     <= GRP_ZERO;
            current_max_r <= ATTN_FP_ZERO;
            partial_sum_r <= ATTN_FP_ZERO;
            sum_exp_r     <= ATTN_FP_ZERO;
            inv_sum_r     <= ATTN_FP_ZERO;
        end else begin
            state_r       <= state_w;
            score_r       <= score_w;
            exp_r         <= exp_w;
            prob_r        <= prob_w;
            grp_idx_r     <= grp_idx_w;
            current_max_r <= current_max_w;
            partial_sum_r <= partial_sum_w;
            sum_exp_r     <= sum_exp_w;
            inv_sum_r     <= inv_sum_w;
        end
    end

endmodule
