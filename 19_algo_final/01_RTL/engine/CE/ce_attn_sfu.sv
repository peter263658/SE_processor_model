/*
    ce_attn_sfu.sv — attention softmax + top-4 packet glue (between QK and Score×V).

    Takes the QK postscale scores for all NUM_HEAD heads (32-frame window each, collected from the QK
    sub-op's o_re stream), runs each head through the verified Softmax_v3 (shift_topk_pow2 fp16-delta),
    and extracts the top-4 packet {local_idx, prob (fp16)} per head (the ≤4 nonzero dense entries; the
    HW Score×V uses the dense fp16 prob directly = fp16(code/256), so no q0.8 re-encode needed). One
    head at a time through the single Softmax_v3 (7 cy/head); fully scoreboarded by the FSM.

    score_buf is written externally during QK (i_wr_*); o_done pulses when all heads are packetized.
    i_valid_mask (T3 warm-up, shared by all heads) gates the window entries: bit d = local_idx d is a
    REAL key (causal triu + 31-window; query frame i has min(i+1,32) valid keys — frames the KV ring
    has actually been written with). Masked lanes are excluded VALUE-INDEPENDENTLY in Softmax_v3
    (lane_valid gates insertion; empty top-k slots weigh 0 so the pow2 denominator sums only the
    survivors) — with <4 valid keys the packet degenerates to min(4, n_valid) live slots (prob=0
    empties; Score×V multiplies them by 0 harmlessly). Steady state: all-ones.
*/
`timescale 1ns/10ps
module ce_attn_sfu
    import Param_def::*;
(
    input  logic i_clk,
    input  logic i_rst_n,

    // score-collect write (from QK o_re stream): score_buf[head][frame] = postscale score
    input  logic                              i_wr_en,
    input  logic [$clog2(NUM_HEAD)-1:0]       i_wr_head,
    input  logic [$clog2(SEQ_LEN)-1:0]        i_wr_frame,
    input  logic [ATTN_FP_WIDTH-1:0]          i_wr_score,

    input  logic                              i_start,        // begin softmax over all heads
    output logic                              o_done,

    // warm-up validity mask (bit d = window local_idx d holds a real key); steady state = all-ones
    input  logic [SEQ_LEN-1:0]                i_valid_mask,

    // top-4 packet per head (idx-ascending; ≤4 valid, prob=0 for empty slots)
    output logic [NUM_HEAD-1:0][3:0][$clog2(SEQ_LEN)-1:0] o_pidx,
    output logic [NUM_HEAD-1:0][3:0][ATTN_FP_WIDTH-1:0]   o_pprob
);
    localparam int HB = $clog2(NUM_HEAD);
    logic [HB:0] head_r;                 // current head (0..NUM_HEAD); declared early for sm_in

    logic [NUM_HEAD-1:0][SEQ_LEN-1:0][ATTN_FP_WIDTH-1:0] score_buf;
    always_ff @(posedge i_clk)
        if (i_wr_en) score_buf[i_wr_head][i_wr_frame] <= i_wr_score;

    // ---- Softmax_v3 ----
    logic sm_valid, sm_ready, sm_ovalid;
    logic [SEQ_LEN-1:0][ATTN_FP_WIDTH-1:0] sm_in, sm_out;
    Softmax u_sm (
        .i_clk(i_clk), .i_rst_n(i_rst_n),
        .i_SM_valid(sm_valid), .i_SM_data(sm_in), .i_SM_mask(i_valid_mask),
        .o_SM_ready(sm_ready), .o_SM_valid(sm_ovalid), .o_SM_data(sm_out)
    );
    assign sm_in = score_buf[head_r];

    // ---- top-4 extraction from the dense softmax (≤4 nonzero), idx-ascending ----
    // PACKET prob = Q0.8-encoded (code/256, prob_max_code=255): the dense 1.0 (w==denom, e.g. a
    // single dominant key at warm-up) SATURATES to 255/256 = fp16 0x3BF8 — the deploy Score×V
    // consumes code/256, verified vs ctx16 warm-up golden (lmul(0.996,V) != lmul(1.0,V)). All
    // other probs are pow2 < 1.0 -> the Q0.8 encode is the identity for them.
    logic [3:0][$clog2(SEQ_LEN)-1:0] x_idx;
    logic [3:0][ATTN_FP_WIDTH-1:0]   x_prob;
    always_comb begin
        int c; c=0; x_idx='0; x_prob='0;
        for (int f=0; f<SEQ_LEN; f++)
            if (sm_out[f] != '0 && c < 4) begin
                x_idx[c]  = f[$clog2(SEQ_LEN)-1:0];
                x_prob[c] = (sm_out[f] == 16'h3C00) ? 16'h3BF8 : sm_out[f];  // 1.0 -> 255/256
                c++;
            end
    end

    // ---- FSM: walk heads through Softmax_v3, capture packet ----
    typedef enum logic [1:0] {S_IDLE, S_FEED, S_WAIT, S_DONE} st_t;
    st_t state;
    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin state<=S_IDLE; head_r<='0; o_done<=1'b0; o_pidx<='0; o_pprob<='0; end
        else begin
            o_done <= 1'b0;
            unique case (state)
                S_IDLE:  if (i_start) begin head_r<='0; state<=S_FEED; end
                S_FEED:  if (sm_ready) state<=S_WAIT;                 // present sm_in (=score_buf[head]) + valid
                S_WAIT:  if (sm_ovalid) begin
                             o_pidx[head_r[HB-1:0]]  <= x_idx;
                             o_pprob[head_r[HB-1:0]] <= x_prob;
                             if (head_r == NUM_HEAD-1) state<=S_DONE;
                             else begin head_r<=head_r+1'b1; state<=S_FEED; end
                         end
                S_DONE:  begin o_done<=1'b1; state<=S_IDLE; end
            endcase
        end
    end
    assign sm_valid = (state == S_FEED) && sm_ready;

endmodule
