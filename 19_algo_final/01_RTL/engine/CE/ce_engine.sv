/*
    ce_engine.sv — CE W4A8 engine (hierarchy: inst_decoder + mem_controller + CE_core(=ce) as direct
    children + behavioral buffers + ce_kv_cache + ppu_const_cache + ce_attn_sfu/Softmax_v3).

    The former `ce_top` control wrapper was DISSOLVED into this engine (T1b 2026-06-10): inst_decoder
    (u_id), mem_controller (u_mc) and CE_core (u_ce) now sit directly at the engine level so they can
    interface the memory macros / top dataflow. Does NOT overwrite the legacy `engine.sv` (old Presto NPU).

    For the FUNCTIONAL / golden path the WBUF and UF are BEHAVIORAL byte memories (init via
    $readmemh from hw_patterns_final/sram_images .bin files); reads model the real path (word-aligned
    window from (addr & ~7) -> ce_operand_gather shifts the (addr&7) offset), registered 1 cycle
    to match the OACC_DLY pipeline. The banked presto SRAM macros + DMA-load are the synth step
    (Step 1a, deferred) — swap the behavioral arrays for wbuf/ufbuf_top later. KV/SKIP + the
    attention datapath (QK/Softmax_v3/Score x V) + the act write-back are TODO (conv path first).

    Memories are module-scope so a golden TB can: $readmemh(img, u_eng.wbuf_mem); preload UF acts.
*/
`timescale 1ns/10ps
module ce_engine
    import Param_def::*;
    import Mem_def::*;
    import ce_subop_pkg::*;
(
    input  logic i_clk,
    input  logic i_rst_n,
    input  logic i_start,
    output logic o_busy,
    output logic o_done,
    output logic [15:0] o_re, o_im,
    output logic o_result_valid,

    // ---- multi-frame ring advance (T3): the frame-loop owner (engine orchestrator / TB) pulses
    //      this ONCE per query frame AFTER the frame's attention completes (e.g. at o_done), at a
    //      frame boundary — NEVER between tied pass A and pass B (both passes share one window).
    //      Unconnected (single-frame TBs): floats z -> never advances (ptr/filled stay 0). ----
    input  logic i_advance,

    // ---- STFT spectrum store (dual-role SKIPBUF, T2 #8): one ear's 64-bin Q2.14 spectrum ----
    // Behavioral one-shot block write on i_spec_wr_en (the synth skipbuf macro is 4x64b dual-port
    // -> 16-row FSM per half at the macro swap). Read is combinational per (ear,bin) for the
    // mask_apply feed. Unconnected (legacy TBs): wr_en floats z -> no writes.
    input  logic              i_spec_wr_en,
    input  logic              i_spec_wr_ear,
    input  logic [63:0][15:0] i_spec_wr_re,    // Q2.14 int16 per bin
    input  logic [63:0][15:0] i_spec_wr_im,
    input  logic              i_spec_rd_ear,
    input  logic [5:0]        i_spec_rd_bin,
    output logic [15:0]       o_spec_re,       // Q2.14 (-> mask_apply i_spec_*)
    output logic [15:0]       o_spec_im
);
    // ===== behavioral buffer memories (byte-addressable; TB $readmemh / preload) =====
    localparam int WBUF_BYTES = 262144;   // 256 KiB image
    localparam int UFBUF_BYTES = 8192;    // 8 KiB
    localparam int SKIPBUF_BYTES = 4096;  // 4 KiB (Mem_def 4 banks x 64b x 128); synth swap = skipbuf_top
    logic [7:0] wbuf_mem [0:WBUF_BYTES-1];
    logic [7:0] uf_mem   [0:UFBUF_BYTES-1];
    // SKIPBUF (T2): re half @[0:2047], im half @[2048:4095]. Per ear/freq layout doc'd in
    // gen_ce_subop_rom.py SKIP_BASE: stage s skip block (compact, skip_indices order) @ base_s
    // (s0@0 s1@64 s2@192 s3@320 s4@448 s5@576, f=0); STFT spec region planned @[1024:..] (task #8).
    logic [7:0] skip_mem [0:SKIPBUF_BYTES-1];

    // ===== ce_top control+datapath =====
    logic [17:0] wr_addr, wi_addr, xr_addr, xi_addr;
    logic        act_en, wbuf_en, wbuf_is_prefetch;
    logic        w_nibble_odd;          // (oc*ic)&1: weight gather starts lane0 at HIGH nibble (odd-ic)
    logic        w_nibble_odd_r;        // 1-cy delayed to align with the registered weight window
    logic        w_is_fp8;              // 1 = FP8 weight (8b/wt); 0 = INT4 (nibble)
    logic        w_is_fp8_r;            // 1-cy delayed (align with registered weight window)
    logic [17:0] wbuf_addr;
    logic [NUM_PE_PG-1:0][7:0] wr_codes, wi_codes;   // INT4 in [3:0] ([7:4]=0) ; FP8 in [7:0]
    logic [NUM_PE_PG-1:0][7:0] xr_codes, xi_codes;
    logic [63:0] const_word;

    // inst_decoder outputs (ce_top dissolved into engine, T1b) — declared here as they are referenced
    // early (is_attn/a_head/PPU feed) before the inst_decoder instantiation further down.
    subop_t      id_subop;
    logic [2:0]  id_phase;
    logic [9:0]  id_oc_idx;
    logic [3:0]  id_ic_chunk;
    logic [2:0]  id_tap;
    CE_LANE_mode id_pe_mode;
    logic [1:0]  id_ppu_mode;
    logic [NUM_PEG-1:0][7:0] id_preshift;
    logic        id_oacc_clr, id_oacc_en, id_ppu_go, id_prefetch, id_cache_half;
    logic        id_oc_complete;
    logic        prefetch_done, drain_done;
    logic [6:0]  id_subop_idx;
    logic        id_pass_sel;     // tied_mha 2-pass select (-> mem_controller UF slice)
    logic [15:0] pre_re, pre_im;  // CE_core pre-cast fp16 (same pipe stage as o_re; skip dual-cast)

    // PPU constants. α/PReLU = identity (0xC00). β (bias) + cast_k are fed LIVE for mha_in_proj (sub-op 31)
    // from a per-oc bias buffer (TB-preload; image const region later), aligned to the oacc->PPU stage =
    // id_oc_idx delayed OACC_DLY+1 (=3). cast_k per qkv region (Q,K=+7, V=+6). Other sub-ops: β=0,cast_k=0.
    logic [11:0] bn_a_re, bn_a_im, prelu_re, prelu_im, scale_re, scale_im;
    logic [15:0] bn_b_re, bn_b_im;
    logic signed [7:0] cast_k;
    assign {bn_a_re,bn_a_im,prelu_re,prelu_im} = {4{12'hC00}};
    logic [15:0] bias_mem_op35_re [0:DIM_SIZE-1], bias_mem_op35_im [0:DIM_SIZE-1]; // output_proj(35) complex β (TB-preload @252896)

    localparam int PPU_STAGE_DLY = 3;                       // OACC_DLY(2)+1 : id_oc_idx -> oacc@PPU
    logic [15:0] bias_mem [0:767];                          // mha_in_proj bias (TB-preload; β-add)
    logic [9:0] ocp [0:PPU_STAGE_DLY-1];  logic [6:0] subp [0:PPU_STAGE_DLY-1];
    always_ff @(posedge i_clk) begin
        ocp[0] <= id_oc_idx;  subp[0] <= u_id.subop_idx;
        for (int k=1;k<PPU_STAGE_DLY;k++) begin ocp[k]<=ocp[k-1]; subp[k]<=subp[k-1]; end
    end
    wire [9:0] oc_ppu = ocp[PPU_STAGE_DLY-1];  wire [6:0] sub_ppu = subp[PPU_STAGE_DLY-1];
    // PPU β + cast_k feed: sub-op 31 (mha_in_proj, real β, cast +7/+6) and sub-op 35 (output_proj, COMPLEX
    // β @252896, cast -1). α/PReLU stay identity (per-head scale is in the dealer, not α). Other sub-ops: β=0.
    assign bn_b_re = (sub_ppu == 7'd31) ? bias_mem[oc_ppu]
                   : (sub_ppu == 7'd35) ? bias_mem_op35_re[oc_ppu[7:0]] : 16'h0;
    assign bn_b_im = (sub_ppu == 7'd35) ? bias_mem_op35_im[oc_ppu[7:0]] : 16'h0;
    assign cast_k  = (sub_ppu == 7'd31) ? 8'sd5                   // in_proj Q/K/V: shared block-float k=-5 (cast +5)
                   : (sub_ppu == 7'd35) ? -8'sd1 : 8'sd0;

    // ===== ATTENTION operand path (QK live, Step A) =====
    // QK sub-op ROM sets oc = NUM_HEAD*SEQ_LEN (=512), so the conv oc-loop flattens (head,frame):
    //   head = oc_idx[8:5], frame = oc_idx[4:0]. K comes from ce_kv_cache, Q from a per-head buffer
    //   (TB-preloaded now; mha_in_proj output later). Operands land on PEG group-0 even lanes (2*d).
    //   QK ×2^-2 (1/sqrt(16)) rides PPU_SCALE i_scale=0xB00 (apot6a a=-2,h2=0). Score×V (sel=V, frame=
    //   packet_idx) + multi-PEG parallelism + the score-collect→Softmax→packet glue are the next layer.
    localparam logic [11:0] SCALE_2M2 = 12'hB00;          // ×2^-2 = 1/sqrt(head_dim=16)
    localparam signed [7:0] K_DESCALE_K = -8'sd5;         // K block-float exp (in_proj cast_k=+5 -> read ×2^-5;
                                                          // pow2-invariant so QK score == cast0 golden)
    // is_attn = QK/Score×V sub-op; active for the WHOLE sub-op (RUN+DRAIN) so the last oc's operand
    // stays muxed while its result drains (oc_idx holds the last value in DRAIN -> operands unchanged).
    wire        is_attn  = (id_pe_mode == MODE_LMUL_ATTN);
    wire [3:0]  a_head   = id_oc_idx[8:5];
    wire [4:0]  a_frame  = id_oc_idx[4:0];

    assign scale_re = is_attn ? SCALE_2M2 : 12'hC00;      // QK descale via PPU_SCALE; else identity
    assign scale_im = is_attn ? SCALE_2M2 : 12'hC00;

    // K/V write-back signals (driven by the chaining handoff #2 gather logic below, after the inst_decoder/CE_core block so o_re
    // / oc_rv / pass_rv are available; declared here so the u_kv write port can bind them). The write
    // always targets the CURRENT frame -> ring slot = ptr (ce_kv_cache owns the pointer, reset 0); pass A
    // (real) writes K_re/V_re, pass B (imag) writes K_im/V_im (kv_wr_pass = pass_rv). The 4 separate paths
    // REPLACE the old single-region + per-pass TB injection (#6 hack). current query = local_idx 0 = ptr.
    localparam signed [7:0] K_VBF = -8'sd5;              // V block-float exp (read ×2^-5; consultant shared k=-5)
    logic                          kv_wr_en, kv_wr_sel, kv_wr_pass;
    logic [$clog2(NUM_HEAD)-1:0]   kv_wr_head;
    logic [HEAD_SIZE-1:0][15:0]    kv_wr_vec;            // [d][7:0] = e4m3 code (code-mode write)

    // forward decls (driven after the inst_decoder/CE_core block): ce_attn_sfu top-4 packet + the Score×V V-gather FSM. After
    // sfu_done the gather reads KVBUF V (sel=1, local_idx=packet idx, ×2^-5) into vbuf_all; score_v(33) RUN is
    // stalled (ce_top i_attn_sv_ready = gather_done) until the gather finishes.
    logic [NUM_HEAD-1:0][3:0][$clog2(SEQ_LEN)-1:0] sfu_pidx;
    logic [NUM_HEAD-1:0][3:0][ATTN_FP_WIDTH-1:0]   sfu_pprob;
    logic                          sfu_done, gather_rd, gather_done;
    logic                          recombine_done;       // tied_mha #6/#7: recombine codes written to UF -> gate output_proj(35)
    logic [$clog2(NUM_HEAD)-1:0]   gh;
    logic [$clog2(SEQ_LEN)-1:0]    gf;

    // KVBUF read mux: QK reads K (pass=pass_sel, sel=0, head=a_head, local_idx=a_frame, ×2^-5); the
    // V-gather reads V (sel=1, head=gh, local_idx=gf, ×2^-5). The cache translates local_idx -> physical
    // ring slot (ptr-local_idx) mod 32, and the write -> slot ptr. i_advance is DORMANT here (one query
    // frame; tied pass A & B share the same window) — top-level multi-frame (#6) pulses it after pass B's
    // score_v. Both passes' K/V live in separate paths (K_re/K_im/V_re/V_im) so there is no per-pass swap.
    logic [HEAD_SIZE-1:0][15:0] kv_rd_fp16;  logic kv_rd_valid;
    logic [$clog2(SEQ_LEN)-1:0] kv_ptr;
    logic [5:0]                 kv_filled;
    ce_kv_cache u_kv (
        .i_clk(i_clk), .i_rst_n(i_rst_n),
        .i_advance(i_advance), .o_ptr(kv_ptr), .o_filled(kv_filled),
        .i_wr_en(kv_wr_en), .i_wr_code_mode(1'b1), .i_wr_pass(kv_wr_pass), .i_wr_sel(kv_wr_sel),
        .i_wr_head(kv_wr_head), .i_wr_cast_k('0), .i_wr_fp16(kv_wr_vec),
        .i_rd_en(gather_rd | is_attn), .i_rd_pass(id_pass_sel), .i_rd_sel(gather_rd ? 1'b1 : 1'b0),
        .i_rd_head(gather_rd ? gh : a_head), .i_rd_local_idx(gather_rd ? gf : a_frame),
        .i_rd_descale_k(gather_rd ? K_VBF : K_DESCALE_K), .o_rd_fp16(kv_rd_fp16), .o_rd_valid(kv_rd_valid)
    );

    // QK vs Score×V distinguished by PPU mode (both MODE_LMUL_ATTN): QK=PPU_SCALE, Score×V=PPU_PASS.
    wire is_qk = is_attn && (id_ppu_mode == PPU_SCALE);
    wire is_sv = is_attn && (id_ppu_mode == PPU_PASS);

    // Q buffer: current frame's Q per head [NUM_HEAD][HEAD_SIZE] fp16 (TB preloads; mha_in_proj later).
    logic [15:0]                q_mem [0:NUM_HEAD*HEAD_SIZE-1];
    logic [HEAD_SIZE-1:0][15:0] q_rd;                     // registered 1-cyc -> aligns with ce_kv_cache
    // Score×V buffers: per head the 4 top-4-packet V-frames [head][slot][dim] + the prob (= dense fp16).
    // TB-preloads now (= V[head][packet_idx[s]] + packet prob); the live gather-from-KVBUF + ce_attn_sfu
    // packet source is the next step. Score×V oc = head*16 + d (head=oc[7:4], d=oc[3:0]).
    logic [15:0] vbuf_all   [0:NUM_HEAD*4*HEAD_SIZE-1];   // [head*4+slot]*16 + dim (live: V-gather fills it)
    logic [9:0]  oc_r;  logic attn_active_r, is_sv_r;
    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin attn_active_r<=1'b0; is_sv_r<=1'b0; oc_r<='0; for (int d=0; d<HEAD_SIZE; d++) q_rd[d]<=16'h0; end
        else begin
            attn_active_r <= is_attn;  is_sv_r <= is_sv;  oc_r <= id_oc_idx;
            for (int d = 0; d < HEAD_SIZE; d++) q_rd[d] <= q_mem[a_head*HEAD_SIZE + d];  // QK Q (aligned)
        end
    end

    // assemble PEG group-0 even lanes: QK = 16 products Q[d]·K[d] ; Score×V = 4 products prob[s]·V[idx[s]][d].
    logic [NUM_PEG-1:0][NUM_PE_PG-1:0][15:0] attn_x, attn_y;
    always_comb begin
        attn_x = '0; attn_y = '0;
        if (is_sv_r) begin                                // Score×V: 4 packet slots on lanes 0,2,4,6
            for (int s = 0; s < 4; s++) begin
                attn_x[0][2*s] = sfu_pprob[oc_r[7:4]][s];                          // live softmax packet prob
                attn_y[0][2*s] = vbuf_all[(oc_r[7:4]*4 + s)*HEAD_SIZE + oc_r[3:0]];// live V-gather
            end
        end else begin                                    // QK: 16 head_dim products on lanes 0,2,..,30
            for (int d = 0; d < HEAD_SIZE; d++) begin
                attn_x[0][2*d] = q_rd[d];
                attn_y[0][2*d] = kv_rd_fp16[d];
            end
        end
    end

    // ===== W4A16 out_proj operand path (handoff #5): INT4 weight + ctx (fp16) on EVEN lanes =====
    // out_proj (sub-op 34, MODE_W4A16) = REAL linear; 16 ic/chunk on PEG0 even lanes (16-bit op mode):
    //   x = INT4 weight (wr_codes[0..15], from the WBUF gather @ the W4A16 8B/chunk stride),
    //   y = ctx (fp16, the score_v output) for this chunk's 16 ic. PEG1..3 = 0 -> oacc_re = Σ W·ctx (real).
    // Dealer stays the conv per-head APoT (ce_top keeps ce_dealer_conv when !i_attn_active). ctx_rd is
    // registered off id_ic_chunk to align with the registered weight window (= mc_x arrival timing).
    logic [15:0] ctx_mem [0:DIM_SIZE-1];                  // score_v ctx (fp16) -> out_proj W4A16 activation
    wire is_w4a16 = (id_pe_mode == MODE_W4A16);
    logic w4a16_active_r;
    logic [HEAD_SIZE-1:0][15:0] ctx_rd;
    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin w4a16_active_r<=1'b0; for (int l=0;l<HEAD_SIZE;l++) ctx_rd[l]<=16'h0; end
        else begin
            w4a16_active_r <= is_w4a16;
            for (int l=0;l<HEAD_SIZE;l++) ctx_rd[l] <= ctx_mem[id_ic_chunk*HEAD_SIZE + l];
        end
    end
    logic [NUM_PEG-1:0][NUM_PE_PG-1:0][15:0] w16_x, w16_y;
    always_comb begin
        w16_x = '0; w16_y = '0;
        for (int l = 0; l < HEAD_SIZE; l++) begin
            w16_x[0][2*l] = {12'd0, wr_codes[l][3:0]};   // INT4 weight on even lane 2l
            w16_y[0][2*l] = ctx_rd[l];                   // ctx fp16 on even lane 2l
        end
    end

    // ===== ce_top DISSOLVED into engine (hierarchy refactor T1b, 2026-06-10): inst_decoder +
    // mem_controller + latency-align + operand mux + CE_core(=ce) are now direct children of the
    // engine level (was a ce_top wrapper). Behavior identical; ports became internal signals. =====
    localparam logic [2:0] P_DRAIN  = 3'd4;
    localparam int OACC_DLY  = 2;   // SRAM read(1) + PEG output reg(1)  [verify in smoke]
    localparam int DRAIN_LAT = 4;   // DRAIN cycles for the last partial to settle through ppu
    localparam int RESV_DLY  = 2;   // drain_done -> o_re valid (oacc -> ppu comb -> ppu reg)

    // ===== inst_decoder (declarations moved up near the top — referenced early in this engine) =====
    // score_v (sub-op 33) has no const block (SCALE_NONE) -> its real "ready" is the softmax packet +
    // V-gather, not the (transient 1-cycle) const-prefetch pulse. So for sub-op 33 the PREFETCH->RUN
    // handshake waits on gather_done (= ce_engine gather_done, a LEVEL that holds), REPLACING the
    // prefetch_done pulse (which fires ~5cy in, long before the ~180cy sfu+gather and would be missed).
    // All other sub-ops use prefetch_done unchanged.
    // sub-op 35 (output_proj) reads its activation from UF (= the tied recombine e4m3 codes that ce_engine
    // writes back during pass-B out_proj). Gate its PREFETCH->RUN on BOTH the weight-scale prefetch AND
    // recombine_done, so the UF read can't race the recombine write-back.
    wire prefetch_done_gated = (id_subop_idx == 7'd33) ? gather_done
                             : (id_subop_idx == 7'd35) ? (prefetch_done & recombine_done)
                             : prefetch_done;

    inst_decoder u_id (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_start(i_start),
        .i_prefetch_done(prefetch_done_gated), .i_drain_done(drain_done),
        .o_busy(o_busy), .o_done(o_done),
        .o_subop(id_subop), .o_subop_idx(id_subop_idx), .o_phase(id_phase),
        .o_oc_idx(id_oc_idx), .o_ic_chunk(id_ic_chunk), .o_tap(id_tap),
        .o_pe_mode(id_pe_mode), .o_ppu_mode(id_ppu_mode), .o_preshift(id_preshift),
        .o_oacc_clr(id_oacc_clr), .o_oacc_en(id_oacc_en), .o_ppu_go(id_ppu_go),
        .o_oc_complete(id_oc_complete),
        .o_prefetch(id_prefetch), .o_cache_half(id_cache_half),
        .o_pass_sel(id_pass_sel)
    );

    assign w_is_fp8 = (id_pe_mode == MODE_LMUL_FP8);   // weight gather width select (-> ce_engine)

    // ===== mem_controller =====
    logic [NUM_PEG-1:0][NUM_PE_PG-1:0][15:0] mc_x, mc_y;
    logic [NUM_PEG-1:0][11:0] mc_dealer;

    mem_controller u_mc (
        .i_clk(i_clk), .i_rst_n(i_rst_n),
        .i_subop(id_subop), .i_phase(id_phase),
        .i_oc_idx(id_oc_idx), .i_ic_chunk(id_ic_chunk), .i_tap(id_tap),
        .i_pass_sel(id_pass_sel),
        .i_prefetch(id_prefetch),
        .o_prefetch_done(prefetch_done), .i_wbuf_const_rdata(const_word),
        .o_wr_byte_addr(wr_addr), .o_wi_byte_addr(wi_addr), .o_w_nibble_odd(w_nibble_odd),
        .o_xr_byte_addr(xr_addr), .o_xi_byte_addr(xi_addr), .o_act_en(act_en),
        .o_wbuf_en(wbuf_en), .o_wbuf_is_prefetch(wbuf_is_prefetch),
        .o_wbuf_byte_addr(wbuf_addr), .o_wbuf_row(), .o_wbuf_bank(),
        .i_wr_codes(wr_codes), .i_wi_codes(wi_codes),
        .i_xr_codes(xr_codes), .i_xi_codes(xi_codes),
        .o_x(mc_x), .o_y(mc_y), .o_dealer_apot(mc_dealer)
    );

    // ===== latency-align OACC control to the operand pipeline (delay by OACC_DLY) =====
    logic [OACC_DLY-1:0] clr_pipe, en_pipe;   // index-based shift: robust for any OACC_DLY>=1
    always_ff @(posedge i_clk or negedge i_rst_n)
        if (!i_rst_n) begin clr_pipe <= '0; en_pipe <= '0; end
        else begin
            clr_pipe[0] <= id_oacc_clr;
            en_pipe [0] <= id_oacc_en;
            for (int k = 1; k < OACC_DLY; k++) begin
                clr_pipe[k] <= clr_pipe[k-1];
                en_pipe [k] <= en_pipe [k-1];
            end
        end
    wire ce_oacc_clr = clr_pipe[OACC_DLY-1];
    wire ce_oacc_en  = en_pipe [OACC_DLY-1];

    // ===== latency-align the per-CHUNK dealer APoT code to the operand pipeline =====
    // The code is combinational off the decoder's ic_chunk @T, but the matching PEG partial only
    // reaches the dealer @T+OACC_DLY (SRAM read + PEG reg). Single-chunk layers have a constant
    // code so this was invisible; MULTI-chunk (ic>32) descaled chunk c with chunk c+2's scale ->
    // wrong OACC. Delay the code by OACC_DLY, exactly like clr/en. (preshift/pe_mode/ppu_mode are
    // per-sub-op constants -> no per-chunk delay needed.)
    logic [NUM_PEG-1:0][11:0] dealer_pipe [0:OACC_DLY-1];
    always_ff @(posedge i_clk or negedge i_rst_n)
        if (!i_rst_n) for (int k = 0; k < OACC_DLY; k++) dealer_pipe[k] <= '0;
        else begin
            dealer_pipe[0] <= mc_dealer;
            for (int k = 1; k < OACC_DLY; k++) dealer_pipe[k] <= dealer_pipe[k-1];
        end
    wire [NUM_PEG-1:0][11:0] ce_dealer_conv = dealer_pipe[OACC_DLY-1];

    // ===== attention / W4A16 operand + dealer select =====
    // ce_engine presents attn_x/y (+ w16_x/y) ALREADY aligned to the conv mc_x/mc_y arrival
    // (registered 1-cyc off the decoder counters), so select directly. QK/Score×V carry no weight descale
    // -> dealer identity; W4A16 (out_proj) DOES carry a per-head weight scale -> keeps the conv dealer.
    wire [NUM_PEG-1:0][NUM_PE_PG-1:0][15:0] ce_x = attn_active_r ? attn_x : (w4a16_active_r ? w16_x : mc_x);
    wire [NUM_PEG-1:0][NUM_PE_PG-1:0][15:0] ce_y = attn_active_r ? attn_y : (w4a16_active_r ? w16_y : mc_y);
    wire [NUM_PEG-1:0][11:0] ce_dealer;
    genvar gd;
    generate for (gd=0; gd<NUM_PEG; gd++) assign ce_dealer[gd] = attn_active_r ? 12'hC00 : ce_dealer_conv[gd]; endgenerate

    // ===== CE_core (verified spine) =====
    CE_core u_ce (
        .i_clk(i_clk), .i_rst_n(i_rst_n),
        .i_pe_mode(id_pe_mode), .i_x(ce_x), .i_y(ce_y),
        .i_preshift(id_preshift), .i_dealer_apot(ce_dealer),
        .i_oacc_en(ce_oacc_en), .i_oacc_clr(ce_oacc_clr),
        .i_ppu_mode(id_ppu_mode),
        .i_bn_alpha_re(bn_a_re), .i_bn_alpha_im(bn_a_im),
        .i_bn_beta_re(bn_b_re),   .i_bn_beta_im(bn_b_im),
        .i_prelu_re(prelu_re),       .i_prelu_im(prelu_im),
        .i_cast_k(cast_k), .i_scale_re(scale_re), .i_scale_im(scale_im),
        .o_re(o_re), .o_im(o_im),
        .o_pre_re(pre_re), .o_pre_im(pre_im)       // pre-cast fp16 (skip-write dual-cast tap)
    );

    // ===== drain_done: DRAIN_LAT cycles into DRAIN (pipe settle) =====
    wire in_drain = (id_phase == P_DRAIN);
    logic [2:0] drain_cnt;
    always_ff @(posedge i_clk or negedge i_rst_n)
        if (!i_rst_n)        drain_cnt <= '0;
        else if (in_drain)   drain_cnt <= (drain_cnt == DRAIN_LAT[2:0]) ? drain_cnt : drain_cnt + 3'd1;
        else                 drain_cnt <= '0;
    assign drain_done = in_drain && (drain_cnt == DRAIN_LAT[2:0]);

    // ===== o_result_valid: ONE pulse per output channel, RESULT_LAT after its o_oc_complete =====
    // o_re = ppu(oacc) registered; oacc settles OACC_DLY+1 after the last accumulate, ppu reg +1
    // => o_re holds the channel result at o_oc_complete + (OACC_DLY+2). Pipelined: many in flight.
    localparam int RESULT_LAT = OACC_DLY + 2;     // = 4 (verify vs golden: sample SAMPLE_LAT=RESULT_LAT-1)
    logic [RESULT_LAT-1:0] rv_pipe;
    always_ff @(posedge i_clk or negedge i_rst_n)
        if (!i_rst_n) rv_pipe <= '0;
        else begin
            rv_pipe[0] <= id_oc_complete;
            for (int k = 1; k < RESULT_LAT; k++) rv_pipe[k] <= rv_pipe[k-1];
        end
    assign o_result_valid = rv_pipe[RESULT_LAT-1];


    // ===== mha_in_proj OUTPUT write-back (chaining handoff #1): Q -> q_mem =====
    // mha_in_proj (sub-op 31) streams the e4m3 Q/K/V codes on o_re per oc (768=[Q256|K256|V256]). The Q
    // (oc<256, = q_mem[oc] head-major) feeds the live QK as fp16 = e4m3(code)·2^k_Q (k_Q=-7). Tracks the
    // oc at o_result_valid (= id_oc_complete + RESULT_LAT=4). (K/V->KVBUF write-back + frame ring = next.)
    localparam int CHAIN_RV_LAT = 4;                       // = ce_top RESULT_LAT
    localparam signed [7:0] K_Q = -8'sd5;                  // Q block-float exponent (dequant ×2^k_Q; consultant shared k=-5)
    logic [9:0] ocv [0:CHAIN_RV_LAT-1];  logic [6:0] subv [0:CHAIN_RV_LAT-1];  logic psv [0:CHAIN_RV_LAT-1];
    always_ff @(posedge i_clk) begin
        ocv[0] <= id_oc_idx;  subv[0] <= u_id.subop_idx;  psv[0] <= id_pass_sel;
        for (int k=1;k<CHAIN_RV_LAT;k++) begin ocv[k]<=ocv[k-1]; subv[k]<=subv[k-1]; psv[k]<=psv[k-1]; end
    end
    wire [9:0] oc_rv = ocv[CHAIN_RV_LAT-1];  wire [6:0] sub_rv = subv[CHAIN_RV_LAT-1];  wire pass_rv = psv[CHAIN_RV_LAT-1];
    logic [15:0] q_deq, q_deq_raw;
    fp8_to_fp16 u_qd (.i_fp8(o_re[7:0]),       .o_fp16(q_deq_raw));     // e4m3 Q code -> fp16
    fp16_scale2 u_qs (.i_x(q_deq_raw), .i_d(K_Q), .o_y(q_deq));        // ×2^k_Q -> Q operand
    always @(posedge i_clk)
        if (o_result_valid && sub_rv == 7'd31 && oc_rv < 10'd256)
            q_mem[oc_rv] <= q_deq;                                      // head-major: q_mem[oc]=Q[oc//16][oc%16]

    // score_v (sub-op 33) ctx output -> ctx_mem (the out_proj W4A16 activation). oc = head*16+d, fp16.
    always @(posedge i_clk)
        if (o_result_valid && sub_rv == 7'd33 && oc_rv < 10'd256)
            ctx_mem[oc_rv] <= o_re;

    // ===== tied_mha #6: 2-pass out_proj post-bias + recombine (single-cycle combinational) =====
    // out_proj(34) is PPU_PASS so o_re = RAW pre-bias oacc. Per-pass +b_o here -> POST-bias a'/b'
    // (so out_proj_a/b match the consultant's t351_a/b, flag #1), then recombine out_re=a'-b' /
    // out_im=a'+b' (b_o cancels in real, doubles in imag), e4m3-cast @ +3 -> output_proj(35) input.
    // 7 SHARED fp16 units (3 add + 2 scale + 2 e4m3). Timing is loose so the whole pass-B chain
    // add(b')->add(a'∓b')->scale2->e4m3 runs COMBINATIONAL in one cycle (no pipeline split):
    //   pass A (rv): out_proj_a[oc] <= fp16_add_sat(o_re, b_o)            (= a')
    //   pass B (rv): b'=fp16_add_sat(o_re,b_o); out_re=a'-b'; out_im=a'+b'; e4m3(+3) -> codes; all @ this edge.
    // recombine_done asserts at pass B's last out_proj channel (oc 255) — zero tail latency.
    logic [15:0] out_proj_a [0:DIM_SIZE-1];          // pass A out_proj POST-bias a' = oacc_a + b_o
    logic [15:0] out_proj_b [0:DIM_SIZE-1];          // pass B out_proj POST-bias b' = oacc_b + b_o
    logic [15:0] bias_mem_outproj [0:DIM_SIZE-1];    // b_o[256] fp16 (TB-preload; WBUF const @253920 later)
    logic [15:0] recomb_re [0:DIM_SIZE-1], recomb_im [0:DIM_SIZE-1];           // out_re/out_im (fp16)
    logic [7:0]  recomb_re_code [0:DIM_SIZE-1], recomb_im_code [0:DIM_SIZE-1]; // e4m3 @ i_cast_k=+3
    // recombine_done is forward-declared up top (used by prefetch_done_gated before that block)

    // post-bias adder (SHARED pass A/B; different cycles): o_re (pre-bias oacc) + b_o[oc] -> a' or b'
    wire [15:0] pb_z;
    fp16_add_sat u_pb_add (.a(o_re), .b(bias_mem_outproj[oc_rv[7:0]]), .z(pb_z));
    // pass-B combinational recombine off a' (stored, this oc) and b' (=pb_z): out_re=a'-b', out_im=a'+b'
    wire [15:0] rc_aprime = out_proj_a[oc_rv[7:0]];                 // pass-A post-bias a' for this oc
    wire [15:0] rc_b_neg  = {~pb_z[15], pb_z[14:0]};                // negate b' = sign-bit flip
    wire [15:0] rc_re_z, rc_im_z;
    fp16_add_sat u_rc_re (.a(rc_aprime), .b(rc_b_neg), .z(rc_re_z));
    fp16_add_sat u_rc_im (.a(rc_aprime), .b(pb_z),     .z(rc_im_z));
    // e4m3 cast @ i_cast_k=+3 (output_proj input block-float k=-3)
    wire [15:0] rc_re_s3, rc_im_s3;  wire [7:0] rc_re_code, rc_im_code;
    fp16_scale2  u_rc_sc_re (.i_x(rc_re_z), .i_d(8'sd3), .o_y(rc_re_s3));
    fp16_scale2  u_rc_sc_im (.i_x(rc_im_z), .i_d(8'sd3), .o_y(rc_im_s3));
    fp16_to_e4m3 u_rc_e4_re (.i_h(rc_re_s3), .o_e4m3(rc_re_code));
    fp16_to_e4m3 u_rc_e4_im (.i_h(rc_im_s3), .o_e4m3(rc_im_code));

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) recombine_done <= 1'b0;
        else begin
            if (i_start) recombine_done <= 1'b0;
            if (o_result_valid && sub_rv == 7'd34 && oc_rv < 10'd256) begin
                if (!pass_rv) out_proj_a[oc_rv[7:0]] <= pb_z;       // pass A: store a'
                else begin                                          // pass B: store b' + recombine + cast (1 cy)
                    out_proj_b[oc_rv[7:0]]     <= pb_z;
                    recomb_re[oc_rv[7:0]]      <= rc_re_z;
                    recomb_im[oc_rv[7:0]]      <= rc_im_z;
                    recomb_re_code[oc_rv[7:0]] <= rc_re_code;
                    recomb_im_code[oc_rv[7:0]] <= rc_im_code;
                    if (oc_rv[7:0] == 8'd255) recombine_done <= 1'b1;
                end
            end
        end
    end

    // #7 feed: write the tied-recombine e4m3 codes into UF as output_proj(35) activation (real=uf[0..255],
    // imag=uf[512..767] = the proj act layout). SEPARATE plain always (not always_ff) so it coexists with
    // the TB's UF preload (models the UF write port). sub-op 35's PREFETCH->RUN is gated on recombine_done
    // (ce_top) so its gather read can't race these writes.
    always @(posedge i_clk)
        if (o_result_valid && sub_rv == 7'd34 && pass_rv && oc_rv < 10'd256) begin
            uf_mem[oc_rv[7:0]]           <= rc_re_code;
            uf_mem[10'd512 + oc_rv[7:0]] <= rc_im_code;
        end

    // ===== LIVE const feed: load PPU β/bias from the WBUF const region (replaces TB-preload) =====
    // At each attention sub-op's FETCH, block-load its β/bias from the WBUF image const region into the
    // PPU bias buffers (bias_mem = sole writer, so no TB-preload / no ICPD). The bn_b_re/im + recombine
    // muxes read these buffers unchanged. Offsets are the frozen memory_map const_blocks. This is the
    // behavioral form (whole block in 1 cy at FETCH, well before the PPU read in RUN); the synth path
    // streams the same bytes into ppu_const_cache during PREFETCH.
    localparam int CB_INPROJ_BIAS  = 254432;   // mha_in_proj(31) bias [768] fp16 ([Q|K|V])
    localparam int CB_OUTPROJ_BO   = 253920;   // mha_out_proj(34) b_o [256] fp16 (recombine per-pass +b_o)
    localparam int CB_OUTPUTPROJ_B = 252896;   // output_proj(35) β [256 re ‖ 256 im] fp16
    wire [6:0] id_sub_le = u_id.subop_idx;
    wire       id_fetch  = (id_phase == 3'd1);   // P_FETCH
    always @(posedge i_clk) begin
        if (id_fetch && id_sub_le == 7'd31)
            for (int o=0;o<768;o++) bias_mem[o]          <= {wbuf_mem[CB_INPROJ_BIAS+2*o+1],  wbuf_mem[CB_INPROJ_BIAS+2*o]};
        if (id_fetch && id_sub_le == 7'd34)
            for (int o=0;o<256;o++) bias_mem_outproj[o]  <= {wbuf_mem[CB_OUTPROJ_BO+2*o+1],    wbuf_mem[CB_OUTPROJ_BO+2*o]};
        if (id_fetch && id_sub_le == 7'd35)
            for (int o=0;o<256;o++) begin
                bias_mem_op35_re[o] <= {wbuf_mem[CB_OUTPUTPROJ_B+2*o+1],     wbuf_mem[CB_OUTPUTPROJ_B+2*o]};
                bias_mem_op35_im[o] <= {wbuf_mem[CB_OUTPUTPROJ_B+512+2*o+1], wbuf_mem[CB_OUTPUTPROJ_B+512+2*o]};
            end
    end

    // ===== mha_in_proj K/V write-back (chaining handoff #2): K/V -> KVBUF (ce_kv_cache, ring slot=ptr) =====
    // oc 256..511 = K (sel=0), 512..767 = V (sel=1); head=oc[7:4], dim=oc[3:0]. The PPU already e4m3-cast
    // K/V (cast_k=+5 -> block-float codes), so o_re[7:0] IS the storage code -> gather 16 dims/head and
    // write the whole head into KVBUF (code-mode, no re-cast). k_K=k_V=-5 ride the QK/ScoreV read descale.
    // The cache stores to the CURRENT frame's ring slot (=ptr); kv_wr_pass picks K_re/V_re (pass A) vs
    // K_im/V_im (pass B). Gather: latch dims 0..14 into kv_codebuf as they stream; at dim==15 assemble
    // {latched[0..14],o_re[15]} combinationally and pulse the synchronous KVBUF write (samples cycle-15 @next edge).
    wire        wb_kv     = o_result_valid && (sub_rv == 7'd31) && (oc_rv >= 10'd256) && (oc_rv < 10'd768);
    wire [3:0]  wb_head   = oc_rv[7:4];
    wire [3:0]  wb_dim    = oc_rv[3:0];
    wire        wb_sel    = oc_rv[9];                                   // oc>=512 -> V (sel=1); 256..511 -> K
    logic [HEAD_SIZE-1:0][7:0] kv_codebuf;                              // dims 0..14 of the in-flight head
    always @(posedge i_clk)
        if (wb_kv) kv_codebuf[wb_dim] <= o_re[7:0];
    always_comb begin
        for (int d = 0; d < HEAD_SIZE; d++) kv_wr_vec[d] = {8'h00, kv_codebuf[d]};
        kv_wr_vec[HEAD_SIZE-1] = {8'h00, o_re[7:0]};                    // dim 15 = current cycle's code
    end
    assign kv_wr_en   = wb_kv && (wb_dim == 4'(HEAD_SIZE-1));
    assign kv_wr_sel  = wb_sel;
    assign kv_wr_head = wb_head;
    assign kv_wr_pass = pass_rv;                                        // pass A (real)->K_re/V_re; pass B->K_im/V_im

    // ===== encoder skip-WRITE (T2 #8): capture encoder pw outputs -> SKIPBUF =====
    // Encoder pw sub-ops (idx 1,3,5,7,9,11 = stage s output, s = idx>>1) stream the stage output
    // at o_result_valid. The static top-50% selection (ce_skip_rom: hit + compact position, in
    // skip_indices FILE order = the verified dump/storage order) routes channel oc_rv to the
    // consuming decoder stage d=5-s block @ SKP_BASE (re half; im at +SKIPBUF_IM_HALF).
    //   s=0..4 (-> dec5..dec1): COPY o_re/o_im[7:0] — same block-float k as the UF consumer
    //   (verified bit-exact, T2 recon). s=5 (-> dec0): CAST mode — dec0's skip k=+1 differs from
    //   the UF consumer k=-7/-6, so re-cast the PRE-CAST fp16 at x2^-1 (each consumer gets its own
    //   single quantization from the same fp16; CONSULT_SKIP_DATAFLOW B1(a), pending rubber-stamp).
    // skip_wr_en: TB-pokeable gate (golden skip-READ TBs preload skip_mem and poke 0 so the walk's
    // encoder passes don't overwrite the preload). Engine orchestration keeps it 1.
    `include "ce_skip_rom.svh"
    localparam int SKIPBUF_IM_HALF = 2048;
    logic skip_wr_en = 1'b1;
    wire is_enc_pw_rv = (sub_rv==7'd1)||(sub_rv==7'd3)||(sub_rv==7'd5)
                      ||(sub_rv==7'd7)||(sub_rv==7'd9)||(sub_rv==7'd11);
    wire [2:0] skp_enc_s = sub_rv[3:1];                 // odd idx 1..11 -> stage 0..5
    logic [9:0] skp_base;                               // dec stage d=5-s block base (f=0)
    always_comb unique case (skp_enc_s)
        3'd0: skp_base = 10'd576;  3'd1: skp_base = 10'd448;  3'd2: skp_base = 10'd320;
        3'd3: skp_base = 10'd192;  3'd4: skp_base = 10'd64;   default: skp_base = 10'd0;
    endcase
    wire [7:0] skp_hp = ce_skip_hitpos(skp_enc_s, oc_rv);   // {hit, pos}
    // dec0 CAST: single-quant the pre-cast fp16 at the dec0 skip block-float k=+1 (cast x2^-1)
    logic [15:0] skp_csc_re, skp_csc_im;  logic [7:0] skp_cc_re, skp_cc_im;
    fp16_scale2  u_skc_re (.i_x(pre_re), .i_d(-8'sd1), .o_y(skp_csc_re));
    fp16_scale2  u_skc_im (.i_x(pre_im), .i_d(-8'sd1), .o_y(skp_csc_im));
    fp16_to_e4m3 u_ske_re (.i_h(skp_csc_re), .o_e4m3(skp_cc_re));
    fp16_to_e4m3 u_ske_im (.i_h(skp_csc_im), .o_e4m3(skp_cc_im));
    always @(posedge i_clk)                              // plain always: skip_mem is TB-preloadable
        if (skip_wr_en && o_result_valid && is_enc_pw_rv && skp_hp[7]) begin
            skip_mem[32'(skp_base) + 32'(skp_hp[6:0])]
                <= (skp_enc_s == 3'd5) ? skp_cc_re : o_re[7:0];
            skip_mem[SKIPBUF_IM_HALF + 32'(skp_base) + 32'(skp_hp[6:0])]
                <= (skp_enc_s == 3'd5) ? skp_cc_im : o_im[7:0];
        end

    // ===== STFT spectrum region (dual-role SKIPBUF, T2 #8): Q2.14 int16 LE =====
    // re half @ SPEC_BASE + ear*128 + bin*2 ; im half mirrored at +SKIPBUF_IM_HALF. Disjoint from
    // the skip blocks (<=704). Behavioral one-shot write (synth: 16-row FSM per half at macro swap).
    localparam int SPEC_BASE = 1024;
    always @(posedge i_clk)
        if (i_spec_wr_en)
            for (int bn = 0; bn < 64; bn++) begin
                {skip_mem[SPEC_BASE + 32'(i_spec_wr_ear)*128 + bn*2 + 1],
                 skip_mem[SPEC_BASE + 32'(i_spec_wr_ear)*128 + bn*2    ]} <= i_spec_wr_re[bn];
                {skip_mem[SKIPBUF_IM_HALF + SPEC_BASE + 32'(i_spec_wr_ear)*128 + bn*2 + 1],
                 skip_mem[SKIPBUF_IM_HALF + SPEC_BASE + 32'(i_spec_wr_ear)*128 + bn*2    ]} <= i_spec_wr_im[bn];
            end
    assign o_spec_re = {skip_mem[SPEC_BASE + 32'(i_spec_rd_ear)*128 + 32'(i_spec_rd_bin)*2 + 1],
                        skip_mem[SPEC_BASE + 32'(i_spec_rd_ear)*128 + 32'(i_spec_rd_bin)*2    ]};
    assign o_spec_im = {skip_mem[SKIPBUF_IM_HALF + SPEC_BASE + 32'(i_spec_rd_ear)*128 + 32'(i_spec_rd_bin)*2 + 1],
                        skip_mem[SKIPBUF_IM_HALF + SPEC_BASE + 32'(i_spec_rd_ear)*128 + 32'(i_spec_rd_bin)*2    ]};

    // ===== WBUF / UF read: word-aligned window (registered 1cy) + ce_operand_gather =====
    // weights: 40 B window covers FP8 (32 weights, 1 B/wt) gather; INT4 uses the low 25 B (17 B of
    // 32 nibbles + up to 1-nibble odd-ic shift). acts: 32 B (32 e4m3). Window = NBYTES+8 for (addr&7).
    logic [(32+8)*8-1:0] win_wr, win_wi;
    logic [(32+8)*8-1:0] win_xr, win_xi;
    logic [2:0] off_wr, off_wi, off_xr, off_xi;
    logic [63:0] win_const;

    // behavioral read of a byte window from (addr & ~7), registered (models SRAM latency 1)
    always_ff @(posedge i_clk) begin
        for (int b = 0; b < 40; b++) begin
            win_wr[b*8 +: 8] <= wbuf_mem[{wr_addr[17:3],3'b0} + b];
            win_wi[b*8 +: 8] <= wbuf_mem[{wi_addr[17:3],3'b0} + b];
        end
        // act window source: UF (default) or SKIPBUF (decoder pw_skip, id_subop.act_is_skip).
        // The select is sampled at the SAME edge as the address (both time-T values) — no extra
        // alignment register needed (unlike w_is_fp8_r, which is consumed combinationally
        // AFTER this register stage).
        for (int b = 0; b < 40; b++) begin
            win_xr[b*8 +: 8] <= id_subop.act_is_skip ? skip_mem[({xr_addr[17:3],3'b0} + b) % SKIPBUF_BYTES]
                                                     : uf_mem  [({xr_addr[17:3],3'b0} + b) % UFBUF_BYTES];
            win_xi[b*8 +: 8] <= id_subop.act_is_skip ? skip_mem[({xi_addr[17:3],3'b0} + b) % SKIPBUF_BYTES]
                                                     : uf_mem  [({xi_addr[17:3],3'b0} + b) % UFBUF_BYTES];
        end
        // const read models an ENABLE-GATED SRAM output reg: it HOLDS when not read, so the
        // value stays valid at ce_prefetch_seq's S_CAP (T+2 after S_RD). A free-running re-read
        // would already show the NEXT word by S_CAP (word_addr increments in S_RD) -> wrong codes.
        if (wbuf_en)
            for (int b = 0; b < 8; b++)
                win_const[b*8 +: 8] <= wbuf_mem[{wbuf_addr[17:3],3'b0} + b];
        off_wr <= wr_addr[2:0]; off_wi <= wi_addr[2:0];
        off_xr <= xr_addr[2:0]; off_xi <= xi_addr[2:0];
        w_nibble_odd_r <= w_nibble_odd;   // align nibble parity to the REGISTERED weight window
        w_is_fp8_r     <= w_is_fp8;       // align weight-format select to the registered window
    end                                   // (continuous RUN advances oc each cycle; combinational
                                          //  w_nibble_odd would be oc+1's parity at the gather)
    assign const_word = win_const;     // const prefetch is word-aligned reads; gather handled in seq

    logic [17*8-1:0] by_wr_i4, by_wi_i4;   // INT4 nibble window (17 B)
    logic [32*8-1:0] by_wr_f8, by_wi_f8;   // FP8 byte window  (32 B)
    logic [32*8-1:0] by_xr, by_xi;
    ce_operand_gather #(.NBYTES(17)) g_wr  (.i_window(win_wr[(17+8)*8-1:0]), .i_byte_off(off_wr), .o_bytes(by_wr_i4));
    ce_operand_gather #(.NBYTES(17)) g_wi  (.i_window(win_wi[(17+8)*8-1:0]), .i_byte_off(off_wi), .o_bytes(by_wi_i4));
    ce_operand_gather #(.NBYTES(32)) g_wrf (.i_window(win_wr), .i_byte_off(off_wr), .o_bytes(by_wr_f8));
    ce_operand_gather #(.NBYTES(32)) g_wif (.i_window(win_wi), .i_byte_off(off_wi), .o_bytes(by_wi_f8));
    ce_operand_gather #(.NBYTES(32)) g_xr  (.i_window(win_xr), .i_byte_off(off_xr), .o_bytes(by_xr));
    ce_operand_gather #(.NBYTES(32)) g_xi  (.i_window(win_xi), .i_byte_off(off_xi), .o_bytes(by_xi));

    // weight unpack: FP8 -> lane l = byte l (8b) ; INT4 -> lane l = nibble (l + w_nibble_odd) (4b,
    // low-first; the odd shift lands the per-oc weight on the active lane for odd-ic). For INT4 the
    // upper 4 bits are 0 so pe_pair (W4A8) reads [3:0]; FP8 uses the full byte (LMUL reads [7:0]).
    genvar l;
    generate
        for (l = 0; l < NUM_PE_PG; l++) begin : g_unpack
            assign wr_codes[l] = w_is_fp8_r ? by_wr_f8[l*8 +: 8] : {4'd0, by_wr_i4[(l + w_nibble_odd_r)*4 +: 4]};
            assign wi_codes[l] = w_is_fp8_r ? by_wi_f8[l*8 +: 8] : {4'd0, by_wi_i4[(l + w_nibble_odd_r)*4 +: 4]};
            assign xr_codes[l] = by_xr[l*8 +: 8];                     // e4m3 byte
            assign xi_codes[l] = by_xi[l*8 +: 8];
        end
    endgenerate

    // ===== ppu_const_cache (2KiB ping-pong) — present; FILL (BN/PReLU/k prefetch) is TODO =====
    logic [127:0] cc_window;
    ppu_const_cache u_cc (
        .i_clk(i_clk), .i_rst_n(i_rst_n),
        .i_fill_en(1'b0), .i_fill_half(1'b0), .i_fill_waddr('0), .i_fill_data('0),
        .i_rd_half(1'b0), .i_rd_waddr('0), .o_rd_window(cc_window)
    );

    // ===== ce_attn_sfu (Softmax_v3 + top-4 packet) — LIVE fed by the QK o_re stream (handoff #4) =====
    // During QK (sub-op 32) each o_result_valid writes score_buf[head][frame] = o_re (postscale score),
    // head=oc[8:5], frame=oc[4:0]. i_start pulses on the LAST QK score (oc=511); the sfu walks 16 heads
    // through Softmax_v3 (~7cy/head) and exposes the top-4 packet (idx-ascending, dense fp16 prob). The
    // packet feeds Score×V (sub-op 33); o_done gates the score_v launch (sequencing in ce_top, handoff #4b).
    wire        sfu_wr_en    = o_result_valid && (sub_rv == 7'd32);
    wire [3:0]  sfu_wr_head  = oc_rv[8:5];
    wire [4:0]  sfu_wr_frame = oc_rv[4:0];
    wire        sfu_start    = o_result_valid && (sub_rv == 7'd32) && (oc_rv == 10'd511);
    // warm-up validity mask (T3): query frame i sees n_valid = min(filled+1, 32) real keys at
    // local_idx 0..n_valid-1 (its own + the filled history; mask = causal triu + 31-window, the
    // ONLY deploy rule — consultant B5). Steady state (filled saturated 32) -> all-ones. With
    // <4 valid the Softmax_v3 top-k degenerates to min(4, n_valid) live packet slots (prob=0
    // empties), pow2 denominator over the survivors — deploy-verified vs mha_module_io_warmup.
    wire [5:0]  attn_n_valid = (kv_filled >= 6'd31) ? 6'd32 : (kv_filled + 6'd1);
    wire [32:0] attn_mask_w  = (33'h1 << attn_n_valid) - 33'h1;
    wire [SEQ_LEN-1:0] attn_valid_mask = attn_mask_w[SEQ_LEN-1:0];

    ce_attn_sfu u_sfu (                                    // sfu_pidx/pprob/done declared up top (forward)
        .i_clk(i_clk), .i_rst_n(i_rst_n),
        .i_wr_en(sfu_wr_en), .i_wr_head(sfu_wr_head), .i_wr_frame(sfu_wr_frame), .i_wr_score(o_re),
        .i_start(sfu_start), .o_done(sfu_done),
        .i_valid_mask(attn_valid_mask),
        .o_pidx(sfu_pidx), .o_pprob(sfu_pprob)
    );

    // ===== Score×V V-gather (handoff #4b): after sfu_done, read KVBUF V at the 4 packet frames/head =====
    // 64 reads (16 head × 4 slot); KVBUF read is registered 1cy -> issue @gcnt, capture @gcnt+1 into
    // vbuf_all[(head*4+slot)*16+d]. gather_rd drives the KVBUF read mux (sel=V, ×2^-5); gather_done gates
    // score_v(33) RUN (ce_top i_attn_sv_ready). Empty packet slots (idx=0,prob=0) read V[.][0] harmlessly
    // (multiplied by prob=0 in the lane). For one query frame this runs once after the softmax.
    typedef enum logic [1:0] {G_IDLE, G_READ, G_RDY} gstate_t;
    gstate_t gstate;
    logic [6:0] gcnt;  logic gcap_v;  logic [6:0] gcap_idx;
    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin gstate<=G_IDLE; gcnt<='0; gather_done<=1'b0; gcap_v<=1'b0; gcap_idx<='0; end
        else begin
            gcap_v   <= (gstate==G_READ) && (gcnt < 7'd64);    // a V read issued this cycle -> capture next
            gcap_idx <= gcnt;
            unique case (gstate)
                G_IDLE: if (sfu_done) begin gcnt<='0; gather_done<=1'b0; gstate<=G_READ; end
                G_READ: begin gcnt<=gcnt+7'd1; if (gcnt==7'd64) gstate<=G_RDY; end
                // RE-ARM for tied 2-pass: hold gather_done through score_v(33), then return to G_IDLE
                // once score_v has left (is_sv low at sub-op 34) so pass B's sfu_done re-triggers a fresh
                // V-gather. (Single-pass: harmless — gather_done is only consumed by sub-op 33's RUN gate.)
                G_RDY:  begin gather_done<=1'b1; if (!is_sv) begin gather_done<=1'b0; gstate<=G_IDLE; end end
                default: gstate<=G_IDLE;
            endcase
            if (gcap_v) for (int d=0; d<HEAD_SIZE; d++) vbuf_all[gcap_idx*HEAD_SIZE + d] <= kv_rd_fp16[d];
        end
    end
    assign gather_rd = (gstate==G_READ) && (gcnt < 7'd64);
    assign gh = gcnt[5:2];                                  // head = gcnt/4
    assign gf = sfu_pidx[gcnt[5:2]][gcnt[1:0]];             // packet frame for slot gcnt%4

endmodule
