/*
    mem_controller.sv — CE Step 2-3 (fuse-independent parts).

    Turns inst_decoder's {subop, phase, counters} into:
      (1) WBUF weight byte addresses for the paired _r/_i tensors (+ bank-split),
      (2) the STATIC PEG-per-stream operand mux (§4: fr->PEG0/2, fi->PEG1/3,
          xr->PEG0/3, xi->PEG1/2),
      (3) WBUF single-port arbitration (PE weight read has priority; const-prefetch
          fills the spare cycle — never the same cycle, by the 1024 vs 512 b/cyc headroom).

    Decision A: each sub-op already pairs _r/_i (subop_t.w_r_base / w_i_base, both explicit
    since the r->i stride varies per layer). The complex combine is hardwired in CE_core;
    here it is just a static route, NO pairing FSM.

    DEALER CODES are formed by ce_dealer_sel (§4 below). Verified offline (consultant +
    builder read): descale is PRE-FUSED — build_sram_images.py folds activation-k into the per-tensor
    WEIGHT_SCALE APoT a/b exponents, so the dealer needs NO runtime k subtraction (NO fuse unit),
    just a per-PEG SELECT (pure mux/offset) gated by subop_t.branches_collapse (now in the ROM,
    from fused_descale_summary):
      branches_collapse=1 (shared-k): broadcast -> PEG0/2 <- wscale_r code, PEG1/3 <- wscale_i (2 codes);
      branches_collapse=0 (split):    each block is kreal||kimag (2x), per-PEG half-select with a
        +12-bit offset -> PEG0<-r/kreal, PEG2<-r/kimag, PEG1<-i/kimag, PEG3<-i/kreal (4 codes).
    The select rides ppu_const_cache + const_field_extract (12-bit, cross-byte). dealer.sv unchanged.
    PRESHIFT RESIDUAL (decided 2026-06-08: (a) rebuild image; range-proven safe). The 2026-05-30
    image stores FULL fused descale (no preshift). Scope is NARROWER than "3 overflow paths":
      - Only the 12 FP8-OUTLIER pw tensors (dec 0/3/4 = 24 R2 WEIGHT_SCALE codes) need residual
        = full - preshift baked in (offline apot_encode round-trips them in 6a range).
      - QK and Score x V do NOT touch WEIGHT_SCALE: QK's 1/sqrt(d)=2^-2 rides PPU SCALE (i_scale),
        Score x V dealer residual = 1. Their protection is purely pe_pair.i_preshift + SCALE-mode.
      - INT4 conv / DW / out_proj: preshift=0, untouched.
    SAFETY INVARIANT: pe_pair.i_preshift and the dealer code flip TOGETHER (preshift=0 <=> full,
    preshift!=0 <=> residual). Double-count happens ONLY if pe_pair sets preshift while the code
    stays full; (a) bakes the pairing into the image (hardest to get wrong).
    URGENCY: FP8-outlier/QK/ScoreV single product > 65504 -> per-product Inf without preshift, so
    those paths are FUNCTIONALLY BROKEN (not lossy) until enabled; must do (a) before decoder-FP8 /
    attention bring-up. preshift=0 everywhere NOW -> self-consistent + correct for INT4-only bring-up.

    ASSUMPTIONS (verify vs golden, task 4/9):
      - PW weight in-tensor layout = oc-major, ic contiguous, INT4 nibble low-first:
        byte(oc,ic_chunk) = w_base + (oc*ic)/2 + ic_chunk*16   (32 INT4 = 16 B / chunk).
      - The 32-element operand code vectors (i_*_codes) are bank-gathered by the buffer
        read path from the addresses this module issues (gather itself is the buffer's job).
      - DEPTHWISE addressing + microarch (no cross-channel reduction; 5-tap OACC accumulate)
        is a DIFFERENT mode, left as a TODO below (PW dominates: projection ~94% of MACs).
      - FP16 operand modes (LMUL_ATTN / W4A16) widen the operand to [15:0]; the route below
        is width-agnostic, only the code->16b packing differs (TODO with the attention glue).
*/
`timescale 1ns/10ps
module mem_controller
    import Param_def::*;
    import Mem_def::*;
    import ce_subop_pkg::*;
(
    input  logic        i_clk,
    input  logic        i_rst_n,

    // ---- from inst_decoder ----
    input  subop_t      i_subop,
    input  logic [2:0]  i_phase,            // P_* (3 = RUN, 2 = PREFETCH)
    input  logic [9:0]  i_oc_idx,
    input  logic [3:0]  i_ic_chunk,
    input  logic [2:0]  i_tap,
    input  logic        i_pass_sel,          // tied_mha 2-pass: pass B (1) reads the x.imag UF slice for mha_in_proj
    input  logic        i_prefetch,         // PREFETCH phase active
    output logic        o_prefetch_done,    // -> inst_decoder.i_prefetch_done
    input  logic [63:0] i_wbuf_const_rdata, // WBUF word returned for a prefetch read (1-cyc latency)

    // ---- WBUF weight byte addresses (PE read, per stream) ----
    output logic [17:0] o_wr_byte_addr,     // real weight tensor
    output logic [17:0] o_wi_byte_addr,     // imag weight tensor
    output logic        o_w_nibble_odd,     // (oc*ic)&1: weight nibble index is ODD (odd-ic, e.g. ic=1)
                                            //   -> the gather must start lane0 at the HIGH nibble

    // ---- UFBUF activation byte addresses ----
    output logic [17:0] o_xr_byte_addr,
    output logic [17:0] o_xi_byte_addr,
    output logic        o_act_en,

    // ---- WBUF single read port (arbitrated) ----
    output logic        o_wbuf_en,
    output logic        o_wbuf_is_prefetch, // 1 = const-prefetch read, 0 = PE weight read
    output logic [17:0] o_wbuf_byte_addr,
    output logic [17:0] o_wbuf_row,
    output logic [WBUF_BANKS_ADDRW-1:0] o_wbuf_bank,

    // ---- gathered operand code vectors (from the buffer read path) ----
    input  logic [NUM_PE_PG-1:0][7:0] i_wr_codes,   // 32 weights, real (INT4 in [3:0]; FP8 in [7:0])
    input  logic [NUM_PE_PG-1:0][7:0] i_wi_codes,   // 32 weights, imag
    input  logic [NUM_PE_PG-1:0][7:0] i_xr_codes,   // 32 e4m3 acts, real
    input  logic [NUM_PE_PG-1:0][7:0] i_xi_codes,   // 32 e4m3 acts, imag

    // ---- routed operands to CE_core (via ce_top) ----
    output logic [NUM_PEG-1:0][NUM_PE_PG-1:0][15:0] o_x,   // weight / Q stream
    output logic [NUM_PEG-1:0][NUM_PE_PG-1:0][15:0] o_y,   // act / K stream

    // ---- dealer-code SELECT (pre-fused WEIGHT_SCALE; filled internally by ce_prefetch_seq) ----
    output logic [NUM_PEG-1:0][11:0]  o_dealer_apot
);
    localparam logic [2:0] P_RUN = 3'd3;

    // ===================================================================
    // (1) WBUF weight byte-address generation (PW; oc-major row-major INT4)
    //     byte(oc,ic_chunk) = w_base + (oc*ic)/2 + ic_chunk*16
    //     DW (is_dw): TODO — depthwise has no cross-channel reduction; addressing +
    //     5-tap OACC microarch differ. PW path below is exact for pointwise.
    // ===================================================================
    // (oc*ic)/2 + ic_chunk*16.  PRECONDITION ic<=256 (asserted in inst_decoder): then
    // oc*ic <= 65536, /2 = 32768, and w_base+pw_off stays inside the weight region
    // (<241602) so the 18-bit byte address never wraps. Verify vs golden (task 4/9).
    // FP8-outlier layers store weights as 1 BYTE/weight (e4m3) -> byte stride = oc*ic + chunk*32.
    // INT4 layers pack 2 nibbles/byte -> byte stride = (oc*ic)>>1 + chunk*16.
    // W4A16 (16-bit op mode) packs 16 INT4/chunk = 8 B/chunk (vs W4A8's 32/chunk = 16 B); the even-lane
    // weight assembly + ctx delivery for out_proj live in ce_engine (this just points the gather window).
    wire is_fp8   = (i_subop.pe_mode == MODE_LMUL_FP8);
    wire is_w4a16 = (i_subop.pe_mode == MODE_W4A16);
    logic [18:0] pw_off;
    assign pw_off = is_fp8   ? ((i_oc_idx * i_subop.ic) + (i_ic_chunk << 5))        // FP8: 1 B/wt, 32/chunk
                  : is_w4a16 ? (((i_oc_idx * i_subop.ic) >> 1) + (i_ic_chunk << 3)) // W4A16: INT4, 16/chunk = 8 B
                             : (((i_oc_idx * i_subop.ic) >> 1) + (i_ic_chunk << 4));// W4A8: INT4, 32/chunk = 16 B
    assign o_wr_byte_addr = i_subop.w_r_base + pw_off;
    assign o_wi_byte_addr = i_subop.w_i_base + pw_off;
    // The >>1 above drops the LSB of the nibble index (oc*ic). For ODD oc*ic (only ic=1 in the
    // PW set: encoder.model.0.0.pointwise) the weight for this oc starts at the HIGH nibble of the
    // addressed byte; export the parity so the gather offsets lane0 to it. No-op (0) for even ic.
    assign o_w_nibble_odd = is_fp8 ? 1'b0 : ((i_oc_idx * i_subop.ic) & 1'b1);  // FP8: byte-aligned

    // ---- UFBUF activation byte addresses (32 e4m3 = 32 B / chunk) ----
    // GROUPED pointwise (decoder pw_x/pw_skip): output channel oc belongs to group
    // g = oc_idx / (oc/groups) and reads input channels [g*ic : (g+1)*ic] (ic = per-group width
    // = the descriptor's ic). oc/groups is a power of 2 (groups in {1,2,4}) -> g = oc_idx >>
    // clog2(oc/groups). groups=1 (encoder/ungrouped) -> g=0 -> grp_off=0 (unchanged path).
    logic [9:0] oc_per_grp;
    always_comb unique case (i_subop.groups)
        3'd1:    oc_per_grp = i_subop.oc;
        3'd2:    oc_per_grp = i_subop.oc >> 1;
        3'd4:    oc_per_grp = i_subop.oc >> 2;
        default: oc_per_grp = i_subop.oc;          // treat unknown as ungrouped
    endcase
    logic [3:0] grp_shift;                          // clog2(oc_per_grp) (oc_per_grp is pow2)
    always_comb begin
        grp_shift = '0;
        for (int k = 0; k < 10; k++) if (oc_per_grp[k]) grp_shift = k[3:0];
    end
    wire [9:0]  grp     = i_oc_idx >> grp_shift;                          // 0..groups-1
    wire [17:0] grp_off = grp * i_subop.ic;                               // input-channel byte offset
    // BLOCK-DIAGONAL (mha_in_proj qkv_groups=4): output oc reuses input group (oc/ic)%4, so the same
    // 256-wide input feeds Q/K/V (oc 0..767). grp_bd = (oc>>clog2(ic)) & 3 ; offset = grp_bd*ic.
    logic [4:0] ic_shift;
    always_comb begin ic_shift='0; for (int k=0;k<10;k++) if (i_subop.ic[k]) ic_shift=k[4:0]; end
    wire [9:0]  grp_bd     = (i_oc_idx >> ic_shift) & 10'd3;
    wire [17:0] grp_off_bd = grp_bd * i_subop.ic;
    logic [17:0] act_off;
    assign act_off = (i_subop.blockdiag ? grp_off_bd : grp_off) + (i_ic_chunk << 5);  // group + ic_chunk*32 B
    // tied_mha 2-pass: pass B feeds x.imag to mha_in_proj (blockdiag=sub-op 31) on the SAME real (xr)
    // stream, reusing the tied weights. TB lays x.real at uf[0:255], x.imag at uf[256:511] -> +256 B.
    // Other sub-ops have blockdiag=0 -> no per-pass offset. xi stream (act_off + UFBUF/2) stays 0 for
    // mha_in_proj (real linear: oacc_re = ΣW·x), so only the xr base moves.
    localparam logic [17:0] MHA_X_IMAG_OFF = 18'd256;
    wire [17:0] xr_pass_off = (i_subop.blockdiag && i_pass_sel) ? MHA_X_IMAG_OFF : 18'd0;
    // decoder pw_skip (act_is_skip): the act windows read SKIPBUF, not UF (ce_engine muxes the
    // window source on the same descriptor flag). skip_mem layout: re half @act_base, im half
    // @act_base + SKIP_IM_HALF (4 KiB split 2048/2048). act_base = the stage's compact skip block
    // (freq position f=0; engine's freq loop adds f*count later, T5). The grouped act_off (group g
    // reads skip[g*ic..]) applies unchanged on top — the compact selected-channel order IS the
    // dump/group order (verified vs skip_indices, T2 recon 2026-06-10).
    localparam logic [17:0] SKIP_IM_HALF = 18'd2048;
    assign o_xr_byte_addr = act_off + (i_subop.act_is_skip ? 18'(i_subop.act_base) : xr_pass_off);
    assign o_xi_byte_addr = act_off + (i_subop.act_is_skip ? 18'(i_subop.act_base) + SKIP_IM_HALF
                                                           : 18'(UFBUF_DEPTH * 8 / 2));
    assign o_act_en       = (i_phase == P_RUN);

    // ===================================================================
    // (3) WBUF single-port arbitration: PE weight read (RUN) has priority;
    //     const-prefetch uses the cycle only when PE is not reading WBUF.
    // ===================================================================
    wire pe_read = (i_phase == P_RUN);
    wire        pf_rd_en;                        // const-prefetch read enable (ce_prefetch_seq, below)
    wire [17:0] pf_rd_addr;                       // const-prefetch byte addr
    assign o_wbuf_en          = pe_read | pf_rd_en;
    assign o_wbuf_is_prefetch = pf_rd_en & ~pe_read;
    // PE read drives the real-stream weight address (imag is a parallel bank read in HW; exposed
    // as o_wi_byte_addr). Prefetch uses ce_prefetch_seq's word address when granted (RUN/PREFETCH
    // are different phases, so pe_read and pf_rd_en are never both asserted).
    assign o_wbuf_byte_addr   = pe_read ? o_wr_byte_addr : pf_rd_addr;

    ce_bank_split #(.BANKS(WBUF_BANKS)) u_split (
        .i_addr (o_wbuf_byte_addr),
        .o_row  (o_wbuf_row),
        .o_bank (o_wbuf_bank),
        .o_byte ()     // byte lane handled by the SRAM/wrapper read path, not routed here
    );

    // ===================================================================
    // (2) STATIC PEG-per-stream operand mux (§4).
    //     PEG0 = fr.xr   PEG1 = fi.xi   PEG2 = fr.xi   PEG3 = fi.xr
    //     weights x: PEG0/2 <- fr,  PEG1/3 <- fi
    //     acts    y: PEG0/3 <- xr,  PEG1/2 <- xi
    //     W4A8: INT4 weight in [3:0], e4m3 act in [7:0]; upper bits 0. (FP16 modes: TODO,
    //     widen code->16b; the route itself is width-agnostic.)
    // ===================================================================
    // valid-lane count for this chunk = min(32, ic - ic_chunk*32). Lanes >= valid are NOT real
    // input channels (partial last chunk, or ic/group < 32 in grouped decoder pw) -> zero their
    // ACT so the product is 0 and the FP16 tree isn't polluted by adjacent group/oc data. (The
    // encoder golden masked this implicitly by zero-padding UF; grouped pw exposes it for real.)
    logic [10:0] ic_rem;
    assign ic_rem = i_subop.ic - (i_ic_chunk << 5);
    wire [5:0] valid_lanes = (ic_rem >= 11'd32) ? 6'd32 : ic_rem[5:0];

    logic [NUM_PE_PG-1:0][15:0] fr, fi, xr, xi;
    genvar l;
    generate
        for (l = 0; l < NUM_PE_PG; l++) begin : g_form
            wire lane_ok = (l < valid_lanes);
            assign fr[l] = {8'd0, i_wr_codes[l]};     // INT4 -> [3:0] ([7:4]=0) ; FP8 -> [7:0]
            assign fi[l] = {8'd0, i_wi_codes[l]};
            assign xr[l] = lane_ok ? {8'd0, i_xr_codes[l]} : 16'd0;   // mask invalid lanes -> 0
            assign xi[l] = lane_ok ? {8'd0, i_xi_codes[l]} : 16'd0;
        end
    endgenerate

    always_comb begin
        o_x[0] = fr; o_y[0] = xr;     // PEG0 = fr.xr
        o_x[1] = fi; o_y[1] = xi;     // PEG1 = fi.xi
        o_x[2] = fr; o_y[2] = xi;     // PEG2 = fr.xi
        o_x[3] = fi; o_y[3] = xr;     // PEG3 = fi.xr
    end

    // ===================================================================
    // (4) DEALER-CODE SELECT — pre-fused WEIGHT_SCALE held in registers, per-PEG
    //     combinational extract gated by branches_collapse. NO runtime k arithmetic.
    //     code index = per-group PW scale: oc_idx*n_chunk + ic_chunk. (Attention scale
    //     is per-head -> different index, TODO with the attention path.) n_codes (split
    //     half-stride) is only consumed when branches_collapse=0 (decoder split, <=256).
    // ===================================================================
    logic [5:0] n_chunk_mc;
    assign n_chunk_mc = (i_subop.ic + 11'd31) >> 5;
    // conv PW per-group scale index: oc*n_chunk <= 512 codes (enc5 pw 128oc x 4grp). Attention
    // proj uses PER-HEAD scale (16 codes) -> a different index (TODO); compute full-width so the
    // assert below flags a non-conv sub-op here instead of silently wrapping.
    // weight-scale index by granularity (subop.scale_mode):
    //   PERGROUP (INT4 conv): per (oc,32-group) -> idx = oc*n_chunk + ic_chunk ; n = oc*n_chunk
    //   PERTENSOR (FP8 outlier): 1 code/branch  -> idx = 0 ; n = 1
    //   PERHEAD (attention proj): per head      -> idx = oc_idx/(oc/NUM_HEAD) ; n = NUM_HEAD
    logic [9:0] per_head_oc;                       // oc per head = oc / NUM_HEAD (NUM_HEAD=16 -> >>4)
    assign per_head_oc = i_subop.oc >> $clog2(NUM_HEAD);
    logic [4:0] head_shift;
    always_comb begin head_shift = '0; for (int k = 0; k < 10; k++) if (per_head_oc[k]) head_shift = k[4:0]; end
    wire [9:0] head_idx = i_oc_idx >> head_shift;
    logic [12:0] code_idx_full, n_codes_full;
    always_comb begin
        unique case (i_subop.scale_mode)
            SCALE_NONE:      begin code_idx_full = 13'd0;                n_codes_full = 13'd0;            end
            SCALE_PERTENSOR: begin code_idx_full = 13'd0;                n_codes_full = 13'd1;            end
            SCALE_PERHEAD:   begin code_idx_full = {3'd0, head_idx};     n_codes_full = 13'(NUM_HEAD);    end
            default:         begin code_idx_full = (i_oc_idx * n_chunk_mc) + i_ic_chunk;
                                   n_codes_full  = (i_subop.oc * n_chunk_mc);                            end
        endcase
        // mha_in_proj block-diagonal: 48 per-head codes [Q_h0..15,K,V], idx = oc/16 (the PERHEAD
        // formula breaks at oc=768 since oc/NUM_HEAD != 16); override here.
        if (i_subop.blockdiag) begin code_idx_full = {3'd0, i_oc_idx[9:4]}; n_codes_full = 13'd48; end
    end
    logic [8:0] code_idx;  logic [9:0] n_codes_b;
    assign code_idx  = code_idx_full[8:0];                     // <=511 for conv
    assign n_codes_b = n_codes_full[9:0];                      // <=512 for conv (split half-stride <=128)
    // full WEIGHT_SCALE block code count: collapse -> 1x ; split kreal||kimag -> 2x (decoder <=256)
    wire [9:0] block_codes = i_subop.branches_collapse ? n_codes_full[9:0] : {n_codes_full[8:0], 1'b0};
`ifndef SYNTHESIS
    always_comb if ((i_phase == P_RUN) && ((n_codes_full > 13'd512) || (code_idx_full > 13'd511)))
        $error("[mem_controller] scale idx/cnt %0d/%0d > conv max 512 (attention per-head scale = TODO)",
               code_idx_full, n_codes_full);
`endif

    // prefetch start = rising edge of the PREFETCH phase
    logic prefetch_d;
    always_ff @(posedge i_clk or negedge i_rst_n)
        if (!i_rst_n) prefetch_d <= 1'b0; else prefetch_d <= i_prefetch;
    wire prefetch_start = i_prefetch & ~prefetch_d;

    // PREFETCH sequencer: streams the r/i WEIGHT_SCALE blocks (12-bit unpack) -> ce_dealer_sel BURST fill
    localparam int PF_FILL_W = 5;
    wire        pf_fill_en, pf_fill_imag;
    wire [8:0]  pf_fill_idx;
    wire [2:0]  pf_fill_n;
    wire [PF_FILL_W*12-1:0] pf_fill_codes;
    ce_prefetch_seq #(.FILL_W(PF_FILL_W)) u_pf (
        .i_clk         (i_clk),       .i_rst_n      (i_rst_n),
        .i_start       (prefetch_start),
        .i_wscale_r    (i_subop.wscale_r), .i_wscale_i (i_subop.wscale_i),
        .i_n_codes     (block_codes),
        .o_rd_en       (pf_rd_en),    .o_rd_byte_addr (pf_rd_addr), .i_rd_data (i_wbuf_const_rdata),
        .o_fill_en     (pf_fill_en),  .o_fill_imag  (pf_fill_imag),
        .o_fill_idx    (pf_fill_idx), .o_fill_n     (pf_fill_n),    .o_fill_codes (pf_fill_codes),
        .o_done        (o_prefetch_done)
    );

    ce_dealer_sel #(.MAXC(512), .FILL_W(PF_FILL_W)) u_dsel (
        .i_clk         (i_clk),
        .i_fill_en     (pf_fill_en),
        .i_fill_imag   (pf_fill_imag),
        .i_fill_idx    (pf_fill_idx),
        .i_fill_n      (pf_fill_n),
        .i_fill_codes  (pf_fill_codes),
        .i_code_idx    (code_idx),
        .i_n_codes     (n_codes_b[8:0]),         // split half-stride <=128 (collapse: unused)
        .i_collapse    (i_subop.branches_collapse),
        .o_dealer_apot (o_dealer_apot)
    );

endmodule

// =======================================================================
// ce_bank_split / ce_dealer_sel / ce_prefetch_seq — moved here from their own files
// (file consolidation 2026-06-10). Still separate instantiated modules; mem_controller
// is their sole parent. (SV forward module refs within a file are legal.)
// =======================================================================
/*
    ce_bank_split.sv — flat WBUF/buffer byte address -> {row, bank, byte}.

    Exactly the §3 rule of SRAM_RTL_MEMORY_HANDOFF_EN.md. A buffer with BANKS banks of
    64-bit words has a row of BANKS*8 bytes, so:
        byte = A & 7
        bank = (A >> 3) & (BANKS-1)
        row  = A >> (3 + log2(BANKS))
    BANKS=16 -> WBUF (row=A>>7); BANKS=8 -> UFBUF (>>6); BANKS=4 -> KV/SKIP (>>5).
    Combinational. House style: instantiated module (not a function).
*/
`timescale 1ns/10ps
module ce_bank_split #(
    parameter int BANKS = 16
)(
    input  logic [17:0]            i_addr,
    output logic [17:0]            o_row,
    output logic [$clog2(BANKS)-1:0] o_bank,
    output logic [2:0]            o_byte
);
    localparam int BW = $clog2(BANKS);
    assign o_byte = i_addr[2:0];
    assign o_bank = i_addr[3 +: BW];        // (A >> 3) & (BANKS-1)
    assign o_row  = i_addr >> (3 + BW);     // A / (BANKS*8)
endmodule
/*
    ce_dealer_sel.sv — per-PEG dealer-code SELECT (Step 2-3 dealer path).

    The whole WEIGHT_SCALE block (r and i tensors) is held in a register array, loaded
    one 12-bit code/cycle during PREFETCH; the 4 per-PEG dealer codes are extracted
    COMBINATIONALLY during RUN. Descale is PRE-FUSED (build_sram_images.py folds
    activation-k into the APoT a/b exponents of WEIGHT_SCALE), so there is NO runtime k
    arithmetic / fuse unit — only a SELECT gated by branches_collapse:

      branches_collapse=1 (shared-k, 1 code/group):
        PEG0,PEG2 <- scale_r[idx]      PEG1,PEG3 <- scale_i[idx]          (2 distinct codes)
      branches_collapse=0 (split, block = kreal[0..n-1] || kimag[n..2n-1]):
        PEG0 <- scale_r[idx]           (r/kreal)   PEG2 <- scale_r[idx+n]  (r/kimag)
        PEG1 <- scale_i[idx+n]         (i/kimag)   PEG3 <- scale_i[idx]    (i/kreal)

    dealer.sv is UNCHANGED (it apot6a_decode's whatever 12-bit code arrives). The complex
    PEG-per-stream mapping (PEG0=fr.xr, PEG1=fi.xi, PEG2=fr.xi, PEG3=fi.xr) is why kreal/kimag
    pair this way (kreal/kimag = static per-PEG slots, 雷⑤). NUM_PEG=4 fixed (complex datapath).

    SAFETY INVARIANT: the loaded code is FULL fused descale iff pe_pair.i_preshift=0; for the
    12 FP8-outlier tensors it must be RESIDUAL (full-preshift) iff i_preshift!=0 — flip together.
*/
`timescale 1ns/10ps
module ce_dealer_sel
    import Param_def::*;
#(
    parameter int MAXC   = 512,  // max WEIGHT_SCALE codes per tensor (enc5 pw: 128 oc x 4 grp = 512)
    parameter int FILL_W = 5     // burst fill width (codes/cycle) = 64b WBUF word / 12b code
)(
    input  logic i_clk,

    // FILL (PREFETCH): a BURST of up to FILL_W 12-bit codes/cycle into consecutive indices of the r or i
    // block (i_fill_codes packs them LSB-first; i_fill_n = how many are valid, 1..FILL_W). The streaming
    // ce_prefetch_seq drains a 64-bit WBUF word as 5 codes/cycle -> matches WBUF read bandwidth.
    input  logic                    i_fill_en,
    input  logic                    i_fill_imag,    // 0 = r block, 1 = i block
    input  logic [$clog2(MAXC)-1:0] i_fill_idx,     // base index
    input  logic [2:0]              i_fill_n,        // valid codes in this burst (1..FILL_W)
    input  logic [FILL_W*12-1:0]    i_fill_codes,    // FILL_W codes, LSB-first

    // SELECT (RUN): combinational
    input  logic [$clog2(MAXC)-1:0] i_code_idx,     // oc_idx*n_chunk + ic_chunk (per-group PW)
    input  logic [$clog2(MAXC)-1:0] i_n_codes,      // codes per branch (split half-stride)
    input  logic                    i_collapse,     // subop_t.branches_collapse
    output logic [NUM_PEG-1:0][11:0] o_dealer_apot
);
    logic [11:0] scale_r [0:MAXC-1];
    logic [11:0] scale_i [0:MAXC-1];

    always_ff @(posedge i_clk) if (i_fill_en) begin
        for (int j = 0; j < FILL_W; j++) if (j < int'(i_fill_n)) begin
            if (i_fill_imag) scale_i[i_fill_idx + j[$clog2(MAXC)-1:0]] <= i_fill_codes[j*12 +: 12];
            else             scale_r[i_fill_idx + j[$clog2(MAXC)-1:0]] <= i_fill_codes[j*12 +: 12];
        end
    end

    logic [$clog2(MAXC)-1:0] idx_hi;
    assign idx_hi = i_code_idx + i_n_codes;          // kimag half (split only)

    always_comb begin
        if (i_collapse) begin
            o_dealer_apot[0] = scale_r[i_code_idx];
            o_dealer_apot[2] = scale_r[i_code_idx];
            o_dealer_apot[1] = scale_i[i_code_idx];
            o_dealer_apot[3] = scale_i[i_code_idx];
        end
        else begin
            o_dealer_apot[0] = scale_r[i_code_idx];  // r / kreal
            o_dealer_apot[2] = scale_r[idx_hi];      // r / kimag
            o_dealer_apot[1] = scale_i[idx_hi];      // i / kimag
            o_dealer_apot[3] = scale_i[i_code_idx];  // i / kreal
        end
    end
endmodule
/*
    ce_prefetch_seq.sv — STREAMING PREFETCH sequencer for the dealer-code fill (Step 2-3).

    On i_start (entering PREFETCH for a sub-op) it streams the two WEIGHT_SCALE blocks (r then i) out of
    WBUF and unpacks the packed 12-bit APoT codes into ce_dealer_sel, then asserts o_done.

    STREAMING (cycle-opt v2): a 64-bit WBUF word holds ~5 codes. Each cycle this FSM both (a) issues the
    NEXT word read (read-ahead, 1 word/cy, throttled by buffer room) AND (b) drains up to FILL_W(=5) codes
    from the running bit buffer into ce_dealer_sel's burst fill. Read latency hides behind the emit, so the
    old 3-cy/word read bubble (S_RD/WAIT/CAP) is gone — prefetch ≈ word-reads ≈ N/5 per block, not N.

    Why an unpacker: WEIGHT_SCALE blocks are BYTE-packed 12-bit codes, NOT word-aligned (e.g. enc0.pw_r @
    byte 241668 = word 30208, byte 4). WBUF is read as 8-byte little-endian words; we read the 8-aligned
    word containing the block start, skip (base&7)*8 bits of the first word, then drain 12-bit codes LSB-first.

    Read timing: o_rd_en + o_rd_byte_addr at T -> i_rd_data valid at T+1 (WBUF const-read latency 1; the
    ce_engine const path is enable-gated and supports back-to-back 1-word/cy reads). rd_pending tracks the
    in-flight read so the returned word is appended exactly once.
    Code count per block = i_n_codes (mem_controller: oc*n_chunk*(collapse?1:2); split block = 2n).
*/
`timescale 1ns/10ps
module ce_prefetch_seq #(
    parameter int FILL_W = 5                  // codes/cycle burst (64b word / 12b code)
)(
    input  logic        i_clk,
    input  logic        i_rst_n,

    input  logic        i_start,             // pulse: begin prefetch for the current sub-op
    input  logic [17:0] i_wscale_r,          // byte offset of the r WEIGHT_SCALE block
    input  logic [17:0] i_wscale_i,          // byte offset of the i WEIGHT_SCALE block
    input  logic [9:0]  i_n_codes,           // codes per block (full block, incl. kimag half if split)

    // WBUF word read (8-aligned byte addr; data 1-cycle later)
    output logic        o_rd_en,
    output logic [17:0] o_rd_byte_addr,
    input  logic [63:0] i_rd_data,

    // ce_dealer_sel BURST fill (up to FILL_W 12-bit codes/cycle, LSB-first)
    output logic        o_fill_en,
    output logic        o_fill_imag,         // 0 = r block, 1 = i block
    output logic [8:0]  o_fill_idx,          // base index
    output logic [2:0]  o_fill_n,            // valid codes this cycle (1..FILL_W)
    output logic [FILL_W*12-1:0] o_fill_codes,

    output logic        o_done
);
    localparam int BW = 256;                  // bit-buffer width (holds leftover + one 64b word, w/ margin)
    typedef enum logic [1:0] {S_IDLE, S_STREAM, S_DONE} state_t;
    state_t state;

    logic        blk;                         // 0 = r, 1 = i
    logic [17:0] rd_addr;                     // 8-aligned byte addr of the next word to read
    logic [7:0]  words_left;                  // words still to issue for this block
    logic        first;                       // next captured word is the block's first (apply start skip)
    logic [5:0]  start_bits;                  // (block base & 7)*8 — bits to skip in the first word
    logic [9:0]  code_idx;                    // codes emitted in current block
    logic [BW-1:0] bit_buf;                   // running LSB-first code buffer (valid bits = bit_cnt)
    logic [8:0]  bit_cnt;
    logic        rd_pending;                  // a read was issued last cycle -> i_rd_data valid now

    function automatic logic [17:0] align8(input logic [17:0] a); return {a[17:3], 3'b000}; endfunction
    // words spanning a block at byte offset `off` with i_n_codes 12-bit codes = ceil((base&7)*8 + n*12)/64)
    function automatic logic [7:0] words_of(input logic [17:0] off);
        logic [13:0] bits;
        bits = {8'd0, off[2:0], 3'b000} + (14'(i_n_codes) * 14'd12);
        return (bits + 14'd63) >> 6;
    endfunction

    // ---- combinational: append the in-flight word, then compute the burst emit ----
    logic [BW-1:0] new_data;  logic [8:0] new_bits;
    always_comb begin
        new_data = '0; new_bits = '0;
        if (rd_pending) begin
            if (first) begin new_data = ({{(BW-64){1'b0}}, i_rd_data} >> start_bits); new_bits = 9'd64 - {3'd0, start_bits}; end
            else        begin new_data = {{(BW-64){1'b0}}, i_rd_data};                 new_bits = 9'd64; end
        end
    end
    wire [BW-1:0] buf_m = bit_buf | (new_data << bit_cnt);     // merge in-flight word at bit_cnt
    wire [8:0]    cnt_m = bit_cnt + new_bits;
    wire [9:0]    rem   = i_n_codes - code_idx;
    logic [2:0]   navail;
    always_comb begin
        if      (cnt_m >= 9'd60) navail = 3'd5;
        else if (cnt_m >= 9'd48) navail = 3'd4;
        else if (cnt_m >= 9'd36) navail = 3'd3;
        else if (cnt_m >= 9'd24) navail = 3'd2;
        else if (cnt_m >= 9'd12) navail = 3'd1;
        else                     navail = 3'd0;
    end
    wire [2:0] emit_n    = (rem < {7'd0, navail}) ? rem[2:0] : navail;   // min(FILL_W slots, codes left)
    wire [8:0] emit_bits = 9'd12 * {6'd0, emit_n};
    wire [9:0] code_next = code_idx + {7'd0, emit_n};
    wire       block_done = (code_next >= i_n_codes);
    // read-ahead: issue next word when more remain, buffer has room, and the block isn't finishing now.
    // gate uses only registered state (+ rd_pending/first counts) so o_rd_en has no i_rd_data data path.
    wire do_read = (state == S_STREAM) && (words_left != 8'd0) && (cnt_m <= 9'd144) && !block_done;

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state<=S_IDLE; blk<=1'b0; rd_addr<='0; words_left<='0; first<=1'b1; start_bits<='0;
            code_idx<='0; bit_buf<='0; bit_cnt<='0; rd_pending<=1'b0; o_fill_en<=1'b0;
        end
        else begin
            o_fill_en <= 1'b0;
            unique case (state)
                S_IDLE: if (i_start) begin
                    blk<=1'b0; rd_addr<=align8(i_wscale_r); start_bits<={i_wscale_r[2:0],3'b000};
                    words_left<=words_of(i_wscale_r); first<=1'b1; code_idx<='0;
                    bit_buf<='0; bit_cnt<='0; rd_pending<=1'b0; state<=S_STREAM;
                end

                S_STREAM: begin
                    // (a) emit burst
                    if (emit_n != 3'd0) begin
                        o_fill_en    <= 1'b1;
                        o_fill_imag  <= blk;
                        o_fill_idx   <= code_idx[8:0];
                        o_fill_n     <= emit_n;
                        o_fill_codes <= buf_m[FILL_W*12-1:0];
                    end
                    // (b) advance the running buffer + read-ahead bookkeeping
                    bit_buf    <= buf_m >> emit_bits;
                    bit_cnt    <= cnt_m - emit_bits;
                    code_idx   <= code_next;
                    if (first && rd_pending) first <= 1'b0;
                    rd_pending <= do_read;
                    if (do_read) begin rd_addr <= rd_addr + 18'd8; words_left <= words_left - 8'd1; end
                    // (c) block transition (overrides the bookkeeping above)
                    if (block_done) begin
                        if (!blk) begin                     // r done -> start i block
                            blk<=1'b1; rd_addr<=align8(i_wscale_i); start_bits<={i_wscale_i[2:0],3'b000};
                            words_left<=words_of(i_wscale_i); first<=1'b1; code_idx<='0;
                            bit_buf<='0; bit_cnt<='0; rd_pending<=1'b0; state<=S_STREAM;
                        end
                        else state <= S_DONE;
                    end
                end

                S_DONE: state <= S_IDLE;
                default: state <= S_IDLE;
            endcase
        end
    end

    assign o_rd_en        = do_read;
    assign o_rd_byte_addr = rd_addr;
    assign o_done         = (state == S_DONE);
endmodule
