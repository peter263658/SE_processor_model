/*
    inst_decoder.sv — CE W4A8 sub-op sequencer (Step 2-2).

    The ONLY "which layer / sub-op / phase" block. Walks the generated 68-entry
    descriptor ROM (ce_subop_rom.svh, from manifest.json) and, per sub-op, runs the
    output-channel x ic-chunk [x DW-tap] accumulate loop, driving:
      - the const PREFETCH window (-> mem_controller -> ppu_const_cache),
      - OACC clr/en during RUN,
      - the per-tensor preshift (overflow fix) + pe/ppu mode (-> CE_core / ce_top),
      - a ppu pulse + result-write strobe during DRAIN.
    mem_controller (Step 2-3) turns the exposed {subop, phase, counters} into the
    actual WBUF/UF/KV/SKIP addresses + the PEG-per-stream operand mux.

    SCOPE / what this does NOT decide (mem_controller + open items own these):
      * COMPLEX FUSION: a HW complex conv pass uses BOTH the _r and _i weight tensors
        on the 4 PEG (PEG0/2=wr, PEG1/3=wi). The ROM lists _r and _i as SEPARATE
        entries (correct per-tensor data); pairing two ROM entries into one 4-PEG pass
        (and the per-PEG preshift/dealer codes that then differ) is mem_controller's map.
        This sequencer walks entries one at a time and broadcasts one preshift to all PEG.
      * within-block dw->pw->BN chaining and attention expansion (QK/Softmax_v3/Score x V
        glue is still '[?]', CE_DATAPATH_DRAFT §9).
    Handshake driven: i_prefetch_done (const streamed), i_drain_done (pipe drained).

    Build deps (filelist): Param_def, Mem_def, then `+incdir+` the dir holding
    memory_map_pkg.svh is NOT needed here (ce_subop_rom.svh carries literal offsets).
*/
`timescale 1ns/10ps
module inst_decoder
    import Param_def::*;
    import Mem_def::*;
    import ce_subop_pkg::*;          // subop_t, PPU_*, N_SUBOP, ce_subop_rom()
(
    input  logic i_clk,
    input  logic i_rst_n,

    input  logic i_start,            // pulse: begin the full sub-op walk
    input  logic i_prefetch_done,    // mem_controller finished streaming this sub-op's const block
    input  logic i_drain_done,       // ce_top: pipeline drained + result written for the tile

    output logic o_busy,
    output logic o_done,             // all N_SUBOP sub-ops complete

    // ---- ppu mode-select localparams are exported so mem_controller/ce_top can compare ----
    output subop_t      o_subop,     // current descriptor (full)
    output logic [6:0]  o_subop_idx,

    output logic [2:0]  o_phase,     // P_* below (for mem_controller)
    output logic [9:0]  o_oc_idx,    // current output channel
    output logic [3:0]  o_ic_chunk,  // current ic chunk (0..n_chunk-1)
    output logic [2:0]  o_tap,       // current DW tap (0..F-1)

    // ---- direct CE_core / ce_top control ----
    output CE_LANE_mode o_pe_mode,                   // = o_subop.pe_mode (convenience)
    output logic [1:0]  o_ppu_mode,                  // = o_subop.ppu_mode
    output logic [NUM_PEG-1:0][7:0] o_preshift,      // per-tensor preshift, broadcast to all PEG
    output logic        o_oacc_clr,
    output logic        o_oacc_en,
    output logic        o_ppu_go,                    // pulse in DRAIN
    output logic        o_oc_complete,               // pulse: this oc's LAST accumulate is in RUN now
                                                     // (-> pipelined result strobe; one per output ch)

    // ---- const prefetch ----
    output logic        o_prefetch,                  // PREFETCH phase active
    output logic        o_cache_half,                // ping-pong half active for read (fill = !this)
    output logic        o_pass_sel                   // tied_mha 2-pass: 0=pass A (x.real), 1=pass B (x.imag)
);
    // PPU_* / subop_t / N_SUBOP / ce_subop_rom() come from ce_subop_pkg (imported above).

    // ----- phases -----
    localparam logic [2:0] P_IDLE=3'd0, P_FETCH=3'd1, P_PREFETCH=3'd2,
                           P_RUN=3'd3, P_DRAIN=3'd4, P_NEXT=3'd5, P_DONE=3'd6;
    logic [2:0] state, nstate;

    // ----- counters / latched descriptor -----
    logic [6:0]  subop_idx;
    subop_t      cur;
    logic [9:0]  oc_idx;
    logic [3:0]  ic_chunk;
    logic [2:0]  tap;
    logic        pass_sel;        // tied_mha: 0=pass A (x.real), 1=pass B (x.imag); attn block runs twice

    // ic chunks: W4A8/FP8 = 32 lanes/chunk (ceil(ic/32)); W4A16 = 16-bit operand mode -> 16 products/PEG
    // (even lanes) -> 16 ic/chunk (ceil(ic/16)). DW (ic=1) -> 1 chunk, F taps drive the loop. ic_chunk[3:0]
    // holds 0..15; W4A16 ic=256 -> 16 chunks (0..15), fits.
    logic [5:0] n_chunk;
    assign n_chunk = (cur.pe_mode == MODE_W4A16) ? ((cur.ic + 11'd15) >> 4)   // 16 ic/chunk
                                                 : ((cur.ic + 11'd31) >> 5);  // 32 ic/chunk
    wire        last_chunk = (ic_chunk == (n_chunk - 6'd1));
    wire        last_tap   = (tap == (cur.F - 3'd1));
    wire        last_oc    = (oc_idx == (cur.oc - 10'd1));
    wire        last_subop = (subop_idx == (N_SUBOP - 1));
    // tied_mha 2-pass: the internal-MHA block (mha_in_proj..mha_out_proj) runs twice — pass A on
    // x.real, pass B on x.imag — then ce_engine recombines (a-b, a+b) before output_proj(35). Loop
    // 34->31 ONCE at the block end (P_NEXT), toggling pass_sel. Each sub-op's inner oc/ic/tap loop is
    // unchanged (runs once per visit). subop_idx walks 30,31,32,33,34,(loop)31,32,33,34,35.
    localparam logic [6:0] SUBOP_ATTN_LO = 7'd31, SUBOP_ATTN_HI = 7'd34;

    // ===================== next-state =====================
    always_comb begin
        nstate = state;
        unique case (state)
            P_IDLE:     if (i_start)         nstate = P_FETCH;
            P_FETCH:                          nstate = P_PREFETCH;
            P_PREFETCH: if (i_prefetch_done) nstate = P_RUN;
            // continuous RUN across oc (pipelined results); DRAIN only to flush the LAST oc.
            P_RUN:      if (last_chunk && last_tap && last_oc) nstate = P_DRAIN;
            P_DRAIN:    if (i_drain_done)    nstate = P_NEXT;
            P_NEXT:                           nstate = last_subop ? P_DONE : P_FETCH;
            // multi-frame (T3): DONE re-arms on i_start (the frame-loop owner launches one walk
            // per query frame); o_done is a LEVEL that holds until the next start.
            P_DONE:     if (i_start)         nstate = P_FETCH;
            default:                          nstate = P_IDLE;
        endcase
    end

    // ===================== sequential =====================
    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state     <= P_IDLE;
            subop_idx <= '0;
            cur       <= '0;
            oc_idx    <= '0;
            ic_chunk  <= '0;
            tap       <= '0;
            pass_sel  <= 1'b0;
        end
        else begin
            state <= nstate;
            unique case (state)
                P_IDLE: if (i_start) begin subop_idx <= '0; pass_sel <= 1'b0; end
                P_DONE: if (i_start) begin subop_idx <= '0; pass_sel <= 1'b0; end  // re-arm (multi-frame)

                P_FETCH: begin
                    cur      <= ce_subop_rom(subop_idx);
                    oc_idx   <= '0;
                    ic_chunk <= '0;
                    tap      <= '0;
                end

                P_RUN: begin
                    // advance ic_chunk (inner) / tap (outer); at oc end, advance oc INLINE so RUN
                    // stays continuous (no per-oc DRAIN bubble). Last oc holds -> nstate=DRAIN.
                    if (last_chunk && last_tap) begin
                        if (!last_oc) begin
                            oc_idx   <= oc_idx + 10'd1;       // start next output channel next cycle
                            ic_chunk <= '0;
                            tap      <= '0;
                        end
                    end
                    else if (last_chunk) begin
                        ic_chunk <= '0;
                        tap      <= tap + 3'd1;
                    end
                    else ic_chunk <= ic_chunk + 4'd1;
                end

                P_DRAIN: ;   // flush the last oc's result; oc advance now happens in P_RUN

                P_NEXT: begin
                    // tied 2-pass loop-back: end of attn pass A (sub-op 34, pass_sel=0) -> sub-op 31 pass B.
                    // (sub-op 34 is never last_subop, so next-state P_NEXT->P_FETCH is unchanged.)
                    if (subop_idx == SUBOP_ATTN_HI && pass_sel == 1'b0) begin
                        subop_idx <= SUBOP_ATTN_LO;
                        pass_sel  <= 1'b1;
                    end
                    else if (!last_subop) subop_idx <= subop_idx + 7'd1;
                end

                default: ;
            endcase
        end
    end

    // ===================== outputs =====================
    assign o_busy      = (state != P_IDLE) && (state != P_DONE);
    assign o_done      = (state == P_DONE);
    assign o_subop     = cur;
    assign o_subop_idx = subop_idx;
    assign o_phase     = state;
    assign o_oc_idx    = oc_idx;
    assign o_ic_chunk  = ic_chunk;
    assign o_tap       = tap;

    assign o_pe_mode   = cur.pe_mode;
    assign o_ppu_mode  = cur.ppu_mode;

    // per-PEG preshift: PEG0/2 = real stream (fr·xr, fr·xi) -> preshift_r ;
    //                   PEG1/3 = imag stream (fi·xi, fi·xr) -> preshift_i .
    // o_preshift is UNSIGNED [7:0] to MATCH CE_core.i_preshift (two's-comp bits preserved;
    // CE_core applies $signed at use). NUM_PEG=4 is fixed for the complex datapath.
    always_comb begin
        o_preshift[0] = cur.preshift_r;
        o_preshift[2] = cur.preshift_r;
        o_preshift[1] = cur.preshift_i;
        o_preshift[3] = cur.preshift_i;
    end

    assign o_prefetch  = (state == P_PREFETCH);
    // ping-pong: read the half we filled for THIS sub-op; toggles per sub-op
    assign o_cache_half = subop_idx[0];
    assign o_pass_sel   = pass_sel;

    // OACC: en every RUN cycle; clr on the first accumulate (ic_chunk==0 && tap==0)
    assign o_oacc_en   = (state == P_RUN);
    assign o_oacc_clr  = (state == P_RUN) && (ic_chunk == '0) && (tap == '0);
    // ppu latches the descaled OACC once the pipe has settled in DRAIN
    assign o_ppu_go    = (state == P_DRAIN) && i_drain_done;
    // one pulse per output channel, at its LAST accumulate cycle -> pipelined result strobe
    assign o_oc_complete = (state == P_RUN) && last_chunk && last_tap;

    // ===================== build-time watch-items (algo-consultant) =====================
    // (1) preshift must be a non-positive pow2 down-shift; and currently 0 everywhere to
    //     match the FULL-descale SRAM images. A nonzero preshift REQUIRES the WBUF scale
    //     codes to be regenerated as residual (full - preshift) — change them together.
    // (3) (preshift, residual) pairing: enforced structurally (both live in one descriptor).
`ifndef SYNTHESIS
    initial begin
        subop_t d;
        for (int s = 0; s < N_SUBOP; s++) begin
            d = ce_subop_rom(s);
            if ((d.preshift_r > 0) || (d.preshift_i > 0))
                $warning("[inst_decoder] sub-op %0d preshift_r=%0d preshift_i=%0d : expected <=0 pow2 down-shift", s, d.preshift_r, d.preshift_i);
            if ((d.preshift_r != 0) || (d.preshift_i != 0))
                $display("[inst_decoder] NOTE sub-op %0d preshift!=0 : WBUF scale codes MUST be residual (full-preshift)", s);
            // precondition: mem_controller's ic_chunk[3:0] + PW byte-addr width assume ic<=256
            if (d.ic > 10'd256)
                $warning("[inst_decoder] sub-op %0d ic=%0d > 256 : violates mem_controller ic<=256 precondition", s, d.ic);
        end
    end
`endif
endmodule
