/*
    tb_inst_decoder_smoke.sv — self-checking smoke for the CE sub-op sequencer (Step 2-2).

    Not a golden test: it drives the prefetch/drain handshakes combinationally (1-cycle each)
    so the FSM walks ALL N_SUBOP sub-ops, and checks at runtime:
      - the latched descriptor o_subop === ce_subop_rom(o_subop_idx) during RUN (FETCH latched ok)
      - o_oacc_en == 1 every RUN cycle; o_oacc_clr only on (ic_chunk==0 && tap==0)
      - subop_idx advances 0..N_SUBOP-1 and the FSM reaches P_DONE
      - no X on the key control outputs while busy
    Pass prints: "[TB] ALL PASS, checked N sub-ops".
*/
`timescale 1ns/10ps
module tb_inst_decoder_smoke;
    import Param_def::*;
    import ce_subop_pkg::*;

    // phase codes (mirror inst_decoder localparams)
    localparam logic [2:0] P_IDLE=3'd0, P_FETCH=3'd1, P_PREFETCH=3'd2,
                           P_RUN=3'd3, P_DRAIN=3'd4, P_NEXT=3'd5, P_DONE=3'd6;

    logic i_clk, i_rst_n, i_start, i_prefetch_done, i_drain_done;
    logic o_busy, o_done;
    subop_t o_subop;
    logic [6:0] o_subop_idx;
    logic [2:0] o_phase;
    logic [9:0] o_oc_idx;
    logic [3:0] o_ic_chunk;
    logic [2:0] o_tap;
    CE_LANE_mode o_pe_mode;
    logic [1:0] o_ppu_mode;
    logic [NUM_PEG-1:0][7:0] o_preshift;
    logic o_oacc_clr, o_oacc_en, o_ppu_go, o_prefetch, o_cache_half;

    inst_decoder u_dut (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_start(i_start),
        .i_prefetch_done(i_prefetch_done), .i_drain_done(i_drain_done),
        .o_busy(o_busy), .o_done(o_done),
        .o_subop(o_subop), .o_subop_idx(o_subop_idx), .o_phase(o_phase),
        .o_oc_idx(o_oc_idx), .o_ic_chunk(o_ic_chunk), .o_tap(o_tap),
        .o_pe_mode(o_pe_mode), .o_ppu_mode(o_ppu_mode), .o_preshift(o_preshift),
        .o_oacc_clr(o_oacc_clr), .o_oacc_en(o_oacc_en), .o_ppu_go(o_ppu_go),
        .o_prefetch(o_prefetch), .o_cache_half(o_cache_half)
    );

    // 1-cycle handshakes so PREFETCH/DRAIN each take one cycle and the walk proceeds
    assign i_prefetch_done = (o_phase == P_PREFETCH);
    assign i_drain_done    = (o_phase == P_DRAIN);

    initial i_clk = 0;
    always #5 i_clk = ~i_clk;

    int err, runs, cyc;
    logic [6:0] max_idx;

    // runtime checks during the walk (all-blocking: TB idiom, avoids B/NB-mix warnings)
    always @(posedge i_clk) if (i_rst_n) begin
        cyc = cyc + 1;
        if (o_phase == P_RUN) begin
            runs = runs + 1;
            if (o_subop !== ce_subop_rom(o_subop_idx)) begin
                err = err + 1; $display("[chk] idx %0d: o_subop != ROM", o_subop_idx);
            end
            if (o_oacc_en !== 1'b1) begin err = err + 1; $display("[chk] idx %0d: oacc_en not 1 in RUN", o_subop_idx); end
            if (((o_ic_chunk==0) && (o_tap==0)) !== o_oacc_clr) begin
                err = err + 1; $display("[chk] idx %0d: oacc_clr mismatch (ic=%0d tap=%0d clr=%b)", o_subop_idx, o_ic_chunk, o_tap, o_oacc_clr);
            end
        end
        if (o_busy && (^{o_oacc_en, o_oacc_clr, o_prefetch} === 1'bx)) begin
            err = err + 1; $display("[chk] X on control output while busy @cyc %0d", cyc);
        end
        if (o_subop_idx > max_idx) max_idx = o_subop_idx;
    end

    initial begin
        i_rst_n = 0; i_start = 0; err = 0; runs = 0; cyc = 0; max_idx = 0;
        repeat (3) @(posedge i_clk); #1 i_rst_n = 1;
        @(posedge i_clk); #1 i_start = 1;          // kick off the walk
        @(posedge i_clk); #1 i_start = 0;

        // run until done, with a watchdog
        while (!o_done && cyc < 2_000_000) @(posedge i_clk);

        $display("==========================================");
        $display("tb_inst_decoder_smoke: walked sub-ops, RUN cycles=%0d, max idx=%0d / %0d", runs, max_idx, N_SUBOP-1);
        $display("  done=%b err=%0d cyc=%0d", o_done, err, cyc);
        $display("==========================================");
        if (o_done && (err == 0) && (max_idx == N_SUBOP-1))
            $display("[TB] ALL PASS, checked %0d sub-ops", N_SUBOP);
        else
            $fatal(1, "[TB] *** FAIL: done=%b err=%0d max_idx=%0d", o_done, err, max_idx);
        $finish;
    end
endmodule
