/*
    tb_kv_ring.sv — focused multi-frame RING test for ce_kv_cache (task #4 verification).

    Exercises the 32-frame streaming ring pointer + the (ptr - local_idx) mod 32 window read + the
    4-path pass addressing + the x 2^-5 descale, against the DEPLOY reference (kv_ring_reference +
    mha_module_io qstar windows). This is the test the single-frame TB-injected #6 chain never touched:
    it advances the ring pointer across 32 frames and reads the window at the query frame.

    Phase 1 (consultant recipe, Item 4):
      stream-write frames 320..351 into the ring (ptr 0->31), one frame per pointer position, using the
      head-major e4m3 ring codes (kv_ring_reference/{K_re,V_re}/head_HH_ring_e4m3.hex, slot order:
      slot0=frame320 .. slot31=frame351). At the query frame 351 (ptr=31), read window entry local_idx
      d (d=0..31) -> physical slot (31-d) mod 32, dequantize (x 2^-5), and compare to the local_idx-order
      qstar_Kwindow / qstar_Vwindow (mha_module_io/real_pass/02_qk/head_HH/). All 16 heads, K and V.
      If the ring pointer + read translation are right, every entry matches (deploy-verified: the descaled
      ring slot (31-d) IS qstar_Kwindow[d]).  pass = re (pass A); the imag paths' separation is covered
      live by the tied 2-pass qk_chain (K_re/K_im/V_re/V_im all exercised there).

    Phase 2 (wraparound self-check): advance the pointer once more (31 -> 0, mod-32 wrap). Confirm o_ptr
      wrapped, and that reading local_idx 0 at ptr=0 returns the SAME data as reading local_idx 31 at
      ptr=31 (both resolve to physical slot 0) -- proving the pointer advance + read translation track the
      wrap. No extra golden needed.
*/
`timescale 1ns/10ps
import Param_def::*;

module tb_kv_ring;
    localparam int NHEAD=16, NKEY=32, HD=16, PERH=NKEY*HD;     // 512 codes/head
    localparam string RING = "../../10_algo/hw_patterns_final/kv_ring_reference/";
    localparam string QK   = "../../10_algo/hw_patterns_final/mha_module_io/real_pass/02_qk/";

    logic i_clk=0, i_rst_n; always #5 i_clk=~i_clk;
    logic                     i_advance;
    logic [$clog2(NKEY)-1:0]  o_ptr;
    logic                     i_wr_en, i_wr_code_mode, i_wr_pass, i_wr_sel;
    logic [$clog2(NHEAD)-1:0] i_wr_head;
    logic signed [7:0]        i_wr_cast_k;
    logic [HD-1:0][15:0]      i_wr_fp16;
    logic                     i_rd_en, i_rd_pass, i_rd_sel;
    logic [$clog2(NHEAD)-1:0] i_rd_head;
    logic [$clog2(NKEY)-1:0]  i_rd_local_idx;
    logic signed [7:0]        i_rd_descale_k;
    logic [HD-1:0][15:0]      o_rd_fp16;
    logic                     o_rd_valid;

    ce_kv_cache u_kv (
        .i_clk(i_clk), .i_rst_n(i_rst_n),
        .i_advance(i_advance), .o_ptr(o_ptr),
        .i_wr_en(i_wr_en), .i_wr_code_mode(i_wr_code_mode), .i_wr_pass(i_wr_pass), .i_wr_sel(i_wr_sel),
        .i_wr_head(i_wr_head), .i_wr_cast_k(i_wr_cast_k), .i_wr_fp16(i_wr_fp16),
        .i_rd_en(i_rd_en), .i_rd_pass(i_rd_pass), .i_rd_sel(i_rd_sel), .i_rd_head(i_rd_head),
        .i_rd_local_idx(i_rd_local_idx), .i_rd_descale_k(i_rd_descale_k),
        .o_rd_fp16(o_rd_fp16), .o_rd_valid(o_rd_valid)
    );

    // [head*512 + slot*16 + dim] e4m3 ring codes (slot order); [head*512 + localidx*16 + dim] qstar fp16
    logic [7:0]  ring_k [0:NHEAD*PERH-1], ring_v [0:NHEAD*PERH-1];
    logic [15:0] qw_k   [0:NHEAD*PERH-1], qw_v   [0:NHEAD*PERH-1];
    logic [15:0] slot0_at_p31 [0:HD-1];                          // capture for the phase-2 wrap check
    string fn;
    integer h, s, d, dd, errK, errV, errWrap;

    initial begin
        // ---- load all 16 heads ----
        for (h=0; h<NHEAD; h++) begin
            fn=$sformatf("%sK_re/head_%02d_ring_e4m3.hex", RING, h);                          $readmemh(fn, ring_k, h*PERH);
            fn=$sformatf("%sV_re/head_%02d_ring_e4m3.hex", RING, h);                          $readmemh(fn, ring_v, h*PERH);
            fn=$sformatf("%shead_%02d/qstar_Kwindow_32x16_localidx_fp16.hex", QK, h);         $readmemh(fn, qw_k, h*PERH);
            fn=$sformatf("%shead_%02d/qstar_Vwindow_32x16_localidx_fp16.hex", QK, h);         $readmemh(fn, qw_v, h*PERH);
        end

        i_advance=0; i_wr_en=0; i_wr_code_mode=1; i_wr_pass=0; i_wr_sel=0; i_wr_head=0; i_wr_cast_k=0;
        i_rd_en=0; i_rd_pass=0; i_rd_sel=0; i_rd_head=0; i_rd_local_idx=0; i_rd_descale_k=-8'sd5;
        for (d=0;d<HD;d++) i_wr_fp16[d]=16'h0;
        i_rst_n=1; repeat(2)@(negedge i_clk); i_rst_n=0; repeat(2)@(negedge i_clk); i_rst_n=1; repeat(2)@(negedge i_clk);
        if (o_ptr!==5'd0) $display("[TB] WARN ptr not 0 after reset: %0d", o_ptr);

        // ===== Phase 1: stream-write frames 320..351 (ptr 0->31), one frame per pointer position =====
        for (s=0; s<NKEY; s++) begin                            // ptr should equal s here
            for (h=0; h<NHEAD; h++) begin                       // K[h] for this frame -> slot ptr
                i_wr_en=1; i_wr_code_mode=1; i_wr_pass=0; i_wr_sel=0; i_wr_head=h[$clog2(NHEAD)-1:0];
                for (d=0;d<HD;d++) i_wr_fp16[d]={8'h00, ring_k[h*PERH + s*HD + d]};
                @(negedge i_clk);
            end
            for (h=0; h<NHEAD; h++) begin                       // V[h] for this frame -> slot ptr
                i_wr_en=1; i_wr_code_mode=1; i_wr_pass=0; i_wr_sel=1; i_wr_head=h[$clog2(NHEAD)-1:0];
                for (d=0;d<HD;d++) i_wr_fp16[d]={8'h00, ring_v[h*PERH + s*HD + d]};
                @(negedge i_clk);
            end
            i_wr_en=0;
            if (s != NKEY-1) begin                              // advance after each frame EXCEPT the query (351)
                if (o_ptr !== s[$clog2(NKEY)-1:0]) $display("[TB] WARN at frame %0d ptr=%0d (exp %0d)", 320+s, o_ptr, s);
                i_advance=1; @(negedge i_clk); i_advance=0;
            end
        end
        if (o_ptr!==5'd31) $display("[TB] WARN query ptr=%0d (exp 31)", o_ptr);
        $display("[TB] wrote frames 320..351, ptr=%0d at query frame 351 (16 heads x 32 frames x {K,V})", o_ptr);

        // ===== Phase 1 read: window at ptr=31, local_idx d -> slot (31-d) mod 32, descale x 2^-5 =====
        errK=0; errV=0;
        for (h=0; h<NHEAD; h++) begin
            for (dd=0; dd<NKEY; dd++) begin                     // K read
                i_rd_en=1; i_rd_pass=0; i_rd_sel=0; i_rd_head=h[$clog2(NHEAD)-1:0];
                i_rd_local_idx=dd[$clog2(NKEY)-1:0]; i_rd_descale_k=-8'sd5;
                @(negedge i_clk);
                if (dd==0 && h==0) for (d=0;d<HD;d++) slot0_at_p31[d]=16'hx;
                for (d=0;d<HD;d++) begin
                    if (o_rd_fp16[d] !== qw_k[h*PERH + dd*HD + d]) begin
                        errK++; if (errK<=8) $display("[TB] K h%0d localidx%0d d%0d got %04h exp %04h",
                                                      h, dd, d, o_rd_fp16[d], qw_k[h*PERH+dd*HD+d]);
                    end
                end
            end
            for (dd=0; dd<NKEY; dd++) begin                     // V read
                i_rd_en=1; i_rd_pass=0; i_rd_sel=1; i_rd_head=h[$clog2(NHEAD)-1:0];
                i_rd_local_idx=dd[$clog2(NKEY)-1:0]; i_rd_descale_k=-8'sd5;
                @(negedge i_clk);
                for (d=0;d<HD;d++) begin
                    if (o_rd_fp16[d] !== qw_v[h*PERH + dd*HD + d]) begin
                        errV++; if (errV<=8) $display("[TB] V h%0d localidx%0d d%0d got %04h exp %04h",
                                                      h, dd, d, o_rd_fp16[d], qw_v[h*PERH+dd*HD+d]);
                    end
                end
            end
        end
        i_rd_en=0;
        $display("[TB] window read @ptr=31 (K): %0d/%0d match   (V): %0d/%0d match",
                 NHEAD*NKEY*HD-errK, NHEAD*NKEY*HD, NHEAD*NKEY*HD-errV, NHEAD*NKEY*HD);

        // ===== Phase 2: capture head-0 local_idx 31 @ptr=31 (= physical slot 0), then wrap ptr 31->0 =====
        i_rd_en=1; i_rd_pass=0; i_rd_sel=0; i_rd_head=0; i_rd_local_idx=5'd31; i_rd_descale_k=-8'sd5;
        @(negedge i_clk);
        for (d=0;d<HD;d++) slot0_at_p31[d]=o_rd_fp16[d];        // physical slot 0 (frame 320) @ptr=31
        i_rd_en=0; @(negedge i_clk);
        i_advance=1; @(negedge i_clk); i_advance=0;             // ptr 31 -> 0 (mod-32 wrap)
        errWrap=0;
        if (o_ptr!==5'd0) begin errWrap++; $display("[TB] WRAP FAIL: ptr after 31+1 = %0d (exp 0)", o_ptr); end
        // at ptr=0, local_idx 0 -> physical slot 0 -> must equal slot0_at_p31 (frame 320, unchanged)
        i_rd_en=1; i_rd_pass=0; i_rd_sel=0; i_rd_head=0; i_rd_local_idx=5'd0; i_rd_descale_k=-8'sd5;
        @(negedge i_clk);
        for (d=0;d<HD;d++) if (o_rd_fp16[d]!==slot0_at_p31[d]) begin
            errWrap++; if (errWrap<=8) $display("[TB] WRAP read d%0d got %04h exp %04h",d,o_rd_fp16[d],slot0_at_p31[d]);
        end
        i_rd_en=0;
        $display("[TB] wraparound (ptr 31->0, slot0 read-through): %s (errWrap=%0d)", errWrap==0?"OK":"FAIL", errWrap);

        if (errK==0 && errV==0 && errWrap==0)
            $display("[TB] ALL PASS (KV ring: pointer advance 0..31 + (ptr-localidx)%%32 read + 4-path + descale -5 vs deploy)");
        else
            $display("[TB] FAIL errK=%0d errV=%0d errWrap=%0d", errK, errV, errWrap);
        $finish;
    end
    initial begin repeat(400000)@(posedge i_clk); $display("[TB] TIMEOUT"); $finish; end
endmodule
