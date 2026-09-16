/*
    tb_kv_cache.sv — self-contained structural smoke for ce_kv_cache (4-path + ring).

    The deploy-reference numeric verification lives in tb_kv_ring.sv (window read vs qstar). This is a
    lightweight, file-free smoke that proves, with e4m3-exact fp16 patterns (so fp16->e4m3->fp16 is the
    identity, cast_k=descale_k=0):
      (1) the 32-frame ring: write a unique-per-slot pattern advancing ptr 0..31, then at ptr=31 read
          window entry local_idx d and confirm it == the pattern written to physical slot (31-d) mod 32;
      (2) 4-path separation: a DIFFERENT pattern written to the K_im path (pass=1) does not perturb the
          pass=0 (K_re) window read;
      (3) K/V split (sel) + per-head/per-dim addressing.
*/
`timescale 1ns/10ps
import Param_def::*;

module tb_kv_cache;
    localparam int NHEAD=16, NKEY=32, HD=16;

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
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_advance(i_advance), .o_ptr(o_ptr),
        .i_wr_en(i_wr_en), .i_wr_code_mode(i_wr_code_mode), .i_wr_pass(i_wr_pass), .i_wr_sel(i_wr_sel),
        .i_wr_head(i_wr_head), .i_wr_cast_k(i_wr_cast_k), .i_wr_fp16(i_wr_fp16),
        .i_rd_en(i_rd_en), .i_rd_pass(i_rd_pass), .i_rd_sel(i_rd_sel), .i_rd_head(i_rd_head),
        .i_rd_local_idx(i_rd_local_idx), .i_rd_descale_k(i_rd_descale_k),
        .o_rd_fp16(o_rd_fp16), .o_rd_valid(o_rd_valid)
    );

    // 8 e4m3-exact fp16 constants (1,2,0.5,3,4,0.25,6,8) -> fp16<->e4m3 round-trip is the identity.
    logic [15:0] e4m3_exact [0:7];
    initial begin e4m3_exact='{16'h3C00,16'h4000,16'h3800,16'h4200,16'h4400,16'h3400,16'h4600,16'h4800}; end
    // pass-0 pattern (depends on slot,head,dim,sel) and a DISTINCT pass-1 pattern (offset index)
    function automatic logic [15:0] pat0(input int s,input int h,input int d,input int sel);
        pat0 = e4m3_exact[(s+h+d+ (sel?5:0)) & 7]; endfunction
    function automatic logic [15:0] pat1(input int s,input int h,input int d);
        pat1 = e4m3_exact[(s+h+d+3) & 7]; endfunction

    integer h, s, d, dd, sel, err;

    initial begin
        i_advance=0; i_wr_en=0; i_wr_code_mode=0; i_wr_pass=0; i_wr_sel=0; i_wr_head=0; i_wr_cast_k=0;
        i_rd_en=0; i_rd_pass=0; i_rd_sel=0; i_rd_head=0; i_rd_local_idx=0; i_rd_descale_k=0;
        for (d=0;d<HD;d++) i_wr_fp16[d]=16'h0;
        i_rst_n=1; repeat(2)@(negedge i_clk); i_rst_n=0; repeat(2)@(negedge i_clk); i_rst_n=1; repeat(2)@(negedge i_clk);

        // ---- write frames 0..31 advancing ptr; pass-0 K/V (real pattern) + pass-1 K (decoy) ----
        for (s=0; s<NKEY; s++) begin
            for (sel=0; sel<2; sel++) for (h=0; h<NHEAD; h++) begin
                i_wr_en=1; i_wr_code_mode=0; i_wr_pass=0; i_wr_sel=sel[0]; i_wr_head=h[$clog2(NHEAD)-1:0]; i_wr_cast_k=0;
                for (d=0;d<HD;d++) i_wr_fp16[d]=pat0(s,h,d,sel);
                @(negedge i_clk);
            end
            for (h=0; h<NHEAD; h++) begin                       // pass-1 K decoy (must not perturb pass-0)
                i_wr_en=1; i_wr_code_mode=0; i_wr_pass=1; i_wr_sel=0; i_wr_head=h[$clog2(NHEAD)-1:0]; i_wr_cast_k=0;
                for (d=0;d<HD;d++) i_wr_fp16[d]=pat1(s,h,d);
                @(negedge i_clk);
            end
            i_wr_en=0;
            if (s!=NKEY-1) begin i_advance=1; @(negedge i_clk); i_advance=0; end
        end

        // ---- read window at ptr=31: local_idx d -> physical slot (31-d); expect pass-0 pattern there ----
        err=0;
        for (sel=0; sel<2; sel++) for (h=0; h<NHEAD; h++) for (dd=0; dd<NKEY; dd++) begin
            i_rd_en=1; i_rd_pass=0; i_rd_sel=sel[0]; i_rd_head=h[$clog2(NHEAD)-1:0];
            i_rd_local_idx=dd[$clog2(NKEY)-1:0]; i_rd_descale_k=0;
            @(negedge i_clk);
            if (o_rd_valid) for (d=0;d<HD;d++) begin
                logic [15:0] exp; exp = pat0(31-dd, h, d, sel);   // physical slot (31-dd) at ptr=31
                if (o_rd_fp16[d]!==exp) begin err++; if (err<=8) $display("[TB] %s h%0d localidx%0d d%0d got %04h exp %04h",
                                                                          sel?"V":"K",h,dd,d,o_rd_fp16[d],exp); end
            end else begin err++; $display("[TB] no rd_valid sel%0d h%0d localidx%0d",sel,h,dd); end
        end
        i_rd_en=0;
        $display("[TB] KV ring round-trip (4-path + (ptr-localidx) read + pass separation): %0d/%0d match",
                 2*NHEAD*NKEY*HD-err, 2*NHEAD*NKEY*HD);
        if (err==0) $display("[TB] ALL PASS (ce_kv_cache 4-path + ring smoke)");
        else        $display("[TB] FAIL err=%0d", err);
        $finish;
    end
    initial begin repeat(300000)@(posedge i_clk); $display("[TB] TIMEOUT"); $finish; end
endmodule
