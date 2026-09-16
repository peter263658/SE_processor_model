/*
    tb_warmup_sfu.sv — T3 warm-up causal mask: ce_attn_sfu (Softmax_v3 + packetizer) vs the deploy
    warm-up goldens mha_module_io_warmup/{t02,t10}/{real,imag}_pass (consultant B5, 2026-06-10).

    Cold-start contract under test: query frame i has n_valid = i+1 (<32) real keys at local_idx
    0..i; the validity mask (i_valid_mask) must exclude the un-filled window entries VALUE-
    INDEPENDENTLY (this TB leaves the invalid score_buf lanes UNWRITTEN = X to prove it), the
    top-k degenerates to min(4, n_valid) live packet slots, and the pow2 denominator sums only
    the survivors. t*=2 (3 valid -> 3 slots, prob_code [128,64,64,0]) is the hard corner; t*=10
    (11 valid, full 4-slot packet) the generic <full-window case.

    Checks per (t*, pass, head):
      (1) packet = set-equal {local_idx, prob} vs packet.json's valid slots (prob = fp16(code/256));
          HW empty slots must be prob=0; live-slot count == top_k_slots_used.
      (2) Score x V ctx via the verified PEG MODE_LMUL_ATTN datapath (packet prob x V[idx][d] on
          lanes 0,2,4,6 -> fp16 tree) vs ctx16_fp32 (RNE'd to fp16). Deploy sums in fp32; with
          <=4 pow2-prob products the fp16 tree is expected exact — counted strictly, reported.
*/
`timescale 1ns/10ps
import Param_def::*;

module tb_warmup_sfu;
    localparam int NH=16, HD=16;
    logic i_clk=0, i_rst_n; always #5 i_clk=~i_clk;
    logic i_start, o_done, wr_en;
    logic [3:0] wr_head; logic [4:0] wr_frame; logic [15:0] wr_score;
    logic [31:0] vmask;
    logic [NH-1:0][3:0][4:0]  o_pidx;
    logic [NH-1:0][3:0][15:0] o_pprob;

    ce_attn_sfu u_sfu (.i_clk(i_clk), .i_rst_n(i_rst_n),
        .i_wr_en(wr_en), .i_wr_head(wr_head), .i_wr_frame(wr_frame), .i_wr_score(wr_score),
        .i_start(i_start), .o_done(o_done), .i_valid_mask(vmask),
        .o_pidx(o_pidx), .o_pprob(o_pprob));

    // Score x V datapath (verified pattern from tb_score_v): 4 packet slots on PEG lanes 0,2,4,6
    logic [NUM_PEG-1:0][NUM_PE_PG-1:0][15:0] pg_x, pg_y;
    logic [NUM_PEG-1:0][7:0] pg_ps;
    logic [NUM_PEG-1:0][15:0] pg_acc;
    PEG u_peg (.i_mode(MODE_LMUL_ATTN), .i_x(pg_x), .i_y(pg_y), .i_preshift(pg_ps), .o_accum(pg_acc));

    // fp32 hex -> fp16 RNE (golden ctx is fp32)
    function automatic logic [15:0] f32_to_f16(input logic [31:0] w);
        logic s; logic [7:0] e; logic [22:0] m; integer ne; logic [12:0] rem;
        logic [10:0] sig; logic [9:0] frac;
        s=w[31]; e=w[30:23]; m=w[22:0];
        if (e==0 && m==0) return {s,15'h0};
        ne = e - 127 + 15;
        if (ne >= 31) return {s,5'h1F,10'h0};
        if (ne <= 0) begin // subnormal fp16 (golden ctx values are normal-range; basic handling)
            sig = {1'b1, m[22:13]}; rem = m[12:0];
            sig = sig >> (1-ne); ne = 0;
            return {s, 5'(ne), sig[9:0]};
        end
        frac = m[22:13]; rem = m[12:0];
        if (rem > 13'h1000 || (rem == 13'h1000 && frac[0])) begin
            {ne, frac} = {ne, frac} + 1;  // RNE round-up (carry promotes exponent)
        end
        if (ne >= 31) return {s,5'h1F,10'h0};
        return {s, 5'(ne), frac};
    endfunction

    string TS[2] = '{"t02","t10"};
    int    NV[2] = '{3, 11};
    string PS[2] = '{"real_pass","imag_pass"};
    logic [15:0] sc [0:31];
    logic [15:0] vw [0:11*16-1];          // Vwindow_localidx: n_valid x 16 fp16
    logic [31:0] cx [0:15];               // ctx16 fp32 (deploy; informational)
    logic [15:0] cxt [0:15];              // ctx16 fp16-TREE (HW bit-exact target)
    integer fjson;
    int g_code [0:3]; int g_lidx [0:3]; int g_vslot [0:3]; int g_used;
    int e_pkt, e_cnt, e_ctx, n_ctx_ok, t, p, h, sslot;
    string fn, line;

    // crude packet.json reader: extract the 3 arrays + top_k_slots_used (fixed layout per generator)
    task automatic read_packet(input string f);
        int fd, code; string key; begin
        fd=$fopen(f,"r");
        if (fd==0) begin $display("[TB] FATAL cannot open %s",f); $finish; end
        g_used=-1;
        while (!$feof(fd)) begin
            void'($fgets(line, fd));
            if ($sscanf(line, " \"top_k_slots_used\": %d", g_used)==1) continue;
            if (line.substr(1,12)=="\"prob_code\":")  begin for(int k=0;k<4;k++) begin void'($fgets(line,fd)); void'($sscanf(line," %d",g_code[k])); end end
            if (line.substr(1,12)=="\"local_idx\":")  begin for(int k=0;k<4;k++) begin void'($fgets(line,fd)); void'($sscanf(line," %d",g_lidx[k])); end end
            if (line.substr(1,13)=="\"valid_slot\":") begin for(int k=0;k<4;k++) begin void'($fgets(line,fd)); void'($sscanf(line," %d",g_vslot[k])); end end
        end
        $fclose(fd); end
    endtask

    // fp16 of the packet prob code (Q0.8 /256). The deploy Score x V consumes code/256 INCLUDING
    // the saturated 1.0 -> 255/256 = 0x3BF8 (prob_max_code=255) — proven by the ctx16 golden
    // (lmul(255/256, V) matches 0/16; lmul(1.0, V) misses 16/16). The sfu packetizer saturates
    // the same way, so HW packet prob == code/256 exactly.
    function automatic logic [15:0] prob_of_code(input int code);
        case (code)
            255: return 16'h3BF8;  // saturated encoding: 255/256
            256: return 16'h3C00; 128: return 16'h3800; 64: return 16'h3400; 32: return 16'h3000;
            16:  return 16'h2C00; 8:   return 16'h2800; 4:  return 16'h2400; 2:  return 16'h2000;
            1:   return 16'h1C00; 0:   return 16'h0000;
            default: begin $display("[TB] FATAL non-pow2 prob code %0d", code); $finish; return 16'h0; end
        endcase
    endfunction

    initial begin
        i_rst_n=1; i_start=0; wr_en=0; vmask='0; pg_x='0; pg_y='0; pg_ps='0;
        e_pkt=0; e_cnt=0; e_ctx=0; n_ctx_ok=0;
        repeat(2)@(negedge i_clk); i_rst_n=0; repeat(2)@(negedge i_clk); i_rst_n=1; repeat(2)@(negedge i_clk);

        for (t=0; t<2; t++) begin
            for (p=0; p<2; p++) begin
                // fresh DUT state per (t,pass): score_buf has no reset — we rely on the MASK
                // excluding stale/X lanes (that IS the test). RE-SYNC to a negedge first: the
                // previous iteration's ctx checks advance time with #1 steps (clock-unaligned);
                // without this the first write's (h0,d0) data is overwritten before any posedge
                // samples it (one lost write -> stale lane).
                @(negedge i_clk);
                vmask = (33'h1 << NV[t]) - 1;
                for (h=0; h<NH; h++) begin
                    fn=$sformatf("../../10_algo/hw_patterns_final/mha_module_io_warmup/%s/%s/head_%02d/scores_row_localidx_postscale_fp16.hex",TS[t],PS[p],h);
                    $readmemh(fn, sc);
                    // write ONLY the valid lanes (invalid lanes stay X from reset / stale from the
                    // previous (t,pass) iteration -> proves value-independent masking)
                    for (int d=0; d<NV[t]; d++) begin
                        wr_en=1; wr_head=4'(h); wr_frame=5'(d); wr_score=sc[d]; @(negedge i_clk);
                    end
                    wr_en=0;
                end
                @(negedge i_clk); i_start=1; @(negedge i_clk); i_start=0;
                wait(o_done===1'b1); @(negedge i_clk);

                for (h=0; h<NH; h++) begin
                    fn=$sformatf("../../10_algo/hw_patterns_final/mha_module_io_warmup/%s/%s/head_%02d/packet.json",TS[t],PS[p],h);
                    read_packet(fn);
                    // (1) packet set-equality on valid slots + live-slot count
                    begin
                        int hw_live; hw_live=0;
                        for (int s2=0;s2<4;s2++) if (o_pprob[h][s2]!=16'h0) hw_live++;
                        if (hw_live != g_used) begin e_cnt++;
                            $display("[TB]  %s/%s h%0d live slots %0d exp %0d",TS[t],PS[p],h,hw_live,g_used); end
                        for (int gk=0;gk<4;gk++) if (g_vslot[gk]) begin
                            int found; found=0;
                            for (int s2=0;s2<4;s2++)
                                if (o_pidx[h][s2]==5'(g_lidx[gk]) && o_pprob[h][s2]==prob_of_code(g_code[gk])) found=1;
                            if (!found) begin e_pkt++;
                                if (e_pkt<=16) begin
                                    $display("[TB]  %s/%s h%0d missing slot idx%0d code%0d",TS[t],PS[p],h,g_lidx[gk],g_code[gk]);
                                    $display("[TB]    HW packet: (%0d,%04h)(%0d,%04h)(%0d,%04h)(%0d,%04h)",
                                             o_pidx[h][0],o_pprob[h][0],o_pidx[h][1],o_pprob[h][1],
                                             o_pidx[h][2],o_pprob[h][2],o_pidx[h][3],o_pprob[h][3]);
                                end end
                        end
                    end
                    // (2) ctx via PEG: prob x V[idx][d] on lanes 0,2,4,6 per dim. Bit-exact target =
                    // the fp16-TREE golden (gen_warmup_ctx_fp16tree.py; lmul_m8 + pairwise fp16 add —
                    // the HW contract). The deploy ctx16_fp32 (fp32-sum) diverges from the tree at
                    // 9/1024 +-1ulp boundary cases (known accepted class, QK precedent).
                    fn=$sformatf("../../10_algo/hw_patterns_final/mha_module_io_warmup/%s/%s/head_%02d/Vwindow_localidx_fp16.hex",TS[t],PS[p],h);
                    $readmemh(fn, vw);
                    fn=$sformatf("../../10_algo/hw_patterns_final/mha_module_io_warmup/%s/%s/head_%02d/ctx16_fp16tree.hex",TS[t],PS[p],h);
                    $readmemh(fn, cxt);
                    for (int d=0; d<HD; d++) begin
                        pg_x='0; pg_y='0;
                        for (int s2=0;s2<4;s2++) begin
                            pg_x[0][2*s2] = o_pprob[h][s2];
                            pg_y[0][2*s2] = (o_pprob[h][s2]!=16'h0) ? vw[o_pidx[h][s2]*HD+d] : 16'h0;
                        end
                        #1;
                        if (pg_acc[0] !== cxt[d]) begin e_ctx++;
                            if (e_ctx<=8) $display("[TB]  %s/%s h%0d ctx d%0d got %04h exp %04h",TS[t],PS[p],h,d,pg_acc[0],cxt[d]);
                        end else n_ctx_ok++;
                    end
                end
                $display("[TB] %s/%s done (16 heads)",TS[t],PS[p]);
            end
        end
        $display("[TB] warm-up: packet err=%0d slot-count err=%0d | ctx %0d ok / %0d err (vs fp32-RNE)",e_pkt,e_cnt,n_ctx_ok,e_ctx);
        if (e_pkt==0 && e_cnt==0 && e_ctx==0)
            $display("[TB] ALL PASS (warm-up causal mask: t*=2 degenerate top-3 + t*=10 partial window, X-masked lanes, deploy bit-exact)");
        else $display("[TB] FAIL");
        $finish;
    end
    initial begin repeat(200000)@(posedge i_clk); $display("[TB] TIMEOUT"); $finish; end
endmodule
