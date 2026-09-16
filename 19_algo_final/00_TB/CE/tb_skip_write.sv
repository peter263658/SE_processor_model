/*
    tb_skip_write.sv — verify the encoder skip-WRITE capture + the STFT spec region (T2 #8).

    Phase 1 (skip-write MECHANISM, self-consistent vs the DUT's own output stream):
      Load the real image + the enc3 input act into UF (nonzero data for every encoder layer with
      ic<=32; enc4/5 read 32 nonzero + zero-pad), run the sub-op walk through the encoder (0..11).
      Each encoder pw output (idx 1,3,5,7,9,11) is live-captured by ce_engine's skip-write into
      SKIPBUF. The TB MIRRORS the contract independently: it samples o_re/o_im (and the pre-cast
      fp16 through its OWN fp16_scale2+fp16_to_e4m3 pair for the enc5->dec0 CAST mode) at the same
      o_result_valid alignment, applies the same hit/pos selection (XMR call into the generated
      ce_skip_rom), and finally compares ALL 188 selected positions x {re,im} halves of skip_mem.
      Wrong position/timing/half => expected entry left X or mismatched => caught. (Real post-BN
      VALUES land when the conv BN live-feed is wired; this proves the capture dataflow.)

    Phase 2 (STFT spec region): write the golden Q2.14 spectrum (mask_apply_golden/L, frames t=0
      and t*=351) through the i_spec_wr port (ear 0/1), read back per bin (o_spec_*) -> must equal
      the int16 codes exactly; then drive the TB-instantiated mask_apply with the stored spec +
      golden mask and check the masked output bit-exact vs masked golden (the dual-role end use).
*/
`timescale 1ns/10ps
import Param_def::*;

module tb_skip_write;
    localparam int WBUF_BYTES=262144, RESULT_LAT=4, NSPEC=25536;  // 64 bins x 399 frames (bin-major)
    logic i_clk=0, i_rst_n; always #5 i_clk=~i_clk;
    logic i_start, o_busy, o_done, o_rv; logic [15:0] o_re, o_im;
    logic        spec_wr_en, spec_wr_ear, spec_rd_ear;
    logic [5:0]  spec_rd_bin;
    logic [63:0][15:0] spec_wr_re, spec_wr_im;
    logic [15:0] spec_o_re, spec_o_im;

    ce_engine u_eng(.i_clk(i_clk),.i_rst_n(i_rst_n),.i_start(i_start),.o_busy(o_busy),.o_done(o_done),
                    .o_re(o_re),.o_im(o_im),.o_result_valid(o_rv),
                    .i_spec_wr_en(spec_wr_en),.i_spec_wr_ear(spec_wr_ear),
                    .i_spec_wr_re(spec_wr_re),.i_spec_wr_im(spec_wr_im),
                    .i_spec_rd_ear(spec_rd_ear),.i_spec_rd_bin(spec_rd_bin),
                    .o_spec_re(spec_o_re),.o_spec_im(spec_o_im));

    // TB-side mirror of the dec0 CAST (independent instances of the verified primitives),
    // driven combinationally from the DUT's pre-cast tap (same pipe stage as o_re).
    logic [15:0] mir_sc_re, mir_sc_im; logic [7:0] mir_cc_re, mir_cc_im;
    fp16_scale2  u_msc_re (.i_x(u_eng.pre_re), .i_d(-8'sd1), .o_y(mir_sc_re));
    fp16_scale2  u_msc_im (.i_x(u_eng.pre_im), .i_d(-8'sd1), .o_y(mir_sc_im));
    fp16_to_e4m3 u_mse_re (.i_h(mir_sc_re), .o_e4m3(mir_cc_re));
    fp16_to_e4m3 u_mse_im (.i_h(mir_sc_im), .o_e4m3(mir_cc_im));

    // TB mask_apply instance (phase 2 end-use check)
    logic [15:0] ma_sr, ma_si, ma_mr, ma_mi, ma_ore, ma_oim;
    mask_apply u_ma (.i_spec_re(ma_sr), .i_spec_im(ma_si), .i_mask_re(ma_mr), .i_mask_im(ma_mi),
                     .o_masked_re(ma_ore), .o_masked_im(ma_oim));

    localparam string IMG ="../../10_algo/hw_patterns_final/sram_images/wbuf_unified_init.bin";
    localparam string ACT ="../../10_algo/hw_patterns_final/sram_images/golden/encoder_model_3_0_pointwise_r__0/act_t0.hex";
    localparam string MAG ="../../10_algo/hw_patterns_final/mask_apply_golden/L/";
    logic [7:0]  img [0:WBUF_BYTES-1], act [0:31];
    logic [15:0] g_sr [0:NSPEC-1], g_si [0:NSPEC-1], g_mr [0:NSPEC-1], g_mi [0:NSPEC-1];
    logic [15:0] g_gre [0:NSPEC-1], g_gim [0:NSPEC-1];
    // expected skip_mem mirror: [stage block byte] (q size 704 covers all 6 blocks)
    logic [7:0] exp_re [0:703], exp_im [0:703];
    integer fd,nrd,b;

    // dec-stage base per encoder stage (mirror of ce_engine skp_base)
    function automatic int base_of(input int s);
        case (s) 0: base_of=576; 1: base_of=448; 2: base_of=320; 3: base_of=192; 4: base_of=64; default: base_of=0; endcase
    endfunction

    initial begin
        i_rst_n=1; i_start=0; spec_wr_en=0; spec_wr_ear=0; spec_rd_ear=0; spec_rd_bin=0;
        for (b=0;b<704;b++) begin exp_re[b]=8'hxx; exp_im[b]=8'hxx; end
        fd=$fopen(IMG,"rb"); nrd=$fread(img,fd); $fclose(fd);
        for(b=0;b<WBUF_BYTES;b++) u_eng.wbuf_mem[b]=img[b];
        for(b=0;b<8192;b++) u_eng.uf_mem[b]=8'h00;
        $readmemh(ACT, act);
        for(b=0;b<32;b++) u_eng.uf_mem[b]=act[b];          // enc3 input: nonzero data for ic<=32 layers
        for(b=0;b<4096;b++) u_eng.skip_mem[b]=8'hxx;       // X canvas: un-written bytes stay X
        $readmemh({MAG,"spec_re.q214.hex"}, g_sr);  $readmemh({MAG,"spec_im.q214.hex"}, g_si);
        $readmemh({MAG,"mask_re.fp16.hex"}, g_mr);  $readmemh({MAG,"mask_im.fp16.hex"}, g_mi);
        $readmemh({MAG,"masked_re.q214.hex"}, g_gre); $readmemh({MAG,"masked_im.q214.hex"}, g_gim);
        $display("[TB] skip_write: image + enc3 act + mask_apply_golden(L) loaded");
        repeat(2)@(negedge i_clk); i_rst_n=0; repeat(2)@(negedge i_clk); i_rst_n=1; repeat(2)@(negedge i_clk);
        @(negedge i_clk); i_start=1; @(negedge i_clk); i_start=0;
    end

    // ===== TB mirror: sample at the same o_oc_complete + RESULT_LAT alignment as the DUT write =====
    wire        docc=u_eng.u_id.o_oc_complete;
    wire [9:0]  doc =u_eng.u_id.oc_idx;
    wire [6:0]  dsub=u_eng.u_id.subop_idx;
    logic [15:0] occ_sr; logic [9:0] oc_sr[0:7]; logic [6:0] sub_sr[0:7];
    always @(posedge i_clk) begin
        occ_sr<={occ_sr[14:0],(i_rst_n&docc)}; oc_sr[0]<=doc; sub_sr[0]<=dsub;
        for(int k=1;k<8;k++) begin oc_sr[k]<=oc_sr[k-1]; sub_sr[k]<=sub_sr[k-1]; end
        if(occ_sr[RESULT_LAT-1] && (sub_sr[RESULT_LAT-1] inside {7'd1,7'd3,7'd5,7'd7,7'd9,7'd11})) begin
            automatic int s   = sub_sr[RESULT_LAT-1] >> 1;
            automatic logic [7:0] hp = u_eng.ce_skip_hitpos(3'(s), oc_sr[RESULT_LAT-1]);
            if (hp[7]) begin
                exp_re[base_of(s)+hp[6:0]] <= (s==5) ? mir_cc_re : o_re[7:0];
                exp_im[base_of(s)+hp[6:0]] <= (s==5) ? mir_cc_im : o_im[7:0];
            end
        end
    end

    logic phase1_ok=0, phase2_ok=0;

    task automatic phase1_check; int er, ei, nw; begin
        er=0; ei=0; nw=0;
        for (int a=0;a<704;a++) begin
            if (exp_re[a]===8'hxx) begin
                // never written by the mirror: the DUT must not have written it either
                if (u_eng.skip_mem[a]!==8'hxx || u_eng.skip_mem[2048+a]!==8'hxx) begin nw++;
                    if (nw<=6) $display("[TB]   unexpected write @%0d (re %02h / im %02h)",a,u_eng.skip_mem[a],u_eng.skip_mem[2048+a]); end
            end else begin
                if (u_eng.skip_mem[a]      !== exp_re[a]) begin er++; if(er<=6) $display("[TB]   re @%0d got %02h exp %02h",a,u_eng.skip_mem[a],exp_re[a]); end
                if (u_eng.skip_mem[2048+a] !== exp_im[a]) begin ei++; if(ei<=6) $display("[TB]   im @%0d got %02h exp %02h",a,u_eng.skip_mem[2048+a],exp_im[a]); end
            end
        end
        $display("[TB] phase1 skip-write capture: re err=%0d im err=%0d stray=%0d (188 positions x2 halves)",er,ei,nw);
        if (er!=0||ei!=0||nw!=0) $display("[TB] PHASE1 FAIL");
        else $display("[TB] PHASE1 PASS (capture timing + hit/pos selection + dec0 CAST mode)");
        phase1_ok = (er==0&&ei==0&&nw==0);
    end endtask

    task automatic phase2_spec; int es, em, t; begin
        es=0; em=0;
        // two frames, two ears: write ear0=t0, ear1=t351; verify storage + mask_apply end-use
        for (int e=0;e<2;e++) begin
            t = (e==0) ? 0 : 351;
            for (int bn=0;bn<64;bn++) begin spec_wr_re[bn]=g_sr[bn*399+t]; spec_wr_im[bn]=g_si[bn*399+t]; end
            spec_wr_ear = e[0]; spec_wr_en = 1; @(negedge i_clk); spec_wr_en = 0; @(negedge i_clk);
        end
        for (int e=0;e<2;e++) begin
            t = (e==0) ? 0 : 351;
            spec_rd_ear = e[0];
            for (int bn=0;bn<64;bn++) begin
                spec_rd_bin = 6'(bn); #1;
                if (spec_o_re!==g_sr[bn*399+t] || spec_o_im!==g_si[bn*399+t]) begin es++;
                    if(es<=6) $display("[TB]   spec e%0d bin%0d got %04h/%04h exp %04h/%04h",e,bn,spec_o_re,spec_o_im,g_sr[bn*399+t],g_si[bn*399+t]); end
                // end-use: stored spec -> mask_apply with the golden mask -> masked golden
                ma_sr=spec_o_re; ma_si=spec_o_im; ma_mr=g_mr[bn*399+t]; ma_mi=g_mi[bn*399+t]; #1;
                if (ma_ore!==g_gre[bn*399+t] || ma_oim!==g_gim[bn*399+t]) begin em++;
                    if(em<=6) $display("[TB]   masked e%0d bin%0d got %04h/%04h exp %04h/%04h",e,bn,ma_ore,ma_oim,g_gre[bn*399+t],g_gim[bn*399+t]); end
            end
        end
        $display("[TB] phase2 spec region: store/fetch err=%0d  mask_apply-chain err=%0d (2 ears x 64 bins)",es,em);
        if (es!=0||em!=0) $display("[TB] PHASE2 FAIL");
        else $display("[TB] PHASE2 PASS (Q2.14 spec store/fetch + mask_apply end-use bit-exact)");
        phase2_ok = (es==0&&em==0);
    end endtask

    initial begin
        @(posedge i_rst_n); wait(i_start===1'b1);
        fork
            begin wait(u_eng.u_id.subop_idx > 7'd11); repeat(10)@(posedge i_clk); end
            begin repeat(400000)@(posedge i_clk); $display("[TB] TIMEOUT subop=%0d",u_eng.u_id.subop_idx); end
        join_any
        phase1_check();
        phase2_spec();
        if (phase1_ok && phase2_ok) $display("[TB] ALL PASS (encoder skip-write + STFT spec region)");
        else $display("[TB] FAIL p1=%0d p2=%0d", phase1_ok, phase2_ok);
        $finish;
    end
endmodule
