/*
    tb_multiframe.sv — T3 LIVE multi-frame: cold-start ring advance + warm-up mask end-to-end.

    Runs THREE query frames through the full sub-op walk (one i_start each), SAME input x every
    frame, KVBUF zero-initialized (= kvbuf_init_zero deploy state), pulsing i_advance after each
    frame (the engine-orchestrator contract: once per frame, at a frame boundary). No history
    preload — the ring fills LIVE from each frame's in_proj write-back.

    Identical inputs give closed-form expectations (no new goldens needed):
      frame 0: filled=0 -> n_valid=1. Sole survivor w=2^0, denom 2^0 -> prob 1.0 SATURATES to
               255/256 (0x3BF8). packet = {(idx0, 0x3BF8)}, live=1.
      frame 1: n_valid=2, scores d0==d1 (same K both slots) -> w=[2^0,2^0], wsum q8=512 (msb 9 <
               thr 725) -> denom 2^9 -> probs 0.5/0.5 (0x3800), idx {0,1}, live=2.
      frame 2: n_valid=3, three equal -> wsum=768 >= 725 -> denom 2^10 (GEOMETRIC nearest; the
               arithmetic next-bit rule would give 2^9) -> probs 0.25 x3 (0x3400), idx {0,1,2}.
    Plus the ring state: after frame f, ptr==f+1, filled==f+1, and K_re/V_re slots 0..f all hold
    the SAME k_code_hw/v_code_hw (the live in_proj write-back, one slot per frame).
    Both tied passes run per frame; the packet checks use the FINAL (pass-B) sfu state with the
    imag-pass closed forms (same equal-score logic); K_re checks cover pass A's path.
*/
`timescale 1ns/10ps
import Param_def::*;

module tb_multiframe;
    localparam int WBUF_BYTES=262144, IC=256, WS_OFF=256256;
    logic i_clk=0, i_rst_n; always #5 i_clk=~i_clk;
    logic i_start, o_busy, o_done, o_rv, i_advance; logic [15:0] o_re, o_im;
    ce_engine u_eng(.i_clk(i_clk),.i_rst_n(i_rst_n),.i_start(i_start),.o_busy(o_busy),.o_done(o_done),
                    .o_re(o_re),.o_im(o_im),.o_result_valid(o_rv),.i_advance(i_advance));

    localparam string GIN ="../../10_algo/hw_patterns_final/sram_images/golden/mha_in_proj_real/";
    localparam string GINI="../../10_algo/hw_patterns_final/sram_images/golden/mha_in_proj_imag/";
    localparam string IMG ="../../10_algo/hw_patterns_final/sram_images/wbuf_unified_init.bin";
    logic [7:0] img [0:WBUF_BYTES-1], fused [0:127], xc [0:IC-1], xc_im [0:IC-1];
    logic [7:0] kc [0:255], vc [0:255], kci [0:255], vci [0:255];
    logic [15:0] qd [0:255];      // golden q_deq (imag pass — q_mem final state)
    integer fd,nrd,b,err;

    // per-frame closed-form packet expectation (imag pass = final sfu state)
    logic [15:0] exp_prob [0:2];
    initial begin exp_prob[0]=16'h3BF8; exp_prob[1]=16'h3800; exp_prob[2]=16'h3400; end

    initial begin
        i_rst_n=1; i_start=0; i_advance=0; err=0;
        fd=$fopen(IMG,"rb"); nrd=$fread(img,fd); $fclose(fd);
        for(b=0;b<WBUF_BYTES;b++) u_eng.wbuf_mem[b]=img[b];
        $readmemh({GIN,"fused_apot.bytes.hex"}, fused);
        for(b=0;b<72;b++) u_eng.wbuf_mem[WS_OFF+b]=fused[b];
        for(b=0;b<8192;b++) u_eng.uf_mem[b]=8'h00;
        $readmemh({GIN, "x_code.hex"}, xc);   $readmemh({GINI,"x_code.hex"}, xc_im);
        for(b=0;b<IC;b++) begin u_eng.uf_mem[b]=xc[b]; u_eng.uf_mem[256+b]=xc_im[b]; end
        $readmemh({GIN, "k_code_hw.hex"}, kc);  $readmemh({GIN, "v_code_hw.hex"}, vc);
        $readmemh({GINI,"k_code_hw.hex"}, kci); $readmemh({GINI,"v_code_hw.hex"}, vci);
        $readmemh({GINI,"q_deq.hex"}, qd);
        for(b=0;b<32768;b++) u_eng.u_kv.kv_mem[b]=8'h00;   // = kvbuf_init_zero (cold start)
        $display("[TB] multiframe: image+fused+x(re,im)+K/V refs loaded; KV zero-init, ptr/filled=0");
        repeat(2)@(negedge i_clk); i_rst_n=0; repeat(2)@(negedge i_clk); i_rst_n=1; repeat(2)@(negedge i_clk);

        for (int f=0; f<3; f++) begin
            // re-load the frame input: UF is the PER-FRAME activation buffer — the walk itself
            // overwrites uf[0..255]/[512..767] with the recombine codes (output_proj input), so
            // each frame's owner (engine orchestrator / TB) rewrites the attention input slices,
            // INCLUDING zeroing [512..767]: that is mha_in_proj's xi stream (act+512), which the
            // real-linear in_proj contract requires to be ZERO (P1/P2 must not contribute).
            for(b=0;b<IC;b++) begin
                u_eng.uf_mem[b]=xc[b]; u_eng.uf_mem[256+b]=xc_im[b]; u_eng.uf_mem[512+b]=8'h00;
            end
            @(negedge i_clk); i_start=1; @(negedge i_clk); i_start=0;
            wait(o_done===1'b0);          // o_done is a LEVEL from the previous walk: wait for the
            wait(o_done===1'b1);          // new walk to take (done falls), THEN for its completion
            // --- live Q stability: q_mem (pass-B final) must equal the golden q_deq EVERY frame
            //     (this is what caught the recombine->uf[512..767] xi-slice pollution) ---
            begin
                int qm; qm=0;
                for (int o=0;o<256;o++) if (u_eng.q_mem[o]!==qd[o]) begin
                    if (qm<3) $display("[TB]  f%0d q_mem[%0d]=%04h exp %04h",f,o,u_eng.q_mem[o],qd[o]);
                    qm++;
                end
                if (qm!=0) begin err+=qm;
                    $display("[TB]  f%0d q_mem vs golden q_deq(imag): %0d/256 mismatch",f,qm); end
            end
            // --- packet (final = pass-B/imag) : live-slot count + idx set + closed-form probs ---
            for (int h=0; h<16; h++) begin
                int live; live=0;
                for (int s=0;s<4;s++) if (u_eng.u_sfu.o_pprob[h][s]!=16'h0) live++;
                if (live != f+1) begin err++;
                    if(err<=8) $display("[TB]  f%0d h%0d live=%0d exp %0d",f,h,live,f+1); end
                for (int s=0;s<=f;s++) begin
                    // idx-ascending: slot s should be (idx=s, prob=exp_prob[f])
                    if (u_eng.u_sfu.o_pidx[h][s] != 5'(s) || u_eng.u_sfu.o_pprob[h][s] !== exp_prob[f]) begin
                        err++;
                        if(err<=8) $display("[TB]  f%0d h%0d slot%0d got (%0d,%04h) exp (%0d,%04h)",
                                            f,h,s,u_eng.u_sfu.o_pidx[h][s],u_eng.u_sfu.o_pprob[h][s],s,exp_prob[f]);
                    end
                end
            end
            // --- ring write: K_re/V_re/K_im/V_im slots 0..f == the same per-frame codes ---
            for (int s=0;s<=f;s++) for (int h=0;h<16;h++) for (int d=0;d<16;d++) begin
                if (u_eng.u_kv.kv_mem[0     + h*512+s*16+d] !== kc [h*16+d]) begin err++;
                    if(err<=12) $display("[TB]  f%0d ring K_re s%0d h%0d d%0d got %02h exp %02h",f,s,h,d,u_eng.u_kv.kv_mem[0+h*512+s*16+d],kc[h*16+d]); end
                if (u_eng.u_kv.kv_mem[16384 + h*512+s*16+d] !== vc [h*16+d]) begin err++;
                    if(err<=12) $display("[TB]  f%0d ring V_re s%0d h%0d d%0d",f,s,h,d); end
                if (u_eng.u_kv.kv_mem[8192  + h*512+s*16+d] !== kci[h*16+d]) begin err++;
                    if(err<=12) $display("[TB]  f%0d ring K_im s%0d h%0d d%0d",f,s,h,d); end
                if (u_eng.u_kv.kv_mem[24576 + h*512+s*16+d] !== vci[h*16+d]) begin err++;
                    if(err<=12) $display("[TB]  f%0d ring V_im s%0d h%0d d%0d",f,s,h,d); end
            end
            // --- advance at the frame boundary; check ptr/filled progression ---
            @(negedge i_clk); i_advance=1; @(negedge i_clk); i_advance=0; @(negedge i_clk);
            if (u_eng.u_kv.o_ptr !== 5'(f+1) || u_eng.u_kv.o_filled !== 6'(f+1)) begin err++;
                $display("[TB]  after f%0d: ptr=%0d filled=%0d exp %0d",f,u_eng.u_kv.o_ptr,u_eng.u_kv.o_filled,f+1); end
            $display("[TB] frame %0d done (n_valid=%0d, packet prob=%04h, ring slots 0..%0d verified)",
                     f, f+1, exp_prob[f], f);
            repeat(4)@(negedge i_clk);
        end

        if (err==0) $display("[TB] ALL PASS (live 3-frame cold start: ring advance + warm-up mask + saturated/0.5/0.25 packets + per-slot K/V writes)");
        else $display("[TB] FAIL err=%0d", err);
        $finish;
    end
    initial begin repeat(900000)@(posedge i_clk); $display("[TB] TIMEOUT"); $finish; end
endmodule
