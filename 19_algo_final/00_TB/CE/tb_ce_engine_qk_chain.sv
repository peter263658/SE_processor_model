/*
    tb_ce_engine_qk_chain.sv — LIVE tied-MHA chain through ce_engine, NOW 2-PASS + recombine (#6).

    ONE i_start walks sub-ops 0..35. The attention block (31 mha_in_proj, 32 QK, 33 score_v, 34
    mha_out_proj) runs TWICE inside inst_decoder: pass A = x.real, pass B = x.imag (tied weights,
    loaded once). After pass B's out_proj, ce_engine recombines out_re=a'-b', out_im=a'+b' (a',b'
    POST-bias = oacc + b_o) and e4m3-casts @ +3 -> output_proj(35) input codes.

    Per-pass data the TB supplies (cast_k is the SAME k=-5 for both, VERIFIED): the x.imag UF slice
    (uf[256..511]) for pass-B mha_in_proj, and BOTH passes' KV history (frames 1..31) preloaded up front
    into the now-4-PATH ce_kv_cache (task #4 DONE): pass A real -> K_re@0/V_re@16384, pass B imag ->
    K_im@8192/V_im@24576. local_idx f -> ring slot (32-f); slot 0 (current = local_idx 0) is written LIVE
    by each pass's in_proj into its own path. The 4 paths exactly fill 32 KiB and coexist, so the old
    pass A->B injection hack is GONE — pass_sel selects the path live (no swap).

    Checks: pass A (real) QK/ctx/out_proj-prebias vs qk_real; pass B (imag) vs qk_imag; the recombine
    POST-bias a'/b' vs recombine_chain/{a,b}_postbias; out_re/out_im vs out_re/out_im.hex (out_im = a'+b'
    is the ONLY path that exercises b_o — consultant flag #2); and the e4m3 @+3 boundary codes vs
    outproj_in_{re,im}_code.hex.
*/
`timescale 1ns/10ps
import Param_def::*;

module tb_ce_engine_qk_chain;
    localparam int WBUF_BYTES=262144, IC=256, WS_OFF=256256, RESULT_LAT=4;
    logic i_clk=0, i_rst_n; always #5 i_clk=~i_clk;
    logic i_start, o_busy, o_done, o_rv; logic [15:0] o_re, o_im;
    ce_engine u_eng(.i_clk(i_clk),.i_rst_n(i_rst_n),.i_start(i_start),.o_busy(o_busy),.o_done(o_done),
                    .o_re(o_re),.o_im(o_im),.o_result_valid(o_rv));

    localparam string GIN  ="../../10_algo/hw_patterns_final/sram_images/golden/mha_in_proj_real/";
    localparam string GQK  ="../../10_algo/hw_patterns_final/sram_images/golden/qk_real/";
    localparam string GINI ="../../10_algo/hw_patterns_final/sram_images/golden/mha_in_proj_imag/";
    localparam string GQKI ="../../10_algo/hw_patterns_final/sram_images/golden/qk_imag/";
    localparam string GRC  ="../../10_algo/hw_patterns_final/sram_images/golden/recombine_chain/";
    localparam string GRB  ="../../10_algo/hw_patterns_final/mha_out_proj_recombine_golden/";
    localparam string GO35 ="../../10_algo/hw_patterns_final/sram_images/golden/mattn_output_proj__0_chain/";
    localparam string IMG  ="../../10_algo/hw_patterns_final/sram_images/wbuf_unified_init.bin";
    logic [7:0]  img [0:WBUF_BYTES-1], fused [0:127], xc [0:IC-1], xc_im [0:IC-1];
    logic [7:0]  kbf [0:8191], vbf [0:8191], kbf_im [0:8191], vbf_im [0:8191];
    // pass A (real) goldens / captures
    logic [15:0] g_post [0:511], got_A [0:511];
    logic [7:0]  g_pidx [0:63]; logic [15:0] g_pprob [0:63];
    logic [15:0] g_ctx [0:255], got_ctx_A [0:255];
    logic [15:0] g_op  [0:255], got_op_A  [0:255];
    // pass B (imag) goldens / captures
    logic [15:0] g_post_im [0:511], got_B [0:511];
    logic [7:0]  g_pidx_im [0:63]; logic [15:0] g_pprob_im [0:63];
    logic [15:0] g_ctx_im [0:255], got_ctx_B [0:255];
    logic [15:0] g_op_im  [0:255], got_op_B  [0:255];
    // recombine goldens (recombine_chain/): post-bias a'/b', fp16 out_re/out_im, e4m3 @+3 codes
    logic [15:0] g_aP [0:255], g_bP [0:255], g_ore [0:255], g_oim [0:255];
    logic [7:0]  g_orec [0:255], g_oimc [0:255];
    // #7: output_proj(35) live e4m3 output codes (computed on the chain recombine input)
    logic [7:0]  g_o35re [0:255], g_o35im [0:255], got_o35re [0:255], got_o35im [0:255];
    integer fd,nrd,b;

    initial begin
        i_rst_n=1; i_start=0;
        for(int o=0;o<512;o++) begin got_A[o]=16'hx; got_B[o]=16'hx; end
        for(int o=0;o<256;o++) begin got_ctx_A[o]=16'hx; got_ctx_B[o]=16'hx; got_op_A[o]=16'hx; got_op_B[o]=16'hx; end
        fd=$fopen(IMG,"rb"); nrd=$fread(img,fd); $fclose(fd);
        for(b=0;b<WBUF_BYTES;b++) u_eng.wbuf_mem[b]=img[b];
        $readmemh({GIN,"fused_apot.bytes.hex"}, fused);
        for(b=0;b<72;b++) u_eng.wbuf_mem[WS_OFF+b]=fused[b];        // in_proj per-head scale (tied; both passes)
        for(b=0;b<8192;b++) u_eng.uf_mem[b]=8'h00;
        $readmemh({GIN, "x_code.hex"}, xc);                        // pass A input act (real) -> uf[0..255]
        $readmemh({GINI,"x_code.hex"}, xc_im);                     // pass B input act (imag) -> uf[256..511]
        for(b=0;b<IC;b++) begin u_eng.uf_mem[b]=xc[b]; u_eng.uf_mem[256+b]=xc_im[b]; end
        // β/bias now LIVE-loaded by ce_engine from the WBUF const region (in_proj@254432, b_o@253920,
        // output_proj β@252896) at each sub-op FETCH — NO TB preload (ce_engine is the sole bias writer).
        // KVBUF history (4-path): preload BOTH passes up front into their separate paths -- pass A (real)
        // -> K_re@0 / V_re@16384, pass B (imag) -> K_im@8192 / V_im@24576. local_idx f -> ring slot (32-f)
        // ((0-f) mod 32; ptr=0, single query frame). slot 0 (current) is left for each pass's in_proj LIVE
        // write. This REPLACES the old single-region + pass A->B injection hack (#6): the 4 paths coexist.
        $readmemh({GQK, "kv_code_K_bf.hex"}, kbf);   $readmemh({GQK, "kv_code_V_bf.hex"}, vbf);
        $readmemh({GQKI,"kv_code_K_bf.hex"}, kbf_im); $readmemh({GQKI,"kv_code_V_bf.hex"}, vbf_im);
        for(int h=0;h<16;h++) for(int f=1;f<32;f++) for(int d=0;d<16;d++) begin
            u_eng.u_kv.kv_mem[0     + h*512 + (32-f)*16 + d] = kbf[(h*32+f)*16+d];     // K_re (pass A real)
            u_eng.u_kv.kv_mem[16384 + h*512 + (32-f)*16 + d] = vbf[(h*32+f)*16+d];     // V_re (pass A real)
            u_eng.u_kv.kv_mem[8192  + h*512 + (32-f)*16 + d] = kbf_im[(h*32+f)*16+d];  // K_im (pass B imag)
            u_eng.u_kv.kv_mem[24576 + h*512 + (32-f)*16 + d] = vbf_im[(h*32+f)*16+d];  // V_im (pass B imag)
        end
        // pass A (real) goldens
        $readmemh({GQK,"qk_chain_post.hex"}, g_post);   $readmemh({GQK,"chain_pidx.hex"}, g_pidx);
        $readmemh({GQK,"chain_pprob.hex"}, g_pprob);    $readmemh({GQK,"chain_ctx.hex"}, g_ctx);
        $readmemh({GQK,"out_proj_oacc_chain.hex"}, g_op);
        // pass B (imag) goldens
        $readmemh({GQKI,"qk_chain_post.hex"}, g_post_im); $readmemh({GQKI,"chain_pidx.hex"}, g_pidx_im);
        $readmemh({GQKI,"chain_pprob.hex"}, g_pprob_im);  $readmemh({GQKI,"chain_ctx.hex"}, g_ctx_im);
        $readmemh({GQKI,"out_proj_oacc_chain.hex"}, g_op_im);
        // recombine goldens
        $readmemh({GRC,"a_postbias.hex"}, g_aP);  $readmemh({GRC,"b_postbias.hex"}, g_bP);
        $readmemh({GRC,"out_re.hex"}, g_ore);     $readmemh({GRC,"out_im.hex"}, g_oim);
        $readmemh({GRC,"outproj_in_re_code.hex"}, g_orec);  $readmemh({GRC,"outproj_in_im_code.hex"}, g_oimc);
        $readmemh({GO35,"out_re_code.hex"}, g_o35re);  $readmemh({GO35,"out_im_code.hex"}, g_o35im); // #7 output_proj live golden
        // output_proj complex β also live-loaded by ce_engine from WBUF @252896 (was TB-preload)
        for(int o=0;o<256;o++) begin got_o35re[o]=8'hxx; got_o35im[o]=8'hxx; end
        $display("[TB] qk_chain 2-pass (4-path KV): image+fused+x(re,im)+bias+b_o+KV(re&im hist preloaded)+goldens loaded");
        repeat(2)@(negedge i_clk); i_rst_n=0; repeat(2)@(negedge i_clk); i_rst_n=1; repeat(2)@(negedge i_clk);
        u_eng.u_kv.filled = 6'd32;   // steady-state frame t*=351: full 32-frame history (warm-up
                                     // mask all-ones; the TB preloads the full window)
        @(negedge i_clk); i_start=1; @(negedge i_clk); i_start=0;
    end

    // (no pass A->B injection: the 4-path KVBUF holds both histories in separate paths -- K_re/K_im/V_re/V_im
    //  -- selected live by pass_sel; the in_proj write fills slot 0 (current) of each pass's path. #4 done.)

    // ===== sample o_re per oc (pass-aware), aligned to o_oc_complete delayed RESULT_LAT =====
    wire        docc=u_eng.u_id.o_oc_complete;
    wire [9:0]  doc =u_eng.u_id.oc_idx;
    wire [6:0]  dsub=u_eng.u_id.subop_idx;
    wire        dps =u_eng.u_id.pass_sel;
    logic [15:0] occ_sr; logic [9:0] oc_sr[0:7]; logic [6:0] sub_sr[0:7]; logic ps_sr[0:7];
    always @(posedge i_clk) begin
        occ_sr<={occ_sr[14:0],(i_rst_n&docc)}; oc_sr[0]<=doc; sub_sr[0]<=dsub; ps_sr[0]<=dps;
        for(int k=1;k<8;k++) begin oc_sr[k]<=oc_sr[k-1]; sub_sr[k]<=sub_sr[k-1]; ps_sr[k]<=ps_sr[k-1]; end
        if(occ_sr[RESULT_LAT-1] && sub_sr[RESULT_LAT-1]==7'd32 && oc_sr[RESULT_LAT-1]<10'd512)
            begin if(!ps_sr[RESULT_LAT-1]) got_A[oc_sr[RESULT_LAT-1]]<=o_re; else got_B[oc_sr[RESULT_LAT-1]]<=o_re; end
        if(occ_sr[RESULT_LAT-1] && sub_sr[RESULT_LAT-1]==7'd33 && oc_sr[RESULT_LAT-1]<10'd256)
            begin if(!ps_sr[RESULT_LAT-1]) got_ctx_A[oc_sr[RESULT_LAT-1]]<=o_re; else got_ctx_B[oc_sr[RESULT_LAT-1]]<=o_re; end
        if(occ_sr[RESULT_LAT-1] && sub_sr[RESULT_LAT-1]==7'd34 && oc_sr[RESULT_LAT-1]<10'd256)
            begin if(!ps_sr[RESULT_LAT-1]) got_op_A[oc_sr[RESULT_LAT-1]]<=o_re; else got_op_B[oc_sr[RESULT_LAT-1]]<=o_re; end
        if(occ_sr[RESULT_LAT-1] && sub_sr[RESULT_LAT-1]==7'd35 && oc_sr[RESULT_LAT-1]<10'd256)  // #7 output_proj e4m3 out
            begin got_o35re[oc_sr[RESULT_LAT-1]]<=o_re[7:0]; got_o35im[oc_sr[RESULT_LAT-1]]<=o_im[7:0]; end
    end

    // ===== full-model latency instrumentation =====
    int pf_cyc=0, run_cyc=0, drain_cyc=0, tot_cyc=0, stall33=0;
    wire [2:0] dph = u_eng.id_phase;
    always @(posedge i_clk) if (i_rst_n && o_busy) begin
        tot_cyc<=tot_cyc+1;
        case (dph)
            3'd2: begin pf_cyc<=pf_cyc+1; if (dsub==7'd33) stall33<=stall33+1; end
            3'd3: run_cyc<=run_cyc+1;
            3'd4: drain_cyc<=drain_cyc+1;
            default: ;
        endcase
    end

    function automatic int cmp16(input logic [15:0] g[0:255], input logic [15:0] x[0:255], input string nm);
        int e; e=0;
        for(int oc=0;oc<256;oc++) if(x[oc]!==g[oc]) begin e++; if(e<=6) $display("[TB]   %s oc%0d got %04h exp %04h",nm,oc,x[oc],g[oc]); end
        return e;
    endfunction

    task automatic report; int erA,erB,ep,h,f, ecA,ecB,eoA,eoB, eaP,ebP,eore,eoim,ecre,ecim, e35r,e35i; begin
        erA=0; erB=0; ep=0;
        for(int oc=0;oc<512;oc++) begin
            if(got_A[oc]!==g_post[oc])    begin erA++; if(erA<=6) $display("[TB]   qkA oc%0d got %04h exp %04h",oc,got_A[oc],g_post[oc]); end
            if(got_B[oc]!==g_post_im[oc]) begin erB++; if(erB<=6) $display("[TB]   qkB oc%0d got %04h exp %04h",oc,got_B[oc],g_post_im[oc]); end
        end
        $display("[TB] pass A QK (real): %0d/512   pass B QK (imag): %0d/512",512-erA,512-erB);
        // pass B packet (post-walk sfu state) vs imag golden (pass A packet validated indirectly via ctx_A)
        for(int hh=0;hh<16;hh++) for(int s=0;s<4;s++) begin
            if(u_eng.u_sfu.o_pidx[hh][s]  !== g_pidx_im[hh*4+s][4:0]) ep++;
            if(u_eng.u_sfu.o_pprob[hh][s] !== g_pprob_im[hh*4+s])     ep++;
        end
        $display("[TB] pass B sfu packet (idx+prob): %0d/128 match",128-ep);
        ecA=cmp16(g_ctx,    got_ctx_A,"ctxA"); ecB=cmp16(g_ctx_im, got_ctx_B,"ctxB");
        $display("[TB] pass A ctx (real): %0d/256   pass B ctx (imag): %0d/256",256-ecA,256-ecB);
        eoA=cmp16(g_op,     got_op_A,"opA");   eoB=cmp16(g_op_im,  got_op_B,"opB");
        $display("[TB] pass A out_proj prebias oacc: %0d/256   pass B: %0d/256",256-eoA,256-eoB);
        // ===== recombine (#6): post-bias a'/b', fp16 out_re/out_im, e4m3 @+3 codes =====
        eaP=cmp16(g_aP, u_eng.out_proj_a,"a'");  ebP=cmp16(g_bP, u_eng.out_proj_b,"b'");
        $display("[TB] recombine post-bias a' (pass A): %0d/256   b' (pass B): %0d/256",256-eaP,256-ebP);
        eore=cmp16(g_ore, u_eng.recomb_re,"re"); eoim=cmp16(g_oim, u_eng.recomb_im,"im");
        $display("[TB] recombine out_re=a'-b': %0d/256   out_im=a'+b' (exercises b_o): %0d/256",256-eore,256-eoim);
        ecre=0; ecim=0;
        for(int oc=0;oc<256;oc++) begin
            if(u_eng.recomb_re_code[oc]!==g_orec[oc]) begin ecre++; if(ecre<=6) $display("[TB]   re_code oc%0d got %02h exp %02h",oc,u_eng.recomb_re_code[oc],g_orec[oc]); end
            if(u_eng.recomb_im_code[oc]!==g_oimc[oc]) begin ecim++; if(ecim<=6) $display("[TB]   im_code oc%0d got %02h exp %02h",oc,u_eng.recomb_im_code[oc],g_oimc[oc]); end
        end
        $display("[TB] e4m3@+3 output_proj-input codes: re %0d/256  im %0d/256",256-ecre,256-ecim);
        // ===== #7: output_proj(35) LIVE on the recombine codes -> e4m3 output (cast_k=-1) =====
        e35r=0; e35i=0;
        for(int oc=0;oc<256;oc++) begin
            if(got_o35re[oc]!==g_o35re[oc]) begin e35r++; if(e35r<=6) $display("[TB]   op35_re oc%0d got %02h exp %02h",oc,got_o35re[oc],g_o35re[oc]); end
            if(got_o35im[oc]!==g_o35im[oc]) begin e35i++; if(e35i<=6) $display("[TB]   op35_im oc%0d got %02h exp %02h",oc,got_o35im[oc],g_o35im[oc]); end
        end
        $display("[TB] #7 output_proj(35) LIVE e4m3 out (cast_k=-1): re %0d/256  im %0d/256",256-e35r,256-e35i);
        if(erA==0&&erB==0&&ep==0&&ecA==0&&ecB==0&&eoA==0&&eoB==0&&eaP==0&&ebP==0&&eore==0&&eoim==0&&ecre==0&&ecim==0&&e35r==0&&e35i==0)
            $display("[TB] ALL PASS (tied 2-pass + recombine + output_proj(35) LIVE — full attention layer bit-exact)");
        else $display("[TB] FAIL qkA=%0d qkB=%0d pkt=%0d ctxA=%0d ctxB=%0d opA=%0d opB=%0d a'=%0d b'=%0d re=%0d im=%0d cre=%0d cim=%0d op35r=%0d op35i=%0d",
                      erA,erB,ep,ecA,ecB,eoA,eoB,eaP,ebP,eore,eoim,ecre,ecim,e35r,e35i);
        $display("[LAT] full walk (subop 0..%0d, o_done=%b): total=%0d  RUN=%0d  PREFETCH=%0d (score_v stall=%0d)  DRAIN=%0d",
                 u_eng.u_id.subop_idx, o_done, tot_cyc, run_cyc, pf_cyc, stall33, drain_cyc);
        $finish; end
    endtask

    initial begin
        @(posedge i_rst_n); wait(i_start===1'b1);
        fork
            begin wait(o_done===1'b1); repeat(10)@(posedge i_clk); report(); end
            begin repeat(600000)@(posedge i_clk); $display("[TB] TIMEOUT subop=%0d",u_eng.u_id.subop_idx); report(); end
        join_any
        $finish;
    end
endmodule
