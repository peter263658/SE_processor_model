/*
    tb_qk_live.sv — LIVE QK through the full ce_engine (Step A operand routing).

    Walks the sub-op ROM to the QK sub-op (idx 32, oc=NUM_HEAD*SEQ_LEN=512). For each oc_idx the
    decoder flattens (head=oc[8:5], frame=oc[4:0]); ce_engine sources Q from q_mem (TB-preloaded) and
    K from ce_kv_cache (TB-preloaded e4m3 codes), assembles them onto PEG group-0 even lanes, runs
    MODE_LMUL_ATTN through CE_core with PPU_SCALE ×2^-2, and streams the postscale score on o_re.
    Compares o_re vs the LIVE QK golden (qk_post_fromcache = qk_score(Q, e4m3-roundtrip(Kwin))), so it
    is bit-exact independent of the (pending) real k_K. Proves: ce_kv_cache→PEG operand routing, the
    Q buffer, the 16-bit attention operand mux, the oc→(head,frame) flatten, and PPU_SCALE, all live.

    Preload: image→wbuf_mem (conv/proj/mha_in_proj prefetch liveness), kv_code_K→u_kv.kv_mem[K region],
    qk_Q→q_mem. +GDIR=<golden qk dir>.
*/
`timescale 1ns/10ps
import Param_def::*;

module tb_qk_live;
    localparam int NHEAD=16, NKEY=32, HD=16, N=NHEAD*NKEY;       // 512
    localparam int WBUF_BYTES=262144, QK_SUBOP=32, RESULT_LAT=4;

    logic i_clk=0, i_rst_n; always #5 i_clk=~i_clk;
    logic i_start, o_busy, o_done, o_rv; logic [15:0] o_re, o_im;
    ce_engine u_eng(.i_clk(i_clk),.i_rst_n(i_rst_n),.i_start(i_start),.o_busy(o_busy),.o_done(o_done),
                    .o_re(o_re),.o_im(o_im),.o_result_valid(o_rv));

    function automatic real f16r(input logic [15:0] h);
        int s,e,f; real v; begin s=h[15];e=h[14:10];f=h[9:0];
            if(e==0) v=(f==0)?0.0:(real'(f)/1024.0)*(2.0**-14);
            else if(e==31) v=1.0e30; else v=(1.0+real'(f)/1024.0)*(2.0**(e-15));
            f16r=s?-v:v; end
    endfunction
    function automatic real absr(input real x); absr=(x<0.0)?-x:x; endfunction

    logic [7:0]  img [0:WBUF_BYTES-1], kcode [0:NHEAD*NKEY*HD-1];
    logic [15:0] qbuf [0:NHEAD*HD-1], gpost [0:N-1], got [0:N-1];
    string GDIR; integer fd,nrd,b;

    initial begin
        i_rst_n=1; i_start=0;
        for(int o=0;o<N;o++) got[o]=16'hx;
        if(!$value$plusargs("GDIR=%s",GDIR)) begin $display("[TB] need +GDIR"); $finish; end
        fd=$fopen({GDIR,"../../wbuf_unified_init.bin"},"rb");
        if(fd==0) fd=$fopen("../../10_algo/hw_patterns_final/sram_images/wbuf_unified_init.bin","rb");
        nrd=$fread(img,fd); $fclose(fd);
        for(b=0;b<WBUF_BYTES;b++) u_eng.wbuf_mem[b]=img[b];
        $readmemh({GDIR,"kv_code_K.hex"}, kcode);
        for(b=0;b<NHEAD*NKEY*HD;b++) u_eng.u_kv.kv_mem[b]=kcode[b];   // K region [0:8191]
        $readmemh({GDIR,"qk_Q.hex"}, qbuf);
        for(b=0;b<NHEAD*HD;b++) u_eng.q_mem[b]=qbuf[b];
        $readmemh({GDIR,"qk_post_fromcache.hex"}, gpost);
        $display("[TB] qk_live: loaded image + kv_code_K(%0d) + q_mem(%0d), golden %0d scores",NHEAD*NKEY*HD,NHEAD*HD,N);
        repeat(2)@(negedge i_clk); i_rst_n=0; repeat(2)@(negedge i_clk); i_rst_n=1; repeat(2)@(negedge i_clk);
        u_eng.u_kv.filled = 6'd32;  // steady-state: full-window mask (TB preloads full history)
        @(negedge i_clk); i_start=1; @(negedge i_clk); i_start=0;
    end

    // sample o_re per oc when the QK sub-op result strobes (o_result_valid = id_oc_complete + RESULT_LAT)
    wire [6:0] dsub=u_eng.u_id.subop_idx;
    wire [9:0] doc =u_eng.u_id.oc_idx;
    wire       docc=u_eng.u_id.o_oc_complete;
    logic [15:0] occ_sr; logic [9:0] oc_sr[0:15]; logic [6:0] sub_sr[0:15];
    always @(posedge i_clk) begin
        occ_sr<={occ_sr[14:0],(i_rst_n&docc)}; oc_sr[0]<=doc; sub_sr[0]<=dsub;
        for(int k=1;k<16;k++) begin oc_sr[k]<=oc_sr[k-1]; sub_sr[k]<=sub_sr[k-1]; end
        if(occ_sr[RESULT_LAT-1] && sub_sr[RESULT_LAT-1]==QK_SUBOP && oc_sr[RESULT_LAT-1]<N)
            got[oc_sr[RESULT_LAT-1]]<=o_re;
    end

    task automatic report; int er; real g,x; begin
        er=0;
        for(int oc=0;oc<N;oc++) begin
            x=f16r(got[oc]); g=f16r(gpost[oc]);
            if(got[oc]!==gpost[oc]) begin er++; if(er<=10) $display("[TB] oc%0d (h%0d f%0d) got %04h exp %04h",oc,oc>>5,oc&31,got[oc],gpost[oc]); end
        end
        $display("[TB] LIVE QK: %0d/%0d postscale scores match (Q@q_mem × K@ce_kv_cache -> PEG -> PPU_SCALE)",N-er,N);
        if(er==0) $display("[TB] ALL PASS (live QK operand routing bit-exact)");
        else $display("[TB] FAIL er=%0d",er);
        $finish; end
    endtask

    initial begin
        @(posedge i_rst_n); wait(i_start===1'b1);
        fork
            begin wait((u_eng.u_id.subop_idx>QK_SUBOP)||(o_done===1'b1)); repeat(8)@(posedge i_clk); report(); end
            begin repeat(400000)@(posedge i_clk); $display("[TB] TIMEOUT subop=%0d",u_eng.u_id.subop_idx); report(); end
        join_any
        $finish;
    end
endmodule
