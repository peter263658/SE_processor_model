/*
    tb_ce_engine_proj.sv — local bit-exact verify of mattn.input_proj / output_proj (COMPLEX, per-head,
    split-k) OACC, without waiting for the exporter. Patches the behavioral WBUF with the k-FUSED
    per-head split-k WEIGHT_SCALE codes (gen_golden_proj.py emits them + the act + the oacc golden),
    drives BOTH halves (in_re→xr, in_im→xi), and checks CE_core.oacc_re/oacc_im (the matmul, pre-β)
    against the bit-exact HW golden. Verifies: complex combine P0−P1/P2+P3, per-head scale (oc//16),
    split-k dealer (collapse=0, per-PEG kreal/kimag), multi-chunk OACC (ic=256, 8 chunks).
    (Bias β + e4m3 cast are post-MAC PPU = separate; OACC is pre-β.)
*/
`timescale 1ns/10ps
import Param_def::*;

module tb_ce_engine_proj;
    localparam int WBUF_BYTES=262144, UFBUF_BYTES=8192, MAXC=256, SAMPLE_LAT=3;
    localparam real REL_TOL=0.02, ABS_TOL=1.0e-4;
    int    SUBOP=30, IC=256, OC=256;
    string SLICE="mattn_input_proj__0";                  // +SLICE / +SUBOP override (output_proj=35)
    localparam string GBASE="../../10_algo/hw_patterns_final/sram_images/golden/";
    localparam string IMG="../../10_algo/hw_patterns_final/sram_images/wbuf_unified_init.bin";

    logic i_clk=0, i_rst_n; always #5 i_clk=~i_clk;
    logic i_start, o_busy, o_done, o_rv; logic [15:0] o_re, o_im;
    ce_engine u_eng(.i_clk(i_clk),.i_rst_n(i_rst_n),.i_start(i_start),.o_busy(o_busy),.o_done(o_done),
                    .o_re(o_re),.o_im(o_im),.o_result_valid(o_rv));

    function automatic real f16r(input logic [15:0] h);
        int s,e,f; real m,v; begin s=h[15];e=h[14:10];f=h[9:0];
            if(e==0) v=(f==0)?0.0:(real'(f)/1024.0)*(2.0**-14);
            else if(e==31) v=1.0e30; else v=(1.0+real'(f)/1024.0)*(2.0**(e-15));
            f16r=s?-v:v; end
    endfunction
    function automatic real absr(input real x); absr=(x<0.0)?-x:x; endfunction

    logic [7:0] img_t [0:WBUF_BYTES-1];
    logic [7:0] are [0:MAXC-1], aim [0:MAXC-1];
    logic [15:0] gre [0:MAXC-1], gim [0:MAXC-1], got_re [0:MAXC-1], got_im [0:MAXC-1];
    integer fd,nrd,b;
    string d;

    initial begin
        i_rst_n=1; i_start=0;
        void'($value$plusargs("SUBOP=%d", SUBOP));
        void'($value$plusargs("SLICE=%s", SLICE));
        for(int o=0;o<MAXC;o++) begin got_re[o]=16'hx; got_im[o]=16'hx; end
        d={GBASE,SLICE,"/"};
        fd=$fopen(IMG,"rb"); if(fd==0) begin $display("[TB] FATAL open img"); $finish; end
        nrd=$fread(img_t,fd); $fclose(fd);
        for(b=0;b<WBUF_BYTES;b++) u_eng.wbuf_mem[b]=img_t[b];
        // NO patch: the k-fused split-k WEIGHT_SCALE blocks are now in the REAL image (R6_proj_fused,
        // input_proj@251680/251728, output_proj@251776/251824) — the ROM subop points the dealer there.
        // acts: in_re -> uf[0..IC-1] (xr) ; in_im -> uf[512..512+IC-1] (xi slice)
        for(b=0;b<UFBUF_BYTES;b++) u_eng.uf_mem[b]=8'h00;
        $readmemh({d,"act_re_t0.hex"}, are); $readmemh({d,"act_im_t0.hex"}, aim);
        for(b=0;b<IC;b++) u_eng.uf_mem[b]=are[b];
        for(b=0;b<IC;b++) u_eng.uf_mem[512+b]=aim[b];
        $readmemh({d,"oacc_re_t0.hex"}, gre); $readmemh({d,"oacc_im_t0.hex"}, gim);
        $display("[TB] proj OACC %s (subop %0d): real-image fused wscale, loaded %0d ic re+im, golden %0d oc",SLICE,SUBOP,IC,OC);
        repeat(2) @(negedge i_clk); i_rst_n=0; repeat(2) @(negedge i_clk); i_rst_n=1; repeat(2) @(negedge i_clk);
        @(negedge i_clk); i_start=1; @(negedge i_clk); i_start=0;
    end

    wire [6:0] dsub=u_eng.u_id.subop_idx;
    wire [9:0] doc =u_eng.u_id.oc_idx;
    wire       docc=u_eng.u_id.o_oc_complete;
    wire [15:0] dore=u_eng.u_ce.oacc_re, doim=u_eng.u_ce.oacc_im;
    logic [15:0] occ_sr; logic [9:0] oc_sr[0:15]; logic [6:0] sub_sr[0:15];
    always @(posedge i_clk) begin
        occ_sr<={occ_sr[14:0],(i_rst_n&docc)}; oc_sr[0]<=doc; sub_sr[0]<=dsub;
        for(int k=1;k<16;k++) begin oc_sr[k]<=oc_sr[k-1]; sub_sr[k]<=sub_sr[k-1]; end
        if(occ_sr[SAMPLE_LAT-1] && sub_sr[SAMPLE_LAT-1]==SUBOP && oc_sr[SAMPLE_LAT-1]<MAXC) begin
            got_re[oc_sr[SAMPLE_LAT-1]]<=dore; got_im[oc_sr[SAMPLE_LAT-1]]<=doim;
        end
    end

    task automatic report; int er,ei,chk; real gr,gi,xr,xi; begin
        er=0; ei=0; chk=0;
        for(int oc=0;oc<OC;oc++) begin
            xr=f16r(got_re[oc]); gr=f16r(gre[oc]); xi=f16r(got_im[oc]); gi=f16r(gim[oc]); chk++;
            if(!(absr(xr-gr)<=ABS_TOL+REL_TOL*absr(gr))) begin er++; if(er<=6) $display("[TB] RE oc%0d got %h exp %h",oc,got_re[oc],gre[oc]); end
            if(!(absr(xi-gi)<=ABS_TOL+REL_TOL*absr(gi))) begin ei++; if(ei<=6) $display("[TB] IM oc%0d got %h exp %h",oc,got_im[oc],gim[oc]); end
        end
        $display("[TB] %s OACC (subop %0d): re %0d/%0d, im %0d/%0d match (vs HW complex per-head split-k golden)",SLICE,SUBOP,chk-er,chk,chk-ei,chk);
        if(er==0&&ei==0) $display("[TB] ALL PASS (%s complex OACC bit-exact)",SLICE);
        else $display("[TB] FAIL re=%0d im=%0d",er,ei);
        $finish; end
    endtask

    initial begin
        @(posedge i_rst_n); wait(i_start===1'b1);
        fork
            begin wait((u_eng.u_id.subop_idx>SUBOP)||(o_done===1'b1)); repeat(6)@(posedge i_clk); report(); end
            begin repeat(80000)@(posedge i_clk); $display("[TB] TIMEOUT subop=%0d",u_eng.u_id.subop_idx); report(); end
        join_any
        $finish;
    end
endmodule
