/*
    tb_ce_engine_mha_in.sv — LIVE mha_in_proj OACC through ce_engine (block-diagonal real MAC).

    Runs sub-op 31 (mha_in_proj): oc=768=[Q256|K256|V256] head-major, ic=64, qkv block-diagonal act slice
    x[((oc/64)%4)*64:], per-head(48) APoT scale idx oc//16. REAL linear (imag UF slice = 0 -> oacc_re =
    Σ W·x). Patches the LOCAL fused-k apot codes @256256 (image has RAW; exporter will add) + loads x_code
    (e4m3 input) into UF real slice. Samples CE_core.oacc_re per oc vs the fp16-tree golden (oacc_t0).
    Proves: block-diagonal addressing + 48-code per-head scale + the real MAC, live through ce_engine.
    (bias-add + e4m3 cast already verified 768/768 in tb_ppu_mha_in.)
*/
`timescale 1ns/10ps
import Param_def::*;

module tb_ce_engine_mha_in;
    localparam int WBUF_BYTES=262144, OC=768, IC=256, SAMPLE_LAT=3, WS_OFF=256256;
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

    localparam string GD="../../10_algo/hw_patterns_final/sram_images/golden/mha_in_proj_real/";
    localparam string IMG="../../10_algo/hw_patterns_final/sram_images/wbuf_unified_init.bin";
    logic [7:0] img [0:WBUF_BYTES-1], fused [0:127], xc [0:IC-1];
    logic [15:0] g_oacc [0:OC-1], got [0:OC-1], g_qdeq [0:255];
    logic [7:0] g_kcode [0:255], g_vcode [0:255];     // K/V e4m3 codes (chain -> KVBUF, handoff #2)
    integer fd,nrd,b;

    initial begin
        i_rst_n=1; i_start=0;
        for(int o=0;o<OC;o++) got[o]=16'hx;
        fd=$fopen(IMG,"rb"); nrd=$fread(img,fd); $fclose(fd);
        for(b=0;b<WBUF_BYTES;b++) u_eng.wbuf_mem[b]=img[b];
        $readmemh({GD,"fused_apot.bytes.hex"}, fused);                 // 72 B (48 codes, 12-bit packed)
        for(b=0;b<72;b++) u_eng.wbuf_mem[WS_OFF+b]=fused[b];           // patch local fused-k scale
        for(b=0;b<8192;b++) u_eng.uf_mem[b]=8'h00;
        $readmemh({GD,"x_code.hex"}, xc);
        for(b=0;b<IC;b++) u_eng.uf_mem[b]=xc[b];                       // x -> UF real slice; imag = 0
        $readmemh({GD,"oacc_t0.hex"}, g_oacc);
        $readmemh({GD,"q_deq.hex"}, g_qdeq);
        $readmemh({GD,"k_code_hw.hex"}, g_kcode);       // K e4m3 codes (chain -> KVBUF sel=0)
        $readmemh({GD,"v_code_hw.hex"}, g_vcode);       // V e4m3 codes (chain -> KVBUF sel=1)
        $readmemh({GD,"bias.hex"}, u_eng.bias_mem);     // mha_in_proj bias -> live PPU β-add
        $display("[TB] mha_in_proj: image + fused@%0d + x_code(%0d) loaded, golden %0d oc",WS_OFF,IC,OC);
        repeat(2)@(negedge i_clk); i_rst_n=0; repeat(2)@(negedge i_clk); i_rst_n=1; repeat(2)@(negedge i_clk);
        @(negedge i_clk); i_start=1; @(negedge i_clk); i_start=0;
    end

    wire [6:0] dsub=u_eng.u_id.subop_idx;
    wire [9:0] doc =u_eng.u_id.oc_idx;
    wire       docc=u_eng.u_id.o_oc_complete;
    wire [15:0] dore=u_eng.u_ce.oacc_re;
    logic [15:0] occ_sr; logic [9:0] oc_sr[0:15]; logic [6:0] sub_sr[0:15];
    always @(posedge i_clk) begin
        occ_sr<={occ_sr[14:0],(i_rst_n&docc)}; oc_sr[0]<=doc; sub_sr[0]<=dsub;
        for(int k=1;k<16;k++) begin oc_sr[k]<=oc_sr[k-1]; sub_sr[k]<=sub_sr[k-1]; end
        if(occ_sr[SAMPLE_LAT-1] && sub_sr[SAMPLE_LAT-1]==31 && oc_sr[SAMPLE_LAT-1]<OC)
            got[oc_sr[SAMPLE_LAT-1]]<=dore;
    end

    task automatic report; int er; begin
        er=0;
        for(int oc=0;oc<OC;oc++) begin
            if(got[oc]!==g_oacc[oc]) begin er++; if(er<=10) $display("[TB] oc%0d (%s) got %04h exp %04h",oc,(oc<256)?"Q":(oc<512)?"K":"V",got[oc],g_oacc[oc]); end
        end
        $display("[TB] mha_in_proj OACC (block-diagonal, 48-scale): %0d/%0d match",OC-er,OC);
        // chaining handoff #1+#2: Q -> q_mem (dequant), K/V -> KVBUF (raw e4m3 code @ frame 31)
        begin int eq,ek,ev; eq=0; ek=0; ev=0;
            for(int oc=0;oc<256;oc++) if(u_eng.q_mem[oc]!==g_qdeq[oc]) begin eq++; if(eq<=6) $display("[TB] q_mem[%0d] got %04h exp %04h",oc,u_eng.q_mem[oc],g_qdeq[oc]); end
            $display("[TB] Q write-back -> q_mem: %0d/256 match",256-eq);
            // 4-path KVBUF: pass A (real) in_proj writes K->K_re@0, V->V_re@16384, current frame -> ring slot 0.
            for(int h=0;h<16;h++) for(int d=0;d<16;d++) begin
                if(u_eng.u_kv.kv_mem[0     + h*512+0*16+d] !== g_kcode[h*16+d]) begin ek++; if(ek<=6) $display("[TB] KVBUF K h%0d d%0d got %02h exp %02h",h,d,u_eng.u_kv.kv_mem[0+h*512+0*16+d],g_kcode[h*16+d]); end
                if(u_eng.u_kv.kv_mem[16384 + h*512+0*16+d] !== g_vcode[h*16+d]) begin ev++; if(ev<=6) $display("[TB] KVBUF V h%0d d%0d got %02h exp %02h",h,d,u_eng.u_kv.kv_mem[16384+h*512+0*16+d],g_vcode[h*16+d]); end
            end
            $display("[TB] K -> KVBUF: %0d/256 match",256-ek);
            $display("[TB] V -> KVBUF: %0d/256 match",256-ev);
            if(er==0 && eq==0 && ek==0 && ev==0) $display("[TB] ALL PASS (mha_in_proj live OACC + Q/K/V write-back bit-exact)");
            else $display("[TB] FAIL oacc_er=%0d q_er=%0d k_er=%0d v_er=%0d",er,eq,ek,ev);
        end
        $finish; end
    endtask

    initial begin
        @(posedge i_rst_n); wait(i_start===1'b1);
        fork
            begin wait((u_eng.u_id.subop_idx>31)||(o_done===1'b1)); repeat(8)@(posedge i_clk); report(); end
            begin repeat(400000)@(posedge i_clk); $display("[TB] TIMEOUT subop=%0d",u_eng.u_id.subop_idx); report(); end
        join_any
        $finish;
    end
endmodule
