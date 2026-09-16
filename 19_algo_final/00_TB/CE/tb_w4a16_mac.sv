/*
    tb_w4a16_mac.sv — verify the mha_out_proj W4A16 MAC (INT4 weight × FP16 context) at PEG+dealer level.
    Per oc: 16 chunks of 16 (W4A16 = 16-bit mode -> 1 product/pe_pair -> 16 products/PEG); each chunk
    drives PEG MODE_W4A16 (int4 weight on i_x[2i][3:0], fp16 ctx on i_y[2i]), fp16-tree(16) -> dealer
    (per-head apot) -> fp16-accumulate -> oacc. Compares vs golden oacc (gen_golden_mha_out_proj, also
    16×16). Proves the W4A16 product + 16-wide tree + per-head dealer + 16-chunk OACC accumulate.
    +GDIR=<golden mha_out_proj dir>.
*/
`timescale 1ns/10ps
import Param_def::*;

module tb_w4a16_mac;
    localparam int OC=256, IC=256, NCH=16, CW=16;

    CE_LANE_mode mode;
    logic [NUM_PE_PG-1:0][15:0] i_x, i_y;
    logic [15:0] peg_accum, dealt, acc, addz;
    logic [11:0] dcode;
    PEG    u_peg (.i_mode(mode), .i_x(i_x), .i_y(i_y), .i_preshift(8'sd0), .o_accum(peg_accum));
    dealer u_dl  (.i_sum(peg_accum), .i_apot(dcode), .o_descaled(dealt));
    // fp16 accumulate (matches OACC's fp16_add): addz = acc + dealt
    DW_fp_add_inst #(.sig_width(10),.exp_width(5),.ieee_compliance(1))
        u_fadd (.inst_a(acc), .inst_b(dealt), .inst_rnd(3'b000), .z_inst(addz), .status_inst());

    logic [3:0]  wint [0:OC*IC-1];
    logic [15:0] ctx  [0:IC-1], goacc [0:OC-1];
    logic [11:0] scode [0:15];
    string GDIR; integer oc, ch, i, er;

    initial begin
        if(!$value$plusargs("GDIR=%s",GDIR)) begin $display("[TB] need +GDIR"); $finish; end
        $readmemh({GDIR,"ctx_fp16.hex"}, ctx);
        $readmemh({GDIR,"oacc_t0.hex"},  goacc);
        $readmemh({GDIR,"w_int4.hex"},   wint);
        $readmemh({GDIR,"scale.hex"},    scode);
        mode = MODE_W4A16;
        er=0;
        for (oc=0; oc<OC; oc++) begin
            acc = 16'h0; dcode = scode[oc>>4];
            for (ch=0; ch<NCH; ch++) begin
                for (i=0;i<NUM_PE_PG;i++) begin i_x[i]=16'h0; i_y[i]=16'h0; end
                for (i=0;i<CW;i++) begin
                    i_x[2*i] = {12'd0, wint[oc*IC + ch*CW + i]};   // int4 weight [3:0]
                    i_y[2*i] = ctx[ch*CW + i];                      // fp16 context
                end
                #1;                       // PEG.o_accum + dealer.dealt settle
                acc = addz;               // acc = fp16_add(acc, dealt)
                #1;
            end
            if (acc !== goacc[oc]) begin er++; if(er<=10) $display("[TB] oc%0d (h%0d) got %04h exp %04h",oc,oc>>4,acc,goacc[oc]); end
        end
        $display("[TB] W4A16 MAC (out_proj): %0d/%0d oacc match",OC-er,OC);
        if(er==0) $display("[TB] ALL PASS (W4A16 INT4xFP16 MAC bit-exact: product + 16-tree + dealer + 16-chunk acc)");
        else $display("[TB] FAIL er=%0d",er);
        $finish;
    end
endmodule
