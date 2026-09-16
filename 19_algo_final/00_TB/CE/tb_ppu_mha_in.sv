/*
    tb_ppu_mha_in.sv — local verify of mha_in_proj's post-MAC PPU (bias-add + e4m3 cast), the "patch
    bias" check. Drives the VERIFIED fp16-tree oacc golden into ppu_fma with the mha_in_proj PPU config:
        PPU_CONV, α=identity(0xC00), PReLU=identity(0xC00), β=in_proj bias[oc] (per-oc, the R²→1 fix),
        cast_k = i_cast_k per qkv region (Q,K=+7 ; V=+6).
    Checks o_re[7:0] (e4m3) bit-exact vs the golden output codes (= e4m3(scale2(oacc+bias,cast_k))), which
    match the deploy Q/K/V_headmajor to a few ULP (fp16-tree-vs-fp32). Proves bias-add + per-region cast.
    +GDIR=<golden mha_in_proj dir>.
*/
`timescale 1ns/10ps
import Param_def::*;

module tb_ppu_mha_in;
    localparam int OC=768;
    localparam logic [1:0] PPU_CONV = 2'd0;

    logic [1:0]  i_mode;
    logic [15:0] i_re, i_im;
    logic [11:0] i_bn_alpha_re, i_bn_alpha_im, i_prelu_re, i_prelu_im, i_scale_re, i_scale_im;
    logic [15:0] i_bn_beta_re, i_bn_beta_im;
    logic signed [7:0] i_cast_k;
    logic [15:0] o_re, o_im;
    ppu_fma u_ppu (.i_mode(i_mode), .i_re(i_re), .i_im(i_im),
        .i_bn_alpha_re(i_bn_alpha_re), .i_bn_alpha_im(i_bn_alpha_im),
        .i_bn_beta_re(i_bn_beta_re), .i_bn_beta_im(i_bn_beta_im),
        .i_prelu_re(i_prelu_re), .i_prelu_im(i_prelu_im), .i_cast_k(i_cast_k),
        .i_scale_re(i_scale_re), .i_scale_im(i_scale_im), .o_re(o_re), .o_im(o_im));

    logic [15:0] oacc [0:OC-1], bias [0:OC-1];
    logic [7:0]  gq [0:255], gk [0:255], gv [0:255], gold [0:OC-1];
    string GDIR; integer er, oc;

    initial begin
        if (!$value$plusargs("GDIR=%s", GDIR)) begin $display("[TB] need +GDIR"); $finish; end
        $readmemh({GDIR,"oacc_t0.hex"}, oacc);
        $readmemh({GDIR,"bias.hex"},    bias);
        $readmemh({GDIR,"q_code_hw.hex"}, gq);
        $readmemh({GDIR,"k_code_hw.hex"}, gk);
        $readmemh({GDIR,"v_code_hw.hex"}, gv);
        for (oc=0; oc<256; oc++) begin gold[oc]=gq[oc]; gold[256+oc]=gk[oc]; gold[512+oc]=gv[oc]; end
        i_mode=PPU_CONV; i_im=16'h0; i_bn_beta_im=16'h0;
        i_bn_alpha_re=12'hC00; i_bn_alpha_im=12'hC00; i_prelu_re=12'hC00; i_prelu_im=12'hC00;
        i_scale_re=12'hC00; i_scale_im=12'hC00;
        $display("[TB] ppu_mha_in: bias-add + cast (Q/K i_cast_k=+7, V=+6), %0d oc", OC);

        er=0;
        for (oc=0; oc<OC; oc++) begin
            i_re = oacc[oc]; i_bn_beta_re = bias[oc];
            i_cast_k = (oc < 512) ? 8'sd7 : 8'sd6;        // Q,K -> +7 ; V -> +6
            #1;
            if (o_re[7:0] !== gold[oc]) begin
                er++; if (er<=8) $display("[TB] oc%0d (%s) got %02h exp %02h", oc, (oc<256)?"Q":(oc<512)?"K":"V", o_re[7:0], gold[oc]);
            end
            #1;
        end
        $display("[TB] mha_in_proj PPU (bias + e4m3 cast): %0d/%0d match", OC-er, OC);
        if (er==0) $display("[TB] ALL PASS (mha_in_proj bias-add + cast bit-exact vs fp16 model)");
        else $display("[TB] FAIL er=%0d", er);
        $finish;
    end
endmodule
