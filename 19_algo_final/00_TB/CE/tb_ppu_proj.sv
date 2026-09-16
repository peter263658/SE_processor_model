/*
    tb_ppu_proj.sv — bit-exact unit test of the PROJ full PPU output (β-add + e4m3 cast).

    The OACC (complex matmul) is already verified bit-exact end-to-end through ce_engine
    (tb_ce_engine_proj.sv: re/im 256/256). This isolates the post-MAC PPU stage: drive the
    VERIFIED oacc golden into ppu_fma with the proj PPU config —
        PPU_CONV, α = identity (0xC00), PReLU = identity (0xC00), β = proj bias (per-oc, image
        LE fp16), cast_k = −(consumer block-float exp)  [input_proj +5, output_proj −2]
    — and check o_re/o_im (e4m3 in [7:0]) bit-exact vs the golden produced by the bit-exact
    fp16 model (gen_golden_proj.py). The deploy cross-check (input_proj 248/252 vs the real
    mha_in_proj consumer codes) is the model-correctness proof; this TB proves RTL == model.

    +GDIR=<golden slice dir>   (e.g. .../golden/mattn_input_proj__0/)
*/
`timescale 1ns/10ps
import Param_def::*;

module tb_ppu_proj;
    localparam int OC = 256;
    localparam logic [1:0] PPU_CONV = 2'd0;

    logic [1:0]  i_mode;
    logic [15:0] i_re, i_im;
    logic [11:0] i_bn_alpha_re, i_bn_alpha_im, i_prelu_re, i_prelu_im, i_scale_re, i_scale_im;
    logic [15:0] i_bn_beta_re, i_bn_beta_im;
    logic signed [7:0] i_cast_k;
    logic [15:0] o_re, o_im;

    ppu_fma u_ppu (
        .i_mode(i_mode), .i_re(i_re), .i_im(i_im),
        .i_bn_alpha_re(i_bn_alpha_re), .i_bn_alpha_im(i_bn_alpha_im),
        .i_bn_beta_re(i_bn_beta_re),   .i_bn_beta_im(i_bn_beta_im),
        .i_prelu_re(i_prelu_re), .i_prelu_im(i_prelu_im), .i_cast_k(i_cast_k),
        .i_scale_re(i_scale_re), .i_scale_im(i_scale_im),
        .o_re(o_re), .o_im(o_im)
    );

    logic [15:0] oacc_re [0:OC-1], oacc_im [0:OC-1], beta_re [0:OC-1], beta_im [0:OC-1];
    logic [7:0]  gold_re [0:OC-1], gold_im [0:OC-1];
    string GDIR;
    integer fd, ck, er, ei, oc;

    initial begin
        if (!$value$plusargs("GDIR=%s", GDIR)) begin $display("[TB] need +GDIR"); $finish; end
        $readmemh({GDIR,"oacc_re_t0.hex"},  oacc_re);
        $readmemh({GDIR,"oacc_im_t0.hex"},  oacc_im);
        $readmemh({GDIR,"beta_re.hex"},     beta_re);
        $readmemh({GDIR,"beta_im.hex"},     beta_im);
        $readmemh({GDIR,"out_re_code.hex"}, gold_re);
        $readmemh({GDIR,"out_im_code.hex"}, gold_im);
        fd = $fopen({GDIR,"cast_k.txt"}, "r");
        if (fd==0) begin $display("[TB] need cast_k.txt"); $finish; end
        void'($fscanf(fd, "%d", ck)); $fclose(fd);

        // proj PPU constants (layer-constant): α=identity, PReLU=identity, cast_k from golden
        i_mode = PPU_CONV;
        i_bn_alpha_re = 12'hC00; i_bn_alpha_im = 12'hC00;   // dealer a=0,h2=0 -> ×1 (identity)
        i_prelu_re    = 12'hC00; i_prelu_im    = 12'hC00;   // apot6b_decode -> k=0,sign=0 (identity)
        i_scale_re    = 12'hC00; i_scale_im    = 12'hC00;   // unused in CONV
        i_cast_k      = ck[7:0];
        $display("[TB] ppu_proj: GDIR=%s  cast_k=%0d  (α/PReLU=identity 0xC00)", GDIR, ck);

        er = 0; ei = 0;
        for (oc = 0; oc < OC; oc++) begin
            i_re = oacc_re[oc]; i_im = oacc_im[oc];
            i_bn_beta_re = beta_re[oc]; i_bn_beta_im = beta_im[oc];
            #1;
            if (o_re[7:0] !== gold_re[oc]) begin
                er++; if (er<=8) $display("[TB] RE oc%0d got %02h exp %02h (oacc %04h β %04h)",
                                          oc, o_re[7:0], gold_re[oc], oacc_re[oc], beta_re[oc]);
            end
            if (o_im[7:0] !== gold_im[oc]) begin
                ei++; if (ei<=8) $display("[TB] IM oc%0d got %02h exp %02h (oacc %04h β %04h)",
                                          oc, o_im[7:0], gold_im[oc], oacc_im[oc], beta_im[oc]);
            end
            #1;
        end
        $display("[TB] proj PPU full output: re %0d/%0d, im %0d/%0d match (β-add + e4m3 cast)",
                 OC-er, OC, OC-ei, OC);
        if (er==0 && ei==0) $display("[TB] ALL PASS (proj β-add + cast bit-exact vs fp16 model)");
        else $display("[TB] FAIL re=%0d im=%0d", er, ei);
        $finish;
    end
endmodule
