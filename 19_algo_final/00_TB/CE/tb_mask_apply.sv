/*
    tb_mask_apply.sv — verify mask_apply (output-stage mode-C complex multiply) vs mask_apply_golden.
    Feeds all 64x399 (bin,frame) operands (spec Q2.14, mask fp16) through the combinational unit and
    checks masked Q2.14 bit-exact. +SIDE=L|R selects the golden side.
*/
`timescale 1ns/10ps
module tb_mask_apply;
    localparam int N = 25536;          // 64 bins x 399 frames (bin-major)
    string SIDE = "L"; string d;
    logic [15:0] sr [0:N-1], si [0:N-1], mr [0:N-1], mi [0:N-1], gre [0:N-1], gim [0:N-1];
    logic [15:0] i_sr, i_si, i_mr, i_mi, o_re, o_im;

    mask_apply dut (.i_spec_re(i_sr), .i_spec_im(i_si), .i_mask_re(i_mr), .i_mask_im(i_mi),
                    .o_masked_re(o_re), .o_masked_im(o_im));

    initial begin
        void'($value$plusargs("SIDE=%s", SIDE));
        d = {"../../10_algo/hw_patterns_final/mask_apply_golden/", SIDE, "/"};
        $readmemh({d,"spec_re.q214.hex"}, sr);   $readmemh({d,"spec_im.q214.hex"}, si);
        $readmemh({d,"mask_re.fp16.hex"}, mr);   $readmemh({d,"mask_im.fp16.hex"}, mi);
        $readmemh({d,"masked_re.q214.hex"}, gre); $readmemh({d,"masked_im.q214.hex"}, gim);
        $display("[TB] mask_apply %s: loaded %0d (bin,frame) operand sets", SIDE, N);
        begin int er, ei; er = 0; ei = 0;
            for (int i = 0; i < N; i++) begin
                i_sr = sr[i]; i_si = si[i]; i_mr = mr[i]; i_mi = mi[i]; #1;
                if (o_re !== gre[i]) begin er++; if (er<=8) $display("[TB] re i%0d sr%04h si%04h mr%04h mi%04h got %04h exp %04h",i,sr[i],si[i],mr[i],mi[i],o_re,gre[i]); end
                if (o_im !== gim[i]) begin ei++; if (ei<=8) $display("[TB] im i%0d got %04h exp %04h",i,o_im,gim[i]); end
            end
            $display("[TB] mask_apply %s: re %0d/%0d  im %0d/%0d match", SIDE, N-er, N, N-ei, N);
            if (er==0 && ei==0) $display("[TB] ALL PASS (mask_apply mode-C complex multiply bit-exact, side %s)", SIDE);
            else $display("[TB] FAIL re=%0d im=%0d", er, ei);
        end
        $finish;
    end
endmodule
