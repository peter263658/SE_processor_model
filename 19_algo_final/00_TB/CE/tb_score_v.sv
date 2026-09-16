/*
    tb_score_v.sv — bit-exact unit test of the CE Score×V attention datapath (MODE_LMUL_ATTN).

    Per (head, out-dim d): context[d] = Σ_{slot=0..3} L-Mul m8( prob[slot], V[ local_idx[slot] ][d] ),
    reduced by the PEG FP16 paired tree. prob = the softmax dense value at the packet idx (= fp16(code/256));
    V = the per-head t* V-window (local_idx order). The 4 products land on PEG lanes 0,2,4,6 (pe_pair LO of
    pairs 0..3; HI forced 0 in attn) so the tree computes (m8_0+m8_1)+(m8_2+m8_3) = fp16_tree([4]) = ctx[d].
    Golden = mha_module_io_fp16tree context (the HW-correct fp16-tree pipeline). Model already == golden 256/256.

    +GDIR=<golden qk dir>  (packet_idx.hex 16×4, sm_dense.hex 16×32, qk_Vwin.hex 16×32×16, sv_ctx.hex 16×16)
*/
`timescale 1ns/10ps
import Param_def::*;

module tb_score_v;
    localparam int NHEAD=16, NKEY=32, HD=16;

    CE_LANE_mode mode;
    logic [NUM_PE_PG-1:0][15:0] i_x, i_y;
    logic signed [7:0] preshift;
    logic [15:0] peg_accum;
    PEG u_peg (.i_mode(mode), .i_x(i_x), .i_y(i_y), .i_preshift(preshift), .o_accum(peg_accum));

    logic [7:0]  pidx [0:NHEAD*4-1];
    logic [15:0] dense [0:NHEAD*NKEY-1], Vwin [0:NHEAD*NKEY*HD-1], gctx [0:NHEAD*HD-1];
    string GDIR;
    integer err, h, d, s, kk;

    initial begin
        if (!$value$plusargs("GDIR=%s", GDIR)) begin $display("[TB] need +GDIR"); $finish; end
        $readmemh({GDIR,"packet_idx.hex"}, pidx);
        $readmemh({GDIR,"sm_dense.hex"},   dense);
        $readmemh({GDIR,"qk_Vwin.hex"},    Vwin);
        $readmemh({GDIR,"sv_ctx.hex"},     gctx);
        mode = MODE_LMUL_ATTN; preshift = 8'sd0;
        $display("[TB] score_v: GDIR=%s  MODE_LMUL_ATTN, preshift=0 (4 packet slots × V on lanes 0,2,4,6)", GDIR);

        err=0;
        for (h=0; h<NHEAD; h++) begin
            for (d=0; d<HD; d++) begin
                for (kk=0; kk<NUM_PE_PG; kk++) begin i_x[kk]=16'h0; i_y[kk]=16'h0; end
                for (s=0; s<4; s++) begin
                    i_x[2*s] = dense[h*NKEY + pidx[h*4+s]];                 // prob = softmax dense @ idx
                    i_y[2*s] = Vwin[(h*NKEY + pidx[h*4+s])*HD + d];         // V[idx][d]
                end
                #1;
                if (peg_accum !== gctx[h*HD+d]) begin
                    err++; if (err<=10) $display("[TB] ctx h%0d d%0d got %04h exp %04h", h, d, peg_accum, gctx[h*HD+d]);
                end
                #1;
            end
        end
        $display("[TB] Score×V context: %0d/%0d match (L-Mul m8 over top-4 packet × V + fp16-tree)",
                 NHEAD*HD-err, NHEAD*HD);
        if (err==0) $display("[TB] ALL PASS (Score×V datapath bit-exact vs fp16-tree golden)");
        else $display("[TB] FAIL err=%0d", err);
        $finish;
    end
endmodule
