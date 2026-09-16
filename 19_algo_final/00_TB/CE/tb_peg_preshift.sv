/*
    tb_peg_preshift.sv — directed self-check for the per-tensor PRE-tree pow2
    `i_preshift` added to pe_pair / PEG (FP16-tree overflow fix, 2026-06-07).

    Why not reuse tb_PEG_w4a8 (DUT vs pe_lane×32)? The golden pe_lane has NO preshift,
    so it can only validate preshift=0. This TB instead validates preshift!=0 by the
    SCALE-EQUIVARIANCE invariant of a pure exponent pre-shift:

        i_preshift=s adds s to every product's biased exponent. FP16 add is exact
        under a shared power-of-2 scaling (mantissa alignment depends only on the
        exponent DIFFERENCE, which is unchanged; rounding decisions are identical).
        => PEG(preshift=s) == ldexp(PEG(preshift=0), s)   EXACTLY,
           provided no product / tree node / result leaves the FP16 normal band
           (no saturation, no flush-to-zero). This holds for L-Mul modes too, since
           the (approximate) mantissa is computed identically in both runs.

    Test plan:
      A. deterministic anchor: all lanes = 1.0 product (acc0=32.0) per mode, sweep s.
      B. randomised: per-lane positive operands chosen so each product ∈ [1,6] and
         every node stays in-band for s∈[-12,+6] -> assert accs === ldexp16(acc0,s)
         across all 4 modes, both signs of s (covers lo+hi slices and 16-bit gating).
      C. boundary: preshift forced large -> product underflow (FTZ->0) / overflow
         (->INF15 -> saturating tree clamps to ±max-normal 0x7BFF). Explicit expected.

    Pass prints: "[TB] ALL PASS, checked N values".
*/
`timescale 1ns/10ps
module tb_peg_preshift;
    import Param_def::*;

    CE_LANE_mode                i_mode;
    logic [NUM_PE_PG-1:0][15:0] i_x, i_y;
    logic signed [7:0]          i_ps;
    logic [15:0]                acc0, accs;

    // same operands into two PEGs: one at preshift=0, one at preshift=i_ps (pure comb)
    PEG u_p0 (.i_mode(i_mode), .i_x(i_x), .i_y(i_y), .i_preshift(8'sd0), .o_accum(acc0));
    PEG u_ps (.i_mode(i_mode), .i_x(i_x), .i_y(i_y), .i_preshift(i_ps),   .o_accum(accs));

    int chk, err;
    int chk_by_mode [0:3];
    int err_by_mode [0:3];

    // fp16 ldexp by exponent-field add (inputs guaranteed normal & in-band by construction)
    function automatic logic [15:0] ldexp16(input logic [15:0] f, input int s);
        logic sgn; logic [4:0] e; logic [9:0] m; int ne;
        sgn = f[15]; e = f[14:10]; m = f[9:0];
        if (e == 5'd0) return f;                              // zero/subnormal (guarded out)
        ne = int'(e) + s;
        if      (ne <= 0)  ldexp16 = {sgn, 15'h0000};         // would FTZ
        else if (ne >= 31) ldexp16 = {sgn, 5'h1F, 10'h000};  // would Inf (out of safe band)
        else               ldexp16 = {sgn, ne[4:0], m};
    endfunction

    // positive operands per mode, product in [1,6]; r selects from {1,2}x{1,2,3}
    task automatic gen_lane(input int mode, input int r, output logic [15:0] xx, output logic [15:0] yy);
        logic [7:0]  fp8  [0:1];   logic [15:0] fp16 [0:1];   logic [3:0] q [0:2];
        fp8[0]  = 8'h38;     fp8[1]  = 8'h40;                 // 1.0, 2.0  (e4m3)
        fp16[0] = 16'h3C00;  fp16[1] = 16'h4000;             // 1.0, 2.0  (fp16)
        q[0] = 4'd1; q[1] = 4'd2; q[2] = 4'd3;               // int4 +1,+2,+3
        xx = '0; yy = '0;
        unique case (mode)
            0: begin xx = {12'd0, q[r % 3]};  yy = {8'd0, fp8[(r/3) % 2]};  end  // W4A8 : int4 x fp8
            1: begin xx = {8'd0, fp8[r % 2]}; yy = {8'd0, fp8[(r/2) % 2]}; end   // FP8  : fp8 x fp8 (m4)
            2: begin xx = fp16[r % 2];        yy = fp16[(r/2) % 2];        end   // QK   : fp16 x fp16 (m8)
            3: begin xx = {12'd0, q[r % 3]};  yy = fp16[(r/3) % 2];        end   // W4A16: int4 x fp16
        endcase
    endtask

    function automatic logic [15:0] rndlane();
        rndlane = {$random} & 16'hFFFF;
    endfunction

    task automatic do_check(input int m, input logic [15:0] exp_v, input string what);
        #1;                                                  // settle both combinational PEGs
        chk++; chk_by_mode[m]++;
        if (accs !== exp_v) begin
            err++; err_by_mode[m]++;
            if (err <= 40)
                $display("MISMATCH [%s] mode=%0d ps=%0d  acc0=%h accs=%h exp=%h",
                         what, m, i_ps, acc0, accs, exp_v);
        end
    endtask

    int rr;
    initial begin
        chk = 0; err = 0;
        for (int m = 0; m < 4; m++) begin chk_by_mode[m] = 0; err_by_mode[m] = 0; end

        // ---- A. deterministic anchor: all lanes product 1.0 -> acc0 = 32.0 (0x5000) ----
        for (int m = 0; m < 4; m++) begin
            i_mode = CE_LANE_mode'(m);
            for (int L = 0; L < NUM_PE_PG; L++) gen_lane(m, 0, i_x[L], i_y[L]);  // r=0 -> product 1.0
            for (int s = -12; s <= 6; s++) begin
                i_ps = s[7:0];
                #1;
                do_check(m, ldexp16(acc0, s), "anchor");
            end
        end

        // ---- B. randomised in-band scale-equivariance, all 4 modes, s in [-12,+6] ----
        for (int m = 0; m < 4; m++) begin
            i_mode = CE_LANE_mode'(m);
            for (int n = 0; n < 3000; n++) begin
                for (int L = 0; L < NUM_PE_PG; L++) begin
                    rr = {$random} & 15;                       // 0..15 -> mapped in gen_lane
                    gen_lane(m, rr, i_x[L], i_y[L]);
                end
                i_ps = 0; #1;                                  // acc0 settles (preshift=0 path)
                // safe-band guard: acc0 must be normal & nonzero (true by construction)
                if (acc0[14:10] == 5'd0 || acc0[14:10] == 5'h1F) continue;
                // pick s in a band that keeps acc0 and the smallest node (>=1.0) in-range
                rr = ({$random} % 19) - 12;                    // s in [-12,+6]
                i_ps = rr[7:0];
                do_check(m, ldexp16(acc0, rr), "rand");
            end
        end

        // ---- C. boundary: FTZ (underflow) and saturating-tree clamp (overflow) ----
        // C1 FTZ: single active lane, product 1.0 (W4A8), huge negative preshift -> 0
        i_mode = MODE_W4A8;
        for (int L = 0; L < NUM_PE_PG; L++) begin i_x[L] = '0; i_y[L] = '0; end
        gen_lane(0, 0, i_x[0], i_y[0]);                        // lane0 product 1.0
        i_ps = -8'sd20;
        do_check(0, 16'h0000, "FTZ");                          // product exp underflows -> 0

        // C2 SAT: single active lane, product 4.0 (W4A8 q=2 x 2.0), huge +preshift
        //         product -> INF15 (0x7C00) in pe_pair -> saturating tree -> 0x7BFF
        for (int L = 0; L < NUM_PE_PG; L++) begin i_x[L] = '0; i_y[L] = '0; end
        i_x[0] = {12'd0, 4'd2}; i_y[0] = {8'd0, 8'h40};        // 2 x 2.0 = 4.0
        i_ps = 8'sd20;
        do_check(0, 16'h7BFF, "SAT");                          // +max-normal (clamped Inf)

        // ---- report ----
        $display("==========================================");
        $display("tb_peg_preshift : PEG(preshift=s) == ldexp(PEG(preshift=0), s)  [+ FTZ/SAT boundary]");
        for (int m = 0; m < 4; m++)
            $display("  mode %0d : checked %0d, err %0d", m, chk_by_mode[m], err_by_mode[m]);
        $display("==========================================");
        if (err == 0) begin
            $display("[TB] ALL PASS, checked %0d values", chk);
            $finish;
        end else begin
            $fatal(1, "[TB] *** FAIL: %0d / %0d mismatches ***", err, chk);
        end
    end
endmodule
