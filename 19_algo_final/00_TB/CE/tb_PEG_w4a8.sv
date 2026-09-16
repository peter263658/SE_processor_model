/*
    tb_PEG_w4a8.sv — self-checking TB for the W4A8 Design-B PEG.

    Strategy: compare the DUT PEG (fused pe_pair = two shared 8-bit slices that
    chain into a 16-bit adder for the 16-bit modes) against a reference PEG_ref
    that uses the validated full pe_lane on ALL 32 lanes (odd lanes gated to 0 in
    16-bit modes, matching pe_pair's hi gating), feeding the SAME 32-way FP16
    saturating adder_tree_fp16. Any o_accum mismatch therefore isolates a pe_pair
    arithmetic / shared-adder / pairing-wiring divergence from pe_lane. Drives
    random operands across all 4 modes.

    (Accumulation is NOT compared against an external fp32 golden here — golden /
    UFBUF / WBUF integration comes later with the CE control path. This TB pins the
    structural + bit-exact-product correctness of the PEG itself.)

    Pass prints: "[TB] ALL PASS, checked N values".
*/
`timescale 1ns/10ps
module tb_PEG_w4a8;
    import Param_def::*;

    CE_LANE_mode               i_mode;
    logic [NUM_PE_PG-1:0][15:0] i_x, i_y;
    logic [15:0]               dut_accum, ref_accum;

    PEG     u_dut (.i_mode(i_mode), .i_x(i_x), .i_y(i_y), .i_preshift(8'sd0), .o_accum(dut_accum));
    PEG_ref u_ref (.i_mode(i_mode), .i_x(i_x), .i_y(i_y), .o_accum(ref_accum));

    integer f_log;
    int chk, err;
    int chk_by_mode [0:3];
    int err_by_mode [0:3];

    localparam int N_PER_MODE = 4000;

    function automatic logic [15:0] rnd16();
        rnd16 = {$random} & 16'hFFFF;
    endfunction

    initial begin
        chk = 0; err = 0;
        for (int m = 0; m < 4; m++) begin chk_by_mode[m] = 0; err_by_mode[m] = 0; end
        f_log = $fopen("peg_w4a8_mismatch.log", "w");

        for (int m = 0; m < 4; m++) begin
            i_mode = CE_LANE_mode'(m);
            for (int n = 0; n < N_PER_MODE; n++) begin
                for (int L = 0; L < NUM_PE_PG; L++) begin
                    i_x[L] = rnd16();
                    i_y[L] = rnd16();
                end
                #1;                                   // settle combinational DUT + REF
                chk++; chk_by_mode[m]++;
                if (dut_accum !== ref_accum) begin
                    err++; err_by_mode[m]++;
                    if (err <= 40)
                        $fwrite(f_log, "MISMATCH mode=%0d n=%0d dut=%h ref=%h\n",
                                m, n, dut_accum, ref_accum);
                end
            end
        end
        $fclose(f_log);

        $display("==========================================");
        $display("tb_PEG_w4a8 : PEG (fused pe_pair, shared 2x8-bit slices) vs PEG_ref (pe_lane x32)");
        for (int m = 0; m < 4; m++)
            $display("  mode %0d : checked %0d, err %0d", m, chk_by_mode[m], err_by_mode[m]);
        $display("==========================================");
        if (err == 0) begin
            $display("[TB] ALL PASS, checked %0d values", chk);
            $finish;
        end else begin
            $fatal(1, "[TB] *** FAIL: %0d / %0d mismatches *** see peg_w4a8_mismatch.log", err, chk);
        end
    end
endmodule


/* ---------------------------------------------------------------------------
   PEG_ref — reference group: full pe_lane on all 32 lanes, odd lanes gated to 0
   in 16-bit modes (matches pe_pair hi gating). Same adder_tree_fp16 as the DUT.
--------------------------------------------------------------------------- */
module PEG_ref
    import Param_def::*;
(   input  CE_LANE_mode                i_mode,
    input  logic [NUM_PE_PG-1:0][15:0] i_x,
    input  logic [NUM_PE_PG-1:0][15:0] i_y,
    output logic [15:0]                o_accum
);
    wire is_16bit = (i_mode == MODE_LMUL_ATTN) || (i_mode == MODE_W4A16);

    logic [NUM_PE_PG-1:0][15:0] prod;

    genvar L;
    generate
        for (L = 0; L < NUM_PE_PG; L++) begin : g_ref
            logic [15:0] p;
            pe_lane u (.i_mode(i_mode), .i_op_x(i_x[L]), .i_op_y(i_y[L]), .o_prod_fp16(p));
            assign prod[L] = (is_16bit && L[0]) ? 16'h0000 : p;   // odd lane = hi -> 0 in 16-bit
        end
    endgenerate

    adder_tree_fp16 u_tree (.i_add_array(prod), .o_accum(o_accum));
endmodule
