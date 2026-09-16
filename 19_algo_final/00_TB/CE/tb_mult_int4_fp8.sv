/*
    tb_mult_int4_fp8.sv  -  Exhaustive sweep over 16 x 256 = 4096 (W, A) pairs
    DUT:    01_RTL/engine/CE/mult_int4_fp8.sv
    Golden: pattern_lane_w4a8/golden_fp16.dat   (產自 tb_mult_int4_fp8.py)

    比對規則：
        - i_w_int4 = 4'b1000 (-8) 列被 TB skip（軟體保證不會出現）
        - 其餘 4096 - 256 = 3840 case 全部與 golden bit-equal
*/
`timescale 1ns/10ps
`define CYCLE        10.0
`define MAX_CYCLE    100000

module tb_mult_int4_fp8;

    initial begin
        $fsdbDumpfile("tb_mult_int4_fp8.fsdb");
        $fsdbDumpvars(0, "+mda+packedOnly");
    end

    // -------- clock --------
    logic i_clk;
    initial              i_clk = 0;
    always #(0.5*`CYCLE) i_clk = ~i_clk;

    // -------- DUT --------
    logic        i_valid;
    logic [3:0]  i_w_int4;
    logic [7:0]  i_a_fp8;
    logic        o_valid;
    logic [15:0] o_prod_fp16;

    mult_int4_fp8 u_dut (.*);

    // -------- golden --------
    parameter int N_W      = 16;
    parameter int N_A      = 256;
    parameter int N_TOTAL  = N_W * N_A;

    logic [15:0] golden [0 : N_TOTAL-1];

    initial $readmemh("./pattern_lane_w4a8/golden_fp16.dat", golden);

    // -------- log --------
    integer f_log;
    int     err_cnt;
    int     chk_cnt;

    // -------- main test --------
    initial begin
        f_log    = $fopen("mismatch.log", "w");
        err_cnt  = 0;
        chk_cnt  = 0;
        i_valid  = 1'b0;
        i_w_int4 = 4'h0;
        i_a_fp8  = 8'h00;

        // wait for golden load + a few clocks settle
        #(`CYCLE * 5);

        for (int w = 0; w < N_W; w++) begin
            // skip raw 4'b1000 (-8) — software clamps to [-7,+7]
            if (w[3:0] == 4'b1000) continue;

            for (int a = 0; a < N_A; a++) begin
                @(negedge i_clk);
                i_valid  = 1'b1;
                i_w_int4 = w[3:0];
                i_a_fp8  = a[7:0];

                // combinational DUT — sample after a small delta into the high phase
                @(posedge i_clk);
                #0.1;

                chk_cnt++;
                if (o_prod_fp16 !== golden[w * N_A + a]) begin
                    err_cnt++;
                    $fwrite(f_log,
                        "MISMATCH idx=%0d  w=%h  a=%h  dut=%h  gold=%h\n",
                        w * N_A + a, w[3:0], a[7:0],
                        o_prod_fp16, golden[w * N_A + a]);
                end
                if (o_valid !== 1'b1) begin
                    err_cnt++;
                    $fwrite(f_log,
                        "VALID-FAIL idx=%0d  w=%h  a=%h  o_valid=%b\n",
                        w * N_A + a, w[3:0], a[7:0], o_valid);
                end
            end
        end

        // valid pass-through check: drop i_valid, expect o_valid=0
        @(negedge i_clk);
        i_valid = 1'b0;
        @(posedge i_clk);
        #0.1;
        if (o_valid !== 1'b0) begin
            err_cnt++;
            $fwrite(f_log, "VALID-PASSTHRU FAIL: i_valid=0 but o_valid=%b\n", o_valid);
        end

        $display("==========================================");
        $display("Lane W4A8 INT4xFP8 -> FP16  exhaustive sweep");
        $display("  checked : %0d cases", chk_cnt);
        $display("  skipped : %0d (w=4'b1000)", N_A);
        if (err_cnt == 0) $display("  RESULT  : *** PASS ***");
        else              $display("  RESULT  : *** FAIL (%0d mismatches) ***  see mismatch.log", err_cnt);
        $display("==========================================");
        $fclose(f_log);
        $finish;
    end

    // -------- timeout --------
    initial begin
        #(`MAX_CYCLE * `CYCLE);
        $display("TIMEOUT");
        $fclose(f_log);
        $finish;
    end

endmodule
