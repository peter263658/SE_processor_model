`timescale 1ns/10ps

`define CYCLE       10.0
`define MAX_CYCLE   10000
`define NUM_TESTS   32
`define ABS_TOL     1.0e-3

import Param_def::*;

module tb_SM;
    localparam int FP16_WIDTH = 16;
    localparam int TOTAL_DATA = `NUM_TESTS * SEQ_LEN;

    logic i_clk;
    logic i_rst_n;

    initial i_clk = 1'b0;
    always #(0.5 * `CYCLE) i_clk = ~i_clk;

    initial begin
        if (ATTN_FP_WIDTH != FP_HP_FULL_WIDTH) begin
            $fatal(1, "tb_SM expects ATTN_FP_WIDTH=16.");
        end
    end


`ifdef FSDB
    initial begin
        $fsdbDumpfile("tb_SM.fsdb");
        $fsdbDumpvars(0, tb_SM, "+mda");
    end
`endif

    logic                                               i_SM_valid;
    logic                                               o_SM_ready;
    logic                                               o_SM_valid;
    logic [SEQ_LEN-1 : 0] [ATTN_FP_WIDTH-1 : 0]         i_SM_data;
    logic [SEQ_LEN-1 : 0]                               i_SM_mask;
    logic [SEQ_LEN-1 : 0] [ATTN_FP_WIDTH-1 : 0]         o_SM_data;

    Softmax u_SM (
        .i_clk      (i_clk),
        .i_rst_n    (i_rst_n),
        .i_SM_valid (i_SM_valid),
        .i_SM_data  (i_SM_data),
        .i_SM_mask  (i_SM_mask),
        .o_SM_ready (o_SM_ready),
        .o_SM_valid (o_SM_valid),
        .o_SM_data  (o_SM_data)
    );

    logic [ATTN_FP_WIDTH-1 : 0] pattern_data   [0 : TOTAL_DATA-1];
    logic [ATTN_FP_WIDTH-1 : 0] pattern_golden [0 : TOTAL_DATA-1];
    logic [SEQ_LEN-1 : 0]       pattern_mask   [0 : `NUM_TESTS-1];

    int send_idx;
    int recv_idx;
    int err_cnt;
    int checked_cnt;

    function automatic real fp16_to_real(input logic [15:0] h);
        int sign;
        int exp;
        int frac;
        real mant;
        real value;
        begin
            sign = h[15];
            exp = h[14:10];
            frac = h[9:0];

            if (exp == 0) begin
                if (frac == 0) begin
                    value = 0.0;
                end else begin
                    mant = real'(frac) / 1024.0;
                    value = mant * (2.0 ** (-14));
                end
            end else if (exp == 31) begin
                value = 1.0e30;
            end else begin
                mant = 1.0 + real'(frac) / 1024.0;
                value = mant * (2.0 ** (exp - 15));
            end

            fp16_to_real = sign ? -value : value;
        end
    endfunction

    function automatic real abs_real(input real x);
        begin
            abs_real = (x < 0.0) ? -x : x;
        end
    endfunction

    task automatic drive_case(input int case_idx);
        int base;
        begin
            base = case_idx * SEQ_LEN;
            for (int i = 0; i < SEQ_LEN; i++) begin
                i_SM_data[i] = pattern_data[base + i];
            end
            i_SM_mask = pattern_mask[case_idx];
            i_SM_valid = 1'b1;
            @(negedge i_clk);
            i_SM_valid = 1'b0;
        end
    endtask

    task automatic check_case(input int case_idx);
        int base;
        real got;
        real exp;
        real diff;
        begin
            base = case_idx * SEQ_LEN;
            $display("[TB] compare case %0d", case_idx);

            for (int i = 0; i < SEQ_LEN; i++) begin
                got = fp16_to_real(o_SM_data[i]);
                exp = fp16_to_real(pattern_golden[base + i]);
                diff = abs_real(got - exp);
                checked_cnt++;

                if (diff > `ABS_TOL) begin
                    $display(
                        "[ERR] case=%0d idx=%0d got=%h (%e) golden=%h (%e) diff=%e",
                        case_idx,
                        i,
                        o_SM_data[i],
                        got,
                        pattern_golden[base + i],
                        exp,
                        diff
                    );
                    err_cnt++;
                end
            end
        end
    endtask

    initial begin
        i_rst_n = 1'b1;
        i_SM_valid = 1'b0;
        i_SM_data = '0;
        i_SM_mask = '0;
        send_idx = 0;
        recv_idx = 0;
        err_cnt = 0;
        checked_cnt = 0;

        $readmemh("./pattern/data.mem", pattern_data);
        $readmemb("./pattern/mask.mem", pattern_mask);
        $readmemh("./pattern/golden.mem", pattern_golden);

        repeat (2) @(negedge i_clk);
        i_rst_n = 1'b0;
        repeat (2) @(negedge i_clk);
        i_rst_n = 1'b1;
        repeat (2) @(negedge i_clk);

        fork
            begin : sender
                while (send_idx < `NUM_TESTS) begin
                    @(negedge i_clk);
                    if (o_SM_ready) begin
                        drive_case(send_idx);
                        $display("[TB] sent case %0d", send_idx);
                        send_idx++;
                    end
                end
            end

            begin : receiver
                while (recv_idx < `NUM_TESTS) begin
                    @(posedge i_clk);
                    if (o_SM_valid) begin
                        check_case(recv_idx);
                        recv_idx++;
                    end
                end
            end
        join

        if (err_cnt == 0) begin
            $display("[TB] ALL PASS, checked %0d values", checked_cnt);
        end else begin
            $display("[TB] FAIL, errors=%0d checked=%0d", err_cnt, checked_cnt);
        end
        $finish;
    end

    initial begin
        #(`MAX_CYCLE * `CYCLE);
        $display("[TB] TIMEOUT send_idx=%0d recv_idx=%0d errors=%0d", send_idx, recv_idx, err_cnt);
        $finish;
    end
endmodule
