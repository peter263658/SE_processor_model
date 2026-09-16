/*
    tb_pe_lane.sv — self-checking TB for pe_lane.sv (mixed-precision zero-mult PE lane)

    Reads pattern_pe_lane/stim.dat (produced by tb_pe_lane.py); each line:
        <mode> <op_x> <op_y> <golden>     (all hex)
    Drives the combinational DUT and bit-compares o_prod_fp16 == golden.
    Pass prints: "[TB] ALL PASS, checked N values".
*/
`timescale 1ns/10ps
`define MAX_LINES 100000

module tb_pe_lane;

    logic [1:0]  i_mode;
    logic [15:0] i_op_x, i_op_y;
    logic [15:0] o_prod_fp16;

    pe_lane u_dut (
        .i_mode      (i_mode),
        .i_op_x      (i_op_x),
        .i_op_y      (i_op_y),
        .o_prod_fp16 (o_prod_fp16)
    );

    integer fd, code, f_log;
    int     chk, err;
    int     err_by_mode [0:3];
    int     chk_by_mode [0:3];

    // scratch read variables (ints for $fscanf %h)
    integer rd_mode, rd_x, rd_y, rd_gold;
    logic [15:0] gold;

    // (waveform dump omitted; self-checking TB. Add Verdi PLI + $fsdbDump* if needed.)

    initial begin
        chk = 0; err = 0;
        for (int m = 0; m < 4; m++) begin err_by_mode[m] = 0; chk_by_mode[m] = 0; end

        fd    = $fopen("./pattern_pe_lane/stim.dat", "r");
        if (fd == 0) begin
            $display("[TB] ERROR: cannot open pattern_pe_lane/stim.dat (run tb_pe_lane.py first)");
            $finish;
        end
        f_log = $fopen("pe_lane_mismatch.log", "w");

        for (int line = 0; line < `MAX_LINES; line++) begin
            code = $fscanf(fd, "%h %h %h %h", rd_mode, rd_x, rd_y, rd_gold);
            if (code != 4) break;                  // EOF / done

            i_mode = rd_mode[1:0];
            i_op_x = rd_x[15:0];
            i_op_y = rd_y[15:0];
            gold   = rd_gold[15:0];
            #1;                                    // settle combinational DUT

            chk++;
            chk_by_mode[i_mode]++;
            if (o_prod_fp16 !== gold) begin
                err++;
                err_by_mode[i_mode]++;
                if (err <= 40)
                    $fwrite(f_log, "MISMATCH mode=%0d x=%h y=%h dut=%h gold=%h\n",
                            i_mode, i_op_x, i_op_y, o_prod_fp16, gold);
            end
        end
        $fclose(fd);
        $fclose(f_log);

        $display("==========================================");
        $display("pe_lane mixed-precision PE lane test");
        for (int m = 0; m < 4; m++)
            $display("  mode %0d : checked %0d, err %0d", m, chk_by_mode[m], err_by_mode[m]);
        $display("==========================================");
        if (err == 0) begin
            $display("[TB] ALL PASS, checked %0d values", chk);
            $finish;
        end else begin
            // non-zero exit so `set -e` / CI catches the failure
            $fatal(1, "[TB] *** FAIL: %0d / %0d mismatches *** see pe_lane_mismatch.log", err, chk);
        end
    end

endmodule
