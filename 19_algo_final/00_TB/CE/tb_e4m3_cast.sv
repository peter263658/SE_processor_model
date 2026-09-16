/*
    tb_e4m3_cast.sv — directed round-trip test for fp16_to_e4m3 (ce_fp_util.sv).
    Closes 雷④ for the cast (caught BUG-1 te>=15 over-clamp and BUG-2 subnormal sh).

    For every e4m3 code c (skip NaN 0x7F/0xFF): build the EXACT fp16 value of c, feed
    fp16_to_e4m3, and require it returns c. Since c's value is exactly representable in
    e4m3, RNE must round back to c. Any mismatch = a cast bug.
    Pass prints "[TB] ALL PASS, checked N".
*/
`timescale 1ns/10ps
module tb_e4m3_cast;

    logic [15:0] h;
    logic [7:0]  q;

    fp16_to_e4m3 u_dut (.i_h(h), .o_e4m3(q));

    // exact e4m3 (OCP, bias 7) -> fp16 value
    function automatic [15:0] e4m3_to_fp16(input [7:0] c);
        logic        s;  logic [3:0] e4;  logic [2:0] m3;
        int          sig, exp2, L, ef;
        logic [20:0] up;
        s = c[7]; e4 = c[6:3]; m3 = c[2:0];
        if (e4 == 4'd0 && m3 == 3'd0) return {s, 15'd0};         // zero
        if (e4 == 0) begin sig = m3;        exp2 = -9;  end       // subnormal: m3 * 2^-9
        else         begin sig = 8 + m3;    exp2 = e4 - 10; end   // normal: (8+m3)*2^(e4-10)
        L = 0;
        for (int i = 0; i < 4; i++) if (sig[i]) L = i;
        up = sig << (10 - L);                                     // leading 1 to bit10
        ef = exp2 + L + 15;                                       // fp16 biased exp
        return {s, ef[4:0], up[9:0]};
    endfunction

    int chk, err;
    logic [7:0] gold;
    initial begin
        chk = 0; err = 0;
        for (int c = 0; c < 256; c++) begin
            if (c == 8'h7F || c == 8'hFF) continue;              // e4m3 NaN, skip
            gold = c[7:0];
            h = e4m3_to_fp16(c[7:0]);
            #1;
            chk++;
            if (q !== gold) begin
                err++;
                if (err <= 30) $display("MISMATCH c=%h fp16=%h dut=%h", c[7:0], h, q);
            end
        end
        $display("==========================================");
        $display("tb_e4m3_cast: fp16_to_e4m3 round-trip over all 254 finite codes");
        $display("  checked %0d, err %0d", chk, err);
        $display("==========================================");
        if (err == 0) $display("[TB] ALL PASS, checked %0d", chk);
        else          $fatal(1, "[TB] *** FAIL: %0d mismatches", err);
        $finish;
    end
endmodule
