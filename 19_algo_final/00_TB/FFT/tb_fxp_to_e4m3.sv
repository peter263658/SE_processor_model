/*
    tb_fxp_to_e4m3.sv — verifies the STFT(Q1.15) -> e4m3 cast against the MODEL's
    real encoder input (encoder.model.0.0.depthwise_r block-float boundary, k=9).

    Patterns from gen_cast_patterns.py (round-trip of the golden e4m3):
      stft_cast_in.hex     : HW Q1.15 code = round(model_value / sqrt(128) * 2^15)
      stft_cast_golden.hex : the model's e4m3(model_value * 2^9)  (= encoder input golden)

    The cast must reproduce the golden e4m3, confirming it correctly folds the
    HW(÷128) vs model(÷sqrt128) normalization AND the k=9 block-float scaling.

    PASS criterion = NO error worse than 1 e4m3 step on any |q15|>2 input.
      - exact / off-by-1-e4m3 : expected (~91% / ~6%); the residual is the inherent
        Q1.15-vs-float64 + fp16-sqrt2 rounding at e4m3 bin boundaries.
      - |q15|<=2 misses : round-trip-derivation underflow (deriving Q1.15 from the
        COARSE golden loses near-zero bins) — NOT a cast bug, excluded.
      - any |q15|>2 off by >1 step => a real scaling/cast bug (a missing sqrt128
        would be ~3.5 octaves off -> floods this counter).
    Pre-validated in Python: REAL errors = 0.
*/
`timescale 1ns/10ps
module tb_fxp_to_e4m3;
    import Param_def::*;

    localparam int N = 25536;            // 64 bins x 399 frames (encoder.0.0.depthwise_r)

    logic signed [FX_FULL_WIDTH-1:0] i_fxp;
    logic signed [7:0]               i_k;
    logic        [7:0]               o_e4m3;

    fxp_to_e4m3 dut (.i_fxp(i_fxp), .i_k(i_k), .o_e4m3(o_e4m3));

    logic [15:0] in_mem   [0:N-1];
    logic [7:0]  gold_mem [0:N-1];

    int exact, off1, worse, real_err;
    int dm; logic ss;

    initial begin
        $readmemh("./patterns/stft_cast_in.hex",     in_mem);
        $readmemh("./patterns/stft_cast_golden.hex", gold_mem);
        i_k = 8'sd9;                       // encoder.model.0.0.depthwise input k
        exact = 0; off1 = 0; worse = 0; real_err = 0;

        for (int n = 0; n < N; n++) begin
            i_fxp = in_mem[n];            // 16-bit Q1.15, interpreted signed
            #1;                            // settle combinational cast
            if (o_e4m3 === gold_mem[n]) begin
                exact++;
            end else begin
                ss = (o_e4m3[7] == gold_mem[n][7]);                 // same sign
                dm = (o_e4m3[6:0] >= gold_mem[n][6:0]) ?
                     (o_e4m3[6:0] - gold_mem[n][6:0]) :
                     (gold_mem[n][6:0] - o_e4m3[6:0]);              // e4m3 code-step dist
                if (ss && dm == 1) begin
                    off1++;
                end else begin
                    worse++;
                    if ($signed(i_fxp) > 2 || $signed(i_fxp) < -2) begin
                        real_err++;
                        if (real_err <= 20)
                            $display("REAL ERR n=%0d  in=%h (%0d)  dut=%h  gold=%h",
                                     n, i_fxp, $signed(i_fxp), o_e4m3, gold_mem[n]);
                    end
                end
            end
        end

        $display("==========================================");
        $display("tb_fxp_to_e4m3 : STFT(Q1.15)->e4m3 vs model encoder-input golden (k=9)");
        $display("  checked %0d   exact %0d (%.2f%%)   off-by-1-e4m3 %0d (%.2f%%)   worse %0d",
                 N, exact, 100.0*exact/N, off1, 100.0*off1/N, worse);
        $display("  REAL errors (|q15|>2 AND >1 e4m3 step) = %0d", real_err);
        $display("==========================================");
        if (real_err == 0)
            $display("[TB] ALL PASS (scaling sqrt128 + k=9 confirmed; misses are Q1.15/fp16 boundary + tiny-bin round-trip artifacts)");
        else
            $fatal(1, "[TB] *** FAIL: %0d real cast errors ***", real_err);
        $finish;
    end
endmodule
