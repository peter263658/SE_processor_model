/*
    tb_istft.sv — directed test for the ISTFT path (STFT_TOP i_stft_mode=ISTFT).

    Per frame: drive a masked onesided spectrum (64 bins 1..64, Q1.15) -> i_spec_*,
    pulse i_istft_load then i_istft_start; capture the 128 serial o_istft_sample and
    compare to the golden real(IFFT(conj-sym spectrum)) from gen_istft_patterns.py.

    This verifies the ON-CHIP IFFT transform (conj_sym_feeder + shared FFT128-as-IFFT +
    istft_reorder). The synthesis window, hop=80 overlap-add and the istft normalization
    are OFF-CHIP (NOT in this block / TB) — see RTL_HANDOFF_FFT_STFT_INTEGRATION §8.

    Golden is the float IFFT of the EXACT quantized spectrum the chip receives, so the
    only gap is fixed-point FFT128 rounding (same nature as the forward path's ±2 LSB).
    PASS: max |o_istft_sample - golden| <= TOL (Q1.15 LSB). A real addressing/conj/scaling
    bug produces full-scale errors (hundreds+ LSB) -> caught. Prints the actual max error.
*/
`timescale 1ns/10ps
module tb_istft;
    import Param_def::*;

    localparam int FRAMES = 4, NB = 64, NT = 128;
    localparam int TOL    = 32;     // Q1.15 LSB; generous for fixed-point IFFT, catches bugs

    logic i_clk, i_rst_n;
    STFT_mode i_stft_mode;
    // STFT-side (unused here)
    logic                      i_new_frame, i_dat_en;
    logic [FX_FULL_WIDTH-1:0]  i_dat_re, i_dat_im;
    logic signed [7:0]         i_k;
    logic                      o_dat_en, o_e4m3_en;
    logic [63:0][FX_FULL_WIDTH-1:0] o_dat_re, o_dat_im;
    logic [63:0][7:0]          o_e4m3_re, o_e4m3_im;
    // ISTFT
    logic [63:0][FX_FULL_WIDTH-1:0] i_spec_re, i_spec_im;
    logic                      i_istft_load, i_istft_start;
    logic                      o_istft_en;
    logic signed [FX_FULL_WIDTH-1:0] o_istft_sample;

    STFT_TOP u_dut (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_stft_mode(i_stft_mode),
        .i_new_frame(i_new_frame), .i_dat_en(i_dat_en), .i_dat_re(i_dat_re), .i_dat_im(i_dat_im),
        .o_dat_en(o_dat_en), .o_dat_re(o_dat_re), .o_dat_im(o_dat_im),
        .i_k(i_k), .o_e4m3_en(o_e4m3_en), .o_e4m3_re(o_e4m3_re), .o_e4m3_im(o_e4m3_im),
        .i_spec_re(i_spec_re), .i_spec_im(i_spec_im),
        .i_istft_load(i_istft_load), .i_istft_start(i_istft_start),
        .o_istft_en(o_istft_en), .o_istft_sample(o_istft_sample)
    );

    initial i_clk = 0;  always #5 i_clk = ~i_clk;

    logic signed [FX_FULL_WIDTH-1:0] spec_re_mem [0:FRAMES*NB-1];
    logic signed [FX_FULL_WIDTH-1:0] spec_im_mem [0:FRAMES*NB-1];
    logic signed [FX_FULL_WIDTH-1:0] gold_mem    [0:FRAMES*NT-1];

    function automatic int absd(input signed [FX_FULL_WIDTH-1:0] a, b);
        absd = (a > b) ? (a - b) : (b - a);
    endfunction

    int e_frame, s_idx, err, maxd;
    // checker: capture 128 serial samples/frame while o_istft_en
    always @(posedge i_clk) begin
        if (o_istft_en) begin
            int gi, d;
            gi = e_frame*NT + s_idx;
            d  = absd(o_istft_sample, gold_mem[gi]);
            if (d > maxd) maxd = d;
            if (d > TOL) begin
                err++;
                if (err <= 20) $display("  ISTFT err f%0d n%0d: dut=%h gold=%h (d=%0d)",
                                        e_frame, s_idx, o_istft_sample, gold_mem[gi], d);
            end
            s_idx = s_idx + 1;
            if (s_idx == NT) begin s_idx = 0; e_frame = e_frame + 1; end
        end
    end

    initial begin
        $readmemh("./patterns/istft_in_re.hex", spec_re_mem);
        $readmemh("./patterns/istft_in_im.hex", spec_im_mem);
        $readmemh("./patterns/istft_golden.hex", gold_mem);
        e_frame=0; s_idx=0; err=0; maxd=0;
        i_rst_n=1; i_stft_mode=ISTFT;
        i_new_frame=0; i_dat_en=0; i_dat_re=0; i_dat_im=0; i_k=8'sd0;
        i_spec_re='0; i_spec_im='0; i_istft_load=0; i_istft_start=0;

        @(posedge i_clk); #1 i_rst_n=0; repeat(3) @(posedge i_clk); #1 i_rst_n=1;

        for (int f = 0; f < FRAMES; f++) begin
            for (int b = 0; b < NB; b++) begin
                i_spec_re[b] = spec_re_mem[f*NB+b];
                i_spec_im[b] = spec_im_mem[f*NB+b];
            end
            @(posedge i_clk); #1 i_istft_load  = 1;
            @(posedge i_clk); #1 i_istft_load  = 0;
            @(posedge i_clk); #1 i_istft_start = 1;
            @(posedge i_clk); #1 i_istft_start = 0;
            repeat(700) @(posedge i_clk);     // drain feeder(128)+FFT(137)+reorder(256)
        end

        $display("==========================================");
        $display("tb_istft : ISTFT path (1 shared FFT128 as IFFT + conj_sym_feeder + istft_reorder)");
        $display("  frames checked=%0d  samples=%0d  max|err|=%0d LSB  (TOL=%0d)", e_frame, e_frame*NT, maxd, TOL);
        $display("==========================================");
        if (e_frame == FRAMES && err == 0)
            $display("[TB] ALL PASS, %0d ISTFT samples within TOL (max err %0d LSB)", FRAMES*NT, maxd);
        else
            $fatal(1, "[TB] *** FAIL: %0d errors (>%0d LSB), frames=%0d, max err %0d ***", err, TOL, e_frame, maxd);
        $finish;
    end
endmodule
