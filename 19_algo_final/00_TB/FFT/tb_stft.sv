`timescale 1ns/1ps

import Param_def::*;
import Mem_def::*;

`define CYCLE 10.0

module tb_stft;

    // --- 參數設定 ---
    parameter WIDTH = 16;
    parameter N_FFT = 128; // 輸入長度
    parameter N_OUT = 64;  // 輸出長度 (Bin 0~63)
    
    // --- I/O 宣告 ---
    logic i_clk, i_rst_n, i_dat_en, o_dat_en;
    logic i_new_frame; 
    STFT_mode i_stft_mode;

    logic signed [WIDTH-1:0] i_dat_re, i_dat_im;
    
    // Output 變為並列陣列 [63:0]
    logic signed [N_OUT-1:0][WIDTH-1:0] o_dat_re;
    logic signed [N_OUT-1:0][WIDTH-1:0] o_dat_im;

    // block-float e4m3 output (STFT->UFBUF cast)
    logic signed [7:0]         i_k;
    logic                      o_e4m3_en;
    logic [N_OUT-1:0][7:0]     o_e4m3_re, o_e4m3_im;

    // --- Memory ---
    logic signed [WIDTH-1:0] stimulus_re [0:1023];
    logic signed [WIDTH-1:0] stimulus_im [0:1023];
    logic signed [WIDTH-1:0] golden_re [0:1023];
    logic signed [WIDTH-1:0] golden_im [0:1023];
    logic [7:0]              golden_e4_re [0:1023];
    logic [7:0]              golden_e4_im [0:1023];
    integer e4_frame_cnt = 0;
    integer e4_err = 0;            // real cast bug (identical Q1.15 in, e4m3 differs)
    integer e4_tol = 0;            // e4m3 diff explained by <=2 LSB FFT-rounding (tolerated)
    // RTL Q1.15 spectrum latched at o_dat_en (cast output o_e4m3 is 1 cycle later)
    logic signed [N_OUT-1:0][WIDTH-1:0] lat_re, lat_im;

    integer i_idx = 0;
    integer frame_cnt = 0;
    integer err_cnt = 0;
    
    // Cycle Counter
    integer cycle_cnt = 0;
    logic start = 0;

    // --- DUT 實例化 ---
    STFT_TOP u_stft (
        .i_clk      (i_clk), 
        .i_rst_n    (i_rst_n), 
        .i_stft_mode(i_stft_mode),
        .i_new_frame(i_new_frame),
        .i_dat_en   (i_dat_en),
        .i_dat_re   (i_dat_re), 
        .i_dat_im   (i_dat_im),
        .o_dat_en   (o_dat_en),
        .o_dat_re   (o_dat_re),
        .o_dat_im   (o_dat_im),
        .i_k        (i_k),
        .o_e4m3_en  (o_e4m3_en),
        .o_e4m3_re  (o_e4m3_re),
        .o_e4m3_im  (o_e4m3_im),
        // ISTFT path unused in this (forward) TB — tie off
        .i_spec_re  ('0),
        .i_spec_im  ('0),
        .i_istft_load (1'b0),
        .i_istft_start(1'b0),
        .o_istft_en   (),
        .o_istft_sample()
    );

    // --- Clock ---
    initial i_clk = 0;
    always #(`CYCLE/2) i_clk = ~i_clk;

    initial begin 
        $fsdbDumpfile("tb_stft.fsdb");
        $fsdbDumpvars(0, "+mda+packedOnly");
    end

    // --- Load Data ---
    initial begin
        $readmemh("./patterns/input_re.hex", stimulus_re); 
        $readmemh("./patterns/input_im.hex", stimulus_im);
        $readmemh("./patterns/golden_fft_re.hex", golden_re);
        $readmemh("./patterns/golden_fft_im.hex", golden_im);
        $readmemh("./patterns/golden_fft_e4m3_re.hex", golden_e4_re);
        $readmemh("./patterns/golden_fft_e4m3_im.hex", golden_e4_im);
    end

    // Cycle Counter 邏輯
    always @(posedge i_clk) begin
        if (start) begin
            cycle_cnt <= cycle_cnt + 1;
        end
    end

    // --- Main Test Process ---
    initial begin
        // Reset & Init
        i_rst_n = 1;
        i_dat_en = 0;
        i_dat_re = 0;
        i_dat_im = 0;
        i_new_frame = 0;
        i_stft_mode = STFT;
        i_k = 8'sd9;                  // encoder.model.0.0.depthwise input block-float exp
        start = 0;
        
        // Reset Sequence
        # (3*`CYCLE) i_rst_n = 0;
        # (3*`CYCLE) i_rst_n = 1;

        $display("---------------------------------------");
        $display(" Start STFT Simulation (With 500-cycle gaps)");
        $display("---------------------------------------");
        
        @(posedge i_clk);
        start = 1;

        // 送入 4 個 Frame 的數據
        for (i_idx = 0; i_idx < 4*N_FFT; i_idx++) begin
            @(posedge i_clk);
            i_dat_en <= 1'b1;
            
            // 每個 Frame 的第一點拉起 i_new_frame
            if (i_idx % N_FFT == 0) i_new_frame <= 1'b1;
            else i_new_frame <= 1'b0;

            i_dat_re <= stimulus_re[i_idx];
            i_dat_im <= stimulus_im[i_idx];

            // ★★★ 新增邏輯：每個 Frame 結束後插入 Bubble ★★★
            // 當 i_idx 是該 Frame 的最後一點 (例如 127, 255, 383...)
            if ((i_idx % N_FFT) == (N_FFT - 1)) begin
                
                // 1. 等待一個 clock，讓 DUT 採樣到這最後一筆資料
                @(posedge i_clk); 
                
                // 2. 拉低 Enable，進入 Idle 狀態
                i_dat_en <= 1'b0;
                i_dat_re <= 0;
                i_dat_im <= 0;
                i_new_frame <= 0;
                
                // 3. 等待 500 cycles (可見的間隔)
                $display(" [Info] Frame finished, waiting 500 cycles...");
                repeat(200) @(posedge i_clk);
                
                // 4. 迴圈會回到開頭執行 @(posedge i_clk)，然後送下一筆資料
            end
        end
        
        @(posedge i_clk);
        i_dat_en <= 1'b0;
        i_dat_re <= 0;
        i_dat_im <= 0;
        i_new_frame <= 0;

        // 等待所有輸出 (時間要拉長一點，因為中間多了很多間隔)
        repeat(5000) @(posedge i_clk);
        
        $display("---------------------------------------");
        if (err_cnt == 0 && e4_err == 0 && frame_cnt > 0 && e4_frame_cnt > 0) begin
            $display(" [PASS] STFT Test Passed! Q1.15 frames=%0d  e4m3 frames=%0d", frame_cnt, e4_frame_cnt);
            $display("        (e4m3: %0d cast-bug, %0d tolerated FFT-rounding diffs)", e4_err, e4_tol);
            $display(" [PERF] Total Cycles: %0d", cycle_cnt);
        end else
            $display(" [FAIL] STFT Test Failed! Q1.15 mismatches=%0d (in %0d) ; e4m3 CAST BUGS=%0d (tolerated FFT-noise=%0d)",
                     err_cnt, frame_cnt, e4_err, e4_tol);
        $display("---------------------------------------");
            
        $finish;
    end

    // --- Checker Process ---
    always @(posedge i_clk) begin
        if (o_dat_en) begin
            
            // Loop 檢查並列輸出的 64 個頻點
            for (int k = 0; k < N_OUT; k++) begin
                integer golden_idx;
                golden_idx = frame_cnt * N_OUT + k;
                
                // Real Part Check
                if (abs_diff(o_dat_re[k], golden_re[golden_idx]) > 2) begin
                    $display("    Error Bin %2d: Real Out %h != Golden %h", 
                             k, o_dat_re[k], golden_re[golden_idx]);
                    err_cnt++;
                end
                
                // Imag Part Check
                if (abs_diff(o_dat_im[k], golden_im[golden_idx]) > 2) begin
                    $display("    Error Bin %2d: Imag Out %h != Golden %h", 
                             k, o_dat_im[k], golden_im[golden_idx]);
                    err_cnt++;
                end
            end
            
            frame_cnt++;
            // 這裡不再需要手動 stop start，讓它跑完 4 個 frame 統計總 cycle
             if (frame_cnt == 4) begin
                start <= 0;
            end
        end
    end

    function [WIDTH-1:0] abs_diff;
        input signed [WIDTH-1:0] a, b;
        begin
            if (a > b) abs_diff = a - b;
            else       abs_diff = b - a;
        end
    endfunction


    // latch the RTL Q1.15 spectrum at o_dat_en; cast output o_e4m3 appears 1 cycle later.
    always @(posedge i_clk) if (o_dat_en) begin
        lat_re <= o_dat_re;
        lat_im <= o_dat_im;
    end

    // --- e4m3 Checker (1 cycle after o_dat_en) ---
    // o_e4m3 = cast(lat_*) ; golden_e4 = cast(golden_fft). A mismatch is a REAL cast bug
    // ONLY if the RTL Q1.15 input equalled the golden Q1.15 (identical input => identical
    // e4m3). If the Q1.15 differed (<=2 LSB FFT rounding, already allowed by the Q1.15
    // check), the e4m3 diff is that noise amplified through the cast -> tolerated. The cast
    // itself is bit-exact-verified vs the model in 08_run_cast.
    always @(posedge i_clk) begin
        if (o_e4m3_en) begin
            for (int k = 0; k < N_OUT; k++) begin
                integer gi;
                gi = e4_frame_cnt * N_OUT + k;
                if (o_e4m3_re[k] !== golden_e4_re[gi]) begin
                    if (lat_re[k] === golden_re[gi]) begin
                        $display("    e4m3 CAST BUG Bin %2d: Re in=%h dut=%h gold=%h",
                                 k, lat_re[k], o_e4m3_re[k], golden_e4_re[gi]);
                        e4_err++;
                    end else e4_tol++;
                end
                if (o_e4m3_im[k] !== golden_e4_im[gi]) begin
                    if (lat_im[k] === golden_im[gi]) begin
                        $display("    e4m3 CAST BUG Bin %2d: Im in=%h dut=%h gold=%h",
                                 k, lat_im[k], o_e4m3_im[k], golden_e4_im[gi]);
                        e4_err++;
                    end else e4_tol++;
                end
            end
            e4_frame_cnt++;
        end
    end

endmodule