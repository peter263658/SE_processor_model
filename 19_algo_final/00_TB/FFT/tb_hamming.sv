`timescale 1ns/1ps
`define CYCLE 10.0


module tb_hamming;

    // --- 參數定義 ---
    parameter WIDTH = 16;
    parameter N_FFT = 128;
    parameter CLK_PERIOD = 10;
    
    // 設定測試資料長度 (需與 Python 生成的數量一致: 4 * 128 = 512)
    parameter TEST_FRAMES = 4;
    parameter TOTAL_SAMPLES = TEST_FRAMES * N_FFT; 

    // --- 訊號宣告 ---
    logic i_clk;
    logic i_rst_n;
    
    // DUT Inputs
    logic i_dat_en;
    logic signed [WIDTH-1:0] i_dat_re;
    logic signed [WIDTH-1:0] i_dat_im;
    logic [6:0] i_counter; // 7 bits for 128 points
    
    // DUT Outputs
    logic o_dat_en;
    logic signed [WIDTH-1:0] o_dat_re;
    logic signed [WIDTH-1:0] o_dat_im;

    // --- 記憶體宣告 (用於儲存 Pattern) ---
    logic signed [WIDTH-1:0] in_re_mem   [0:TOTAL_SAMPLES-1];
    logic signed [WIDTH-1:0] in_im_mem   [0:TOTAL_SAMPLES-1];
    logic signed [WIDTH-1:0] gold_re_mem [0:TOTAL_SAMPLES-1];
    logic signed [WIDTH-1:0] gold_im_mem [0:TOTAL_SAMPLES-1];

    // --- 統計變數 ---
    integer err_count = 0;
    integer out_idx = 0; // 追蹤目前比對到第幾筆輸出
    
    // --- 實例化 DUT ---
    Hamming #(
        .FX_FULL_WIDTH(WIDTH) // 注意: 確認 Hamming 模組內的參數名稱是否為 FX_FULL_WIDTH
    ) u_hamming (
        .i_clk      (i_clk),
        .i_rst_n    (i_rst_n),
        .i_counter  (i_counter),
        .i_dat_en   (i_dat_en),
        .i_dat_re   (i_dat_re),
        .i_dat_im   (i_dat_im),
        .o_dat_en   (o_dat_en),
        .o_dat_re   (o_dat_re),
        .o_dat_im   (o_dat_im)
    );

    // --- 1. Clock Generation ---
    initial begin
        i_clk = 0;
        forever #(CLK_PERIOD/2) i_clk = ~i_clk;
    end

    initial begin 
        $fsdbDumpfile("tb_hamm.fsdb");
        $fsdbDumpvars(0, "+mda+packedOnly");
    end

    // --- 2. Load Patterns & Drive Inputs ---
    initial begin
        // 讀取由 Python 生成的 Hex 檔案
        // 確保路徑與 Python 設定的 OUTPUT_DIR 一致
        $readmemh("./patterns/input_re.hex", in_re_mem);
        $readmemh("./patterns/input_im.hex", in_im_mem);
        $readmemh("./patterns/golden_hamming_re.hex", gold_re_mem);
        $readmemh("./patterns/golden_hamming_im.hex", gold_im_mem);

        // 初始化
        i_rst_n = 1;
        i_dat_en = 0;
        i_dat_re = 0;
        i_dat_im = 0;
        i_counter = 0;
        
        // Reset 釋放
        # (3* `CYCLE) i_rst_n = 0;
        # (3* `CYCLE) i_rst_n = 1;


        $display("-------------------------------------------");
        $display(" Simulation Start: Driving %0d samples", TOTAL_SAMPLES);
        $display("-------------------------------------------");

        // 輸入驅動迴圈
        for (int i = 0; i < TOTAL_SAMPLES; i++) begin
            @(posedge i_clk); // 等待正緣
            
            i_dat_en <= 1'b1;
            
            // 從記憶體讀取資料
            i_dat_re <= in_re_mem[i];
            i_dat_im <= in_im_mem[i];
            
            // [關鍵] i_counter 邏輯檢查
            // Python/Loop 是 0~127, 但硬體 case 是 1~128
            // 且每個 Frame (128點) 都要重置
            i_counter <= (i % N_FFT); 
        end

        // 輸入結束
        @(posedge i_clk);
        i_dat_en  <= 1'b0;
        i_dat_re  <= 0;
        i_dat_im  <= 0;
        i_counter <= 0;

        // 等待所有輸出比對完成 (由下方的 Monitor 區塊控制結束)
        // 設定一個 Timeout 防止模擬卡死
        repeat(200) @(posedge i_clk); 
        if (out_idx < TOTAL_SAMPLES) begin
            $display("Error: Simulation timed out! Not all outputs received.");
            $display("Expected: %0d, Received: %0d", TOTAL_SAMPLES, out_idx);
            $finish;
        end
    end

    // --- 3. Output Monitor & Automatic Check ---
    // 獨立的 Process 監控輸出，不用擔心 Pipeline Latency
    always @(posedge i_clk) begin
        if (o_dat_en) begin
            // 檢查是否超出預期範圍
            if (out_idx >= TOTAL_SAMPLES) begin
                $display("Warning: Received more outputs than expected!");
            end else begin
                // 比對邏輯
                // 這裡使用 !== 來捕捉 X (不定態)
                if (o_dat_re !== gold_re_mem[out_idx] || o_dat_im !== gold_im_mem[out_idx]) begin
                    $display("[Error] Mismatch at Index %0d (Frame %0d, Bin %0d)", 
                             out_idx, out_idx / N_FFT, (out_idx % N_FFT) + 1);
                    $display("    Real: Expected %h, Got %h", gold_re_mem[out_idx], o_dat_re);
                    $display("    Imag: Expected %h, Got %h", gold_im_mem[out_idx], o_dat_im);
                    err_count++;
                end
                                    $display("    Real: Expected %h, Got %h", gold_re_mem[out_idx], o_dat_re);
                    $display("    Imag: Expected %h, Got %h", gold_im_mem[out_idx], o_dat_im);
            end
            
            out_idx++; // 推進比對索引
            
            // 如果全部比對完成，結束模擬
            if (out_idx == TOTAL_SAMPLES) begin
                repeat(5) @(posedge i_clk); // 多等幾個 Cycle 讓波形好看
                $display("\n-------------------------------------------");
                if (err_count == 0) begin
                    $display(" TEST PASSED! All %0d samples match Golden Model.", TOTAL_SAMPLES);
                end else begin
                    $display(" TEST FAILED! Found %0d mismatches.", err_count);
                end
                $display("-------------------------------------------\n");
                $finish;
            end
        end
    end

    // Debug: 當 Reset 期間如果有不定態輸出，可以打開這行檢查
    // always @(o_dat_re) if(o_dat_en === 1'bx) $display("Output is X!");

endmodule