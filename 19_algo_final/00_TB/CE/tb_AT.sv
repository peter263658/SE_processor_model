`timescale 1ns/10ps

`define CYCLE             10.0
`define MAX_CYCLE         20000 // Increase max cycle for more test data
`define TEST_SAMPLE       1

import Param_def::*;
import Mem_def::*;

module tb_AT;

    // ====================================================================
    // Parameters
    // ====================================================================
    // Test Mode Selection: 0 = GLAC (Accumulation), 1 = PSUM (Vector Add)
    parameter TEST_MODE_SEL = 0; 
    
    // Data Definitions
    parameter INPUT_NUM     = 64;           // Transaction size
    parameter NUM_CHUNKS    = 16;           // Total chunks (Aligned with Python)
    parameter TOTAL_SAMPLES = INPUT_NUM * NUM_CHUNKS; // 64 * 16 = 1024

    // File Paths (Modify based on Python generation)
    // Default path: ./pattern_at_fp16/
    string file_in0;
    string file_in1;
    string file_out;

    initial begin
        if (TEST_MODE_SEL == 0) begin
            file_in0 = "./pattern_at_fp16/glac_input0.dat";
            file_in1 = "./pattern_at_fp16/glac_input1.dat"; // All zeros for GLAC
            file_out = "dump_glac_output.dat";
            $display("=== Mode: GLAC (Global Accumulation) ===");
        end else begin
            file_in0 = "./pattern_at_fp16/psum_input0.dat";
            file_in1 = "./pattern_at_fp16/psum_input1.dat";
            file_out = "dump_psum_output.dat";
            $display("=== Mode: PSUM (Partial Sum A+B) ===");
        end
    end

    // ====================================================================
    // Signal Declaration
    // ====================================================================
    logic i_clk;
    logic i_rst_n;
    logic sim_begin;
    
    // DUT Signals
    AT_mode                                  i_AT_mode;
    logic [INPUT_NUM -1 : 0] [CLPX_FP_WIDTH-1 : 0]   i_AT_dat0;
    logic [INPUT_NUM -1 : 0] [CLPX_FP_WIDTH-1 : 0]   i_AT_dat1; // Support Input B
    logic                                    i_AT_valid;
    logic                                    o_AT_ready;
    logic                                    o_AT_valid;
    logic [INPUT_NUM -1 : 0] [CLPX_FP_WIDTH-1 : 0]   o_AT_data;
    logic                                    o_AT_last;
    logic                                    i_AT_last;

    // Buffer for all test data
    logic [CLPX_FP_WIDTH - 1 : 0] total_input_data_0 [0 : TOTAL_SAMPLES - 1];
    logic [CLPX_FP_WIDTH - 1 : 0] total_input_data_1 [0 : TOTAL_SAMPLES - 1];

    // ====================================================================
    // Clock & Reset
    // ====================================================================
    initial i_clk = 0;
    always #(0.5*`CYCLE) i_clk = ~i_clk; 

    initial begin
        $fsdbDumpfile("tb_AT.fsdb");
        $fsdbDumpvars(0, "+mda+packedOnly");
        // $fsdbDumpvars(0, tb_AT); // Safer dumping
        
        i_rst_n = 1;
        sim_begin = 0;
        
        $display("Loading test data...");
        $readmemh(file_in0, total_input_data_0);
        $readmemh(file_in1, total_input_data_1);
        $display("Test data loaded. Total samples: %0d", TOTAL_SAMPLES);

        #(`CYCLE*3) i_rst_n = 0;
        #(`CYCLE*3) i_rst_n = 1;
                    sim_begin = 1;
    end
    
    // Watchdog
    initial begin
        #(`MAX_CYCLE * `CYCLE)
        $display("Error: Runtime exceeded!");
        $finish;
    end

    // ====================================================================
    // DUT Instantiation
    // ====================================================================
    CE_AT u_CE_AT(.*);

    // Input Initialization
    initial begin
        // i_AT_dat0 = '0;
        // i_AT_dat1 = '0;
        // i_AT_valid = 0;
        // i_AT_last = 0;
        
        // Set DUT Mode
        if (TEST_MODE_SEL == 0) i_AT_mode = AT_GLAC;
        else                    i_AT_mode = AT_PSUM;
    end

    // ====================================================================
    // Driver FSM (Fixed for Timing Alignment)
    // ====================================================================
    int chunk_idx; 

    // 1. 輸出邏輯 (Combinational Logic) - 維持不變
    assign i_AT_valid = sim_begin && (chunk_idx < NUM_CHUNKS);
    assign i_AT_last  = (chunk_idx == NUM_CHUNKS - 1);

    always_comb begin
        i_AT_dat0 = '0;
        i_AT_dat1 = '0;
        if (i_AT_valid) begin
            for (int i = 0; i < INPUT_NUM; i++) begin
                i_AT_dat0[i] = total_input_data_0[chunk_idx * INPUT_NUM + i];
                i_AT_dat1[i] = total_input_data_1[chunk_idx * INPUT_NUM + i];
            end
        end
    end

    // --------------------------------------------------------------------
    // [關鍵修正]：引入 handshake 偵測
    // 我們必須在 Posedge 檢查是否成交，才能決定是否換資料
    // --------------------------------------------------------------------
    logic handshake_occurred;

    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) 
            handshake_occurred <= 0;
        else 
            // 紀錄：剛剛這個 Posedge 有沒有發生 (Valid & Ready)？
            handshake_occurred <= (i_AT_valid && o_AT_ready);
    end

    // 2. 狀態更新 (Sequential Logic @ Negedge)
    // 利用 handshake_occurred 訊號來決定是否換下一筆
    always @(negedge i_clk) begin
        if (!i_rst_n) begin
            chunk_idx <= 0;
        end else if (sim_begin) begin
            
            // 只有當「上一個 Posedge 已經成功傳輸」後，才切換到下一筆
            if (handshake_occurred) begin
                $display("[Driver] Chunk %0d sent (Latched).", chunk_idx);
                chunk_idx <= chunk_idx + 1;
            end
            
        end
    end

    // ====================================================================
    // Monitor / Dumper (Output Checker)
    // ====================================================================
    integer f_out_handle;
    int dump_cnt;

    initial begin
        f_out_handle = $fopen(file_out, "w");
        dump_cnt = 0;
    end

    always @(negedge i_clk) begin
        if (o_AT_valid) begin
            // 1. Dump Action
            
            if (TEST_MODE_SEL == 0) begin
                // === GLAC Mode ===
                $fwrite(f_out_handle, "%h\n", o_AT_data[0]);
            end else begin
                // === PSUM Mode ===
                for (int k = 0; k < INPUT_NUM; k++) begin
                    $fwrite(f_out_handle, "%h\n", o_AT_data[k]);
                end
            end
            
            // 2. Check o_AT_last signal
            if (dump_cnt == NUM_CHUNKS - 1) begin
                // This is the last chunk, o_AT_last should be 1
                if (o_AT_last !== 1'b1) begin
                    $display("\n[Error] Last Flag Mismatch at chunk #%0d!", dump_cnt);
                    $display("        Expected: o_AT_last = 1");
                    $display("        Actual:   o_AT_last = 0");
                    $display("Simulation FAILED due to missing last flag.");
                    // $finish; // Stop on error
                end else begin
                    $display("[Check] Last chunk detected with o_AT_last = 1. (Correct)");
                end
            end else begin
                // Not the last chunk, o_AT_last should be 0
                if (o_AT_last !== 1'b0) begin
                    $display("\n[Error] Premature Last Flag at chunk #%0d!", dump_cnt);
                    $display("        Expected: o_AT_last = 0");
                    $display("        Actual:   o_AT_last = 1");
                    $display("Simulation FAILED due to unexpected last flag.");
                    // $finish; // Stop on error
                end
            end
            
            // 3. Update counter
            dump_cnt++;
            
            // 4. Termination Check
            if (dump_cnt >= NUM_CHUNKS) begin
                // Wait a bit before finish
                #(3* `CYCLE); 
                
                $display("========================================");
                $display("Simulation Completed Successfully!");
                $display("Total Chunks Processed: %d", dump_cnt);
                $display("Verified: o_AT_last behavior is correct.");
                $display("Output saved to: %s", file_out);
                $display("========================================");
                $fclose(f_out_handle);
                $finish;
            end
        end
    end

endmodule