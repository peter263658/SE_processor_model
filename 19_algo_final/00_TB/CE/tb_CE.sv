`timescale 1ns/10ps

`define CYCLE       10.0
`define MAX_CYCLE   50000


import Param_def::*;
import Mem_def::*;

module tb_CE;

    // ====================================================================
    // Parameters
    // ====================================================================
    parameter NUM_BANKS     = 16;
    parameter BANK_WIDTH    = 128; // 128 bits per addr
    parameter BANK_DEPTH    = 256; 
    
    // Derived
    parameter WORD_WIDTH    = 32;  
    parameter WORDS_PER_BANK= BANK_WIDTH / WORD_WIDTH; // 4 words
    parameter TOTAL_WIDTH   = NUM_BANKS * BANK_WIDTH;  // 2048 bits (Logical Line)

    // ====================================================================
    // Signals (Banked SRAMs)
    // ====================================================================
    logic i_clk;
    logic i_rst_n;
    logic i_start;
    logic o_done;
    
    // 定義 16 組 SRAM，每組寬 128 bit，深 256
    // 這裡使用 3D array: [Bank][Addr][Data] 
    // 或者用 generate block 裡的 local variable 比較符合物理擺放，
    // 但為了 TB 容易 access，我們宣告一個 Global 的 Array

    logic [BANK_WIDTH-1:0] ifbuf_banks [0:NUM_BANKS-1][0:BANK_DEPTH-1];
    logic [BANK_WIDTH-1:0] wbuf_banks  [0:NUM_BANKS-1][0:BANK_DEPTH-1];
    logic [BANK_WIDTH-1:0] ofbuf_banks [0:NUM_BANKS-1][0:BANK_DEPTH-1]; // DUT writes here
    logic [BANK_WIDTH-1:0] gold_banks  [0:NUM_BANKS-1][0:BANK_DEPTH-1]; // Reference

    // ====================================================================
    // Generate Block for Loading Files
    // ====================================================================
    genvar i;
    generate
        for (i = 0; i < NUM_BANKS; i++) begin : SRAM_LOADER
            initial begin
                // 使用 $readmemh 分別讀入對應的 Bank 檔案
                // 檔名格式: ifbuf_bank0.dat, ifbuf_bank1.dat ...
                $readmemh($sformatf("./pattern_banked/ifbuf_bank%0d.dat", i), ifbuf_banks[i]);
                $readmemh($sformatf("./pattern_banked/wbuf_bank%0d.dat",  i), wbuf_banks[i]);
                $readmemh($sformatf("./pattern_banked/ofbuf_golden_bank%0d.dat", i), gold_banks[i]);
                
                // Clear Output Buffer
                for(int k=0; k<BANK_DEPTH; k++) ofbuf_banks[i][k] = '0;
            end
        end
    endgenerate

    // ====================================================================
    // Clock & Simulation Flow
    // ====================================================================
    initial i_clk = 0;
    always #(0.5*`CYCLE) i_clk = ~i_clk;

    initial begin
        $fsdbDumpfile("tb_CE.fsdb");
        $fsdbDumpvars(0, tb_CE);
        
        $display("============================================");
        $display("   Conv Engine TB (16 Banks x 128-bit)      ");
        $display("============================================");

        // Reset
        i_rst_n = 1; i_start = 0;
        #(`CYCLE*5) i_rst_n = 0;
        #(`CYCLE*5) i_rst_n = 1;
        
        // Start
        $display("[TB] Starting Engine...");
        #(`CYCLE) i_start = 1;
        #(`CYCLE) i_start = 0;
        
        // Wait Done
        wait(o_done);
        $display("[TB] Done Signal Received.");
        
        // Check
        check_results();
        $finish;
    end
    
    // Watchdog
    initial begin
        #(`MAX_CYCLE * `CYCLE);
        $display("[Error] Timeout!");
        $finish;
    end

    // ====================================================================
    // Verification Task
    // ====================================================================
    task check_results;
        int err_cnt = 0;
        int mismatch_found;
        logic [BANK_WIDTH-1:0] dut_val, gold_val;
        logic [WORD_WIDTH-1:0] dut_w, gold_w;
        
        $display("[Check] Verifying Output Banks...");
        
        // 逐一檢查每一個 Address
        for (int addr = 0; addr < BANK_DEPTH; addr++) begin
            mismatch_found = 0;
            
            // 檢查該 Address 下的每一個 Bank
            for (int bank = 0; bank < NUM_BANKS; bank++) begin
                dut_val  = ofbuf_banks[bank][addr];
                gold_val = gold_banks[bank][addr];
                
                if (dut_val !== gold_val) begin
                    // 為了除錯方便，我們進一步印出是哪一個 Word 錯了
                    for (int w = 0; w < WORDS_PER_BANK; w++) begin
                        dut_w  = dut_val[w*WORD_WIDTH +: WORD_WIDTH];
                        gold_w = gold_val[w*WORD_WIDTH +: WORD_WIDTH];
                        
                        if (dut_w !== gold_w && (dut_w != 0 || gold_w != 0)) begin
                            if (err_cnt < 20) begin
                                $display("[Err] Addr: %0d | Bank: %0d | Word: %0d | DUT: %h | Gold: %h", 
                                         addr, bank, w, dut_w, gold_w);
                            end
                            err_cnt++;
                        end
                    end
                end
            end
        end
        
        if (err_cnt == 0) begin
            $display("============================================");
            $display("   PASS: All Bank outputs match!            ");
            $display("============================================");
        end else begin
            $display("============================================");
            $display("   FAIL: Found %0d mismatches.              ", err_cnt);
            $display("============================================");
        end
    endtask

    // ====================================================================
    // DUT Mockup (測試用)
    // ====================================================================
    // 這裡模擬 DUT 行為：當 start 後，把 Golden 寫入 ofbuf_banks
    // 你的真實 DUT 應該連接 ifbuf_banks/wbuf_banks 的 output ports
    // 並驅動 ofbuf_banks 的 write enable/data ports
    initial begin
        o_done = 0;
        forever begin
            @(posedge i_clk);
            if (i_start) begin
                repeat(50) @(posedge i_clk); // 模擬計算時間
                
                // Cheat: Copy golden to output
                for (int b = 0; b < NUM_BANKS; b++) begin
                    for (int a = 0; a < BANK_DEPTH; a++) begin
                        ofbuf_banks[b][a] = gold_banks[b][a];
                    end
                end
                
                o_done = 1;
                @(posedge i_clk) o_done = 0;
            end
        end
    end

endmodule