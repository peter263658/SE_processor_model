`timescale 1ns/1ps

`define CYCLE       50.0
`define MAX_CYCLE   50000

import Param_def::*;
import Mem_def::*;


module tb_wbuf ();


    // Clock and Reset
    logic i_clk;
    logic i_rst_n;

    // Layer execution mode
    Exec_mode     i_exec_mode;
    AE_mode       i_ae_mode;
    CE_mode       i_ce_mode;
    CE_LAYER_mode i_ce_layer;

    // Input loading to SRAM (sequential)
    logic                    i_top_valid;
    logic                    o_top_ready;
    logic [CHIP_I_WIDTH-1:0] i_top_data;

    // WBUF SRAM interface (for output to MAC, parallel, only load no write)
    logic i_load_valid;
    logic o_load_ready;
    logic o_load_valid;

    // MAC control interface
    logic [WBUF_BANKS-1:0]                   i_core_ena;
    logic [WBUF_BANKS-1:0]                   i_core_enb;
    logic [WBUF_BANKS-1:0][WBUF_ADDRW-1:0]   i_core_addra;
    logic [WBUF_BANKS-1:0][WBUF_ADDRW-1:0]   i_core_addrb;
    logic [WBUF_BANKS-1:0][WBUF_WIDTH-1:0]   o_core_douta;
    logic [WBUF_BANKS-1:0][WBUF_WIDTH-1:0]   o_core_doutb;

    // Instantiate wbuf_top
    wbuf_top u_wbuf_top (
        .i_clk         (i_clk),
        .i_rst_n       (i_rst_n),
        .i_exec_mode   (i_exec_mode),
        .i_ae_mode     (i_ae_mode),
        .i_ce_mode     (i_ce_mode),
        .i_ce_layer    (i_ce_layer),
        .i_top_valid   (i_top_valid),
        .o_top_ready   (o_top_ready),
        .i_top_data    (i_top_data),
        .i_load_valid  (i_load_valid),
        .o_load_ready  (o_load_ready),
        .o_load_valid  (o_load_valid),
        .i_core_ena    (i_core_ena),
        .i_core_enb    (i_core_enb),
        .i_core_addra  (i_core_addra),
        .i_core_addrb  (i_core_addrb),
        .o_core_douta  (o_core_douta),
        .o_core_doutb  (o_core_doutb)
    );

    // Clock generation
    initial begin
        i_clk = 0;
        forever #(`CYCLE/2.0) i_clk = ~i_clk;
    end

    // Dump waveform
    initial begin
        $fsdbDumpfile("tb_wbuf.fsdb");
        $fsdbDumpvars(0, tb_wbuf);
        $fsdbDumpMDA();
    end

    // Simulation timeout
    initial begin
        #(`CYCLE * `MAX_CYCLE);
        $display("=================================================");
        $display("  SIMULATION TIMEOUT!!!");
        $display("=================================================");
        $finish;
    end

    localparam TOTAL_LINES = (WBUF_BANKS * WBUF_DEPTH * 64) / CHIP_I_WIDTH;

    // Pattern memory
    logic [CHIP_I_WIDTH-1:0] pattern_mem [0 : TOTAL_LINES-1];
    integer data_cnt;

    // Main Test Sequence
    initial begin
        // Initialize Signals
        i_rst_n      = 0;
        i_exec_mode  = EXEC_ENC;
        i_ae_mode    = AE_IDLE;
        i_ce_mode    = CE_DW;
        i_ce_layer   = CE_L1;

        i_top_valid  = 0;
        i_top_data   = 0;
        i_load_valid = 0;
        i_core_ena   = 0;
        i_core_enb   = 0;
        
        for (int i = 0; i < WBUF_BANKS; i++) begin
            i_core_addra[i] = 0;
            i_core_addrb[i] = 0;
        end
        data_cnt     = 0;

        // Load Patterns
        // Using $readmemh to read 128-bit hex patterns into memory
        $readmemh("pattern/wbuf_in.dat", pattern_mem);

        // Reset
        #20 i_rst_n = 1;

        // Wait a few cycles
        @(posedge i_clk);
        @(posedge i_clk);
        @(posedge i_clk);

        $display("-----------------------------------------");
        $display("-   Start Loading WBUF Data patterns    -");
        $display("-----------------------------------------");

        // 1. Input Loading (Sequential)
        while (data_cnt < TOTAL_LINES) begin
            i_top_valid <= 1'b1;
            i_top_data  <= pattern_mem[data_cnt];
            
            // Wait until the cycle where o_top_ready is high
            do begin
                @(posedge i_clk);
            end while (o_top_ready !== 1'b1);
            
            data_cnt++;
        end

        // Drop valid
        i_top_valid <= 1'b0;

        // Wait for o_top_ready to drop (indicates weights_store_done)
        while (o_top_ready == 1'b1) begin
            @(posedge i_clk);
        end
        $display("Write completed. o_top_ready is low.");

        $display("-----------------------------------------");
        $display("-      Start Reading MAC Data           -");
        $display("-----------------------------------------");
        
        // 2. Output Reading (Parallel)
        // Enable reading for all banks
        @(posedge i_clk);
        i_core_ena   <= {WBUF_BANKS{1'b1}};
        i_core_enb   <= {WBUF_BANKS{1'b1}};

        // Read a few addresses testing the logic
        for (int addr = 0; addr < 10; addr++) begin
            // Drive same address to all banks (both ears calculated concurrently)
            for (int b = 0; b < WBUF_BANKS; b++) begin
                i_core_addra[b] <= addr;
                i_core_addrb[b] <= addr + 1;
            end
            
            i_load_valid <= 1'b1;
            
            // i_load_valid maintains its value until o_load_ready triggers transfer
            @(posedge i_clk);
            while (o_load_ready !== 1'b1) begin
                @(posedge i_clk);
            end
            
            // Wait for data valid output (could be delayed, simplifed inline check)
            if (o_load_valid) begin
                $display("Time: %0t, Addr: %0d, douta[0]: %h", $time, addr, o_core_douta[0]);
            end
        end

        // Finish reading
        @(posedge i_clk);
        i_load_valid <= 1'b0;
        i_core_ena   <= 0;
        i_core_enb   <= 0;

        #100
        $display("Test finished.");
        $finish;
    end


endmodule
