/* 
    tb_PEG.sv - Check the correctness of Processing Engine Group with random generated data
*/
`timescale 1ns/10ps

`define CYCLE               10.0
`define MAX_CYCLE           10000
`define TEST_SAMPLE         1

import Param_def::*;

module tb_PEG;

    // // Dump Signal
    // initial begin
    //     $dumpfile("tb_CE_PEG.vcd");
    //     $dumpvars;
    // end
    initial begin 
        $fsdbDumpfile("tb_PEG.fsdb");
        $fsdbDumpvars(0, "+mda+packedOnly");
    end


    parameter INPUT_NUM = 64;
    parameter WEIGHT_NUM = 5;

    
    // Initialize clock
    logic i_clk;
    initial              i_clk = 0;
    always #(0.5*`CYCLE) i_clk = ~i_clk; 

    // Reset and termination condition
    logic   i_rst_n;
    initial i_rst_n = 1;
    logic   sim_begin;
    initial sim_begin = 0;

    // Load test data
    logic   [INPUT_NUM  - 1 : 0][CLPX_FP_WIDTH-1 : 0]  input_data;
    logic   [WEIGHT_NUM - 1 : 0][CLPX_FP_WIDTH-1 : 0]  weight_data;

    logic   [CLPX_FP_WIDTH - 1 : 0] buff_input  [0 : INPUT_NUM-1];
    logic   [CLPX_FP_WIDTH - 1 : 0] buff_weight [0 : WEIGHT_NUM-1];

    initial begin
        $display("Loading test data...");

        // Load input 
        $readmemh("./pattern_fp16/Layer0_1to8/input.dat", buff_input);
        for (int i = 0; i < INPUT_NUM; i++) begin
            input_data[i] = buff_input[i];
        end

        // Load weights 
        $readmemh("./pattern_fp16/Layer0_1to8/dw_weight.dat", buff_weight);
        for (int i = 0; i < WEIGHT_NUM; i++) begin
            weight_data[i] = buff_weight[i];
        end

        $display("Test data loaded successfully.");
        $display("Start Simulation.");

        #(`CYCLE*3)     i_rst_n     = 0;
        #(`CYCLE)       i_rst_n     = 1;
                        sim_begin   = 1;
    end
    
    initial begin
        #(`MAX_CYCLE * `CYCLE)
        $display("Runtime exceeded!");
        $finish;
    end

    // PEG Interface
    logic   [CLPX_FP_WIDTH-1 : 0] i_PEG_dat0;
    logic   [CLPX_FP_WIDTH-1 : 0] i_PEG_dat1;
    CE_mode                       i_PE_mode; // 0: CMAC 1: CACC
    logic                         i_PEG_valid;
    logic                         o_PEG_valid;
    logic   [CLPX_FP_WIDTH-1 : 0] o_PEG_data;


    CE_PEG u_CEPEG(.*);

    initial begin
        i_PEG_valid = 0;
        i_PEG_dat0  = 0;
        i_PEG_dat1  = 0;
        i_PE_mode = CE_ACCM; // Set to CACC mode
    end

    // Simulation Control
    typedef enum integer { 
        IDLE,
        CALC,
        COMPLETE
    } Sim_phase;
    
    typedef enum integer { 
        SET_CONFIG,
        DATA_STREAM,
        WAIT_RESULT
    } Sub_phase;

    Sim_phase sim_phase;
    Sub_phase sub_phase;
    int       sample_cnt;
    int       stream_cnt;
    
    initial begin
        sim_phase  = IDLE;
        sub_phase  = SET_CONFIG;
        sample_cnt = 0;
        stream_cnt = 0;
    end

    // Simulation
    always @(negedge i_clk) begin
        // Set default control signal
        i_PEG_valid = 0;
        
        case (sim_phase)
            IDLE: begin
                if (sim_begin) begin
                    sim_phase = CALC;
                    sub_phase = SET_CONFIG;
                end
            end
            
            CALC: begin
                case (sub_phase)
                    SET_CONFIG: begin
                        $display("Setting config for PEG calculation"); 
                        sub_phase  = DATA_STREAM;
                        stream_cnt = 0;
                    end
                    
                    DATA_STREAM: begin
                        if (stream_cnt == 0) begin
                            $display("Start data streaming.");
                        end 

                        i_PEG_valid = 1;
                        i_PEG_dat0  = input_data[stream_cnt];
                        i_PEG_dat1  = weight_data[stream_cnt];
                        
                        stream_cnt++;
                        
                        // Check if all data streamed
                        if (stream_cnt >= INPUT_NUM) begin
                            sample_cnt++;
                            sub_phase = WAIT_RESULT;
                            $display("Data streaming completed. Total samples: %0d", stream_cnt);
                        end
                    end
                    
                    WAIT_RESULT: begin
                        if (o_PEG_valid) begin
                            $display("PEG output valid received.");
                            sim_phase  = COMPLETE;
                            sample_cnt = 0;
                            sub_phase  = SET_CONFIG;
                        end
                    end
                endcase
            end
            
            COMPLETE: begin
                // Stay in COMPLETE state, wait for dump to finish
            end
        endcase
    end

    // Dump results for further check
    typedef enum integer { 
        WAIT_DUMP,
        DUMPING,
        DUMP_END
    } Dump_phase;
    
    Dump_phase dump_phase;
    logic [CLPX_FP_WIDTH-1 : 0] o_PEG_dump;
    
    initial begin
        dump_phase = WAIT_DUMP;
        o_PEG_dump = 0;
    end        

    integer f_PEG;
    initial begin
        f_PEG = $fopen("CEPEG_output.dat", "w");
        if (f_PEG == 0) begin
            $display("Error: Cannot open output file!");
            $finish;
        end
    end

    always @(negedge i_clk) begin
        case (dump_phase)
            WAIT_DUMP: begin
                if (o_PEG_valid) begin
                    o_PEG_dump = o_PEG_data;
                    dump_phase = DUMPING;
                    $display("Capturing PEG output data...");
                end
            end
            
            DUMPING: begin
                $fwrite(f_PEG, "%h\n", o_PEG_dump);
                dump_phase = DUMP_END;
            end
            
            DUMP_END: begin
                $fclose(f_PEG);
                $display("========================================");
                $display("Stream count: %d", stream_cnt);
                $display("========================================");
                $display("Successfully dumped PEG results to CEPEG_output.dat");
                $display("Further verification using Python script is needed.");
                $display("Simulation completed!");
                $finish;
            end
        endcase
    end

endmodule