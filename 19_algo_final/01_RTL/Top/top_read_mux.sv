/**********************************************************
* Filename: top_read_mux.sv
* Description: read data mux (choose the channel: if/w)
* Parameters:
* Note: 
* Review History: 2023 
**********************************************************/


// control the data to each channel, some data like shared exp must be saved into extra buffer
module top_read_mux 
    import NPU_def::*;
(
    input  logic i_clk,
    input  logic i_rst_n,

    // top load command signals   
    input  logic [2:0] i_io_top_npu_cmd, // LOAD MEM => 3'd0: IF, 3'd1: W
    input  logic       i_io_top_lcmd_valid,
    // input  logic       i_io_top_exec_valid,

    // engine load command signals (tc = top control)
    output logic tc_engine_load_IF_valid,
    input  logic tc_engine_load_IF_ready,
    input  logic tc_engine_load_IF_idle,
    output logic tc_engine_load_W_valid,
    input  logic tc_engine_load_W_ready,
    input  logic tc_engine_load_W_idle,

    // TDMA read channel
    input  logic [ITDMA_WIDTH-1:0] i_io_tdma_rd_data,
    output logic                   o_io_tdma_rd_ready,
    input  logic                   i_io_tdma_rd_valid,

    // engine input feature channel
    output logic                   tc_engine_IF_valid,
    input  logic                   tc_engine_IF_ready,
    output logic [ITDMA_WIDTH-1:0] tc_engine_IF_data,

    // engine weight channel
    output logic                   tc_engine_W_valid,
    input  logic                   tc_engine_W_ready,
    output logic [ITDMA_WIDTH-1:0] tc_engine_W_data

    // output logic                   tc_engine_load_w_valid,
    // output logic                   tc_engine_load_if_valid

    // input valid control
    // output logic                  in_if_sram_valid,
    // output logic                  in_w_sram_valid
);

    logic  engine_load_IF_valid_r, engine_load_IF_valid_w;
    logic  engine_load_W_valid_r,  engine_load_W_valid_w;
    assign tc_engine_load_IF_valid =  engine_load_IF_valid_r;
    assign tc_engine_load_W_valid  =  engine_load_W_valid_r;

    logic                   tc_engine_IF_valid_w, tc_engine_IF_valid_r;
    logic [ITDMA_WIDTH-1:0] tc_engine_IF_data_w, tc_engine_IF_data_r;
    logic                   tc_engine_W_valid_w, tc_engine_W_valid_r;
    logic [ITDMA_WIDTH-1:0] tc_engine_W_data_w, tc_engine_W_data_r;

    assign tc_engine_IF_valid = tc_engine_IF_valid_r;
    assign tc_engine_IF_data = tc_engine_IF_data_r;
    assign tc_engine_W_valid = tc_engine_W_valid_r;
    assign tc_engine_W_data = tc_engine_W_data_r;


    // engine read data MUX states
    typedef enum logic [1:0] {
        S_IDLE,
        S_READ_IF,
        S_READ_W
    } State;
    State state_r, state_w;
    // assign o_top_load_idle = (state_r == S_IDLE);
    
    // assign tc_engine_load_w_valid  = (state_r == S_READ_W);
    // assign tc_engine_load_if_valid = (state_r == S_READ_IF);

    // top read data MUX
    always_comb begin
        o_io_tdma_rd_ready   = 0;
        tc_engine_IF_valid_w = tc_engine_IF_valid_r;
        tc_engine_IF_data_w  = tc_engine_IF_data_r;
        tc_engine_W_valid_w  = tc_engine_W_valid_r;
        tc_engine_W_data_w   = tc_engine_W_data_r;
        // in_w_sram_valid_w  = 0;
        // in_if_sram_valid_w = 0;

        case(state_r)
            S_READ_IF: begin
                // in_if_sram_valid_w = 1;
                // in_w_sram_valid_w  = 0;
                tc_engine_IF_valid_w = i_io_tdma_rd_valid;
                o_io_tdma_rd_ready   = tc_engine_IF_ready;
                tc_engine_IF_data_w  = i_io_tdma_rd_data;
            end
            S_READ_W: begin
                // in_w_sram_valid_w  = 1;
                // in_if_sram_valid_w = 0;
                tc_engine_W_valid_w = i_io_tdma_rd_valid;
                o_io_tdma_rd_ready  = tc_engine_W_ready;
                tc_engine_W_data_w  = i_io_tdma_rd_data;
            end
        endcase
    end
    always_ff @ (posedge i_clk or negedge i_rst_n) begin
        if (~i_rst_n) begin
            tc_engine_IF_valid_r <= 0;
            tc_engine_IF_data_r  <= 0;
            tc_engine_W_valid_r  <= 0;
            tc_engine_W_data_r   <= 0;
        end
        else begin
            tc_engine_IF_valid_r <= tc_engine_IF_valid_w;
            tc_engine_IF_data_r  <= tc_engine_IF_data_w;
            tc_engine_W_valid_r  <= tc_engine_W_valid_w;
            tc_engine_W_data_r   <= tc_engine_W_data_w;
        end
    end

    // top read data MUX FSM
    always_comb begin
        engine_load_IF_valid_w = engine_load_IF_valid_r;
        engine_load_W_valid_w  = engine_load_W_valid_r;
        state_w                = state_r;

        case (state_r)
            S_IDLE: begin
                if (i_io_top_lcmd_valid) begin
                    case (i_io_top_npu_cmd) // synopsys parallel_case
                        3'd0: begin
                            engine_load_IF_valid_w = ~(tc_engine_load_IF_valid & tc_engine_load_IF_ready);
                            state_w                = ~engine_load_IF_valid_w ? S_READ_IF : state_r;
                        end
                        3'd1: begin
                            engine_load_W_valid_w = ~(tc_engine_load_W_valid & tc_engine_load_W_ready);
                            state_w               = ~engine_load_W_valid_w ? S_READ_W : state_r;
                        end
                    endcase
                end
            end
            S_READ_IF: begin
                if (tc_engine_load_IF_idle) begin
                    engine_load_IF_valid_w = 0;
                    state_w = S_IDLE;
                end
            end
            S_READ_W: begin
                if (tc_engine_load_W_idle) begin
                    engine_load_W_valid_w = 0;
                    state_w = S_IDLE;
                end
            end
        endcase
    end
    always_ff @ (posedge i_clk or negedge i_rst_n) begin
        if (~i_rst_n) begin
            engine_load_IF_valid_r <= 0;
            engine_load_W_valid_r  <= 0;
            state_r                <= S_IDLE;
        end
        else begin
            engine_load_IF_valid_r <= engine_load_IF_valid_w;
            engine_load_W_valid_r  <= engine_load_W_valid_w;
            state_r                <= state_w;
        end
    end
    
endmodule