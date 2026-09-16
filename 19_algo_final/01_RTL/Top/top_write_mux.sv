/**********************************************************
* Filename: top_write_mux.sv
* Description: read data mux (choose the channel: if/w/b)
* Parameters:
* Note: 
* Review History: 2023 
**********************************************************/

module top_write_mux 
    import NPU_def::*;
(
    input  logic i_clk,
    input  logic i_rst_n,

    // top store command signals
    // input  logic i_io_top_store_OFp_valid,
    // output logic o_io_top_store_OFp_ready,
    // output logic o_io_top_store_idle,

    // engine store command signals
    output logic tc_engine_store_OF_valid,
    input  logic tc_engine_store_OF_ready,
    input  logic tc_engine_store_OF_idle,

    // TDMA write channel
    output logic                   o_io_tdma_wr_valid,
    input  logic                   i_io_tdma_wr_ready,
    output logic [OTDMA_WIDTH-1:0] o_io_tdma_wr_data,

    // engine output feature channel
    input  logic                   tc_engine_OF_valid,
    output logic                   tc_engine_OF_ready,
    input  logic [OTDMA_WIDTH-1:0] tc_engine_OF_data

);
    logic io_tdma_wr_valid_w, io_tdma_wr_valid_r;
    assign o_io_tdma_wr_valid = io_tdma_wr_valid_w;
    logic [OTDMA_WIDTH-1:0] io_tdma_wr_data_r, io_tdma_wr_data_w;
    assign o_io_tdma_wr_data = io_tdma_wr_data_w;

    logic engine_store_OF_valid_r, engine_store_OF_valid_w;
    assign tc_engine_store_OF_valid = engine_store_OF_valid_r;

    // engine write data MUX states
    typedef enum logic [1:0] {
        S_IDLE,
        S_engine_OF
    } State;
    State state_r, state_w;

    // engine write data MUX
    always_comb begin
        io_tdma_wr_valid_w = 0;
        io_tdma_wr_data_w  = 0;
        tc_engine_OF_ready = 0;

        if (state_r == S_engine_OF) begin // 跟 read_mux 功能相反
            io_tdma_wr_valid_w = tc_engine_OF_valid;
            tc_engine_OF_ready = i_io_tdma_wr_ready;
            io_tdma_wr_data_w  = tc_engine_OF_data;
        end
    end

    // engine read data MUX FSM
    always_comb begin
        // top_store_OF_ready_w  = 0;
        engine_store_OF_valid_w = engine_store_OF_valid_r;
        state_w                      = state_r;

        case (state_r)
            S_IDLE: begin
                engine_store_OF_valid_w = ~(tc_engine_store_OF_valid & tc_engine_store_OF_ready);
                // top_store_OF_ready_w  = ~engine_store_OF_valid_w;
                state_w                      = ~engine_store_OF_valid_w ? S_engine_OF : state_r;
            end
            S_engine_OF: begin // for TDMA output
                state_w = tc_engine_store_OF_idle ? S_IDLE : state_r;
            end

        endcase
    end
    always_ff @ (posedge i_clk or negedge i_rst_n) begin
        if (~i_rst_n) begin
            // top_store_OF_ready_r  <= 0;
            engine_store_OF_valid_r <= 0;
            state_r                      <= S_IDLE;
            // io_tdma_wr_data_r            <= 0;
            // io_tdma_wr_valid_r           <= 0;
        end
        else begin
            // top_store_OF_ready_r  <= top_store_OF_ready_w;
            engine_store_OF_valid_r <= engine_store_OF_valid_w;
            state_r                      <= state_w;
            // io_tdma_wr_data_r            <= io_tdma_wr_data_w;
            // io_tdma_wr_valid_r           <= io_tdma_wr_valid_w;
        end
    end

endmodule