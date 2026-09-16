/**********************************************************
* Filename: top_exe_mux.sv
* Description: output if/w to MAC for executing
* Parameters:
* Note: 
* Review History: 2024
**********************************************************/


module top_exe_mux 
    import NPU_def::*;
(
    input  logic i_clk,
    input  logic i_rst_n,

    input  logic [2:0] i_io_top_npu_cmd, // EXEC => 3'd0: Q, 3'd1: K, 3'd2: V, 3'd3: S, 3'd4: O
    output logic [2:0] exec_cmd,


    // top execution control signals
    input  logic i_io_top_exec_valid,
    output logic o_io_top_exec_ready,
    // output logic o_top_exec_idle,

    // engine execution control signals
    output logic tc_engine_exec_valid,
    input  logic tc_engine_exec_ready,
    input  logic tc_engine_exec_idle

);
    // output assignments
    // logic top_exec_ready_r, top_exec_ready_w;
    // assign o_top_exec_ready = top_exec_ready_r;

    logic engine_exec_valid_r, engine_exec_valid_w;
    assign tc_engine_exec_valid = engine_exec_valid_r;

    // if layer type is Conv2d, ConvTranspose2d, or local-max filter,
    //     use engine, otherwise use simd_engine
    // logic layer_is_conv;
    // assign layer_is_conv = (layer_type == `PRESTO_LAYER_TYPE_CONV2D) ||
    //                        (layer_type == `PRESTO_LAYER_TYPE_CONVTRANSPOSE2D) ||
    //                        (layer_type == `PRESTO_LAYER_TYPE_LMAXFILT);

    // engine execution MUX states
    typedef enum logic [1:0] {
        S_IDLE,
        S_EX
        // S_VPU
    } State;
    State state_r, state_w;
    // assign o_top_exec_idle = (state_r == S_IDLE);

    logic io_top_exec_valid_r;
    logic [2:0] exec_cmd_w, exec_cmd_r;
    assign exec_cmd = exec_cmd_r;


    // engine execution MUX FSM
    always_comb begin
        o_io_top_exec_ready = tc_engine_exec_ready;
        // o_io_top_exec_ready  = 0;
        engine_exec_valid_w  = engine_exec_valid_r;
        state_w              = state_r;
        exec_cmd_w           = exec_cmd_r;

        case (state_r)
            S_IDLE: begin
                if (io_top_exec_valid_r) begin
                    engine_exec_valid_w = ~(tc_engine_exec_valid & tc_engine_exec_ready);
                    // top_exec_ready_w    = ~engine_exec_valid_w;
                    state_w             = ~engine_exec_valid_w ? S_EX : state_r;
                    exec_cmd_w          = i_io_top_npu_cmd;
                end
            end
            S_EX: begin
                if (tc_engine_exec_idle) begin
                    state_w = S_IDLE;
                    engine_exec_valid_w = 0;
                end
            end
        endcase
    end
    always_ff @ (posedge i_clk or negedge i_rst_n) begin
        if (~i_rst_n) begin
            // top_exec_ready_r    <= 0;
            io_top_exec_valid_r <= 0;
            engine_exec_valid_r <= 0;
            state_r             <= S_IDLE;
            exec_cmd_r          <= 0;
        end
        else begin
            // top_exec_ready_r    <= top_exec_ready_w;
            io_top_exec_valid_r <= i_io_top_exec_valid;
            engine_exec_valid_r <= engine_exec_valid_w;
            state_r             <= state_w;
            exec_cmd_r          <= exec_cmd_w;
        end
    end
    
endmodule