/**********************************************************
* Filename: top.sv
* Description: read data mux (choose the channel: if/w/b)
* Parameters:
* Note: 
* Review History: 2024 
**********************************************************/

module top 
    import NPU_def::*;
(
    // 3+64+16+8+4+9=104
    input  logic                        i_clk,
    input  logic                        i_rst_n,

    // TDMA read channel
    input  logic [     ITDMA_WIDTH-1:0] i_io_tdma_rd_data,
    input  logic                        i_io_tdma_rd_valid,
    output logic                        o_io_tdma_rd_ready,

    input  logic [                 2:0] i_io_top_npu_cmd,
    input  logic                        i_io_top_lcmd_valid,
    input  logic                        i_io_top_exec_valid,  // execution channel valid
    output logic                        o_io_top_exec_ready,  // execution channel ready

    // TDMA write channel
    output logic [     OTDMA_WIDTH-1:0] o_io_tdma_wr_data,
    output logic                        o_io_tdma_wr_valid,
    input  logic                        i_io_tdma_wr_ready,

    input  logic [DEBUG_MODE_WIDTH-1:0] i_io_debug_cmd,
    output logic [DEBUG_DATA_WIDTH-1:0] o_io_debug_data
   
);
    logic [DEBUG_DATA_WIDTH-1:0] core_ctrl_debug_data;
    logic [DEBUG_DATA_WIDTH-1:0] core_ctrl_ifbuf_debug_data;
    logic [DEBUG_DATA_WIDTH-1:0] core_ctrl_wbuf_debug_data;
    logic [DEBUG_DATA_WIDTH-1:0] core_ctrl_ofbuf_debug_data;
    logic [DEBUG_DATA_WIDTH-1:0] core_mac_array1_debug_data;
    logic [DEBUG_DATA_WIDTH-1:0] core_mac_array2_debug_data;
    logic [DEBUG_DATA_WIDTH-1:0] core_dispatcher1_debug_data;
    logic [DEBUG_DATA_WIDTH-1:0] core_dispatcher2_debug_data;
    logic [DEBUG_DATA_WIDTH-1:0] spec_eope_debug_data;
    logic [DEBUG_DATA_WIDTH-1:0] spec_qtk_add_debug_data;
    logic [DEBUG_DATA_WIDTH-1:0] spec_qt_quant_debug_data;
    logic [DEBUG_DATA_WIDTH-1:0] spec_filter_debug_data;
    logic [DEBUG_DATA_WIDTH-1:0] tensor_engine1_debug_data;
    logic [DEBUG_DATA_WIDTH-1:0] tensor_engine2_debug_data;
    logic [DEBUG_DATA_WIDTH-1:0] ifbuf_debug_data_top;
    logic [DEBUG_DATA_WIDTH-1:0] ifbuf_debug_data_ctrl;
    logic [DEBUG_DATA_WIDTH-1:0] ifbuf_debug_data_sram;
    logic [DEBUG_DATA_WIDTH-1:0] wbuf_debug_data_top;
    logic [DEBUG_DATA_WIDTH-1:0] wbuf_debug_data_ctrl;
    logic [DEBUG_DATA_WIDTH-1:0] wbuf_debug_data_sram;
    logic [DEBUG_DATA_WIDTH-1:0] ofbuf_debug_data_top;
    logic [DEBUG_DATA_WIDTH-1:0] ofbuf_debug_data_ctrl;
    logic [DEBUG_DATA_WIDTH-1:0] spec_bitmask;

    logic [DEBUG_DATA_WIDTH-1:0] debug_data_w, debug_data_r;
    assign o_io_debug_data = debug_data_r;

    logic [DEBUG_MODE_WIDTH-1:0] debug_cmd_r;

    logic io_tdma_rd_ready_w, io_tdma_rd_ready_r;
    assign o_io_tdma_rd_ready = io_tdma_rd_ready_r;

    logic io_top_exec_ready_w, io_top_exec_ready_r;
    assign o_io_top_exec_ready = io_top_exec_ready_r;

    logic [OTDMA_WIDTH-1:0] io_tdma_wr_data_w, io_tdma_wr_data_r;
    assign o_io_tdma_wr_data = io_tdma_wr_data_r;

    logic io_tdma_wr_valid_w, io_tdma_wr_valid_r;
    assign o_io_tdma_wr_valid = io_tdma_wr_valid_r;

    // engine load control signals
    logic tc_engine_load_IF_valid;
    logic tc_engine_load_IF_ready;
    logic tc_engine_load_IF_idle;
    logic tc_engine_load_W_valid;
    logic tc_engine_load_W_ready;
    logic tc_engine_load_W_idle;

    // engine input feature channel
    logic                   tc_engine_IF_valid;
    logic                   tc_engine_IF_ready;
    logic [ITDMA_WIDTH-1:0] tc_engine_IF_data;

    // engine weight channel
    logic                   tc_engine_W_valid;
    logic                   tc_engine_W_ready;
    logic [ITDMA_WIDTH-1:0] tc_engine_W_data;

    // logic tc_engine_load_w_valid;
    // logic tc_engine_load_if_valid;

    // each data has its own ready and valid signal
    // engine read data MUX module instance
    top_read_mux u_top_read_mux (
        .i_clk                   (i_clk                   ),
        .i_rst_n                 (i_rst_n                 ),

        .i_io_top_npu_cmd        (i_io_top_npu_cmd        ),
        .i_io_top_lcmd_valid     (i_io_top_lcmd_valid     ),

        .tc_engine_load_IF_valid (tc_engine_load_IF_valid ),
        .tc_engine_load_IF_ready (tc_engine_load_IF_ready ),
        .tc_engine_load_IF_idle  (tc_engine_load_IF_idle  ),

        .tc_engine_load_W_valid  (tc_engine_load_W_valid  ),
        .tc_engine_load_W_ready  (tc_engine_load_W_ready  ),
        .tc_engine_load_W_idle   (tc_engine_load_W_idle   ),

        .i_io_tdma_rd_data       (i_io_tdma_rd_data       ),
        .o_io_tdma_rd_ready      (io_tdma_rd_ready_w      ),
        .i_io_tdma_rd_valid      (i_io_tdma_rd_valid      ),

        .tc_engine_IF_valid      (tc_engine_IF_valid      ),
        .tc_engine_IF_ready      (tc_engine_IF_ready      ),
        .tc_engine_IF_data       (tc_engine_IF_data       ),

        .tc_engine_W_valid       (tc_engine_W_valid       ),
        .tc_engine_W_ready       (tc_engine_W_ready       ),
        .tc_engine_W_data        (tc_engine_W_data        )
        
        // .tc_engine_load_w_valid  (tc_engine_load_w_valid ),
        // .tc_engine_load_if_valid (tc_engine_load_if_valid)

    );

    // engine store control signals
    logic tc_engine_store_OF_valid;
    logic tc_engine_store_OF_ready;
    logic tc_engine_store_OF_idle;

    // engine output feature channel
    logic                   tc_engine_OF_valid;
    logic                   tc_engine_OF_ready;
    logic [OTDMA_WIDTH-1:0] tc_engine_OF_data;


    // // each data has its own ready and valid signal
    // // engine write data MUX module instance
    top_write_mux u_top_write_mux (
        .i_clk                    (i_clk                    ),
        .i_rst_n                  (i_rst_n                  ),

        .tc_engine_store_OF_valid (tc_engine_store_OF_valid ),
        .tc_engine_store_OF_ready (tc_engine_store_OF_ready ),
        .tc_engine_store_OF_idle  (tc_engine_store_OF_idle  ),

        .o_io_tdma_wr_valid       (io_tdma_wr_valid_w       ),
        .i_io_tdma_wr_ready       (i_io_tdma_wr_ready       ),
        .o_io_tdma_wr_data        (io_tdma_wr_data_w        ),

        .tc_engine_OF_valid       (tc_engine_OF_valid       ),
        .tc_engine_OF_ready       (tc_engine_OF_ready       ),
        .tc_engine_OF_data        (tc_engine_OF_data        )
    );

    // // engine execution control signals
    logic tc_engine_exec_valid;
    logic tc_engine_exec_ready;
    logic tc_engine_exec_idle;

    logic [2:0] exec_cmd;

    // // engine execution control MUX module instance
    top_exe_mux u_top_exe_mux (
        .i_clk                 (i_clk                ),
        .i_rst_n               (i_rst_n              ),

        .i_io_top_npu_cmd      (i_io_top_npu_cmd     ),
        .exec_cmd              (exec_cmd             ),
        .i_io_top_exec_valid   (i_io_top_exec_valid  ),
        .o_io_top_exec_ready   (io_top_exec_ready_w  ),

        .tc_engine_exec_valid  (tc_engine_exec_valid ),
        .tc_engine_exec_ready  (tc_engine_exec_ready ),
        .tc_engine_exec_idle   (tc_engine_exec_idle  )
    );

    // engine module instance
    engine u_engine (
        .i_clk                       (i_clk                      ),
        .i_rst_n                     (i_rst_n                    ),

        .tc_IF_load_valid            (tc_engine_load_IF_valid    ),
        .tc_IF_load_ready            (tc_engine_load_IF_ready    ),
        .tc_IF_load_idle             (tc_engine_load_IF_idle     ),
        .tc_W_load_valid             (tc_engine_load_W_valid     ),
        .tc_W_load_ready             (tc_engine_load_W_ready     ),
        .tc_W_load_idle              (tc_engine_load_W_idle      ),

        .tc_IF_valid                 (tc_engine_IF_valid         ),
        .tc_IF_ready                 (tc_engine_IF_ready         ),
        .tc_IF_data                  (tc_engine_IF_data          ),

        .tc_W_valid                  (tc_engine_W_valid          ),
        .tc_W_ready                  (tc_engine_W_ready          ),
        .tc_W_data                   (tc_engine_W_data           ),

        .tc_OF_store_valid           (tc_engine_store_OF_valid   ),
        .tc_OF_store_ready           (tc_engine_store_OF_ready   ),
        .tc_OF_store_idle            (tc_engine_store_OF_idle    ),

        .tc_OF_valid                 (tc_engine_OF_valid         ),
        .tc_OF_ready                 (tc_engine_OF_ready         ),
        .tc_OF_data                  (tc_engine_OF_data          ),

        .tc_exec_valid               (tc_engine_exec_valid       ),
        .tc_exec_ready               (tc_engine_exec_ready       ),
        .tc_exec_idle                (tc_engine_exec_idle        ),
        .top_npu_cmd                 (i_io_top_npu_cmd           ),
        .exec_cmd                    (exec_cmd                   ),
        .top_debug_cmd               (debug_cmd_r                ),

        .core_ctrl_debug_data        (core_ctrl_debug_data       ),
        .core_ctrl_ifbuf_debug_data  (core_ctrl_ifbuf_debug_data ),
        .core_ctrl_wbuf_debug_data   (core_ctrl_wbuf_debug_data  ),
        .core_ctrl_ofbuf_debug_data  (core_ctrl_ofbuf_debug_data ),
        .core_mac_array1_debug_data  (core_mac_array1_debug_data ),
        .core_mac_array2_debug_data  (core_mac_array2_debug_data ),
        .core_dispatcher1_debug_data (core_dispatcher1_debug_data),
        .core_dispatcher2_debug_data (core_dispatcher2_debug_data),
        .spec_eope_debug_data        (spec_eope_debug_data       ),
        .spec_qtk_add_debug_data     (spec_qtk_add_debug_data    ),
        .spec_qt_quant_debug_data    (spec_qt_quant_debug_data   ),
        .spec_filter_debug_data      (spec_filter_debug_data     ),
        .tensor_engine1_debug_data   (tensor_engine1_debug_data  ),
        .tensor_engine2_debug_data   (tensor_engine2_debug_data  ),
        .ifbuf_debug_data_top        (ifbuf_debug_data_top       ),
        .ifbuf_debug_data_ctrl       (ifbuf_debug_data_ctrl      ),
        .ifbuf_debug_data_sram       (ifbuf_debug_data_sram      ),
        .wbuf_debug_data_top         (wbuf_debug_data_top        ),
        .wbuf_debug_data_ctrl        (wbuf_debug_data_ctrl       ),
        .wbuf_debug_data_sram        (wbuf_debug_data_sram       ),
        .ofbuf_debug_data_top        (ofbuf_debug_data_top       ),
        .ofbuf_debug_data_ctrl       (ofbuf_debug_data_ctrl      ),
        .spec_bitmask                (spec_bitmask               )
    );

    // debug counter
    logic [DEBUG_DATA_WIDTH-1:0] debug_cnt_r, debug_cnt_w;
    assign debug_cnt_w = debug_cnt_r + 1;
    always_ff @( posedge i_clk or negedge i_rst_n ) begin 
        if (~i_rst_n) begin
            debug_cnt_r <= 0;
        end
        else begin
            debug_cnt_r <= debug_cnt_w;
        end
    end

    always_comb begin
        debug_data_w = debug_data_r;
        case (debug_cmd_r)
            DM_CORE_CTRL:        debug_data_w = core_ctrl_debug_data;
            DM_CORE_CTR_IFBUF:   debug_data_w = core_ctrl_ifbuf_debug_data;
            DM_CORE_CTRL_WBUF:   debug_data_w = core_ctrl_wbuf_debug_data;
            DM_CORE_CTRL_OFBUF:  debug_data_w = core_ctrl_ofbuf_debug_data;
            DM_CORE_MAC1:        debug_data_w = core_mac_array1_debug_data;
            DM_CORE_MAC2:        debug_data_w = core_mac_array2_debug_data;
            DM_CORE_DPR1:        debug_data_w = core_dispatcher1_debug_data;
            DM_CORE_DPR2:        debug_data_w = core_dispatcher2_debug_data;
            DM_SPEC_EOPE:        debug_data_w = spec_eope_debug_data;
            DM_SPEC_QTKA:        debug_data_w = spec_qtk_add_debug_data;
            DM_SPEC_QTQ:         debug_data_w = spec_qt_quant_debug_data;
            DM_SPEC_FILT:        debug_data_w = spec_filter_debug_data;
            DM_SPEC_TE1:         debug_data_w = tensor_engine1_debug_data;
            DM_SPEC_TE2:         debug_data_w = tensor_engine2_debug_data;
            DM_IFBUF_TOP:        debug_data_w = ifbuf_debug_data_top;
            DM_IFBUF_CTRL:       debug_data_w = ifbuf_debug_data_ctrl;
            DM_IFBUF_SRAM:       debug_data_w = ifbuf_debug_data_sram;
            DM_WBUF_TOP:         debug_data_w = wbuf_debug_data_top;
            DM_WBUF_CTRL:        debug_data_w = wbuf_debug_data_ctrl;
            DM_WBUF_SRAM:        debug_data_w = wbuf_debug_data_sram;
            DM_OFBUF_TOP:        debug_data_w = ofbuf_debug_data_top;
            DM_OFBUF_CTRL:       debug_data_w = ofbuf_debug_data_ctrl;
            DM_TOP_CNT:          debug_data_w = debug_cnt_r;
            DM_SPEC_BM:          debug_data_w = spec_bitmask;
            DM_JUMP_SPEC1:       debug_data_w = spec_eope_debug_data;
            DM_JUMP_SPEC2:       debug_data_w = spec_qtk_add_debug_data;
        endcase
    end

    always_ff @ (posedge i_clk or negedge i_rst_n) begin
        if (~i_rst_n) begin
            debug_cmd_r         <= 0;
            debug_data_r        <= 0;
            io_tdma_rd_ready_r  <= 0;
            io_top_exec_ready_r <= 0;
            io_tdma_wr_data_r   <= 0;
            io_tdma_wr_valid_r  <= 0;
        end
        else begin
            debug_cmd_r         <= i_io_debug_cmd;
            debug_data_r        <= debug_data_w;
            io_tdma_rd_ready_r  <= io_tdma_rd_ready_w;
            io_top_exec_ready_r <= io_top_exec_ready_w;
            io_tdma_wr_data_r   <= io_tdma_wr_data_w;
            io_tdma_wr_valid_r  <= io_tdma_wr_valid_w;
        end
    end


endmodule