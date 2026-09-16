module CHIP 
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

    logic                        i_clk_pad;              
    logic                        i_rst_n_pad;            
    logic [     ITDMA_WIDTH-1:0] i_io_tdma_rd_data_pad;  
    logic                        i_io_tdma_rd_valid_pad; 
    logic                        o_io_tdma_rd_ready_pad; 
    logic [                 2:0] i_io_top_npu_cmd_pad;   
    logic                        i_io_top_lcmd_valid_pad;
    logic                        i_io_top_exec_valid_pad;
    logic                        o_io_top_exec_ready_pad;
    logic [     OTDMA_WIDTH-1:0] o_io_tdma_wr_data_pad;  
    logic                        o_io_tdma_wr_valid_pad; 
    logic                        i_io_tdma_wr_ready_pad; 
    logic [DEBUG_MODE_WIDTH-1:0] i_io_debug_cmd_pad;     
    logic [DEBUG_DATA_WIDTH-1:0] o_io_debug_data_pad;    

    // core
    top u_top (
        .i_clk               (i_clk_pad              ),
        .i_rst_n             (i_rst_n_pad            ),
        .i_io_tdma_rd_data   (i_io_tdma_rd_data_pad  ),
        .i_io_tdma_rd_valid  (i_io_tdma_rd_valid_pad ),
        .o_io_tdma_rd_ready  (o_io_tdma_rd_ready_pad ),
        .i_io_top_npu_cmd    (i_io_top_npu_cmd_pad   ),
        .i_io_top_lcmd_valid (i_io_top_lcmd_valid_pad),
        .i_io_top_exec_valid (i_io_top_exec_valid_pad),
        .o_io_top_exec_ready (o_io_top_exec_ready_pad),
        .o_io_tdma_wr_data   (o_io_tdma_wr_data_pad  ),
        .o_io_tdma_wr_valid  (o_io_tdma_wr_valid_pad ),
        .i_io_tdma_wr_ready  (i_io_tdma_wr_ready_pad ),
        .i_io_debug_cmd      (i_io_debug_cmd_pad     ),
        .o_io_debug_data     (o_io_debug_data_pad    )
    );

    PDIDGZ IPAD_I_CLK (.PAD (i_clk), .C (i_clk_pad));
    PDIDGZ IPAD_I_RST_N (.PAD (i_rst_n), .C (i_rst_n_pad));
    PDIDGZ IPAD_I_DATA_0 (.PAD (i_io_tdma_rd_data[0]), .C (i_io_tdma_rd_data_pad[0]));
    PDIDGZ IPAD_I_DATA_1 (.PAD (i_io_tdma_rd_data[1]), .C (i_io_tdma_rd_data_pad[1]));
    PDIDGZ IPAD_I_DATA_2 (.PAD (i_io_tdma_rd_data[2]), .C (i_io_tdma_rd_data_pad[2]));
    PDIDGZ IPAD_I_DATA_3 (.PAD (i_io_tdma_rd_data[3]), .C (i_io_tdma_rd_data_pad[3]));
    PDIDGZ IPAD_I_DATA_4 (.PAD (i_io_tdma_rd_data[4]), .C (i_io_tdma_rd_data_pad[4]));
    PDIDGZ IPAD_I_DATA_5 (.PAD (i_io_tdma_rd_data[5]), .C (i_io_tdma_rd_data_pad[5]));
    PDIDGZ IPAD_I_DATA_6 (.PAD (i_io_tdma_rd_data[6]), .C (i_io_tdma_rd_data_pad[6]));
    PDIDGZ IPAD_I_DATA_7 (.PAD (i_io_tdma_rd_data[7]), .C (i_io_tdma_rd_data_pad[7]));
    PDIDGZ IPAD_I_DATA_8 (.PAD (i_io_tdma_rd_data[8]), .C (i_io_tdma_rd_data_pad[8]));
    PDIDGZ IPAD_I_DATA_9 (.PAD (i_io_tdma_rd_data[9]), .C (i_io_tdma_rd_data_pad[9]));
    PDIDGZ IPAD_I_DATA_10 (.PAD (i_io_tdma_rd_data[10]), .C (i_io_tdma_rd_data_pad[10]));
    PDIDGZ IPAD_I_DATA_11 (.PAD (i_io_tdma_rd_data[11]), .C (i_io_tdma_rd_data_pad[11]));
    PDIDGZ IPAD_I_DATA_12 (.PAD (i_io_tdma_rd_data[12]), .C (i_io_tdma_rd_data_pad[12]));
    PDIDGZ IPAD_I_DATA_13 (.PAD (i_io_tdma_rd_data[13]), .C (i_io_tdma_rd_data_pad[13]));
    PDIDGZ IPAD_I_DATA_14 (.PAD (i_io_tdma_rd_data[14]), .C (i_io_tdma_rd_data_pad[14]));
    PDIDGZ IPAD_I_DATA_15 (.PAD (i_io_tdma_rd_data[15]), .C (i_io_tdma_rd_data_pad[15]));
    PDIDGZ IPAD_I_DATA_16 (.PAD (i_io_tdma_rd_data[16]), .C (i_io_tdma_rd_data_pad[16]));
    PDIDGZ IPAD_I_DATA_17 (.PAD (i_io_tdma_rd_data[17]), .C (i_io_tdma_rd_data_pad[17]));
    PDIDGZ IPAD_I_DATA_18 (.PAD (i_io_tdma_rd_data[18]), .C (i_io_tdma_rd_data_pad[18]));
    PDIDGZ IPAD_I_DATA_19 (.PAD (i_io_tdma_rd_data[19]), .C (i_io_tdma_rd_data_pad[19]));
    PDIDGZ IPAD_I_DATA_20 (.PAD (i_io_tdma_rd_data[20]), .C (i_io_tdma_rd_data_pad[20]));
    PDIDGZ IPAD_I_DATA_21 (.PAD (i_io_tdma_rd_data[21]), .C (i_io_tdma_rd_data_pad[21]));
    PDIDGZ IPAD_I_DATA_22 (.PAD (i_io_tdma_rd_data[22]), .C (i_io_tdma_rd_data_pad[22]));
    PDIDGZ IPAD_I_DATA_23 (.PAD (i_io_tdma_rd_data[23]), .C (i_io_tdma_rd_data_pad[23]));
    PDIDGZ IPAD_I_DATA_24 (.PAD (i_io_tdma_rd_data[24]), .C (i_io_tdma_rd_data_pad[24]));
    PDIDGZ IPAD_I_DATA_25 (.PAD (i_io_tdma_rd_data[25]), .C (i_io_tdma_rd_data_pad[25]));
    PDIDGZ IPAD_I_DATA_26 (.PAD (i_io_tdma_rd_data[26]), .C (i_io_tdma_rd_data_pad[26]));
    PDIDGZ IPAD_I_DATA_27 (.PAD (i_io_tdma_rd_data[27]), .C (i_io_tdma_rd_data_pad[27]));
    PDIDGZ IPAD_I_DATA_28 (.PAD (i_io_tdma_rd_data[28]), .C (i_io_tdma_rd_data_pad[28]));
    PDIDGZ IPAD_I_DATA_29 (.PAD (i_io_tdma_rd_data[29]), .C (i_io_tdma_rd_data_pad[29]));
    PDIDGZ IPAD_I_DATA_30 (.PAD (i_io_tdma_rd_data[30]), .C (i_io_tdma_rd_data_pad[30]));
    PDIDGZ IPAD_I_DATA_31 (.PAD (i_io_tdma_rd_data[31]), .C (i_io_tdma_rd_data_pad[31]));
    PDIDGZ IPAD_I_DATA_32 (.PAD (i_io_tdma_rd_data[32]), .C (i_io_tdma_rd_data_pad[32]));
    PDIDGZ IPAD_I_DATA_33 (.PAD (i_io_tdma_rd_data[33]), .C (i_io_tdma_rd_data_pad[33]));
    PDIDGZ IPAD_I_DATA_34 (.PAD (i_io_tdma_rd_data[34]), .C (i_io_tdma_rd_data_pad[34]));
    PDIDGZ IPAD_I_DATA_35 (.PAD (i_io_tdma_rd_data[35]), .C (i_io_tdma_rd_data_pad[35]));
    PDIDGZ IPAD_I_DATA_36 (.PAD (i_io_tdma_rd_data[36]), .C (i_io_tdma_rd_data_pad[36]));
    PDIDGZ IPAD_I_DATA_37 (.PAD (i_io_tdma_rd_data[37]), .C (i_io_tdma_rd_data_pad[37]));
    PDIDGZ IPAD_I_DATA_38 (.PAD (i_io_tdma_rd_data[38]), .C (i_io_tdma_rd_data_pad[38]));
    PDIDGZ IPAD_I_DATA_39 (.PAD (i_io_tdma_rd_data[39]), .C (i_io_tdma_rd_data_pad[39]));
    PDIDGZ IPAD_I_DATA_40 (.PAD (i_io_tdma_rd_data[40]), .C (i_io_tdma_rd_data_pad[40]));
    PDIDGZ IPAD_I_DATA_41 (.PAD (i_io_tdma_rd_data[41]), .C (i_io_tdma_rd_data_pad[41]));
    PDIDGZ IPAD_I_DATA_42 (.PAD (i_io_tdma_rd_data[42]), .C (i_io_tdma_rd_data_pad[42]));
    PDIDGZ IPAD_I_DATA_43 (.PAD (i_io_tdma_rd_data[43]), .C (i_io_tdma_rd_data_pad[43]));
    PDIDGZ IPAD_I_DATA_44 (.PAD (i_io_tdma_rd_data[44]), .C (i_io_tdma_rd_data_pad[44]));
    PDIDGZ IPAD_I_DATA_45 (.PAD (i_io_tdma_rd_data[45]), .C (i_io_tdma_rd_data_pad[45]));
    PDIDGZ IPAD_I_DATA_46 (.PAD (i_io_tdma_rd_data[46]), .C (i_io_tdma_rd_data_pad[46]));
    PDIDGZ IPAD_I_DATA_47 (.PAD (i_io_tdma_rd_data[47]), .C (i_io_tdma_rd_data_pad[47]));
    PDIDGZ IPAD_I_DATA_48 (.PAD (i_io_tdma_rd_data[48]), .C (i_io_tdma_rd_data_pad[48]));
    PDIDGZ IPAD_I_DATA_49 (.PAD (i_io_tdma_rd_data[49]), .C (i_io_tdma_rd_data_pad[49]));
    PDIDGZ IPAD_I_DATA_50 (.PAD (i_io_tdma_rd_data[50]), .C (i_io_tdma_rd_data_pad[50]));
    PDIDGZ IPAD_I_DATA_51 (.PAD (i_io_tdma_rd_data[51]), .C (i_io_tdma_rd_data_pad[51]));
    PDIDGZ IPAD_I_DATA_52 (.PAD (i_io_tdma_rd_data[52]), .C (i_io_tdma_rd_data_pad[52]));
    PDIDGZ IPAD_I_DATA_53 (.PAD (i_io_tdma_rd_data[53]), .C (i_io_tdma_rd_data_pad[53]));
    PDIDGZ IPAD_I_DATA_54 (.PAD (i_io_tdma_rd_data[54]), .C (i_io_tdma_rd_data_pad[54]));
    PDIDGZ IPAD_I_DATA_55 (.PAD (i_io_tdma_rd_data[55]), .C (i_io_tdma_rd_data_pad[55]));
    PDIDGZ IPAD_I_DATA_56 (.PAD (i_io_tdma_rd_data[56]), .C (i_io_tdma_rd_data_pad[56]));
    PDIDGZ IPAD_I_DATA_57 (.PAD (i_io_tdma_rd_data[57]), .C (i_io_tdma_rd_data_pad[57]));
    PDIDGZ IPAD_I_DATA_58 (.PAD (i_io_tdma_rd_data[58]), .C (i_io_tdma_rd_data_pad[58]));
    PDIDGZ IPAD_I_DATA_59 (.PAD (i_io_tdma_rd_data[59]), .C (i_io_tdma_rd_data_pad[59]));
    PDIDGZ IPAD_I_DATA_60 (.PAD (i_io_tdma_rd_data[60]), .C (i_io_tdma_rd_data_pad[60]));
    PDIDGZ IPAD_I_DATA_61 (.PAD (i_io_tdma_rd_data[61]), .C (i_io_tdma_rd_data_pad[61]));
    PDIDGZ IPAD_I_DATA_62 (.PAD (i_io_tdma_rd_data[62]), .C (i_io_tdma_rd_data_pad[62]));
    PDIDGZ IPAD_I_DATA_63 (.PAD (i_io_tdma_rd_data[63]), .C (i_io_tdma_rd_data_pad[63]));
    PDIDGZ IPAD_I_RD_VALID (.PAD (i_io_tdma_rd_valid), .C (i_io_tdma_rd_valid_pad));
    PDIDGZ IPAD_I_NPU_CMD_0 (.PAD (i_io_top_npu_cmd[0]), .C (i_io_top_npu_cmd_pad[0]));
    PDIDGZ IPAD_I_NPU_CMD_1 (.PAD (i_io_top_npu_cmd[1]), .C (i_io_top_npu_cmd_pad[1]));
    PDIDGZ IPAD_I_NPU_CMD_2 (.PAD (i_io_top_npu_cmd[2]), .C (i_io_top_npu_cmd_pad[2]));
    PDIDGZ IPAD_I_LCMD_VALID (.PAD (i_io_top_lcmd_valid), .C (i_io_top_lcmd_valid_pad));
    PDIDGZ IPAD_I_EXEC_VALID (.PAD (i_io_top_exec_valid), .C (i_io_top_exec_valid_pad));
    PDIDGZ IPAD_I_WR_READY (.PAD (i_io_tdma_wr_ready), .C (i_io_tdma_wr_ready_pad));
    PDIDGZ IPAD_I_DEBUG_CMD_0 (.PAD (i_io_debug_cmd[0]), .C (i_io_debug_cmd_pad[0]));
    PDIDGZ IPAD_I_DEBUG_CMD_1 (.PAD (i_io_debug_cmd[1]), .C (i_io_debug_cmd_pad[1]));
    PDIDGZ IPAD_I_DEBUG_CMD_2 (.PAD (i_io_debug_cmd[2]), .C (i_io_debug_cmd_pad[2]));
    PDIDGZ IPAD_I_DEBUG_CMD_3 (.PAD (i_io_debug_cmd[3]), .C (i_io_debug_cmd_pad[3]));
    PDIDGZ IPAD_I_DEBUG_CMD_4 (.PAD (i_io_debug_cmd[4]), .C (i_io_debug_cmd_pad[4]));

    PDO24CDG OPAD_O_RD_READY (.I (o_io_tdma_rd_ready_pad), .PAD (o_io_tdma_rd_ready));
    PDO24CDG OPAD_O_EXEC_READY (.I (o_io_top_exec_ready_pad), .PAD (o_io_top_exec_ready));
    PDO24CDG OPAD_O_DATA_0 (.I (o_io_tdma_wr_data_pad[0]), .PAD (o_io_tdma_wr_data[0]));
    PDO24CDG OPAD_O_DATA_1 (.I (o_io_tdma_wr_data_pad[1]), .PAD (o_io_tdma_wr_data[1]));
    PDO24CDG OPAD_O_DATA_2 (.I (o_io_tdma_wr_data_pad[2]), .PAD (o_io_tdma_wr_data[2]));
    PDO24CDG OPAD_O_DATA_3 (.I (o_io_tdma_wr_data_pad[3]), .PAD (o_io_tdma_wr_data[3]));
    PDO24CDG OPAD_O_DATA_4 (.I (o_io_tdma_wr_data_pad[4]), .PAD (o_io_tdma_wr_data[4]));
    PDO24CDG OPAD_O_DATA_5 (.I (o_io_tdma_wr_data_pad[5]), .PAD (o_io_tdma_wr_data[5]));
    PDO24CDG OPAD_O_DATA_6 (.I (o_io_tdma_wr_data_pad[6]), .PAD (o_io_tdma_wr_data[6]));
    PDO24CDG OPAD_O_DATA_7 (.I (o_io_tdma_wr_data_pad[7]), .PAD (o_io_tdma_wr_data[7]));
    PDO24CDG OPAD_O_DATA_8 (.I (o_io_tdma_wr_data_pad[8]), .PAD (o_io_tdma_wr_data[8]));
    PDO24CDG OPAD_O_DATA_9 (.I (o_io_tdma_wr_data_pad[9]), .PAD (o_io_tdma_wr_data[9]));
    PDO24CDG OPAD_O_DATA_10 (.I (o_io_tdma_wr_data_pad[10]), .PAD (o_io_tdma_wr_data[10]));
    PDO24CDG OPAD_O_DATA_11 (.I (o_io_tdma_wr_data_pad[11]), .PAD (o_io_tdma_wr_data[11]));
    PDO24CDG OPAD_O_DATA_12 (.I (o_io_tdma_wr_data_pad[12]), .PAD (o_io_tdma_wr_data[12]));
    PDO24CDG OPAD_O_DATA_13 (.I (o_io_tdma_wr_data_pad[13]), .PAD (o_io_tdma_wr_data[13]));
    PDO24CDG OPAD_O_DATA_14 (.I (o_io_tdma_wr_data_pad[14]), .PAD (o_io_tdma_wr_data[14]));
    PDO24CDG OPAD_O_DATA_15 (.I (o_io_tdma_wr_data_pad[15]), .PAD (o_io_tdma_wr_data[15]));
    PDO24CDG OPAD_O_VALID (.I (o_io_tdma_wr_valid_pad), .PAD (o_io_tdma_wr_valid));
    PDO24CDG OPAD_O_DEBUG_DATA_0 (.I (o_io_debug_data_pad[0]), .PAD (o_io_debug_data[0]));
    PDO24CDG OPAD_O_DEBUG_DATA_1 (.I (o_io_debug_data_pad[1]), .PAD (o_io_debug_data[1]));
    PDO24CDG OPAD_O_DEBUG_DATA_2 (.I (o_io_debug_data_pad[2]), .PAD (o_io_debug_data[2]));
    PDO24CDG OPAD_O_DEBUG_DATA_3 (.I (o_io_debug_data_pad[3]), .PAD (o_io_debug_data[3]));
    PDO24CDG OPAD_O_DEBUG_DATA_4 (.I (o_io_debug_data_pad[4]), .PAD (o_io_debug_data[4]));
    PDO24CDG OPAD_O_DEBUG_DATA_5 (.I (o_io_debug_data_pad[5]), .PAD (o_io_debug_data[5]));
    PDO24CDG OPAD_O_DEBUG_DATA_6 (.I (o_io_debug_data_pad[6]), .PAD (o_io_debug_data[6]));
    PDO24CDG OPAD_O_DEBUG_DATA_7 (.I (o_io_debug_data_pad[7]), .PAD (o_io_debug_data[7]));
    PDO24CDG OPAD_O_DEBUG_DATA_8 (.I (o_io_debug_data_pad[8]), .PAD (o_io_debug_data[8]));
    PDO24CDG OPAD_O_DEBUG_DATA_9 (.I (o_io_debug_data_pad[9]), .PAD (o_io_debug_data[9]));
    PDO24CDG OPAD_O_DEBUG_DATA_10 (.I (o_io_debug_data_pad[10]), .PAD (o_io_debug_data[10]));
    PDO24CDG OPAD_O_DEBUG_DATA_11 (.I (o_io_debug_data_pad[11]), .PAD (o_io_debug_data[11]));
    PDO24CDG OPAD_O_DEBUG_DATA_12 (.I (o_io_debug_data_pad[12]), .PAD (o_io_debug_data[12]));
    PDO24CDG OPAD_O_DEBUG_DATA_13 (.I (o_io_debug_data_pad[13]), .PAD (o_io_debug_data[13]));
    PDO24CDG OPAD_O_DEBUG_DATA_14 (.I (o_io_debug_data_pad[14]), .PAD (o_io_debug_data[14]));
    PDO24CDG OPAD_O_DEBUG_DATA_15 (.I (o_io_debug_data_pad[15]), .PAD (o_io_debug_data[15]));

    // output pins x driving factor
    // 35 x 1.239 = 43.365
    // divided by the driving value of VDD and VSS
    // 43.365 / 1.6 = 27.1 -> 28 (IO VDD pad)
    // 43.365 / 1.5 = 28.9 -> 29 (IO VSS pad)
    // total number of IO pad pins = 57

    // 1 power pad control pin
    
    // core power pins = 256 - 77 - 57 - 35 - 1 = 86/2 => 43
    // 77+35=112

endmodule