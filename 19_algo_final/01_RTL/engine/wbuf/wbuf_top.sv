/***************************************************************************
* Filename: wbuf_top.sv
* Description: top modules of all w_srams, control the DMA input to SRAMs
* Parameters:
* Note: core data out has output buffer
* Review History: 2026
* TODO: 
    1. add some debug signal
// 
***************************************************************************/

module wbuf_top
    import Param_def::*;
    import Mem_def::*;
(
    input  logic i_clk,
    input  logic i_rst_n,

    // dedicate the layer and mode
    input  Exec_mode                    i_exec_mode,        // from CPU_controller
    input  AE_mode                      i_ae_mode,
    input  CE_mode                      i_ce_mode,
    input  CE_LAYER_mode                i_ce_layer,
    
    // for input loading to SRAM (sequential)
    input  logic                        i_top_valid,      // input data valid
    output logic                        o_top_ready,      // output receive data ready
    input  logic [CHIP_I_WIDTH-1:0]     i_top_data,       
    
    // WBUF SRAM interface (for output to MAC, parallel, only load no write)
    input  logic                        i_load_valid,  // load command valid
    output logic                        o_load_ready,  // load command ready
    output logic                        o_load_valid,   // load command idle

    input  logic [WBUF_BANKS-1:0]                   i_core_ena,    // SRAM port a enable
    input  logic [WBUF_BANKS-1:0]                   i_core_enb,    // SRAM port b enable
    input  logic [WBUF_BANKS-1:0][WBUF_ADDRW-1:0]   i_core_addra,  // SRAM port a address
    input  logic [WBUF_BANKS-1:0][WBUF_ADDRW-1:0]   i_core_addrb,  // SRAM port b address
    output logic [WBUF_BANKS-1:0][WBUF_WIDTH-1:0]   o_core_douta,  // SRAM port a output data
    output logic [WBUF_BANKS-1:0][WBUF_WIDTH-1:0]   o_core_doutb  // SRAM port b output data

    // output logic [DEBUG_DATA_WIDTH-1:0] debug_data_top,
    // output logic [DEBUG_DATA_WIDTH-1:0] debug_data_ctrl,
    // output logic [DEBUG_DATA_WIDTH-1:0] debug_data_sram

);

    logic [WBUF_BANKS-1:0]                  sram_buf_wea;
    logic [WBUF_BANKS-1:0]                  sram_buf_ena;
    logic [WBUF_BANKS-1:0][WBUF_ADDRW-1:0] sram_buf_addra;
    logic [WBUF_BANKS-1:0][WBUF_WIDTH-1:0] sram_buf_dina;
    logic [WBUF_BANKS-1:0][WBUF_WIDTH-1:0] sram_buf_douta;

    logic [WBUF_BANKS-1:0]                  sram_buf_web;
    logic [WBUF_BANKS-1:0]                  sram_buf_enb;
    logic [WBUF_BANKS-1:0][WBUF_ADDRW-1:0]  sram_buf_addrb;
    logic [WBUF_BANKS-1:0][WBUF_WIDTH-1:0]  sram_buf_dinb;
    logic [WBUF_BANKS-1:0][WBUF_WIDTH-1:0]  sram_buf_doutb;

    wbuf_sram u_wbuf_sram_inst (
        .i_clk      (i_clk      ),
        .i_rst_n    (i_rst_n    ),
        // port a
        .sram_ena   (sram_buf_ena   ),
        .sram_wea   (sram_buf_wea   ),
        .sram_addra (sram_buf_addra ),
        .sram_dina  (sram_buf_dina  ),
        .sram_douta (sram_buf_douta ),
        // port b
        .sram_enb   (sram_buf_enb   ),
        .sram_web   (sram_buf_web   ),
        .sram_addrb (sram_buf_addrb ),
        .sram_dinb  (sram_buf_dinb  ),
        .sram_doutb (sram_buf_doutb )
    );


    wbuf_ctrl u_wbuf_ctrl_inst (
    .i_clk              (i_clk  ),
    .i_rst_n            (i_rst_n),
    .i_exec_mode        (i_exec_mode    ),
    .i_ae_mode          (i_ae_mode      ),
    .i_ce_mode          (i_ce_mode      ),
    .i_top_valid        (i_top_valid    ),      // input data valid
    .o_top_ready        (o_top_ready    ),      // output receive data ready
    .i_top_data         (i_top_data     ),      
    .i_load_valid       (i_load_valid   ),      // load command valid
    .o_load_ready       (o_load_ready   ),      // load command ready
    .o_load_valid       (o_load_valid   ),      // load command idle
    .i_core_ena         (i_core_ena     ),      // SRAM port a enable
    .i_core_enb         (i_core_enb     ),      // SRAM port b enable
    .i_core_addra       (i_core_addra   ),      // SRAM port a address
    .i_core_addrb       (i_core_addrb   ),      // SRAM port b address
    .o_sram_buf_wea     (sram_buf_wea   ),
    .o_sram_buf_ena     (sram_buf_ena   ),
    .o_sram_buf_addra   (sram_buf_addra ),
    .o_sram_buf_dina    (sram_buf_dina  ),
    .o_sram_buf_web     (sram_buf_web   ),
    .o_sram_buf_enb     (sram_buf_enb   ),
    .o_sram_buf_addrb   (sram_buf_addrb ),
    .o_sram_buf_dinb    (sram_buf_dinb  )
    );

    logic [WBUF_BANKS-1:0][WBUF_WIDTH-1:0]   core_douta_r;  // SRAM port a output data
    logic [WBUF_BANKS-1:0][WBUF_WIDTH-1:0]   core_doutb_r;

    always_ff @ (posedge i_clk or negedge i_rst_n) begin 
        if (~i_rst_n) begin
            core_douta_r <= '0;
            core_doutb_r <= '0;
        end else begin
            core_douta_r <= sram_buf_douta;
            core_doutb_r <= sram_buf_doutb;
        end
    end
    // always_comb begin
    //     debug_data_top        = '0;
    //     debug_data_top[  1:0] = state_r;
    //     debug_data_top[  3:2] = ictrl_en[1:0];
    //     debug_data_top[  5:4] = ictrl_we[1:0];
    //     debug_data_top[  7:6] = buf0_en[1:0];
    //     debug_data_top[  9:8] = buf0_we[1:0];
    //     debug_data_top[13:10] = i_top_data[3:0];
    //     debug_data_top[   14] = i_top_valid;
    //     debug_data_top[   15] = o_top_ready;
    // end


endmodule