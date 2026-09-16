/****************************************************************************
* Filename: wbuf_ctrl.sv
* Description: use for B matrix for A @ B or Weight (FWP)
* Parameters:
* Note: 
    1. compute the address of SRAM and determine the input of SRAM from DMA
    2. compute the address according to the layer
    3. CHIP input bandwidth = 128bit
    
* Review History: 2023 
* TODO: 
****************************************************************************/

module wbuf_ctrl 
    import Param_def::*;
    import Mem_def::*;
(
    input  logic i_clk,
    input  logic i_rst_n,

    // dedicate the layer 
    input  Exec_mode                    i_exec_mode,
    input  AE_mode                      i_ae_mode,
    input  CE_mode                      i_ce_mode,
    
    // for input loading to SRAM (sequential)
    input  logic                        i_top_valid,      // input data valid
    output logic                        o_top_ready,      // output receive data ready
    input  logic [CHIP_I_WIDTH-1:0]     i_top_data,       
    
    // WBUF SRAM interface (for output to MAC, parallel, only load no write)
    input  logic                        i_load_valid,  // load command valid
    output logic                        o_load_ready,  // load command ready
    output logic                        o_load_valid,

    // for MAC control
    input  logic [WBUF_BANKS-1:0]                   i_core_ena,    // SRAM port a enable
    input  logic [WBUF_BANKS-1:0]                   i_core_enb,    // SRAM port b enable
    input  logic [WBUF_BANKS-1:0][WBUF_ADDRW-1:0]   i_core_addra,  // SRAM port a address
    input  logic [WBUF_BANKS-1:0][WBUF_ADDRW-1:0]   i_core_addrb,  // SRAM port b address

    // for SRAM interface
    output  logic [WBUF_BANKS-1:0]                 o_sram_buf_wea,
    output  logic [WBUF_BANKS-1:0]                 o_sram_buf_ena,
    output  logic [WBUF_BANKS-1:0][WBUF_ADDRW-1:0] o_sram_buf_addra,
    output  logic [WBUF_BANKS-1:0][WBUF_WIDTH-1:0] o_sram_buf_dina,
    output  logic [WBUF_BANKS-1:0]                 o_sram_buf_web,
    output  logic [WBUF_BANKS-1:0]                 o_sram_buf_enb,
    output  logic [WBUF_BANKS-1:0][WBUF_ADDRW-1:0] o_sram_buf_addrb,
    output  logic [WBUF_BANKS-1:0][WBUF_WIDTH-1:0] o_sram_buf_dinb
);

    // chip input and output
    logic                        i_top_valid_r, i_top_valid_w;   // input data valid
    logic                        o_top_ready_r, o_top_ready_w;   // output receive data ready
    logic [CHIP_I_WIDTH-1:0]     i_top_data_r, i_top_data_w;   

    // control signal
    logic   i_load_valid_w, i_load_valid_r;
    logic   o_load_ready_w, o_load_ready_r;
    logic   o_load_valid_w, o_load_valid_r;

    // // sram input and output 
    // logic [WBUF_BANKS-1:0]                 o_sram_buf_wea_r, o_sram_buf_wea_w;
    // logic [WBUF_BANKS-1:0]                 o_sram_buf_ena_r, o_sram_buf_ena_w;
    // logic [WBUF_BANKS-1:0][WBUF_ADDRW-1:0] o_sram_buf_addra_r, o_sram_buf_addra_w;
    // logic [WBUF_BANKS-1:0][WBUF_DEPTH-1:0] o_sram_buf_dina_r, o_sram_buf_dina_w;
    // logic [WBUF_BANKS-1:0]                 o_sram_buf_web_r, o_sram_buf_web_w;
    // logic [WBUF_BANKS-1:0]                 o_sram_buf_enb_r, o_sram_buf_enb_w;
    // logic [WBUF_BANKS-1:0][WBUF_ADDRW-1:0] o_sram_buf_addrb_r, o_sram_buf_addrb_w;
    // logic [WBUF_BANKS-1:0][WBUF_DEPTH-1:0] o_sram_buf_dinb_r, o_sram_buf_dinb_w;

    
    logic [WBUF_BANKS-1:0]                 i_core_ena_r, i_core_ena_w;
    logic [WBUF_BANKS-1:0]                 i_core_enb_r, i_core_enb_w;
    logic [WBUF_BANKS-1:0][WBUF_ADDRW-1:0] i_core_addra_r, i_core_addra_w;
    logic [WBUF_BANKS-1:0][WBUF_ADDRW-1:0] i_core_addrb_r, i_core_addrb_w;



    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (~i_rst_n) begin 
            i_top_valid_r <= '0;
            i_top_data_r  <= '0;
        end else begin 
            i_top_valid_r <= i_top_valid_w;
            i_top_data_r  <= i_top_data_w;
        end
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (~i_rst_n) begin 
            i_load_valid_r <= '0;
            o_load_valid_r <= '0;
            o_load_ready_r <= '0;
        end
        else begin 
            i_load_valid_r <= i_load_valid_w;
            o_load_valid_r <= o_load_valid_w;
            o_load_ready_r <= o_load_ready_w;
        end
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (~i_rst_n) begin 
            i_core_ena_r <= '0;
            i_core_enb_r <= '0;
            i_core_addra_r <= '0;
            i_core_addrb_r <= '0;
        end
        else begin 
            i_core_ena_r <= i_core_ena_w;
            i_core_enb_r <= i_core_enb_w;
            i_core_addra_r <= i_core_addra_w;
            i_core_addrb_r <= i_core_addrb_w;
        end
    end

    assign i_top_valid_w = i_top_valid;
    assign i_top_data_w  = i_top_data;

    assign i_load_valid_w = i_load_valid;


    assign i_core_ena_w = i_core_ena;
    assign i_core_enb_w = i_core_enb;
    assign i_core_addra_w = i_core_addra;
    assign i_core_addrb_w = i_core_addrb;

    typedef enum logic [1:0] {
        S_IDLE,
        S_LOAD,                 // used for weights loading only (AE mode)
        S_STORE,                // used for weights storing only (Init mode and AE mode)
        S_LOAD_STORE            // used for weights loading and MAC reading at the same time (CE mode)
    } State;
        
    State state_w, state_r;
    
    // record the weights is empty or not
    logic weight_load_done;
    // recored the weighhs is full or not
    logic weight_store_done;

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (~i_rst_n) begin 
            state_r <= S_STORE;
        end else begin 
            state_r <= state_w;
        end   
    end

    always_comb begin
        state_w = state_r;
        case (state_r)
            S_IDLE: begin
                if (i_top_valid) begin
                    state_w = S_LOAD;
                end
            end
            S_LOAD: begin
                if (weight_load_done)   state_w = S_IDLE;
                else                    state_w = S_LOAD;
            end
            S_STORE: begin
                if (weight_store_done)  state_w = S_IDLE;
                else                    state_w = S_STORE;
            end
            default: begin
                state_w = S_IDLE;
            end
        endcase
    end

    

    // control the store process
    logic [WBUF_BANKS_ADDRW-1:0] bank_num_wr_w, bank_num_wr_r;
    
    always_ff @ (posedge i_clk or negedge i_rst_n) begin
        if (~i_rst_n) begin 
            bank_num_wr_r <= '0;
        end else begin 
            bank_num_wr_r <= bank_num_wr_w;
        end
    end

    always_comb begin 
        weight_store_done = '0;
        bank_num_wr_w = bank_num_wr_r;
        case (state_r) 
            S_STORE: begin
                
            end
            default: begin 
                weight_store_done = 0;
                bank_num_wr_w = bank_num_wr_r;
            end
        endcase
    end

    


    // always_comb begin
    //     debug_data      = '0;
    //     debug_data[1:0] = state_r;
    //     debug_data[3:2] = bank_num_wr_r[1:0];
    //     debug_data[7:4] = sram_temp_r[3:0];
    //     debug_data[9:8] = lstate_r[1:0];
    //     debug_data[10 ] = load_enab;
    //     debug_data[11 ] = load_valid;
    //     debug_data[12 ] = load_ready;
    //     debug_data[13 ] = load_idle;
    //     debug_data[14 ] = load_done;
    //     debug_data[15 ] = i_top_valid;
    // end

endmodule