/**************************************************
* Filename: ufbuf_top.sv
* Description: combined ifbuf and ofbuf, using ping-pong buffer. 
* Parameters:
* Note: 
    1. Change buf state in IDLE state
* Review History: 2023 
**************************************************/

module ufbuf_top
    import Param_def::*;
    import Mem_def::*;
(
    input  logic i_clk,
    input  logic i_rst_n,

    // using for 
    input  logic                    swap_valid,             // swap buffer

    input  logic                    store_valid,                // store command valid
    input  logic                    store_done,                 // store command done
    output logic                    store_ready,                // store command ready
    output logic                    store_idle,                 // store command idle

    input  logic                    load_valid,                 // load command valid
    input  logic                    load_done,                  // load command done
    output logic                    load_ready,                 // load command ready
    output logic                    load_idle,                  // load command idle

    // load and store seperately, but using ping-pong buffer
    input  logic [UFBUF_BANKS_2-1:0]                  load_buf_ena,    // load SRAM enable
    input  logic [UFBUF_BANKS_2-1:0][UFBUF_ADDRW-1:0] load_buf_addra,  // load SRAM address
    output logic [UFBUF_BANKS_2-1:0][UFBUF_WIDTH-1:0] load_buf_douta,  // load SRAM output data
    input  logic [UFBUF_BANKS_2-1:0]                  load_buf_enb,    // load SRAM enable
    input  logic [UFBUF_BANKS_2-1:0][UFBUF_ADDRW-1:0] load_buf_addrb,  // load SRAM address
    output logic [UFBUF_BANKS_2-1:0][UFBUF_WIDTH-1:0] load_buf_doutb,  // load SRAM output data

    input  logic [UFBUF_BANKS_2-1:0]                  store_buf_ena,    // Store SRAM enable
    input  logic [UFBUF_BANKS_2-1:0][UFBUF_ADDRW-1:0] store_buf_addra,  // Store SRAM address
    input  logic [UFBUF_BANKS_2-1:0][UFBUF_WIDTH-1:0] store_buf_dina,   // Store SRAM output data
    input  logic [UFBUF_BANKS_2-1:0]                  store_buf_enb,    // Store SRAM enable
    input  logic [UFBUF_BANKS_2-1:0][UFBUF_ADDRW-1:0] store_buf_addrb,  // Store SRAM address
    input  logic [UFBUF_BANKS_2-1:0][UFBUF_WIDTH-1:0] store_buf_dinb,  // Store SRAM output data

);



    // conv_ofbuf states
    typedef enum logic [1:0] {
        S_IDLE,
        S_LOAD,
        S_STORE
    } State;
    State state_r, state_w;

    typedef enum logic [1:0] {
        S_BUF0,         // if/of
        S_BUF1          // of/if
    } BufferState;

    BufferState bufstate_r, bufstate_w;

    always_ff @ (posedge i_clk or negedge i_rst_n) begin
        if (~i_rst_n) begin
            state_r    <= S_IDLE;
            bufstate_r <= S_BUF0;
        end
        else begin
            state_r <= state_w;
            bufstate_r <= bufstate_w;
        end
    end


    always_comb begin 
        case(state_r)
            S_IDLE: begin
                if (store_valid) begin
                    state_w = S_STORE;
                end else if (load_valid) begin 
                    state_w = S_LOAD;
                end else begin 
                    state_w = S_IDLE;
                end
            end
            S_LOAD: begin
                if (load_done) begin
                    state_w = S_IDLE;
                end else begin 
                    state_w = S_LOAD;
                end
            end
            S_STORE: begin
                if (store_done) begin
                    state_w = S_IDLE;
                end else begin 
                    state_w = S_STORE;
                end
            end
            default: begin 
                state_w = S_IDLE;
            end
        endcase
    end

    // input buffer and output buffer
    logic [UFBUF_BANKS_2-1:0]                  load_buf_ena_w, load_buf_ena_r;    // load SRAM enable
    logic [UFBUF_BANKS_2-1:0][UFBUF_ADDRW-1:0] load_buf_addra_w, load_buf_addra_r;  // load SRAM address
    logic [UFBUF_BANKS_2-1:0][UFBUF_WIDTH-1:0] load_buf_douta_w, load_buf_douta_r;  // load SRAM output data
    logic [UFBUF_BANKS_2-1:0]                  load_buf_enb_w, load_buf_enb_r;    // load SRAM enable
    logic [UFBUF_BANKS_2-1:0][UFBUF_ADDRW-1:0] load_buf_addrb_w, load_buf_addrb_r;  // load SRAM address
    logic [UFBUF_BANKS_2-1:0][UFBUF_WIDTH-1:0] load_buf_doutb_w, load_buf_doutb_r;  // load SRAM output data

    logic [UFBUF_BANKS_2-1:0]                  store_buf_ena_w, store_buf_ena_r;    // Store SRAM enable
    logic [UFBUF_BANKS_2-1:0][UFBUF_ADDRW-1:0] store_buf_addra_w, store_buf_addra_r;  // Store SRAM address
    logic [UFBUF_BANKS_2-1:0][UFBUF_WIDTH-1:0] store_buf_dina_w, store_buf_dina_r;   // Store SRAM output data
    logic [UFBUF_BANKS_2-1:0]                  store_buf_enb_w, store_buf_enb_r;    // Store SRAM enable
    logic [UFBUF_BANKS_2-1:0][UFBUF_ADDRW-1:0] store_buf_addrb_w, store_buf_addrb_r;  // Store SRAM address
    logic [UFBUF_BANKS_2-1:0][UFBUF_WIDTH-1:0] store_buf_dinb_w, store_buf_dinb_r;  // Store SRAM output data
    
    always_ff @ (posedge i_clk or negedge i_rst_n)begin 
        if (~i_rst_n) begin
            load_buf_ena_r <= '0;
            load_buf_addra_r <= '0;
            load_buf_douta_r <= '0;
            load_buf_enb_r <= '0;
            load_buf_addrb_r <= '0;
            load_buf_doutb_r <= '0;
            store_buf_ena_r <= '0;
            store_buf_addra_r <= '0;
            store_buf_dina_r <= '0;
            store_buf_enb_r <= '0;
            store_buf_addrb_r <= '0;
            store_buf_dinb_r <= '0;
        end
        else begin
            load_buf_ena_r <= load_buf_ena_w;
            load_buf_addra_r <= load_buf_addra_w;
            load_buf_douta_r <= load_buf_douta_w;
            load_buf_enb_r <= load_buf_enb_w;
            load_buf_addrb_r <= load_buf_addrb_w;
            load_buf_doutb_r <= load_buf_doutb_w;

            store_buf_ena_r <= store_buf_ena_w;
            store_buf_addra_r <= store_buf_addra_w;
            store_buf_dina_r <= store_buf_dina_w;
            store_buf_enb_r <= store_buf_enb_w;
            store_buf_addrb_r <= store_buf_addrb_w;
            store_buf_dinb_r <= store_buf_dinb_w;
        end
    end

    // input buffer
    always_comb begin 
            load_buf_ena_w      = load_buf_ena;
            load_buf_addra_w    = load_buf_addra;
            load_buf_enb_w      = load_buf_enb;
            load_buf_addrb_w    = load_buf_addrb;

            store_buf_ena_w     = store_buf_ena;
            store_buf_addra_w   = store_buf_addra;
            store_buf_dina_w    = store_buf_dina;
            store_buf_enb_w     = store_buf_enb;
            store_buf_addrb_w   = store_buf_addrb;
            store_buf_dinb_w    = store_buf_dinb;
    end


    assign load_buf_douta = load_buf_douta_r;
    assign load_buf_doutb = load_buf_doutb_r;


    always_comb begin 
        case(bufstate_r)
            S_BUF0: begin
                if (swap_valid && state_r == S_IDLE) begin
                    bufstate_w = S_BUF1;
                end else begin 
                    bufstate_w = S_BUF0;
                end
            end
            S_BUF1: begin
                if (swap_valid && state_r == S_IDLE) begin
                    bufstate_w = S_BUF0;
                end else begin 
                    bufstate_w = S_BUF1;
                end
            end
            default: begin 
                bufstate_w = S_BUF0;
            end
        endcase
    end



    logic [UFBUF_BANKS-1:0]                  sram_buf_wea;
    logic [UFBUF_BANKS-1:0]                  sram_buf_ena;
    logic [UFBUF_BANKS-1:0][UFBUF_ADDRW-1:0] sram_buf_addra;
    logic [UFBUF_BANKS-1:0][UFBUF_WIDTH-1:0] sram_buf_dina;
    logic [UFBUF_BANKS-1:0][UFBUF_WIDTH-1:0] sram_buf_douta;

    logic [UFBUF_BANKS-1:0]                  sram_buf_web;
    logic [UFBUF_BANKS-1:0]                  sram_buf_enb;
    logic [UFBUF_BANKS-1:0][UFBUF_ADDRW-1:0] sram_buf_addrb;
    logic [UFBUF_BANKS-1:0][UFBUF_WIDTH-1:0] sram_buf_dinb;
    logic [UFBUF_BANKS-1:0][UFBUF_WIDTH-1:0] sram_buf_doutb;


    // assign signal for ping-pong buffer
    always_comb begin 
        for (int i = 0; i < UFBUF_BANKS_2; i++) begin 
            load_buf_douta_w[i] = 'bx;
            load_buf_doutb_w[i] = 'bx;
        end
        case(bufstate_r)
            S_BUF0: begin
                if (state_r == S_LOAD) begin 
                    for (int i = 0; i < UFBUF_BANKS_2; i++) begin 
                        // enable ofbuf
                        sram_buf_ena    [i+UFBUF_BANKS_2] = load_buf_ena_r[i];
                        sram_buf_addra  [i+UFBUF_BANKS_2] = load_buf_addra_r[i];
                        sram_buf_enb    [i+UFBUF_BANKS_2] = load_buf_enb_r[i];
                        sram_buf_addrb  [i+UFBUF_BANKS_2] = load_buf_addrb_r[i];
                        sram_buf_wea    [i+UFBUF_BANKS_2] = 'b1;
                        sram_buf_web    [i+UFBUF_BANKS_2] = 'b1;
                        sram_buf_dina   [i+UFBUF_BANKS_2] = 'bx;
                        sram_buf_dinb   [i+UFBUF_BANKS_2] = 'bx;
                        load_buf_douta_w  [i]             = sram_buf_douta[i+UFBUF_BANKS_2];
                        load_buf_doutb_w  [i]             = sram_buf_doutb[i+UFBUF_BANKS_2];

                        // disable ifbuf
                        sram_buf_ena   [i] = 'b0;
                        sram_buf_addra [i] = 'bx;
                        sram_buf_dina  [i] = 'bx;
                        sram_buf_enb   [i] = 'b0;
                        sram_buf_addrb [i] = 'bx;
                        sram_buf_dinb  [i] = 'bx;
                        sram_buf_wea   [i] = 'b1;
                        sram_buf_web   [i] = 'b1;
                    end
                end
                else if (state_r == S_STORE) begin 
                    for (int i = 0; i < UFBUF_BANKS_2; i++) begin 
                        // enable ifbuf
                        sram_buf_ena   [i] = store_buf_ena_r[i];
                        sram_buf_addra [i] = store_buf_addra_r[i];
                        sram_buf_dina  [i] = store_buf_dina_r[i];
                        sram_buf_enb   [i] = store_buf_enb_r[i];
                        sram_buf_addrb [i] = store_buf_addrb_r[i];
                        sram_buf_dinb  [i] = store_buf_dinb_r[i];
                        sram_buf_wea   [i] = ~store_buf_ena_r[i];
                        sram_buf_web   [i] = ~store_buf_enb_r[i];

                        // disable ofbuf
                        sram_buf_ena    [i+UFBUF_BANKS_2] = 'b0;
                        sram_buf_addra  [i+UFBUF_BANKS_2] = 'bx;
                        sram_buf_enb    [i+UFBUF_BANKS_2] = 'b0;
                        sram_buf_addrb  [i+UFBUF_BANKS_2] = 'bx;
                        sram_buf_wea    [i+UFBUF_BANKS_2] = 'b1;
                        sram_buf_web    [i+UFBUF_BANKS_2] = 'b1;
                        sram_buf_dina   [i+UFBUF_BANKS_2] = 'bx;
                        sram_buf_dinb   [i+UFBUF_BANKS_2] = 'bx;
                    end
                end     
                else begin 
                    for (int i = 0; i < UFBUF_BANKS; i++) begin 
                        sram_buf_ena    [i] = 'b0;
                        sram_buf_addra  [i] = 'bx;
                        sram_buf_dina   [i] = 'bx;
                        sram_buf_enb    [i] = 'b0;
                        sram_buf_addrb  [i] = 'bx;
                        sram_buf_dinb   [i] = 'bx;
                        sram_buf_wea    [i] = 'b1;
                        sram_buf_web    [i] = 'b1;
                    end                    
                end              
            end
            S_BUF1: begin
                if (state_r == S_LOAD) begin 
                    for (int i = 0; i < UFBUF_BANKS_2; i++) begin 
                        // enable ofbuf
                        sram_buf_ena    [i] = load_buf_ena_r[i];
                        sram_buf_addra  [i] = load_buf_addra_r[i];
                        sram_buf_enb    [i] = load_buf_enb_r[i];
                        sram_buf_addrb  [i] = load_buf_addrb_r[i];
                        sram_buf_wea    [i] = 'b1;
                        sram_buf_web    [i] = 'b1;
                        sram_buf_dina   [i] = 'bx;
                        sram_buf_dinb   [i] = 'bx;
                        load_buf_douta_w  [i] = sram_buf_douta[i];
                        load_buf_doutb_w  [i] = sram_buf_doutb[i];

                        // diable ifbuf
                        sram_buf_ena    [i+UFBUF_BANKS_2] = 'b0;
                        sram_buf_addra  [i+UFBUF_BANKS_2] = 'bx;
                        sram_buf_enb    [i+UFBUF_BANKS_2] = 'b0;
                        sram_buf_addrb  [i+UFBUF_BANKS_2] = 'bx;
                        sram_buf_wea    [i+UFBUF_BANKS_2] = 'b1;
                        sram_buf_web    [i+UFBUF_BANKS_2] = 'b1;
                        sram_buf_dina   [i+UFBUF_BANKS_2] = 'bx;
                        sram_buf_dinb   [i+UFBUF_BANKS_2] = 'bx;
                    end
                end
                else if (state_r == S_STORE) begin 
                    for (int i = 0; i < UFBUF_BANKS_2; i++) begin 
                        // enable ifbuf
                        sram_buf_ena   [i+UFBUF_BANKS_2] = store_buf_ena_r[i];
                        sram_buf_addra [i+UFBUF_BANKS_2] = store_buf_addra_r[i];
                        sram_buf_dina  [i+UFBUF_BANKS_2] = store_buf_dina_r[i];
                        sram_buf_enb   [i+UFBUF_BANKS_2] = store_buf_enb_r[i];
                        sram_buf_addrb [i+UFBUF_BANKS_2] = store_buf_addrb_r[i];
                        sram_buf_dinb  [i+UFBUF_BANKS_2] = store_buf_dinb_r[i];
                        sram_buf_wea   [i+UFBUF_BANKS_2] = ~store_buf_ena_r[i];
                        sram_buf_web   [i+UFBUF_BANKS_2] = ~store_buf_enb_r[i];

                        // disable ofbuf
                        sram_buf_ena    [i] = 'b0;
                        sram_buf_addra  [i] = 'bx;
                        sram_buf_enb    [i] = 'b0;
                        sram_buf_addrb  [i] = 'bx;
                        sram_buf_wea    [i] = 'b1;
                        sram_buf_web    [i] = 'b1;
                        sram_buf_dina   [i] = 'bx;
                        sram_buf_dinb   [i] = 'bx;
                    end
                end
                else begin 
                    for (int i = 0; i < UFBUF_BANKS; i++) begin 
                        sram_buf_ena    [i] = 'b0;
                        sram_buf_addra  [i] = 'bx;
                        sram_buf_dina   [i] = 'bx;
                        sram_buf_enb    [i] = 'b0;
                        sram_buf_addrb  [i] = 'bx;
                        sram_buf_dinb   [i] = 'bx;
                        sram_buf_wea    [i] = 'b1;
                        sram_buf_web    [i] = 'b1;
                    end
                end
            end
            default: begin 
                for (int i = 0; i < UFBUF_BANKS; i++) begin 
                    sram_buf_ena    [i] = 'b0;
                    sram_buf_addra  [i] = 'bx;
                    sram_buf_dina   [i] = 'bx;
                    sram_buf_enb    [i] = 'b0;
                    sram_buf_addrb  [i] = 'bx;
                    sram_buf_dinb   [i] = 'bx;
                    sram_buf_wea    [i] = 'b1;
                    sram_buf_web    [i] = 'b1;
                end
                for (int j = 0; j < UFBUF_BANKS_2; j++) begin 
                    load_buf_douta_w[j] = 'bx;
                    load_buf_doutb_w[j] = 'bx;
                end
            end
        endcase
    end



    assign store_ready = (state_r == S_IDLE);
    assign store_idle  = (state_r != S_STORE);
    assign load_ready  = (state_r == S_IDLE);
    assign load_idle   = (state_r != S_LOAD);


    ufbuf_sram # (
        .SRAM_BANKS (UFBUF_BANKS),
        .SRAM_WIDTH (UFBUF_WIDTH),
        .SRAM_DEPTH (UFBUF_DEPTH)
    ) u_ufbuf_sram_inst (
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

endmodule