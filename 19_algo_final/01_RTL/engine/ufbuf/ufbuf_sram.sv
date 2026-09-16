/**************************************************************************
* Filename: ofbuf_sram.sv
* Description: for reading/writing psum
* Parameters:
* Note: 2p sram to read/write different addr in same cycle (ref: mac_array)
* Review History: 2023 
**************************************************************************/

module ufbuf_sram 
    import Param_def::*;
    import Mem_def::*;
(
    input  logic i_clk,
    input  logic i_rst_n,

    input  logic [UFBUF_BANKS-1:0]                 sram_ena,
    input  logic [UFBUF_BANKS-1:0]                 sram_wea, 
    input  logic [UFBUF_BANKS-1:0][UFBUF_ADDRW-1:0] sram_addra,
    input  logic [UFBUF_BANKS-1:0][UFBUF_WIDTH-1:0] sram_dina,
    output logic [UFBUF_BANKS-1:0][UFBUF_WIDTH-1:0] sram_douta, 
    input  logic [UFBUF_BANKS-1:0]                 sram_enb,    
    input  logic [UFBUF_BANKS-1:0]                 sram_web, 
    input  logic [UFBUF_BANKS-1:0][UFBUF_ADDRW-1:0] sram_addrb, 
    input  logic [UFBUF_BANKS-1:0][UFBUF_WIDTH-1:0] sram_dinb,  
    output logic [UFBUF_BANKS-1:0][UFBUF_WIDTH-1:0] sram_doutb  
);
    logic [UFBUF_BANKS-1:0]                 ena_r;
    logic [UFBUF_BANKS-1:0]                 wea_r;
    logic [UFBUF_BANKS-1:0][UFBUF_ADDRW-1:0] addra_r;
    logic [UFBUF_BANKS-1:0][UFBUF_WIDTH-1:0] dina_r;
    logic [UFBUF_BANKS-1:0][UFBUF_WIDTH-1:0] douta_r, douta_w;

    logic [UFBUF_BANKS-1:0]                 enb_r;
    logic [UFBUF_BANKS-1:0]                 web_r;    
    logic [UFBUF_BANKS-1:0][UFBUF_ADDRW-1:0] addrb_r;
    logic [UFBUF_BANKS-1:0][UFBUF_WIDTH-1:0] dinb_r;
    logic [UFBUF_BANKS-1:0][UFBUF_WIDTH-1:0] doutb_r, doutb_w;

    assign sram_douta = douta_r;
    assign sram_doutb = doutb_r;

    // register input signals if PRESTO_BUF_GATE_IN is defined
    // generate
    `ifdef PRESTO_BUF_GATE_IN
    // if (PRESTO_BUF_GATE_IN) begin
        always_ff @ (posedge i_clk or negedge i_rst_n) begin
            if (~i_rst_n) begin
                ena_r   <= '0;
                wea_r   <= '0;
                addra_r <= '0;
                dina_r  <= '0;


                enb_r   <= '0;
                web_r   <= '0;
                addrb_r <= '0;
                dinb_r  <= '0;
            end
            else begin
                ena_r   <= sram_ena;
                wea_r   <= sram_wea;
                addra_r <= sram_addra;
                dina_r  <= sram_dina;

                enb_r   <= sram_enb;
                web_r   <= sram_web;
                addrb_r <= sram_addrb;
                dinb_r  <= sram_dinb;
            end
        end
    // end
    `else 
    // else begin
        always_comb begin
                ena_r   = sram_ena;
                wea_r   = sram_wea;
                addra_r = sram_addra;
                dina_r  = sram_dina;

                enb_r   = sram_enb;
                web_r   = sram_web;
                addrb_r = sram_addrb;
                dinb_r  = sram_dinb;
        end
    // end
    `endif
    // endgenerate

    // register output signals if PRESTO_BUF_GATE_OUT is defined
    // generate
    `ifdef PRESTO_BUF_GATE_OUT
    // if (PRESTO_BUF_GATE_OUT) begin
        always_ff @ (posedge i_clk or negedge i_rst_n) begin
            if (~i_rst_n) begin
                douta_r <= '0;
                doutb_r <= '0;
            end
            else begin
                douta_r <= douta_w;
                doutb_r <= doutb_w;
            end
        end
    // end else begin
    `else
        always_comb begin
            douta_r = douta_w;
            doutb_r = doutb_w;  
        end
    // end
    `endif
    // endgenerate

    generate 
        // SRAM module instances
        for (genvar gi = 0; gi < UFBUF_BANKS; gi = gi + 1) begin: bank
            presto_dpram # (
                .WIDTH (UFBUF_WIDTH ),
                .DEPTH (UFBUF_DEPTH )
            ) inst (
                .i_clk    (i_clk     ),
                .i_rst_n  (i_rst_n   ),
                // port a
                .ena      (ena_r[gi]  ),
                .wea      (wea_r[gi]  ),
                .addra    (addra_r[gi]),
                .dina     (dina_r[gi] ),
                .douta    (douta_w[gi]),
                // port b
                .enb      (enb_r[gi]  ),
                .web      (web_r[gi]),
                .addrb    (addrb_r[gi]),
                .dinb     (dinb_r[gi] ),
                .doutb    (doutb_w[gi])
            );
        end
    endgenerate
endmodule
