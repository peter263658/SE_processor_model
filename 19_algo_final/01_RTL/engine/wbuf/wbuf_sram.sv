/**************************************************
* Filename: wbuf_sram.sv
* Description: using dual port SRAM, for double bandwidth
* Parameters:
* Note: maybe should adjust to two port regfile
* TODO: should add debug signal 
* Review History: 2026
**************************************************/

module wbuf_sram 
    import Param_def::*;
    import Mem_def::*;
(
    input  logic i_clk,
    input  logic i_rst_n,

    // port a
    input  logic [WBUF_BANKS-1:0]                 sram_ena,
    input  logic [WBUF_BANKS-1:0]                 sram_wea, 
    input  logic [WBUF_BANKS-1:0][WBUF_ADDRW-1:0] sram_addra,
    input  logic [WBUF_BANKS-1:0][WBUF_WIDTH-1:0] sram_dina,
    output logic [WBUF_BANKS-1:0][WBUF_WIDTH-1:0] sram_douta, 
    // port b
    input  logic [WBUF_BANKS-1:0]                 sram_enb,    
    input  logic [WBUF_BANKS-1:0]                 sram_web, 
    input  logic [WBUF_BANKS-1:0][WBUF_ADDRW-1:0] sram_addrb, 
    input  logic [WBUF_BANKS-1:0][WBUF_WIDTH-1:0] sram_dinb,  
    output logic [WBUF_BANKS-1:0][WBUF_WIDTH-1:0] sram_doutb  
);
    logic [WBUF_BANKS-1:0]                 ena_r;
    logic [WBUF_BANKS-1:0]                 wea_r;
    logic [WBUF_BANKS-1:0][WBUF_ADDRW-1:0] addra_r;
    logic [WBUF_BANKS-1:0][WBUF_WIDTH-1:0] dina_r;
    logic [WBUF_BANKS-1:0][WBUF_WIDTH-1:0] douta_r, douta_w;

    logic [WBUF_BANKS-1:0]                 enb_r;
    logic [WBUF_BANKS-1:0]                 web_r;    
    logic [WBUF_BANKS-1:0][WBUF_ADDRW-1:0] addrb_r;
    logic [WBUF_BANKS-1:0][WBUF_WIDTH-1:0] dinb_r;
    logic [WBUF_BANKS-1:0][WBUF_WIDTH-1:0] doutb_r, doutb_w;

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
        for (genvar gi = 0; gi < WBUF_BANKS; gi = gi + 1) begin: bank
            presto_dpram # (
                .WIDTH (WBUF_WIDTH ),
                .DEPTH (WBUF_DEPTH )
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

    // always_comb begin
    //     debug_data = {en_r[0], we_r[0], addr_r[0][1:0], din_r[0][6:0], dout_w[0][6:0]};
    // end

endmodule
