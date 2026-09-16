/**************************************************************************
* Filename: skipbuf_sram.sv
* Description: SKIP-connection buffer SRAM (decoder skip branch). Banked
*              TRUE dual-port (presto_dpram): port A and port B are each R/W,
*              so the buffer supports 2R or 2W in one cycle (e.g. store on A
*              while gathering on B). presto_2pram is only 1W1R and has no
*              64-wide macro — presto_dpram has rf_dp_128x64 for this size.
* Parameters:  SKIPBUF_BANKS x SKIPBUF_WIDTH x SKIPBUF_DEPTH (Mem_def: 4 x 64 x 128)
* Note:        PRESTO_BUF_GATE_IN/OUT (`ifdef`) default = pass-through, read latency 1.
* Review History: 2026-06-07 rewritten from stale wbuf copy (presto_2pram->presto_dpram)
**************************************************************************/

module skipbuf_sram
    import Param_def::*;
    import Mem_def::*;
(
    input  logic i_clk,
    input  logic i_rst_n,
    // port A : read / write
    input  logic [SKIPBUF_BANKS-1:0]                    sram_ena,
    input  logic [SKIPBUF_BANKS-1:0]                    sram_wea,
    input  logic [SKIPBUF_BANKS-1:0][SKIPBUF_ADDRW-1:0] sram_addra,
    input  logic [SKIPBUF_BANKS-1:0][SKIPBUF_WIDTH-1:0] sram_dina,
    output logic [SKIPBUF_BANKS-1:0][SKIPBUF_WIDTH-1:0] sram_douta,
    // port B : read / write
    input  logic [SKIPBUF_BANKS-1:0]                    sram_enb,
    input  logic [SKIPBUF_BANKS-1:0]                    sram_web,
    input  logic [SKIPBUF_BANKS-1:0][SKIPBUF_ADDRW-1:0] sram_addrb,
    input  logic [SKIPBUF_BANKS-1:0][SKIPBUF_WIDTH-1:0] sram_dinb,
    output logic [SKIPBUF_BANKS-1:0][SKIPBUF_WIDTH-1:0] sram_doutb
);
    logic [SKIPBUF_BANKS-1:0]                    ena_r, wea_r, enb_r, web_r;
    logic [SKIPBUF_BANKS-1:0][SKIPBUF_ADDRW-1:0] addra_r, addrb_r;
    logic [SKIPBUF_BANKS-1:0][SKIPBUF_WIDTH-1:0] dina_r, dinb_r;
    logic [SKIPBUF_BANKS-1:0][SKIPBUF_WIDTH-1:0] douta_r, douta_w, doutb_r, doutb_w;

    assign sram_douta = douta_r;
    assign sram_doutb = doutb_r;

    // register input signals if PRESTO_BUF_GATE_IN is defined (default: pass-through)
    `ifdef PRESTO_BUF_GATE_IN
        always_ff @ (posedge i_clk or negedge i_rst_n) begin
            if (~i_rst_n) begin
                ena_r <= '0; wea_r <= '0; addra_r <= '0; dina_r <= '0;
                enb_r <= '0; web_r <= '0; addrb_r <= '0; dinb_r <= '0;
            end
            else begin
                ena_r <= sram_ena; wea_r <= sram_wea; addra_r <= sram_addra; dina_r <= sram_dina;
                enb_r <= sram_enb; web_r <= sram_web; addrb_r <= sram_addrb; dinb_r <= sram_dinb;
            end
        end
    `else
        always_comb begin
            ena_r = sram_ena; wea_r = sram_wea; addra_r = sram_addra; dina_r = sram_dina;
            enb_r = sram_enb; web_r = sram_web; addrb_r = sram_addrb; dinb_r = sram_dinb;
        end
    `endif

    // register output signals if PRESTO_BUF_GATE_OUT is defined (default: pass-through)
    `ifdef PRESTO_BUF_GATE_OUT
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
    `else
        always_comb begin
            douta_r = douta_w;
            doutb_r = doutb_w;
        end
    `endif

    // SRAM bank instances (true dual-port)
    generate
        for (genvar gi = 0; gi < SKIPBUF_BANKS; gi = gi + 1) begin: bank
            presto_dpram # (
                .WIDTH (SKIPBUF_WIDTH),
                .DEPTH (SKIPBUF_DEPTH)
            ) inst (
                .i_clk   (i_clk      ),
                .i_rst_n (i_rst_n    ),
                .ena     (ena_r  [gi]),
                .wea     (wea_r  [gi]),
                .addra   (addra_r[gi]),
                .dina    (dina_r [gi]),
                .douta   (douta_w[gi]),
                .enb     (enb_r  [gi]),
                .web     (web_r  [gi]),
                .addrb   (addrb_r[gi]),
                .dinb    (dinb_r [gi]),
                .doutb   (doutb_w[gi])
            );
        end
    endgenerate
endmodule
