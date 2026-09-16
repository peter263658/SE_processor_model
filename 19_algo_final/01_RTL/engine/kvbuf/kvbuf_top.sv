/**************************************************
* Filename: kvbuf_top.sv
* Description: top wrapper of the attention KV-cache buffer. Thin banked
*              true-dual-port pass-through over kvbuf_sram (presto_dpram).
*              Both ports are R/W so the future mem_controller can run 2R or
*              2W in a cycle. Intended usage: port A = in_proj store (write K/V
*              e4m3 into the ring), port B = QK read (all 32 frames) and
*              Score x V read (top-4). The ring address generation / frame
*              pointer lives in mem_controller (Step 2), not here.
* Parameters:  KVBUF_BANKS x KVBUF_WIDTH x KVBUF_DEPTH (Mem_def: 4 x 64 x 1024)
* Note:        Rewritten 2026-06-07 from a stale ofbuf copy (was instantiating
*              ofbuf_sram/ofbuf_ctrl with undefined OFBUF*_ params). The old
*              kvbuf_ctrl is deprecated/unused and superseded by mem_controller.
**************************************************/

module kvbuf_top
    import Param_def::*;
    import Mem_def::*;
(
    input  logic i_clk,
    input  logic i_rst_n,

    // Port A : R/W (in_proj store K/V)
    input  logic [KVBUF_BANKS-1:0]                   a_en,
    input  logic [KVBUF_BANKS-1:0]                   a_we,
    input  logic [KVBUF_BANKS-1:0][KVBUF_ADDRW-1:0]  a_addr,
    input  logic [KVBUF_BANKS-1:0][KVBUF_WIDTH-1:0]  a_din,
    output logic [KVBUF_BANKS-1:0][KVBUF_WIDTH-1:0]  a_dout,

    // Port B : R/W (QK all-32-frames / Score x V top-4 read)
    input  logic [KVBUF_BANKS-1:0]                   b_en,
    input  logic [KVBUF_BANKS-1:0]                   b_we,
    input  logic [KVBUF_BANKS-1:0][KVBUF_ADDRW-1:0]  b_addr,
    input  logic [KVBUF_BANKS-1:0][KVBUF_WIDTH-1:0]  b_din,
    output logic [KVBUF_BANKS-1:0][KVBUF_WIDTH-1:0]  b_dout
);

    kvbuf_sram u_kvbuf_sram (
        .i_clk      (i_clk  ),
        .i_rst_n    (i_rst_n),
        // port A
        .sram_ena   (a_en   ),
        .sram_wea   (a_we   ),
        .sram_addra (a_addr ),
        .sram_dina  (a_din  ),
        .sram_douta (a_dout ),
        // port B
        .sram_enb   (b_en   ),
        .sram_web   (b_we   ),
        .sram_addrb (b_addr ),
        .sram_dinb  (b_din  ),
        .sram_doutb (b_dout )
    );

endmodule
