/***************************************************************************
* Filename: skipbuf_top.sv
* Description: top wrapper of the decoder SKIP-connection buffer. Thin banked
*              true-dual-port pass-through over skipbuf_sram (presto_dpram).
*              Both ports are R/W so the future mem_controller can run 2R or
*              2W in a cycle. Intended usage: port A = decoder-side store
*              (write skip features), port B = gather read (static top-50%
*              channels, groups_skip=2). The gather address remap / sequencing
*              lives in mem_controller (Step 2), not here.
* Parameters:  SKIPBUF_BANKS x SKIPBUF_WIDTH x SKIPBUF_DEPTH (Mem_def: 4 x 64 x 128)
* Note:        Rewritten 2026-06-07 from a stale wbuf copy; the old DMA-load
*              skipbuf_ctrl is deprecated/unused and superseded by mem_controller.
***************************************************************************/

module skipbuf_top
    import Param_def::*;
    import Mem_def::*;
(
    input  logic i_clk,
    input  logic i_rst_n,

    // Port A : R/W (decoder-side store / write)
    input  logic [SKIPBUF_BANKS-1:0]                    a_en,
    input  logic [SKIPBUF_BANKS-1:0]                    a_we,
    input  logic [SKIPBUF_BANKS-1:0][SKIPBUF_ADDRW-1:0] a_addr,
    input  logic [SKIPBUF_BANKS-1:0][SKIPBUF_WIDTH-1:0] a_din,
    output logic [SKIPBUF_BANKS-1:0][SKIPBUF_WIDTH-1:0] a_dout,

    // Port B : R/W (gather read)
    input  logic [SKIPBUF_BANKS-1:0]                    b_en,
    input  logic [SKIPBUF_BANKS-1:0]                    b_we,
    input  logic [SKIPBUF_BANKS-1:0][SKIPBUF_ADDRW-1:0] b_addr,
    input  logic [SKIPBUF_BANKS-1:0][SKIPBUF_WIDTH-1:0] b_din,
    output logic [SKIPBUF_BANKS-1:0][SKIPBUF_WIDTH-1:0] b_dout
);

    skipbuf_sram u_skipbuf_sram (
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
