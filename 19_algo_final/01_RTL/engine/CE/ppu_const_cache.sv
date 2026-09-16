/**************************************************************************
* Filename: ppu_const_cache.sv
* Description: PPU-local constant cache (register file, NOT an SRAM macro).
*   Ping-pong: 2 x (WORDS_PER_HALF x 64b). mem_controller fills one 64-bit
*   WBUF const word per cycle during a sub-op's PREFETCH phase (into the
*   INACTIVE half); dealer/ppu read packed codes during RUN/DRAIN (from the
*   ACTIVE half). The PE never reads the WBUF const region during MAC.
*   Read returns a 128-bit window {mem[waddr+1], mem[waddr]} so a 12-bit code
*   that straddles a 64-bit word boundary can be sliced by const_field_extract.
* Parameters: WORDS_PER_HALF (128 -> 1KiB/half, 2KiB total; per-sub-op peak 768B).
* Note: registered read (latency 1). Behavioral 2-word read; for synth split
*   even/odd banks or add a 2nd read port. See CE_STEP2_DESIGN.md §2.
* Review History: 2026-06-07 created (Step 2-1)
**************************************************************************/

module ppu_const_cache #(
    parameter int WORDS_PER_HALF = 128
)(
    input  logic i_clk,
    input  logic i_rst_n,

    // FILL : one 64-bit WBUF const word per cycle (mem_controller, PREFETCH phase)
    input  logic                              i_fill_en,
    input  logic                              i_fill_half,    // which ping-pong half to fill
    input  logic [$clog2(WORDS_PER_HALF)-1:0] i_fill_waddr,
    input  logic [63:0]                       i_fill_data,

    // READ : registered 128-bit window {word[waddr+1], word[waddr]} (RUN/DRAIN phase)
    input  logic                              i_rd_half,      // active half to read
    input  logic [$clog2(WORDS_PER_HALF)-1:0] i_rd_waddr,
    output logic [127:0]                      o_rd_window
);
    // 2 halves x WORDS_PER_HALF words of 64 bits
    logic [63:0] mem [0:1][0:WORDS_PER_HALF-1];

    // FILL (synchronous write into the selected half)
    always_ff @ (posedge i_clk) begin
        if (i_fill_en)
            mem[i_fill_half][i_fill_waddr] <= i_fill_data;
    end

    // READ (registered, latency 1). Upper word forced 0 at the top of the half
    // (a packed field never straddles past the last valid const word in practice).
    logic [63:0] rd_lo_r, rd_hi_r;
    always_ff @ (posedge i_clk or negedge i_rst_n) begin
        if (~i_rst_n) begin
            rd_lo_r <= '0;
            rd_hi_r <= '0;
        end
        else begin
            rd_lo_r <= mem[i_rd_half][i_rd_waddr];
            rd_hi_r <= (i_rd_waddr == (WORDS_PER_HALF-1)) ? 64'b0
                                                          : mem[i_rd_half][i_rd_waddr + 1'b1];
        end
    end

    assign o_rd_window = {rd_hi_r, rd_lo_r};

endmodule


/**************************************************************************
* Filename: const_field_extract (in ppu_const_cache.sv)
* Description: combinational slice of one packed const code out of the
*   128-bit read window. 12-bit fields = APoT 6a/6b weight/BN/PReLU codes;
*   16-bit fields = BN bias fp16 or activation-k (u8, low byte). Field start
*   bit within the window is i_bit_off = (byte_in_block*8 + code_idx*W) & 63.
*   House style: instantiated module (not a function) for waveform visibility.
**************************************************************************/
module const_field_extract (
    input  logic [127:0] i_window,    // {word[waddr+1], word[waddr]} from ppu_const_cache
    input  logic [6:0]   i_bit_off,   // start bit of the field within the window (0..127)
    input  logic         i_is16,      // 1 = 16-bit field (fp16 / u8-k); 0 = 12-bit (apot 6a/6b)
    output logic [15:0]  o_field
);
    logic [127:0] shifted;
    assign shifted  = i_window >> i_bit_off;
    assign o_field  = i_is16 ? shifted[15:0] : {4'b0, shifted[11:0]};
endmodule
