/*
    ce_operand_gather.sv — select an NBYTES operand window from a banked SRAM row read.

    ce_top issues a BYTE address per stream (o_wr_addr / o_xr_addr ...); the buffer read
    returns a contiguous WINDOW of consecutive 64-bit words starting at (addr & ~7). This
    module shifts out the (addr & 7) byte offset and exposes the NBYTES the stream needs:
      - W4A8 weights: NBYTES = 16  -> 32 INT4 nibbles (one ic-chunk, low-nibble first)
      - e4m3 acts:    NBYTES = 32  -> 32 e4m3 bytes
    Combinational (1-cycle read latency lives in the SRAM, matching ce_top OACC_DLY=2).

    The caller must supply a window of at least (NBYTES + 8) bytes so any byte offset fits.
    BYTE-ALIGNMENT / ROW-SPANNING vs the 128-byte WBUF row is the integration detail to
    pin in the golden sim (task 4/9): if a 16-byte chunk straddles a 128-byte row boundary
    the caller must read both rows into the window. Here we just shift+slice.
*/
`timescale 1ns/10ps
module ce_operand_gather #(
    parameter int NBYTES = 16                       // bytes the stream consumes
)(
    input  logic [(NBYTES+8)*8-1:0] i_window,       // consecutive words from byte (addr & ~7)
    input  logic [2:0]              i_byte_off,      // addr & 7
    output logic [NBYTES*8-1:0]     o_bytes
);
    logic [(NBYTES+8)*8-1:0] shifted;
    assign shifted = i_window >> {i_byte_off, 3'b000};   // drop (addr&7)*8 bits
    assign o_bytes = shifted[NBYTES*8-1:0];
endmodule
