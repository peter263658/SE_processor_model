/********************************************************************
* Filename: presto_spram.sv
* Authors:
*     Wen-Cong Huang
* Description:
*     Single-port SRAM modules for Presto
* Parameters:
*     - WIDTH: SRAM data width
*     - DEPTH: SRAM entry depth
*     - ADDRW: SRAM address width
* Note:
*     - Replace with real SRAMs when synthesizing the design.
*     - Use `if ((WIDTH == X) && (DEPTH == Y))` for SRAM selection,
*       if no matching WIDTH and DEPTH, call $error() function.
* Review History:
*     2021.04.10    Wen-Cong Huang
*********************************************************************/

module presto_spram # (
    parameter WIDTH = 128,
    parameter DEPTH = 256,
    parameter ADDRW = $clog2(DEPTH)
)(
    input  logic             i_clk,
    input  logic             i_rst_n,
    input  logic             en,
    input  logic             we,
    input  logic [ADDRW-1:0] addr,
    input  logic [WIDTH-1:0] din,
    output logic [WIDTH-1:0] dout
);
    initial $display("SPRAM (%m)\n    WIDTH: %5d, DEPTH: %7d", WIDTH, DEPTH);

`ifndef BEHAV
    generate
    // TN28+
    // kv cache
    if ((WIDTH == 64) && (DEPTH == 2048)) begin
        rf_sp_2048x64 mem_macro_inst (
            .CLK(i_clk), .CEN(~en), .WEN(~we), .A(addr), .Q(dout), .D(din),
            .CENY(), .WENY(), .AY(), .DY(), .EMA(3'b0), .EMAW(2'b0), .EMAS(1'b0), .TEN(1'b1), .BEN(1'b1), .TCEN(1'b1), .TWEN(1'b0), .TA(9'b0), .TD(128'b0), .TQ(128'b0), .RET1N(1'b1), .STOV(1'b0)
        );
    end
    else begin
        $error("\n*PRESTO_ERROR: invalid SPRAM configuration, WIDTH: %d, DEPTH: %d\n    %m\n", WIDTH, DEPTH);
    end
    endgenerate
`else
    logic [WIDTH-1:0] mem [DEPTH-1:0];

    always_ff @ (posedge i_clk) begin
        dout <= (en && !we) ? mem[addr] : {WIDTH{1'bx}};
    end

    always_ff @ (posedge i_clk) begin
        if (en && we) begin
            mem[addr] <= din;
        end
    end
`endif  // BEHAV
endmodule
