/********************************************************************
* Filename: presto_2pram.sv
* Authors:
*     Wen-Cong Huang
* Description:
*     Two-port SRAM modules for Presto
* Parameters:
*     - WIDTH: SRAM data width
*     - DEPTH: SRAM entry depth
*     - ADDRW: SRAM address width
* Note:
*     - SRAMs are replace with real SRAM macros during synthesize.
*     - Use `if ((WIDTH == X) && (DEPTH == Y))` for SRAM selection,
*       if no matching WIDTH and DEPTH, call $error() function.
* Review History:
*     2021.04.10    Wen-Cong Huang
*********************************************************************/

module presto_dpram # (
    parameter WIDTH = 128,
    parameter DEPTH = 256,
    parameter ADDRW = $clog2(DEPTH)
)
(
    input  logic             i_clk,
    input  logic             i_rst_n,
    // Port A: read / write
    input  logic             ena,
    input  logic             wea,
    input  logic [ADDRW-1:0] addra,
    input  logic [WIDTH-1:0] dina,
    output logic [WIDTH-1:0] douta,
    // Port B: read / write
    input  logic             enb,
    input  logic             web,
    input  logic [ADDRW-1:0] addrb,
    input  logic [WIDTH-1:0] dinb,
    output logic [WIDTH-1:0] doutb
);
    initial $display("2PRAM (%m)\n    WIDTH: %5d, DEPTH: %7d", WIDTH, DEPTH);

`ifndef BEHAV
    generate
    // TN28+
    if ((WIDTH == 64) && (DEPTH == 1024)) begin
        rf_dp_1024x64 mem_macro_inst (
            .AA(addra), .DA(dina), .WEBA(wea), .CEBA(~ena), .CLK(i_clk),
            .AB(addrb), .DB(dinb), .WEBB(web), .CEBB(~enb),
            .QA(douta), .QB(doutb), 
            .RTSEL(2'b00), .WTSEL(2'b00), .PTSEL(2'b00)
        );
    end
    else if ((WIDTH == 64) && (DEPTH == 128)) begin
        rf_dp_128x64 mem_macro_inst (
            .AA(addra), .DA(dina), .WEBA(wea), .CEBA(~ena), .CLK(i_clk),
            .AB(addrb), .DB(dinb), .WEBB(web), .CEBB(~enb),
            .QA(douta), .QB(doutb), 
            .RTSEL(2'b00), .WTSEL(2'b00), .PTSEL(2'b00)
        );
    end

    else begin
        $error("\n*PRESTO_ERROR: invalid 2PRAM configuration, WIDTH: %d, DEPTH: %d\n    %m\n", WIDTH, DEPTH);
    end
    endgenerate
`else
    // ---------------------------------------------------------------
    // Behavior model: True Dual-Port SRAM
    //   Port A: independent read/write
    //   Port B: independent read/write
    //   Collision policy (matches real SRAM macro):
    //     - WA vs WB same address : both writes corrupted -> mem[addr] = X
    //     - WA vs RB same address : doutb is unknown     -> doutb = X
    //     - WB vs RA same address : douta is unknown     -> douta = X
    // ---------------------------------------------------------------
    logic [WIDTH-1:0] mem [DEPTH-1:0];

    // ---- Port A read, Port B read ----
    always_ff @(posedge i_clk) begin
        if (ena && !wea) begin
            // Read: if B is writing to same address, result is X
            if (enb && web && (addrb == addra))
                douta <= {WIDTH{1'bx}};
            else
                douta <= mem[addra];
        end
        if (enb && !web) begin
            // Read: if A is writing to same address, result is X
            if (ena && wea && (addra == addrb))
                doutb <= {WIDTH{1'bx}};
            else
                doutb <= mem[addrb];
        end
    end

    // ---- Port A write, Port B write (single driver for mem) ----
    always_ff @(posedge i_clk) begin
        if (ena && wea && enb && web && (addra == addrb)) begin
            // Write-Write collision: both corrupt
            mem[addra] <= {WIDTH{1'bx}};
        end else begin
            if (ena && wea)
                mem[addra] <= dina;
            if (enb && web)
                mem[addrb] <= dinb;
        end
    end
`endif  // BEHAV
endmodule
