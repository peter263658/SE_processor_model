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

module presto_2pram # (
    parameter WIDTH = 128,
    parameter DEPTH = 256,
    parameter ADDRW = $clog2(DEPTH)
)
(
    input  logic             i_clk,
    input  logic             i_rst_n,
    input  logic             ena,
    input  logic             wea,
    input  logic [ADDRW-1:0] addra,
    input  logic [WIDTH-1:0] dina,
    input  logic             enb,
    input  logic [ADDRW-1:0] addrb,
    output logic [WIDTH-1:0] doutb
);
    initial $display("2PRAM (%m)\n    WIDTH: %5d, DEPTH: %7d", WIDTH, DEPTH);

`ifndef BEHAV
    generate
    // TN40G
    if ((WIDTH == 256) && (DEPTH == 16)) begin
        rf_2p_256x16 mem_macro_inst (
            .CENYA(), .AYA(), .CENYB(), .AYB(), .DYB(),
            .CLKA(i_clk), .CLKB(i_clk),
            .CENA(~enb), .AA(addrb), .QA(doutb),
            .CENB(~(ena&wea)), .AB(addra), .DB(dina),
            .EMAA(3'b0), .EMASA(1'b0), .EMAB(3'b0), .EMAWB(2'b0), .TENA(1'b1), .BENA(1'b1), .TCENA(1'b1), 
            .TAA(4'b0), .TQA(256'b0), .TENB(1'b1), .TCENB(1'b1), .TAB(4'b0), .TDB(256'b0), .RET1N(1'b1), .STOVA(1'b0), .STOVB(1'b0), .COLLDISN(1'b1)
        );
    end
    else if ((WIDTH == 256) && (DEPTH == 32)) begin
        rf_2p_256x32 mem_macro_inst (
            .CENYA(), .AYA(), .CENYB(), .AYB(), .DYB(),
            .CLKA(i_clk), .CLKB(i_clk),
            .CENA(~enb), .AA(addrb), .QA(doutb),
            .CENB(~(ena&wea)), .AB(addra), .DB(dina),
            .EMAA(3'b0), .EMASA(1'b0), .EMAB(3'b0), .EMAWB(2'b0), .TENA(1'b1), .BENA(1'b1), .TCENA(1'b1), 
            .TAA(5'b0), .TQA(256'b0), .TENB(1'b1), .TCENB(1'b1), .TAB(5'b0), .TDB(256'b0), .RET1N(1'b1), .STOVA(1'b0), .STOVB(1'b0), .COLLDISN(1'b1)
        );
    end
    else if ((WIDTH == 256) && (DEPTH == 64)) begin
        rf_2p_256x64 mem_macro_inst (
            .CENYA(), .AYA(), .CENYB(), .AYB(), .DYB(),
            .CLKA(i_clk), .CLKB(i_clk),
            .CENA(~enb), .AA(addrb), .QA(doutb),
            .CENB(~(ena&wea)), .AB(addra), .DB(dina),
            .EMAA(3'b0), .EMASA(1'b0), .EMAB(3'b0), .EMAWB(2'b0), .TENA(1'b1), .BENA(1'b1), .TCENA(1'b1), 
            .TAA(6'b0), .TQA(256'b0), .TENB(1'b1), .TCENB(1'b1), .TAB(6'b0), .TDB(256'b0), .RET1N(1'b1), .STOVA(1'b0), .STOVB(1'b0), .COLLDISN(1'b1)
        );
    end
    else if ((WIDTH == 256) && (DEPTH == 8)) begin
        rf_2p_256x8 mem_macro_inst (
            .CENYA(), .AYA(), .CENYB(), .AYB(), .DYB(),
            .CLKA(i_clk), .CLKB(i_clk),
            .CENA(~enb), .AA(addrb), .QA(doutb),
            .CENB(~(ena&wea)), .AB(addra), .DB(dina),
            .EMAA(3'b0), .EMASA(1'b0), .EMAB(3'b0), .EMAWB(2'b0), .TENA(1'b1), .BENA(1'b1), .TCENA(1'b1), 
            .TAA(3'b0), .TQA(256'b0), .TENB(1'b1), .TCENB(1'b1), .TAB(3'b0), .TDB(256'b0), .RET1N(1'b1), .STOVA(1'b0), .STOVB(1'b0), .COLLDISN(1'b1)
        );
    end
    else begin
        $error("\n*PRESTO_ERROR: invalid 2PRAM configuration, WIDTH: %d, DEPTH: %d\n    %m\n", WIDTH, DEPTH);
    end
    endgenerate
`else
    logic [WIDTH-1:0] mem [DEPTH-1:0];
    // assert when write en and read en simultaneously
    //assert ((ena & enb & wea) && (addrb == addra)) $error("Retention Occur!!!!!");

    always_ff @ (posedge i_clk) begin
        doutb <= enb ? mem[addrb] : {WIDTH{1'bx}};
    end

    always_ff @ (posedge i_clk) begin
        if (ena && wea) begin
            mem[addra] <= dina;
        end
    end
`endif  // BEHAV
endmodule
