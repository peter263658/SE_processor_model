/********************************************************************
* Filename: clk_gen.sv
* Authors:
*     Wen-Cong Huang
* Description:
*     Clock and reset generator for simulation
* Parameters:
*     - CYCLE_TIME:  clock cycle period (ns)
*     - MAX_CYCLE:   maximum cycle count until $finish
*     - RESET_DELAY: delay util async_rst is deasserted (ns)
* Note:
*
* Review History:
*     2021.04.10    Wen-Cong Huang
*********************************************************************/

module clk_gen # (
    parameter CYCLE_TIME  = 10.0,
    parameter MAX_CYCLE   = 10000,
    parameter RESET_DELAY = (5.5 * CYCLE_TIME)
)(
    output logic i_clk,
    output logic async_rst,
    output logic async_rst_n
);

    localparam HALF_CYCLE = (CYCLE_TIME / 2.0);
    always # (HALF_CYCLE) i_clk = ~i_clk;

    initial begin
        i_clk = 1'b1;
        async_rst_n = 1; async_rst = 0; # (              0.25 * CYCLE_TIME );
        async_rst_n = 0; async_rst = 1; # (RESET_DELAY - 0.25 * CYCLE_TIME );
        async_rst_n = 1; async_rst = 0; # (         MAX_CYCLE * CYCLE_TIME );
        $finish;
    end
endmodule
