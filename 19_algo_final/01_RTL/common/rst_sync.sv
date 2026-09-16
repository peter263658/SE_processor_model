/********************************************************************
* Filename: rst_sync.sv
* Authors:
*     Wen-Cong Huang
* Description:
*     Reset synchronizer for simulation
* Parameters:
*
* Note:
*
* Review History:
*     2021.04.10    Wen-Cong Huang
*********************************************************************/

module rst_sync (
    input  logic i_clk,
    input  logic async_rst,
    output logic rst
);
    initial begin
        rst = 0;
        forever @ (posedge i_clk) rst = async_rst;
    end
endmodule
