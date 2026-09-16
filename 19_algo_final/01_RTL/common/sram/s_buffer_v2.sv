
module s_buffer_v2 
    import NPU_def::*;
(
    input  logic i_clk,
    input  logic i_rst_n,

    // for formal S calc. // write
    input  logic s_calc_valid,
    input  logic [5:0] buf_cnt,
    input  logic [8-1:0][SUM_WIDTH-1:0] s_calc_data,

    // for formal O calc. // read
    input  logic o_calc_valid,
    output logic [64-1:0][8-1:0][FRC_WIDTH-1:0] o_calc_data


);

    logic [4096-1:0][SUM_WIDTH-1:0] smem_w, smem_r;

    always_comb begin
        smem_w = smem_r;
        if (s_calc_valid) begin
            for (int j = 0; j < 8; j++) begin
                smem_w[buf_cnt*8+j] = s_calc_data[j];
            end
        end
    end
    always_comb begin
        o_calc_data = '0;
        if (o_calc_valid) begin
            for (int i = 0; i < 64; i++) begin
                for (int j = 0; j < 8; j++) begin
                    o_calc_data[i][j] = {smem_r[i*8+j][SUM_WIDTH-1], smem_r[i*8+j][FRC_WIDTH-2:0]};
                end
            end
        end
    end
    always_ff @ (posedge i_clk or negedge i_rst_n) begin
        if (~i_rst_n) begin
            smem_r <= '0;
        end
        else begin
            smem_r <= smem_w;
        end
    end

endmodule