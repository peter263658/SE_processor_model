
module s_buffer_v1 
    import NPU_def::*;
(
    input  logic i_clk,
    input  logic i_rst_n,

    // for speculation stage 2
    input  logic spec2_psum_valid,
    input  logic [ 8-1:0][SUM_WIDTH-1:0] spec2_rpsum,

    input  logic                         spec2_access,
    input  logic        [$clog2(64)-1:0] spec_cnt,
    output logic [64-1:0][SUM_WIDTH-1:0] spec2_s_hat,

    input  logic cnt_clear,

    // for formal S calc. // write
    input  logic s_calc_valid,
    input  logic [5:0] buf_cnt,
    input  logic [8-1:0][SUM_WIDTH-1:0] s_calc_data,

    // for formal O calc. // read
    input  logic o_calc_valid,
    output logic [64-1:0][8-1:0][FRC_WIDTH-1:0] o_calc_data

);

    logic         [$clog2(512)-1:0] scnt_w, scnt_r;
    logic [4096-1:0][SUM_WIDTH-1:0] smem_w, smem_r;
    always_comb begin
        spec2_s_hat = '0;
        if (spec2_access) begin
            for (int i = 0; i < 64; i++) begin
                spec2_s_hat[i] = smem_r[spec_cnt*64+i];
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

    always_comb begin
        smem_w = smem_r;
        scnt_w = scnt_r;
        if (spec2_psum_valid) begin
            scnt_w = scnt_r + 1;
            for (int j = 0; j < 8; j++) begin
                smem_w[scnt_r*8+j] = spec2_rpsum[j];
            end
        end
        else if (s_calc_valid) begin
            for (int j = 0; j < 8; j++) begin
                smem_w[buf_cnt*8+j] = s_calc_data[j];
            end
        end
        else if (cnt_clear) begin
            scnt_w = 0;
        end
    end
    always_ff @ (posedge i_clk or negedge i_rst_n) begin
        if (~i_rst_n) begin
            smem_r <= '0;
            scnt_r <= 0;
        end
        else begin
            smem_r <= smem_w;
            scnt_r <= scnt_w;
        end
    end

endmodule
