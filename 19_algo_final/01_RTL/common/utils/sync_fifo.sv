module sync_fifo # (
    parameter DATA_WIDTH = 8,
    parameter FIFO_DEPTH = 5,
    parameter FULL_THRES = (FIFO_DEPTH-1),
    parameter PTR_WIDTH = $clog2(FIFO_DEPTH)
)(
    input  logic                  clk,
    input  logic                  rst_n,
    input  logic                  ipush,
    input  logic [DATA_WIDTH-1:0] idata,
    input  logic                  ipop,
    output logic [DATA_WIDTH-1:0] odata,
    output logic                  full,
    output logic                  empty,
    output logic [   PTR_WIDTH:0] size
);
    // if (!(FULL_THRES < FIFO_DEPTH))
    //     $error("\n*ERROR: FULL_THRES should be less than FIFO_DEPTH, FULL_THRES: %d, FIFO_DEPTH: %d\n    %m\n", FULL_THRES, FIFO_DEPTH);

    logic [PTR_WIDTH-1:0] iptr_r, iptr_w;
    logic [PTR_WIDTH-1:0] optr_r, optr_w;
    logic [  PTR_WIDTH:0] size_r, size_w;
    assign empty = (size_r == 0);
    assign full = (size_r == FIFO_DEPTH);
    assign size = size_r;

    logic [FIFO_DEPTH-1:0][DATA_WIDTH-1:0] mem_r, mem_w;

    logic [DATA_WIDTH-1:0] odata_r, odata_w;
    assign odata = odata_r;

    always_comb begin
        iptr_w   = (ipush && !full) ? iptr_r + 1 : iptr_r;
        optr_w   = (ipop && !empty) ? optr_r + 1 : optr_r;
        odata_w  = mem_r[optr_w];
        size_w   = size_r;
        if (ipush && !full) begin
            size_w = size_r + 1;
        end
        else if (ipop && !empty) begin
            size_w = size_r - 1;
        end
    end
    always_ff @ (posedge clk or negedge rst_n) begin
        if (~rst_n) begin
            iptr_r   <= 0;
            optr_r   <= 0;
            size_r   <= 0;
            odata_r  <= 0;
        end
        else begin
            iptr_r   <= iptr_w;
            optr_r   <= optr_w;
            size_r   <= size_w;
            odata_r  <= odata_w;
        end
    end

    always_comb begin
        mem_w = mem_r;
        if (ipush) begin
            mem_w[iptr_r] = idata;
        end
    end
    always_ff @ (posedge clk or negedge rst_n) begin
        if (~rst_n) begin
            mem_r <= '0;
        end
        else begin
            mem_r <= mem_w;
        end
    end
endmodule