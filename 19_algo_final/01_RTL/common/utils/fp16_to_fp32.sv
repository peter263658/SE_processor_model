/*
    fp16_to_fp32.sv — exact IEEE half (1+5+10) -> single (1+8+23) widening.

    Pure combinational. Handles zero / normal / subnormal (leading-one normalise) / inf-nan.
    Used by mask_apply (the output-stage mode-C complex multiply) to bring the fp16 decoder
    mask into the fp32 product datapath. Exact: fp16 ⊂ fp32 (every fp16 value is representable
    in fp32), so the only work is exponent rebias (+112) and, for subnormals, normalisation.
*/
`timescale 1ns/10ps
module fp16_to_fp32 (
    input  logic [15:0] i_h,
    output logic [31:0] o_f
);
    wire        s = i_h[15];
    wire [4:0]  e = i_h[14:10];
    wire [9:0]  m = i_h[9:0];

    // leading-zero count of the 10-bit subnormal mantissa (0..9; 10 only when m==0)
    logic [3:0] lz;
    always_comb begin
        casez (m)
            10'b1?????????: lz = 4'd0;
            10'b01????????: lz = 4'd1;
            10'b001???????: lz = 4'd2;
            10'b0001??????: lz = 4'd3;
            10'b00001?????: lz = 4'd4;
            10'b000001????: lz = 4'd5;
            10'b0000001???: lz = 4'd6;
            10'b00000001??: lz = 4'd7;
            10'b000000001?: lz = 4'd8;
            10'b0000000001: lz = 4'd9;
            default:        lz = 4'd10;   // m == 0
        endcase
    end
    wire [9:0]  sub_man = m << (lz + 4'd1);          // drop the leading 1, left-align the frac bits

    logic [7:0]  exp32;
    logic [22:0] man32;
    always_comb begin
        if (e == 5'd0 && m == 10'd0) begin           // zero
            exp32 = 8'd0;            man32 = 23'd0;
        end else if (e == 5'd31) begin               // inf / nan (passthrough; not expected for a mask)
            exp32 = 8'hFF;          man32 = {m, 13'd0};
        end else if (e == 5'd0) begin                // subnormal fp16 -> normal fp32
            exp32 = 8'd112 - {4'd0, lz};             // unbiased (9-lz-24) + 127 = 112-lz
            man32 = {sub_man, 13'd0};
        end else begin                               // normal
            exp32 = e + 8'd112;                       // (e-15)+127
            man32 = {m, 13'd0};
        end
    end
    assign o_f = {s, exp32, man32};
endmodule
