/*
    Module:     Hamming Windows Multiplication for STFT
    Note:       Support 2 Multiplication 
*/

module Hamming 
    import Param_def::*;
    import Mem_def::*;
(
    input  logic                        i_clk,         //  Master Clock
    input  logic                        i_rst_n,       //  Active low Asynchronous Reset
    input  logic [6:0]                  i_counter,     //  current number of the input signal
    input  logic                        i_dat_en,      //  Input Data Enable
    input  logic [FX_FULL_WIDTH-1:0]    i_dat_re,      //  Input Data (Real)
    input  logic [FX_FULL_WIDTH-1:0]    i_dat_im,      //  Input Data (Imag)
    output logic                        o_dat_en,      //  Output Data Enable
    output logic [FX_FULL_WIDTH-1:0]    o_dat_re,      //  Output Data (Real)
    output logic [FX_FULL_WIDTH-1:0]    o_dat_im       //  Output Data (Imag)        
);



    logic                               o_dat_en_w, o_dat_en_r;
    logic signed [FX_FULL_WIDTH-1:0]    o_dat_re_w, o_dat_re_r;
    logic signed [FX_FULL_WIDTH-1:0]    o_dat_im_w, o_dat_im_r;

    logic signed [FX_FULL_WIDTH-1:0]   i_mul_a_re, i_mul_b_re;
    logic signed [FX_FULL_WIDTH-1:0]   i_mul_a_im, i_mul_b_im;


    logic signed [2*FX_FULL_WIDTH-1:0] temp_re, temp_im;
    logic signed [2*FX_FULL_WIDTH:0] temp_round_re, temp_round_im;


    // Q1.15 (hamming) * Q2.14 (input) --> Q1.15 (output)
    localparam SHIFT_R        = 14;       
    localparam FX_MAX         = 16'sh7FFF;
    localparam FX_MIN         = 16'sh8000;
    localparam ROUND_BIAS     = 32'sh0000_2000; // 2^13 = round bias for >>SHIFT_R(14)

    // hamming coef 
    always_comb begin 
        i_mul_a_re = 0;
        if (i_dat_en) begin 
            case(i_counter)
                0: i_mul_a_re = 16'h0A3D;
                1: i_mul_a_re = 16'h0A50;
                2: i_mul_a_re = 16'h0A87;
                3: i_mul_a_re = 16'h0AE3;
                4: i_mul_a_re = 16'h0B64;
                5: i_mul_a_re = 16'h0C08;
                6: i_mul_a_re = 16'h0CD1;
                7: i_mul_a_re = 16'h0DBC;
                8: i_mul_a_re = 16'h0ECB;
                9: i_mul_a_re = 16'h0FFB;
                10: i_mul_a_re = 16'h114D;
                11: i_mul_a_re = 16'h12BF;
                12: i_mul_a_re = 16'h1451;
                13: i_mul_a_re = 16'h1601;
                14: i_mul_a_re = 16'h17CF;
                15: i_mul_a_re = 16'h19B9;
                16: i_mul_a_re = 16'h1BBE;
                17: i_mul_a_re = 16'h1DDE;
                18: i_mul_a_re = 16'h2016;
                19: i_mul_a_re = 16'h2265;
                20: i_mul_a_re = 16'h24CA;
                21: i_mul_a_re = 16'h2743;
                22: i_mul_a_re = 16'h29CF;
                23: i_mul_a_re = 16'h2C6B;
                24: i_mul_a_re = 16'h2F18;
                25: i_mul_a_re = 16'h31D2;
                26: i_mul_a_re = 16'h3498;
                27: i_mul_a_re = 16'h3769;
                28: i_mul_a_re = 16'h3A42;
                29: i_mul_a_re = 16'h3D22;
                30: i_mul_a_re = 16'h4007;
                31: i_mul_a_re = 16'h42F0;
                32: i_mul_a_re = 16'h45D9;
                33: i_mul_a_re = 16'h48C2;
                34: i_mul_a_re = 16'h4BA9;
                35: i_mul_a_re = 16'h4E8C;
                36: i_mul_a_re = 16'h5169;
                37: i_mul_a_re = 16'h543E;
                38: i_mul_a_re = 16'h570A;
                39: i_mul_a_re = 16'h59CA;
                40: i_mul_a_re = 16'h5C7E;
                41: i_mul_a_re = 16'h5F22;
                42: i_mul_a_re = 16'h61B7;
                43: i_mul_a_re = 16'h643A;
                44: i_mul_a_re = 16'h66A9;
                45: i_mul_a_re = 16'h6903;
                46: i_mul_a_re = 16'h6B47;
                47: i_mul_a_re = 16'h6D73;
                48: i_mul_a_re = 16'h6F85;
                49: i_mul_a_re = 16'h717D;
                50: i_mul_a_re = 16'h7359;
                51: i_mul_a_re = 16'h7518;
                52: i_mul_a_re = 16'h76B9;
                53: i_mul_a_re = 16'h783C;
                54: i_mul_a_re = 16'h799E;
                55: i_mul_a_re = 16'h7ADF;
                56: i_mul_a_re = 16'h7BFE;
                57: i_mul_a_re = 16'h7CFB;
                58: i_mul_a_re = 16'h7DD5;
                59: i_mul_a_re = 16'h7E8C;
                60: i_mul_a_re = 16'h7F1F;
                61: i_mul_a_re = 16'h7F8D;
                62: i_mul_a_re = 16'h7FD7;
                63: i_mul_a_re = 16'h7FFB;
                64: i_mul_a_re = 16'h7FFB;
                65: i_mul_a_re = 16'h7FD7;
                66: i_mul_a_re = 16'h7F8D;
                67: i_mul_a_re = 16'h7F1F;
                68: i_mul_a_re = 16'h7E8C;
                69: i_mul_a_re = 16'h7DD5;
                70: i_mul_a_re = 16'h7CFB;
                71: i_mul_a_re = 16'h7BFE;
                72: i_mul_a_re = 16'h7ADF;
                73: i_mul_a_re = 16'h799E;
                74: i_mul_a_re = 16'h783C;
                75: i_mul_a_re = 16'h76B9;
                76: i_mul_a_re = 16'h7518;
                77: i_mul_a_re = 16'h7359;
                78: i_mul_a_re = 16'h717D;
                79: i_mul_a_re = 16'h6F85;
                80: i_mul_a_re = 16'h6D73;
                81: i_mul_a_re = 16'h6B47;
                82: i_mul_a_re = 16'h6903;
                83: i_mul_a_re = 16'h66A9;
                84: i_mul_a_re = 16'h643A;
                85: i_mul_a_re = 16'h61B7;
                86: i_mul_a_re = 16'h5F22;
                87: i_mul_a_re = 16'h5C7E;
                88: i_mul_a_re = 16'h59CA;
                89: i_mul_a_re = 16'h570A;
                90: i_mul_a_re = 16'h543E;
                91: i_mul_a_re = 16'h5169;
                92: i_mul_a_re = 16'h4E8C;
                93: i_mul_a_re = 16'h4BA9;
                94: i_mul_a_re = 16'h48C2;
                95: i_mul_a_re = 16'h45D9;
                96: i_mul_a_re = 16'h42F0;
                97: i_mul_a_re = 16'h4007;
                98: i_mul_a_re = 16'h3D22;
                99: i_mul_a_re = 16'h3A42;
                100: i_mul_a_re = 16'h3769;
                101: i_mul_a_re = 16'h3498;
                102: i_mul_a_re = 16'h31D2;
                103: i_mul_a_re = 16'h2F18;
                104: i_mul_a_re = 16'h2C6B;
                105: i_mul_a_re = 16'h29CF;
                106: i_mul_a_re = 16'h2743;
                107: i_mul_a_re = 16'h24CA;
                108: i_mul_a_re = 16'h2265;
                109: i_mul_a_re = 16'h2016;
                110: i_mul_a_re = 16'h1DDE;
                111: i_mul_a_re = 16'h1BBE;
                112: i_mul_a_re = 16'h19B9;
                113: i_mul_a_re = 16'h17CF;
                114: i_mul_a_re = 16'h1601;
                115: i_mul_a_re = 16'h1451;
                116: i_mul_a_re = 16'h12BF;
                117: i_mul_a_re = 16'h114D;
                118: i_mul_a_re = 16'h0FFB;
                119: i_mul_a_re = 16'h0ECB;
                120: i_mul_a_re = 16'h0DBC;
                121: i_mul_a_re = 16'h0CD1;
                122: i_mul_a_re = 16'h0C08;
                123: i_mul_a_re = 16'h0B64;
                124: i_mul_a_re = 16'h0AE3;
                125: i_mul_a_re = 16'h0A87;
                126: i_mul_a_re = 16'h0A50;
                127: i_mul_a_re = 16'h0A3D;
                default: i_mul_a_re = 0;
            endcase
        end
    end


    assign i_mul_a_im = i_mul_a_re;
    assign i_mul_b_re = i_dat_re;
    assign i_mul_b_im = i_dat_im;


    assign temp_re = $signed(i_mul_a_re) * $signed(i_mul_b_re);
    assign temp_im = $signed(i_mul_a_im) * $signed(i_mul_b_im);

    assign temp_round_re = $signed({temp_re[2*FX_FULL_WIDTH-1],temp_re}) + ROUND_BIAS;
    assign temp_round_im = $signed({temp_im[2*FX_FULL_WIDTH-1],temp_im}) + ROUND_BIAS;     

    assign o_dat_re_w = ($signed(temp_round_re[2*FX_FULL_WIDTH:SHIFT_R]) > $signed({1'b0, FX_MAX}))? FX_MAX: ($signed(temp_round_re[2*FX_FULL_WIDTH:SHIFT_R]) < $signed({1'b1, FX_MIN}))? FX_MIN: $signed(temp_round_re[SHIFT_R+:FX_FULL_WIDTH]);
    assign o_dat_im_w = ($signed(temp_round_im[2*FX_FULL_WIDTH:SHIFT_R]) > $signed({1'b0, FX_MAX}))? FX_MAX: ($signed(temp_round_im[2*FX_FULL_WIDTH:SHIFT_R]) < $signed({1'b1, FX_MIN}))? FX_MIN: $signed(temp_round_im[SHIFT_R+:FX_FULL_WIDTH]);


    always_comb begin 
        o_dat_en_w = 0;
        if (i_dat_en) begin 
            o_dat_en_w = 'b1;
        end
    end


    always_ff @ (posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            o_dat_re_r <= 0;
            o_dat_im_r <= 0;
            o_dat_en_r <= 0;
        end
        else begin
            o_dat_re_r <= o_dat_re_w;
            o_dat_im_r <= o_dat_im_w;
            o_dat_en_r <= o_dat_en_w;
        end
    end

    assign o_dat_re = o_dat_re_r;
    assign o_dat_im = o_dat_im_r;
    assign o_dat_en = o_dat_en_r;


endmodule