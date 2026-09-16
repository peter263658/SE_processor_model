/*
    Module:     fft bit reorder
    Note:       support fft reorder and save the whole frame data
    Behavior:   
*/

module Bit_Reorder     
    import Param_def::*;
    import Mem_def::*;
(   input  logic                     i_clk,
    input  logic                     i_rst_n,
    input  logic                     i_new_frame,   //  New frame FFT point
    input  logic                     i_dat_en,      //  Input Data Enable
    input  logic [FX_FULL_WIDTH-1:0] i_dat_re,      //  Input Data (Real)
    input  logic [FX_FULL_WIDTH-1:0] i_dat_im,      //  Input Data (Imag)

    output logic                               o_dat_en,        //  Output Data Enable
    output logic [63 : 0][FX_FULL_WIDTH-1 : 0] o_dat_re,        //  Output Data (Real), only save 64 BIN
    output logic [63 : 0][FX_FULL_WIDTH-1 : 0] o_dat_im         //  Output Data (Imag), only save 64 BIN
);    

    logic [63 : 0][FX_FULL_WIDTH-1 : 0] fft_bank_re_w, fft_bank_re_r;
    logic [63 : 0][FX_FULL_WIDTH-1 : 0] fft_bank_im_w, fft_bank_im_r;

    logic [FFT_SIZE_BIT-1 : 0] bit_counter;
    logic [FFT_SIZE_BIT-1 : 0] bit_reverse_counter;
    logic                      o_dat_en_w, o_dat_en_r;
    
    
    always_ff @ (posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            bit_counter <= 0;
            o_dat_en_r  <= 0;
        end
        else if (i_new_frame) begin
            bit_counter <= 0;
            o_dat_en_r  <= 0;
        end
        else begin
            if (i_dat_en) bit_counter <= bit_counter + 1;
            // clean 1-cycle pulse: fires the cycle after the last (bit_counter==127)
            // sample, self-clears when i_dat_en drops (do NOT rely on the top FSM to gate).
            o_dat_en_r  <= i_dat_en & o_dat_en_w;
        end
    end

    assign bit_reverse_counter = {bit_counter[0], bit_counter[1], bit_counter[2], bit_counter[3], bit_counter[4], bit_counter[5], bit_counter[6]};

    // Model (DCNN feature_extractors.Stft) drops DC (bin0) and keeps bins 1..64
    //   (onesided n_dft=128 -> 65 bins, then narrow(start=1) -> 64 bins, Nyquist kept).
    // So keep freq bins 1..64 and store bin k at bank index k-1:
    //   bank[0]=bin1 (lowest non-DC) ... bank[63]=bin64 (Nyquist).
    logic       keep_bin;
    logic [6:0] bank_idx;
    assign keep_bin = i_dat_en && (bit_reverse_counter >= 7'd1) && (bit_reverse_counter <= 7'd64);
    assign bank_idx = bit_reverse_counter - 7'd1;   // 0..63 when keep_bin

    always_comb begin
        fft_bank_re_w = fft_bank_re_r;
        fft_bank_im_w = fft_bank_im_r;
        if (keep_bin) begin
            fft_bank_re_w[bank_idx[5:0]] = i_dat_re;
            fft_bank_im_w[bank_idx[5:0]] = i_dat_im;
        end

    end

    always_ff @ (posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            fft_bank_re_r <= 0;
            fft_bank_im_r <= 0;
        end
        else begin
            fft_bank_im_r <= fft_bank_im_w;
            fft_bank_re_r <= fft_bank_re_w;
        end
    end

    assign o_dat_en_w = (bit_counter == 'd127)? 1 : 0;
    assign o_dat_en = o_dat_en_r;
    assign o_dat_re = fft_bank_re_r;
    assign o_dat_im = fft_bank_im_r;
    


endmodule