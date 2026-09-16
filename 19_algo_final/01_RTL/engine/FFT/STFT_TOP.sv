/*
    Module:     STFT
    Note:       support stft, output with parallel 64 BIN output with natural order
    Behavior:   
    TODO:       should add istft function and support hop data, so save the current fft data.
*/


module STFT_TOP     
    import Param_def::*;
    import Mem_def::*;
(
    input  logic                     i_clk,         //  Master Clock
    input  logic                     i_rst_n,       //  Active low Asynchronous Reset
    input  STFT_mode                 i_stft_mode,        //  mode
    input  logic                     i_new_frame,
    input  logic                     i_dat_en,      //  Input Data Enable
    input  logic [FX_FULL_WIDTH-1:0] i_dat_re,      //  Input Data (Real)
    input  logic [FX_FULL_WIDTH-1:0] i_dat_im,      //  Input Data (Imag)

    output logic                             o_dat_en,      //  Output Data Enable
    output logic [63 : 0][FX_FULL_WIDTH-1:0] o_dat_re,     //  Output Data (Real)  Q1.15
    output logic [63 : 0][FX_FULL_WIDTH-1:0] o_dat_im,     //  Output Data (Imag)  Q1.15

    //  block-float e4m3 output for UFBUF: cast folds HW(÷128) vs model(÷√128)
    //  normalization + 2^i_k block-float exp (encoder.model.0.0.depthwise input = 9).
    //  o_e4m3_* valid on o_e4m3_en (= o_dat_en delayed 1 by the cast register).
    input  logic signed [7:0]                i_k,           //  consuming boundary block-float exp
    output logic                             o_e4m3_en,
    output logic [63 : 0][7:0]               o_e4m3_re,
    output logic [63 : 0][7:0]               o_e4m3_im,

    //  ISTFT path (reuses FFT128 as IFFT, i_stft_mode==ISTFT). Masked onesided spectrum
    //  in (bins 1..64, Q1.15, DC dropped); per-frame 128 synth-windowed time samples out
    //  (serial). hop overlap-add + normalization are OFF-CHIP / TB (see handoff).
    input  logic [63 : 0][FX_FULL_WIDTH-1:0] i_spec_re,     //  masked bins 1..64 (Q1.15)
    input  logic [63 : 0][FX_FULL_WIDTH-1:0] i_spec_im,
    input  logic                             i_istft_load,  //  latch masked spectrum
    input  logic                             i_istft_start, //  start IFFT stream (128)
    output logic                             o_istft_en,    //  time-sample valid (serial)
    output logic signed [FX_FULL_WIDTH-1:0]  o_istft_sample
);


    // input buffer
    logic                       i_hamm_dat_en;
    logic [FX_FULL_WIDTH-1 : 0] i_hamm_dat_re;
    logic [FX_FULL_WIDTH-1 : 0] i_hamm_dat_im;
    logic                       o_hamm_dat_en;
    logic [FX_FULL_WIDTH-1 : 0] o_hamm_dat_re;
    logic [FX_FULL_WIDTH-1 : 0] o_hamm_dat_im;
    logic [6               : 0] i_hamm_counter;
    logic [6               : 0] i_hamm_counter_buf;

    logic                       i_FFT_dat_en;
    logic [FX_FULL_WIDTH-1 : 0] i_FFT_dat_re;
    logic [FX_FULL_WIDTH-1 : 0] i_FFT_dat_im;
    logic                       o_FFT_dat_en;
    logic [FX_FULL_WIDTH-1 : 0] o_FFT_dat_re;
    logic [FX_FULL_WIDTH-1 : 0] o_FFT_dat_im;

    // bit reorder
    logic                     i_BR_new_frame;   //  New frame FFT point
    logic                     i_BR_dat_en;      //  Input Data Enable
    logic [FX_FULL_WIDTH-1:0] i_BR_dat_re;      //  Input Data (Real)
    logic [FX_FULL_WIDTH-1:0] i_BR_dat_im;      //  Input Data (Imag)

    logic                               o_BR_dat_en;        //  Output Data Enable
    logic [63 : 0][FX_FULL_WIDTH-1 : 0] o_BR_dat_re;        //  Output Data (Real), only save 64 BIN
    logic [63 : 0][FX_FULL_WIDTH-1 : 0] o_BR_dat_im;        //  Output Data (Imag), only save 64 BIN

    typedef enum logic [0 : 0] { 
        IDLE,    
        CACL    
    } state_fsm;

    state_fsm state_r, state_w;

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n)   state_r <= IDLE;
        else            state_r <= state_w;
    end

    always_comb begin 
        case(state_r) 
            IDLE: begin
                if (i_new_frame)    state_w = CACL;
                else                state_w = IDLE;
            end
            CACL: begin
                if (o_BR_dat_en)    state_w = IDLE;
                else                state_w = CACL;
            end
            default: state_w = IDLE;
        endcase
    end



    always_ff @(posedge i_clk) begin 
        i_hamm_dat_en <= i_dat_en;
        i_hamm_dat_re <= i_dat_re;
        i_hamm_dat_im <= i_dat_im;
        i_hamm_counter_buf <= i_hamm_counter;
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) i_hamm_counter <= 0;
        else if (i_dat_en) i_hamm_counter <= i_hamm_counter + 1;
        else if (!i_dat_en) i_hamm_counter <= 0;
    end

    Hamming u_hamming (
        .i_clk      (i_clk),
        .i_rst_n    (i_rst_n),
        .i_counter  (i_hamm_counter_buf),
        .i_dat_en   (i_hamm_dat_en),
        .i_dat_re   (i_hamm_dat_re),
        .i_dat_im   (i_hamm_dat_im),
        .o_dat_en   (o_hamm_dat_en),
        .o_dat_re   (o_hamm_dat_re),
        .o_dat_im   (o_hamm_dat_im)
    );


    assign i_FFT_dat_en = o_hamm_dat_en;     //  STFT analysis path (Hamming -> FFT)
    assign i_FFT_dat_re = o_hamm_dat_re;
    assign i_FFT_dat_im = o_hamm_dat_im;

    //  ---- ISTFT front: conj-symmetric feeder (masked spectrum -> IFFT input) ----
    wire                       is_istft = (i_stft_mode == ISTFT);
    logic                      feeder_en;
    logic [FX_FULL_WIDTH-1:0]  feeder_re, feeder_im;
    conj_sym_feeder u_feeder (
        .i_clk     (i_clk),
        .i_rst_n   (i_rst_n),
        .i_load    (i_istft_load),
        .i_spec_re (i_spec_re),
        .i_spec_im (i_spec_im),
        .i_start   (i_istft_start),
        .o_en      (feeder_en),
        .o_re      (feeder_re),
        .o_im      (feeder_im)
    );

    //  ---- shared FFT128: STFT = Hamming->FFT ; ISTFT = feeder->IFFT (÷128 = 1/N) ----
    wire                      fft_di_en = is_istft ? feeder_en : i_FFT_dat_en;
    wire [FX_FULL_WIDTH-1:0]  fft_di_re = is_istft ? feeder_re : i_FFT_dat_re;
    wire [FX_FULL_WIDTH-1:0]  fft_di_im = is_istft ? feeder_im : i_FFT_dat_im;

    FFT128 #(.WIDTH(FX_FULL_WIDTH))
    uFFT (
        .i_clk      (i_clk),
        .i_rst_n    (i_rst_n),
        .di_en      (fft_di_en),
        .di_re      (fft_di_re),
        .di_im      (fft_di_im),
        .do_en      (o_FFT_dat_en),
        .do_re      (o_FFT_dat_re),
        .do_im      (o_FFT_dat_im)
    );


    assign i_BR_dat_en = o_FFT_dat_en;
    assign i_BR_dat_re = o_FFT_dat_re;
    assign i_BR_dat_im = o_FFT_dat_im;
    assign i_BR_new_frame = i_new_frame;


    Bit_Reorder u_bit_reorder(
    .i_clk          (i_clk),
    .i_rst_n        (i_rst_n),
    .i_new_frame    (i_BR_new_frame),     //  New frame FFT point
    .i_dat_en       (i_BR_dat_en),        //  Input Data Enable
    .i_dat_re       (i_BR_dat_re),        //  Input Data (Real)
    .i_dat_im       (i_BR_dat_im),        //  Input Data (Imag)
    .o_dat_en       (o_BR_dat_en),        //  Output Data Enable
    .o_dat_re       (o_BR_dat_re),        //  Output Data (Real), only save 64 BIN
    .o_dat_im       (o_BR_dat_im)         //  Output Data (Imag), only save 64 BIN
    );

    //  STFT outputs gated off in ISTFT mode (FFT128 then carries IFFT data, not a spectrum)
    assign o_dat_en = (!is_istft && state_r == CACL) ? o_BR_dat_en : 1'b0;
    assign o_dat_re = o_BR_dat_re;
    assign o_dat_im = o_BR_dat_im;

    //  STFT(Q1.15) -> block-float e4m3 for UFBUF (see fxp_to_e4m3 / stft_fp8_cast).
    stft_fp8_cast u_fp8 (
        .i_clk     (i_clk),
        .i_rst_n   (i_rst_n),
        .i_dat_en  (o_dat_en),       //  FSM-gated single-cycle frame-valid
        .i_k       (i_k),
        .i_dat_re  (o_dat_re),       //  Q1.15, 64 bins (1..64)
        .i_dat_im  (o_dat_im),
        .o_dat_en  (o_e4m3_en),
        .o_e4m3_re (o_e4m3_re),
        .o_e4m3_im (o_e4m3_im)
    );

    //  ISTFT back: IFFT serial output (bit-reversed time) -> reorder + synth Hamming.
    //  hop overlap-add + normalization are OFF-CHIP / TB.
    logic                            iobuf_en;
    logic signed [FX_FULL_WIDTH-1:0] iobuf_sample;
    istft_reorder u_iobuf (
        .i_clk    (i_clk),
        .i_rst_n  (i_rst_n),
        .i_dat_en (o_FFT_dat_en),    //  IFFT serial output
        .i_dat_re (o_FFT_dat_re),    //  take real (imag ~0)
        .o_en     (iobuf_en),
        .o_sample (iobuf_sample)
    );
    assign o_istft_en     = is_istft ? iobuf_en : 1'b0;
    assign o_istft_sample = iobuf_sample;

endmodule