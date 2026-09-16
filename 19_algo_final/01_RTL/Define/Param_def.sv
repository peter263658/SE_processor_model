/*
    Parameter & configuration definition
*/

package Param_def;

    // // softmax
    parameter SEQ_LEN           = 32;
    parameter SEQ_LEN_BIT       = $clog2(SEQ_LEN);
    parameter SM_SCAN_LANES      = 8;            // Softmax_v3 Top-K scan lanes/cycle (its own param;
                                                 // CE's NUM_PEG=4 below = # of complex PE groups)
    parameter NUM_SM_INST       = 8;
    /*
        STFT configuration
    */
    parameter HAMM_MUL_NUM      = 4;
    parameter FX_FRAC           = 15;
    parameter FX_INT            = 1;
    parameter FX_FULL_WIDTH     = FX_FRAC + FX_INT;

    /*
        Universal configuration
    */

    // using fp8 e4m3 for activation
    parameter FP_EXPN_WIDTH     = 4;
    parameter FP_FRAC_WIDTH     = 3;
    parameter FP_FULL_WIDTH     = FP_EXPN_WIDTH + FP_FRAC_WIDTH + 1;
    parameter FP_ZERO           = 8'b0;

    // high precision floating point (FP16 = 1+5+10, bias 15)
    parameter FP_HP_EXPN_WIDTH     = 5;
    parameter FP_HP_FRAC_WIDTH     = 10;
    parameter FP_HP_FULL_WIDTH     = FP_HP_EXPN_WIDTH + FP_HP_FRAC_WIDTH + 1;   // = 16
    parameter FP_HP_ZERO           = 16'b0;

    // FP16 attention score / softmax datapath (= FP_HP / FP16, per CLAUDE.md). Used by Softmax_v3.
    parameter ATTN_FP_EXPN_WIDTH   = FP_HP_EXPN_WIDTH;   // 5
    parameter ATTN_FP_FRAC_WIDTH   = FP_HP_FRAC_WIDTH;   // 10
    parameter ATTN_FP_WIDTH        = FP_HP_FULL_WIDTH;   // 16


    parameter FFT_SIZE          = 128;
    parameter FFT_SIZE_BIT      = $clog2(FFT_SIZE);
    parameter SLIDE_WIN_LEN     = 32;
    parameter SLIDE_WIN_LEN_BIT = $clog2(SLIDE_WIN_LEN);
    parameter HEAD_SIZE         = 16;      // dim 512->256, head size = 256/16=16
    parameter NUM_HEAD          = 16;
    parameter DIM_SIZE          = HEAD_SIZE * NUM_HEAD;
    parameter DIM_SIZE_BIT      = $clog2(DIM_SIZE);

    // PEG specification
    parameter NUM_PE_PG         = 32;                           // number of the PE per group
    parameter NUM_PEG           = 4;                            // number of PE group
    parameter NUM_PE            = NUM_PE_PG * NUM_PEG;          // number of the PE in total 


    // IO specification
    parameter   CHIP_I_WIDTH   = 128;
    parameter   CHIP_O_WIDTH   = 8;

    // DEBUG signal
    parameter  DEBUG_WIDTH = 16;


    // exec mode, indicate which I/O mode doing now
    typedef enum logic [2 : 0] {
        EXEC_ENC,
        EXEC_ATTN,
        EXEC_DEC,
        EXEC_IDLE,
        EXEC_FEAT
    } Exec_mode;

    // CE mode definition
    typedef enum logic [1 : 0] {
        CE_DW,
        CE_PW,
        CE_IDLE
    } CE_mode;

    typedef enum logic [0 : 0] {
        AT_GLAC,      // vector accumulate
        AT_PSUM     // partial sum
    } AT_mode;

    typedef enum logic [2 : 0] {
        CE_L1,
        CE_L2,
        CE_L3,
        CE_L4,
        CE_L5,
        CE_L6
    } CE_LAYER_mode;


    typedef enum logic [0 : 0] {
        STFT,
        ISTFT
    } STFT_mode;

    typedef enum logic [1 : 0] {
        MODE_W4A8,
        MODE_LMUL_FP8,
        MODE_LMUL_ATTN,
        MODE_W4A16
    } CE_LANE_mode;


endpackage
