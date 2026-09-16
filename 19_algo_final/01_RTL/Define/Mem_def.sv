/*
    Memory spec definition
*/

package Mem_def;

    // weight buffer  single port SRAM 
    parameter WBUF_BANKS  = 16; 
    parameter WBUF_WIDTH  = 64; 
    parameter WBUF_DEPTH  = 2048; 
    parameter WBUF_ADDRW  = $clog2(WBUF_DEPTH);
    parameter WBUF_LAST_ADDR = WBUF_DEPTH - 1;
    parameter WBUF_BANKS_ADDRW = $clog2(WBUF_BANKS);

    // KV cache  dual port sram
    parameter KVBUF_BANKS = 4;
    parameter KVBUF_WIDTH = 64;
    parameter KVBUF_DEPTH = 1024;
    parameter KVBUF_ADDRW = $clog2(KVBUF_DEPTH);
    parameter KVBUF_LAST_ADDR = KVBUF_DEPTH - 1;

    // dual-port SRAM     
    parameter UFBUF_BANKS = 8;
    parameter UFBUF_WIDTH = 64;
    parameter UFBUF_DEPTH = 128; 
    parameter UFBUF_ADDRW = $clog2(UFBUF_DEPTH);
    parameter UFBUF_LAST_ADDR = UFBUF_DEPTH - 1;
    // used for seperate the input feature and output feature
    parameter UFBUF_BANKS_2 = 4;

    // SKIP Buffer dp-sram
    parameter SKIPBUF_BANKS = 4;
    parameter SKIPBUF_WIDTH = 64;
    parameter SKIPBUF_DEPTH = 128;
    parameter SKIPBUF_ADDRW = $clog2(SKIPBUF_DEPTH);
    parameter SKIPBUF_LAST_ADDR = SKIPBUF_DEPTH - 1;

    typedef enum logic {  
        SRAMA_R_IDLE,
        SRAMA_R
    } SRAM_RCFG;

    typedef enum logic {  
        SRAMA_W_IDLE,
        SRAMA_W
    } SRAM_WCFG;

    // MB mode definition
    typedef enum logic [1 : 0] {
        MB_IDLE,
        MB_QK,   
        MB_PV,    
        MB_NM     
    } MB_mode;



    /////////////////////////////////////////////////////////////////////////
    //                              Debug                                  //
    /////////////////////////////////////////////////////////////////////////   

    
    // Debug signal 
    parameter DEBUG_DATA_WIDTH = 16;
    parameter DEBUG_MODE_NUM   = 32;
    parameter DEBUG_MODE_WIDTH = $clog2(DEBUG_MODE_NUM);


    // DM: debug mode 
    parameter DM_CORE_CTRL          = 5'd0;
    parameter DM_CORE_CTR_IFBUF     = 5'd1;
    parameter DM_CORE_CTRL_WBUF     = 5'd2;
    parameter DM_CORE_CTRL_OFBUF    = 5'd3;
    parameter DM_CORE_MAC1          = 5'd4;
    parameter DM_CORE_MAC2          = 5'd5;
    parameter DM_CORE_DPR1          = 5'd6;
    parameter DM_CORE_DPR2          = 5'd7;
    parameter DM_SPEC_EOPE          = 5'd8;
    parameter DM_SPEC_QTKA          = 5'd9;
    parameter DM_SPEC_QTQ           = 5'd10;
    parameter DM_SPEC_FILT          = 5'd11;
    parameter DM_SPEC_TE1           = 5'd12;
    parameter DM_SPEC_TE2           = 5'd13;
    parameter DM_IFBUF_TOP          = 5'd14;
    parameter DM_IFBUF_CTRL         = 5'd15;
    parameter DM_IFBUF_SRAM         = 5'd16;
    parameter DM_WBUF_TOP           = 5'd17;
    parameter DM_WBUF_CTRL          = 5'd18;
    parameter DM_WBUF_SRAM          = 5'd19;
    parameter DM_OFBUF_TOP          = 5'd20;
    parameter DM_OFBUF_CTRL         = 5'd21;
    parameter DM_TOP_CNT            = 5'd22; // a
    parameter DM_SPEC_BM            = 5'd23; // b3

    parameter DM_JUMP_SPEC1         = 5'd30; // b1
    parameter DM_JUMP_SPEC2         = 5'd31; // b2


endpackage
