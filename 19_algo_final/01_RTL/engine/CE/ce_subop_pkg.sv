/*
    ce_subop_pkg.sv — CE sub-op descriptor type + generated ROM (Step 2-2).

    Holds the packed descriptor `subop_t`, the PPU mode codes, and the generated
    `ce_subop_rom()` function (from manifest.json via gen_ce_subop_rom.py). Kept in a
    PACKAGE so the type can be used in module port lists (e.g. inst_decoder.o_subop).

    Compile order (filelist): Param_def.sv -> ce_subop_rom.svh present -> THIS -> inst_decoder.sv.
*/
package ce_subop_pkg;
    import Param_def::*;

    // PPU mode-select codes (match ppu_fma localparams)
    localparam logic [1:0] PPU_CONV = 2'd0, PPU_PASS = 2'd1, PPU_SCALE = 2'd2;

    // weight-scale granularity -> dealer code index formula (mem_controller §4)
    //   PERGROUP: per (oc, 32-ic-group)  idx = oc*n_chunk + ic_chunk     (INT4 conv pointwise)
    //   PERTENSOR: 1 code/branch          idx = 0                        (FP8-outlier)
    //   PERHEAD: per head                 idx = oc / (oc/NUM_HEAD)        (attention projections)
    //   NONE: no weight scale            n_codes = 0 -> prefetch no-op   (QK / Score×V; dealer=identity)
    localparam logic [1:0] SCALE_PERGROUP = 2'd0, SCALE_PERTENSOR = 2'd1, SCALE_PERHEAD = 2'd2,
                           SCALE_NONE = 2'd3;

    // per-COMPLEX-sub-op descriptor (decision A): each entry pairs the _r and _i weight
    // tensors of one complex MAC. PEG0/2 <- (w_r_base, wscale_r, k_real, preshift_r);
    // PEG1/3 <- (w_i_base, wscale_i, k_imag, preshift_i). Field order MUST match
    // ce_subop_rom.svh emission (gen_ce_subop_rom.py).
    typedef struct packed {
        CE_LANE_mode       pe_mode;      // MODE_W4A8 / LMUL_FP8 / LMUL_ATTN / W4A16
        logic [1:0]        ppu_mode;     // PPU_CONV / PASS / SCALE
        logic              is_dw;        // depthwise (F taps accumulate in OACC)
        logic [9:0]        ic;           // in channels (contraction); DW = 1
        logic [9:0]        oc;           // out channels
        logic [2:0]        F;            // taps (DW = 5, PW = 1)
        logic [2:0]        groups;       // decoder pw groups_x=4 / skip=2; else 1
        logic [17:0]       w_r_base;     // WBUF byte offset, REAL weight tensor (PEG0/2)
        logic [17:0]       w_i_base;     // WBUF byte offset, IMAG weight tensor (PEG1/3); stride varies
        logic [17:0]       wscale_r;     // weight-scale const byte offset (real)
        logic [17:0]       wscale_i;     // weight-scale const byte offset (imag)
        logic [17:0]       bn_ptr;       // BN-scale const byte offset (per block; 0 if no BN)
        logic [17:0]       prelu_ptr;    // PReLU const byte offset (per block; 0 if no PReLU)
        logic              has_bn;
        logic              has_prelu;
        logic signed [7:0] k_real;       // block-float k, real act (input_activation.descale_pow2_exp)
        logic signed [7:0] k_imag;       // block-float k, imag act
        logic signed [7:0] preshift_r;   // per-tensor pre-tree pow2 (real, PEG0/2); 0 = full descale in dealer
        logic signed [7:0] preshift_i;   // per-tensor pre-tree pow2 (imag, PEG1/3)
        logic              branches_collapse; // 1 = single fused code (kreal==kimag): PEG0/2<-r, PEG1/3<-i (2 codes);
                                              // 0 = kreal||kimag (split): per-PEG half-select +12bit (4 codes)
        logic [1:0]        scale_mode;        // SCALE_PERGROUP / PERTENSOR / PERHEAD (weight-scale index)
        logic              blockdiag;         // 1 = mha_in_proj qkv block-diagonal: act slice
                                              // x[((oc/ic)%4)*ic:], scale idx = oc/16 (48 codes)
        logic              act_is_skip;       // 1 = decoder pw_skip: act windows read SKIPBUF, not UF
        logic [11:0]       act_base;          // act base byte offset (skip: stage block base @ f=0;
                                              // re half; im at +SKIP_IM_HALF). UF sub-ops: 0.
    } subop_t;

    // localparam int N_SUBOP; function automatic subop_t ce_subop_rom(input int unsigned idx);
    `include "ce_subop_rom.svh"

endpackage
