/*
    ppu_fma.sv — post-MAC FMA / PPU, ONE complex unit (re + im), mode-select.

    Modes (雷③ — do NOT hardwire BN+PReLU+cast; attention skips them):
      PPU_CONV  : BN affine (α·x + β) → PReLU (pow2 ±2^k) → e4m3 cast (next-layer k)
                  → o_re/o_im = e4m3 in [7:0]   (block-float activation out)
      PPU_PASS  : passthrough OACC fp16 (e.g. Score×V ctx → out_proj input)
      PPU_SCALE : APoT scale only (e.g. QK ×2^-2·2^-k descale before softmax) → fp16

    Zero-multiplier: α·x and the scales reuse `dealer` (2^a±2^b exponent-add);
    PReLU is pow2 (`fp16_scale2`); cast is `fp16_to_e4m3` (RNE [雷④, confirm vs fp8_qat]).
    BN α assumed overall-positive (this ckpt: overall_sign=+1); if a future retrain
    makes γ negative the 6a format needs an overall-sign bit — assert in packer.

    Throughput: drains 1 complex output / ⌈ic/32⌉ cycles → 1 unit keeps up; CE may
    instantiate a small pool (4) for margin. Combinational (regs live in CE/OACC).
*/
`timescale 1ns/10ps
module ppu_fma
    import Param_def::*;
(
    input  logic [1:0]  i_mode,        // PPU_CONV / PPU_PASS / PPU_SCALE
    input  logic [15:0] i_re,          // OACC real
    input  logic [15:0] i_im,          // OACC imag
    // BN (per output channel, complex): α = APoT-6a, β = fp16 fused bias
    input  logic [11:0] i_bn_alpha_re,
    input  logic [11:0] i_bn_alpha_im,
    input  logic [15:0] i_bn_beta_re,
    input  logic [15:0] i_bn_beta_im,
    // PReLU (per module, pow2 6b, real/imag scalars)
    input  logic [11:0] i_prelu_re,
    input  logic [11:0] i_prelu_im,
    // e4m3 cast: next layer's block-float exponent k (signed)
    input  logic signed [7:0] i_cast_k,
    // SCALE mode APoT code (e.g. QK descale), real/imag
    input  logic [11:0] i_scale_re,
    input  logic [11:0] i_scale_im,
    output logic [15:0] o_re,          // CONV: {8'b0, e4m3}; PASS/SCALE: fp16
    output logic [15:0] o_im,
    // pre-cast fp16 (CONV: post BN+PReLU, BEFORE x2^cast_k + e4m3). A SECOND consumer with a
    // different block-float k casts from THIS value (skip-write dual-cast, CONSULT B1(a)): each
    // consumer gets its own single quantization from the same fp16. PASS/SCALE: = the mode output.
    output logic [15:0] o_pre_re,
    output logic [15:0] o_pre_im
);
    localparam logic [1:0] PPU_CONV  = 2'd0;
    localparam logic [1:0] PPU_PASS  = 2'd1;
    localparam logic [1:0] PPU_SCALE = 2'd2;

    // ---- CONV: BN (α·x via dealer) + β ----
    logic [15:0] bn_ax_re, bn_ax_im, bn_re, bn_im;
    dealer u_bn_re (.i_sum(i_re), .i_apot(i_bn_alpha_re), .o_descaled(bn_ax_re));
    dealer u_bn_im (.i_sum(i_im), .i_apot(i_bn_alpha_im), .o_descaled(bn_ax_im));
    fp16_add_sat u_bnb_re (.a(bn_ax_re), .b(i_bn_beta_re), .z(bn_re));
    fp16_add_sat u_bnb_im (.a(bn_ax_im), .b(i_bn_beta_im), .z(bn_im));

    // ---- CONV: PReLU pow2 (y<0 → y·±2^k) ----
    logic signed [7:0] pk_re, pk_im;  logic ps_re, ps_im;
    apot6b_decode u_pd_re (.i_code(i_prelu_re), .o_k(pk_re), .o_sign(ps_re));
    apot6b_decode u_pd_im (.i_code(i_prelu_im), .o_k(pk_im), .o_sign(ps_im));
    logic [15:0] psc_re, psc_im, prelu_re, prelu_im;
    fp16_scale2 u_ps_re (.i_x(bn_re), .i_d(pk_re), .o_y(psc_re));
    fp16_scale2 u_ps_im (.i_x(bn_im), .i_d(pk_im), .o_y(psc_im));
    assign prelu_re = bn_re[15] ? {bn_re[15] ^ ps_re, psc_re[14:0]} : bn_re;
    assign prelu_im = bn_im[15] ? {bn_im[15] ^ ps_im, psc_im[14:0]} : bn_im;

    // ---- CONV: e4m3 cast with next-layer k ----
    logic [15:0] cst_re, cst_im;  logic [7:0] e4_re, e4_im;
    fp16_scale2 u_ck_re (.i_x(prelu_re), .i_d(i_cast_k), .o_y(cst_re));
    fp16_scale2 u_ck_im (.i_x(prelu_im), .i_d(i_cast_k), .o_y(cst_im));
    fp16_to_e4m3 u_e4_re (.i_h(cst_re), .o_e4m3(e4_re));
    fp16_to_e4m3 u_e4_im (.i_h(cst_im), .o_e4m3(e4_im));

    // ---- SCALE: APoT-only (reuse dealer) ----
    logic [15:0] sc_re, sc_im;
    dealer u_sc_re (.i_sum(i_re), .i_apot(i_scale_re), .o_descaled(sc_re));
    dealer u_sc_im (.i_sum(i_im), .i_apot(i_scale_im), .o_descaled(sc_im));

    always_comb begin
        case (i_mode)
            PPU_CONV:  begin o_re = {8'h00, e4_re}; o_im = {8'h00, e4_im}; end
            PPU_SCALE: begin o_re = sc_re;          o_im = sc_im;          end
            default:   begin o_re = i_re;           o_im = i_im;           end  // PPU_PASS
        endcase
    end

    // pre-cast tap (skip-write dual-cast): CONV = post-BN/PReLU fp16; PASS/SCALE = the mode output.
    always_comb begin
        case (i_mode)
            PPU_CONV:  begin o_pre_re = prelu_re; o_pre_im = prelu_im; end
            PPU_SCALE: begin o_pre_re = sc_re;    o_pre_im = sc_im;    end
            default:   begin o_pre_re = i_re;     o_pre_im = i_im;     end  // PPU_PASS
        endcase
    end
endmodule

// =======================================================================
// dealer — moved here from dealer.sv (file consolidation 2026-06-10; still a separate
// instantiated module, co-located with ppu_fma — its sibling descale unit under CE_core).
// =======================================================================
/*
    dealer.sv — per-PEG delayed APoT descale (one bolted to each PEG tree output).

    o = i_sum * (2^a [+/- 2^b])   where the 12-bit APoT-6a code is the FUSED descale
    (weight APoT exponent − activation 2^-k ; k baked in per-PEG static slot, §2/§2.1
    of CE_DATAPATH_DRAFT). Zero-multiplier: two exponent-adds + one FP16 add/sub.

    NOTE (2026-06-07 overflow fix): when a per-tensor coarse pow2 `i_preshift` is folded into
    the pe_pair product exponent upstream, this code carries the RESIDUAL descale
    (full_descale − preshift). For per-group INT4 paths preshift=0 → code unchanged. The logic
    here is identical either way; only the value the controller programs differs.

    Placement (雷②): BETWEEN the 32-way tree and OACC — per-group, pre-OACC. There
    are NUM_PEG(=4) dealers, one per PEG, so all 4 group-partials descale in parallel.
    Input/output FP16. Combinational.
*/
`timescale 1ns/10ps
module dealer
    import Param_def::*;
(
    input  logic [15:0] i_sum,      // PEG pre-descale group partial (FP16, saturating tree)
    input  logic [11:0] i_apot,     // fused APoT-6a descale code for this PEG/group
    output logic [15:0] o_descaled  // FP16
);
    logic signed [7:0] a, b;
    logic              has_second, res_sign;
    apot6a_decode u_dec (.i_code(i_apot), .o_a(a), .o_b(b),
                         .o_has_second(has_second), .o_res_sign(res_sign));

    logic [15:0] term_a, term_b, term_b_sel;
    fp16_scale2 u_a (.i_x(i_sum), .i_d(a), .o_y(term_a));
    fp16_scale2 u_b (.i_x(i_sum), .i_d(b), .o_y(term_b));

    // second term: −2^b if residual_sign else +2^b ; dropped if has_second=0
    assign term_b_sel = has_second ? {res_sign ^ term_b[15], term_b[14:0]} : 16'h0000;

    fp16_add_sat u_sum (.a(term_a), .b(term_b_sel), .z(o_descaled));
endmodule
