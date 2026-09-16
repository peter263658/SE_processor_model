/*
    ce_kv_cache.sv — attention KV cache: 4-path pass-indexed history + 32-frame ring (task #4).

    Holds K and V for the 32-frame causal sliding window of the tied-MHA 2-pass attention, as raw
    e4m3 block-float codes (1 B each). The tied attention runs M(x.real) [pass A] then M(x.imag)
    [pass B]; the two passes produce ENTIRELY different K/V histories (~99% of entries differ), so a
    single region cannot serve both. The deploy therefore uses FOUR independent paths that exactly
    fill the 32 KiB KVBUF (memory_map.json["kvbuf"], consultant pin 2026-06-09):

        path   region        byte offset   rows
        K_re   pass A, K      0            0..255     descale x 2^-5
        K_im   pass B, K      8192         256..511   descale x 2^-5
        V_re   pass A, V      16384        512..767   descale x 2^-5
        V_im   pass B, V      24576        768..1023  descale x 2^-5
        ---- 4 x 8192 = 32768 B = KVBUF capacity (zero slack); e4m3 1-byte is what makes it fit ----

    Element address (consultant element_addressing): byte = path_base + head*512 + frame_slot*16 + dim
        path_base = sel*16384 + pass*8192   (sel 0=K/1=V, pass 0=re/1=im)
        head_stride 512 (= 32 frame x 16 dim), frame_stride 16, dim_stride 1.

    RING POINTER (32-frame streaming window, NOT a 63-frame chunk window). One ptr shared by all heads:
        write:   current frame -> ring slot = ptr            (the in_proj store writes local_idx 0 = ptr)
        read:    window entry local_idx d -> slot (ptr - d) mod 32   (local_idx 0 = newest = ptr)
        advance: ptr = (ptr + 1) mod 32 AFTER a frame's QK + score_v complete (i_advance pulse).
    SEQ_LEN = 32 = 2^5, so the (ptr - local_idx) mod 32 is a free 5-bit subtraction. Reset ptr = 0.
    Verified vs the deploy reference (kv_ring_reference.json): dequant(ring[(31-d) mod 32]) == the
    local_idx-order qstar_Kwindow[d] for all 16 heads at the query frame (ptr=31).

    Block-float exponent rides i_*_k:
        write fp16-mode:  code = fp16_to_e4m3( fp16 * 2^i_wr_cast_k )   (legacy tb_kv_cache path)
        write code-mode:  store i_wr_fp16[d][7:0] raw                  (in_proj PPU already e4m3-cast)
        read:             fp16 = fp8_to_fp16(code) * 2^i_rd_descale_k   (i_rd_descale_k = +k = -5)
    k_Q = k_K = k_V = -5 (shared, pass-independent; only the K/V DATA is pass-split).

    STORAGE: behavioral byte memory (golden-ready, TB-preloadable). The synth-time swap is the banked
    presto macro kvbuf_top (4 x 64b x 1024 = 32 KiB dual-port); the {path,head,slot,dim} offset here maps
    to its bank/row split (deferred synth task with WBUF->presto_spram). Read is registered (1-cycle,
    models SRAM latency); the address translation + cast are combinational.
*/
`timescale 1ns/10ps
module ce_kv_cache
    import Param_def::*;
    import Mem_def::*;
(
    input  logic i_clk,
    input  logic i_rst_n,

    // ---- RING pointer (one shared by all heads; reset 0, advance after each frame's QK+score_v) ----
    input  logic                                  i_advance,     // pulse: ptr = (ptr+1) mod 32
    output logic [$clog2(SEQ_LEN)-1:0]            o_ptr,         // current ring pointer (debug / chaining)
    output logic [5:0]                            o_filled,      // COMPLETED frames, saturates at 32:
                                                                 // warm-up validity = min(filled+1, 32)
                                                                 // valid keys for the in-flight frame

    // ---- WRITE port (in_proj store): the CURRENT frame's (pass,sel,head) = HEAD_SIZE codes -> slot=ptr ----
    // i_wr_code_mode=0 -> fp16 source, cast (x 2^cast_k) -> e4m3 (legacy/tb_kv_cache path);
    // i_wr_code_mode=1 -> source is ALREADY an e4m3 code (the in_proj PPU output); store [7:0] raw.
    input  logic                                  i_wr_en,
    input  logic                                  i_wr_code_mode,// 1 = store raw e4m3 code from [7:0]
    input  logic                                  i_wr_pass,     // 0=re (pass A / x.real), 1=im (pass B / x.imag)
    input  logic                                  i_wr_sel,      // 0=K, 1=V
    input  logic [$clog2(NUM_HEAD)-1:0]           i_wr_head,
    input  logic signed [7:0]                     i_wr_cast_k,   // = -k_blockfloat (fp16-mode only)
    input  logic [HEAD_SIZE-1:0][15:0]            i_wr_fp16,

    // ---- READ port (QK all-frames / Score×V top-4): window entry local_idx -> slot (ptr-local_idx) ----
    input  logic                                  i_rd_en,
    input  logic                                  i_rd_pass,
    input  logic                                  i_rd_sel,
    input  logic [$clog2(NUM_HEAD)-1:0]           i_rd_head,
    input  logic [$clog2(SEQ_LEN)-1:0]            i_rd_local_idx, // window position (0 = current/newest = ptr)
    input  logic signed [7:0]                     i_rd_descale_k, // = +k_blockfloat (= -5)
    output logic [HEAD_SIZE-1:0][15:0]            o_rd_fp16,
    output logic                                  o_rd_valid
);
    localparam int PER_HEAD    = SEQ_LEN * HEAD_SIZE;          // 32*16 = 512 bytes / head
    localparam int PASS_STRIDE = NUM_HEAD * PER_HEAD;          // 16*512 = 8192 bytes / path (one pass of K or V)
    localparam int SEL_STRIDE  = 2 * PASS_STRIDE;              // 16384 bytes / (K-block or V-block)
    localparam int KV_BYTES    = 4 * PASS_STRIDE;              // 32768 bytes = 4 paths = KVBUF capacity

    logic [7:0] kv_mem [0:KV_BYTES-1];                         // behavioral; synth swap = kvbuf_top

    // path_base + head*512 + frame_slot*16 + dim  (frame_slot is the PHYSICAL ring slot)
    function automatic int kv_off(input logic pass,
                                  input logic sel,
                                  input logic [$clog2(NUM_HEAD)-1:0] h,
                                  input logic [$clog2(SEQ_LEN)-1:0]  slot);
        kv_off = (sel ? SEL_STRIDE : 0) + (pass ? PASS_STRIDE : 0) + h*PER_HEAD + slot*HEAD_SIZE;
    endfunction

    // ---- RING pointer + filled counter (cold-start warm-up) ----
    // plain `always` (not always_ff) so a steady-state TB may poke `filled = 32` (full-window
    // mask) without ICPD — same TB-pokeable convention as kv_mem / skip_wr_en.
    logic [$clog2(SEQ_LEN)-1:0] ptr;
    logic [5:0]                 filled;     // completed frames, saturating at SEQ_LEN(32)
    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            ptr    <= '0;
            filled <= '0;
        end else if (i_advance) begin
            ptr    <= ptr + 1'b1;                              // 5-bit add wraps mod 32
            if (filled != 6'(SEQ_LEN)) filled <= filled + 6'd1;
        end
    end
    assign o_ptr    = ptr;
    assign o_filled = filled;

    // write goes to the current frame's slot (= ptr); read translates local_idx -> physical slot.
    wire [$clog2(SEQ_LEN)-1:0] wr_slot = ptr;
    wire [$clog2(SEQ_LEN)-1:0] rd_slot = ptr - i_rd_local_idx; // 5-bit sub = (ptr - local_idx) mod 32

    // ---- WRITE: fp16 -> (x 2^cast_k) -> e4m3 (or raw code), store HEAD_SIZE bytes (synchronous) ----
    logic [HEAD_SIZE-1:0][15:0] wr_scaled;
    logic [HEAD_SIZE-1:0][7:0]  wr_code;
    genvar wd;
    generate
        for (wd = 0; wd < HEAD_SIZE; wd++) begin : g_wr_cast
            fp16_scale2     u_ws (.i_x(i_wr_fp16[wd]), .i_d(i_wr_cast_k), .o_y(wr_scaled[wd]));
            fp16_to_e4m3    u_we (.i_h(wr_scaled[wd]),                    .o_e4m3(wr_code[wd]));
        end
    endgenerate

    // plain `always` (not always_ff) so a golden TB may also $readmemh/poke-preload kv_mem.
    always @(posedge i_clk) begin
        if (i_wr_en)
            for (int d = 0; d < HEAD_SIZE; d++)
                kv_mem[kv_off(i_wr_pass, i_wr_sel, i_wr_head, wr_slot) + d]
                    <= i_wr_code_mode ? i_wr_fp16[d][7:0] : wr_code[d];
    end

    // ---- READ: registered byte fetch (SRAM 1-cyc), then combinational e4m3 -> fp16 (x 2^descale_k) ----
    logic [HEAD_SIZE-1:0][7:0] rd_code_r;
    logic signed [7:0]         rd_k_r;
    logic                      rd_valid_r;
    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin rd_valid_r <= 1'b0; rd_k_r <= '0; end
        else begin
            rd_valid_r <= i_rd_en;
            rd_k_r     <= i_rd_descale_k;
            if (i_rd_en)
                for (int d = 0; d < HEAD_SIZE; d++)
                    rd_code_r[d] <= kv_mem[kv_off(i_rd_pass, i_rd_sel, i_rd_head, rd_slot) + d];
        end
    end

    logic [HEAD_SIZE-1:0][15:0] rd_deq;
    genvar rd;
    generate
        for (rd = 0; rd < HEAD_SIZE; rd++) begin : g_rd_cast
            logic [15:0] rd_fp16_raw;
            fp8_to_fp16 u_rf (.i_fp8(rd_code_r[rd]), .o_fp16(rd_fp16_raw));
            fp16_scale2 u_rs (.i_x(rd_fp16_raw), .i_d(rd_k_r), .o_y(rd_deq[rd]));
        end
    endgenerate

    assign o_rd_fp16  = rd_deq;
    assign o_rd_valid = rd_valid_r;

endmodule
