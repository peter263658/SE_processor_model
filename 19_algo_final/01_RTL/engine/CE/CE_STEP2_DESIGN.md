# CE W4A8 — Step 2 control/integration DETAILED DESIGN (2026-06-07)

How to write `ppu_const_cache → inst_decoder → mem_controller → ce_top → ce_engine.sv`.
This is the contract: each module below has a fixed port list + behavior so they wire up
without rework. Build + elaborate-audit each in order. Companion: `CE_HANDOFF_NEXT.md` (§3
CE_core contract, §4 timing), `CE_IMPL_NOTES.md`, `CE_PROGRESS.md`.
Authoritative offsets: `10_algo/hw_patterns_final/sram_images/memory_map_pkg.svh` (include it —
DO NOT hand-transcribe). Params: `Param_def` (NUM_PEG=4, NUM_PE_PG=32, CE_LANE_mode), `Mem_def`.

## 0. Geometry recap (so the loops are right)
- **W4A8 lane count = 128 = NUM_PEG(4) × NUM_PE_PG(32).** CE_core processes, per cycle, ONE
  ic-chunk (32 input channels) of ONE output channel, for all 4 complex streams in parallel:
  PEG0=fr·xr, PEG1=fi·xi, PEG2=fr·xi, PEG3=fi·xr.
- To cover ic: loop `ic_chunk = 0..⌈ic/32⌉−1`, accumulate in OACC. DW conv: ×5 taps in OACC.
- To cover oc: loop output-channel tiles.
- WBUF supplies 1024 b/cyc (16 banks × 64b); PE demand for one ic-chunk×4 streams = 4×32×(4b w +
  8b act). Weights INT4 are 4b; one 64b WBUF word = 16 INT4 = half an ic-chunk → 2 WBUF words/cyc
  for weights of one stream. Const-prefetch fills the spare bank-cycles (never same-cycle as PE read).

## 1. Const region (from memory_map_pkg.svh) — what ppu_const_cache holds
WBUF const region base byte = `WBUF_CONST_BASE` = 241664. Three sub-regions, all 12-bit-packed
except BN bias (fp16) and activation-k (u8):
| sub-region | byte range | content | decode (`DEC`) |
|---|---|---|---|
| weight_apot_scale | 241664 .. ~247215 | per-tensor APoT weight-scale codes (per-group(32)/head/chan) | 0 = apot_6a |
| bn_scale / bn_bias / prelu | ~247216 .. 251623 | BN APoT(6a) + BN bias fp16 + PReLU pow2(6b) | 0 / 2=fp16 / 1=apot_6b |
| activation_k | 251624 .. 251676 | block-float k table, `u8 = k + 32` | 3 = u8_plus32 |

Each const block is `{OFF, BYTES, DEC}` in `memory_map_pkg.svh` (66 weight + 103 const blocks).
**12-bit packing is contiguous**: code `i` lives at bit `i*12` (a block of N codes = `ceil(N*12/8)`
bytes, e.g. 8 codes = 12 bytes), so codes cross 64-bit WBUF word boundaries → extraction needs a
2-word window. Decode enums:
- **6a** (weight/BN scale): `[11:7]=a_exp+24 | [6:2]=b_exp+24 | [1]=has_second | [0]=residual_sign`
  → `2^a (± 2^b if has_second)`. This is exactly `ce_fp_util::apot6a_decode`.
- **6b** (PReLU): `[11:7]=k_exp+24 | [6:2]=0 | [1]=0 | [0]=overall_sign` → `±2^k`. `apot6b_decode`.
  PReLU is per-module scalar (num_parameters=1) → broadcast, NOT per-channel.
- **fp16** BN bias: raw 16-bit.
- **u8+32** k: signed k = u8 − 32.

## 2. `ppu_const_cache.sv` (Step 2-1 — written this round)
Ping-pong register file. mem_controller fills it from WBUF const words during a sub-op's PREFETCH
phase; dealer/ppu read codes during RUN/DRAIN. PE never reads WBUF const during MAC.
```
module ppu_const_cache #(parameter WORDS_PER_HALF = 128)   // 128×64b = 1KiB/half, 2KiB total
( i_clk, i_rst_n,
  // FILL (from mem_controller; one 64b WBUF const word per cycle)
  input i_fill_en, input i_fill_half, input [$clog2(WORDS_PER_HALF)-1:0] i_fill_waddr,
  input [63:0] i_fill_data,
  // READ window (registered, latency 1): returns word[waddr] and word[waddr+1] as 128b
  input i_rd_half, input [$clog2(WORDS_PER_HALF)-1:0] i_rd_waddr,
  output [127:0] o_rd_window );     // {mem[waddr+1], mem[waddr]}
```
Companion combinational extractor (house style = module, not function):
```
module const_field_extract ( input [127:0] i_window, input [6:0] i_bit_off,  // bit within window
  input i_is16, output [15:0] o_field );  // 12-bit (apot/prelu) or 16-bit (fp16/k) field
  assign o_field = i_is16 ? (i_window >> i_bit_off)[15:0]
                          : {4'b0, (i_window >> i_bit_off)[11:0]};
```
Read protocol: code index `c` of a block at byte `base` → bit `B = (base − WBUF_CONST_BASE)*8 +
c*W` (W=12 or 16); `waddr = B>>6` (within the half), `bit_off = B & 63`. mem_controller issues
`i_rd_waddr=waddr`; next cycle `const_field_extract(o_rd_window, bit_off, is16)` gives the code;
then `apot6a_decode` / `apot6b_decode` / fp16 / (u8−32). Ping-pong: fill `!active_half` while RUN
reads `active_half`; swap at sub-op boundary.

## 3. `inst_decoder.sv` (Step 2-2) — the only "which layer/sub-op/phase" block
**GRANULARITY = per-COMPLEX-sub-op (decision A, 2026-06-08):** one CE_core 4-PEG pass IS the
atomic complex MAC, so each descriptor PAIRS the _r/_i weight tensors (both `w_*_base` explicit —
stride varies) and carries `k_real/k_imag` + `preshift_r/preshift_i`. PEG0/2 ← real, PEG1/3 ← imag.
ROM = **36 entries** (30 conv pairs + 6 attention). The PEG-per-stream mux in mem_controller is a
STATIC route, NOT a runtime pairing FSM. Authoritative struct: `ce_subop_pkg.sv::subop_t`.
**Descriptor ROM** — one entry per COMPLEX sub-op, packed struct (older per-tensor sketch below
superseded by the paired struct):
```
typedef struct packed {
  CE_LANE_mode pe_mode;     // MODE_W4A8 / LMUL_FP8 / LMUL_ATTN / W4A16
  logic [1:0]  ppu_mode;    // 0 CONV / 1 PASS / 2 SCALE
  logic [9:0]  ic, oc;      // contraction / output dims
  logic [4:0]  F;           // taps (DW=kernel, PW=1)
  logic [2:0]  groups;      // grouping (decoder pw groups_x=4 / skip=2)
  logic [17:0] w_base;      // WBUF weight byte offset  (from memory_map_pkg W_*_OFF)
  logic [1:0]  act_src;     // 0 UF / 1 KV / 2 SKIP
  logic [13:0] act_base;    // act buffer base
  logic [1:0]  out_dst;     // 0 UF / 1 KV / 2 SKIP
  logic [17:0] const_ptr;   // WBUF const byte offset for this sub-op's block group
  logic signed [5:0] preshift; // operand pow2 pre-shift (overflow fix hook; 0 = none)
} subop_t;
```
Populate the ROM from `memory_map_pkg.svh` localparams (W_*_OFF / C_*_OFF). **Walk order =
HANDOFF_HW §8/§9 execution order**: `enc0..5 {dw,pw}` → MHA sub-ops (in_proj → Q/K/V proj store
KV → QK SCALE → Softmax_v3 → Score×V PASS → out_proj W4A16) → `dec0..5 {pw_x, pw_skip, dwT}`.
The 12 decoder-pointwise outlier layers (dec stage 0/3/4 ×{x,skip}×{r,i}) get `pe_mode=LMUL_FP8`;
all other pw get `MODE_W4A8`; QK/Score×V get `LMUL_ATTN`; MHA out_proj `W4A16`.

**Per-sub-op FSM** (drives mem_controller + CE_core ctrl):
```
FETCH    : latch descriptor[subop_idx]
PREFETCH : stream this sub-op's const block(s) WBUF→ppu_const_cache (fill !active_half);
           assert i_fill_* for BYTES/8 cycles; on done, flip active_half
RUN      : for oc_tile: for ic_chunk 0..⌈ic/32⌉−1 (DW: ×F taps):
             emit WBUF weight addr + act addr (mem_controller), i_dealer_apot[0..3],
             i_oacc_clr=(first chunk/tap), i_oacc_en=1
DRAIN    : OACC settle (pipeline 2 regs) → pulse ppu (i_ppu_mode + BN/PReLU/cast/scale codes)
           → write o_re/o_im to out_dst (mem_controller)
NEXT     : oc_tile++ ; at layer end, UFBUF swap; subop_idx++ (or attention branch to Softmax_v3)
```
Counters: `subop_idx`, `oc_tile`, `ic_chunk`, `tap`. All addresses derive from
descriptor.base + counter×stride (strides from ic/oc/F/groups). Emit `valid`/`done` up to ce_top.

## 4. `mem_controller.sv` (Step 2-3) — addr/we + operand route + WBUF arb
From inst_decoder counters:
- **WBUF**: weight read addr = `w_base + (oc_tile,ic_chunk,tap)·stride`; bank-split
  `row=A>>7, bank=(A>>3)&0xF, byte=A&7` (64b word, little-endian, no byte-swap). Single-port:
  **arbitrate** PE weight read (priority) vs const-prefetch (fills idle bank-cycles only; never
  same-cycle — guaranteed by 1024 supply vs 512 demand headroom).
- **UF/KV/SKIP**: read act per `act_src`/`act_base`+counter; write result per `out_dst`. These are
  the new presto_dpram 2-port buffers (port A write / port B read), driven directly here.
- **PEG-per-stream operand mux** (STATIC routing, §4 handoff): weights `i_x[0]=i_x[2]=fr`,
  `i_x[1]=i_x[3]=fi`; acts `i_y[0]=i_y[3]=xr`, `i_y[1]=i_y[2]=xi`. The 32 lanes of a PEG = the
  32 in-channels of the ic-chunk.
- **dealer codes**: `i_dealer_apot[0]=fused(w_r,k_real)`, `[1]=fused(w_i,k_imag)`,
  `[2]=fused(w_r,k_imag)`, `[3]=fused(w_i,k_real)` — read from ppu_const_cache via
  `const_field_extract` + apot6a_decode; kreal/kimag are STATIC per-PEG (雷⑤), never a runtime branch.
- **OVERFLOW pre-shift** (`descriptor.preshift`, ✅ datapath DONE 2026-06-07): drive
  `CE_core.i_preshift[g]` (signed [7:0]) per PEG with the per-tensor coarse pow2. pe_pair now folds
  it into all 4 product exponents (zero HW; preshift=0 → bit-exact). mem_controller must: set
  preshift per sub-op (INT4/DW/out_proj = 0; FP8 ~2^-3/-4; QK 2^-10; ScoreV 2^-13) AND program the
  dealer code as the RESIDUAL (full − preshift). Per-group INT4 stays preshift=0 (雷②). See
  `project_fp16_tree_overflow` memo / CE_IMPL_NOTES §2.

## 5. `ce_top.sv` (Step 2-4) — CE_core + handshake + latency
Wraps `CE_core` (the verified spine). Adds the valid/ready/done handshake to inst_decoder and
accounts for **SRAM read 1cy + 2 CE_core pipe regs**: operand addr@T → SRAM data@T+1 → PEG reg@T+2
→ dealer(comb) → OACC@T+3 → ppu(comb) → o_re/o_im reg@T+4. So `done` for an output channel lags the
last `i_oacc_en` by the fixed pipe depth; shift the write-strobe to mem_controller accordingly.
Ports: down = {i_pe_mode, i_x/i_y already muxed by mem_controller OR mem_controller feeds ce_top
which feeds CE_core, pick one — recommend mem_controller→ce_top→CE_core so ce_top owns latency},
i_dealer_apot, i_oacc_clr/en, i_ppu_mode + all ppu const inputs, i_cast_k, i_scale_*; up = o_re,
o_im, o_valid (pipe-aligned).

## 6. `ce_engine.sv` (Step 2-5) — new top (do NOT overwrite legacy engine.sv yet)
Instantiate: `wbuf_top`(presto_dpram now; spram is a later Step 1a) · `ufbuf_top` · `kvbuf_top` ·
`skipbuf_top` · `Softmax_v3` (`01_RTL/engine/Softmax/Softmax_v3.sv`) · `ppu_const_cache` ·
`inst_decoder` · `mem_controller` · `ce_top`. Wire per §3–§5. Keep legacy `tc_*` DMA handshake
ports if a host shell needs them. `import Param_def + Mem_def` (correct case). MHA: QK → CE_core
SCALE → Softmax_v3 (i_SM_data[32][16]+mask → o_SM_data dense ≤4 nz) → Score×V CE_core PASS →
out_proj W4A16. Softmax-packet→m8 operand glue still `[?]` (CE_DATAPATH_DRAFT §9) — stub + flag.

## 7. Build + verify order (no VCS license → elaborate-audit each)
1. ✅ ppu_const_cache (+ const_field_extract) — audited.
2. ✅ inst_decoder (+ ce_subop_pkg + generated ce_subop_rom.svh via gen_ce_subop_rom.py) —
   audited 3-way (elaborates, 0 blockers). 68 sub-ops (60 conv generated + 8 attention
   hand-authored PROVISIONAL). preshift=0 everywhere (matches full-descale images).
   OPEN for mem_controller: complex _r/_i → 4-PEG fusion, within-block dw→pw→BN chaining,
   attention QK/Softmax/Score×V expansion + dims.
3. mem_controller — addr math + PEG route; check bank-split + dealer code indexing.
4. ce_top — latency alignment.
5. ce_engine — single-layer smoke (drive enc0.pw with real WBUF image + golden `layer_io_fp32`,
   tolerance on accum). This is the §9 golden close-out for the control path.
Each step: run the elaborate-audit workflow pattern used for the buffers before moving on.
```
