# CE W4A8 core — HANDOFF for next implementer (2026-06-07)

You are continuing a SystemVerilog RTL build of the BCCTN W4A8 zero-multiplier
Compute Engine. The PE layer + CE datapath spine are DONE & verified; your job is
the buffer adaptation + the control/integration. Read this top-to-bottom, then the
3 companion docs in this dir.

## 0. Read order
1. This file.
2. `CE_DATAPATH_DRAFT.md` — datapath decisions (authoritative, has algo-consultant review).
3. `CE_IMPL_NOTES.md` — implementation/handoff of the spine.
4. `CE_PROGRESS.md` — status table.
5. Repo context: `/home/r12k41024/Research/CLAUDE.md` (build/sim env, project map).
6. SRAM map: `10_algo/docs/SRAM_RTL_MEMORY_HANDOFF_EN.md` + `10_algo/hw_patterns_final/sram_images/memory_map_pkg.svh` (tensor/const offsets — DO NOT hand-transcribe).
7. Op→hw mapping: `HANDOFF_HW_ZERO_MULTIPLIER_CORE.md`; pitfalls: `10_algo/docs/RTL_HANDOFF_PE_LANE_DATAPATH_2026-06-06.md` (雷①–⑤).

## 1. DONE + VERIFIED (don't rewrite)
PE layer (`01_RTL/engine/CE/`): `PEG.sv` (PEG = 16×pe_pair + 32-way FP16 saturating
tree `adder_tree_fp16`+`fp16_add_sat`; helpers `pe_fp_unpack`/`pe_int4_csd`),
`pe_pair.sv` (fused 2×8b precision-scalable). Golden ref `00_TB/CE/pe_lane_ref.sv`.
→ `cd 00_TB/CE && ./06_run_PEG_w4a8` = **16000 vec / 4 modes / 0 err**.

CE spine: `ce_fp_util.sv` (fp16_scale2, apot6a/6b_decode, fp16_to_e4m3 — e4m3 BUG-1/2
fixed), `dealer.sv`, `oacc.sv`, `ppu_fma.sv`, `CE_core.sv` (+2 pipe regs).
→ `vcs -f 00_TB/CE/ce_core_file.f ...` smoke o_re=0x5000 PASS;
  `vcs -f 00_TB/CE/e4m3_file.f ...` e4m3 round-trip 254/0.
(Build env: export VCS vars per CLAUDE.md; DW needs `-y .../dw/sim_ver +incdir+...`.)

## 2. LOCKED decisions
Datapath: FP16+saturating tree (not fp32) · complex = PEG-per-stream · OACC 2 banks
re/im · complex k = 2 static slots (no runtime select) · 4 dealer / OACC 2-bank /
PPU pool 4–8 (NO dedicated BN/PReLU/cast lanes) · DW 5-tap accumulate in OACC ·
accum compared with TOLERANCE (per-product bit-exact, tree fp16 rounding diverges
from fp32 ref).
Integration (user-approved): (1) WBUF single-port `presto_spram` + wbuf_ctrl
arbitrates PE-read vs const-prefetch; (2) UFBUF unified ping-pong REPLACES legacy
IFBUF+OFBUF; (3) OVERWRITE legacy `engine.sv`; (4) ✅ DONE — skipbuf/kvbuf rewritten on
`presto_dpram` (true dual-port = 2R/2W; NOT `presto_2pram` which is 1W1R) — see §5.

## 3. CE_core interface you must drive (the contract)
```
module CE_core (
  i_clk, i_rst_n,
  input CE_LANE_mode i_pe_mode,                       // MODE_W4A8/LMUL_FP8/LMUL_ATTN/W4A16 (Param_def)
  input [NUM_PEG-1:0][NUM_PE_PG-1:0][15:0] i_x, i_y,  // NUM_PEG=4, NUM_PE_PG=32; per-PEG operands
  input [NUM_PEG-1:0][7:0] i_preshift,                // per-PEG per-tensor pre-tree pow2 (overflow fix; 0=bit-exact)
  input [NUM_PEG-1:0][11:0] i_dealer_apot,            // per-PEG fused APoT-6a descale code (RESIDUAL = full − preshift)
  input i_oacc_en, i_oacc_clr,                        // clr=load first chunk/tap, en=advance
  input [1:0] i_ppu_mode,                             // 0=CONV 1=PASS 2=SCALE (ppu_fma localparams)
  input [11:0] i_bn_alpha_re/_im,                     // BN APoT-6a (per out-ch)
  input [15:0] i_bn_beta_re/_im,                      // BN fused bias fp16
  input [11:0] i_prelu_re/_im,                        // PReLU pow2 6b
  input signed [7:0] i_cast_k,                        // next-layer block-float k for e4m3
  input [11:0] i_scale_re/_im,                        // SCALE-mode APoT (QK ×2^-2·2^-k)
  output [15:0] o_re, o_im );                         // CONV: e4m3 in [7:0]; PASS/SCALE: fp16
```
Sub-block contracts: `dealer(i_sum, i_apot[11:0]) → o_descaled` = sum·(2^a±2^b).
`oacc(i_clk,i_rst_n,i_en,i_clr,i_p0..i_p3) → o_re,o_im` with re+=p0−p1, im+=p2+p3.
`ppu_fma(...)` modes CONV/PASS/SCALE. `PEG(i_mode,i_x[32][16],i_y[32][16])→o_accum`.

## 4. Datapath behavior + timing (controller MUST honor)
- **PEG-per-stream routing** (mem_controller drives, STATIC per PEG):
  weights: `i_x[0]=i_x[2]=fr`, `i_x[1]=i_x[3]=fi`;
  acts:    `i_y[0]=i_y[3]=xr`, `i_y[1]=i_y[2]=xi`.
  Within a PEG, the 32 lanes = the 32 input-channels of one ic-chunk; W4A8 weight in
  operand[3:0] (INT4), fp8 act in operand[7:0]; FP16 modes use full [15:0].
- **dealer codes** per PEG (fused descale = weight APoT − act 2^-k):
  `i_dealer_apot[0]=fused(w_r,k_real)`, `[1]=fused(w_i,k_imag)`,
  `[2]=fused(w_r,k_imag)`, `[3]=fused(w_i,k_real)`. From WBUF const region per layer.
- **OACC accumulate**: for each output channel, loop ic_chunk = 0..⌈ic/32⌉−1 (DW: ×5
  taps). i_oacc_clr=1 on the FIRST chunk/tap, en=1 every chunk. After last chunk the
  re/im banks hold the result.
- **PIPELINE LATENCY** (2 new regs): operands at cycle T → `peg_accum_r` at T+1 →
  dealer comb → assert `i_oacc_clr/en` at **T+1** → OACC reg at T+2 → ppu comb →
  `o_re/o_im` reg at **T+3**. Add SRAM read (addr@T → data@T+1) in front. The
  inst_decoder counters + valid/done must shift to match.
- **PPU modes**: conv layers = CONV (BN→PReLU→e4m3, out e4m3[7:0]); QK = SCALE
  (×2^-2·2^-k via i_scale, fp16 out) → Softmax_v3; Score×V/out_proj feed = PASS (fp16).
- **e4m3 cast** uses NEXT layer's k; rounding RNE (verified self-consistent; final
  bit-exact vs `fp8_qat` is the golden TODO §9).

## 5. NEXT — Step 1: BUFFERS (foundation; elaborate each before Step 2)
Current state (verified by inspection):
- `ufbuf/ufbuf_top.sv`+`ufbuf_sram.sv` = CORRECT unified ping-pong (presto_dpram).
  Use as the TEMPLATE. (FSM S_IDLE/LOAD/STORE; bufstate BUF0/BUF1; swap_valid.)
- `wbuf/` = real wbuf but uses `presto_dpram` (DUAL-port). **TASK**: convert to
  `presto_spram` (SINGLE-port) + make `wbuf_ctrl` arbitrate PE-weight-read vs
  const-prefetch on the one port (headroom: 1024 b/cyc supply vs 512 b/cyc PE demand
  → interleave prefetch in spare bank-cycles; never same-cycle conflict).
- `skipbuf/` = ✅ **DONE (2026-06-07)**: rewritten on `presto_dpram` (true dual-port).
  `skipbuf_sram.sv` (`module skipbuf_sram`, `SKIPBUF_*` 4×64×128, presto_dpram banks,
  `ifdef` gate like ufbuf) + `skipbuf_top.sv` (thin 2-port pass-through, ports A/B each
  R/W). `skipbuf_ctrl.sv` → deprecated stub (old wbuf-load FSM removed; gather sequencing
  → mem_controller). NOTE: handoff first said `presto_2pram`, but that macro is **1W1R**
  and has **no 64-wide config**; `presto_dpram` (`rf_dp_128x64`) is correct for 2R/2W +
  this size. Static top-50% gather (groups_skip=2) addressing is mem_controller's job.
- `kvbuf/` = ✅ **DONE (2026-06-07)**: `kvbuf_top.sv` no longer instantiates the
  undefined `ofbuf_sram`/`ofbuf_ctrl` (those used `OFBUF*_` params defined NOWHERE — worse
  than "legacy", a from-scratch rewrite). Now `kvbuf_sram.sv` (`module kvbuf_sram`,
  `KVBUF_*` 4×64×1024, `presto_dpram`=`rf_dp_1024x64`) + `kvbuf_top.sv` (thin 2-port
  pass-through). `kvbuf_ctrl.sv` → deprecated stub. K/V e4m3 ring addressing (in_proj
  write; QK all-32-frames, Score×V top-4) → mem_controller (Step 2).
- `common/sram/`: `presto_spram.sv`(SP), `presto_dpram.sv`(true dual RW = 2R/2W),
  `presto_2pram.sv`(**1W1R only**) all EXIST & behavioral-correct (read latency=1 registered;
  active-high en/we wrap active-low CEN/WEN; W/W,W/R collision→X). `PRESTO_BUF_GATE_IN/OUT`
  undefined = pass-through — use `` `ifdef `` (NOT `generate if`, the macro is not a
  Param_def constant; the old kvbuf/skipbuf `generate if` would not elaborate).
  `presto_dpram` real macros ALREADY include `rf_dp_1024x64` (KV) + `rf_dp_128x64` (SKIP);
  only WBUF 64×2048 (`presto_spram`, single-port) needs a macro added before synth
  (BEHAV sim OK now). `presto_2pram` has only 256-wide configs → unusable for 64-wide KV/SKIP.
- `Mem_def.sv`: WBUF 16×64×2048, KVBUF 4×64×1024, UFBUF 8×64×128 (UFBUF_BANKS_2=4 ✓),
  SKIPBUF 4×64×128. `Param_def.sv`: NUM_PEG=4, NUM_PE_PG=32, CE_LANE_mode enum.

## 6. NEXT — Step 2: CONTROL + integration
Build (new modules + OVERWRITE engine.sv):
- `ppu_const_cache` — 2KiB regfile; prefetch a sub-op's APoT/BN/PReLU/k from WBUF
  const region; feeds dealer/ppu codes. (PPU never reads WBUF during MAC.)
- `inst_decoder` (the only "which layer/sub-op/phase" block):
  63-sub-op ROM descriptor `{pe_mode, ppu_mode, ic, oc, F, groups, w_base, act_src
  (UF/KV/SKIP), act_base, out_dst, const_ptr}` (offsets from `memory_map_pkg.svh`;
  walk order = HANDOFF_HW §8/§9 execution order: enc0..5{dw,pw} → MHA sub-ops →
  dec0..5{pw_x,pw_skip,dwT}). FSM/sub-op: FETCH→PREFETCH(const→ppu cache)→RUN(oc-tile
  × ic_chunk [×5 DW taps]: emit addrs + i_dealer_apot[0..3] + clr/en)→DRAIN(oacc
  settle→pulse ppu→write out)→NEXT(tile / layer-end UFBUF swap).
- `mem_controller`: from inst_decoder counters → WBUF/UF/KV/SKIP addr+we, ping-pong
  half, and the PEG-per-stream operand mux (§4). WBUF single-port arbitration here.
- `ce_top`: wraps CE_core + valid/ready/done handshake to inst_decoder; accounts for
  SRAM 1cy + the 2 pipe regs.
- `engine.sv` (OVERWRITE): instantiate wbuf_top(spram)/ufbuf_top/kvbuf_top/skipbuf_top/
  Softmax_v3(`01_RTL/engine/Softmax/Softmax_v3.sv`)/ppu_const_cache/inst_decoder/
  mem_controller/ce_top. Keep the legacy `tc_*` load/store DMA handshake ports if a
  host shell needs them; drop core_top/ifbuf_top/ofbuf_top; fix `import MEM_def` →
  `import Mem_def`. (Legacy `Mem_Ctrl_Top.sv` is a double-drive stub — ignore/delete.)
MHA specifics: QK uses CE_core SCALE mode → Softmax_v3 (I/O: i_SM_data[32][16]+mask,
o_SM_data[32][16] dense ≤4 nz) → Score×V via CE_core (m8, PASS) → out_proj (W4A16).
Softmax packet→m8 operand glue is still `[?]` — see CE_DATAPATH_DRAFT §9.

## 7. Gotchas / contracts
- 雷②: dealer is per-group, BETWEEN tree & OACC (already so in CE_core). Don't move.
- 雷③: PPU mode-select; attention does NO BN/PReLU/cast.
- 雷④: e4m3 cast RNE — verify bit-exact vs `fp8_qat` with golden (cast self-test passes).
- 雷⑤: kreal/kimag = static per-PEG dealer code; never a runtime branch.
- ppu_fma assumes BN `overall_sign=+1` (no overall-sign path). Confirm the sram builder
  asserts overall_sign==1, else a γ<0 retrain silently flips BN sign.
- Tree accum: compare with TOLERANCE, not bit-exact (fp16 rounding).
- Real SRAM macro swap is a pre-synth TODO (behavioral now).

## 8. Test commands (from 00_TB/CE/, after export VCS env)
- PE: `./06_run_PEG_w4a8`  → "[TB] ALL PASS, checked 16000".
- spine smoke: `vcs -f ce_core_file.f -full64 -R +v2k +notimingcheck -sverilog -timescale=1ns/10ps -debug_access+all -y /usr/cad/synopsys/synthesis/cur/dw/sim_ver +incdir+/usr/cad/synopsys/synthesis/cur/dw/sim_ver +libext+.v`
- e4m3: `vcs -f e4m3_file.f -full64 -R +v2k -sverilog -timescale=1ns/10ps -debug_access+all`

## 9. Open verification (do with real patterns)
Golden per-layer numeric sim: drive dealer/oacc/ppu with EXPORTED codes from
`10_algo/hw_patterns_final/` (weights/scales/BN/PReLU/act-k) for one real layer
(e.g. encoder.model.0 pw), compare to `debug_reference/layer_io_fp32` with tolerance
on the accumulation; per-product bit-exact. This is the largest remaining risk —
it closes 雷④ end-to-end and validates the dealer/ppu code decode against real data.

## 10. Suggested order
Step 1a WBUF→spram+arb · ✅1b skipbuf rewrite (DONE, presto_dpram) · ✅1c kvbuf rewrite (DONE, presto_dpram).
Step 2a ppu_const_cache · 2b inst_decoder · 2c mem_controller · 2d ce_top · 2e
engine.sv + single-layer integration smoke. Then §9 golden.
