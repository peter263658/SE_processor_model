# CE W4A8 zero-multiplier core — progress (2026-06-07)

Status report for algorithm-side consultant review. Companion docs in same dir:
`CE_DATAPATH_DRAFT.md` (datapath decisions), `CE_IMPL_NOTES.md` (implementation/handoff).
All sims: VCS (export VCS env first — see CLAUDE.md "Simulation Commands").

---

## ✅ Done + verified

### A. PE layer (Design B precision-scalable, fused)
| file | content | verify |
|---|---|---|
| `01_RTL/engine/CE/pe_pair.sv` | fused cell: two 8-bit slices, 16-bit modes chain carry into one 16-bit add; lo pack 16b / hi pack 8b; PEG-per-stream operands | — |
| `01_RTL/engine/CE/PEG.sv` | `PEG` (16×pe_pair + 32-way **FP16 saturating** tree → o_accum), `adder_tree_fp16`, `fp16_add_sat`, shared helpers `pe_fp_unpack`/`pe_int4_csd` | — |
| `00_TB/CE/pe_lane_ref.sv` | `pe_lane` + `pe_slice_addsub` (moved out of PEG.sv; TB-only golden) | — |
| `00_TB/CE/tb_PEG_w4a8.sv` (`./06_run_PEG_w4a8`) | DUT(pe_pair) vs PEG_ref(pe_lane×32), 4 modes random | **16000 vec / 4 modes / 0 err** |

### B. CE datapath spine
| file | content | verify |
|---|---|---|
| `01_RTL/engine/CE/ce_fp_util.sv` | `fp16_scale2` (pow2 scale = exp-add), `apot6a_decode`, `apot6b_decode`, `fp16_to_e4m3` (RNE; **BUG-1/2 fixed**) | — |
| `01_RTL/engine/CE/dealer.sv` | per-PEG APoT `2^a±2^b` delayed fused descale (4 instances) | — |
| `01_RTL/engine/CE/oacc.sv` | complex 2-bank: `re+=P0−P1`, `im+=P2+P3`, clr(load)/en(advance) | — |
| `01_RTL/engine/CE/ppu_fma.sv` | mode-select: CONV(BN α·x+β → PReLU pow2 → e4m3 cast) / PASS / SCALE | — |
| `01_RTL/engine/CE/CE_core.sv` | spine PEG×4→dealer×4→oacc→ppu_fma + **2 pipe regs** (PEG-out stage1, ppu-out stage3) | — |
| `00_TB/CE/tb_ce_core_smoke.sv` (`vcs -f ce_core_file.f`) | W4A8: PEG0=32×(1×1.0)=32.0 through full spine | **o_re=0x5000 (32.0) PASS** |
| `00_TB/CE/tb_e4m3_cast.sv` (`vcs -f e4m3_file.f`) | all 254 finite e4m3 codes: e4m3→fp16→cast == code | **254 / 0 err (雷④ cast closed)** |

### C. Locked datapath decisions
- FP16 + saturating add tree (clamp ±65504, not fp32).
  > ✅ **RESOLVED (2026-06-07):** per-path scan done. INT4/DW/out_proj safe (≤0.19×). FP8 outlier
  > (4.2×)/QK(33×)/ScoreV(5.7×) overflowed (both operands full-range; single product alone Inf).
  > FIX: per-PEG `i_preshift` (per-tensor coarse pow2) folded into pe_pair's 4 product exponents
  > (zero HW; =0 → bit-exact); dealer keeps the residual (full − preshift). Post-fix all ≤0.49×,
  > 0 overflow. See `project_fp16_tree_overflow` memo + CE_IMPL_NOTES §2. TB re-run (16000/0 +
  > preshift≠0 directed case) owed on a VCS machine.
- Complex = **PEG-per-stream**: PEG0=fr·xr, PEG1=fi·xi, PEG2=fr·xi, PEG3=fi·xr; OACC `re+=P0−P1`, `im+=P2+P3` (2 banks).
- Complex activation k = **two static slots** (k_real, k_imag) wired per-PEG (PEG0/3→k_real, PEG1/2→k_imag); "shared" layers export equal codes; **zero runtime select** (k fused into the dealer's APoT exponent per PEG).
- Unit counts: **4 dealer** (1/PEG, pre-OACC) · **OACC 1 unit = 2 re/im banks** · **PPU pool 4–8 shared** (no dedicated BN/PReLU/cast lanes).
- DW conv 5-tap accumulated in OACC (no dedicated pre-accumulator).
- Latency operand→o_re = PEG-reg(1) + oacc(1) + ppu-reg(1) = 3 (+ SRAM read 1).
- Accumulation NOT bit-exact vs fp32 golden (FP non-assoc + dtype) → **compare accum with tolerance**; per-product is bit-exact.

---

## 🟡 Consultant task-list status
| # | task | status |
|---|---|---|
| 1 | BUG-1 `te>=15→16` (e4m3 256–448 normals) | ✅ fixed + verified |
| 2 | BUG-2 subnormal `sh=1-te→8-te` | ✅ fixed + verified |
| 3 | e4m3 round-trip TB | ✅ 254/0 (`tb_e4m3_cast.sv`) |
| 5/11 | PEG-out + ppu-out pipe regs, retime | ✅ added, smoke re-verified |
| 6 | Mem_def `UFBUF_BANKS_2=4` | ✅ already correct |
| B1 | skipbuf/kvbuf were stale wbuf/ofbuf copies | ✅ **DONE (2026-06-07)** — both rewritten on `presto_dpram` (true 2-port = 2R/2W); `skipbuf_*`/`kvbuf_*` module names + SKIPBUF_*/KVBUF_* params + `ifdef` gate; tops = thin 2-port pass-throughs; old `*_ctrl` → deprecated stubs (sequencing → mem_controller). `presto_2pram` was wrong (1W1R, no 64-wide macro). |
| B2 | `presto_2pram` "undefined" | ⚪ **false alarm** — `01_RTL/common/sram/presto_2pram.sv` exists (consultant's RTL_review/ copy lacked it); kvbuf/skipbuf use it fine |
| B3 | `Mem_Ctrl_Top.sv` double-drive | ⚪ legacy stub; superseded by new `mem_controller` |
| 7 | builder asserts BN `overall_sign==1` | 🔲 to confirm in sram builder (ppu_fma has no BN overall-sign path; assumes +1) |
| 4/9 | golden per-layer numeric sim (dealer/oacc/ppu vs exported codes, tolerance on accum) | 🔲 **largest remaining verification** (needs real WBUF image + controller) |
| 8 | inst_decoder + mem_controller + engine top | 🔲 plan done; awaiting decision approval (below) |
| — | WBUF single-port `presto_spram` (currently `presto_dpram`) + wbuf_ctrl arbitration | 🔲 structural; do at integration |
| 10 | presto behavioral macros for 64×2048 (WBUF) + 64×1024 (KV) | 🔲 pre-synth (BEHAV sim OK now) |
| — | real SRAM macro swap | 🔲 pre-synth reminder |

presto behavioral signals: read latency = 1 cyc registered, active-high en/we wrap active-low CEN/WEN; W/W and W/R collision → X (matches macro). `presto_*` interfaces verified OK; `PRESTO_BUF_GATE_IN/OUT` default = pass-through (latency stays 1).

---

## 🔲 Engine integration — plan (4 decisions awaiting approval before writing)
New top (NOT overwriting legacy `engine.sv` = old Presto NPU: `MEM_def`/IFBUF/OFBUF/core_top):
```
ce_engine (new)
├─ wbuf_top     presto_SPRAM 16×64×2048   weights R0/R1 + const R2..R5b
├─ ufbuf_top    presto_DPRAM 8×64×128     act ping-pong (BANKS_2=4 per half)
├─ kvbuf_top    presto_dpram 4×64×1024    K/V e4m3 ring (true 2-port: 2R/2W)   ✅ rewritten
├─ skipbuf_top  presto_dpram 4×64×128     decoder skip (top-50%, groups_skip=2) ✅ rewritten
├─ Softmax_v3                              attention SFU
├─ ppu_const_cache  2KiB regfile          prefetched APoT/BN/PReLU/k
├─ inst_decoder   FSM: which layer/sub-op/phase
├─ mem_controller addr/we + operand-route
└─ CE_top → CE_core (+2 pipe regs)
```
- **inst_decoder**: 63-sub-op ROM `{pe_mode, ppu_mode, ic, oc, F, groups, w_base, act_src(UF/KV/SKIP), act_base, out_dst, const_ptr}` (offsets from `memory_map_pkg.svh`); FSM per sub-op `FETCH→PREFETCH(const→ppu cache)→RUN(oc-tile × ic_chunk [×5 DW tap]: addr + i_dealer_apot[0..3] + clr/en)→DRAIN(oacc settle→pulse ppu→write)→NEXT(tile / layer-end UFBUF swap)`.
- **mem_controller**: counters→addr/we, ping-pong half, PEG-per-stream operand mux (fr→x[0,2], fi→x[1,3], xr→y[0,3], xi→y[1,2]). WBUF single-port → interleave const-prefetch in spare bank-cycles (1024 vs 512 b/cyc).
- **handshake CE_top↔inst_decoder**: down valid/clr/en/ppu_mode/ppu_go; up ready/done; account for SRAM 1cy + 2 pipe regs.

**Decisions to confirm:**
1. WBUF single-port `presto_spram` + wbuf_ctrl arbitration (PE-read vs const-prefetch).
2. UFBUF unified ping-pong replaces legacy IFBUF+OFBUF (W4A8 topology).
3. New top file `ce_engine.sv`; legacy `engine.sv` untouched.
4. ✅ DONE — skipbuf/kvbuf rewritten on `presto_dpram` (2R/2W; presto_2pram is 1W1R + has no 64-wide macro). Passive 2-port wrappers; role sequencing (skip gather / KV ring) deferred to mem_controller.

---

## Open / next
1. Engine integration (inst_decoder + mem_controller + ce_top + ce_engine + behavioral buffers) — after decision approval.
2. Golden per-layer numeric verification (the big one) — with real WBUF image + tolerance on accum.
3. WBUF single-port conversion; skipbuf/kvbuf adaptation; BN overall_sign assert; presto macros + real SRAM macro swap (pre-synth).
