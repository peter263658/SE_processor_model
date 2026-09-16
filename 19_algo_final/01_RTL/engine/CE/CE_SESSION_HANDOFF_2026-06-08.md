# CE W4A8 — session handoff for review (2026-06-08)

File-by-file guide to everything changed this session, so you can review by opening each file.
Three workstreams: **A.** FP16-tree overflow fix (touched VERIFIED PE RTL), **B.** buffer rewrites
(presto_dpram), **C.** Step-2 control/integration (new). Companion specs: `CE_STEP2_DESIGN.md`,
`CE_IMPL_NOTES.md`, `CE_PROGRESS.md`, `CE_HANDOFF_NEXT.md`.

## 0. Verification status at a glance
| What | How verified | Result |
|---|---|---|
| PE preshift fix (preshift=0 bit-exact) | VCS `06_run_PEG_w4a8` 16000/0 + `07_run_peg_preshift` (preshift≠0 directed) | ✅ real VCS |
| e4m3 cast / CE_core spine | VCS `07_run_e4m3` 254/0 · `08_run_ce_core` o_re=0x5000 | ✅ real VCS |
| inst_decoder sequencer | VCS `09_run_inst_decoder` walk all sub-ops, descriptor==ROM, DONE, 0 err | ✅ real VCS |
| mem_controller + ce_bank_split | VCS `mem_controller_file.f` elaborate+run clean | ✅ real VCS |
| ce_subop_pkg/rom, ppu_const_cache, ce_dealer_sel, buffers | adversarial multi-agent elaborate-audit (no VCS on my box) | 🟡 audited, VCS pending |
| Per-layer numeric (dealer/ppu codes vs golden) | — | 🔲 the big TODO (task 4/9) |

> ⚠️ The struct grew (`branches_collapse`) and `ce_dealer_sel` was added AFTER your last VCS run —
> re-run `09_run_inst_decoder` + `mem_controller_file.f` to re-confirm 🟡 rows.

---

## A. FP16-tree overflow fix (per-PEG `i_preshift`) — touched VERIFIED RTL
Root cause + data: `project_fp16_tree_overflow` memo / `CE_IMPL_NOTES §2`. Per-PEG signed pow2
folded into the pe_pair product exponent (zero new arithmetic unit). `preshift=0` ⇒ bit-exact to old.

| file | change | review focus |
|---|---|---|
| `pe_pair.sv` | +`input signed [7:0] i_preshift`; `preshift_ext` added to all 4 product exponents (`int4_expf_l/lmul_expf_l/int4_expf_h/lmul_expf_h`) BEFORE the ≤0/≥31 clamp | preshift=0 ⇒ +0 ⇒ no change; all 4 paths covered; signed add |
| `PEG.sv` | +`i_preshift` port, broadcast to all 16 pe_pair | wiring |
| `CE_core.sv` | +`input [NUM_PEG-1:0][7:0] i_preshift`, `$signed(i_preshift[g])` per PEG | wiring |
| `dealer.sv` | doc only — note: code carries RESIDUAL when preshift≠0 (logic unchanged) | no logic change |
| `tb_PEG_w4a8.sv`, `tb_ce_core_smoke.sv` | drive `i_preshift=0` (keep bit-exact) | — |
| `tb_peg_preshift.sv`, `07_run_peg_preshift` | **(consultant-authored)** directed preshift≠0 invariant test | scale-equivariance |

**SAFETY INVARIANT** (review the whole fix against this): `pe_pair.i_preshift` and the dealer code
flip together — `preshift=0 ⇔ full descale`, `preshift≠0 ⇔ residual = full−preshift`. Currently 0
everywhere ⇒ self-consistent. Enabling FP8/QK/ScoreV needs decision (a) below.

---

## B. Buffer rewrites — presto_dpram (true dual-port, 2R/2W)
Reason: `presto_2pram` is 1W1R and has NO 64-wide macro; `presto_dpram` has `rf_dp_1024x64` (KV) +
`rf_dp_128x64` (SKIP). Modeled on the verified `ufbuf_sram`.

| file | change | review focus |
|---|---|---|
| `engine/skipbuf/skipbuf_sram.sv` | rewritten: `presto_dpram`, `SKIPBUF_*` 4×64×128, `ifdef` gate | was half-renamed stale wbuf copy |
| `engine/skipbuf/skipbuf_top.sv` | rewritten: thin 2-port pass-through (A=store, B=gather) | no FSM (mem_controller drives) |
| `engine/skipbuf/skipbuf_ctrl.sv` | → deprecation stub (old wbuf-load FSM removed; git has it) | dead file, comment-only |
| `engine/kvbuf/kvbuf_sram.sv` | rewritten: `presto_dpram`, `KVBUF_*` 4×64×1024 | — |
| `engine/kvbuf/kvbuf_top.sv` | rewritten: dropped undefined `ofbuf_sram/ofbuf_ctrl`+`OFBUF*_` params | was a stale ofbuf copy |
| `engine/kvbuf/kvbuf_ctrl.sv` | → deprecation stub | dead file |

Role sequencing (skip top-50% gather, KV ring) is intentionally NOT here — it's mem_controller's job.

---

## C. Step-2 control/integration (all NEW)
Datapath spine (`CE_core` etc.) was already verified; this is the controller that drives it.
Full spec: `CE_STEP2_DESIGN.md`. Build order = the order below.

### C1. `ppu_const_cache.sv` (+ `const_field_extract`)
2 KiB ping-pong const RF. Fill one 64-bit WBUF const word/cycle (PREFETCH); registered 128-bit
window read; `const_field_extract` slices a 12-bit (apot) or 16-bit (fp16/k) field, handling the
12-bit cross-64b-word packing. Review: the 128-bit window = `{mem[w+1],mem[w]}` for cross-boundary.

### C2. descriptor ROM (generated)
| file | role | review focus |
|---|---|---|
| `10_algo/scripts/gen_ce_subop_rom.py` | generator: manifest.json → ROM; offsets = literals from memory_map_pkg.svh; `branches_collapse` from `fused_descale_summary.json` | the field mapping + pairing |
| `ce_subop_rom.svh` (generated) | **36 sub-ops = 30 conv pairs + 6 attention**, 20 fields each | spot-check a few vs memory_map |
| `ce_subop_pkg.sv` | `subop_t` (20 fields) + `PPU_*` + ROM include (package so the type works in ports) | **DECISION A**: each entry PAIRS _r/_i (both `w_*_base` explicit, stride varies); `k_real/k_imag`, `preshift_r/i`, `branches_collapse` |

`branches_collapse`: =1 shared-k (encoder all, decoder dw) → 2 dealer codes; =0 split (decoder
pw_x/skip all) → kreal‖kimag, 4 codes. Attention rows are **PROVISIONAL** (dims/mode rough; finalize
with the softmax→m8 glue).

### C3. `inst_decoder.sv` (the only "which sub-op/phase" block)
FSM `FETCH→PREFETCH→RUN→DRAIN→NEXT` + counters `oc_idx / ic_chunk / tap`. Drives `o_oacc_clr/en`,
`o_ppu_go`, `o_prefetch`, and `o_preshift` (PEG0/2←preshift_r, PEG1/3←preshift_i). Build-time asserts:
preshift≤0 note, `ic≤256` precondition. Review: the RUN accumulate-loop off-by-one (verified clean
in audit + VCS smoke walks all sub-ops). `o_preshift` is unsigned to match CE_core (two's-comp bits).

### C4. `ce_bank_split.sv`
Combinational flat-byte-addr → {row,bank,byte} per SRAM_RTL_MEMORY_HANDOFF §3 (BANKS-parameterized:
WBUF 16 / UF 8 / KV·SKIP 4). Exact, no assumptions.

### C5. `mem_controller.sv` — 4 parts (3 fuse-independent VCS-clean + dealer SELECT)
| part | what | status / review |
|---|---|---|
| (1) PW WBUF weight addr-gen | `w_*_base + (oc*ic)/2 + ic_chunk*16` + bank-split | **ASSUMES oc-major row-major INT4** — verify vs golden 4/9. ic≤256 precondition |
| (2) STATIC §4 operand mux | PEG0=fr·xr, PEG1=fi·xi, PEG2=fr·xi, PEG3=fi·xr (x:0/2←fr,1/3←fi ; y:0/3←xr,1/2←xi) | audit "PERFECT"; an x/y or index swap silently corrupts re=P0−P1/im=P2+P3 — **review this route carefully** |
| (3) WBUF single-port arb | PE-read priority, prefetch only when `~pe_read` | mutual-exclusion |
| (4) dealer SELECT | instantiates `ce_dealer_sel`; `code_idx=oc*n_chunk+ic_chunk`, `n_codes=oc*n_chunk` | per-group PW index (attention per-head = TODO) |

### C6. `ce_dealer_sel.sv` — the dealer SELECT (this turn)
Whole WEIGHT_SCALE block in a register array (fill 1 code/cycle in PREFETCH); combinational per-PEG
extract. Descale is PRE-FUSED ⇒ NO runtime k arithmetic, only a select gated by `branches_collapse`:
- collapse=1: `PEG0,PEG2←scale_r[idx]`, `PEG1,PEG3←scale_i[idx]`.
- collapse=0: `PEG0←scale_r[idx]`(r/kreal), `PEG2←scale_r[idx+n]`(r/kimag), `PEG1←scale_i[idx+n]`(i/kimag), `PEG3←scale_i[idx]`(i/kreal).
**Review focus**: the kreal/kimag/r/i per-PEG assignment matches the PEG streams (PEG0=fr·xr→r/kreal …)
and the consultant's spec (concat order kreal-first, +n half-stride). `dealer.sv` itself unchanged.

### C7. `ce_prefetch_seq.sv` — PREFETCH sequencer (closes the dealer fill loop)
On entering PREFETCH it streams the two WEIGHT_SCALE blocks (r then i) from WBUF and **unpacks the
byte-unaligned packed 12-bit APoT codes** into `ce_dealer_sel` (1 code/cycle), then `o_done`. FSM
`S_IDLE→RD→WAIT→CAP→EMIT` with a 128-bit bit-accumulator (skip `(base&7)*8` on the first word, drain
12-bit LSB-first, refill when <12 bits). Integrated into mem_controller: it generates the prefetch
WBUF address (fed to the single-port arbitration), consumes `i_wbuf_const_rdata`, drives the dealer
fill, and `o_prefetch_done`→inst_decoder. **Audit (2-agent) confirmed unpacker logic correct
(byte-align skip, append, emit, off-by-one, FSM)**; one index-width blocker fixed (conv ≤512 codes
→ MAXC=512; the `oc*n_chunk` index is conv-PW-only, attention per-head scale = TODO, caught by a
sim assert). **Review focus**: the bit-accumulator math (`>>(base&7)*8` first word, `<<bit_cnt`
append) + the r→i block transition; throughput is ~8 cy/word (read FSM not pipelined — fine, PREFETCH).

### C8. `ce_top.sv` — control+datapath integration (Step 2-4)
Wires inst_decoder → mem_controller → CE_core and OWNS the pipeline-latency alignment. Buffers +
ppu_const_cache stay in ce_engine; here their interfaces are PORTS (smoke-testable). **Latency
(the only new timing — sub-module arithmetic is already verified)**: OACC clr/en delayed `OACC_DLY=2`
(SRAM read + PEG reg) to align with operands reaching CE_core; `drain_done` = `DRAIN_LAT=4` cycles
into DRAIN; `o_result_valid` = `RESV_DLY=2` after drain_done. **These 3 constants are PINNED BY THE
SINGLE-LAYER SMOKE — verify before trusting.** Audit (2-agent): elaborates GREEN, all 3 instance
port/width matches, handshake loop FF-broken (no comb loop), shift/counter logic correct (rewritten
index-based so robust for any DLY≥1). **Review focus**: the 3 latency constants vs the real pipe.

### C9. `ce_operand_gather.sv` — banked-window → operand codes
Combinational: shift a word-aligned SRAM read window by `(addr&7)*8` and slice `NBYTES` (16 for
32 INT4 weights, 32 for 32 e4m3 acts). The synth banked-SRAM read path; ce_engine uses it.

### C10. `ce_engine.sv` — W4A8 engine top (Step 2-5)
ce_top + **behavioral byte-addressable WBUF/UF memories** (module-scope arrays, golden TB does
`$readmemh(img, u_eng.wbuf_mem)` + preloads UF acts) + 4× `ce_operand_gather`. Reads model the real
path: word-aligned window from `(addr&~7)`, registered 1 cy (matches OACC_DLY), gather shifts the
offset, then INT4-nibble (low-first) / e4m3-byte unpack → ce_top operand codes; the const word feeds
the prefetch. Does NOT overwrite legacy `engine.sv`. **NOT yet wired** (next steps, were dead stubs):
ppu_const_cache (BN/PReLU/k prefetch), Softmax_v3 (attention — needs SEQ_LEN/ATTN_FP_WIDTH uncommented
in Param_def), KV/SKIP, act write-back. **Review focus**: the byte/row alignment of the window reads
+ the INT4-nibble unpack vs the real WBUF layout — PIN with the golden per-layer sim (task 4/9).
Banked presto SRAM macros + DMA-load = synth (Step 1a, deferred).

---

## D. TBs / runners / filelists (new this session)
`07_run_e4m3`, `08_run_ce_core`, `09_run_inst_decoder` (+ `tb_inst_decoder_smoke.sv`,
`inst_decoder_file.f`), `mem_controller_file.f`, `e4m3_file.f`, `ce_core_file.f`. All follow the `06_`
belt-and-suspenders pattern (tee log + `grep -q "[TB] ALL PASS"`).

---

## E. Open decisions / TODO (what's NOT done)
1. **(a) preshift residual — DECIDED (a), BUILDER task.** Re-bake the 12 FP8-outlier pw tensors
   (24 R2 WEIGHT_SCALE codes) with `residual = full−preshift` (range-proven safe). QK/ScoreV do NOT
   touch WEIGHT_SCALE (QK via PPU SCALE `i_scale`, ScoreV residual=1). Do before decoder-FP8 / attention
   bring-up — those paths are functionally broken (per-product Inf) without preshift. INT4-only now is fine.
2. **Prefetch sequencer**: dealer-code fill ✅ DONE (`ce_prefetch_seq`). STILL TODO: the BN/PReLU/k
   prefetch into `ppu_const_cache` (the PPU/ppu_fma side; the WBUF read can share the same prefetch
   read path). Also: the `i_wbuf_const_rdata` word-return path needs wiring to the WBUF wrapper in ce_top.
3. **DEPTHWISE addressing/microarch** (no cross-channel reduction; 5-tap OACC) — mem_controller PW-only so far.
4. **Attention expansion** (QK/Softmax_v3/Score×V glue `[?]`) + attention per-head scale index.
5. **`ce_top`** ✅ DONE (Step 2-4, audited). NEXT: a **single-layer smoke TB** to PIN the 3 latency
   constants (OACC_DLY/DRAIN_LAT/RESV_DLY) — drive one conv sub-op's operand codes + const word,
   check o_re/o_im + o_result_valid timing. Then **`ce_engine.sv`** (wrap ce_top + buffers +
   ppu_const_cache + Softmax_v3; do NOT overwrite legacy `engine.sv`).
6. **WBUF → presto_spram + arbitration** (Step 1a) still deferred.
7. **Golden per-layer numeric sim** (task 4/9) — the largest remaining verification; validates the
   oc-major layout assumption + the dealer codes end-to-end.

## F. How to re-verify (VCS machine)
```bash
cd 00_TB/CE
./06_run_PEG_w4a8 ; ./07_run_peg_preshift ; ./07_run_e4m3 ; ./08_run_ce_core
./09_run_inst_decoder                                  # re-run: struct grew (branches_collapse)
vcs -f mem_controller_file.f -R -full64 +v2k -sverilog -timescale=1ns/10ps -debug_access+all
./10_run_ce_top      # ce_top single-layer LIVENESS smoke: walks conv sub-ops, checks no
                     # hang / no X / o_result_valid pulses + OBSERVES latency. Prints the first
                     # few results + the cycle they land (use to PIN OACC_DLY/DRAIN_LAT/RESV_DLY).
```
The ce_top smoke is liveness/integration only (fixed pattern; exact numerics = the per-layer golden
sim, task 4/9). It stops before the first attention sub-op (idx 30; attention per-head scale = TODO).
