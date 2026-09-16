# CE W4A8 Per-Layer Golden Sim — Handoff (2026-06-08)

Task 4/9: numerically verify the W4A8 conv core against the **real fp32 reference golden**.

## STATUS: all 6 encoder pointwise layers BIT-EXACT (376 channels, 0 ulp)

| sub-op | layer | ic | oc | chunks | result |
|--------|-------|----|----|--------|--------|
| 1  | encoder.model.0.0.pointwise | 1   | 8   | 1 | **8/8 bit-exact** |
| 3  | encoder.model.1.0.pointwise | 8   | 16  | 1 | **16/16 bit-exact** |
| 5  | encoder.model.2.0.pointwise | 16  | 32  | 1 | **32/32 bit-exact** |
| 7  | encoder.model.3.0.pointwise | 32  | 64  | 1 | **64/64 bit-exact** |
| 9  | encoder.model.4.0.pointwise | 64  | 128 | 2 | **128/128 bit-exact** |
| 11 | encoder.model.5.0.pointwise | 128 | 128 | 4 | **128/128 bit-exact** |

Each `oacc_re[oc]` matches a BIT-EXACT fp16 model (same paired tree + APoT dealer + cross-chunk
OACC accumulate) that itself reconstructs the independent fp32 reference (slope/affine R²=1). The
sweep exercises single-product (ic=1) → full 32-way tree (ic=32) → multi-chunk OACC accumulate
(ic=64/128). Run: `./12_run_ce_engine_golden` (enc0) or compile once + sweep (see §C).

This note documents (A) the decoded golden contract, (B) the new files, (C) how to run, (D) the
**three** integration findings the sweep surfaced (all FIXED), (E) the path to decoder/post-BN golden.

Everything here is **review-then-run**: I cannot run VCS on my side. The Python golden
generator is fully self-validated (below); the SV TB needs your VCS.

---

## A. Decoded + validated golden contract

The fp32 per-module reference (`debug_reference/layer_io_fp32/layers/<slice>/output_0.fp32.hex`)
for a pointwise `_r` branch is an **exact per-channel affine** of its input (regression R²=1.0,
no PReLU kink):

```
fp32_golden[oc, t] = wf_fused[oc] * xdeq[t] + conv_bias[oc]
```

| term | source (all in-repo) |
|------|----------------------|
| `wf_fused[oc] = int4_w[oc] * apot_scale[oc]` | `wbuf_unified_init.bin` @ weight off + WEIGHT_SCALE off (APoT 2^a±2^b, FUSED descale incl. 2^-k_act) |
| `xdeq[t] = e4m3_decode(act_code[t])` | `activations_fp8/<slice>/input_0.e4m3.u8.bin`; per-tensor block-float `k = -8` |
| `conv_bias[oc]` | ORIGINAL per-channel conv bias. Deploy build sets `conv_bias_fused_into_bn_bias=true`, so the **HW conv core does NOT add it** — it reappears in the PPU BN-bias FMA. (Hence `conv_bias != bn_bias_*_fused` 1:1; the stored fused bias already mixes BN β + scale.) |

**Independent (non-circular) tie-in = the slope.** From the image bytes alone (INT4 + APoT
decode) `A_hw[oc] = wf_fused[oc] * 2^(-k)` matches the golden's regressed slope `A_golden[oc]`
to **2.2e-08 for all 8 channels** — see the generator's `[CHECK]` lines. This proves the
INT4-decode + APoT `2^a±2^b` descale + e4m3 numeric chain the RTL implements is correct against
the fp32 reference, independent of the bias.

**What the CE conv core reproduces** (gather → INT4 CSD → FP16 sat-tree → APoT dealer descale →
OACC) is the **pre-bias partial**:

```
oacc_re[oc] = wf_fused[oc] * xdeq[t]   == fp32_golden[oc,t] - conv_bias[oc]
```

so the SV TB observes `CE_core.oacc_re` (PPU is benign-stubbed in ce_engine → `o_re` is NOT this).

Capstone Python check (all pass, `gen_golden_enc0_pw.py`):
- image INT4 @128 == per-tensor `w_int4.packed.bin` → `[-7,7,-7,-7,-7,7,7,7]`
- image APoT codes @241668 decode == per-tensor `scale_apot_*_fused` a/b exps (8/8)
- `wf_fused*xdeq + conv_bias` reconstructs fp32 golden to **6e-08**.

---

## B. New files

| file | role |
|------|------|
| `10_algo/scripts/gen_golden_enc0_pw.py` | self-validating golden generator. Emits, under `hw_patterns_final/sram_images/golden/`: `act_enc0_pwr_t0.hex` (1 byte), `golden_enc0_pwr_preBN_t0.hex` (8× fp16 = compare target), plus full-`T` and post-bias variants + `…_meta.json`. Prints the slope tie-in + full-recon checks. |
| `10_algo/scripts/wbuf_bin2hex.py` | `.bin` → byte/line hex for `$readmemh` (the TB uses `$fread` instead, so this is optional). |
| `00_TB/CE/tb_ce_engine_golden.sv` | the per-layer golden TB (details below). |
| `00_TB/CE/tb_ce_engine_golden_file.f` | filelist (ce_engine spine + TB). |
| `00_TB/CE/12_run_ce_engine_golden` | runner (env + auto-regen golden + VCS). |

Modified: `01_RTL/engine/CE/ce_engine.sv` — const-read now ENABLE-GATED (finding #1 below).

---

## C. How to run

```bash
cd 00_TB/CE
./12_run_ce_engine_golden        # regenerates golden if missing, then VCS
```

The TB:
1. `$fread`s the real 256 KiB `wbuf_unified_init.bin` into the behavioral `wbuf_mem`.
2. Zeros `uf_mem`, places **only** `act[0]` at `uf[0]` (ic=1 ⇒ lanes 1..31 must be 0, else they
   pollute the FP16 tree — there is no ic<32 lane mask yet).
3. Pulses `i_start`, lets the decoder walk to sub-op 1 (`encoder.model.0.0.pointwise`),
   hierarchically samples `CE_core.oacc_re` at each `drain_done` tagged by `oc_idx`.
4. Compares the 8 OACC values against `golden_enc0_pwr_preBN_t0.hex` (rel tol 4%, ~1–2 fp16 ulp).

Hierarchical probes used: `u_eng.u_top.u_id.{subop_idx,oc_idx}`, `u_eng.u_top.drain_done`,
`u_eng.u_top.u_ce.{oacc_re,oacc_im}`.

---

## D. Three findings this sweep surfaced (all FIXED)

### Finding #1 (FIXED here) — const-read latency in the behavioral WBUF

`ce_prefetch_seq` issues a word address in `S_RD` and reads `i_rd_data` in `S_CAP` (T+2),
assuming an **enable-gated SRAM whose output register HOLDS**. The original `ce_engine`
behavioral const read was **free-running** (re-read every cycle), so by `S_CAP` it already
showed the *next* word (`word_addr` increments in `S_RD`) → the dealer would latch wrong APoT
codes and **every** oc would fail.

Fix (ce_engine.sv): gate the `win_const` update with `wbuf_en` so it holds between reads —
exactly an enable-gated SRAM output reg. PE weight/act reads stay free-running (they are
consumed at T+1, no hold needed). This is **behavioral scaffolding only** (not synthesizable
RTL); the real `presto_spram` swap (Step 1a) will hold naturally.

### Finding #2 (OPEN) — ic=1 odd-oc weight nibble addressing

`mem_controller` PW addr-gen: `pw_off = (oc*ic)>>1 + ic_chunk*16`. The `>>1` **drops the
`(oc*ic)&1` nibble**. The gather always puts the low nibble of the byte address on lane 0, so
for **odd `oc*ic`** (only happens at **ic=1**, i.e. `encoder.model.0.0.pointwise`) the lane-0
weight is `w[oc-1]` instead of `w[oc]`.

Expected first-run result (current RTL): **PASS on oc 0,2,4,6** (even — correct nibble) **and
oc 3,7** (odd but `w[oc-1]==w[oc]` here), **FAIL on oc 1,5** (`w[oc-1]≠w[oc]`, e.g. oc1 reads
`w[0]=-7` for `w[1]=+7` → sign flip). The PASS set already validates gather → CSD → FP16 tree →
dealer → OACC against the real fp32 golden — the milestone. The two FAILs localize the gap.

**Fix (DONE + verified):** threaded a nibble-parity bit `o_w_nibble_odd = (oc*ic)&1`
`mem_controller` → `ce_top` → `ce_engine`, widened the weight gather window 16→17 B
(`ce_operand_gather`), and offset the lane-`l` nibble extract by `nibble_odd`. No-op for even-ic
(the common case). enc0 → 8/8 bit-exact after the fix.

### Finding #3 (FIXED here) — multi-chunk dealer APoT scale misalignment (REAL HW bug)

The most important find: for ic>32 (multi-chunk: enc4 2×, enc5 4×) the per-(oc,32-ic-group) APoT
dealer code is generated combinationally from the decoder's `ic_chunk` counter @T, but the matching
PEG partial only reaches the dealer @T+OACC_DLY (SRAM read + PEG reg). So chunk `c` was descaled with
chunk `c+2`'s scale → wrong cross-chunk OACC sum. **Single-chunk layers (ic≤32) have a constant code,
so this was invisible** — and the `10_run_ce_top` liveness smoke (no numeric check) passed it too. The
sweep caught it: enc4 28/128, enc5 3/128 (only the oc whose two chunk-scales happened to be equal).

Fix (ce_top): delay `mc_dealer` by `OACC_DLY=2` through a shift register, exactly like `clr_pipe`/
`en_pipe`. `preshift`/`pe_mode`/`ppu_mode` are per-sub-op constants → no per-chunk delay needed.
After the fix: enc4 128/128, enc5 128/128 bit-exact; single-chunk unchanged; ce_top smoke still PASS.
Lesson: liveness ≠ numerics — only the golden sweep across ic catches per-chunk scheduling bugs.

---

## E. Toward the full golden

enc0 (ic=1) is the *hello-world* golden: 1 active lane, single product — it does **not**
exercise the 32-wide FP16 reduction or the multi-ic nibble layout. The representative golden is
an **even-ic** pointwise:

- `gen_golden_enc0_pw.py` is enc0-specific; generalize it (parameterize the dotted layer name;
  pull offsets from `memory_map.json` `tensors`/`const_blocks`, act from
  `activations_fp8/<slice>`, golden from `layer_io_fp32/layers/<slice>`).
- Recommended target: **sub-op 7 = `encoder.model.3.0.pointwise` (ic=32, oc=64)** — fills the 32
  lanes exactly (1 ic_chunk), clean byte-aligned addressing, exercises the full saturating tree.
- Pin the multi-ic **activation layout** (channel-major `[ic, T]` vs minor) when loading one
  timestep's 32 ic values into `uf[0..31]` — for enc0 (ic=1) it was moot.
- Post-BN (full `o_re`) golden needs the PPU const FILL (BN scale/bias/PReLU/cast) wired into
  `ppu_const_cache` (still TODO) and compares against `golden_enc0_pwr_full*` with an e4m3-cast
  tolerance. The pre-BN OACC golden here needs none of that.

Open downstream (unchanged): DW addressing/microarch, attention QK/Softmax/ScoreV glue +
per-head scale index, act write-back, WBUF→presto_spram, preshift-residual image rebuild
(decision (a)) before FP8-outlier/attention bring-up.

---

## F. Cycle optimization v1 — per-oc DRAIN eliminated (conv 19,368 → 12,658, −35%)

VCS-measured (ce_top smoke, conv sub-ops 0–29), phase breakdown:

| | before | after |
|---|---|---|
| total | 19,368 | **12,658** |
| RUN (compute) | 3,876 | **3,876 = IDEAL** |
| PREFETCH | (folded) | **8,569 (68%)** |
| DRAIN | (per-oc) | 150 (per-sub-op) |

**What changed:** the per-output-channel DRAIN bubble (4 cy each) is gone — `inst_decoder` now does
CONTINUOUS RUN across oc (advance `oc_idx` INLINE at `last_chunk && last_tap`; DRAIN once per sub-op
only, to flush the last channel). A new `o_oc_complete` strobe (one pulse per oc, at its last
accumulate) drives `ce_top.o_result_valid` through a `RESULT_LAT = OACC_DLY+2 = 4` shift register
(replacing the drain-based valid). RUN is now exactly the theoretical ideal.

**4th alignment bug it exposed (FIXED):** with continuous RUN, `oc_idx` advances every cycle, so the
combinational `w_nibble_odd` was oc+1's parity at the (registered) weight-gather cycle. Fixed by
delaying it one cycle (`w_nibble_odd_r`) to match the registered weight window. Only ic=1 felt it
(even-ic parity is always 0). All 6 golden layers stay bit-exact; smoke still 1372 results / 0 X.

**Verify:** golden sweep (`SAMPLE_LAT=3` = `RESULT_LAT−1`, the oacc-settle latency) + `10_run_ce_top`
(`phase breakdown` line). The golden TB now samples `oacc_re` at `o_oc_complete + SAMPLE_LAT` (works
for both the old hold-during-DRAIN and the new 1-cycle-valid timing).

**v2 (next, bigger — has an architecture choice):** PREFETCH (8,569, 68%) is now the bottleneck. It is
serial — 1 code/cycle, ~3-cy/word read stall, and r-block then i-block — but RUN consumes 1 r + 1 i
code per cycle, so prefetch ≈ 2.2× RUN and cannot be hidden by simple overlap. To approach RUN (~4–5k
total): parallel r/i streams + multi-code/cycle unpack (multi-write-port `ce_dealer_sel`) + stream/
ping-pong overlap with RUN (the `ppu_const_cache` ping-pong + `i_cache_half` hooks exist). ⚠️ Confirm
target clock first: `pe_pair` says 20 MHz (→ 100k-cy budget, already fits at 12,658); at 2.5 MHz
(12,500-cy budget) v2 is needed once attention (~6.5k) is added.
