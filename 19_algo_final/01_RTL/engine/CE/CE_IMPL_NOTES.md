# CE W4A8 datapath — implementation notes (2026-06-07)

Companion to `CE_DATAPATH_DRAFT.md`. Covers what is WRITTEN + VERIFIED so far, how
the datapath flows, and how the (deferred) control FSM + buffer module should be written.

## 1. Files (new this round)
| file | module(s) | role |
|---|---|---|
| `ce_fp_util.sv` | `fp16_scale2`, `apot6a_decode`, `apot6b_decode`, `fp16_to_e4m3` | shared zero-mult FP helpers |
| `dealer.sv` | `dealer` | per-PEG APoT 2^a±2^b fused descale (×4 in CE) |
| `oacc.sv` | `oacc` | complex 2-bank accumulator (re/im) |
| `ppu_fma.sv` | `ppu_fma` | post-MAC: CONV(BN→PReLU→e4m3) / PASS / SCALE |
| `CE_core.sv` | `CE_core` | datapath SPINE: PEG×4→dealer×4→oacc→ppu_fma |
| `PEG.sv` (edited) | PEG, adder_tree_fp16, fp16_add_sat, pe_fp_unpack, pe_int4_csd | pe_lane + pe_slice_addsub MOVED out |
| `00_TB/CE/pe_lane_ref.sv` | `pe_lane`, `pe_slice_addsub` | TB-only golden reference |
| `00_TB/CE/tb_ce_core_smoke.sv` + `ce_core_file.f` | — | CE_core compile + sanity |

Legacy `CE.sv` (FP16 engine, module `CE`) left intact — new spine is `CE_core`.

## 2. Datapath (CE_core), all FP16 internal
```
                 i_x[g], i_y[g]  (per-PEG operands, routed by controller)
                       │
   ┌── PEG[0..3] ──────┤  pe_pair fused 2×8b + 32-way saturating tree
   │   o_accum[g]  (pre-descale group partial, FP16, saturating ±65504)
   │      │
   │   dealer[g]:  o = sum·2^a [± 2^b]   (fused descale code i_dealer_apot[g];
   │      │         k baked in per-PEG static slot; APoT-6a; zero-mult)
   │      ▼   dealt[0]=P0(fr·xr) dealt[1]=P1(fi·xi) dealt[2]=P2(fr·xi) dealt[3]=P3(fi·xr)
   └── oacc:  re += (P0 − P1)   im += (P2 + P3)     (i_clr=load first chunk/tap, i_en=advance)
          │   (cross-ic-chunk AND DW 5-tap accumulate live here; 2 FP16 banks)
          ▼   oacc_re, oacc_im
       ppu_fma (i_ppu_mode):
          CONV  : BN(α·x via dealer + β) → PReLU(pow2 ±2^k) → e4m3 cast(next-layer k) → o=[7:0] e4m3
          PASS  : o = oacc (fp16)                      (Score×V ctx → out_proj)
          SCALE : o = dealer(oacc, scale code) (fp16)  (QK ×2^-2·2^-k before softmax)
          ▼ o_re, o_im
```
- Zero-multiplier throughout: every scale is `fp16_scale2` (exponent-add) ± `fp16_add_sat`.
- PEG-per-stream (§2 draft): controller feeds PEG0=fr·xr, PEG1=fi·xi, PEG2=fr·xi, PEG3=fi·xr;
  dealer codes per-PEG are static-fused (weight APoT − act kreal/kimag). re/im combine is hardwired.
- Unit counts realised: 4 dealer (1/PEG), 1 oacc (2 banks), 1 ppu_fma (pool of 1 for now; can replicate to 4).
- ✅ **Precision (RESOLVED 2026-06-07):** per-path headroom verified (golden + calib bound).
  INT4 conv/proj/DW/out_proj safe (≤0.19×). FP8 outlier (4.2×) / QK (33×) / Score×V (5.7×)
  OVERFLOWED (both operands scaled to full e4m3 range; single product alone > 65504 → pe_pair
  per-product Inf, so widening the tree cannot help). FIX APPLIED: per-PEG `i_preshift` (per-tensor
  coarse pow2) folded into pe_pair's 4 product exponents (int4/lmul × lo/hi), zero extra HW;
  dealer carries the residual (full − preshift). Post-fix: all paths ≤0.49× / 0 overflow
  (QK 2^-10→0.03×, ScoreV 2^-13→0.001×, FP8 2^-3/-4→0.49×). `preshift=0` → INT4/DW bit-exact.
  Controller programs preshift (per-tensor) + residual; INT4 per-group path keeps preshift=0 (雷②).
  TB re-run on a VCS machine still owed: confirm 16000/0 + add a `preshift≠0` directed case.

## 3. Verification
- `06_run_PEG_w4a8`: PEG (pe_pair) vs PEG_ref(pe_lane×32) = **16000 vec / 4 modes / 0 err** (also re-confirms pe_lane move).
- `tb_ce_core_smoke`: W4A8, PEG0 = 32×(1×1.0)=32.0, dealer ×1, PASS → **o_re=0x5000(32.0), o_im=0 PASS** (spine elaborates + runs).
- NOT yet verified: bit-accumulation vs external fp32 golden; dealer/ppu APoT/BN/PReLU/cast numerics vs golden codes; e4m3 cast RNE vs `fp8_qat` (雷④). These need the controller + real WBUF const codes + golden patterns.

## 4. The control / memory module (NEXT — deferred per request)
`CE_core` is pure datapath: operands, descale codes, OACC ctrl, PPU consts are PORTS.
The next module (`CE_top` / mem+layer controller) must provide:

### 4a. Buffers to instantiate (reuse existing *_top, wrap presto behavioral SRAM)
| buffer | module | feeds |
|---|---|---|
| WBUF | `wbuf_top` (wbuf_ctrl+wbuf_sram) | weights → i_x/i_y; const region (APoT/BN/PReLU/k) → PPU-const-cache → dealer/ppu codes |
| UFBUF | `ufbuf_top` (ping-pong) | layer act in (load) → i_y; FMA out (store) ← o_re/o_im. swap per layer |
| KVBUF | `kvbuf_top` | K/V (in_proj write) ; QK/Score×V read |
| SKIPBUF | `skipbuf_top` | decoder PW skip branch (static **top-50%** channels, groups_skip=2) |
> ⚠️ SRAM is behavioral (`presto_dpram`); **real SRAM macro swap is a TODO — remind before tape-out / synth.**

### 4b. FSM (layer/sub-op sequencer) — how to write it
Outer: walk the 63 layers (encoder 6×{dw,pw} → MHA sub-ops → decoder 6×{pw_x,pw_skip,dwT}); per layer set `i_pe_mode`, `i_ppu_mode`, and the const-block prefetch.
Per layer, a sub-FSM:
```
LOAD_CONST : prefetch this sub-op's APoT/BN/PReLU/k from WBUF const region → PPU const cache
RUN        : for each output-channel tile:
               for ic_chunk in 0..⌈ic/32⌉-1 (and DW: ×5 taps):
                 drive WBUF weight read + UFBUF/KVBUF act read → i_x/i_y (PEG-per-stream routing)
                 i_dealer_apot[0..3] = fused descale for this chunk's 4 streams
                 i_oacc_clr = (first chunk/tap),  i_oacc_en = 1
               # OACC now holds re/im for this output channel
               ppu_fma (i_ppu_mode + BN/PReLU/cast codes) → write UFBUF(store) / KVBUF
DONE→swap UFBUF halves
```
Notes: drain rate must be ≥4 group-partials/cycle (PE never stalls); small layers underfill PE (front-end duty, not a CE-unit issue). Attention: QK uses SCALE mode (×2^-2·2^-k) → Softmax_v3 (`01_RTL/engine/Softmax/Softmax_v3.sv`) → Score×V via PEG (m8, PASS post) → out_proj (W4A16). Softmax packet→m8 operand glue still `[?]` (§9 draft).

## 5. Open / TODO
- controller + buffer instantiation (§4) — the real "串起來" with sequencing.
- bit-exact verification vs golden (dealer/ppu codes, e4m3 RNE 雷④, accumulation tolerance).
- Score×V packet→m8 glue; KV ring addressing under 4-bank×1024.
- real SRAM macro swap.
- optional: replicate ppu_fma to a pool of 4 (throughput margin; 1 suffices per §4a draft).
