# SRAM Memory Map & Weight Placement Spec (strict-18 deploy)

Date: 2026-06-03. Authoritative for the SRAM-image / TB-pattern generator.
Basis: `export_out_strict18/` RTL-preferred compact files (in_proj `[12,64,64]`,
decoder grouped pointwise, FP8 outlier e4m3). All byte counts measured from the
actual `.bin` files, not estimated.

Deploy ckpt: `ckpt/strict_core_pow2_blockfloat_deploy_ep3_vctk0.8357_timit0.7898.ckpt`.

---

## 1. SRAM configuration (FROZEN)

```verilog
// WBUF: single-port UHD, full model resident (weights + ALL constants)
parameter WBUF_BANKS = 16;
parameter WBUF_WIDTH = 64;
parameter WBUF_DEPTH = 2048;   // 256 KiB

// KVBUF: true dual-port, K/V raw e4m3 ring for tied real/imag MHA
parameter KV_BANKS = 4;
parameter KV_WIDTH = 64;
parameter KV_DEPTH = 1024;     // 32 KiB

// UFBUF: true dual-port, current-frame activations / scratch
parameter UF_BANKS = 8;
parameter UF_WIDTH = 64;
parameter UF_DEPTH = 128;      // 8 KiB

// SKIPBUF: true dual-port, top-50% skip subset only
parameter SKIP_BANKS = 4;
parameter SKIP_WIDTH = 64;
parameter SKIP_DEPTH = 128;    // 4 KiB
```

**No separate CONSTBUF.** Packed constants live inside the WBUF constants region
(§3). A small **PPU local const cache** (register file, NOT an SRAM macro, §5)
prefetches the per-sub-op constants so the PPU never reads WBUF during the PE MAC
phase.

### Throughput check
| buffer | supply | peak demand | ok |
|---|---|---|---|
| WBUF single-port | 16×64 = 1024 b/cyc | INT4 128-lane 512 b; FP8 64-lane 512 b | ✅ |
| KVBUF 2R | 4×64×2 = 512 b/cyc = 64 B | L-Mul m8 64-lane 64 B/cyc | ✅ |
| UFBUF 2R | 8×64×2 = 1024 b/cyc = 128 B | W4A8 128-lane 128 B/cyc | ✅ |
| SKIPBUF 2R | 4×64×2 = 512 b/cyc = 64 B | groups_skip=2, 64-lane | ✅ (note A) |

> Note A — SKIPBUF: decoder skip pointwise is `groups_skip=2`; most stages have
> small skip channel counts so 64-lane is sufficient. If RTL wants full 128 unique
> A8/cycle on the skip path, raise SKIP_BANKS to 8 (8 KiB). Default 4 banks.

---

## 2. WBUF region budget (measured, compact basis)

| Region | Contents | Bytes | KiB |
|---|---|---:|---:|
| R0 weight INT4 | 54 tensors, compact (in_proj `[12,64,64]`, grouped decoder) | 223,312 | 218.1 |
| R1 weight FP8 | 12 decoder outlier tensors, e4m3 | 17,024 | 16.6 |
| **weight subtotal** | | **240,336** | **234.7** |
| R2 weight-side APoT scale | **3700 codes** × 12b (3548 fused INT4 + **128 MHA non-fused base** + 24 FP8 outlier fused descale) | 5,550 | 5.42 |
| R3 BN APoT scale | 1250 (r+i) × 12b | 1,875 | 1.83 |
| R4 BN bias fp16 | 1250 (r+i) × 16b | 2,500 | 2.44 |
| R5 PReLU (6b) | 22×12b | ~33 | 0.03 |
| R5b act-k | 53×u8 | 53 | 0.05 |
| **constants subtotal** | | **~10,013** | **~9.8** |
| **WBUF used (measured, 12bit pack)** | | **251,677** | **245.8** |
| **WBUF capacity** | | **262,144** | **256.0** |
| **slack** | | **10,467** | **10.2 (4.0%)** |

> **Offset/address table is NOT a WBUF payload region.** Per-tensor and per-const
> WBUF offsets live in `memory_map.json` (host/RTL metadata) and are compiled into
> `memory_map_pkg.svh` localparams (§7). The WBUF image holds only weight codes +
> packed constants; do not reserve a WBUF region for the offset table.

> **R2 count correction (codex):** weight-side scale is **3700**, not 3548. 3548
> is fused INT4 only; add 128 MHA non-fused base + 24 FP8 outlier fused.

> WBUF is **95.7% full** — the constants MUST be packed (12b APoT, fp16 bias). Do
> not store int16 exponents / int8 signs / fp32 bias in the SRAM image (that is
> ~24 KiB of scale and would overflow). Debug `.fp32` files stay out of the image.

---

## 3. WBUF address layout (two regions, bank-uniform)

WBUF is a flat 16×64×2048 = 262,144 B space. Address = `depth*128 + bank*8 + byte`
(16 banks × 8 B/word = 128 B per depth row). Two address regions:

```
depth row   bytes                 region
0     .. 1887   [0 .. 241,602]     R0+R1   WEIGHT CODE region   (PE reads here)
1888  .. ~1966  [241,664 .. 251,676] R2..R5b CONSTANTS region  (prefetched to PPU cache)
~1967 .. 2047   slack (10,467 B)
```
(measured, 12bit pack; `const_base = 241,664`, row 1888.)

- **Weight code region first**, each tensor **64 B aligned** (one full depth-row
  half / clean nibble boundary). Tensor order = §4 execution order so DMA-free
  resident layout matches the controller's natural walk.
- **Constants region after** all weight code. Sub-regions R2→R5b in that order;
  each const *block* is independently packed and has an explicit `wbuf_offset` in
  `memory_map.json` (so it is directly DMA-able to the PPU cache).
- Region boundary (`const_base`) is emitted in `memory_map.json` and
  `sram_config.vh` as `WBUF_CONST_BASE_ROW`. The offset table itself is metadata,
  not a WBUF region (§2 note).

### Why bank-uniform (not split banks)
Single-port WBUF cannot serve PE and PPU on the same bank in the same cycle.
Rather than dedicating banks, we **prefetch** (§5): during the MAC phase the PPU
reads its local cache, not WBUF, so PE owns all 16 banks. The weight/const split
is by **address (depth)**, and the two are accessed in **different pipeline
phases**, never concurrently on WBUF.

---

## 4. Weight tensor ordering (execution order = resident layout)

Per §8/§9 of `HANDOFF_HW_ZERO_MULTIPLIER_CORE.md`:

1. Encoder stage 0..5 — each: `depthwise_r, depthwise_i, pointwise_r, pointwise_i`
2. MHA — `input_proj.fc_r, input_proj.fc_i`, internal `in_proj_weight` (compact
   `[12,64,64]`), internal `out_proj`, `output_proj.fc_r, output_proj.fc_i`
3. Decoder stage 0..5 — each: `pointwise_{r,i}_{x,skip}`, `depthwise_transpose_{r,i}`

Hard layout rules (verifier-enforced):
- MHA internal in_proj **MUST** use compact `w_int4.blocks.packed.bin` (24 KiB),
  never dense `[768,256]` (96 KiB).
- Decoder pointwise **MUST** use grouped shapes (`groups_x=4`, `groups_skip=2`).
- FP8 outlier (decoder stage 0/3/4, 12 tensors) use `w_fp8.e4m3.bin`.
- **No** dense / `_debug` / `effective` / `.fp32` weight files in the image.

---

## 5. PPU local const cache (Option 1.5 — adopted)

**Decision (codex):** layer-level full serialization is rejected — it makes
MAC+PPU ≈ 17K cycles/frame, but 2.5 MHz @ 200 fps allows only 12.5K cycles/frame.
Instead constants stay in WBUF but are **prefetched** to a small PPU-local
register file so PE(layer N+1) and PPU(layer N) overlap.

```
MAC phase:      PE reads WBUF weight region only.
descale phase:  PPU reads PPU-local const cache (prefetched), NOT WBUF.
prefetch:       before each sub-op, controller copies that sub-op's const block
                WBUF const region -> PPU local cache.
```

### Cache sizing (measured worst case, from `memory_map.json` `max_prefetch_unit_bytes`)
| prefetch granularity | max unit | cache cap |
|---|---:|---|
| `--prefetch stage` (whole stage at once) | 899 B (encoder.model.5.0) | **4 KiB** |
| **`--prefetch subop`** (each const block a separate unit) | **768 B** (enc5 pointwise weight scale, 512 grp×12b) | **2 KiB** ✅ |

> The sub-op peak is **768 B**, not 896 B: BN scale and BN bias are emitted as
> **separate prefetch units** (R3 vs R4), so they are never fetched together. The
> largest single unit is the enc5 pointwise weight-scale block (512 groups × 12b).
> RTL must not assume it prefetches BN scale+bias as one unit.

**Recommendation: per-sub-op prefetch, PPU local const cache = 2 KiB**
(768 B peak + double-buffer → builder reports `double_buffer_cache_bytes`). If RTL
prefers whole-stage prefetch, use 4 KiB. This is a register file / latch array,
**not** counted as one of the 4 SRAM macros.

`memory_map.json` emits per-layer `const_prefetch` lists and reports
`max_const_prefetch_bytes` so RTL can size the cache.

---

## 6. APoT constant encoding

Unified **12-bit packed** code. Two interpretations of the low 2 bits, selected by
which constant region the code lives in (region tag in `memory_map.json`):

### 6a. APoT scale code (weight scale R2, BN scale R3) — 2-term `2^a ± 2^b`
```
[11:7] a_exp   5b  biased (+24), covers fused range [-24,+7]
[6:2]  b_exp   5b  biased (+24)
[1]    has_second  1b   1 = second term 2^b present
[0]    sign_lo     1b   role depends on region (see below)
```
- **Weight scale (R2):** overall magnitude < 1, overall positive. `has_second`
  is constant 1 in this ckpt (packer asserts). `sign_lo` = **residual (second-term)
  sign** (1 ⇒ `−2^b`). Ignored only if `has_second=0` (does not occur for weights).
- **BN scale (R3):** γ may be negative ⇒ needs an overall sign. In this ckpt the
  exported `overall_sign` is constant **+1** (packer asserts == 1). `sign_lo` =
  **residual sign** (BN `residual_sign ∈ {−1,0,+1}`; 0 occurs iff `has_second=0`,
  in which case `sign_lo` is ignored). If a future retrain makes γ negative, the
  packer's `overall_sign==1` assert fires — extend the format then (e.g. steal an
  unused a_exp code point or widen to 13b); do NOT silently drop it.

### 6b. PReLU pow2 code (R5) — single term `±2^k`, NO second term
```
[11:7] k_exp   5b  biased (+24), measured range [-8,-1]
[6:2]  (zero)  5b
[1]    has_second = 0   (always — PReLU is single pow2)
[0]    overall_sign 1b  1 = negative slope (−2^k); 0 = positive
```
- **Distinct from 6a:** for PReLU, `bit0` is the **overall slope sign**, NOT a
  residual sign, and it is **always meaningful** (not gated by has_second). 7 of 22
  slopes are negative in this ckpt — dropping bit0 here would flip them positive.
  RTL must decode R5 with the 6b rule, R2/R3 with the 6a rule.

### Notes
- Fused weight scale a_exp range measured [-19,-1], b_exp [-23,-3] ⇒ 5b biased.
- BN bias: **fp16**, validated **no MBSTOI regression** (VCTK 0.8357 / TIMIT
  0.7898 unchanged to 4 dp) — **not bit-exact to fp32** (waveform maxabs not
  measured).
- activation-k (R5b): stored as `u8 = k + 32` (k range [-2,9]).

**Alternative (RTL-simpler):** 16-bit aligned APoT (pad 12→16) — ~13–14 KiB total
constants, still fits 256 KiB, cleaner word alignment, no cross-word unpack.
Builder supports both via `--scale-pack {12bit,16bit}`.

---

## 7. Required outputs
```
sram_images/wbuf_unified_init.bin     # <= 262,144 B, weights + packed constants
sram_images/kvbuf_init_zero.bin       # 32 KiB zeros
sram_images/ufbuf_init_zero.bin       # 8 KiB zeros
sram_images/skipbuf_init_zero.bin     # 4 KiB zeros
sram_images/memory_map.json           # per-region + per-tensor offset/len + const_prefetch
sram_images/sram_config.vh            # SRAM params + WBUF_CONST_BASE_ROW + PPU_CONST_CACHE_BYTES
sram_images/memory_map_pkg.svh        # RTL-friendly localparams (tensor/const offsets) compiled from JSON
sram_images/README.md
```
Regenerate: `conda activate bcctn && python software_to_hardware_18/build_sram_images.py`
(needs numpy from the `bcctn` env; base Python will fail).

## 8. Verification (verifier must pass all)
- [ ] WBUF image ≤ 262,144 B; report per-region usage / slack.
- [ ] Decode WBUF weight-code region → byte-compare to `tb_pattern/weights` (compact).
- [ ] Decode packed APoT/BN/PReLU/act-k → compare to source exponent/sign fields.
- [ ] Assert no dense/debug/effective/fp32 weight file included.
- [ ] Assert MHA in_proj uses compact `[12,64,64]`; decoder uses grouped shapes.
- [ ] Assert `max_const_prefetch_bytes` ≤ PPU cache cap (2 KiB sub-op / 4 KiB stage).
- [ ] R2 weight-side scale count == 3700.
```
