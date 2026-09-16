# SRAM / WBUF RTL Memory Handoff (English)

Date: 2026-06-03

This document hands off the SRAM / WBUF initialization contract for the strict-18 pow2 block-float deploy. Authoritative artifacts:

- `hw_patterns_final/sram_images/wbuf_unified_init.bin`
- `hw_patterns_final/sram_images/memory_map.json`
- `hw_patterns_final/sram_images/memory_map_pkg.svh`
- `hw_patterns_final/sram_images/sram_config.vh`
- `software_to_hardware_18/build_sram_images.py`
- `software_to_hardware_18/verify_sram_image_standalone.py`

`memory_map_pkg.svh` is the RTL-friendly complete offset table. If this document disagrees with `memory_map.json` / `memory_map_pkg.svh`, use `memory_map.json` and `memory_map_pkg.svh` as the source of truth.

---

## 1. Frozen SRAM configuration

```verilog
parameter WBUF_BANKS = 16;
parameter WBUF_WIDTH = 64;
parameter WBUF_DEPTH = 2048;   // 256 KiB, single-port UHD, full model resident

parameter KVBUF_BANKS = 4;
parameter KVBUF_WIDTH = 64;
parameter KVBUF_DEPTH = 1024;  // 32 KiB, dual-port

parameter UFBUF_BANKS = 8;
parameter UFBUF_WIDTH = 64;
parameter UFBUF_DEPTH = 128;   // 8 KiB, dual-port

parameter SKIPBUF_BANKS = 4;
parameter SKIPBUF_WIDTH = 64;
parameter SKIPBUF_DEPTH = 128; // 4 KiB, dual-port

parameter WBUF_CONST_BASE_ROW = 1888;
parameter WBUF_USED_BYTES = 251677;
parameter PPU_CONST_CACHE_BYTES = 2048;
```

WBUF is a single-port UHD SRAM and holds the full model: weight codes plus packed constants. There is no separate CONSTBUF. Runtime constants are prefetched from WBUF into a PPU-local const cache, so the PPU does not read WBUF during the PE MAC phase.

---

## 2. WBUF flat image layout

WBUF image is a single flat byte stream:

```text
file: hw_patterns_final/sram_images/wbuf_unified_init.bin
size: 262,144 B
used: 251,677 B
slack: 10,467 B
```

Flat regions:

| Region | Start byte | End byte inclusive | Bytes | Purpose |
|---|---:|---:|---:|---|
| weight_code | 0 | 241,602 | 241,603 | 66 weight tensors, 64B aligned |
| row-align padding | 241,603 | 241,663 | 61 | padding before constants |
| weight_apot_scale | 241,664 | 247,215 | 5,552 | 3700 packed 12-bit APoT scale codes |
| bn_prelu | 247,216 | 251,623 | 4,408 | BN APoT, BN bias fp16, PReLU codes |
| activation_k | 251,624 | 251,676 | 53 | block-float activation exponent table |
| slack | 251,677 | 262,143 | 10,467 | unused |

`WBUF_CONST_BASE_ROW = 1888`, so `1888 * 128 = 241,664`, the first byte of the constants region.

Note: the top-level `memory_map.json` region keys are `weight_code`, `weight_apot_scale`, `bn_prelu`, and `constants`. The table above lists `activation_k` separately for RTL readability; in JSON it is a `const_block` inside the `constants` region.

---

## 3. Bank import rule

### WBUF

WBUF has 16 banks, 64-bit per bank. One depth row contains:

```text
16 banks * 8 B = 128 B / row
```

For a flat byte address `A` in `wbuf_unified_init.bin`:

```text
row      = A >> 7          // divide by 128
bank     = (A >> 3) & 0xF  // 0..15
byte_sel = A & 0x7         // byte inside 64-bit bank word
```

For bank `b`, row `r`, load bytes:

```text
base = r * 128 + b * 8
bank_word_bytes = image[base : base + 8]
```

If `$readmemh` / SRAM init expects a conventional 64-bit hex word with MSB on the left, write:

```verilog
word64 = { image[base+7], image[base+6], ..., image[base+0] };
```

This keeps `word64[7:0]` equal to the lowest byte address. Do not reinterpret or repack the payload while splitting banks.

Fixed byte-order invariant:

```verilog
// little-endian inside each 64-bit bank word
// for i = 0..7:
//   word64[8*i +: 8] == image[base+i]
```

That is, `image[base+0]` must land in `word64[7:0]`, and `image[base+7]` must land in `word64[63:56]`. Do not apply an extra byte-swap.

### KVBUF / UFBUF / SKIPBUF

The zero-init files use the same flat-to-bank rule with their own row size:

| Buffer | Banks | Width | Row bytes | Address mapping |
|---|---:|---:|---:|---|
| KVBUF | 4 | 64b | 32 B | `row=A>>5`, `bank=(A>>3)&3`, `byte=A&7` |
| UFBUF | 8 | 64b | 64 B | `row=A>>6`, `bank=(A>>3)&7`, `byte=A&7` |
| SKIPBUF | 4 | 64b | 32 B | `row=A>>5`, `bank=(A>>3)&3`, `byte=A&7` |

Current zero-init files:

- `kvbuf_init_zero.bin`: 32,768 B
- `ufbuf_init_zero.bin`: 8,192 B
- `skipbuf_init_zero.bin`: 4,096 B

---

## 4. Weight locations

Use `memory_map_pkg.svh` in RTL/TB. It contains:

```verilog
localparam int N_WEIGHT_TENSORS = 66;
localparam int N_CONST_BLOCKS   = 103;
localparam int WBUF_CONST_BASE  = 241664;
```

Every weight has:

```verilog
localparam int W_<NAME>_OFF = ...;
localparam int W_<NAME>_LEN = ...;
```

Weight payload conventions:

- INT4 weights: packed signed 4-bit two's-complement nibbles, low nibble first in the byte stream. Do not offset-by-7 decode.
- FP8 outlier weights: raw `e4m3` bytes, one byte per code.
- MHA internal `in_proj_weight` uses compact block-diagonal storage `[12,64,64]`, 24,576 B. Do not use dense `[768,256]`.
- Decoder pointwise tensors are already grouped (`groups_x=4`, `groups_skip=2`) in the exported storage.

Important offsets:

| Tensor | Format | Offset | Length |
|---|---|---:|---:|
| `mattn.input_proj/fc_r` | INT4 | 29,056 | 32,768 |
| `mattn.input_proj/fc_i` | INT4 | 61,824 | 32,768 |
| `mattn.mattn.MultiheadAttention_re/in_proj_weight` | INT4 compact | 94,592 | 24,576 |
| `mattn.mattn.MultiheadAttention_re/out_proj` | INT4 | 119,168 | 32,768 |
| `mattn.output_proj/fc_r` | INT4 | 151,936 | 32,768 |
| `mattn.output_proj/fc_i` | INT4 | 184,704 | 32,768 |
| `decoder.model.0.0.pointwise_r_x` | FP8 e4m3 | 217,472 | 4,096 |
| `decoder.model.0.0.pointwise_i_x` | FP8 e4m3 | 221,568 | 4,096 |
| `decoder.model.0.0.pointwise_r_skip` | FP8 e4m3 | 225,664 | 4,096 |
| `decoder.model.0.0.pointwise_i_skip` | FP8 e4m3 | 229,760 | 4,096 |

The complete 66-weight table is in `memory_map_pkg.svh`; do not manually transcribe all offsets into RTL.

---

## 5. Constants locations and decode

Each constants block has:

```verilog
localparam int C_<NAME>_OFF   = ...;
localparam int C_<NAME>_BYTES = ...;
localparam int C_<NAME>_DEC   = ...;
```

Decode enum:

| `DEC` | Meaning | Used for |
|---:|---|---|
| 0 | `apot_6a` | weight APoT scale, BN APoT scale |
| 1 | `apot_6b` | PReLU pow2 slope |
| 2 | `fp16` | BN bias |
| 3 | `u8_plus32` | activation block-float exponent `k` |

### APoT 6a code

12-bit packed code:

```text
[11:7] a_exp + 24
[6:2]  b_exp + 24
[1]    has_second
[0]    residual_sign, 1 means negative residual term
```

Use for weight scale and BN scale.

### PReLU 6b code

12-bit packed code:

```text
[11:7] k_exp + 24
[6:2]  zero
[1]    has_second = 0
[0]    overall_sign, 1 means negative slope
```

This is intentionally different from 6a. Do not gate PReLU sign by `has_second`.

### Activation k table

Stored as one byte per activation boundary:

```text
stored_u8 = k + 32
k = stored_u8 - 32
```

---

## 6. PPU const prefetch

Runtime uses `--prefetch subop`.

```text
max prefetch unit: 768 B
PPU local const cache: 2048 B
double-buffer cache recommendation: 4096 B
```

Before each sub-op, copy the needed constants block from WBUF constants region into the PPU local const cache. During PE MAC, PE reads WBUF weight region and PPU reads only local cache.

`2048 B` is the capacity of one PPU const cache bank and is sufficient for the largest single sub-op const block. If the controller must prefetch next constants while the PPU is still consuming current constants, use `2 * 2048 B = 4096 B` as a ping-pong cache: one side feeds the PPU, while the other side is filled from WBUF. Without ping-pong, the controller must pause at sub-op boundaries to prefetch and must not overlap "prefetch next block" with "PPU reads current block".

Largest prefetch blocks:

| Unit | Offset | Bytes | Decode |
|---|---:|---:|---|
| `encoder.model.5.0.pointwise_r:weight_scale` | 243,540 | 768 | APoT 6a |
| `encoder.model.5.0.pointwise_i:weight_scale` | 244,308 | 768 | APoT 6a |
| `encoder.model.4.0:bn_bias` | 248,452 | 512 | fp16 |
| `encoder.model.5.0:bn_bias` | 249,351 | 512 | fp16 |
| `decoder.model.0.0:bn_bias` | 250,250 | 512 | fp16 |

BN scale and BN bias are separate prefetch units; the peak is therefore 768 B, not 896 B.

---

## 7. RTL integration flow

1. Include `sram_config.vh` for SRAM dimensions.
2. Include `memory_map_pkg.svh` for weight and constants offsets.
3. Split `wbuf_unified_init.bin` into WBUF bank init words using the mapping in §3.
4. Initialize KVBUF / UFBUF / SKIPBUF using the zero-init files and their mapping in §3.
5. During each op, read the weight tensor from `W_<NAME>_OFF/LEN`.
6. Prefetch constants from `C_<NAME>_OFF/BYTES/DEC` into PPU local const cache before the corresponding sub-op.
7. Decode constants according to `DEC`.
8. Run RTL one-frame TB and compare against the existing golden / TB pattern.

Regenerate images:

```bash
cd /raid/R12K41024/18_w4a8_pow2_mixed_lmul_hw_handoff_20260530
conda activate bcctn
python software_to_hardware_18/build_sram_images.py --scale-pack 12bit --prefetch subop
python software_to_hardware_18/verify_sram_image_standalone.py
```

Standalone verification result:

```text
[standalone-verify] 168 block checks over 66 weights + 103 const blocks
[standalone-verify] RESULT: PASS  (0 failures)
```

---

## 8. Do not change these assumptions

- Do not load debug `.fp32`, `effective`, or dense MHA files into WBUF.
- Do not re-pack INT4 nibbles while banking; preserve byte stream exactly.
- Do not treat PReLU 6b sign as APoT residual sign.
- Do not make the PPU read WBUF during the PE MAC phase; use the local const cache.
- Do not reserve WBUF space for offset tables. Offsets are metadata in `memory_map.json` / `memory_map_pkg.svh`.

