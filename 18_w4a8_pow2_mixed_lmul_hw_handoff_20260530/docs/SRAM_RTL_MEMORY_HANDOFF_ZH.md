# SRAM / WBUF RTL Memory Handoff (中文)

日期: 2026-06-03

本文件是 strict-18 pow2 block-float deploy 的 SRAM / WBUF 匯入交接說明。完整數值來源是:

- `hw_patterns_final/sram_images/wbuf_unified_init.bin`
- `hw_patterns_final/sram_images/memory_map.json`
- `hw_patterns_final/sram_images/memory_map_pkg.svh`
- `hw_patterns_final/sram_images/sram_config.vh`
- `software_to_hardware_18/build_sram_images.py`
- `software_to_hardware_18/verify_sram_image_standalone.py`

`memory_map_pkg.svh` 是 RTL-friendly 的完整 offset table。若本文和 `memory_map.json` / `memory_map_pkg.svh` 有任何差異，以 `memory_map.json` 和 `memory_map_pkg.svh` 為準。

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

WBUF 是 single-port UHD SRAM，存整個模型的 weight code 和 packed constants。沒有獨立 CONSTBUF。constants 在 runtime 由 WBUF 預取到 PPU local const cache，PPU 在 MAC phase 不讀 WBUF，因此 PE 和 PPU 不搶 single-port WBUF。

---

## 2. WBUF flat image layout

WBUF image 是單一 flat byte stream:

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

`WBUF_CONST_BASE_ROW = 1888`, so `1888 * 128 = 241,664`, which is the first byte of the constants region.

注意: `memory_map.json` 的 top-level region key 只有 `weight_code`, `weight_apot_scale`, `bn_prelu`, `constants`。上表把 `activation_k` 獨立列出是為了 RTL 讀取方便；在 JSON 裡它是 `constants` region 內的一個 `const_block`。

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

If your `$readmemh` / SRAM init expects a conventional 64-bit hex word with MSB on the left, write:

```verilog
word64 = { image[base+7], image[base+6], ..., image[base+0] };
```

This keeps `word64[7:0]` equal to the lowest byte address. Do not reinterpret or repack the payload while splitting banks.

不可變 byte-order rule:

```verilog
// little-endian inside each 64-bit bank word
// for i = 0..7:
//   word64[8*i +: 8] == image[base+i]
```

也就是說 `image[base+0]` 必須落在 `word64[7:0]`，`image[base+7]` 必須落在 `word64[63:56]`。不要額外 byte-swap。

### KVBUF / UFBUF / SKIPBUF

The zero-init files use the same flat-to-bank rule with their own row size:

| Buffer | Banks | Width | Row bytes | Address mapping |
|---|---:|---:|---:|---|
| KVBUF | 4 | 64b | 32 B | `row=A>>5`, `bank=(A>>3)&3`, `byte=A&7` |
| UFBUF | 8 | 64b | 64 B | `row=A>>6`, `bank=(A>>3)&7`, `byte=A&7` |
| SKIPBUF | 4 | 64b | 32 B | `row=A>>5`, `bank=(A>>3)&3`, `byte=A&7` |

The current init files are all zero:

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

`2048 B` 是單一 PPU const cache bank 的容量，足以容納最大的單一 sub-op const block。若 controller 要在 PPU 使用 current constants 的同時預取 next constants，請用 `2 * 2048 B = 4096 B` ping-pong cache；一邊供 PPU 讀取，一邊從 WBUF 預取下一個 block。若沒有 ping-pong，就必須在 sub-op 邊界停下來預取，不能把「預取下一個 block」和「PPU 讀當前 block」重疊。

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
- Do not make PPU read WBUF during PE MAC phase; use local const cache.
- Do not reserve WBUF space for offset tables. Offsets are metadata in `memory_map.json` / `memory_map_pkg.svh`.

