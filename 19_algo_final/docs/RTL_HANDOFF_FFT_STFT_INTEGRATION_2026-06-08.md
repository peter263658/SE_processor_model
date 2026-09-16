# RTL_HANDOFF: STFT / FFT → e4m3 → UFBUF integration

Date: 2026-06-08
Base candidate: `strict_core_pow2_blockfloat_deploy_ep3` (model side); RTL in `19_algo_final/01_RTL/engine/FFT/`
Handoff chain: [RTL_FIX_FP16_TREE_PRESHIFT_2026-06-07.md](RTL_FIX_FP16_TREE_PRESHIFT_2026-06-07.md) (CE side) · model STFT = `code/DCNN/feature_extractors.py`

對象:把 STFT/FFT 接進 engine(UFBUF / controller)的人。先讀 §3 五個雷,那是接錯會 silent-wrong 的地方。

---

## 0. Dataflow

```
audio (Q2.14, i_dat_re/im, i_dat_en, i_new_frame@first sample)
  → Hamming        (symmetric Hamming window, Q1.15 out)            Hamming.sv
  → FFT128         (radix-2² SDF, ÷128 scaling, bit-reversed out)   FFT128.sv
  → Bit_Reorder    (natural order, bins 1..64, 64 parallel, Q1.15)  Bit_Reorder.sv
  → [o_dat_re/im (Q1.15, 64), o_dat_en]            ← existing Q1.15 spectrum output
  → stft_fp8_cast  (×√128 norm + ×2^i_k, → e4m3)   stft_fp8_cast.sv / fxp_to_e4m3.sv
  → [o_e4m3_re/im (8b, 64), o_e4m3_en]             ← block-float activation → UFBUF
```
All three new/changed cast files live in `01_RTL/engine/FFT/`. `STFT_TOP` now emits **both** the Q1.15 spectrum and the e4m3 block-float (cast wired in internally).

---

## 1. STFT_TOP interface

| port | dir | width | notes |
|---|---|---|---|
| `i_clk` / `i_rst_n` | in | 1 | **async active-low** (`posedge clk or negedge i_rst_n`) — matches CE |
| `i_stft_mode` | in | `STFT_mode` | STFT (ISTFT path = TODO) |
| `i_new_frame` | in | 1 | pulse on the **first sample** of a 128-sample frame |
| `i_dat_en` / `i_dat_re` / `i_dat_im` | in | 1 / 16 / 16 | input sample, **Q2.14** (per Hamming.sv) |
| `i_k` | in | signed 8 | **block-float exponent of the consumer** = **9** for encoder.model.0.0.depthwise input |
| `o_dat_en` | out | 1 | Q1.15 spectrum valid — **single-cycle** pulse |
| `o_dat_re` / `o_dat_im` | out | `[63:0][15:0]` | Q1.15, **64 bins, natural order** (see §3.1) |
| `o_e4m3_en` | out | 1 | e4m3 valid — **= o_dat_en delayed 1 cycle** (cast register) |
| `o_e4m3_re` / `o_e4m3_im` | out | `[63:0][7:0]` | block-float e4m3 → UFBUF |

Latency: input→`o_dat_en` ≈ FFT128 137 + window/reorder (frame-buffered; one pulse per frame). `o_e4m3_en` = `o_dat_en` + 1.

---

## 2. The cast (fxp_to_e4m3) — bit-exact reference

Reconciles HW vs model STFT scale **and** the block-float exponent in one shot. `i_k=9`:
```python
# Q1.15 HW STFT code c (= model_value / sqrt(128) in Q1.15)  ->  e4m3 code
#   e4m3( c * sqrt(128) * 2^k / 2^15 ) = e4m3( c * sqrt(2) * 2^(k-12) )
SQRT2_FP16 = 1.4140625                         # fp16 0x3DA8
f_int = fp16(c)                                # DW_fp_i2flt (int16->fp16, RNE)
f_rt2 = fp16(f_int * SQRT2_FP16)               # DW_fp_mult (fp16 RNE)  — the only real mult
f_sc  = fp16(f_rt2 * 2.0**(k-12))              # fp16_scale2 (exponent add)
e4m3  = fp16_to_fp8(f_sc)                       # RNE, saturate ±448  (common/utils/fp16_to_fp8.sv)
```
`sqrt(128) = 2^3.5 = 2^3 · √2`: the **2^3** folds into the `k-12` shift (`12 = 15 − 3`); only the **√2** is a real fp16 multiply (frame-rate, not in the MAC loop). `i_k` is a port so the cell serves other boundaries.

---

## 3. 五個雷(接錯會 silent-wrong)

### 3.1 ⭐ Bin convention = FFT bins **1..64** (DC dropped, Nyquist kept)
The model (`feature_extractors.Stft`) does `y.narrow(dim=-2, start=1, …)` → drops DC (bin0), keeps bins 1..64 (Nyquist bin64 included). `Bit_Reorder` matches: `bank[j] = FFT bin (j+1)`, so `o_*[0]` = bin1 (lowest non-DC), `o_*[63]` = bin64 (Nyquist). **UFBUF / encoder must index freq this way.** (Old code kept 0..63 = DC..bin63 — every bin was off, fixed 2026-06-07.)

### 3.2 ⭐ √128 normalization is in the cast, **not** the FFT
HW FFT128 = `÷128` (radix-2² `>>1`/stage). Model `torch.stft(normalized=True)` = `÷√128`. So **HW Q1.15 spectrum = model × 2⁻³·⁵ (≈ 1/11.3, NON-pow2)**. The cast restores ×√128. ⚠️ Do **not** feed the raw Q1.15 (`o_dat_re/im`) to anything expecting model scale — it's 11.3× small. Only the e4m3 output is at model scale.

### 3.3 ⭐ `i_k` is locked to 9 for the encoder; weight descale assumes it
e4m3 = `e4m3(model × 2^9)`. The encoder's fused weight-descale (`weight_APoT − k`) is **baked with k=9**. Drive `i_k=9`. A different k ⇒ every encoder conv output off by `2^(k−9)`.

### 3.4 ⭐ e4m3 output is **parallel 64-bin, single-cycle valid**
`o_e4m3_re/im` = 64 bins × 8b, valid only while `o_e4m3_en`=1 (one cycle). UFBUF write must capture all 64 (re+im) that cycle, or latch. Same for the Q1.15 path (`o_dat_en`).

### 3.5 ⭐ Single-frame FSM — **no hop / overlap yet**
`STFT_TOP` processes ONE 128-sample frame per `i_new_frame` (FSM IDLE→CACL→IDLE), no overlap. Model uses **hop=80** (overlapping 128-windows, 399 frames/sec of audio). Streaming integration must feed overlapping windows from an **input ring buffer (hop=80)** — that buffer is **not yet built**. (ISTFT/decoder reconstruction also TODO; see STFT_TOP header.)

---

## 4. Verification status

| test | run | what it checks | result |
|---|---|---|---|
| FFT core | `00_TB/FFT/01_run` (TB128) | FFT128 numerics, bit-reversed→reorder | (rerun on VCS) |
| Hamming | `00_TB/FFT/02_hamm` | window mult Q-format | (rerun on VCS) |
| **STFT top** | `00_TB/FFT/03_top` (tb_stft) | Q1.15 spectrum vs `golden_fft` **+ e4m3 vs cast(golden)** | **PASS** (VCS): Q1.15 4/4 frames 0 mismatch; e4m3 0 cast-bug, 9 tolerated FFT-rounding diffs |
| **STFT→e4m3 cast vs MODEL** | `00_TB/FFT/08_run_cast` (tb_fxp_to_e4m3) | cast vs real `encoder.0.0.depthwise_r` e4m3 golden (round-trip) | **PASS** (VCS); REAL errors (|q15|>2, >1 e4m3 step) = 0 |
| **ISTFT path** | `00_TB/FFT/09_run_istft` (tb_istft) | on-chip IFFT (conj_sym_feeder + shared FFT128 + istft_reorder) vs `real(IFFT(spec))`; window/OLA/norm off-chip | **PASS** (VCS): 512/512 samples, **max err 2 LSB** (= forward FFT128 fidelity) |

- `08` is the **model-tie**: it confirms √128 + k=9 reproduce the model's actual encoder input. Python pre-val: 25536 samples, exact 90.84%, off-by-1 5.67%, **real errors 0**.
- `03` is the **integration/numeric** check (HW-numeric, random input): STFT_TOP Q1.15 vs `golden_fft` (tol ≤2 LSB) and e4m3 vs `cast(golden_fft)` (tol ≤1 e4m3 step). The e4m3 golden uses the **same cast model that 08 validated in VCS** (verified identical in Python).
- **NOT tested end-to-end**: real-wav → STFT_TOP → cast → model encoder golden (needs hop + binaural + 399 frames). Covered in two halves: STFT numerics (03) + cast-vs-model (08).
- ⚠️ ~6% of bins land **±1 e4m3 step** vs the model (Q1.15-vs-float64 + fp16-√2, ~1e-4). Benign for a 3-bit activation; if you want it tighter, do the √2 in fp32. Confirm with a full-pipeline MBSTOI eval before tape-out.

Run (any VCS server with the 06/07/08 env): `cd 00_TB/FFT && ./03_top && ./08_run_cast && ./09_run_istft`. Patterns auto-regen via `tb_FFT.py` / `gen_cast_patterns.py` / `gen_istft_patterns.py`. **All three PASS on VCS (2026-06-08).**

---

## 5. Files this round

| file | change |
|---|---|
| `01_RTL/engine/FFT/fxp_to_e4m3.sv` | NEW — 1-element Q1.15→e4m3 cast (√128 + 2^k); DW i2flt + DW mult + fp16_scale2 + fp16_to_fp8 |
| `01_RTL/engine/FFT/stft_fp8_cast.sv` | NEW — 64-bin re/im wrapper (128 cast cells), registered, frame-rate |
| `01_RTL/engine/FFT/STFT_TOP.sv` | + `i_k` input, `o_e4m3_en/re/im` outputs, instantiates `stft_fp8_cast` (Q1.15 outputs unchanged) |
| `00_TB/FFT/tb_stft.sv` | + e4m3 checker (vs `golden_fft_e4m3`, tol ≤1 step), drives `i_k=9` |
| `00_TB/FFT/tb_fxp_to_e4m3.sv` + `gen_cast_patterns.py` + `cast_file.f` + `08_run_cast` | NEW — cast vs model encoder e4m3 golden |
| `00_TB/FFT/tb_FFT.py` | + generates `golden_fft_e4m3_{re,im}.hex` = cast(STFT golden, k=9) |
| `00_TB/FFT/{FFT,STFT,Hamm}_file.f`, `03_top` | path fix `01_RTL/FFT/`→`engine/FFT/`; DW `-y` → `/usr/cad/.../cur`; env + grep |

Prior FFT review fixes (bins, async reset, `i_clk/i_rst_n` rename, Bit_Reorder pulse, di_count): see commit / memory `project_bcctn_fft_stft_review_2026-06-07`.

---

## 6. Open items for integration

1. **Input ring buffer (hop=80)** feeding overlapping 128-windows → `i_new_frame`/`i_dat_*` (§3.5). Biggest missing piece for streaming.
2. **Wire `o_e4m3_re/im` (+`o_e4m3_en`) → UFBUF write**; capture all 64×(re,im) on the valid cycle.
3. **Binaural**: encoder is per-ear; STFT_TOP is single-channel. Two instances or time-multiplex (L/R share weights per ASIC spec).
4. ~~ISTFT~~ **DONE this round (§7)** — `i_stft_mode==ISTFT`, reuses FFT128 as IFFT. Still need: mask complex-mult upstream + off-chip OLA/normalization (§8).
5. **Two e4m3 casts coexist** (`common/utils/fp16_to_fp8` here vs `engine/CE/ce_fp_util.fp16_to_e4m3` in PE) — both RNE; consider unifying.
6. **Real SRAM macro** for any buffering (per the CE handoff's pre-synth reminder).

---

## 7. ISTFT mode (reuses FFT128 as IFFT) — added 2026-06-08

`IFFT(X) = conj(FFT128(conj(X)))`, and **FFT128's ÷128 IS the IFFT 1/N** (free). Output of a
real signal is real → take real, no output conj. Verified bit-exact vs `np.fft`
(`real(FFT128(emit))/128 == x − DC`, err 1.7e-16; DC is intentionally dropped by the model).

**KEPT / reused (ONE shared FFT128, no new big block):** FFT128 (the IFFT), plus
DelayBuffer/Twiddle/Butterfly inside it. FFT128 is **time-shared** between STFT
(Hamming→FFT) and ISTFT (feeder→IFFT) via `i_stft_mode` mux on its `di_*`. **The ISTFT
add-on is exactly TWO small modules** on top of that one FFT128 (confirmed: 1×FFT128,
1×conj_sym_feeder, 1×istft_reorder; the analysis Hamming is the only Hamming instance).

**ADDED (small):**
| file | role |
|---|---|
| `conj_sym_feeder.sv` | load 64 masked bins (1..64) → stream 128 conj-sym serial to FFT128 (DC=0; `bin(128−k)=conj(bin k)`; conj folded for the IFFT trick). Addressing verified bit-exact vs np.fft. |
| `istft_reorder.sv` | FFT128 serial out (bit-reversed time) → 128-buf bit-reverse reorder → take real → serial samples. **No window** (synth Hamming is OFF-CHIP — avoids a Q2.14-vs-Q1.15 mismatch and keeps the add-on to 2 modules). |
| `STFT_TOP.sv` | + ISTFT ports, `conj_sym_feeder`, FFT128 `di_*` mux, `istft_reorder`; STFT outputs gated off in ISTFT mode |

Output is **raw Q1.15 IFFT time samples** (pre-window, pre-OLA, pre-normalization).

**ISTFT interface (STFT_TOP, `i_stft_mode==ISTFT`):**
- in: `i_spec_re/im[63:0]` (masked bins 1..64, Q1.15), `i_istft_load` (latch), `i_istft_start` (stream 128).
- out: `o_istft_en` + `o_istft_sample` (serial, 128 synth-windowed Q1.15 time samples/frame).
- sequence: drive `i_istft_load` with the 64-bin masked spectrum → pulse `i_istft_start` → after FFT128 latency, `o_istft_en` streams 128 samples.

**Still off-chip / upstream (NOT in this block):** the mask complex-multiply, hop OLA, normalization (§8).

---

## 8. TB / off-chip responsibilities (so we don't forget what the TB must do)

The chip does **per-frame transforms only**. Framing, overlap-add, normalization, and the
mask multiply live in the TB / controller. Checklist:

### 8a. Forward STFT TB (analysis)
- Feed **overlapping 128-sample windows, hop=80** (frame m = `samples[m*80 : m*80+128]`), `i_new_frame` on the first sample of each — the **hop ring buffer is the TB's job, not on-chip**.
- First output only after 128 samples buffered (`center=False`, no padding).
- (verification: `03_top` already does this for 4 frames + checks Q1.15 and e4m3.)

### 8b. Mask multiply (between decoder and ISTFT)
- `masked = cspecs × mask` — **per-bin COMPLEX multiply** (config `masking_mode='C'`; mathematically same as 'E'; cartesian → no abs/angle/exp). `cspecs` = forward STFT (64 bins, Q1.15); `mask` = decoder output (complex, 64 bins). Done by CE or a small complex-mult; **feed the result into `i_spec_*`**. Confirm the mask's Q-format and that the product lands in Q1.15.
- Model pads DC=0 after masking (64→65); on-chip the feeder already inserts DC=0, so feed only bins 1..64.

### 8c. ISTFT TB (synthesis) — per ear
- For each frame: drive `i_spec_re/im` = the masked 64 bins (Q1.15), pulse `i_istft_load` then `i_istft_start`; capture the 128 serial `o_istft_sample` = **raw IFFT real time samples (Q1.15, NO window)**.
- **Synthesis window OFF-CHIP**: multiply each frame's 128 samples by the Hamming synth window (the chip no longer applies it — kept off-chip with OLA to avoid a Q-format mismatch).
- **Overlap-add (hop=80) OFF-CHIP**: `out[m*80 + n] += win_s[n] * frame_m[n]`, n=0..127.
- **Normalization OFF-CHIP**: divide each output sample by `Σ_m win²[...]` (the istft `normalized=True` window-energy norm). **Non-pow2 scaling trap** — analogous to the forward √128 (§3.2); must be applied here or the waveform amplitude is wrong.
- Directed transform check exists: `09_run_istft` (`tb_istft.sv`, `gen_istft_patterns.py`) feeds a known Q1.15 spectrum and checks `o_istft_sample` vs `real(IFFT(conj-sym spec))` (the on-chip IFFT only; window/OLA/norm not included). PASS = max err ≤ TOL(32 LSB); reports actual max.
- **Inverse √128 trap**: forward HW = model/√128 (cast restores it). For ISTFT the FFT128 ÷128 = IFFT 1/N; the model istft `normalized=True` = ÷√N + window-norm. Reconcile the leftover √N + window-norm **in the off-chip OLA**, consistent with the forward √128 — get these two consistent or the round-trip amplitude is off.
- Output length (`center=False`): `(frames−1)*80 + 128`.

### 8d. Binaural
- L and R are separate passes through the (weight-shared) datapath. TB/controller loops over 2 ears; STFT_TOP/ISTFT are channel-agnostic. Keep L/R separate in UFBUF and in the OLA output (two waveforms).

### 8e. End-to-end (optional, system-level)
- Real-wav → STFT → cast → CE → decoder → mask → ISTFT → OLA → waveform, compared at **MBSTOI** (not bit-exact e4m3 — fixed-point FFT vs float model diverges by more than the cast's ±1 step). The two-halves (`03_top` STFT numerics + `08_run_cast` cast-vs-model) already cover the forward path in pieces.
