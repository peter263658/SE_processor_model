# Relay packet — MHA chain blockers (2026-06-09)

The live attention **compute** chain (in_proj → QK → softmax → score_v) is RTL-verified bit-exact
(handoffs #1–#4b). The remaining links (out_proj → recombine → output_proj) and the output stage
(mask-apply) are blocked on 4 items below. Each has: context · the ask · what RTL needs back.
Routing: **[ALGO]** = algorithm consultant · **[EXPORTER]** = the `software_to_hardware_18` export tooling ·
**[STFT]** = the STFT/FFT datapath owner.

---

## 1. [EXPORTER] out_proj bias is unmapped (sub-op 34, W4A16)

**Context.** `gen_golden_mha_out_proj.py:68` adds a per-channel FP16 bias post-OACC
(`out = fp16_add_sat(oacc, bias)`, from `…/out_proj/bias.fp16.bin`). But the ROM descriptor
`ce_subop_rom.svh:50` (sub-op 34) sets `ppu_mode = PPU_PASS, has_bn = 0`, with **no bias const pointer**.
So the bias exists in the golden but maps to no PPU op / const region — the RTL has nowhere to read it from.

**Ask.** Map the out_proj bias into the PPU β path, exactly like `mha_in_proj` already does (which is
`bias=True` and emits an in_proj `bias.fp16.bin` + a β feed). Concretely for sub-op 34: set a bias-adding
PPU mode (β-add on the PASS path, α=identity) and emit the bias const-block offset in the descriptor +
`memory_map`. Both `_real` and `_imag` passes need it (only `_real` golden exists on disk today;
`gen_golden_mha_out_proj.py` supports the `imag` arg but it hasn't been emitted).

**What RTL needs back.** Updated `ce_subop_rom` descriptor for sub-op 34 with the bias pointer + the
emitted `mha_out_proj_{real,imag}` bias const, so out_proj-live adds bias via the existing β path (no new
adder, no ROM hand-edit). (The MAC itself is already RTL-verified 256/256.)

---

## 2. [ALGO] output_proj cast_k = −1 or −2? (sub-op 35, outer GroupedLinear)

**Context.** output_proj PPU output e4m3-casts: `out_code = fp16_to_e4m3(fp16_scale2(oacc+β, cast_k))`,
with `cast_k = −(decoder-stage-0 consumer block-float exponent)`. There are **three conflicting values**:
- on-disk golden `golden/mattn_output_proj__0/cast_k.txt` = **−2** (written 2026-06-08 22:47),
- current script `gen_golden_proj.py:201` dict = **−1** (edited 2026-06-09 02:08, AFTER the golden → the
  on-disk `out_{re,im}_code.hex` is **stale** vs the script),
- script comment `:200`: deploy meta `descale_pow2_exp = +2` would give cast_k=−2, but notes it "was NOT
  cross-checked".
They differ by a **2× block-float shift**, so the e4m3 output codes (which feed decoder stage-0) won't
match a script rerun until pinned.

**Ask.** Confirm the correct `cast_k` for output_proj = −(decoder stage-0 **input** block-float exponent).
Is it **−1** or **−2**? (Measured against the actual dec0 consumer, like the input_proj cast_k=+5 /
output_proj earlier −1 measurement.)

**What RTL needs back.** The confirmed value → we regen `mattn_output_proj__0/out_{re,im}_code.hex` and use
it as the RTL match target. (`oacc_{re,im}_t0.hex`, `beta_{re,im}.hex` are cast_k-independent, already
usable — the MAC + bias path can be built now; only the final e4m3 output codes are gated.)

---

## 3. [ALGO] mask-apply: dump the fp16-mask masked-spectrum golden

**Context.** `RESPONSE_MASK_APPLY_NUMERICS_2026-06-09.md` answered the numerics (mask = **fp16** not e4m3;
spec = **Q2.14**; mode-C `real=sr·mr−si·mi, imag=sr·mi+si·mr`, b=0..63; F.pad DC bin0=0). The RTL bit-exact
target is the **fp16-mask** masked spectrum (not fp32 deploy). The offered golden
`hw_patterns_final/mask_apply_golden/mask_apply_golden.json` **does not exist on disk yet**.

**Ask.** Please dump the fp16-mask masked-spectrum golden, per (bin 0..63, frame), as **hex codes**:
`spec_re/im` (Q2.14 16-bit code), `mask_re/im` (fp16 16-bit code), `masked_re/im` (the fp16-mask result,
Q2.14 code). Plus state the **exact rounding mode** the reference uses for fp16→Q2.14 (round-half-to-even
vs round-half-up) so the RTL `fp16_to_q214_sat` matches bit-exactly. (Full 64-bin × N-frame, like the
softmax_stepwise dump — the worked frame-38/bins-1..8 in the response is enough to prototype but not to
regression.)

**Also note (prerequisite glue).** The mask must reach mask-apply at fp16, so the **last decoder layer
(`decoder.model.5.0.pointwise_x`, sub-ops 28/29) must emit fp16 (PASS, NO e4m3 cast)** — currently those
ROM entries are `PPU_CONV` (which would e4m3-cast). Confirm the last-layer output should be fp16-PASS for
the mask path (e4m3 mask → 1.9% RMS, ~60× the iSTFT budget, per the response table).

**What RTL needs back.** The golden JSON + rounding mode → mask_apply RTL is fully specced (fp16×Q2.14, two
helper modules) and can be built + bit-exact-verified.

---

## 4. [STFT] Confirm the spec / iSTFT-input / mask-apply bus is Q2.14 (not Q1.15)

**Context.** The consultant flagged `|spec|max = 1.46`, `|masked| = 1.38` — both **exceed ±1 and clip in
Q1.15**. But `Param_def.sv` declares `FX_FRAC=15, FX_INT=1` (Q1.15); `Hamming.sv:35` comments "Q2.14
input", `istft_reorder.sv` comments "Q1.15". So the FFT bus convention is inconsistent with the data range
on the mask-apply / iSTFT-input path.

**Ask.** Confirm the STFT-output → mask-apply → iSTFT-input bus is **Q2.14** (range [−2,2), step 2⁻¹⁴), not
Q1.15. If the STFT RTL currently emits Q1.15, it needs to carry Q2.14 on this path (or the bus be
reinterpreted as Q2.14 with the Hamming/FFT scaling adjusted).

**What RTL needs back.** Q-format confirmation → sets the mask_apply helper `FXP_FRAC = 14` (vs 15). This is
a one-parameter flip but must match the STFT side to avoid a 2× scale error into iSTFT.

---

## Status / what unblocks what
- **#1 (bias map)** unblocks **out_proj-live (#5)** end-to-end (the W4A16 ctx-delivery operand path is RTL
  work I can do once the bias path is defined). **#6 recombine** (`(a−b), (a+b)` FP16 adds) follows.
- **#2 (cast_k)** unblocks **output_proj (#7)** golden match (MAC+bias buildable now; only e4m3 output gated).
- **#3 + #4 (mask golden + Q2.14)** unblock **mask-apply** bit-exact verification (the unit is fully specced).
