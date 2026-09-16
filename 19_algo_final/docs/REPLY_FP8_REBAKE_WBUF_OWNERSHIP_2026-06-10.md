# REPLY — FP8 preshift re-bake: wbuf packing facts + ownership split (2026-06-10)

RTL-side answers to `RESPONSE_FP16_PPU_EVAL_2026-06-10.md` questions #1/#2, plus two
same-day RTL findings that bear directly on the re-bake.

## Q1 — wbuf re-pack: YES the FP8 descale is already wired into WBUF; you own the re-pack

**There is no `fp8_descale_apot` region because none is needed.** The 12 outlier tensors'
descale codes are ordinary `const_blocks` entries of `kind: "weight_scale"` with
`wbuf_bytes: 3` (= **2 × 12-bit APoT split codes, `kreal || kimag`, LSB-first packing**)
in the R3/R4 const region — `build_sram_images.py` evidently already packs them, because
the shipped image contains them and the RTL reads them today:

| tensor (dotted)                          | wbuf_offset |
|------------------------------------------|-------------|
| decoder.model.0.0.pointwise_{r,i}_x      | 245268 / 245271 |
| decoder.model.0.0.pointwise_{r,i}_skip   | 245274 / 245277 |
| decoder.model.3.0.pointwise_{r,i}_x      | 247104 / 247107 |
| decoder.model.3.0.pointwise_{r,i}_skip   | 247110 / 247113 |
| decoder.model.4.0.pointwise_{r,i}_x      | 247164 / 247167 |
| decoder.model.4.0.pointwise_{r,i}_skip   | 247170 / 247173 |

(dec5 pointwise is the INT4 oc=1 special, not FP8 — its 3-byte blocks @247200+ are
plain per-tensor INT4 scales and are NOT part of the re-bake.)

The sub-op ROM rows already point at them (e.g. row 13 `dec0.pw_x`: `wsr=245268,
wsi=245271`, `SCALE_PERTENSOR`), and the per-sub-op const prefetch reads them like any
other weight_scale block. So:

- **You own the re-pack** (builder + ckpt + bins live in the source package; this repo
  cannot run `build_sram_images.py`). Re-pack = same builder run against the re-baked
  bins; the 12 blocks are overwritten **in place** — `a/b += |preshift|` stays inside the
  5-bit coded field (outlier APoT range [−16,−6] → [−13,−2]/[−12,−2] after +3/+4), so
  **sizes and offsets must come out identical**. Please confirm memory_map.json diffs are
  bytes-only (no offset churn); if the builder renumbers anything, flag it before handing
  off.
- **We own the RTL pickup**, all mechanical: descriptor `preshift_{r,i}` already exist in
  `subop_t` (currently 0) and the PEG lo-path already adds `preshift_ext` into both the
  INT4 and L-Mul product exponents (unit-tested: `07_run_peg_preshift`). On image+manifest
  delivery we (1) regen `ce_subop_rom.svh` with psr/psi from `preshift_manifest.json`,
  (2) add the preshift term to the `lmul_fp8` replica in the golden generators (they parse
  image+ROM directly, so everything downstream regenerates automatically), (3) rerun the
  full regression incl. the new full-frame e2e.

Deliverables we need: re-packed `wbuf_unified_init.bin`, (bytes-only-diff)
`memory_map.json`, `preshift_manifest.json`.

## Same-day RTL findings (full-frame closure, t0=351) — both strengthen the re-bake

1. **The preshift=0 overflow is LIVE in deploy data, not just headroom math.** The new
   full-frame e2e (real t=351, both ears, enc→attn→dec chained) hit it: dec4.pw_x
   (pos 1, oc 1), ic-chunk 0 — two opposite-sign **Inf** L-Mul products collide in the
   fp16 tree → DW_fp_add emits qNaN `0x7C01` → the APoT dealer (pure exponent shift)
   turns the NaN into finite garbage. We measured the DW semantics and made the python
   replica DW-true (NaN → `0x7C01`, sign dropped; Inf+finite → ±`0x7BFF`), so RTL ==
   replica even through the pathology — but the produced value is garbage either way.
   Your re-bake (post-fix max |node| ≤ 0.49×, 0 overflow) makes the whole NaN path
   unreachable. Please confirm the re-baked weights kill this exact event (dec4 t=351)
   — it's a nice witness case.
2. **`fp16_to_e4m3` subnormal fix (BUG-3, RTL + 5 replicas).** The cast flushed
   te=−3 (|x| ∈ [2⁻¹⁰, 2⁻⁹)) to zero instead of RNE-rounding up to the min subnormal —
   caught by the dec0 skip CAST (×2⁻¹) at t=351, 8 bytes. Fixed to true RNE = torch
   behavior, so your MBSTOI eval (torch casts) already covers the fixed RTL. No action
   needed on your side; FYI for your `verify_fp16_ppu_chain.py` reconstruction (one
   plausible contributor to your extra ~15-pt enc5 divergence).

## Q2 — what to drill into

Yes to your own suggestion, and it's the only quality gap left:
**per-decoder-stage fp16 WITH the re-baked preshift** — i.e. let the 12 FP8-outlier
multiplies go through the fp16 tree (instead of the fp32 bypass your injection kept) on
top of the re-baked weights, per stage. That's exactly the hardware datapath
post-re-bake, and it's the one segment MBSTOI hasn't certified yet. SNR breakdowns are
not needed — the headline + decoder-fp16-with-preshift closes everything we consume.
