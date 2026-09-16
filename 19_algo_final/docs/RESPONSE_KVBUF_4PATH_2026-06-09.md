# Response — KVBUF 4-path pin + addressing contract (2026-06-09)

Answers `CONSULT_KVBUF_4PATH_2026-06-09.md`. All 4 items resolved; verified against the
deploy capture.

## Item 1 — 4-path PINNED into memory_map.json ✅

`hw_patterns_final/sram_images/memory_map.json` now has a top-level **`kvbuf`** block
(`scripts/patch_memmap_kvbuf_4path.py`, non-destructive `.pre_kvbuf.bak`). Regen
`memory_map_pkg.svh` from it on your side.

| path | rows | byte offset | bytes | descale |
|---|---|---|---|---|
| K_re | 0..255 | 0 | 8192 | ×2⁻⁵ |
| K_im | 256..511 | 8192 | 8192 | ×2⁻⁵ |
| V_re | 512..767 | 16384 | 8192 | ×2⁻⁵ |
| V_im | 768..1023 | 24576 | 8192 | ×2⁻⁵ |

4 × 8192 = **32768 B = KVBUF capacity (zero slack)**. `kvbuf_init_zero.bin` = 32768 B,
all-zero — **no resize**, the 4 regions above are just its partitioning. (e4m3 1-byte is
what makes it fit; fp16 would be 64 KiB.)

## Item 2 — addressing contract CONFIRMED ✅

- in_proj write address = `{path, head, frame_slot, dim}`,
  `byte = path_base + head*512 + frame_slot*16 + dim` (head_stride 512, frame 16, dim 1;
  16 heads × 32 frames × 16 dim e4m3 = 8192 B/path).
- **pass A (x.real) → writes K_re, V_re ; pass B (x.imag) → writes K_im, V_im.**
  QK pass A reads K_re / pass B reads K_im; score_v pass A reads V_re / pass B reads V_im.
  ✅ correct.
- **descale on read: K ×2⁻⁵, V ×2⁻⁵, pass-INDEPENDENT** — CORRECTED 2026-06-09: the
  deploy uses a SHARED block-float exponent k=−5 for Q/K/V (`mattn.mattn.activation_quantizer`,
  calib amax 11.45 → −5). Full-seq |K|max=7.5, |V|max=13 OVERFLOW e4m3 at −7/−6; exact only at −5. Only the K/V **data** is pass-split, not the
  scaling. (Earlier "−7/−6" was a find_k artifact on frame-351 small values — wrong for the full tensor.) K_re vs K_im differ **98.6%** of entries → single region genuinely can't serve
  both (matches your ~99%). ✅ confirmed.

## Item 3 — ring frame indexing CONFIRMED ✅

32-frame streaming ring (NOT the 63-frame chunk window):
- **write:** frame t → `ring[ptr]` (all heads share `ptr`).
- **read:** window entry `local_idx d` → physical slot `(ptr − d) mod 32`. `local_idx 0`
  = current/newest = `ptr`.
- **advance:** `ptr = (ptr + 1) mod 32` AFTER frame t's QK + score_v complete.

Verified (`scripts/gen_kv_ring_reference.py`) on the t*=351 window (ptr=31): the map
`frame 320→slot0/local_idx31 … frame 351→slot31/local_idx0` is self-consistent, and
`K_block[31−d] == qstar_Kwindow[d]` across **all 16 heads** — i.e. `(ptr−d) mod 32` picks
exactly the right historical frame.

## Item 4 — verification level + ring reference DELIVERED

The single-frame per-pass golden verifies the **window-read numerics** (QK over the
32-frame history) but NOT the ring **pointer/wraparound**. So: `hw_patterns_final/kv_ring_reference/`:
- `kv_ring_reference.json` — the frame↔write_slot↔local_idx map + read/write/advance formula.
- `K_re_head00_ring_e4m3.{hex,u8}`, `V_re_head00_ring_e4m3.{hex,u8}` — K/V e4m3 ring codes
  in **write/slot order** (slot0=frame320 … slot31=frame351), head 0.

**Multi-frame ring test:** stream-write frames 320..351 into `ring[0..31]` (ptr 0→31);
then at frame 351 (ptr=31), read `local_idx d → ring[(31−d) mod 32]`, run QK, and compare
against `mha_module_io/real_pass/02_qk/head_HH/qstar_scores_row32_localidx_postscale_*`
(QK at query 351, local_idx order). If your ptr/wraparound is right, it matches; this
exercises the ring pointer that the single-frame TB-injected test never touched. (All 16
heads' fp16 K/V windows are in `mha_module_io/02_qk/`; I dumped head-0 e4m3 ring codes as
the worked example — say the word for all 16 / both passes as ring-order e4m3.)

## Summary
- Item 1 (pin) — done in `memory_map.json['kvbuf']`; regen `memory_map_pkg.svh`.
- Items 2,3 — confirmed (pass→path split correct; descale pass-independent; ring formula
  `(ptr−local_idx) mod 32`, advance-after).
- Item 4 — ring reference + multi-frame test recipe delivered.
