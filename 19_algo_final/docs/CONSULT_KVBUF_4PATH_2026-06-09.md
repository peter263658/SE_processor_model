# Consultant ask — KVBUF 4-path pin + addressing contract (2026-06-09)

Context: tied-MHA #6/#7 are LIVE + bit-exact. The live 2-pass needs **per-pass K/V history** —
measured: `qk_real` vs `qk_imag` K/V histories are ~99% different (all 32 frames), so a single
KVBUF region cannot serve both passes. You already confirmed the **4-path KV (K_re/K_im/V_re/V_im)
fits 32 KiB exactly** (e4m3 1-byte; 4 × 256 rows × 32 B = 1024 rows = KVBUF_DEPTH). For #6/#7 the
TB currently INJECTS the imag history at the pass A→B boundary to model this; to make it real I need
the items below. (Tracked as RTL task #4, BLOCKED on item 1.)

## What I need from you (algorithm/exporter side)

1. **PIN the 4-path layout into `memory_map.json`** (same as you did for the proj/bias consts), so my
   ROM/golden generators + RTL reference symbols, not hand-transcribed offsets. Proposed (your message):
   | path  | rows      | byte offset | bytes |
   |-------|-----------|-------------|-------|
   | K_re  | 0..255    | 0           | 8192  |
   | K_im  | 256..511  | 8192        | 8192  |
   | V_re  | 512..767  | 16384       | 8192  |
   | V_im  | 768..1023 | 24576       | 8192  |
   Please emit region names + offsets (KVBUF) in `memory_map.json` (+ `memory_map_pkg.svh` if you
   regen it) and confirm `kvbuf_init_zero.bin` (32768 B) partitioning matches (no resize).

2. **Confirm the per-pass write/read addressing contract**:
   - in_proj write address = `{path, head, frame, dim}`, path ∈ {K_re,K_im,V_re,V_im}.
   - pass A (x.real) writes **K_re, V_re**; pass B (x.imag) writes **K_im, V_im** (pass bit selects
     the re/im half). QK pass A reads K_re, pass B reads K_im; score_v pass A reads V_re, pass B V_im.
   - descale on read: K ×2⁻⁷, V ×2⁻⁶ (pass-independent — already verified). Confirm this split is right.

3. **Confirm the ring-buffer frame indexing** (you flagged: 32-frame streaming ring, NOT the 63-frame
   chunk window). For the live multi-frame HW I need the exact mapping: each new frame writes the
   current K/V to `ring[ptr]`; QK/score_v read window entry `local_idx` from physical frame
   `(ptr − local_idx) mod 32`? (current frame = local_idx 0 is already pinned.) Please confirm the
   ptr advance + local_idx→physical formula so the window read picks the right historical frames.

4. **Verification level**: is the existing per-pass golden (`qk_{real,imag}` with their own 32-frame
   histories) sufficient to verify the 4-path addressing for ONE query frame? Or do you want a
   multi-frame ring golden (write N frames, then verify the window read at frame N)? For #6/#7 the
   single-frame per-pass goldens sufficed (TB-injected); the ring pointer itself is unverified until a
   multi-frame test exists — let me know if you can dump a small multi-frame ring reference.

Items 2–3 are the only RTL-facing unknowns; item 1 unblocks the implementation. cast_k is
pass-independent (verified k_Q=k_K=−7, k_V=−6), so only the K/V DATA is pass-split, not the scaling.
