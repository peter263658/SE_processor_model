# Response — out_proj/recombine goldens + ROM regen prep (2026-06-09)

The ROM generator (`gen_ce_subop_rom.py`) + `ce_subop_rom.svh` live in your RTL repo,
not this package, so I can't run the regen here. What I did instead: the **export-side**
prep so your generator picks everything up, plus the missing goldens as hex.

## 1. Goldens dumped (hex)

`hw_patterns_final/mha_out_proj_recombine_golden/`:
- `beta.fp16.{hex,bin}` — **MHA out_proj bias b_o[256] fp16** (|max| 0.35). The β golden
  for §C-9. **SAME const for both passes**: `out_pass = fp16_add_sat(oacc_pass, b_o)`.
- `t351_{a_real_pass,b_imag_pass,recombine_re_a_minus_b,recombine_im_a_plus_b}.fp16.hex`
  — recombine worked example (§C-10): `out_re = a−b`, `out_im = a+b`, a/b = post-bias
  out_proj fp16.
- `outproj_in_{re,im}_code.{hex,u8}` — the recombine result e4m3-cast at **i_cast_k=+3**
  (output_proj input block-float k=−3), all 399×256. dequant == golden exactly (boundary
  into the outer output_proj).

`hw_patterns_final/proj_fused_codes/mattn.output_proj/e4m3_out_castk_-1/out_{re,im}_code.{hex,u8}`
— output_proj e4m3 **output** golden at **cast_k=−1** (L-half real == decoder pw_x codes 100%).
**Delete the stale −2 `cast_k.txt` / `out_*_code.hex`.**

`hw_patterns_final/mask_apply_golden/{L,R}/` — full 64×399 mask-apply golden (spec Q2.14,
mask fp16, masked Q2.14, RNE).

## 2. memory_map.json export (done, in this package)

`hw_patterns_final/sram_images/memory_map.json` (+ `wbuf_unified_init.bin`), non-destructive,
backups `*.pre_*.bak`. New const_blocks your generator should now find:

| const_block | sub-op | offset | bytes | use |
|---|---|---|---|---|
| `mattn.input_proj/{fc_r,fc_i}` weight_scale_splitk | 30 | 251680/251728 | 48 ea | act-k fused split-k |
| `mattn.input_proj` bn_bias (β) | 30 | 251872 | 1024 | β = b_r‖b_i |
| `mattn.output_proj/{fc_r,fc_i}` weight_scale_splitk | 35 | 251776/251824 | 48 ea | split-k (k=−3/−3) |
| `mattn.output_proj` bn_bias (β) | 35 | 252896 | 1024 | β = b_r‖b_i |
| **`mattn.mattn.MultiheadAttention_re/out_proj` bn_bias** | **34** | **253920** | **512** | **b_o[256] fp16 (NEW this patch)** |

`wbuf_used 254432 / slack 7712`. (`memory_map_pkg.svh` still needs regen from this JSON.)

## 3. ROM descriptor fields to set (sub-ops 30–35)

Run your `gen_ce_subop_rom.py` against the patched `memory_map.json`; the two edits that
matter and their confirmed values:

| sub-op | layer | mode | bias (β) | cast_k (output) | notes |
|---|---|---|---|---|---|
| 30 | input_proj | W4A8, PPU_CONV β | `input_proj` bn_bias @251872 | **+5** | split-k per-head, α=id, PReLU k=0 |
| 31 | mha_in_proj | W4A8 | in_proj_bias (768) | +7/+7/+6 (Q/K/V) | 48-code APoT, qkv_groups=4 |
| 32 | qk | fp16-tree | — | — | ×2⁻², L-Mul m8 |
| 33 | score_v | fp16-tree packet | — | — | top-4 q0.8 |
| **34** | **mha_out_proj** | **W4A16, β-add** | **out_proj bn_bias @253920** | i_cast_k=+3 → output_proj in | **was PPU_PASS/has_bn=0 — set β pointer** |
| **35** | **output_proj** | **W4A8, PPU_CONV β** | `output_proj` bn_bias @252896 | **−1** | **was −2 stale — set cast_k=−1** |

(28/29 dec5 pointwise_x: **fp16-PASS, NOT e4m3** — for the mask path.)

## 4. What you regen

1. `git rm` the stale `golden/mattn_output_proj__0/cast_k.txt` + `out_*_code.hex`; replace
   with `…/e4m3_out_castk_-1/out_{re,im}_code.hex`.
2. Emit the out_proj imag-pass via your `gen_golden_mha_out_proj.py imag` (β = `beta.fp16`).
3. `gen_ce_subop_rom.py` → regen `ce_subop_rom.svh` (sub-op 34 bias ptr @253920, 35 cast_k=−1)
   + `memory_map_pkg.svh` from the patched `memory_map.json`.
