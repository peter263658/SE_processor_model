# Deliverables for the HW advisor — proj fused scale + bias (§3 done, 2026-06-08)

Answers the exporter request `EXPORTER_REQUEST_PROJ_FUSED_2026-06-08.md`. §1 (k-fused
split-k weight scale) and §2 (proj bias β) are generated, verified (R²=1), and §3
**patched into the deploy SRAM image**. Hand the advisor the files below.

## 1. Patched SRAM image + memory map  ← the main thing

`hw_patterns_final/sram_images/`
- `wbuf_unified_init.bin` — patched (256 KB, size unchanged). New const data appended
  in the previously-free slack `[251680, 253920)`; **all pre-existing bytes untouched**.
- `memory_map.json` — patched: +6 `const_blocks`, new region `proj_fused`,
  `wbuf_used_bytes 251677→253920`, `wbuf_slack_bytes 10467→8224`.
- Backups: `*.pre_projfused.bak` (restore to undo).
- `memory_map_pkg.svh` is **NOT** regenerated — regenerate it from the patched JSON
  with your packer (only new symbols were added; existing ones are unchanged).

### New const_blocks (concrete offsets)

| dotted | kind | offset | bytes | decode / layout |
|---|---|---|---|---|
| mattn.input_proj/fc_r | weight_scale_splitk | 251680 | 48 | apot_6a 12-bit, `kreal[16]‖kimag[16]` (k=−7/−6) |
| mattn.input_proj/fc_i | weight_scale_splitk | 251728 | 48 | apot_6a 12-bit, `kreal[16]‖kimag[16]` |
| mattn.output_proj/fc_r | weight_scale_splitk | 251776 | 48 | apot_6a 12-bit (k=−3/−3) |
| mattn.output_proj/fc_i | weight_scale_splitk | 251824 | 48 | apot_6a 12-bit |
| mattn.input_proj | bn_bias (proj β) | 251872 | 1024 | fp16, `beta_re[256]‖beta_im[256]` |
| mattn.output_proj | bn_bias (proj β) | 252896 | 1024 | fp16, `beta_re[256]‖beta_im[256]` |

12-bit packing = contiguous, **LSB-first** (verified bit-exact against the existing
image blocks). `weight_scale` raw-16 proj blocks are left in place but tagged
`superseded_by: weight_scale_splitk` — **do not use the raw ones** (no act-k → off by 2^k).

## 2. Standalone code/bias artifacts (inspection / re-export source)

`hw_patterns_final/proj_fused_codes/` (+ same files dropped additively into
`tb_pattern/weights/<layer>/`):
- `<layer>/<half>/scale_apot_{a,b}_exp_fused_act_{kreal,kimag}.int16` — fused exponents
  (decoder-identical naming).
- `<layer>/<half>/scale_apot_code_fused_act_{kreal,kimag}.u16` and
  `scale_apot_code_splitk32.u16` — packed 12-bit codes (the second is the 32-code
  `kreal‖kimag` block, ready to drop into a const_block).
- `<layer>/bn_bias_{r,i}_fused.fp16` — β = (b_r, b_i).
- `README.md`, `manifest.json`.

## 3. CE subop config for input_proj / output_proj

```
pe_mode        = W4A8
ppu_mode       = PPU_CONV          (β-add path)
  alpha        = identity 2^0      (apot_6a code 0xC60; weight scale is in the dealer)
  prelu        = identity (k=0,+)  (proj has no activation)
  beta         = proj β (plain add: β_re→real acc, β_im→imag acc; NOT b_r∓b_i)
  cast_k       = consumer's block-float exponent (see §5)
has_bn         = 1
branches_collapse = 0              (split-k; output_proj k_real==k_imag so its two halves are equal)
scale index    = per-head, head = oc // 16, 16 codes / half
```
Per-PEG dealer code (matches `dealer.sv` + `CE_DATAPATH §2.1`):
`P0=fc_r·x_real→fc_r kreal[head]`, `P1=fc_i·x_imag→fc_i kimag[head]`,
`P2=fc_r·x_imag→fc_r kimag[head]`, `P3=fc_i·x_real→fc_i kreal[head]`. preshift=0.

## 4. Verification

`scripts/patch_memmap_proj_fused.py` reads back the **patched image** and reconstructs
`out = fused_descale·Σ(int4·e4m3) + β` (complex) vs the fp32 golden:

| layer | R² | max_abs |
|---|---|---|
| mattn.input_proj | **1.00000000** | 1.2e-4 |
| mattn.output_proj | **1.00000000** | 2.4e-4 |

(neg-controls earlier: raw apot no-k → R²=−7500 / −49.8; drop bias → 0.987 / 0.999999.)

## 5. Consumer cast-k (for the PPU e4m3 cast on the proj OUTPUT) — pinned 2026-06-09

**Sign convention:** the consumer stores the proj output as block-float
`value = e4m3_code · 2^(k_cons)`. The PPU cast does `e4m3 = round(fp16 · 2^(i_cast_k))`,
so **`i_cast_k = −k_cons`** (NOT k_cons). Both consumer exponents are measured EXACT
(e4m3_code·2^k == fp32 golden, maxabs=0), so this is deploy-cross-verified, not tentative:

| proj | consumer | consumer k (block-float) | **PPU i_cast_k** |
|---|---|---|---|
| input_proj | `mha_in_proj` (`mattn.mattn.activation_quantizer`, amax 11.45) | −5 | **+5** |
| output_proj | decoder (after `chunk(2)` → L/R 128ch each; `decoder.model.0.0.pointwise_x`, amax 576/704) | **+1** | **−1** |

- input_proj **i_cast_k = +5** (single k; agrees with the advisor's +5).
- output_proj **i_cast_k = −1** — **corrects the tentative −2.** The MHA output is
  `chunk(2,dim=1)` into L/R (128ch each) and fed to the shared decoder; the decoder's
  first op (`pointwise_x`) quantizes that input at **k=+1** (verified bit-exact, both
  real & imag), so i_cast_k = −1. (The −2 came from a meta `descale_pow2_exp` not
  cross-checked against the activation export.)
- Alternative: if the cast is deferred to the consumer's own input-quant, the proj PPU
  just does β-add + PASS fp16 and no cast_k is needed.

## Reproduce / undo
```bash
cd /raid/R12K41024/19_algo_final
conda run --no-capture-output -n bcctn python scripts/gen_proj_fused_codes_and_verify.py  # regen codes + bias
conda run --no-capture-output -n bcctn python scripts/patch_memmap_proj_fused.py          # patch image + verify
# undo: restore hw_patterns_final/sram_images/*.pre_projfused.bak
```
