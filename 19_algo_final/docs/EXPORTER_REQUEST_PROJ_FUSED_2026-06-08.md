# Exporter request — mattn.input_proj / output_proj fused scale + bias (2026-06-08)

The HW contract for `mattn.input_proj` and `mattn.output_proj` is fully verified (R²=1, matmul abs
sd=2.8e-7). To run them on the deploy SRAM image, the exporter needs to add **two** things to the
image for each of these two layers (each has `fc_r` + `fc_i`). Today the image ships only the RAW
per-head apot (16 codes) and no proj bias in a const slot — that's the gap.

Convention is IDENTICAL to the decoder split-k layers (which already ship correctly), just applied to
the per-head proj scale. Per-PEG fuse mapping (matches dealer.sv + CE_DATAPATH §2.1):
`P0=fc_r·x_real, P1=fc_i·x_imag, P2=fc_r·x_imag, P3=fc_i·x_real`.

## 1. k-fused per-head split-k WEIGHT_SCALE codes (replace the raw-only blocks)

For each of `fc_r`, `fc_i`: emit a **2×16 = 32-code** block = `kreal[16] || kimag[16]` (apot_6a 12-bit,
same packing as decoder `scale_apot_*_fused_act_kreal/kimag`):

- `kreal[head] = apot6a( a_exp + k_real, b_exp + k_real, has_second, sign )`  (fuse k_real into per-head apot)
- `kimag[head] = apot6a( a_exp + k_imag, b_exp + k_imag, has_second, sign )`  (fuse k_imag)
- `a_exp/b_exp/has_second/sign` = the existing RAW per-head apot for that head; only the two exponents
  shift by k (has_second & sign unchanged). 16 heads → head = oc // 16.
- `k_real / k_imag` = the real / imag **activation block-float exponents** of that layer's input:

  | layer        | k_real | k_imag |
  |--------------|--------|--------|
  | input_proj   | −7     | −6     |
  | output_proj  | −3     | −3     |

  (output_proj split-k degenerates since k_real==k_imag, but still emit both 16-code halves for uniform
  control.) Sanity: input_proj raw a_exp=−4 → kreal a_exp=−11, kimag a_exp=−10.

- The dealer reads, per the split (`branches_collapse=0`): PEG0 ← `ws_r[head]` (kreal half), PEG2 ←
  `ws_r[head+16]` (kimag half); PEG1 ← `ws_i[head+16]` (kimag), PEG3 ← `ws_i[head]` (kreal).
- preshift = 0 (proj is overflow-safe, ≤0.17× of fp16 max), so the dealer carries the FULL fused
  descale, no residual.

## 2. proj bias β in a const slot (per-oc, fp16)

- `β_re = fc_r/bias.fp16` (256), `β_im = fc_i/bias.fp16` (256) — already exist in the weight dirs,
  just not placed in an image const block. Place them in a **BN-β const slot** so the PPU β-add reads
  them. **Plain add** (β_re to the real accumulator, β_im to the imag): NOT the conv `b_r ∓ b_i` fold.
- bias is significant (input ≈0.46, output ≈0.64 abs) — must not be dropped.

## 3. memory_map / manifest / subop ROM

- Add the fused-code const_blocks (32 codes / branch, `decode=apot_6a`, `pack=12bit`) + the bias
  const_blocks, with offsets, to `memory_map.json` (`const_blocks`).
- The CE subop for input_proj/output_proj should then be: `pe_mode=W4A8`, `ppu_mode=PPU_CONV`
  (β-add path, α=identity 2^0, PReLU k=0, β=bias, cast k = consumer's block-float exp),
  `has_bn=1`, `branches_collapse=0` (split-k), scale index = per-head (oc//16, 16 codes/half).
  `gen_ce_subop_rom.py` will pick these up from the const_blocks once present.

## Consumer cast exponents (for the PPU e4m3 cast k)
- input_proj output → consumed as e4m3 by `mattn.mha_in_proj` (use its input block-float exp).
- output_proj output → consumed as e4m3 by the decoder (its input block-float exp).

Once the image has (1)+(2), the RTL bring-up is just wiring (per-head dealer, split-k, PPU β, complex
P0–P3 are all already implemented + verified on conv/decoder). We are verifying locally in the
meantime by patching the behavioral image with exactly these codes.
