# Response — skip dataflow + frame orchestration (2026-06-10)

Answers `CONSULT_SKIP_DATAFLOW_2026-06-10.md`. Measured against the deploy ckpt/golden wav.

## A — rubber-stamp

**A1. skip = encoder-stage output, pure code copy (same k). ✓** Confirmed for s=1,2,5
(51072/51072). The mechanism is identical for all s, so s=3,4 hold too (their pw_skip
activation files sit under the FP8-stage naming; self-verify is fine). HW skip-write =
on write, select channels per `skip_indices`, store the original e4m3 code (compact,
file order = storage order).

**A2. dec0 skip = enc5 output, re-quantized to k=+1. ✓** Confirmed: `decoder.model.0.0.
pointwise_{r,i}_skip` input is **k=+1, value range ±4–5, exact recovery** — the SAME
tensor as the input_proj input (enc5 output), just stored at a different exponent. The
62~63/64 residual you saw is the **double-quantization** (you re-quantized from the
already-e4m3@−7 codes); a single quant from fp16 (strategy a) is exact. dec0 skip is NOT
a separate tensor.

**A3. mattn input 256 = binaural concat, L=0..127 / R=128..255. ✓** From
`binaural_attention_model.forward`: `enc_combined = cat([encoder_out_l[-1],
encoder_out_r[-1]], dim=1)` → **ear A (L) = ch 0..127, ear B (R) = ch 128..255**, L first.
The output `chunk(2, dim=1)` splits back in the same order (attn_l = 0..127).

## B

**B1. dec0 skip quant → use (a) pre-cast fp16 dual-cast.** The k=+1-vs-−7 is **per-consumer
calibration, NOT a bug**: input_proj has its own key (calib amax 3.29/3.55 → k=−7/−6); dec0
pw shares the **`split_pointwise_cat` calibration** (amax real 621 / imag 646 → **k=+1**),
applied to BOTH the x-branch and the skip-branch. The skip values (enc5, ±3.5–5) are stored
at the over-provisioned shared k=+1 — recovered **exactly** (err=0). **The "±896" is the
calibration amax (≈621·... → k=+1), not the actual values (±4–5).** So 2^8 vs input_proj is
expected, not a dump bug.
→ **The (a)-golden already exists:** the deploy's `decoder.model.0.0.pointwise_{r,i}_skip`
e4m3 export IS the single-quant (deploy quantizes the fp16 enc5 output directly to k=+1).
Use it as the dec0 skip-write target. Do NOT re-quantize from the attention's e4m3@−7 codes
(that's (b), adds the 1–3/64 boundary error).

**B2. `__N` semantics: __0=L(earA).real-input, __1=L.imag-input, __2=R(earB).real-input,
__3=R.imag-input.** `apply_complex` calls the real conv `dr` on `input.real` then
`input.imag` (each contributes to out.real / out.imag), per ear (L forward, then R) → 4
calls. All from the **same forward / same wav** as the attention capture. The attention
(`mha_module_io`) is the bottleneck `cat[L,R]` (not "a chunk"); **t*=351 is a time-frame
index** — all conv `__N` dumps have T=399, so their frame 351 aligns with the attention
t*=351. skip/conv golden uses `__0` (L.real) → same source.

**B3. dw per-position golden — DELIVERED.** `hw_patterns_final/dw_position_golden/{enc1,dec1,dec4}/`
(weight `Cx5`, bias `C`, input `CxFinxT`, output `CxFoutxT` hex + `dw_position_golden.json`
tap-map). Reproduce-from-tapmap == deploy layer_io (enc1 err 0, dec1/dec4 err ~1e-7):

- **(iii) encoder dw = Conv2d, stride 2, pad 2, k5, NO bias:** `out[c,q,t] = Σ_k w[c,k]·
  in[c, 2q−2+k, t]` (zero-pad). enc1: 8ch 32→16. e.g. q0←{(in2,k2),(in3,k3)... → (0,2)(1,3)(2,4)},
  q1←(0,0)(1,1)(2,2)(3,3)(4,4), q2←(2,0)…(6,4).
- **(ii) decoder dw = ConvTranspose2d, stride 2, pad 2, k5, output_pad 1, + per-channel BIAS:**
  `out[c,q,t] = Σ_{(p,k): q=2p−2+k} w[c,k]·in[c,p,t] + bias[c]`. dec1: 64ch 2→4. tap map:
  **q0←(p0,k2)+(p1,k0); q1←(p0,k3)+(p1,k1); q2←(p0,k4)+(p1,k2); q3←(p1,k3).** dec4: 8ch 16→32.
  ⚠️ **The decoder dw has a per-channel bias (out += bias_C); the encoder dw does NOT.** (This
  was the missing piece — without the bias, F.conv_transpose2d is 0.55 off.)

**B4. attention ↔ conv reshape/ear contract. ✓**
- enc5 (per ear [128, F=1]) → MHA input [256]: F=1, so it's **ch direct, no freq fold**
  (flatten of [256ch,1freq] = 256). Ear order = A3 (L 0..127, R 128..255).
- MHA output [256] → `chunk(2,dim=1)` → dec input per ear [128,1] (L 0..127, R 128..255).
- **decoder runs once per ear, shared weights, time-mux, NO cross-ear terms** (two
  independent `self.decoder(attn_l,…)` / `(attn_r,…)` calls).

**B5. warm-up mask. ✓ — golden DELIVERED.** Confirmed the deploy mask for t<31 is ONLY
`triu(diag=1)` (causal) + `dist > window(31) → −inf` — **no other special case**. So query
`i` valid keys = `[max(0,i−31), i]` (i+1 valid for i<31; <4 valid for i<3).
**`hw_patterns_final/mha_module_io_warmup/{t02,t10}/{real,imag}_pass/head_HH/`** (16 heads ×
2 passes): `Kwindow/Vwindow_localidx` (only the valid keys), `scores_row_localidx_{pre,post}scale`,
`softmax_dense_localidx`, `ctx16`, `packet.json`. Cold-start behavior confirmed:
- **t*=2 (3 valid keys, <4):** top-4 packet uses **only 3 slots** — `valid_slot=[1,1,1,0]`,
  `prob_code=[128,64,64,0]` (probs 0.5/0.25/0.25), softmax nonzero=3. The 4th slot is zero/invalid.
- **t*=10 (11 valid):** full top-4 — `prob_code=[128,64,64,64]`, all 4 valid.
So when <4 keys are valid the packetizer emits `min(4, n_valid)` slots and the pow2 denom is
over the survivors; the RTL per-frame validity mask must produce exactly this.

**B6. skip freq-layout. ✓** pw at output position f consumes `skip[:, f]` = the `count_s`
skip channels at freq f — **correct.** The dump is `[C, F, T]` **channel-major**
(`skip[ci,f,t] = data[ci·F·T + f·T + t]`, i.e. F-major within a channel). So a per-position
read gathers the `count_s` channels at **stride F·T** (not contiguous in the dump layout). If
you prefer contiguous per-position reads, re-lay-out to position-major on skip-write; both
are fine as long as the read addressing matches. Capacity arithmetic (2816 skip + 512 spec =
3328 ≤ 4 KiB) checks out.

## What I delivered / next
- Delivered: B3 dw goldens (`dw_position_golden/`); all A/B confirmations.
- Offer: B5 warm-up golden (t*=10, t*=2) — quick, on request.
- ⚠️ Two things to carry into RTL: **(B1)** dec0 skip uses the existing `pointwise_*_skip`
  e4m3 export as the k=+1 single-quant target (not a re-quant); **(B3)** decoder dw has a
  per-channel bias.
