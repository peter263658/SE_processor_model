# MHA golden + input_proj/output_proj contract — answers for the HW advisor (2026-06-08)

Two requests answered here, in order:

- **Part A** — inner ComplexMHA per-module golden (in_proj → QKᵀ → softmax/packet →
  Score×V → out_proj), incl. KV-cache layout and complex/tied handling.
- **Part B** — exact `mattn.input_proj` / `mattn.output_proj` computation contract
  (resolves the R²≈0 blocker).

Everything below was produced from the deploy point
`strict_core_pow2_blockfloat_deploy_ep3_vctk0.8357_timit0.7898.ckpt` on the golden
wav `p225_p225_009_az-40_snr+0.0.wav`, and is **verified against the shipped golden
bit-exactly** (patched-forward output == `layer_io_fp32/output_waveform.fp32.hex`,
max_abs_diff = 0).

Config that fixes every constant below: `ckpt/strict_core_deploy_config.yaml`.
`embed_dim=256, num_heads=16, head_dim=16, T=399`, sliding window = 32 keys.

---

## Part A — inner MHA golden

**Patterns:** `hw_patterns_final/mha_module_io/` (see its `README.md`).
**Generator:** `scripts/dump_mha_module_io.py`.

Representative slice: query `t* = 351` (newest query of chunk-aligned block
`[320..351]`, so its 32-key causal window is full); all 16 heads; both passes.

### A0. Complex / tied handling (this is the addressing-critical part)

`tied_mha=true`: one real attention block `M` runs twice, once per input half:

```
a = M(x_r)    b = M(x_i)
ComplexMHA(x_r + i·x_i) = (a − b) + i·(a + b)
```

- `real_pass/` = `M(x_r)`, `imag_pass/` = `M(x_i)`. **Same weights**, different input.
- The `(a−b, a+b)` recombine happens in the caller, **after** `out_proj` — it is
  NOT inside the QK/softmax/Score×V datapath. So the core processes two
  independent real passes and the complex glue is two adds at the very end.
- This halves both the parameter set and the softmax passes vs an untied complex MHA.

### A1. `mha_in_proj`  (`01_in_proj/`)
- Input = MHA input for `t*` `[256]` (one real/imag half), `input_x_fp32`.
- `mha.in_proj_weight` is `[768,256]` = stacked `[Wq;Wk;Wv]`, **block-diagonal
  `qkv_groups=4`** (each of Q/K/V is 4 independent 64×64 blocks; off-block
  weights are masked to 0). Bias `[768]`.
- Output `qkv_concat_fp32` `[768] = [Q256 | K256 | V256]` (raw `F.linear` output).
- `{Q,K,V}_headmajor` `[16 heads, 16]`: the reshaped per-head Q/K/V, head-major
  (`reshape(T,H,D).transpose → (H,T,D)`; head `h` owns embed lanes `[16h:16h+16]`).
  **These are post-activation-quant (e4m3 block-float) fp16 values** — i.e. the
  actual operands of QKᵀ and the KV-cache contents (see A6).

### A2. QKᵀ per head  (`02_qk/head_HH/`)
- `Q_block_32x16`, `K_block_32x16`, `V_block_32x16` = the 32 queries / 32 keys of
  block `[320..351]`.
- `scores_block_32x32_prescale` = `Q_block·K_blockᵀ` via **L-Mul m8 (bit-exact)**,
  fp16. `scores_block_32x32_postscale` = ×2⁻² (the QK scale).
- **QK scale = ×2⁻²** (pow2 mode: `1/√head_dim = 1/√16 = 1/4 = 2⁻²` → arithmetic
  shift right by 2; verified pre/post ratio = 4.0).
- `mask_block_32x32` = additive mask (`−inf` outside the causal sliding window).
- ⚠️ The dense `[32×32]` is exact **only for the last row** (`t*=351`). For
  complete per-query data use `qstar_scores_row32_localidx_*` `[32]` (the newest
  query's full window, in `local_idx` order) + `qstar_{K,V}window_32x16_localidx`.

### A3. Softmax + top-4 packet  (`03_softmax/head_HH/`)
- `qstar_softmax_dense32_localidx` `[32]`: softmax output, **dense with exactly
  ≤4 nonzeros** (`shift_topk_pow2_clamp8_nomantissa`, top-4). Weights are pow2
  fractions normalized by a **pow2 denominator** → every kept prob is a negative
  power of two (e.g. 0.5, 0.25, 0.125), which is why the packet codes are clean.
- Hardware top-4 packet (4 slots, emit order = descending prob):
  - `qstar_packet_prob_code_u8` `[4]` — **8-bit prob code, `prob = code/256`** (`prob_scale=256`, max code 255).
  - `qstar_packet_local_idx_u8` `[4]` — **5-bit `local_idx`** (`local_idx = t* − key`, 0..31).
  - `qstar_packet.json` also gives `abs_key_frame` and `valid_slot`.
- Example (real_pass, `t*=351`):

  | head | prob_code[4] | local_idx[4] |
  |---|---|---|
  | 0 | `[128,32,32,32]` | `[9,10,15,25]` |
  | 1 | `[128,64,64,64]` | `[14,15,29,20]` |
  | 4 | `[128,64,64,32]` | `[0,6,28,25]` |

  (top-1 is always code 128 = 0.5; full 16-head table in the patterns.)

### A4. Score×V  (`04_score_v/head_HH/`)
- Input = top-4 packet (A3) + the head's V window (`qstar_Vwindow_32x16_localidx`).
- `out = Σ_{slot=0..3} L-Mul_m8( prob_topk[slot], V[ local_idx[slot] ] )`,
  `prob_topk = code/256`. (`score_v_mode=lmul`, bit-exact.)
- Output `qstar_context16` `[16]` per head, fp16.
- Cross-check: vs an exact-mul dequant reconstruction, cos-sim = 0.99998 (the
  small gap is exactly the L-Mul m8 approximation).

### A5. `mha_out_proj`  (`05_out_proj/`)
- Input `input_ctx256` `[256]` = the 16 per-head `[16]` contexts concatenated
  (head-major: head `h` → lanes `[16h:16h+16]`).
- `out = F.linear(ctx, out_proj.weight, out_proj.bias)`, `out_proj.weight` `[256,256]`.
- Output `output256_fp32` `[256]` = this pass's MHA-half result (`a` or `b`).
- This is **W4A16**: the out_proj input is the fp16 Score×V accumulator, not an
  e4m3-block-float A8 activation.

### A6. KV-cache layout & dtype (addressing)
- Layout: **`[head][frame][head_dim]`** = `(H=16, T, D=16)` (that is exactly how
  `{Q,K,V}` are captured; `q.reshape(B,T,H,D).transpose(1,2)`).
- Sliding window = 32 frames; `local_idx = query − key ∈ [0,31]`, `local_idx 0` =
  newest (the query's own frame), `local_idx 31` = oldest.
- **Dtype: e4m3 block-float.** K/V come out of in_proj, are quantized by the
  activation quantizer to e4m3 (raw e4m3 code + a per-tensor pow2 shared
  exponent), then cast to fp16 for the L-Mul accumulate. Since e4m3 ⊂ fp16, the
  fp16 cache value is the exact e4m3 grid value (the dumped `*_fp16` files are the
  HW number; `*_fp32` are the same value upcast for reference).

### A7. Numeric path summary (deploy)
| op | datapath |
|---|---|
| in_proj QKV | INT4×e4m3 (block-diagonal `qkv_groups=4`), per-head APoT weight scale |
| QKᵀ | L-Mul **m8** bit-exact, fp16 accum, windowed chunk=32, window=32 |
| QK scale | **×2⁻²** (pow2 shift) |
| softmax | `shift_topk_pow2_clamp8_nomantissa`, top-4, pow2 denom |
| packet | top-4, q0.8 (`code/256`, 8-bit, ≤255), `local_idx` 5-bit |
| Score×V | L-Mul **m8** bit-exact over 4 packet slots · V |
| out_proj | `F.linear`, W4A16 (fp16 ScoreV input) |

---

## Part B — `mattn.input_proj` / `mattn.output_proj` contract (R²≈0 fix)

Both are a single **complex64 `GroupedLinear`** with `groups=1` (dense). The
export splits the one complex weight into two real halves — `fc_r = W.real`,
`fc_i = W.imag`, each `[256,256]`. **`module_type=_Half` just labels "one
real/imag half of a logically-complex parent"; it is NOT a special computation.**

Verified by reconstructing the projection from the deploy model's effective
weight and matching the shipped fp32 golden I/O
(`scripts/verify_proj_contract.py`):

| convention | input_proj R² | output_proj R² |
|---|---|---|
| **`out = x @ Wᵀ`, `W[oc,ic]`, re=xr·Wr−xi·Wi, im=xr·Wi+xi·Wr** | **1.000000** | **1.000000** |
| `out = x @ W` (W as `[ic,oc]`) | −0.857 | −0.083 |
| conj combine re=xr·Wr+xi·Wi | −0.763 | −0.309 |
| real-only (drop imag W) | 0.523 | 0.672 |

The R²≈0 you saw is the second/third row — wrong index transpose or wrong complex
sign. The correct contract:

### B1. Index convention
`W.shape = [out_features=256, in_features=256] = W[oc, ic]` (PyTorch `nn.Linear`).
`F.linear(x, W)` computes `out[...,oc] = Σ_ic x[...,ic] · W[oc, ic]`, i.e.
**`out = x @ Wᵀ`** (matrix form: `out = W @ x` per token). So `W[oc, ic]`, with the
**output channel as the row**.

### B2. Complex combine (4-product)
With `x = x_r + i·x_i`, `W = W_r + i·W_i` (`W_r=fc_r`, `W_i=fc_i`):
```
out_re[oc] = Σ_ic ( x_r[ic]·W_r[oc,ic] − x_i[ic]·W_i[oc,ic] )
out_im[oc] = Σ_ic ( x_r[ic]·W_i[oc,ic] + x_i[ic]·W_r[oc,ic] )
```
Maps 1:1 onto the CE complex MAC: `P0=fr·xr, P1=fi·xi, P2=fr·xi, P3=fi·xr`,
`re = P0 − P1`, `im = P2 + P3`, with `fr=W_r, fi=W_i`. (Bias is complex, added after.)

### B3. Layout
Activation tensor is `[T=399 frame, 256 channel]`, **channel is the inner/
contiguous axis** (per frame, 256 channel values contiguous). The linear runs
independently per frame; `in_features`/`out_features` are the channel axis, the
399 frames are the token/batch axis.

### B4. Per-head (16) scale
The INT4 export scales these projections **per output-head**:
`int4_per_head_linear` reshapes `W[256,256] → [16 heads, 16 rows, 256]`, so
**head(oc) = oc // 16** — every group of 16 consecutive output rows shares one
APoT scale. Scale tensor shape `[16,1,1]`. **`fc_r` and `fc_i` carry independent
per-head APoT scales** (each half quantized separately). Effective weight:
`W_r[oc,ic] = q_r[oc,ic] · scale_r[oc//16]` (q ∈ INT4 `[−7,7]`), likewise `W_i`.

> This confirms your `mem_controller` per-head dealer index `oc_idx >> clog2(256/16)
> = oc >> 4 = oc // 16`, `n = NUM_HEAD = 16`. In the complex MAC, the descale of
> P0/P2 (the `W_r` products) uses `scale_r[oc//16]`; P1/P3 (the `W_i` products)
> use `scale_i[oc//16]`.

### B5. e4m3 activation ↔ fp32 golden correspondence (confirmed)
The `tb_pattern/activations_fp8/mattn_{input,output}_proj__0/input_0.{real,imag}.e4m3.u8`
codes, dequantized as **`value = e4m3_code · 2^(shared_exp)`**, equal the
`layer_io_fp32` fp32 golden input **exactly** (max_abs_diff = 0):

| activation | real shared_exp | imag shared_exp |
|---|---|---|
| input_proj input | **−7** | **−6** |
| output_proj input | **−3** | **−3** |

So the fp32 golden input *is* the dequantized block-float input (real and imag
each have their own per-tensor pow2 exponent). Use the e4m3 u8 code as the MAC
operand and apply the per-tensor `2^exp` as part of the post-MAC descale.

---

---

## Part C — QK accumulator: deploy fp32-sum vs HW fp16 adder-tree (2026-06-08)

**Question.** Deploy QK reduction is `products.sum` = fp32 accumulate, round once.
The CE PEG is the 32-way `adder_tree_fp16` (paired binary tree, fp16 RNE +
saturate-to-±65504 at **every** node, `MODE_LMUL_ATTN`). So the HW will NOT be
bit-exact to the deploy QK scores. Reuse the conv fp16 tree, or add a QK-only
fp32 accumulator?

**What I changed.** Added a faithful model of `adder_tree_fp16` and gated the QK
reduction on a flag (`attention_datapath.qk_accumulate`: `fp32` default / `fp16_tree`),
in `code/DCNN/utils/complexPyTorch/complexLayers.py`
(`_fp16_add_sat`, `_fp16_tree_reduce`, `reduce_mode` on `_lmul_matmul` /
`_lmul_windowed_qk_matmul`). Default `fp32` reproduces the shipped golden
**bit-exactly** (verified). Tree topology = adjacent-pair `(0,1)(2,3)…` over the
16 head_dim products (the 32-way tree's unused upper lanes are 0, so the +0 levels
are exact no-ops). Pre-tree pow2 preshift is a global scale → identical fp16
mantissas → it suffices to tree-reduce the existing dequantized products.

**Measured impact (`scripts/dump_mha_qk_fp16tree.py`, golden wav, both passes):**

| metric | fp32-sum vs fp16-tree |
|---|---|
| QK raw scores changed | 32.4% of in-window entries (rest bit-identical) |
| QK score abs diff | max 0.0625, p99.9 0.0156, **mean 8.5e-4** |
| QK score ULP@magnitude | **mean 1.1 ULP**, 67.5% ==0, **91.5% ≤1 ULP**, 95.7% ≤2 ULP; >4 ULP only 2.3% (cancellation on ~0 scores) |
| **top-4 packet key-set identical** | **99.81%** (24 / 12768 query·head rows differ) |
| **top-1 (highest-prob) key flip** | **0.078%** (10 / 12768) |
| packet prob-codes changed | 0.135% (69 / 51072) |
| Score×V context abs diff | max 3.53 |
| MHA-half output | cos **0.99995** |
| full output waveform | cos **0.99992**, max_abs 0.034 |

So the deploy-vs-HW gap is ~1 ULP typical (a few ULP under cancellation), top-k
selection is 99.8% stable, and the utterance output is perceptually identical —
MBSTOI will not move.

**Recommendation: Option 1 — reuse the conv fp16 tree (`adder_tree_fp16`).**
- **Zero extra hardware** — QK shares the existing 32-way fp16 saturating tree
  (it's already `MODE_LMUL_ATTN` on the same `pe_pair`/`PEG`). A QK-only fp32
  accumulator means a second, wider (fp32) adder tree used for one op — poor
  area/power for a sub-ULP, top-k-stable, MBSTOI-neutral difference.
- **Numerically safe** — the QK score only (a) selects the top-4 keys and (b)
  feeds a pow2-quantized no-mantissa softmax; both are robust to ~1 ULP. The
  design already tolerates much larger L-Mul / pow2 perturbations.
- **Verification strategy** — make the **`fp16_tree` model the bit-exact golden**
  the RTL must match (`hw_patterns_final/mha_module_io_fp16tree/`); treat the
  deploy fp32 eval as the MBSTOI quality reference and compare RTL→deploy golden
  with a tolerance (≈ a few fp16 ULP on scores), NOT bit-exact.
- Score×V stays bit-exact either way (4 terms, pow2 probs, no accumulation
  rounding pressure) — confirmed.

> ⚠️ One thing to confirm on your side: the model assumes the 16 QK products land
> in PEG lanes `0..15` (so the tree pairs head_dim `(0,1)(2,3)…`). If your
> controller maps the 16 products to different lanes, the tree pairing — and thus
> the bit-exact golden — changes; tell me the lane order and I'll regenerate.

**fp16-tree golden:** `hw_patterns_final/mha_module_io_fp16tree/` (QK scores +
Score×V + packet for `t*`, both passes; everything else identical to
`mha_module_io/`). Diff it head-to-head against `mha_module_io/` (fp32-sum).

---

---

## Part D — input_proj / output_proj: act-k fusion & bias (RTL open questions, 2026-06-08)

Both answers are already DECIDED by `01_RTL/engine/CE/CE_DATAPATH_DRAFT.md §2.1`
and the existing decoder path; the proj layers just haven't been brought in line
with it yet. Numbers verified against the deploy ckpt.

### D1. act-k — FUSE into per-head codes (split-k), do NOT apply 2^k at runtime

**Answer: fuse.** Emit k-fused per-head APoT codes (split `k_real` / `k_imag`)
for `input_proj`/`output_proj` `fc_r`/`fc_i`, exactly like the decoder already
ships (`scale_apot_{a,b}_exp_fused_act_kreal/kimag`). The CE dealer loads one
pre-fused code per PEG — no runtime k stage.

Why (not a preference — it's the decided datapath):
- `dealer.sv` is specified to consume a **single fused APoT code per PEG**
  ("k baked in per-PEG static slot"). `CE_DATAPATH_DRAFT §2.1`: *"always carry
  two static k slots `k_real`,`k_imag`, wired per-PEG by slice … the dealer just
  loads the right pre-fused exponent."* The decoder already does this.
- act-k is a **static calibrated per-tensor** pow2 (no runtime amax) and is
  constant along the contraction axis, so folding it into the per-head post-acc
  descale is exact (same algebra as the §1.2 preshift proof). Verified fusion
  convention: `fused_exp = raw_weight_exp + act_k` (checked on decoder
  `pointwise_r_x`: fused−raw = −6 = that layer's act-k).
- Runtime 2^k buys nothing and costs an extra exponent-add + subop k plumbing +
  control. The raw-only proj export is just an **exporter gap** — close it by
  emitting the fused files for the proj too.

Per-PEG wiring (matches `CE_IMPL_NOTES` PEG map `P0=fr·xr, P1=fi·xi, P2=fr·xi,
P3=fi·xr` and §2.1 `x_real→k_real`, `x_imag→k_imag`):

| PEG | weight half | activation | fused dealer code |
|---|---|---|---|
| PEG0 | fc_r | x_real | `ws_r ⊕ k_real` |
| PEG1 | fc_i | x_imag | `ws_i ⊕ k_imag` |
| PEG2 | fc_r | x_imag | `ws_r ⊕ k_imag` |
| PEG3 | fc_i | x_real | `ws_i ⊕ k_real` |

Fuse = add k to **both** `a_exp` and `b_exp` (`2^a±2^b → 2^(a+k)±2^(b+k)`;
`has_second`/`residual_sign` unchanged). `preshift=0` for the proj (per the
2026-06-07 overflow table the proj is safe ≤0.17×), so the dealer carries the
**full** fused descale, no residual split.

Concrete (raw `a_exp` per head; verified):

| layer | k_real / k_imag | raw a_exp | fused a_exp: PEG0/PEG3 (k_real) | PEG1/PEG2 (k_imag) |
|---|---|---|---|---|
| input_proj | −7 / −6 | −4 (all heads) | **−11** | **−10** |
| output_proj | −3 / −3 | −4 or −3 | raw−3 | raw−3 (k collapses) |

For `output_proj`, `k_real==k_imag` so the two fused variants are identical
(still write both slots for uniform control, per §2.1). If the CE instead uses
raw apot without applying k, `input_proj` is wrong by **2^7=128×** on the
x_real-driven PEGs and **2^6=64×** on the x_imag PEGs — i.e. broken, not "off".

### D2. bias — add it in the PPU β path (the hardware already exists)

**Answer: add the per-oc complex bias via PPU β** (`ppu_fma` `u_bnb_re/u_bnb_im`).
The proj currently routes to `PPU_PASS`, which skips β — that's why the bias is
dropped. The fix is to route `input_proj`/`output_proj` through the affine β-add,
not PASS.

- The proj is a **plain complex linear** (not tied), so the bias adds **directly**
  — `β_re = b_r`, `β_im = b_i` (`b_r = fc_r/bias`, `b_i = fc_i/bias`, per-oc, 256).
  **This is NOT the conv `b_r∓b_i` fold** (that fold is conv-specific). Verified:
  `out_re += b_r, out_im += b_i` reproduces the fp32 golden at R²=1.0 (Part B).
- Domain: bias is post-descale fp16, per output channel; `oacc` already holds the
  fully-descaled linear sum (weight scale is applied in the dealer), so β is added
  on the `oacc` output in the PPU — correct domain.
- Bias is significant (`input_proj` |b|max 0.46, `output_proj` 0.64) — dropping it
  visibly corrupts the output.

**Recommended routing (reuses the existing CONV datapath, zero new hardware):**
run the proj subop through `PPU_CONV` with
- `α_re = α_im = APoT(2^0)` (identity — there is no BatchNorm; the weight scale is
  already in the dealer),
- PReLU `k=0, sign=+` (identity — the proj has no activation),
- `β_re = b_r`, `β_im = b_i`,
- e4m3 cast `k = next-stage block-float exponent` **iff** the proj output is
  stored as block-float for its consumer (it is: `input_proj`→MHA in_proj input,
  `output_proj`→decoder both re-quantize to e4m3).

i.e. set the proj subop `has_bn=1` with the identity α and PReLU above (not PASS).
If you'd rather keep the proj output as plain fp16 (defer the e4m3 cast to the
consumer), add a minimal `PPU_PASS+β` mode instead (`o = oacc + β`; the adders
already exist). Either way the invariant is: **β is added on the descaled
accumulator, before any cast.**

### D3. Generated codes + end-to-end verification (R²=1)

`scripts/gen_proj_fused_codes_and_verify.py` emits the fused codes + bias and
verifies them by reconstructing the projection from INT4 weight codes × e4m3
activation codes × **the generated fused descale** + bias, vs the fp32 golden:

| layer | k_real / k_imag | VERIFY (fused codes + bias) | neg-ctrl: raw apot (no k) | neg-ctrl: drop bias |
|---|---|---|---|---|
| input_proj | −7 / −6 | **R² = 1.00000000** (max_abs 1.2e-4) | R² = −7500 | R² = 0.987 |
| output_proj | −3 / −3 | **R² = 1.00000000** (max_abs 2.4e-4) | R² = −49.8 | R² = 0.999999 |

(`R²=1` re-confirmed reading the on-disk `.bin` files; max_abs is fp32
reassociation only. The negative controls prove both fixes are mandatory: raw
apot without act-k is off by 2^k → R²≪0; dropping the bias loses real variance.)

**Artifacts** — `hw_patterns_final/proj_fused_codes/<layer>/`:
- `<half>/scale_apot_{a,b}_exp_fused_act_{kreal,kimag}.int16` — fused exponents
  (same names/format as the decoder). `<half>/scale_apot_code_fused_act_*.u16` —
  the packed 12-bit APoT-6a code (`[11:7]=a+24 [6:2]=b+24 [1]=has_second [0]=res_sign`).
- `bn_bias_{r,i}_fused.fp16` — PPU β = (b_r, b_i), per-oc.
- Same files dropped **additively** into `tb_pattern/weights/<layer>/` (new
  filenames only; nothing overwritten). `manifest.json` has per-layer summary.

Per-PEG dealer code, input_proj head 0 (concrete): PEG0 `0x6A3` (a=−11,b=−16),
PEG1 `0x72E` (−10,−13), PEG2 `0x727` (−10,−15), PEG3 `0x6AA` (−11,−14). Identity
BN-α (for the bias-only PPU path) = `0xC60` (a=b=0, no second term).

### D4. Patched into the deploy SRAM image (§3 done)

`scripts/patch_memmap_proj_fused.py` appended the 6 new const_blocks (4 split-k
weight-scale + 2 proj β) into the WBUF slack `[251680, 253920)` of
`hw_patterns_final/sram_images/{wbuf_unified_init.bin, memory_map.json}` —
non-destructive (pre-existing bytes untouched, `*.pre_projfused.bak` backups,
slack 10467→8224 B). Read-back from the patched image reconstructs both projections
at **R²=1**. PPU cast (`i_cast_k = −k_consumer`): input_proj **+5** (MHA in_proj k=−5);
output_proj **−1** (MHA out is `chunk(2)`→L/R 128ch → decoder, which quantizes at
k=+1 — verified bit-exact; corrects the tentative −2). **Full advisor hand-off list:
`docs/PROJ_FUSED_DELIVERABLES_2026-06-08.md`** (offsets, subop config, cast-k).

---

---

## Part E — Softmax score precision: deploy fp16-delta vs RTL sq4 (2026-06-09)

**Issue (advisor).** RTL `Softmax_v3` (== `tb_SM.py`) truncates the score to **sq4**
(Q.4 = 1/16) *before* the shift-topk delta. Deploy `_shift_topk_softmax` sees the
**full-fp16** score (`score_quant: none`), so `x = score − max` is fp16 and only the
exact max gets `2^0`; sq4 collapses near-ties (<1/16) to delta 0 → `2^0` → flatter.

**Confirmed in code:** deploy attn is full fp16 (no sq4); sq4 is an RTL-only choice,
**never part of the trained/gated model** (gate = 0.8357/0.7898).

**Measured downstream cost** (`scripts/verify_softmax_sq4_vs_deploy.py`, golden wav,
both passes; deploy run reproduces golden bit-exact):

| | sq4 vs deploy softmax | fp16-tree QK (Part C, benign) |
|---|---|---|
| top-4 keyset identical | **86.8%** | 99.81% |
| top-1 key flip | **4.7%** | 0.078% |
| Score×V context cos | **0.991** | ~exact |
| output waveform | cos 0.9992, **rel_rms 3.9%** | cos 0.99992 |

**Decision: Option 1 — compute the top-4 delta in fp16 (drop sq4), bit-exact to deploy.**
- sq4's divergence is ~50× the (benign) QK fp16-tree effect and was never quality-gated;
  3.9% output RMS is too large to ship without re-running MBSTOI.
- Area cost is negligible: top-4-only — one fp16 subtract `(score−max)`, `×1.5` (`x+x>>1`),
  `floor` (mantissa mask), then the existing clamp/ldexp. sq4 needs a truncate anyway.
- This makes the **whole attention bit-exact to deploy** (QK fp16-tree packet is already
  16/16; softmax fp16-delta → 16/16).
- The deploy fp16-delta softmax golden **already ships**: `mha_module_io/*/03_softmax/`
  (`qstar_softmax_dense32`, `qstar_packet*`). Validate the fixed `Softmax_v3` against it
  and update `tb_SM.py` / `00_TB/SoftMax` golden to match.
- Keep the QK fp16-tree accumulation (Part C) — that one is benign and free.

### Answers to the consult (`CONSULT_SOFTMAX_SCORE_DELTA_PRECISION_2026-06-09.md`)

**Q1 — canonical?** The deploy full-fp16 `floor(1.5·(score−max))` (peaky). It is the
trained + MBSTOI-gated model; sq4 is an RTL-only deviation.

**Q2 — was sq4 ever quality-checked / what was deploy measured with?** Deploy
0.8357/0.7898 was measured with **full-fp16 delta** — proven by the deploy config
`strict_core_deploy_config.yaml`: `score_quant: none` + `shift_topk_pow2_clamp8_nomantissa`
→ `_shift_topk_softmax` on raw fp16 scores. **sq4 was never MBSTOI-evaluated**: the only
score-quant knob is `score_quant ∈ {none, fp8_e4m3, fixed}` and every gated run used
`none`; the score-delta fixed-point precision was never swept. So matching deploy is the
safe sign-off target. (Forward-level cost of sq4 already measured in the table above:
3.9% output RMS, 13% keyset divergence — not benign. A full MBSTOI A/B is runnable via
`eval.py --hydra_overrides model.attention_datapath.score_quant=fixed score_fixed_frac_bits=4`
if a number is wanted, but the decision doesn't hinge on it.)

**Q3 — exact RTL spec (bit-exact to deploy):**
```
x      = fp16_sub(topk_score, row_max)              # both fp16 (PPU-SCALE QK out)
scaled = fp16_add_sat(x, fp16_scale2(x, -1))        # = RNE_fp16(1.5·x), NOT wide-exact
scaled = clamp(scaled, -8, 0)                        # shift_topk_pow2_clamp8
k      = floor(scaled)            in [-8,0]           # use_mantissa=False
weight = 1 << (k)  (i.e. 2^k)
denom  = 2^round(log2( Σ weight ))                    # nearest
prob   = weight / denom ; code = round(prob·256) clamp[0,255]
```
⚠️ **`1.5·x` must be RNE-rounded to fp16 before `floor`, not computed exact.** Audit over
the whole golden: exact `x + x>>1`→floor differs from deploy's `fp16(1.5x)`→floor on
**18/50880** top-k entries (boundary cases where the fp16 round crosses an integer). The
advisor's proposed wide-fixed-point `x + x>>1` would mismatch deploy there; `fp16_add(x, x·2^-1)`
(reusing the existing `fp16_add_sat`) is bit-exact and cheap.

**Step-by-step golden:** `mha_module_io/softmax_stepwise/softmax_stepwise_golden.json`
(per head, t*=351, both passes: topk scores, row_max, x, 1.5x, k, weight, denom, prob,
q0.8 code) + `*_k_floor.int16` / `*_prob_code_q08.u8`. Recomputed codes == shipped deploy
packet **32/32 head-passes**. Head-1 example matches the consult exactly:
local_idx [14,29,15,20] → x [0,−.0054,−.0239,−.0708] → 1.5x [0,−.0081,−.0359,−.106]
→ k [0,−1,−1,−1] → code [128,64,64,64], denom 2.0.

## Reproduce
```bash
cd /raid/R12K41024/19_algo_final
conda run --no-capture-output -n bcctn python scripts/verify_softmax_sq4_vs_deploy.py  # Part E sq4 vs deploy
conda run --no-capture-output -n bcctn python scripts/dump_mha_module_io.py     # Part A patterns + bit-exact check
conda run --no-capture-output -n bcctn python scripts/verify_proj_contract.py   # Part B contract proof
conda run --no-capture-output -n bcctn python scripts/dump_mha_qk_fp16tree.py   # Part C fp32-sum vs fp16-tree + tree golden
conda run --no-capture-output -n bcctn python scripts/gen_proj_fused_codes_and_verify.py  # Part D fused codes + bias, R²=1 verify
```
