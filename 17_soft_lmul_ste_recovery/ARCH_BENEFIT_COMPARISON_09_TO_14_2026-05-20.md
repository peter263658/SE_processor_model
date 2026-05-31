# Architecture Benefit Comparison: 09_weight_tied_mha → 14_low_power_arch_qat

Date: 2026-05-20

Scope: every architecture and RTL-facing change developed across folders 09
through 14, expressed in terms of what each one BUYS (parameters, MAC,
bandwidth, RTL simplicity) and what it COSTS (quality, spatial cue, QAT
budget). Internal development code-names (P0/P1/P2/P3/P4 and 14A/14B/14C/14D)
have been replaced with descriptive titles in this document so that anyone
without the original task-tracking context can follow the rationale. Folder
numbers (09, 10, ..., 14) are kept because they map directly to source
directories.

**MAC accounting**: all MAC numbers in this document are **per 5 ms hop**
= **per spectrogram frame** = the arithmetic the hardware must finish before
the next STFT hop arrives. At 16 kHz with hop = 80 samples this is also 1/200
of the per-second integral. `scripts/analyze_low_power_arch.py` is the
canonical source. Older folder-12 results docs report per-utterance decoder
pointwise MAC at the ~30–60 M scale; those numbers are converted to per-frame
where they appear (§3.4) so this document stays consistent.

**Hardware connectivity convention**: when a row says "QKV groups=4" or
"decoder pointwise groups_x=4 / groups_skip=2", the RTL stores **only** the
diagonal weight blocks and **does not** compute the off-diagonal positions.
The hardware contract is "block-diagonal weights, materialised once".

PyTorch implementation differs between the two grouping mechanisms — only the
QKV linear uses a forward-time `weight × block_diagonal_mask` (see §3.8); the
decoder grouped pointwise and the skip-subset use **real** `Conv2d(groups=…)`
plus an in-channel slice (see `DCNN/models/model.py:416` and
`DCNN/utils/complexPyTorch/complexLayers.py:1911`), so they **do** reduce
PyTorch parameter count. The hardware view is identical in both cases (only
diagonal blocks materialised); the parameter columns below reflect the
PyTorch reality.

STFT / iSTFT runs on a **separate on-chip unit** per the BCCTN ASIC spec and is
**not** counted in the MAC budgets here. The 10_stft_fixpoint_ptq folder
explores STFT reduced precision in isolation; it is not on the production
stack.

## 0. Executive Summary

The final low-power production stack is:

```text
W4A8 strict QAT (base)
+ tied complex MHA                                              (folder 09)
+ shift-topK pow2 minimal-area softmax                          (folder 09 → 11)
+ decoder grouped pointwise: groups_x=4, groups_skip=2          (folder 12)
+ decoder skip subset: top-50% skip channels by L1 magnitude    (folder 12)
+ FP16 attention accumulator                                    (folder 13)
+ L-Mul M4 attention multiplier backend, QAT-anchored ckpt      (folder 13)
+ block-diagonal grouped QKV projection: qkv_groups=4, 30-ep QAT (folder 14)
+ sparse top-4 packet attention probabilities + bit-exact L-Mul (folder 14)
```

Composite full-eval result for this stack (script-level config in §5):

| Stack | VCTK MBSTOI | Gate 0.8302 | TIMIT MBSTOI | Gate 0.7869 | Verdict |
|---|---:|---:|---:|---:|---|
| grouped-QKV-4 30ep + packet+bit-exact-LMul | 0.8354 | +0.0052 | 0.7880 | +0.0011 | **final production candidate** |

Main benefit summary:

| Area | Benefit |
|---|---|
| Parameters | tied MHA removes the independent imaginary MHA path. Under the active 09 config (`embed_dim=256`, `num_heads=16`): 830,300 → 567,132 trainable parameter elements, **−31.7%** total; MHA-local reduction is **−50%**. With decoder grouped-pointwise + skip-subset applied, the config-aware inspector reports 485,332 parameter elements. Two ways to express the final reduction: **−41.5 %** vs the original untied dense-decoder 09 reference (485,332 / 830,300), or **−35.2 %** vs an untied baseline that already has the same final decoder topology (485,332 / 748,500). The first answers "what did the full sequence buy"; the second isolates the tied-MHA contribution within the final architecture. |
| Decoder MAC | grouped pointwise + skip subset reduce decoder pointwise MAC by **−74.7%** and conv-only MAC by **−46.5%** vs the dense decoder baseline. |
| Attention MAC / operating point | block-diagonal QKV grouping at groups=4 reduces attention-equivalent MAC from 1.39 M → 1.10 M (**−21%**) and converts 64 lanes @ 5 MHz from failing duty 1.02 to passing duty 0.81. |
| Softmax RTL | replaces exp/div softmax with the top-k shift/pow2 path: no general multiplier, no divider, no exp LUT in the probability block; output is a 4-entry sparse row instead of a dense 32-entry row. |
| Attention multipliers | L-Mul M4 (and its bit-exact variant in folder 14) replaces FP-multiplier behaviour on QK and Score×V with an RTL-compatible add/shift backend. MAC **count** is unchanged; multiplier **energy and area** are the win. |
| Attention probability storage | sparse top-4 q0.8 packet stores 52 bits/row vs 256 bits/row for a dense q0.8 row and 512 bits/row for dense FP16 → **−89.8%** vs FP16. |
| RTL behaviour | local index bounded to 5 bits, causal window explicit, packet validation stays enabled as internal invariant, QK path is windowed/chunked, Score×V uses packet gather instead of dense `(B,H,T,T,D)` materialization. |

The 14 stack trades some TIMIT headroom vs the earlier folder-13 stack but
buys the concrete low-power operating point: 64 lanes @ 5 MHz within the 5 ms
hop budget.

### 0.1 Master Rollup — what each stage buys and costs

This table is the one-page version of the rest of the document. Each row is a
distinct architectural change that was eventually adopted on the final stack.
"Quality" is the full-VCTK MBSTOI at the moment that stage was introduced,
against whichever reference was used in the corresponding folder's results.
Deferred / rejected experiments are listed in §3 but not here.

| Stage (folder) | Parameter Δ | MAC Δ (per frame) | Bandwidth / SRAM Δ | RTL friendliness | VCTK MBSTOI | TIMIT MBSTOI |
|---|---|---|---|---:|---:|---:|
| Tied complex MHA (09) | −31.7 % total params / −50 % MHA-local | MHA passes 4 → 2 per frame (MHA compute halved) | MHA weight DMA halved (one weight load services both re and im) | datapath runs 2× instead of 4× | 0.8383 ref | n/a |
| Shift-topK pow2 softmax (09 → 11) | 0 | softmax block: dense 32 entries → top-4 sparse; **separate hardware unit, not on 64-lane MAC array** | softmax dense row replaced by 4-entry sparse output | removes exp LUT / divider / general multiplier from softmax block | 0.8360 | 0.7927 |
| Decoder grouped pointwise (12, groups_x=4 / groups_skip=2) | real Conv2d groups; reduces params (joint with next row ≈ −14 % vs tied-only baseline 567,132 → 485,332) | decoder pointwise **−62.16 %** per frame (≈ 188 K → 71 K) | grouped weight load fits weight-stationary dataflow | block-diagonal connectivity, no dynamic indexing | 0.8361 | 0.7923 |
| Decoder skip subset (12, top-50 % by L1) | reduces in_skip_channels (joint with above ≈ −14 % vs tied-only) | decoder pointwise **−74.74 %** cumulative per frame (≈ 188 K → 48 K), conv-only **−46.52 %** | skip-channel read bandwidth **−50 %** | static skip index ROM, frozen at training | 0.8352 | 0.7919 |
| FP16 attention accumulator (13) | 0 | 0 (MAC count unchanged) | scratch register fp32 → fp16 (narrower datapath) | uniform fp16 attention scratch | 0.8352 (isolated) | 0.7914 (isolated) |
| L-Mul M4 on QK + Score×V (13, QAT-anchored) | 0 | MAC **count** unchanged; multiplier **energy / area** ↓ | unchanged | FP multiplier → add/shift backend; uniform attention multiplier | 0.8364 | 0.7925 |
| Grouped QKV projection (14, qkv_groups=4, 30-ep QAT) | 0 (RTL stores only the diagonal 4 × 64 × 64 blocks for each Q/K/V projection) | **QKV layer −75 % (393 K → 98 K per frame), attention block −28 %, whole-model per-frame −21 % (1.39 M → 1.10 M)** | Q/K/V W4 set drops **96 KB → 24 KB**, so Q/K/V fit a 32 KB WBUF in one pass; O remains dense 32 KB unless `out_proj_groups>1` | block-diagonal Q/K/V weight matrix | 0.8353 | 0.7881 |
| Sparse top-4 q0.8 packet + bit-exact L-Mul (14) | 0 | 0 | probability row **512 → 52 bits (−89.8 %)** | packet gather replaces dense `(B,H,T,T,D)` Score×V; 5-bit local index | 0.8360 (base ckpt) | n/a at base ckpt; 0.7903 in 10-ep control |
| **Composite: grouped-QKV-4 30ep + packet+bit-exact-LMul** | combined | combined | combined | combined | **0.8354** | **0.7880** |

Two reading rules for this table:

1. The composite row is **not** the sum of the individual rows — composing the
   grouped projection with the packet/bit-exact-LMul attention datapath is
   measured to be lossless within metric noise (Δ < 0.0002 in either
   direction). Numerical (probability/multiplier) and structural (connectivity)
   changes are orthogonal.
2. Every "MAC Δ" entry is **per 5 ms hop = per frame**. The percentage
   figures (−62.16 %, −74.74 %, −75 %, −21 %) are scale-invariant; the
   absolute MAC figures in parentheses are the per-frame values used by the
   operating-point analyzer.

## 1. Baselines And Gates

Quality gates used by the later specs:

| Dataset | MBSTOI floor |
|---|---:|
| VCTK full paper-style | 0.8302 |
| TIMIT full paper-style | 0.7869 |
| VCTK 20-pair sanity | 0.8031 |

The 20-pair sanity gate is roughly 0.027 looser than the full VCTK spec gate,
so a 20-pair "pass" does not guarantee a full-set "pass". This is documented
empirically in §3.8 (grouped projection) where the 10-epoch QAT cleared 20-pair
but missed full VCTK by ~0.001.

Important reference points (full paper-style, 750 pairs/SNR averaged across
8 SNRs unless noted):

| Reference | VCTK | TIMIT | Notes |
|---|---:|---:|---|
| Tied-MHA strict (09 final) | 0.8383 | — | strict tied-MHA run; the closest "untouched" reference for the binaural acoustic quality |
| Shift-topK pow2 softmax primary (11) | 0.8360 | 0.7927 | replaces exp/div softmax with top-k + pow2 weights |
| Decoder grouped pointwise (12) | 0.8361 | 0.7923 | adds grouped pointwise on top of shift-topK |
| Decoder grouped + skip-subset (12) | 0.8352 | 0.7919 | adds top-50% skip channel selection |
| L-Mul attention backend, QK-QAT lineage (13) | 0.8364 | 0.7925 | L-Mul M4 on QK and Score×V |
| Folder-13 composite with deferred q0.8 dense + FP8 W4-scale (13) | 0.8326 | 0.7889 | passed hard floors but missed the 0.835 VCTK soft target; q0.8-dense and FP8-W4-scale were therefore deferred from 13 |
| Folder-14 final (grouped-QKV-4 30ep + packet+bit-exact-LMul) | 0.8354 | 0.7880 | current low-power production stack |

Config sanity check — do **not** infer the final architecture from each
folder's `config/config.yaml` alone:

| Folder | Base config model shape | Base config extra knobs | Interpretation |
|---|---|---|---|
| 09_weight_tied_mha | embed_dim=256, num_heads=16, attn_win=31, tied_mha=true | `attention_softmax.mode=softmax` | base tied-MHA model; softmax RTL modes are script / Hydra overrides |
| 10_stft_fixpoint_ptq | same 256/16/31/tied shape | `model.stft_fixed_point.enabled=true` | STFT precision experiment; not a final model architecture |
| 11_weight_tied_mha_fast_mbstoi | same shape | `attention_softmax.mode=softmax` | eval infrastructure snapshot; RTL softmax modes are overrides |
| 12_decoder_grouped_pointwise | same shape | `decoder_grouped_pointwise.mode=off` by default | grouped + skip-subset are not base-config defaults; scripts override |
| 13_aggressive_algo_qat | same shape | `attention_datapath` exists but disabled by default; decoder default off | L-Mul, FP16 accumulator, etc. are Hydra/script overrides on top of the base |
| 14_low_power_arch_qat | same shape | `attention_datapath` and `low_power` blocks exist but default disabled | final stack requires the explicit overrides listed in §5 |

## 2. Benefit Matrix By Architecture

| Folder / change | Architecture added | Main benefit | Quality result | Current decision |
|---|---|---|---|---|
| 09 tied MHA | tied complex MHA (`MHA_re == MHA_im`) | removes independent imaginary MHA weights; halves real-MHA passes | VCTK 0.8383 (strict tied); spatial cues preserved post-QAT | adopted |
| 09 → 11 shift-topK pow2 softmax | minimal-area softmax: top-k=4, pow2 weights, no mantissa | removes exp/div LUT; sparse 4-entry probability row | VCTK 0.8360 (conservative); minimal-area variant also 0.8360 | adopted |
| 10 STFT/iSTFT reduced precision | fixed/floating-point PTQ around the STFT/iSTFT boundary | explores cheap front-end FFT | only smoke / quick SNR0 evidence, not full sign-off | not final; reference only |
| 11 metric parallelization | MBSTOI eval parallelization | eval infrastructure; makes full sweeps tractable | byte-identical to serial baseline | adopted as tooling |
| 12 decoder grouped pointwise | groups_x=4 / groups_skip=2 inside the decoder pointwise branches | cuts decoder pointwise compute while preserving skip topology | VCTK 0.8361, TIMIT 0.7923 | adopted |
| 12 decoder skip subset (top-50% by L1) | keeps only the top-50% skip channels selected by validation-set L1 magnitude | halves skip-channel bandwidth, further cuts decoder pointwise MAC | VCTK 0.8352, TIMIT 0.7919 | adopted |
| 13 FP16 attention accumulator | narrows attention scratch datapath to fp16 | uniform fp16 inside attention; aligned with hardware datapath | isolated VCTK 0.8352, TIMIT 0.7914 | adopted |
| 13 dense q0.8 probability storage | 8-bit probability storage on a dense 32-entry row | row bandwidth halved | isolated pass, but stacked with FP8 W4-scale regressed in 13 | deferred in 13; reworked as the sparse packet path in 14 |
| 13 FP8 W4-scale SRAM | compresses W4 per-group scales from FP32 → FP8 E4M3 | reduces scale-table SRAM | isolated pass, stacked regression | deferred |
| 13 narrower attention score storage | FP8 / fixed-Q score format | would narrow score registers | FP8 top-k Jaccard failed; fixed-Q QAT failed | rejected |
| 13 L-Mul M4 on QK + Score×V | replaces FP multiplier backend with L-Mul-style add/shift | uniform attention multiplier backend; multiplier-energy reduction | VCTK 0.8364, TIMIT 0.7925 | adopted |
| 14 multiplier-free affine variants | PReLU pow2 / BN affine pow2 / tied real-imag BN scale | converts small affine multipliers toward shifts | 20-pair sanity + 10-epoch QAT only; spatial cue regression on some variants | not on the final critical path; see §3.9 for detail |
| 14 sparse top-4 q0.8 packet + bit-exact L-Mul | packetised sparse attention probabilities; RTL-bit-exact L-Mul simulator | packet row 52 bits, 5-bit local index, packet gather for Score×V | full VCTK 0.8360 at base ckpt; composite with grouped projection passes both gates | adopted on the final stack |
| 14 grouped QKV projection, qkv_groups=4 | block-diagonal connectivity inside the Q/K/V projections only | reduces attention-equivalent MAC; enables 64 lanes @ 5 MHz | qkv_groups=4 + 30-ep QAT: VCTK 0.8353, TIMIT 0.7881 | adopted on the final stack |
| 14 all-attention groups=4 (rejected) | grouping applied to input proj + QKV + out proj + output proj | larger MAC cut (−57 % attention) | sanity collapsed to 0.6819 | rejected by quality |

## 3. Detailed Benefits

### 3.1 Tied Complex MHA (folder 09)

Mechanism:

```text
Before: real MHA path and imag MHA path have independent weight tensors.
After : the imag-side MHA shares the real-side MHA weights.
```

Benefit under the active 09 config (`embed_dim=256`, `num_heads=16`):

| Item | Before | After | Reduction |
|---|---:|---:|---:|
| Total trainable parameter elements | 830,300 | 567,132 | −31.7 % |
| MHA-local params (re/out + im/out) | 526,336 | 263,168 | −50.0 % |
| Real-MHA passes per inference | 4 | 2 | −50.0 % |
| Softmax serializations per inference | 4 | 2 | −50.0 % |
| `mattn` block compute per step (from 09 README) | ~1.5 M ops | ~0.95 M ops | ~−37 % |

This is the cleanest parameter-reduction story in the whole sequence: it
removes an entire duplicate MHA path while keeping binaural quality intact
after QAT.

Note: the old checked-in `09_weight_tied_mha/model_inspect.csv` and
`model_size.md` were generated with default constructor settings
(`embed_dim=512`, `num_heads=32`) rather than the active 09 config
(`embed_dim=256`, `num_heads=16`, `tied_mha=true`). Use the updated
config-aware `model_inspect.py` output for parameter claims.

The final folder-14 stack instantiates to 485,332 parameter elements. This is
**lower** than tied-only 567,132 because the decoder grouped-pointwise +
skip-subset (12) use real `Conv2d(groups=…)` and a real in_skip_channels
reduction, so they remove ~81 K parameter elements from the decoder
pointwise branches.

Two ways to read the final reduction depending on the question being asked:

| Denominator | Reduction | Question it answers |
|---|---:|---|
| vs original untied dense-decoder 09 reference (830,300) | **−41.5 %** | what the full 09 → 14 sequence buys |
| vs untied baseline with the same final decoder topology (748,500) | **−35.2 %** | what tied MHA contributes on top of the final decoder |

The grouped-QKV layer in folder 14 is a separate story: its PyTorch
implementation uses a forward-time mask, so by itself it does **not** reduce
PyTorch parameter count. Its benefit lives entirely in MAC accounting and
RTL block-diagonal storage (see §3.8 and §3.10).

### 3.2 Shift-TopK Pow2 Softmax (folders 09 → 11)

Mechanism:

```text
score window ≤ 32 entries
  → score-domain top-k=4
  → subtract row max
  → clamp and approximate weights as powers of two
  → denominator rounded to power of two
  → sparse scatter to 4 nonzero probabilities
```

Hardware benefit:

| Before | After |
|---|---|
| exp LUT or exp approximation | removed |
| true division / reciprocal | removed |
| dense 32-entry softmax row | top-4 fused sparse row |
| general softmax multiplier path | replaced by add/subtract, compare, leading-zero detect, barrel shifts |

Quality:

| Variant | VCTK MBSTOI | ILD | IPD | Decision |
|---|---:|---:|---:|---|
| softmax baseline | 0.8373 | 3.2505 | 15.68 | reference |
| shift-topK pow2 (conservative primary) | 0.8360 | 3.2304 | 15.69 | adopted |
| shift-topK pow2 clamp8 no-mantissa (minimal-area final) | 0.8360 | 3.2356 | 15.69 | **final** |

The important benefit is not MAC count. It is removing the expensive nonlinear
softmax hardware block.

### 3.3 STFT / iSTFT Reduced Precision (folder 10)

`10_stft_fixpoint_ptq` added fixed-point and reduced floating-precision
simulation around the STFT / iSTFT boundary.

Observed quick SNR-0 smoke (10 files):

| Mode | MBSTOI | ILD | IPD |
|---|---:|---:|---:|
| FP32 STFT | 0.8153 | 5.1264 | 20.8854 |
| FP16 × FP16 STFT | 0.8171 | 5.1228 | 20.9623 |
| FP8 × FP8 STFT | 0.8170 | 5.2563 | 21.3142 |

Interpretation: useful feasibility evidence for STFT boundary precision, but
**not a signed-off architecture** because the available evidence is smoke / quick
run, not full 750-pair VCTK + TIMIT.

STFT / iSTFT is implemented on a separate on-chip unit and is not counted in
the MAC budgets used elsewhere in this document; reduced-precision STFT
would buy memory bandwidth around the front-end, not core MAC.

### 3.4 Decoder Grouped Pointwise And Skip Subset (folder 12)

The decoder pointwise layers are split into an `x` branch and a `skip` branch,
and each branch is independently grouped with **real `Conv2d(groups=…)`**
(not a mask × dense-weight multiplication). The skip branch is further
reduced to the **top 50 % of skip channels** by an in-place
`in_skip_channels` slice; the selection is fixed once from a
validation-set L1 magnitude scan and frozen as a ROM at training time. Both
mechanisms drop real parameters from the model — they are not mask-based.

Benefit (per frame; absolute numbers converted from the per-utterance
totals in the folder-12 results docs ÷ 200 hops per 1 s clip):

| Stack | Decoder pointwise MAC / frame | Δ | Conv-only MAC / frame | Δ | VCTK | TIMIT |
|---|---:|---:|---:|---:|---:|---:|
| dense baseline | ~188,500 | 0 % | ~302,800 | 0 % | 0.8360 | 0.7927 |
| grouped pointwise (`groups_x=4`, `groups_skip=2`) | ~71,300 | **−62.16 %** | ~185,600 | **−38.70 %** | 0.8361 | 0.7923 |
| grouped + top-50 % skip subset | ~47,600 | **−74.74 %** | ~162,000 | **−46.52 %** | 0.8352 | 0.7919 |

The percentages are scale-invariant and match the folder-12 results docs
exactly; the per-frame absolute numbers are integer-divided approximations.
The analyzer's `decoder_conv_p2 = 122,624` per frame is the canonical post-
state number, slightly below the ~162 K estimate because the analyzer
accounts only for the layers it models, not the full decoder branch.

Why this matters:

| Benefit | Explanation |
|---|---|
| MAC | the deepest decoder pointwise layers dominate conv compute (>80 % of model FLOPs); grouping hits the high-MAC part directly |
| Bandwidth | skip-subset reads only half of the selected skip channels per layer |
| RTL simplicity | the skip-index table is a frozen ROM, no dynamic selection at inference |
| Quality | VCTK and TIMIT remain comfortably above the later hard floors |

### 3.5 Attention Datapath Narrowing (folder 13, mixed outcomes)

Folder 13 tested a family of narrower attention datapaths / storage formats.
Outcomes diverged: some were adopted, some were deferred when stacking
showed regression, and one (narrower score storage) was rejected outright.

| Variant | Mechanism | Result | Decision |
|---|---|---|---|
| FP16 attention accumulator | fp16 scratch for QK / softmax / Score×V scratch | isolated VCTK 0.8352, TIMIT 0.7914 | adopted (part of final stack) |
| Dense q0.8 probability storage | 8-bit dense probability row | isolated pass; stacked with FP8 W4-scale regressed to 0.8326 / 0.7889 | deferred from 13; reworked as the sparse packet path in 14 |
| FP8 E4M3 storage for W4 per-group scales | compresses scale SRAM | isolated pass; stacked regression as above | deferred |
| Narrower attention score storage (FP8 / fixed-Q) | narrows score registers | FP8 top-k Jaccard failed; fixed-Q QAT failed | rejected |

Important lesson encoded here: **not every isolated win composes**. The full
folder-13 candidate stack with q0.8-dense + FP8-W4-scale dropped to VCTK
0.8326 and TIMIT 0.7889 — it still passed the hard floors but missed the 0.835
VCTK soft target and used most of the TIMIT margin. This is why those two
storage features were not shipped from 13 as dense storage. The 14 sparse
packet path is the rework that survives composition (see §3.7).

### 3.6 L-Mul Attention Multiplier Backend (folder 13)

L-Mul (logarithmic-multiply approximation, M4 = 4-bit mantissa) replaces the
FP-multiplier backend for the QK and Score×V matrix multiplies inside attention.

| Variant | QK mul | Score×V mul | VCTK | TIMIT | Decision |
|---|---|---|---:|---:|---|
| FP MAC reference | FP | FP | 0.8352 | 0.7919 | reference |
| QK = L-Mul M4 (QAT) | L-Mul | FP | 0.8365 | 0.7926 | feasible |
| Score×V = L-Mul M4 (no train) | FP | L-Mul | 0.8351 | 0.7920 | feasible |
| both = L-Mul M4 (no train) | L-Mul | L-Mul | 0.8349 | 0.7920 | feasible |
| Adopted lineage: QK-QAT ckpt + L-Mul inference on both | L-Mul | L-Mul | 0.8364 | 0.7925 | **adopted** |

Hardware benefit:

| Before | After |
|---|---|
| FP multiplier backend for QK and Score×V | L-Mul M4 backend for both |
| mixed backend if only QK is changed | uniform attention multiplier backend |
| multiplier-heavy attention matmuls | add / shift style approximate multiplier |

This is **multiplier-energy / area** reduction, **not** MAC-count reduction.

### 3.7 Sparse Top-4 Packet + Bit-Exact L-Mul (folder 14)

Folder 14 reworks the deferred dense-q0.8 idea from folder 13 into a **sparse
top-4 packet** representation and combines it with a **bit-exact L-Mul** simulator
that mirrors the RTL adder/shifter behaviour exactly.

Two separate, orthogonal mechanisms — both can be on at once and they do
different jobs:

- `model.attention_datapath.prob_quant` — a **value-grid** quantizer on the
  dense probability row (the folder-13 idea). On the final stack this is set
  to `none`; the dense path is not used.
- `model.attention_datapath.prob_packet.*` — a **sparse top-k packet** encoder
  that keeps only the top-4 indices per row and stores each as `(5-bit local
  index, 8-bit q0.8 probability)`. On the final stack this is `enabled=true`
  with `top_k=4`, `local_idx_bits=5`, `score_v_mode=lmul`.

Packet storage benefit:

| Representation | Bits per row | Reduction vs FP16 dense |
|---|---:|---:|
| Dense FP16 probability row (32 entries × 16 bit) | 512 | reference |
| Dense q0.8 probability row (32 entries × 8 bit) | 256 | −50.0 % |
| Top-4 q0.8 packet row (4 × (5+8) bit) | 52 | **−89.8 %** |

Algorithmic / RTL benefit:

| Feature | Benefit |
|---|---|
| 5-bit local index | bounded local addressing inside the 32-entry causal window; matches the hardware index field |
| Window-aware top-k | no illegal packet indices at early rows (the bug that surfaced in mid-cycle and was fixed via window-masked top-k + safe-index + `scatter_add` reconstruction) |
| `validate_local_idx=true` (kept on) | stays as an internal invariant assertion; should never fire for well-formed inputs |
| Packet Score×V gather | avoids dense `(B, H, T, T, D)` materialization; matches packet-consumer hardware |
| Windowed / chunked QK L-Mul | avoids dense full-window matmul memory pressure |

Quality (full VCTK, 8 packet/L-Mul variants on the base ckpt without further QAT):

| Variant | Mechanism | Full VCTK MBSTOI | vs gate 0.8302 |
|---|---|---:|---:|
| dense q0.8 probability (`b0`) | grid quantization only | 0.8365 | +0.0063 |
| packet dequant-to-fp16 Score×V (`b1`) | packet store + fp16 read-back | 0.8365 | +0.0063 |
| packet fixed shift-add Score×V (`b2`) | packet store + integer gather | 0.8366 | +0.0064 |
| packet L-Mul Score×V (`b3`) | packet store + L-Mul gather | 0.8361 | +0.0059 |
| bit-exact L-Mul on QK only (`c0`) | windowed-chunk QK with bit-exact simulator | 0.8365 | +0.0063 |
| bit-exact L-Mul on Score×V only (`c1`) | dense Score×V with bit-exact simulator | 0.8364 | +0.0062 |
| bit-exact L-Mul on both (`c2`) | combines `c0` + `c1` | 0.8364 | +0.0062 |
| **packet + bit-exact L-Mul on QK (`c3`)** | the final attention datapath | **0.8360** | **+0.0058** |

All eight variants clear the spec gate by +0.006 from the base ckpt — **no QAT
is required on the attention datapath side**. The most aggressive variant
(`c3`) sits within metric noise of the cleanest one.

### 3.8 Block-Diagonal Grouped QKV Projection (folder 14)

The MHA QKV projection (only the QKV layer, not input proj / out proj /
output proj) is reduced to a block-diagonal connectivity matrix: with
`qkv_groups=4`, the 256-dim QKV layer becomes four independent 64-dim
sub-matmuls, each operating on its own 64-dim input slice. The winning setting
is `qkv_groups=4` with a 30-epoch QAT cycle.

#### Compute and operating-point impact (per 5 ms hop, analyzer accounting)

| Component | Dense | `qkv_groups=4` | Δ |
|---|---:|---:|---:|
| Encoder conv | 202,880 | 202,880 | 0 |
| Decoder conv (post grouped-pointwise + skip-subset) | 122,624 | 122,624 | 0 |
| Attention input proj (E→E) | 262,144 | 262,144 | 0 (not grouped) |
| **Attention QKV (E→3E, tied 2-pass)** | **393,216** | **98,304** | **−294,912 (−75 %)** |
| Attention out_proj (E→E, tied 2-pass) | 131,072 | 131,072 | 0 (not grouped) |
| Attention output proj (E→E) | 262,144 | 262,144 | 0 (not grouped) |
| QK + Score×V (matmuls inside attention) | 18,432 | 18,432 | 0 |
| **Total per hop** | **1,392,512** | **1,097,600** | **−294,912 (−21 %)** |

Three readings of the same 21 % number:

- the **QKV layer itself** is cut by 75 % (393 K → 98 K MAC / hop);
- the **attention block as a whole** is cut by 28 % (1.05 M → 0.75 M MAC / hop on
  projections + qk_scorev);
- the **whole-model per-hop MAC** is cut by 21 % (1.39 M → 1.10 M).

| Variant | Total per-hop MAC | 64 lanes @ 5 MHz duty (utilization 0.85) | Verdict |
|---|---:|---:|---|
| Dense attention | 1,392,512 | **1.024** → fails 5 ms hop | reference |
| `qkv_groups=4` | 1,097,600 | **0.807** → meets 5 ms hop | **adopted** |
| All-attention groups=4 (input + QKV + out + output, rejected) | 606,080 | 0.446 → meets | rejected by quality |

The 5 ms hop budget at 64 lanes @ 5 MHz (85 % utilization) is
64 × 5 MHz × 5 ms × 0.85 ≈ 1.36 M MACs. Dense attention overshoots by
~32 K; `qkv_groups=4` lands 262 K under the budget. **The 21 % is exactly the
amount needed to convert this hop from failing to passing at 5 MHz**, which is
why this stage moves the operating point even though 21 % is "only" 21 %.

#### PyTorch mask vs RTL block storage (QKV only)

Within the folder-14 production stack, **only the QKV `GroupedLinear`** is
implemented in PyTorch as a forward-time mask × dense weight; the decoder
grouped pointwise and skip-subset use real `Conv2d(groups=…)` and a real
channel slice (see §3.4). So the "PyTorch doesn't actually save anything"
discussion below applies **only** to QKV. The QKV PyTorch forward is roughly:

```python
def forward(self, x):                                                # GroupedLinear (complexLayers.py)
    mask = _linear_group_mask(self.weight, self.groups)              # block-diagonal 0/1
    weight = self.weight * mask                                      # zeros off-diagonal blocks
    return F.linear(x, weight, self.bias)
```

Consequences in PyTorch for this layer:

- The full 256 × 256 QKV weight tensor is still stored.
- The full 256 × 256 matmul is still executed; off-diagonal results are
  multiplied by 0 but the multiplications still run.
- Gradient is 0 at off-diagonal positions, so those entries do not update
  during QAT. The block-diagonal pattern is enforced structurally at every
  forward.

The analyzer MAC numbers above (and the operating-point claim) describe the
RTL view: only the four 64 × 64 diagonal blocks of each Q/K/V projection are
stored, loaded, and computed. The PyTorch mask is used only to (i) verify
quality under block-diagonal connectivity and (ii) train the QAT checkpoint
that the RTL will load. The production `out_proj` remains dense because
`out_proj_groups=1` (see §3.10).

#### Hardware-side savings beyond the MAC count

The QKV-grouping payoff at the RTL level shows up in three places that the
per-hop MAC number alone does not capture:

**Weight SRAM (32 KB WBUF target):**

| Configuration | Q/K/V combined | O / out_proj | Q/K/V + O combined | 32 KB WBUF implication |
|---|---:|---:|---:|---|
| FP32 dense | 768 KB | 256 KB | 1 MB | no — many DMA passes |
| FP8 dense | 192 KB | 64 KB | 256 KB | no — 8 passes |
| W4 dense | 96 KB | 32 KB | 128 KB | Q/K/V need 3 passes, then O needs 1 pass |
| **W4 + `qkv_groups=4`** | **24 KB** | **32 KB** | **56 KB** | Q/K/V fit in one pass; O remains a separate dense pass |

With tied MHA (re/im share the same weight), the grouped Q/K/V working set
becomes a single 24 KB load that fits the WBUF. The dense O projection is still
a separate 32 KB load. This lets the Q/K/V weight DMA hide behind compute via
double-buffering, but it should not be described as a single Q+K+V+O WBUF pass.

**DMA bandwidth (DRAM → WBUF):**

Dense W4 needs three Q/K/V passes plus one O pass. `qkv_groups=4` collapses
the Q/K/V side to one pass plus the unchanged dense O pass. For Q/K/V weights
alone that is **−75 % bandwidth** (96 KB → 24 KB). Including the dense O
projection, production Q/K/V+O weight traffic is **128 KB → 56 KB** (−56 %).
Compared with an untied dense real/imag baseline, tied-MHA plus QKV grouping
puts the Q/K/V-side traffic at roughly 1/8; for Q/K/V+O combined it is about
56/256 ≈ 22 % of that untied dense baseline.

**Compute cycles on the 64-lane MAC array:**

For one output row of a grouped 256 × 256 Q/K/V projection, dense execution takes
256/64 × 256 = 1024 cycles. With `qkv_groups=4`, each row decomposes into four
independent 64 × 64 sub-matmuls; each sub-matmul saturates the 64-lane array
in 64 cycles, and the full row takes 4 × 64 = 256 cycles — a **4× speedup per
output row** without leaving any lanes idle (the saving is "we don't do the
0-multiply at all", not "we do it and discard"). This is the per-cycle cousin
of the per-hop MAC reduction above.

#### Quality

| Variant | VCTK | vs gate 0.8302 | TIMIT | vs gate 0.7869 | Decision |
|---|---:|---:|---:|---:|---|
| `qkv_groups=2`, 10-epoch QAT | 0.8295 | −0.0007 | 0.7861 | −0.0008 | misses both gates |
| `qkv_groups=2`, 30-epoch QAT | 0.8326 | +0.0024 | 0.7859 | −0.0010 | VCTK only; TIMIT still misses |
| `qkv_groups=4`, 10-epoch QAT | 0.8300 | −0.0002 | 0.7845 | −0.0024 | misses both gates |
| **`qkv_groups=4`, 30-epoch QAT** | **0.8353** | **+0.0051** | **0.7881** | **+0.0012** | **adopted** |
| `qkv_groups=4` 30ep + packet+bit-exact-LMul attention | 0.8354 | +0.0052 | 0.7880 | +0.0011 | **final composite** |

**Counter-intuitive but verified**: the more aggressive grouping (groups=4,
−21 % per-hop MAC) **outperforms** the safer one (groups=2, −14 %) after a
30-epoch QAT, even though the 10-epoch result was the other way around. Likely
explanation: groups=4 forces the model to commit earlier to a sparse-attention
solution; the 30-epoch budget is needed to drive the model into that solution,
which is why the 10-epoch checkpoint looked worse. **Operational implication:
the QAT budget for any new grouped-projection variant must not be cut below
30 epoch from a sanity number alone.**

This is the key low-power architecture in folder 14: it converts the attention
block from "needs more lanes or a faster clock" into a feasible 64-lane,
5 MHz operating point.

### 3.9 Multiplier-Free Affine Variants (folder 14, deferred from final stack)

Folder 14 also tested four affine variants intended to replace small per-channel
multiplies with shifts:

| Variant | Mechanism | 20-pair base | 10-ep QAT 20-pair | SegSNR Δ (20-pair, post-QAT) | Hardware implication |
|---|---|---:|---:|---:|---|
| PReLU pow2 (`a0`) | PReLU slope quantized to nearest 2^k | 0.8104 | — (no QAT needed) | +2.47 dB | PReLU slope → barrel shift; no multiplier needed |
| PReLU pow2 + 10-ep QAT (`a1`) | as above with retrain | 0.8104 | 0.8092 | +2.37 dB | same |
| BN affine pow2, untied (`a2`) | BN scale quantized to 2^k (separate real / imag scales) | 0.8012 | 0.8062 | +0.92 dB | BN scale → barrel shift; two scales per layer |
| BN affine pow2 + tied real/imag (`a3`) | as above with shared real/imag scale | 0.8078 | 0.8070 | −0.52 dB | BN scale → barrel shift; one shared scale per layer (further SRAM saving) |

Observed caveats:

- PReLU pow2 alone (`a0`) needs no QAT; clean win on the MBSTOI side.
- BN pow2 quantization is **approximate but MBSTOI-recoverable** with short
  QAT (the `a2` row went 0.8012 → 0.8062, +0.005 above the sanity gate).
- The tied real/imag variant (`a3`) **passes MBSTOI but regresses SegSNR by
  −0.5 dB at 20-pair sanity**, and IPD error rises (16.57 → 18.21 deg). Not
  free; needs full-eval validation before deployment.

Not on the final critical path because the production stack (§5) already
meets both spec gates without these. They remain on the table as an
**additional** multiplier-area saving if a future tape-out needs more
aggressive area reduction — the BN scale paths could be removed entirely in
RTL if the QAT budget can accept the tied/untied trade-offs.

### 3.10 SRAM / DMA Bandwidth Impact

This is the practical reason the grouped-QKV-4 result matters more than the
21 % MAC number suggests. The ASIC has a 32 KB WBUF (per the BCCTN ASIC spec
reference). The final stack does not make the whole Q+K+V+O set fit in one
pass; it makes the Q/K/V set fit in one pass while leaving O as a separate
dense pass.

Production MHA projection weight footprint:

| Configuration | Q/K/V combined | O / out_proj | Q/K/V + O combined | 32 KB WBUF implication |
|---|---:|---:|---:|---|
| FP32 dense reference | 768 KB | 256 KB | 1 MB | no — many passes |
| FP8 dense | 192 KB | 64 KB | 256 KB | no — 8 passes |
| W4 dense | 96 KB | **32 KB** | **128 KB** | Q/K/V need 3 passes, then O needs 1 pass |
| W4 + `qkv_groups=4` | **24 KB** | **32 KB** | **56 KB** | Q/K/V fit in one pass; O remains one dense pass |

Combined with tied MHA (only the real-side weight needs to live in WBUF), the
final stack reduces the Q/K/V transfer to one WBUF-resident batch. Weight DMA
can run double-buffered behind the previous frame's compute for Q/K/V, while
the dense O projection still consumes its own WBUF pass.

Attention probability storage SRAM (per query row, full causal window of 32
keys; 200 query rows per 1 s clip at hop 80 / 16 kHz):

| Representation | Bits / row | Per-row storage | Per-1 s clip (200 rows) |
|---|---:|---:|---:|
| Dense FP16 | 512 | 64 B | ~12.8 KB |
| Dense q0.8 | 256 | 32 B | ~6.4 KB |
| Top-4 q0.8 packet | 52 | 6.5 B | **~1.3 KB** |

The packet path reduces the on-chip probability scratch to about 1.3 KB per
1 s of audio — relevant for the KV-cache budget if the design ever holds
multiple frames of attention probabilities simultaneously.

## 4. Cumulative MAC Rollup (per frame)

All numbers in this section are **per 5 ms hop = per spectrogram frame**.
The encoder is unchanged across the whole sequence; decoder enters the
analyzer post grouped-pointwise + skip-subset; attention changes in two
places (tied MHA halves the imaginary pass; `qkv_groups=4` cuts the QKV
layer). Source: `scripts/analyze_low_power_arch.py` plus the percent
reductions cross-referenced from §3.4.

| Stage | Encoder | Decoder (post grouping + skip-subset) | Attention block | qk_scorev | Total per frame | Δ vs dense baseline |
|---|---:|---:|---:|---:|---:|---:|
| Pre-tied dense (estimated; analyzer does not provide this row) | 202,880 | 122,624 | ~1,440,000 (4-pass MHA, dense QKV) | 18,432 | ~1,784,000 | reference |
| + tied MHA (halves MHA passes; analyzer's "f0_dense") | 202,880 | 122,624 | 1,048,576 | 18,432 | **1,392,512** | **−22 %** |
| + grouped QKV (`qkv_groups=4`) — analyzer's "f3_qkv_g4" | 202,880 | 122,624 | 753,664 | 18,432 | **1,097,600** | **−38 %** |

Per-frame deltas in this table come from MHA-side changes only because the
analyzer's decoder figure already reflects the grouped-pointwise + skip-subset
post-state (see §3.4 for the within-decoder reductions, also per-frame).
Shift-topK softmax does not show up in this MAC accounting — it is a
different hardware block (no general multiplier; not on the 64-lane MAC
array).

### 4.1 Two readings of "QKV grouping only saves 21 % per frame"

The user-facing question "why does QKV grouping look small in the rollup?"
has two answers, both correct at different scales:

- The QKV layer **itself** is cut by 75 % (393 K → 98 K per hop). That is not
  small.
- The QKV layer is one of four MHA projection layers (input / QKV / out /
  output). The other three are not grouped in the production stack, so they
  dilute the relative saving when measured at the attention-block level (−28 %)
  or whole-model per-hop level (−21 %).
- The 21 % at whole-model per-hop is also the **exact amount** needed to push
  64 lanes @ 5 MHz from duty 1.024 to duty 0.807. The operating-point payoff
  is binary (fits the 5 ms hop or doesn't), not linear in MAC.

If a future ASIC iteration tightens the hop budget further, the natural next
target is one of the three currently-ungrouped projections (input proj, out
proj, output proj). Earlier sanity (`d2 / d3 / d4` variants in §6 of the 14
TODO) showed those collapse without QAT; reaching them would require their
own 30-ep QAT cycle.

## 5. Final Architecture Definition

The final low-power architecture is:

```text
Base lineage:
  W4A8 strict QAT
  tied complex MHA
  shift-topK pow2 minimal-area softmax
  decoder grouped pointwise (groups_x=4 / groups_skip=2)
  decoder skip subset (top 50 % skip channels by L1, frozen at training time)
  FP16 attention accumulator
  L-Mul M4 attention multiplier backend, QAT-anchored ckpt

Folder-14 additions:
  training.qat_scope=full
  training.top_k_attention=4
  training.fp16_accumulator=true
  training.int4_scale_format=fp32
  model.attention_softmax.mode=shift_topk_pow2_clamp8_nomantissa
  model.decoder_grouped_pointwise.mode=grouped
  model.decoder_grouped_pointwise.groups_x=4
  model.decoder_grouped_pointwise.groups_skip=2
  model.decoder_grouped_pointwise.skip_select_ratio=0.5
  model.decoder_grouped_pointwise.skip_select_mode=topk_l1
  model.decoder_grouped_pointwise.skip_indices_path=Checkpoints/p2_skip_indices_g42_topk_l1_r050.pt
  model.attention_datapath.enabled=true
  model.attention_datapath.score_quant=none
  model.attention_datapath.prob_quant=none
  model.low_power.mha_projection.enabled=true
  model.low_power.mha_projection.qkv_groups=4
  model.attention_datapath.lmul_enabled=true
  model.attention_datapath.lmul_targets=[qk]
  model.attention_datapath.lmul_mantissa_bits=4
  model.attention_datapath.lmul_bit_exact=true
  model.attention_datapath.prob_packet.enabled=true
  model.attention_datapath.prob_packet.top_k=4
  model.attention_datapath.prob_packet.local_idx_bits=5
  model.attention_datapath.prob_packet.score_v_mode=lmul
  model.attention_datapath.prob_packet.validate_local_idx=true
  model.attention_datapath.prob_packet.prob_bits=8
  model.attention_datapath.prob_packet.prob_scale=256
  model.attention_datapath.prob_packet.prob_max_code=255
  model.attention_datapath.prob_packet.causal_window=null   # null = couple to 2**local_idx_bits = 32
  model.attention_datapath.prob_packet.renormalize=false
```

The last five lines come from `config/config.yaml` defaults but are listed
explicitly here so that an RTL handoff that copies this override block does
not silently inherit the wrong defaults if a future config-rebase changes
them.

**Critical override interaction (don't copy this list naively without reading
it)**: in the final packet path, `lmul_targets=[qk]` controls only the QK
matmul. The Score×V side is driven through the packet consumer via
`prob_packet.score_v_mode=lmul`, so Score×V also goes through the L-Mul
backend — just via the packet path, not via `lmul_targets`. Writing
`lmul_targets=[qk, score_v]` here would not produce the final composite stack;
it would double-route Score×V through both the dense L-Mul path and the
packet consumer.

Similarly, `prob_quant=none` (value-grid quantizer off) and
`prob_packet.enabled=true` (sparse top-4 packet on) are intentionally both
configured: they are two independent mechanisms (§3.7) and the final stack uses
the sparse packet only.

Recommended one-sentence architecture claim:

```text
The final design keeps W4A8 / tied-MHA quality, removes the expensive nonlinear
softmax block, compresses decoder pointwise compute by 74.7 %, uses L-Mul /
bit-exact L-Mul for the attention matmuls, packetises attention probabilities
to 52 bits per row, and adds block-diagonal grouped QKV so the attention block
meets a 64-lane @ 5 MHz, 5 ms-hop operating point while passing both VCTK and
TIMIT spec gates.
```

## 6. What Not To Claim

| Claim to avoid | Correct wording |
|---|---|
| "Dense q0.8 probability storage is shipped from folder 13." | Dense q0.8 was isolated-pass but deferred in 13 because of stacked regression. The shipped path is the folder-14 **sparse top-4 packet** reformulation, without the FP8 W4-scale change. |
| "All-attention grouping gives −57 % MAC and should ship." | It gives −57 % attention MAC analytically but sanity-scale quality collapsed to 0.6819. Rejected. |
| "20-pair sanity pass implies full-eval pass." | The sanity gate is ~0.027 looser than the full VCTK spec gate. Final decisions require full VCTK + TIMIT. |
| "L-Mul reduces MAC count." | L-Mul keeps operation count unchanged; the win is multiplier energy / area. |
| "STFT FP8 / fixed-point is final." | Only smoke evidence in folder 10; not signed off. |
| "BN pow2 / tied real-imag is free." | 20-pair MBSTOI looks fine, but the tied variant regresses SegSNR by ~0.5 dB at sanity scale; full-eval validation is pending. Not on the final critical path. |
| "Grouped QKV at 10-epoch QAT is enough." | The 10-epoch checkpoint misses both VCTK and TIMIT gates. 30-epoch QAT is required for the production result. |
| "`lmul_targets=[qk, score_v]` is the final stack." | The final stack uses `lmul_targets=[qk]` plus `prob_packet.score_v_mode=lmul`. See §5 critical interaction note. |

## 7. Source Files Used

| Source | What it supports |
|---|---|
| 09_weight_tied_mha/README.md | tied MHA local benefit, 4 → 2 real-MHA passes |
| 09_weight_tied_mha/model_inspect.py | config-aware full-parameter reduction calculation |
| 09_weight_tied_mha/results/compare_table_vctk_averages.csv | tied-MHA full-eval reference |
| 09_weight_tied_mha/RTL_HANDOFF_SOFTMAX_2026-05-14.md | softmax RTL sweep and final minimal-area mode |
| 10_stft_fixpoint_ptq/README.md and quick result dirs | STFT reduced-precision status |
| 11_weight_tied_mha_fast_mbstoi/HANDOFF_MBSTOI_PARALLEL_2026-05-14.md | eval parallelization tooling |
| 12_decoder_grouped_pointwise/RESULTS_P1_GROUPED_2026-05-17.md | grouped decoder pointwise results |
| 12_decoder_grouped_pointwise/RESULTS_P2_SKIP_2026-05-18.md | decoder skip subset results |
| 13_aggressive_algo_qat/RESULTS_P3_ATTENTION_FP8_FINAL_2026-05-18.md | attention datapath narrowing decisions |
| 13_aggressive_algo_qat/RESULTS_P4_LMUL_FEASIBILITY_FINAL_2026-05-18.md | L-Mul attention backend decisions |
| 13_aggressive_algo_qat/RTL_HANDOFF_FULL_STACK_2026-05-19.md | integrated handoff and deferred dense-storage features |
| 14_low_power_arch_qat/STATUS_LOW_POWER_2026-05-19.md | final folder-14 results, including composite eval |
| 14_low_power_arch_qat/results/low_power_analysis/*.json | MAC, packet bits, and operating-point accounting |
| 14_low_power_arch_qat/results/low_power_composite/d1_qkv_g4_plus_c3_full/ | final composite full-eval result |
