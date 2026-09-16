# Architecture Progression — Condensed Results

Per stage: the hardware-architecture change, and what it buys (parameters / MAC /
quality). Quality = full paper-style VCTK & TIMIT MBSTOI at the stage it was
introduced. Gates: VCTK ≥ 0.835 (hard floor 0.8302), TIMIT ≥ 0.7880.

Two parameter views: **RTL** = weights the chip actually stores/computes;
PyTorch state_dict counts differ for masked/grouped layers (noted where it
matters). MAC = per 5 ms frame.

The progression below now starts from the original full-size BCCTN reference
(`fft_len=512`, `embed_dim=512`) and then separates two kinds of changes:

- **Architecture downsizing / structural compression**: reduces tensor sizes,
  stored weights, and MAC workload.
- **Numerical / hardware arithmetic conversion**: keeps the topology mostly fixed
  but changes precision, scaling, and multiplier implementation.

---

## Original BCCTN → hardware-sized working point (pre-09)

These rows explain why the later 09→18 work starts from an already downsized
baseline instead of the paper-like BCCTN. The original model is kept here as the
quality / size reference.

| Stage / source | Architecture / analysis result | Params / workload implication | Quality / decision |
|---|---|---|---|
| **Original BCCTN reference** (`/raid/R12K41024/BCCTN/config/model.yaml`) | `win_len=400`, `win_inc=100`, `fft_len=512`; 6 encoder/decoder stages `kernel_num=[16,32,64,128,256,256]`; `embed_dim=512`, `num_heads=32`; untied real/imag MHA; dense decoder | **3,973,288 params** = encoder 274,988 + transformer 3,151,360 + decoder 546,940. Transformer alone is ~79% of params. Archived flow shape: STFT `[1,256,321]`, transformer `[1,256,4,321]`. Profiled compute (hooks on the `model.yaml` arch, random init) ≈ **9.4 G MACs / 2 s** = **≈ 29 M MAC per 100-sample frame** (~4.7 G/s): convT 4.86 G + conv 2.43 G + MHA 1.77 G + linear 0.34 G. Full conv (`(5,2)` kernel) + dense decoder + untied 32-head full `T×T` attention; complex = 4 real sub-convs as executed. **Own basis** (hop 100, fft 512); not normalized to the deploy's 1.10 M multiply-equivalent/frame. | Full-size quality reference: **VCTK 0.8519 / TIMIT 0.8114** (`BCCTN/results_May21/...`). Too large for the target WBUF/MAC budget. |
| **Hardware-sized FP8/QAT working point** (`main_model` / `01_utilization_analysis`) | Move to `fft_len=128`, `win_len=128`, `win_inc=80`; keep 6 stages; reduce transformer to `embed_dim=256`, `num_heads=16`, sliding `attn_win=31`; depthwise-separable conv path | Parameter basis becomes **830,300** before tied-MHA / decoder grouping. Utilization analysis: attention is **77.6%** of MACs; decoder 15.0%; encoder 7.4%. The 1-second profiler counted **1.687G MACs** in its own convention; use it for bottleneck attribution, not as the normalized per-frame number below. | Hardware-sized FP8/QAT baseline quality: **VCTK 0.8399 / TIMIT 0.7966** (`main_model/results_May21/...`). This becomes the practical starting point for 09+. |
| **01 Utilization analysis** | Weight-stationary 256-MAC analysis found the model is globally compute-bound, but layer bottlenecks differ: early/depthwise conv are activation-DMA/under-utilized, attention projections are compute-heavy, QK/Score×V are activation-transport heavy | Average utilization barely changes with W4A8 (45.3%→45.4%), so W4A8 is **not** chosen to raise MAC utilization. The useful gain is weight SRAM/DMA and WBUF residency. | Direction: attack attention weight size / transport first, then remove softmax and multiplier cost. |
| **02 Weight bit-width sensitivity** | Weight-only PTQ sweep on the FP8-QAT model: activations remain FP8, weights are tested as INT8/INT4/INT3/FP4 | Uniform INT4 gives **50% weight/DMA reduction** with moderate local metric loss (SI-SNR −0.56 dB, ILD +0.21 dB, IPD +0.64°). INT3 mixed precision saves more bytes but fails by cumulative error (SI-SNR −3.08 dB). | Decision: use **W4A8** as the main path; avoid W3. Add per-group/per-head scaling rather than global scale. |
| **07 Mixed-precision W4A8 QAT design** | INT4 for most conv/linear/MHA weights, FP8 activations, fp16 accumulator. Keep decoder pointwise outliers in FP8; MHA scales per-head; QK/Score×V need fp16 accumulator | Early finding: decoder PW stages 0/3/4 contain outliers; MHA head groups are non-uniform; QK dynamic range exceeds FP8. | This becomes the numerical template later refined into W4A8 + APoT scales + L-Mul + pow2 block-float activations. |

Net original → hardware-sized baseline: parameters **3.97M → 0.83M** (about
**−79%**) before the 09→14 structural compression. Quality drops by about
0.012 VCTK MBSTOI and 0.015 TIMIT MBSTOI, but the model becomes small enough to
start fitting the target hardware constraints.

Compute drops on the same arc: the original profiles at **≈ 29 M MAC/frame**
(hop-100 basis) and the final deploy is **1.10 M multiply-equivalent/frame** —
roughly **20–27× fewer MAC** (basis-dependent: different hop, FFT, and complex-mult
convention), achieved by full→depthwise-separable conv, dense→grouped decoder with
top-50% skip, untied full attention → tied + windowed + grouped-QKV + top-4, and
lower FFT/freq resolution. The original number is profiled (forward hooks, random
init) and reported on its own basis; it is *not* normalized to the deploy frame.

---

## Structural compression (folders 09 → 14, starting from the hardware-sized baseline)

| Stage | HW architecture change | Params | MAC/frame | VCTK | TIMIT |
|---|---|---|---|---|---|
| Hardware-sized baseline (pre-09 untied, dense decoder) | `fft_len=128`, `embed_dim=256`, `num_heads=16`; FP8/QAT working point | **830,300** | ~1.67 M | 0.8399 | 0.7966 |
| **09 Tied complex MHA** | `MHA_re == MHA_im`: one weight set, datapath runs 2× not 4× | 830,300 → **567,132** (−31.7%; MHA-local −50%) | MHA passes 4→2 | 0.8383 | — |
| **09→11 Shift-topK pow2 softmax** | top-4 + pow2 weights, no mantissa; **removes exp/div LUT**; 4-entry sparse prob row | — | softmax mult/div → shift | 0.8360 | — |
| **12 Decoder grouped pointwise** (groups_x=4, groups_skip=2) | real `Conv2d(groups)`, block-diagonal; no dynamic indexing | 567,132 → **485,332** (−14%) | decoder PW −62%; whole 1.67→**1.44 M** (−14%) | 0.8361 | 0.7923 |
| **12 Decoder skip subset** (top-50% by L1) | keep only top-50% skip channels (fixed index) | (in above) | halves skip bandwidth | 0.8352 | 0.7919 |
| **13 FP16 attention accumulator** | attention scratch narrowed to fp16 | — | — | 0.8352 | 0.7914 |
| **13 L-Mul M4 on QK + Score×V** | FP multiplier backend → L-Mul add/shift | — | attn multiplier energy ↓ (count same) | 0.8364 | 0.7925 |
| **14 Grouped QKV projection** (qkv_groups=4, 30-ep QAT) | block-diagonal Q/K/V; only diagonal 64×64 blocks exist | RTL Q/K/V 196,608 → **49,152** (−75%); total RTL **337,876** | QKV −75%, attn −28%, whole 1.39→**1.10 M** (−21%); **duty 1.02→0.81** | 0.8353 | 0.7881 |
| **14 Sparse top-4 q0.8 packet + bit-exact L-Mul** | packetised sparse prob; 52 bits/row, 5-bit local idx | — | packet gather for Score×V | 0.8360 | — |
| **= 14 composite (production)** | grouped-QKV-4 + packet + L-Mul | PyTorch 485,332 / **RTL 337,876** (−59.3% vs 09) | **1.10 M** (−21% vs dense QKV) | **0.8354** | **0.7880** |

Net 09→14: RTL weights **−59.3%**, MAC/frame **1.67 M → 1.10 M**, both gates passed.
QKV W4 weight set 96 KB → 24 KB ⇒ fits the 32 KB WBUF in one pass.

---

## MAC reduction steps (only changed workload)

This table intentionally lists only the stages where the normalized neural-core
workload changes. Later numerical stages (15→18) keep the same 1.10 M
multiply-equivalent workload and change *how* each operation is implemented.

| Step | MAC / workload change | Why it changes |
|---|---|---|
| Hardware-sized FP8/QAT baseline | establishes the normalized working point at **~1.67 M MAC/frame** | `fft_len=128`, `embed_dim=256`, `num_heads=16`, sliding `attn_win=31`; the original BCCTN 512/512 model is not normalized to this final frame basis |
| 09 Tied complex MHA | MHA real passes **4 → 2** | real/imag MHA weights are tied; the tied block runs two shared real-MHA passes and combines `(a−b)+i(a+b)` |
| 12 Decoder grouped pointwise | whole model **1.67 M → 1.44 M/frame** (−14%); decoder pointwise local −62% | decoder pointwise layers become static block-diagonal grouped convs: `groups_x=4`, `groups_skip=2` |
| 14 Grouped QKV projection | whole model **1.39 M → 1.10 M/frame** (−21%); QKV local −75%; attention local −28%; duty **1.02 → 0.81** | `qkv_groups=4` keeps only diagonal 64×64 Q/K/V blocks |

Notes: the 12 skip subset halves skip activation bandwidth and is already folded
into the grouped-decoder result above. Top-4 sparse attention changes transport
and multiplier implementation, not the normalized MAC count.

---

## Parameter / stored-weight reduction steps (only changed storage)

This table separates logical PyTorch parameter counts from the RTL stored-weight
basis where masking/grouping removes weights that the chip never stores.

| Step | Parameter / stored-weight change | Why it changes |
|---|---|---|
| Original BCCTN → hardware-sized baseline | **3,973,288 → 830,300** params (−79.1%) | FFT/model downsizing: `fft_len=512→128`, `embed_dim=512→256`, `num_heads=32→16`, smaller hardware-oriented working point |
| 09 Tied complex MHA | **830,300 → 567,132** params (−31.7%; MHA-local −50%) | one shared MHA weight set is used for both tied real-MHA passes |
| 12 Decoder grouped pointwise | **567,132 → 485,332** params (−14.4%) | decoder pointwise layers become grouped/block-diagonal; fixed top-50% skip subset removes unused skip inputs |
| 14 Grouped QKV projection | PyTorch count remains **485,332**, but RTL stored weights become **337,876**; Q/K/V RTL **196,608 → 49,152** (−75%) | `qkv_groups=4` means off-diagonal Q/K/V blocks do not exist in RTL |
| W4A8 / INT4 storage | same logical weight count, but stored bytes are about **half** of FP8 for INT4 layers | final RTL weights are INT4 except the documented FP8 outlier tensors |
| Cumulative original → final RTL basis | **3,973,288 → 337,876** (−91.5%) before INT4 byte savings | hardware-sized model + tied MHA + grouped decoder + grouped QKV |

For plotting a single "parameter / stored-weight reduction" curve, use this
series. The first point is the original PyTorch parameter count because that model
was not RTL-targeted; the last point switches to the RTL stored-weight basis
because grouped QKV removes weights that the chip does not store.

| Plot point | Value to plot | Basis |
|---|---:|---|
| Original BCCTN (`fft_len=512`, `embed_dim=512`) | **3,973,288** | PyTorch params = encoder 274,988 + transformer 3,151,360 + decoder 546,940 |
| Hardware-sized FP8/QAT baseline | **830,300** | PyTorch params |
| 09 tied complex MHA | **567,132** | PyTorch params after MHA tying |
| 12 decoder grouped pointwise / skip subset | **485,332** | PyTorch params after decoder grouping |
| 14 / 18 final RTL stored-weight basis | **337,876** | RTL stored/computed weights after grouped QKV; use this for chip storage |

---

## Why W4A8 was kept through the architecture work

The early analysis did not choose W4A8 because it made the PE array busier. It
chose W4A8 because it changed the memory/weight-residency problem:

- Original attention dominated the model: the 01 analysis measured attention at
  **77.6%** of MACs.
- MHA weights were the large reusable object. The 02 report estimated the MHA
  real+imag FP8 matrices at **512 KB**, far beyond a 64 KB WBUF; uniform INT4 cuts
  that to **256 KB** before later structural grouping.
- Layer-wise PTQ showed INT4 was the lowest robust uniform weight width. INT3
  looked fine one layer at a time, but failed when applied together because
  errors accumulated through the network.
- The final 14 grouped-QKV step is what makes W4 really fit the smaller WBUF:
  QKV weight storage goes from **96 KB → 24 KB** after grouping, fitting the final
  32 KB WBUF target.
- The later 18 work keeps the same W4A8 topology but removes the remaining
  multipliers: INT4×activation becomes shift-add, scales become APoT/pow2, and
  FP8 outlier/attention multiplies become L-Mul.

So W4A8 is a **memory/SRAM/DMA choice first**, and a multiplier-removal platform
second. The observed MBSTOI recovery comes from QAT/finetune under the deployed
arithmetic, not from W4A8 alone.

---

## Numerical / multiplier-free (folders 15 → 18)

Structure (params, MAC count) is unchanged from the 14 composite; these stages
change the **arithmetic and bit-width**, not the topology. They remove every
multiplier from the neural core and halve weight memory.

| Stage | HW architecture change | Weight mem / arithmetic | VCTK | TIMIT |
|---|---|---|---|---|
| **W4A8** | INT4 weights + FP8 activations; scale: PW per-group(32) / DW per-channel / MHA per-head | weight SRAM/DMA **halved** vs FP8 | (within noise of 0.8354) | — |
| **APoT weight scale** (apot_clip[-16,-1]) | per-group scale = `2^a ± 2^b` = 2 shift + 1 add (was a real mult) | scale mult → shift-add | 0.8350 | 0.7892 |
| **Conv shift-add** | INT4 code × FP8 = shift-add MAC + delayed per-group scale | conv multiplier → shift-add | — | — |
| **Attention L-Mul m8 + FP8-outlier L-Mul m4** | QK/Score×V and 12 FP8 PW layers via L-Mul | attn/outlier multiplier → exp-add + mantissa-add | — | — |
| **BN fold → APoT[-9,4]** | fold to `a_q·x + b`, quantize effective `a=γ/√(var+ε)` | BN multiplier → 2 shift + 1 add | — | — |
| **PReLU pow2[-8,0]** | slope → `±2^k` | PReLU multiplier → sign-mux + 1 shift | — | — |
| **= 18 strict core (exact descale)** | **whole neural core zero-multiplier** | shift / add / mux only; no mult, no divider, no exp LUT | 0.835531 | 0.790234 |
| **= 18 DEPLOY (delayed APoT descale, FT)** | + FP8-outlier descale = delayed 2-term APoT shift-add `[-16,-6]` (no per-output mult); activations still **dynamic** per-frame amax/descale | strictly shift/add/mux | 0.8365 | 0.7912 |
| **★ = 18 pow2 BLOCK-FLOAT DEPLOY (FT)** | + activations = **e4m3 raw code + per-tensor pow2 shared exponent**; the activation descale fuses into the weight descale as a single post-acc **exponent offset** (shift-add). **No runtime activation amax/descale.** eager==fused exactly (relL2=0) ⇒ eval == HW | strictly shift/add/mux; **zero runtime activation scaling** | **0.8357** | **0.7898** |

Net 14→18: same params/MAC, weights INT4 (half SRAM/DMA), **every multiply in the
neural core replaced by shift/add/mux**. The exact-descale strict core
(0.835531 / 0.790234) still implied one real multiply per output on the 12 FP8
outlier layers (the per-tensor descale); the **deploy point** makes that descale a
delayed 2-term APoT shift-add (`[-16,-6]`) and a 5-epoch recovery finetune *under*
that arithmetic lifts it to **0.8365 / 0.7912** — both gates passed and beating
the multiplier-using 14 composite (0.8354 / 0.7880) on both sets.

(MAC/frame is kept as a **multiply-equivalent algorithmic count** — the number of
multiply-accumulate operations the dataflow performs — even though each is now a
shift-add, not a hardware multiplier. It is the workload/throughput metric, not a
multiplier count.)

**pow2 block-float activation deploy (current).** The delayed-APoT deploy
(0.8365) still scaled *activations* dynamically (per-frame amax → scale → descale,
real runtime work). The block-float deploy makes every activation an **e4m3 raw
code + a single per-tensor pow2 shared exponent** (offline calibrated, `floor(log2
448/amax)`). Because the activation scale is one power of two, `LMul(code·2^k, w)
== 2^k·LMul(code, w)` exactly, so the activation descale `2^-k` fuses into each
layer's post-accumulation weight descale as a pure **exponent offset** (the weight
APoT a/b exponents shift by −k — shift-add, no multiply). Net effect: **all runtime
activation amax/descale is removed**, every quantized layer keeps exactly one
post-acc descale, and `eager==fused` is bit-exact (verified relL2 = 0 on all 48
outlier L-Mul calls), so the validated MBSTOI **is** the deployed hardware number.
A 5-epoch recovery FT under pow2 activations recovers and exceeds the general-float
calibrated point: **VCTK 0.8357 / TIMIT 0.7898** (both gates passed). Cost vs the
dynamic-activation deploy: −0.0008 VCTK in exchange for zero runtime activation
scaling. Exception: MHA `out_proj` input is the fp16 Score×V accumulator (W4A16,
INT4 shift-add) — not block-float A8; QK activation descale is applied on the QK
accumulator **before** softmax (logit temperature).

Excluded from the zero-multiplier core claim (DSP front/back-end, separate
units): STFT / iSTFT, window multiply, waveform-domain mask apply.

---

## Headline

| | Weights / params basis | MAC/frame | Multipliers in core | VCTK / TIMIT |
|---|---|---|---|---|
| Original BCCTN reference (`fft_len=512`, `embed_dim=512`) | 3,973,288 PyTorch params | ≈ 29 M MAC / 100-sample frame (~4.7 G/s) — own basis, **not** normalized to the 1.10 M deploy frame | yes | 0.8519 / 0.8114 |
| Hardware-sized FP8/QAT baseline (pre-09) | 830,300 | ~1.67 M | yes | 0.8399 / 0.7966 |
| Production (14 composite) | 337,876 | 1.10 M | yes | 0.8354 / 0.7880 |
| Strict core (18, exact descale) | 337,876 (INT4) | 1.10 M | **none** | 0.835531 / 0.790234 |
| Deploy (18, delayed APoT descale, dynamic act) | 337,876 (INT4) | 1.10 M | **none** | 0.8365 / 0.7912 |
| **★ Deploy (18, pow2 block-float act)** | 337,876 (INT4) | 1.10 M | **none** + **no runtime act scaling** | **0.8357 / 0.7898** |

From the original full-size BCCTN reference to the final RTL weight set, the
stored/computed parameter basis goes **3,973,288 → 337,876** (about **−91.5%**)
before counting the extra byte savings from INT4 storage. The quality cost versus
the original reference is about −0.016 VCTK MBSTOI / −0.022 TIMIT MBSTOI, while
the final neural core removes every general multiplier, divider, and exp LUT.

The exact-descale strict core stored the per-tensor FP8-outlier descale as an exact
fp32 scale (would need one real multiply per output on the 12 outlier layers). The
**deploy point** replaces it with a **delayed 2-term APoT descale** (`2^a±2^b`,
range `[-16,-6]`) folded into the L-Mul output — fully shift-add, no multiplier
anywhere. A 5-epoch recovery finetune *under* the delayed APoT descale
(`BCCTN_OUTLIER_DELAYED_DESCALE=1`) recovers and surpasses the exact-descale point:
**VCTK 0.8365 / TIMIT 0.7912** (both above gates 0.835 / 0.7880, and above the
exact-descale 0.8355 / 0.7902).

Current deploy ckpt (full path):
`/raid/R12K41024/18_w4a8_pow2_mixed_lmul_hw_handoff_20260530/ckpt/strict_core_pow2_blockfloat_deploy_ep3_vctk0.8357_timit0.7898.ckpt`

Key source anchors:

- Original BCCTN config/result: `/raid/R12K41024/BCCTN/config/model.yaml`,
  `/raid/R12K41024/BCCTN/results_May21/vctk_paper_style/`.
- Hardware-sized FP8/QAT baseline result: `/raid/R12K41024/main_model/results_May21/vctk_paper_style/`.
- Utilization motivation: `/raid/R12K41024/01_utilization_analysis/01_utilization_analysis_report.md`.
- W4A8 bit-width motivation: `/raid/R12K41024/02_weights_bitwidth/02_weights_bitwidth_analysis_summary.md`.
- Mixed-precision QAT design: `/raid/R12K41024/07_mixed_precision_qat/07_mixed_precision_qat_handoff.md`.
- Per-op hardware mapping: `HANDOFF_HW_ZERO_MULTIPLIER_CORE.md` (§3.4 = descale).
