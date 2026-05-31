# Architecture Progression — Condensed Results

Per stage: the hardware-architecture change, and what it buys (parameters / MAC /
quality). Quality = full paper-style VCTK & TIMIT MBSTOI at the stage it was
introduced. Gates: VCTK ≥ 0.835 (hard floor 0.8302), TIMIT ≥ 0.7880.

Two parameter views: **RTL** = weights the chip actually stores/computes;
PyTorch state_dict counts differ for masked/grouped layers (noted where it
matters). MAC = per 5 ms frame.

---

## Structural compression (folders 09 → 14)

| Stage | HW architecture change | Params | MAC/frame | VCTK | TIMIT |
|---|---|---|---|---|---|
| Baseline (untied, dense decoder) | — | 830,300 | ~1.67 M | ref | ref |
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

| | RTL weights | MAC/frame | Multipliers in core | VCTK / TIMIT |
|---|---|---|---|---|
| Baseline (09 untied) | 830,300 | ~1.67 M | yes | ~0.838 / — |
| Production (14 composite) | 337,876 | 1.10 M | yes | 0.8354 / 0.7880 |
| Strict core (18, exact descale) | 337,876 (INT4) | 1.10 M | **none** | 0.835531 / 0.790234 |
| Deploy (18, delayed APoT descale, dynamic act) | 337,876 (INT4) | 1.10 M | **none** | 0.8365 / 0.7912 |
| **★ Deploy (18, pow2 block-float act)** | 337,876 (INT4) | 1.10 M | **none** + **no runtime act scaling** | **0.8357 / 0.7898** |

The exact-descale strict core stored the per-tensor FP8-outlier descale as an exact
fp32 scale (would need one real multiply per output on the 12 outlier layers). The
**deploy point** replaces it with a **delayed 2-term APoT descale** (`2^a±2^b`,
range `[-16,-6]`) folded into the L-Mul output — fully shift-add, no multiplier
anywhere. A 5-epoch recovery finetune *under* the delayed APoT descale
(`BCCTN_OUTLIER_DELAYED_DESCALE=1`) recovers and surpasses the exact-descale point:
**VCTK 0.8365 / TIMIT 0.7912** (both above gates 0.835 / 0.7880, and above the
exact-descale 0.8355 / 0.7902).

Deploy ckpt (full path):
`/raid/R12K41024/18_w4a8_pow2_mixed_lmul_hw_handoff_20260530/ckpt/strict_core_apot_descale_deploy_ep2_vctk0.8365_timit0.7912.ckpt`
Per-op hardware mapping: `HANDOFF_HW_ZERO_MULTIPLIER_CORE.md` (§3.4 = descale).
