# Paper Feature Summary — BCCTN Binaural Speech Enhancement × Zero-Multiplier Accelerator (2026-07-07)

> Purpose: complete feature inventory before paper writing. Each step lists **what was done**
> and **what benefit it bought (with numbers)**.
> Sources: `19_algo_final/docs/RESULTS_ARCH_PROGRESSION.md` (authoritative progression),
> `ARCHITECTURE_SUMMARY_STRICT18.md`, `HANDOFF_HW_ZERO_MULTIPLIER_CORE.md`,
> `20_pesq_finetune/docs/STATUS_CHIP_PAPER_COMPARISON_POSITIONING_2026-06-28.md`,
> `演算法端_品質對齊_SPEC.md`, and per-directory README / RESULTS / HANDOFF docs.
> Chinese version: `PAPER_FEATURE_SUMMARY_2026-07-07.zh.md`.

---

## 0. One-sentence positioning (headline claim)

> **The first binaural, spatial-cue-preserving speech-enhancement accelerator**: matches/exceeds
> the best monaural SE chips on standard quality (STOI **96.85 > 95.1, CICC'26-PKU**) while uniquely
> preserving the ILD/IPD spatial cues that monaural designs cannot (MBSTOI **0.836**), in a
> **pure-digital, zero-multiplier W4A8 neural core** (every multiply in Conv/MHA/BN/PReLU/softmax
> realized as shift/add/mux — no general multiplier, no divider, no exp LUT), with the **entire
> model resident in a single 256 KiB WBUF** (no runtime DRAM weight streaming).
>
> Competitors are labeled **venue-institution** (JSSC'20-NTU / CICC'26-PKU / A-SSCC'25-Tsinghua /
> JSSC'24-SNU; P-numbering abolished, MIT VLSID'25 dropped — decided 2026-07-07).

---

## 1. Global progression table

| Stage | Weight basis | MAC/frame (multiply-equivalent) | Multipliers in core | VCTK / TIMIT MBSTOI |
|---|---:|---:|---|---|
| Original BCCTN (fft 512, embed 512) | 3,973,288 | ≈29 M (own basis) | yes | 0.8519 / 0.8114 |
| Hardware-sized FP8/QAT baseline | 830,300 | ~1.67 M | yes | 0.8399 / 0.7966 |
| 14 composite (production) | 337,876 (RTL) | 1.10 M | yes | 0.8354 / 0.7880 |
| 18 strict core (exact descale) | 337,876 (INT4) | 1.10 M | **none** | 0.8355 / 0.7902 |
| 18 deploy (delayed APoT, dynamic act) | 337,876 (INT4) | 1.10 M | **none** | 0.8365 / 0.7912 |
| **★ 18 deploy (pow2 block-float act)** | 337,876 (INT4) | 1.10 M | **none** + **no runtime act scaling** | **0.8357 / 0.7898** |

- Parameters (stored-weight basis): **3,973,288 → 337,876 (−91.5%)**, before INT4 byte halving.
- Compute: **≈29 M → 1.10 M multiply-equivalent/frame (~20–27×; bases differ, must be footnoted)**.
- Quality cost vs the original full-size reference: about −0.016 VCTK / −0.022 TIMIT MBSTOI,
  in exchange for a fully multiplier-free core.
- Deploy gates: VCTK ≥ 0.835 (deploy 0.8357), TIMIT ≥ 0.788 (deploy 0.7898) — **both passed**.
- Deploy ckpt: `18_w4a8_pow2_mixed_lmul_hw_handoff_20260530/ckpt/strict_core_pow2_blockfloat_deploy_ep3_vctk0.8357_timit0.7898.ckpt`

---

## 2. Phase 0 — Algorithm baseline and architecture downsizing (BCCTN → hardware-sized)

| Feature | What was done | Benefit |
|---|---|---|
| **BCCTN baseline** (ICASSP'24 lineage) | Complex encoder–decoder (6 stages) + complex MHA bottleneck + per-ear complex ratio masks; loss = SNR + 10·STOI + ILD + 10·IPD; L/R shared weights | Binaural enhancement that **preserves ILD/IPD spatial cues** — the project's USP from the start |
| **Depthwise-separable complex conv** | Full `(5,2)` conv → depthwise + pointwise `(5,1)` | Large conv param/MAC reduction (conv is >80% of FLOPs); the architectural basis for all later quantization |
| **Causal + sliding-window attention** (attn_win=31) | Causal padding; attention restricted to a 31-frame window | Low-latency streaming; bounded S×S softmax cost; finite KV history → small KVBUF |
| **Architecture downsizing** | `fft_len 512→128`, `win 400/100→128/80`, `embed_dim 512→256`, `heads 32→16` | Params **3.97 M → 0.83 M (−79.1%)** at only −0.012 VCTK / −0.015 TIMIT; model starts fitting the hardware budget |

---

## 3. Phase 1 — Hardware-aware analysis (01–06, no retraining, pure profiling)

| # | What was done | Key numbers | Decision bought (benefit) |
|---|---|---|---|
| **01 Utilization/bottleneck** | Per-layer MAC/DMA cycles and utilization on a 256-MAC weight-stationary array. ⚠️ Original run profiled the embed-512/32-head variant; re-profiled 2026-07-07 at the deploy config (see verification report) | Deploy basis (E=256/16): total **0.823 G MACs/s**, attention = **54.0%** of MACs (decoder 30.8%, encoder 15.2%); the historical **77.6%** is the E=512 variant. Encoder depthwise utilization 13–17% (decoder DW-transpose ~29%); W4A8 barely moves average utilization (45.1%→45.1%) | **W4A8's value is SRAM/DMA, not utilization**; primary target set to attention weight size/transport — conclusion identical on both bases |
| **01-8kHz sample-rate study** | 16 kHz vs 8 kHz-upsampled input comparison | After 8 kHz, **ILD/IPD get worse than noisy** (ILD 4.82→5.98 dB) | **Cannot drop to 8 kHz**: spatial-cue fidelity needs full bandwidth → 16 kHz locked |
| **02 Weight bit-width sweep** | Weight-only PTQ: INT8/INT4/INT3/FP4, per-layer and full-model. ✅ **Re-run at n=500/seed 42 (2026-07-07), artifacts retained** (`results_rerun_2026-07-07/`) — cite these numbers | Uniform INT4: SI-SNR **−0.38 dB** (below the 0.5 dB threshold), ILD +0.20 dB, IPD +0.56°, **DMA −50%**; INT3 collapses (**−4.36 dB**); per-layer W3 deltas each ~0.0x dB yet mixed = −2.31 dB combined (savings 61.5%); tier 29 C + 1 B reproduced exactly | **W4A8 chosen**; proved per-layer analysis is systematically optimistic (cross-layer error accumulation) → reject W3; adopt per-group/per-head scaling |
| **03 Dynamic-range analysis** | Per-tensor/channel/group/head statistics over 104 weight tensors | **5 decoder pointwise tensors (stages 0/3/4) kurtosis>5 → keep FP8** (verified); real/imag can share a scale (verified); MHA head groups non-uniform → per-head: **8/12 matrices at the deploy-aligned 16-head split** (2026-07-07 re-split, `results_regen_2026-07-07_h16/`; the report's 10/12 used a 32-way split on the 16-head model). ✅ "42/42 convs need per-channel/group" re-supported by the 2026-07-07 fixed-analyzer regen (`results_regen_2026-07-07/`, worst var_max 0.526/0.494; group-size means exactly reproduce the report's table) | Direct quantization recipe: PW per-group, DW per-channel, MHA per-head, outlier layers FP8, shared re/im scale (scale overhead <1%) |
| **04 Sparsity analysis** | Weight/activation/attention-score sparsity at multiple thresholds | Weight sparsity is random (unstructured, 43/52 layers); all 16 heads contribute; no dead positions inside the W=31 window; PReLU activation sparsity ~19–62% but dynamic (note: the report's "6 dense = Q/K/V/O" is mislabeled — 5 conv + 1 MHA) | **Negative result that saved hardware cost**: no zero-skip/pruning hardware; effort concentrated on W4A8 and attention |
| **05 Conv reuse / DMA breakdown** | Cross-layer weight similarity, SVD low-rank, temporal reuse, DMA share | No shareable conv weights; conv kernel_t=1 → no cross-frame reuse; **MHA weight DMA 640 KB ≈ 4× all pointwise convs combined** | Confirms **MHA is the DMA bottleneck**: prioritize W4 on MHA (4×); DW gets a freq-row sliding buffer |
| **06 Attention mixed precision** | Per-sub-op dynamic range + injected fake-quant + full-model top-k eval. ✅ **Re-run at n=20 (2026-07-07)** closing the audit's n=2 gap (n=2 originals preserved in `results_n2_2026-04-16/`) | **QK^T \|max\| = 1049 > e4m3 448** (n=2 probe: 911 — conclusion strengthened) → needs FP16 accumulator; **12/16 sharp heads confirmed at n=20**; top-4 keeps prob-row L1 error low | **Top-4 sparse attention** is the one recommendation that survived to deploy (quality re-confirmed at scale in 07: 0.8340 / 14: 0.8360); the "int16 accumulator / all-fp16" table was superseded by 13/14/18 (FP16 accumulator + W4A8 + L-Mul m8) |

---

## 4. Phase 2 — Quantization + structural compression (07–14)

Quality reference: VCTK noisy MBSTOI 0.7861; FP8/QAT baseline 0.8399.

| # | Feature | What was done | Benefit (numbers) |
|---|---|---|---|
| **07/08** | **W4A8 mixed-precision QAT recipe** | INT4 weights (PW per-group-32, DW per-channel, MHA per-head) + FP8 e4m3 activations + fp16 accumulator + top-4; decoder PW outliers (stages 0/3/4) kept FP8 | MBSTOI 0.8340 (gsz32+topk4); weight memory vs 512-dim FP8 **−82% DMA** (2.80 MB→487 KB, ~5.75×); fp16 accumulation and top-4 proven "free" (Δ<0.001) |
| **09** | **Weight-tied complex MHA** ★ architectural novelty | `MHA_re ≡ MHA_im`: one weight set; complex output composed from two real passes as `(a−b)+i(a+b)` | MHA params **526,336→263,168 (−50%)**; real passes **4→2**; softmax serializations 4→2; quality **improves** (finetuned 0.8411 > untied 0.8397; strict W4A8 0.8383); total params 830,300→**567,132 (−31.7%)** |
| **09→11** | **Exp-free shift-top4-pow2 softmax** (`shift_topk_pow2_clamp8_nomantissa`) | max-shift normalization → top-4 select → prob = `2^clamp(⌊score⌋,−8,0)` (no mantissa) → pow2 denominator shift-divide | **Removes exp LUT, divider, multiplier**; needs only a top-k selector + adder/subtractor + barrel shifter + LZD; MBSTOI 0.8360 (Δ−0.0013 vs exact softmax); robust cross-domain on TIMIT (+0.0616 enhancement, no collapse) |
| **10** | **Fixed-point STFT/iSTFT** | STFT front/back-end in 16-bit fixed point (Q1.14/Q5.10) and an fp8×fp8 sweep | fp8×fp8 STFT MBSTOI 0.8170 ≈ fp32 0.8153; 16-bit fixed point matches float to 4 decimals → **cheap fixed-point FFT hardware justified** |
| **11** | **Parallel MBSTOI evaluation infrastructure** | Split GPU inference from CPU metric computation via a process pool | **~13× eval speedup** (byte-identical results) → made the large softmax/QAT sweeps tractable (methodology contribution, implementation-details material) |
| **12** | **Decoder grouped pointwise + skip subset** | Split PW into x-branch/skip-branch, each grouped (**g42**: groups_x=4, groups_skip=2); keep only **top-50%** skip channels by L1 (static index → ROM) | decoder PW MAC **−74.7%**, all-conv MAC −46.5%; whole model **1.67→1.44 M MAC/frame (−14%)**; params 567,132→**485,332 (−14.4%)**; skip bandwidth halved; quality 0.8352/0.7919 (within run-to-run noise) |
| **13** | **Attention datapath narrowing + L-Mul m4 feasibility** | (a) all attention scratch in fp16; (b) QK^T and Score×V via **L-Mul** (logarithmic multiply: exponent add + mantissa add, no full multiplier) | fp16 accumulation 0.8352/0.7914 (free); L-Mul m4 after QAT **0.8364/0.7925 (above baseline)**; ~**−70% multiplier energy** (MAC count unchanged); score-domain quantization systematically ruled out (double discretization) |
| **14** | **Grouped QKV (qkv_groups=4) + top-4 q0.8 sparse packet** ★ production stack | Keep only diagonal 64×64 Q/K/V blocks (30-epoch QAT); pack softmax probs into a 4-entry q0.8 packet (8-bit prob + 5-bit local index) | QKV MAC/params **−75%** (RTL 196,608→49,152); whole model **1.39→1.10 M MAC/frame (−21%)**; 64 lanes@5 MHz duty **1.02→0.81 (failing→passing)**; QKV weights 96 KB→24 KB; prob row **512→52 bits (−89.8%)**; composite **0.8354 / 0.7880, both gates passed**; total RTL weights **337,876 (−59.3% vs 09)** |

---

## 5. Phase 3 — All-L-Mul exploration & quality recovery (15–17, exploratory, NOT in deploy)

| # | What was done | Result | Meaning for the paper |
|---|---|---|---|
| **15 all-L-Mul eval** | Pushed L-Mul m4 to **every** Conv/ConvTranspose/Linear/MHA multiply | Best after QAT/finetune: VCTK 0.8264–0.8281, still **~0.009** below gate | Defines the L-Mul applicability boundary: **L-Mul for attention + FP8 outliers, INT4 shift-add for everything else** is the deployable combination |
| **16 KD/heatmap recovery** | Per-layer error heatmap, output-level knowledge distillation (teacher = production), bias correction | Heatmap hotspots = deepest decoder PW; best KD recovery **0.8288** (still ~0.007 short); bias correction ineffective | Ablation evidence: the all-L-Mul gap is not caused by accumulator/packet/top-k (fp32 accumulator gives no lift), supporting the deploy's mixed-multiplier design |
| **17 L-Mul-aware STE** | Three L-Mul backward variants (identity / autograd_lmul ≈0.675× gradient damping / manual grad_scale), forward bit-exact | Smoke-verified, **not launched** (contingent backup) | Future-work material; suggests the recovery plateau may sit in the STE gradient path |

---

## 6. Phase 4 — Strict zero-multiplier core (18, the deploy model)

Topology/MAC identical to 14 (337,876 / 1.10 M); every item below turns a multiply into shift/add/mux:

| Feature | What was done | Benefit |
|---|---|---|
| **APoT weight scales** (`2^a ± 2^b`, exp∈[−16,−1]) | per-group/per-channel/per-head scales converted from a real multiply to 2 shifts + 1 add | Scale multipliers eliminated; 0.8350/0.7892 (lossless) |
| **Conv INT4 shift-add** | INT4 code × e4m3 activation = CSD shift-adds on the significand (\|q\|≤7 ⇒ ≤2 add/sub); descale delayed to once-per-output after accumulation | Conv multipliers eliminated; algebraically equivalent to per-weight dequant (bit-exact under wide accumulation) |
| **L-Mul m8 (QK/Score×V) + m4 (12 FP8 outlier layers)** | FP multiply → sign-XOR + exponent add + one (m+2)-bit mantissa add | Attention/outlier multipliers eliminated (m4 = 6-bit adder, m8 = 10-bit adder) |
| **BN fold → APoT[−9,4]** | BN folded to `a_q·x + b` (signed APoT); conv bias fused into BN bias | BN multiplier → 2 shifts + 1 add |
| **PReLU pow2 slope** (`±2^k`, k∈[−8,0]) | Slope quantized to a signed power of two | PReLU multiplier → sign-mux + 1 shift |
| **Delayed 2-term APoT descale** (outlier layers, [−16,−6]) | Per-tensor descale of the 12 FP8 outlier layers converted from an exact fp32 multiply to a delayed APoT shift-add, with a 5-epoch recovery finetune under that arithmetic | The last multiplier removed; **0.8365/0.7912, beating the multiplier-using 14 composite (0.8354/0.7880)** |
| **★ pow2 block-float activations** | Activations = e4m3 raw code + **per-tensor pow2 shared exponent** (offline calibration `k=⌊log2(448/amax)⌋`, 53 boundaries); the `2^-k` descale is **fused** into the post-acc weight descale as an exponent offset | **All runtime amax/scale/descale removed**; `LMul(code·2^k,w)==2^k·LMul(code,w)` holds exactly → **eager==fused bit-exact (relL2=0 on all 48 outlier L-Mul calls)** → the software MBSTOI *is* the silicon number |
| **Training under the deployed arithmetic** (QAT/recovery finetune) | The final model is finetuned under arithmetic bit-exact to the hardware (L-Mul, delayed APoT, pow2 block-float) | Quality gates hold on the true hardware path, not an approximation; **deploy 0.8357 / 0.7898** |

**Core claim**: the neural core (Conv/MHA/BN/PReLU/softmax of encoder+MHA+decoder) has
**zero general multipliers, zero dividers, zero exp LUTs**. Out of scope (separate DSP
front/back-end): STFT/iSTFT, windowing, waveform-domain mask application.

---

## 7. Phase 5 — Hardware handoff & RTL (18_handoff / 19_algo_final)

| Feature | Content | Status / benefit |
|---|---|---|
| **Handoff package `hw_patterns_final/`** | `tb_pattern/` (packed INT4/FP8 weight codes, APoT/pow2 scale codes, fused descale, per-layer block-float activations, BN/PReLU constants, skip indices; manifest: 63 weight layers, 12 FP8 outlier, 54 INT4, missing=0) + `sram_images/` (`wbuf_unified_init.bin` + memory map) + golden `.hex` | Single source of truth for RTL; `18_hw_handoff` keeps the ckpt and the full regeneration toolchain |
| **Whole model resident on-chip** | WBUF 256 KiB single-port: weights 240,336 B (INT4 223,312 + FP8 17,024) + constants ~10 KB = **251,677 B (95.7% full)** | **No runtime DRAM weight streaming** — achieved jointly by −91.5% params + INT4 + tied MHA + grouped QKV; kills the original DMA bottleneck outright |
| **CE datapath** | `PEG×4 → (fused APoT descale) dealer×4 → OACC (2-bank complex accumulator) → PPU_fma`; 64 complex lanes (128 real); one configurable add/shift slice covers all 4 modes (m4 / W4A8 / m8 / W4A16) | No dedicated multiplier in any mode; complex multiply = 4 PEG streams + OACC add/sub composition |
| **Shared PPU** | Descale+BN+PReLU+requant-to-e4m3 in one shared post-MAC unit. ⚠️ The RTL implements a **single complex PPU** (one re+im pair; throughput sufficient at 1 complex output per ⌈ic/32⌉ cycles) — the "~8-lane pool" was an early algorithm-side sizing suggestion; the paper describes the RTL | Area efficiency: attention (the largest compute block — 54.0% on the deploy basis; ~78% on the old E=512 basis) has no BN/PReLU, so dedicated lanes would idle |
| **Buffer design** | KVBUF 32 KiB (dual-port ring; tied MHA's 2 passes ⇒ 2× K/V history), UFBUF 8 KiB, SKIPBUF 4 KiB (top-50% skip) | Memory hierarchy frozen (2026-06-03 SRAM spec) |
| **RTL verification status** (updated 2026-07-07 per HW side) | Bit-exact golden ladder all green: PEG 16,000 vec × 4 modes 0 err; per-layer goldens; chained MHA; **full-frame e2e + boot e2e bit-exact**; T6 real TSMC SRAM macros in; multi-mode tapeout build; SpyGlass clean | **Measured frame latency 28,831 cy = 1.442 ms warm / 31,554 cy = 1.578 ms cold @ 20 MHz** (~29–32% of the 5 ms budget). Only remaining gap: **synthesis/measured area & power (comparison-table gap)** |

---

## 8. Phase 6 — Paper quality positioning (20_pesq_finetune)

| Feature | What was done | Result / claim |
|---|---|---|
| **VoiceBank+DEMAND alignment eval** | 824 files, standard protocol, binaural render (az0/az70), deploy model zero-shot | **STOI 0.9685 (96.85) > 95.1 (CICC'26-PKU) > 91.44 (JSSC'24-SNU)** — highest among all chips reporting STOI; PESQ nb 2.89→3.01 / wb 2.00→2.27; SI-SDR 17.9→17.1 (healthy) |
| **PESQ↔spatial-cue tradeoff ablation** (dose-dependent, proven) | Warm-start from deploy with increasing PESQ loss weight | w=0: MBSTOI 0.834/ILD 3.33/IPD 16.0 → w18: 0.823/3.44/17.7 → w30: 0.773/3.98/21.6. **Higher PESQ weight monotonically wrecks spatial cues** → "quantization preserves cues; PESQ finetuning is what erodes them"; turns the weakness into evidence of design intent |
| **Honest PESQ framing** | Report wb/nb separately labelled; **never** put our nb next to CICC'26-PKU's wb 3.34; cite that the original BCCTN paper reports no PESQ (its objective is spatial-cue preservation, not PESQ — full paper-ready narrative in `FEATURES_en.md` E6) | High-PESQ variants (mix_ep19 nb 3.324) collapse SI-SDR to 11.9 and MBSTOI to 0.802 → **report deploy, not a gamed point** |
| **Quantization does not sacrifice spatial fidelity** (same-test-set, same-metric A/B) | Official BCCTN FP32 ckpt (11M params) vs our W4A8 deploy on identical 64 VCTK files + identical metric | **IPD 18.0° vs FP32 17.3° (tie), ILD 3.76 vs 4.59 dB** → publishable claim: "the W4A8 zero-multiplier core preserves interaural cues on par with the FP32 reference"; also proved the paper's 0.4 dB/5° is authors'-test-set-specific — **absolute ILD/IPD is not cross-test-set comparable** |
| **Competitor positioning table** (labels: venue-institution, P-numbering abolished 2026-07-07; MIT VLSID'25 dropped — not a top venue) | JSSC'20-NTU (Lee) / **CICC'26-PKU (Zhi; VB+DEMAND, PESQ wb 3.34/STOI 95.1, D-CIM)** / A-SSCC'25-Tsinghua (Wu; visual-aided) / JSSC'24-SNU (Park; multiplier-less mono) | All **four** competitors are **monaural with zero spatial cues** → MBSTOI/ILD/IPD uncontested; hardware differentiation: pure-digital zero-multiplier (more portable than CICC'26-PKU's D-CIM; vs JSSC'24-SNU we run the much harder binaural transformer) |

---

## 9. Claim list ready for the paper

1. **First binaural, cue-preserving SE accelerator** (all 4 compared chips — JSSC'20-NTU, CICC'26-PKU, A-SSCC'25-Tsinghua, JSSC'24-SNU — are monaural).
2. **Zero-multiplier neural core**: Conv/MHA/BN/PReLU/softmax all shift/add/mux; no multiplier, divider, or exp LUT.
3. **All weights on-chip**: 337,876 weights (INT4+FP8) + constants fit one 256 KiB WBUF (95.7% full).
4. **Static calibration, zero runtime activation scaling**: pow2 block-float + fused exponent offset; eager==fused bit-exact (relL2=0) → **the reported MBSTOI is the silicon number**.
5. **STOI 96.85 > 95.1 (CICC'26-PKU) > 91.44 (JSSC'24-SNU)** (same-dataset direct comparison for the PKU chip); MBSTOI 0.836 + ILD/IPD fidelity as unique metrics.
6. **Compression numbers**: params −91.5% (3.97 M→338 K), MACs −20~27× (29 M→1.10 M/frame), attention probability transport −89.8% (512→52 bits/row), MHA weight storage 512 KB→24 KB (QKV).
7. **Tied complex MHA**: −50% MHA params/softmax serializations with a quality *gain* (0.8411 > 0.8397) — the architectural novelty.
8. **Quantization does not sacrifice spatial fidelity**: W4A8 vs FP32, same test set: IPD 18.0° vs 17.3°.
9. **PESQ↔spatial-cue tradeoff ablation**: dose-dependent evidence supporting the "we chose to preserve spatial cues" narrative.
10. **Trained under the deployed arithmetic**: QAT/recovery finetuning ran directly under bit-exact hardware arithmetic; the deploy point beats the multiplier-based production stack (0.8365/0.7912 vs 0.8354/0.7880; pow2 block-float variant 0.8357/0.7898).

---

## 10. Writing caveats (align these before quoting numbers)

1. **Do not mix parameter bases**: analyses 01/02 used the old 512-dim/32-head variant (3.97 M); 03/04/06 used 256-dim; dir 07's `unified_analysis_report` computes 2.54 M on yet another counting basis. **The paper should quote the `RESULTS_ARCH_PROGRESSION.md` series** (3,973,288 → 830,300 → 567,132 → 485,332 → 337,876 RTL). The per-directory `model_size.md` files are stale boilerplate — do not cite them.
2. **WBUF capacity**: early analysis docs mention 64 KB/32 KB targets; the **frozen design is 256 KiB with full residency** (2026-06-03 SRAM spec supersedes; the HANDOFF §4/§9 streaming narrative is stale).
3. **MAC-number bases**: the original 29 M/frame is profiled on its own hop-100/fft-512 basis and is **not directly comparable** to the deploy's 1.10 M (hop-80/fft-128); footnote any plot.
4. **Group size**: analysis 03 recommended gsz=256, but the deploy uses **gsz=32** (the sweet spot from the 07 sweep). RESOLVED 2026-07-07: 03's `results_0410/summary.md`/CSV BN/PReLU rows are pre-bugfix artifacts that invert the trend; restricted to conv+MHA weight tensors (n=56) the correct trend is per-tensor 7.63e-4 → gsz16 2.76e-4 (best) → gsz32 2.93e-4 → gsz256 3.36e-4. Never cite "gsz256 = 0.28×, best".
5. **Window inconsistency — RESOLVED 2026-07-07**: training and RTL both use **Hamming** (RTL bit-exact vs deploy, T4); `FUSIONS.md`'s "Hann" was a doc error and now carries a dated erratum.
6. **Absolute ILD/IPD is not cross-test-set comparable** (the original paper's 0.4 dB/5° is specific to its unreleased test set); report improvement-over-noisy and same-test-set FP32-vs-W4A8 preservation instead.
7. **The consolidated per-SNR table — LOCKED 2026-07-07**: `18_..._hw_handoff_20260530/results/apot_descale_eval/ft_pow2_ep3_full/per_snr_locked_paper_table.csv` (VCTK+TIMIT × 8 SNR × noisy/enh/Δ × 5 metrics; per-SNR means reproduce 0.8357/0.7898; canonical averages = the `*_averages.txt`). Per-SNR STOI/ESTOI/SI-SDR on VCTK/TIMIT would still need a re-run if wanted.
8. **Chip power/energy/area/node numbers are TBD** pending RTL bring-up — the open comparison-table gap.
9. **All-L-Mul (dirs 15–17) is NOT in the deploy** (exploratory); deploy L-Mul is only on QK/Score×V (m8) and the 12 FP8 outlier layers (m4) — do not conflate.
10. **All PESQ-leaning finetunes fall below the MBSTOI gate** → the final reported model = deploy (the low/no-PESQ-pressure operating point).
11. **"Attention = 77.6% of MACs" is the embed-512 variant, not the deploy model.** Deploy basis (E=256/16, re-profiled 2026-07-07): total 0.823 G MACs/s, attention 54.0 / decoder 30.8 / encoder 15.2. "Attention is the largest block" holds on both bases; label whichever basis you quote. ⚠️ The re-profile is a **third MAC basis** (complex→4-real convention, pre-09 untied architecture; ÷200 fps ≈ 4.1 M/frame) — NOT convertible to the 1.10 M multiply-equivalent/frame deploy number; use it for percent shares only, never divide the two. Full audit: `RESULTS_ANALYSIS_VERIFICATION_01_06_2026-07-07.md`.
12. **Analysis 06 re-run at n=20 (2026-07-07)** — QK^T 1049>448 and 12/16 sharp heads confirmed; cite the n=20 numbers. Its IPD ~1.4 rad remains a saturated raw-STFT metric — never place next to the 15–21° eval numbers. Cite 07/14 for top-4 quality.
13. **05 low-rank "+18–27%" savings hold only at the 95%-energy threshold** (all flip negative at 99%); 05's "num_heads=32" is a doc error (actual 16, no numbers affected); MHA 640 KB counts outer complex projections real-only (~768 KB fully complex).
14. **04's "6 dense = Q/K/V/O projections" is mislabeled** — 5 of the 6 are conv layers; the sparsity conclusion itself is verified and safe.
15. **Paper figures**: regenerated 2026-07-07 with corrected data in `paper_figures/` (SVG + EMF for Visio + PNG); per-figure sources and mandatory caption caveats in `paper_figures/CAPTIONS.md`.
