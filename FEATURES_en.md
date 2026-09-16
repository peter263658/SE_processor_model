# BCCTN Speech-Enhancement Accelerator — Consolidated Feature Inventory (English)

> **Version v2 — 2026-07-07.** v1 (repo docs + RTL state) has been merged and cross-verified
> against the **algorithm consultant's list** (`Journal/PAPER_FEATURE_SUMMARY_2026-07-07.md`,
> cited as [Consultant]). New series: **P** (design-space analyses), **X** (exploratory, NOT in
> deploy), **E** (quality positioning / competitors).
> Chinese version: `FEATURES_zh.md` (IDs correspond one-to-one). Conflicts & open items: §9/§10.

---

## 0. System at a Glance (Headline)

**One-sentence positioning** [Consultant §0]:
> **The first binaural, spatial-cue-preserving speech-enhancement accelerator**: matches/exceeds
> the best monaural SE chips on standard quality (STOI **96.85 > 95.1 (CICC'26-PKU) > 91.44
> (JSSC'24-SNU)**) while uniquely preserving ILD/IPD spatial cues (MBSTOI **0.836**); a
> **pure-digital zero-multiplier W4A8 neural core** (every multiply in Conv/MHA/BN/PReLU/softmax
> realized as shift/add/mux — no general multiplier, divider, or exp LUT); the **entire model
> resident in one 256 KiB WBUF** (no runtime DRAM weight streaming).

**Competitor labeling convention (decided 2026-07-07, replaces all P1–P5 numbering):**
competitors are referred to as **venue-year + institution** — **JSSC'20-NTU** (Lee et al.),
**CICC'26-PKU** (Zhi et al.), **A-SSCC'25-Tsinghua** (Wu et al.), **JSSC'24-SNU** (Park et al.).
This kills the P-number ambiguity between the SPEC and STATUS docs, and avoids collision with the
P-series analysis IDs below. The **MIT VLSID'25** paper is **dropped from the comparison** (not a
top venue; no PDF on file). Comparison set = these **four** monaural chips.

| Item | Value |
|---|---|
| Task | Binaural SE via complex ratio mask, preserving ILD/IPD; loss = SNR + 10·STOI + ILD + 10·IPD |
| Real-time | 16 kHz, fft 128 / hop 80 = **5 ms/frame** hard deadline (8 kHz rejected — see P2) |
| Model | BCCTN (ICASSP'24 lineage): complex U-Net (6 stages) + tied complex MHA bottleneck |
| Parameters | **337,876 (RTL)**, from 3,973,288 → **−91.5%** (before INT4 byte halving) |
| MAC / frame | ~29 M → **1.10 M** (multiply-equivalent; bases differ — must footnote, §9-3) |
| Numerics | **W4A8**: INT4 + FP8 E4M3 block-float (static pow2 exponents, zero runtime scaling); FP16 accumulation |
| Core claim | **Zero-multiplier neural core** + **eager==fused==HW bit-exact (relL2=0)** → software MBSTOI *is* the silicon number |
| Quality (MBSTOI) | deploy **VCTK 0.8357 / TIMIT 0.7898** (gates 0.835/0.788 both passed; HW-chain-certified 0.8356/0.7897) |
| Quality (standard protocol) | VoiceBank+DEMAND zero-shot: **STOI 0.9685** (> CICC'26-PKU 0.951 > JSSC'24-SNU 0.9144), PESQ nb 2.89→3.01 / wb 2.00→2.27, SI-SDR 17.9→17.1 [Consultant §8] |
| Process / clock | TSMC 28nm ("HPC+" qualifier not found in docs — confirm before citing), **20 MHz** (PERIOD = 50 ns) |
| Compute array | **128 real lanes = 64 complex lanes** (4 PEGs × 32; 64 lanes in 16-bit modes), precision-reconfigurable |
| On-chip SRAM | **300 KiB**: WBUF 256 (95.7% full) + KVBUF 32 + UFBUF 8 + SKIPBUF 4; **zero weight DMA** at inference |
| Frame latency | warm **28,831 cy = 1.442 ms**; cold boot **31,554 cy = 1.578 ms** (~29–32% of 5 ms) |
| Chip I/O | 148 pads; 128-bit load stream / 8-bit byte-serial output; 6 measurement modes + MISR debug |
| Pipeline | **STFT → enc → MHA → dec → mask → iSTFT all on-chip** (overlap-add off-chip) |
| Area / power | **TBD** (backfill after synthesis/measurement — the one open comparison-table gap [Consultant §10-8]) |

---

## 1. Algorithm / Model-Architecture Features (A-series)

**A1 — Complex binaural BCCTN topology (ICASSP'24 lineage).** STFT → 6-stage complex depthwise-separable conv encoder → tied complex MHA bottleneck → 6-stage complex transpose-conv decoder → per-ear complex ratio masks → iSTFT. **Loss = SNR + 10·STOI + ILD + 10·IPD**; L/R share weights, concatenated only at the bottleneck (per-ear C·F=128, L‖R=256) → spatial-cue preservation is the project's USP from the start. [HANDOFF §8, Consultant §2]

**A2 — Complex conv = 4 real sub-convolutions.** `y_re = fr(x_re) − fi(x_im)`, `y_im = fr(x_im) + fi(x_re)`; each real sub-weight is reused with both activation branches (batch-of-2 weight reuse). [HANDOFF §3.1a]

**A3 — Causal sliding-window MHA.** embed_dim=256, 16 heads, head_dim=16, attn_win=31 (≤31 past + 1 current = 32 entries); streaming: one query per frame; the only cross-frame state is a 31-frame K/V ring → bounded KVBUF. [HANDOFF §8]

**A4 — Tied complex MHA (2 real passes + recombine) ★ architectural novelty.** `MHA_re ≡ MHA_im = M`: run twice (a=M(x.re), b=M(x.im)), recombine `(a−b)+i(a+b)`. **MHA params 526,336→263,168 (−50%)**, real passes 4→2, softmax serializations 4→2, total params 830,300→567,132 (−31.7%); quality **improves** (finetuned 0.8411 > untied 0.8397). Each pass keeps its own K/V history → 4-path KV in hardware (H7). [Consultant §4-09, 09_tied_mha]

**A5 — Block-diagonal QKV projection (qkv_groups=4, 30-epoch QAT).** Only the diagonal 64×64 blocks (compact `[12,64,64]`); out 768=[Q|K|V] head-major, 48 per-head APoT scales, bias=True. **QKV MAC/params −75%** (RTL 196,608→49,152), attention −28%, model 1.39→1.10 M (−21%); **64 lanes @5 MHz duty 1.02→0.81 (failing→passing)**; QKV weights 96→24 KB. [Consultant §4-14]

**A6 — Grouped decoder split-pointwise + static top-50% skip.** `pointwise_x(groups=4) + pointwise_skip(top-50%-by-L1 subset, groups=2)` (fixed indices → ROM) → depthwise-transpose (freq ×2) → BN+PReLU. Model 1.67→1.44 M MAC (−14%), params 567,132→485,332 (−14.4%), skip bandwidth halved; quality 0.8352/0.7919 (within run noise). ✅ **−62% vs −74.7% RESOLVED (2026-07-07)**: same counting basis (conv-only module-call MAC, hooks on Conv2d/ConvTranspose2d, excl. MHA Linear/attention/STFT), different feature scope — **−62.16% = grouping alone (P1 g42: dec_pw 37,698,560→14,264,320)**; **−74.74% = grouping + top-50% skip subset (P2 g42+s050_topk: →9,526,528; all-conv −46.52%) = the deploy configuration → the paper cites −74.7%** (optionally decomposed). The whole-model 1.67→1.44 M is the separate normalized basis incl. MHA — don't put on the same axis. [PROG stage 12, Consultant §4-12, RESPONSE_OPEN_ITEMS §1]

**A7 — Frequency-only depthwise-separable convs.** Full (5,2) conv → depthwise + pointwise (5,1) (the main cut on conv's >80% FLOP share); stride (2,1); conv stages frame-independent. Complex channels [16,32,64,128,256,256]. [Consultant §2, HANDOFF §8.1]

**A8 — Top-4 sparse attention + q0.8 packet.** 4-entry sparse probability packet (8-bit q0.8 + 5-bit local index): **probability row 512→52 bits (−89.8%)**; effective Score×V positions 31→4 (−87% attention-side compute); near-zero quality impact (SI-SNR Δ<0.02 dB). Current frame = local_idx 0. [Consultant §3-06/§4-14]

**A9 — Architecture downsizing (fft 512→128, win 400/100→128/80, embed 512→256, heads 32→16).** Params **3.97 M→0.83 M (−79.1%)** for only −0.012 VCTK / −0.015 TIMIT — the model starts fitting the hardware budget. [Consultant §2]

**A10 — Full quality-progression table (every step's cost quantified).** See §0 and `RESULTS_ARCH_PROGRESSION.md`: 0.8519 (3.97 M original) → 0.8399 (HW-sized) → … → **0.8357/0.7898 (strict-18 block-float deploy)**. Total cost vs full-size original ≈ −0.016 VCTK / −0.022 TIMIT, in exchange for a fully multiplier-free core. [Consultant §1]

---

## 2. Design-Space Analysis Features (P-series, dirs 01–06+10: profiling that bought decisions) [Consultant §3]

**P1 — Utilization / bottleneck (01).** Per-layer profiling on a 256-MAC weight-stationary array. ⚠️ **Audit 2026-07-07: the original run profiled the embed-512/32-head variant (profiler hardcodes 512), NOT the deploy model.** Re-profiled at the deploy config (E=256/16, same analytical model, 512 numbers reproduced first): total **0.823 G MACs/s**; attention = **54.0% of MACs** (dec 30.8% / enc 15.2%). The historical 77.6/15.0/7.4 split is the E=512 basis — label whichever basis is quoted. Encoder depthwise utilization 13–17% (decoder DW-transpose ≈29%, the report's "all 13–17%" was wrong); W4A8 barely moves average utilization (45.1→45.1% at deploy basis) → **W4A8's value is SRAM/DMA, not utilization** (MHA weights 512→256 KB); the primary target became attention weight size/transport — conclusion identical on both bases. Per-layer rows: `paper_figures/data/util_rows_e256.csv`; full audit: `RESULTS_ANALYSIS_VERIFICATION_01_06_2026-07-07.md`. **Basis warning (third MAC number):** 0.823 G/s ÷ 200 fps ≈ 4.1 M/frame on the profiler's own convention (complex→4-real ×4, QK/Score×V ×8, and the *pre-09 untied dense-QKV architecture*) — it is NOT convertible to the 1.10 M multiply-equivalent/frame deploy number (different counting convention AND different architecture stage; the same normalized basis puts the pre-09 baseline at 1.67 M). Use the re-profile **only for shares/bottleneck attribution**, never divide it against 1.10 M (§9-23).

**P2 — 8 kHz sample-rate rejection (01-8kHz).** With 8 kHz-upsampled input, **ILD/IPD become worse than noisy** (ILD 4.82→5.98 dB) → spatial cues need full bandwidth; **16 kHz locked**. (A negative result that killed a wrong direction.)

**P3 — Weight bit-width sweep (02).** Weight-only PTQ INT8/INT4/INT3/FP4 → **W4A8 chosen**; per-layer analysis proved systematically optimistic (cross-layer error accumulation) → reject W3, adopt per-group/per-head scaling. ✅ **Re-run at n=500/seed 42 (2026-07-07, artifacts retained in `results_rerun_2026-07-07/`)** — cite these numbers: uniform INT4 = SI-SNR **−0.38 dB** (below the 0.5 dB threshold), ILD +0.20 dB, IPD +0.56°, **DMA −50%**; INT3 collapses (**−4.36 dB**; per-layer W3 deltas each ~0.0x dB yet mixed 29×W3+1×W4 = −2.31 dB combined, savings 61.5%); tier count **29 C + 1 B reproduced exactly** (settles the report's internal 29+1-vs-30 contradiction). The original report's absolute values (baseline 10.49 etc.) came from a different eval subset and remain historical-only.

**P4 — Dynamic-range analysis (03).** Stats over 104 weight tensors: **5 decoder-PW tensors (stages 0/3/4) with kurtosis>5 → keep FP8** (verified exact against `dynamic_range_table.csv`; expands to 12 tensors in deploy: r/i × x/skip); real/imag can share a scale (std_ratio 0.988/1.003, verified); MHA head groups non-uniform → per-head: **8/12 matrices at the deploy-aligned 16-head split** (`results_regen_2026-07-07_h16/`, worst output_proj.imag 5.6e-3; fig05 uses this) — the original report's 10/12 (its text says 9) was a 32-way split on the 16-head model; quote the 16-head number. Produced the quantization recipe directly (scale overhead <1%). ✅ **Group-size contradiction RESOLVED (audit 2026-07-07)**: the report's trend (finer group = lower MSE) is correct; `results_0410/summary.md`'s "gsz256 = 0.28×, best" averages in BN/PReLU rows produced by a pre-bugfix analyzer (non-physical MSE-halving with group size) — never cite it. Recomputed on conv+MHA weight tensors only (n=56): per-tensor 7.63e-4 → gsz16 2.76e-4 (best) → **gsz32 2.93e-4 (deployed, from the 07 QAT sweep)** → gsz256 3.36e-4. ✅ **"42/42 convs need per-channel/group" re-supported (2026-07-07)**: the fixed analyzer was re-run (`results_regen_2026-07-07/`) — `per_channel_variance.csv` now exists with **42/42 conv tensors var_max > 1e-4** (worst 0.526/0.494, matching the report), and the regen's all-row group-size means **exactly reproduce the report's published table** (per-tensor 1.314e-3 → gsz16 9.572e-4 (0.73×) → gsz32 1.048e-3 → gsz256 1.202e-3, monotone) — the report is now fully artifact-backed; cite the regen dir.

**P5 — Sparsity analysis (04; a cost-saving negative result).** Weight zeros are random (unstructured; 43/52 layers); all 16 heads contribute; no dead positions within the W=31 window; PReLU activation sparsity ~19–62% but dynamic → **no zero-skip/pruning hardware**; effort concentrated on W4A8 + attention. ✅ Audit 2026-07-07: all headline numbers reproduce from `results/` (authoritative dir; `results_test/` is an earlier partial run); the repo-TODO "still has bugs" note post-dates the results but no residual computational bug was found. One wording fix: the report's "6 dense = Q/K/V/O projections" is mislabeled — 5 of the 6 dense layers are convs, 1 is MHA output_proj.

**P6 — Conv reuse / DMA breakdown (05).** No shareable conv weights (adjacent *pointwise* cosine ≤0.556; adjacent depthwise reaches 0.62 — say "pointwise"); kernel_t=1 → no cross-frame reuse; **MHA weight DMA 640 KB ≈ 4× all pointwise convs combined** → MHA confirmed as THE DMA bottleneck; W4 prioritized on MHA; DW gets a freq-row sliding buffer. ✅ Audit 2026-07-07: arithmetic verified; the 640 KB counts the outer complex projections real-only (fully complex ≈768 KB — dominance claim unaffected); the report's "num_heads=32" is a doc error (actual 16; no number depends on it); low-rank "+18–27%" savings hold **only at the 95%-energy threshold** (all flip negative at 99%) — do not quote as free savings.

**P7 — Attention mixed precision (06).** **QK^T dynamic range exceeds FP8** → wide accumulation required; 12/16 heads are sharp; top-k=4 leaves the probability-row L1 error negligible. ✅ **Re-run at n=20 (2026-07-07, closing the audit's n=2 gap)**: QK^T |max| = **1049 > 448** (n=2 probe said 911 — conclusion strengthened), Q ±51 / K ±71 / V ±104, **12/16 sharp heads confirmed**; artifacts in `06_attn_mised_precision/results/` (n=2 originals preserved at `results_n2_2026-04-16/`). Its "int16 QK^T accumulator / all-fp16" recommendation was **superseded** by the deployed FP16-accumulator + W4A8 + L-Mul m8 path (dirs 13/14/18); **top-4 is the one recommendation that survived**, and its quality evidence should be cited from the at-scale evals (07: 0.8340 full paper-style; 14: packet 0.8360), not from 06. Also: 06's IPD ~1.42–1.46 rad is a saturated all-bin raw-STFT metric (constant across k) — never place next to the 15–21° eval numbers; 06's `results/summary.md` "V has smaller range/std than Q" is inverted vs its own data.

**P8 — Fixed-point STFT/iSTFT feasibility (10).** 16-bit fixed point (Q1.14/Q5.10) matches float to 4 decimals; fp8×fp8 STFT MBSTOI 0.8170 ≈ fp32 0.8153 → **cheap fixed-point FFT hardware justified** (RTL realization: H18).

**P9 — Parallel MBSTOI evaluation infrastructure (11).** GPU inference / CPU metric computation split via a process pool: **~13× speedup**, byte-identical → made the large softmax/QAT sweeps tractable (methodology material).

---

## 3. Quantization / Numerics Features (Q-series) — the Zero-Multiplier Core

**Q1 — W4A8 base format.** INT4 symmetric (q∈[−7,7]) + FP8 E4M3 + FP16 accumulation. W4A8 is first a **memory/DMA choice** (evidenced by P1/P3), and second the multiplier-removal platform. [HANDOFF §1]

**Q2 — Block-float FP8 activations (static per-tensor pow2 exponent, zero runtime scaling) ★.** Activation = raw e4m3 code + offline calibration `k=⌊log2(448/amax)⌋` (53 boundaries); `2^-k` fuses into the post-acc descale as an exponent offset. Since k is a pure power of two, `LMul(code·2^k,w)==2^k·LMul(code,w)` holds exactly → **eager==fused==HW bit-exact (relL2=0 on all 48 outlier L-Mul calls) → the software MBSTOI is the silicon number**. [HANDOFF §0, Consultant §6]

**Q3 — APoT weight scales (`2^a±2^b`, exp∈[−16,−1]), three granularities.** PW per-group-32 / DW per-channel / MHA per-head; scale multiply → 2 shifts + 1 add; quality 0.8350/0.7892 (lossless). [Consultant §6]

**Q4 — Conv INT4 shift-add + delayed descale.** INT4 code × e4m3 = CSD shift-adds on the significand (|q|≤7 ⇒ ≤2 add/sub); descale applied once per output post-accumulation — algebraically equivalent to per-weight dequant, bit-exact under wide accumulation. [Consultant §6]

**Q5 — L-Mul (m8 attention / m4 outliers).** sign-XOR + exponent add + one (m+2)-bit mantissa add (m4 = 6-bit, m8 = 10-bit adder); **~−70% multiplier energy** (MAC count unchanged); post-QAT **0.8364/0.7925, above baseline**. Score-domain quantization systematically ruled out (double discretization). [Consultant §4-13]

**Q6 — Pow2 Top-K softmax (`shift_topk_pow2_clamp8_nomantissa`).** Top-4 select → center on max, 1.5×, clamp [−8,0], floor → `2^k` (Q0.8, no mantissa) → denominator = sum of 4 weights rounded to the geometrically nearest pow2 → scatter shift. **Removes exp LUT, divider, multiplier** — needs only a top-k selector + adder + barrel shifter + LZD; 0.8360 (Δ−0.0013); robust cross-domain on TIMIT (+0.0616). Ablation: floor −0.0089, ceil −0.0203, taylor2 0.7770. [SOFTMAX handoff, Consultant §4-09→11]

**Q7 — Pow2 PReLU.** `±2^k`, k∈[−8,0], signed (7 of 22 slopes negative) → sign mux + 1 shift. [HANDOFF §3.6]

**Q8 — Folded-APoT BN (standalone per-channel affine — NOT foldable into complex conv) ★.** fr feeds both re/im outputs → folding would double the weights → post-MAC `y=a_q·x+b` (a_q = APoT[−9,4]; ~34% of channels need positive exponents; b = fp16 incl. conv bias). **= the necessity argument for the post-MAC PPU.** [HANDOFF §3.5]

**Q9 — 12 decoder FP8 outlier layers + delayed 2-term APoT descale ([−16,−6]).** Per-tensor scale 448/amax; the delayed-APoT approximation error recovered by a **5-epoch finetune under the deployed arithmetic** — **0.8365/0.7912, beating the multiplier-using 14 composite (0.8354/0.7880)**. [Consultant §6]

**Q10 — W4A16 exception (MHA out_proj).** The only 16-bit-activation op: consumes the Score×V fp16 accumulator directly (e4m3 loses precision); reuses the same chained adder slices. [HANDOFF §0/§4]

**Q11 — Training under the deployed arithmetic (QAT / recovery finetune) ★.** The final model is finetuned under arithmetic bit-exact to the hardware (L-Mul, delayed APoT, pow2 block-float); the quality gates hold on the **true hardware path**. Deploy ckpt md5 `673de90a44dc13a4770cafc5a223d161`; epoch picked by post-hoc MBSTOI screening (ep3 > ep2/ep4). [Consultant §6]

**Q12 — Shared Q/K/V block-float k=−5.** Calibration amax 11.45; the QK descale `2^-(k_q+k_k+2)` applied pre-softmax (logit temperature); downstream cast_k values pinned (output_proj = −1, proven by 100% consumer e4m3 reproduction). [RESPONSE_ALGO_INVENTORY]

**Scope of the zero-multiplier claim**: inside the neural core (Conv/MHA/BN/PReLU/softmax) — **zero general multipliers, zero dividers, zero exp LUTs**. Out of scope (separate DSP): STFT/iSTFT, windowing, mask-apply. [HANDOFF §0/§6, Consultant §6]

---

## 4. Exploratory Work (X-series, dirs 15–17: **NOT in deploy**; ablation/boundary material) [Consultant §5]

**X1 — All-L-Mul push (15).** L-Mul m4 on every Conv/Linear/MHA multiply: best after QAT/finetune 0.8264–0.8281, still ~0.009 below gate → **defines the L-Mul applicability boundary**: L-Mul for attention + FP8 outliers, INT4 shift-add elsewhere, is the deployable combination.

**X2 — KD / heatmap recovery (16).** Per-layer error heatmap (hotspots = deepest decoder PW), output-level KD (best 0.8288, still ~0.007 short), bias correction ineffective; fp32 accumulator gives no lift → the gap is not accumulator/packet/top-k — supporting the deploy's mixed-multiplier design.

**X3 — L-Mul-aware STE (17).** Three backward variants (identity / autograd ≈0.675× gradient damping / manual grad_scale), forward bit-exact; smoke-verified, **not launched** (contingent backup) → future work: the recovery plateau may sit in the STE gradient path.

> ⚠️ Writing red line: **dirs 15–17 are NOT in the deploy**; deploy L-Mul covers only QK/Score×V (m8) and the 12 FP8 outliers (m4) — do not conflate. [Consultant §10-9]

---

## 5. Hardware-Architecture Features (H-series)

### 5.1 Compute Fabric

**H1 — Precision-reconfigurable PE lane (4 modes, one datapath).** Atom = 8-bit significand slice; adjacent slices run "2×8b independent ↔ 1×16b carry-chained"; `W4A8`/`LMUL_FP8`(m4) → **128 lanes**, `LMUL_ATTN`(m8)/`W4A16` → **64 lanes** (= one configurable add/shift slice covers all 4 modes [Consultant §7]). INT4 as CSD ≤2 terms; fully combinational; per-tensor `preshift` folded into product exponents (H6). [PEG.sv]

**H2 — PEG organization (4×32 = 128 real lanes = 64 complex lanes).** Each PEG = 16 pe_pairs + **32-way FP16 saturating tree** (±65504 clamp; worst pre-descale partial 100,352 > FP16 max). **PEG-per-stream complex mapping**: PEG0=fr·xr, PEG1=fi·xi, PEG2=fr·xi, PEG3=fi·xr — complex multiply = 4 PEG streams + OACC add/sub composition (no multiplier). [CE_core.sv]

**H3 — Dealer (delayed APoT descale).** Per-PEG 12-bit `2^a±2^b` decode = 2 exponent adjusts + 1 fp16 add; between tree and OACC (positionally required for per-group scales); all three granularities share the array; split-k supports complex branches with different exponents. [ppu_fma.sv]

**H4 — OACC (complex accumulation + 128-context RF).** `re+=P0−P1`, `im+=P2+P3`; under A+ a {pos, ear}-indexed RF + completion snapshot; W4A16 4-PEG real mode sums left-associatively for bit-exact order preservation. [oacc.sv]

**H5 — PPU (post-MAC; CONV/PASS/SCALE).** BN affine (APoT α + fp16 β) → pow2 PReLU → e4m3 cast (cast_k = −consumer k); PASS = fp16 passthrough; SCALE = QK ×2^-2·2^-k. ⚠️ Implemented as a **single complex PPU (one re+im pair)** — not the "~8-lane PPU pool" in [Consultant §7] (an early algorithm-side sizing suggestion). The paper should describe the RTL (throughput sufficient: 1 complex output per ⌈ic/32⌉ cycles). [ppu_fma.sv]

**H6 — FP16-tree overflow management (preshift/residual decomposition).** INT4/DW/out_proj safe (≤0.19×); FP8 outlier (4.2×) / QK (33×) / Score×V (5.7×) overflow → move the **pure-pow2 component** of the descale into the operand-unpack exponent (zero cost); non-pow2 residual stays in the dealer; flip-together invariant. After the algorithm-side re-bake of the 12 FP8 tensors: MBSTOI certified 0.8356/0.7897. [fp16-tree analysis]

### 5.2 Attention Hardware

**H7 — 4-path KV ring cache (32 KiB, exactly full).** The tied passes' real/imag histories differ entirely → K_re/K_im/V_re/V_im at 8 KiB each; ring pointer, read `slot=(ptr−local_idx) mod 32`; stores e4m3 block-float codes (k=−5). **32 KiB = 4×2×32×256×1 B exactly.** [ce_kv_cache.sv]

**H8 — Softmax_v3 hardware (multiplier/divider/exp-LUT-free).** `IDLE→SCAN(4 cy, 8 lanes/cy)→CALC→DONE`, ~7 cy/head; chained Top-4 insertion sort; fp16-delta (subtract-max → RNE 1.5× → floor clamp8); geometrically-nearest pow2 denominator (√2 threshold); packet saturation 1.0→255/256; warm-up degradation + causal validity mask — **bit-exact vs deploy**. [Softmax_v3.sv]

**H9 — Attention scheduling overlap.** Per-head softmax hidden under QK's next head (stall 210→2 cy); just-in-time V-gather; out_proj W4A16 across 4 PEGs (RUN 8,192→2,048). [HW_HIERARCHY §3.0]

### 5.3 Memory Subsystem

**H10 — Full-model-resident WBUF (256 KiB single-port; zero weight DMA) ★.** Entire model resident: manifest **63 weight layers = 66 packed tensors (54 INT4 + 12 FP8)** + 103 const blocks; weights 240,336 B + ~10 KB constants = **251,677 B (95.7% full)**. Achieved jointly by "−91.5% params + INT4 + tied MHA + grouped QKV" — **kills the original DMA bottleneck outright** (P6's 640 KB/frame MHA transport → zero). [SRAM_MEMORY_MAP_SPEC, Consultant §7]

**H11 — 12-bit constant packing + PPU const cache.** 3,700 APoT scales + 1,250×2 BN + 22 PReLU + 53 act-k, all 12-bit-packed into the WBUF constants region (~9.8 KiB) — no SCALE/BNBUF macros; 6a/6b decode formats; a **2 KiB ping-pong RF const cache** prefetches per sub-op → the PPU never touches WBUF during MAC. [SRAM_RTL_MEMORY_HANDOFF_EN]

**H12 — Frozen SRAM (300 KiB) + real TSMC macros (T6).** WBUF rf_sp_2048x64×16 / KVBUF rf_dp_1024x64×4 / UFBUF rf_dp_128x64×8 / SKIPBUF rf_dp_128x64×4; WBUF 3-row ring, UF bank interleave, byte-write masks; all regressions bit-exact. [RTL_HANDOFF_T6]

**H13 — Dual-role SKIPBUF.** One 4 KiB buffer holds the top-50% sparse skips (188-entry capture LUT) + the **STFT spectrum in Q2.14** parked until the decoder emits the mask. [skip dataflow T2]

### 5.4 Control / Orchestration

**H14 — Two-level control: a 36-entry sub-op ROM (auto-generated from the algorithm manifest) + a 33-step frame sequencer.** FSM `FETCH→PREFETCH→(DWLOAD)→RUN→DRAIN→NEXT`; attention [31,34] loops via pass_sel for the tied 2-pass under one i_start. [inst_decoder.sv, ce_frame_seq.sv]

**H15 — A+ weight-stationary binaural dataflow.** oc → chunk/tap → position×{L,R}: each weight read once per (oc, chunk) → conv WBUF reads **20.2k→3.9k/frame (÷5.2, theoretical floor)**; latency unchanged, goldens preserved bit-exactly; attention inherently ear-merged. [t5-orchestration]

**H16 — Depthwise 5-tap spatial parallelism + line buffer.** enc dw RUN 6,720→1,344, dec dw 13,440→2,688 (**5× each**); inputs read once; left-associative 20-dealer order preservation. [HW_HIERARCHY §3.0]

**H17 — Three-generation prefetch optimization (conv 19,368→4,977 cy, −74%).** v1 DRAIN-bubble removal (RUN = ideal 3,876); v2 burst emit (1→5 codes/cy); v3 streaming read-ahead (PREFETCH 888). [CE_PREFETCH_OPT_LOG]

### 5.5 Front/Back-End DSP and Chip Level

**H18 — On-chip STFT/iSTFT (shared FFT128).** Radix-2² SDF dual-mode; **Hamming** window (fixed-point justified by P8 → 16-bit); `stft_q214_cast` (single ×4√2 integer multiply, exhaustively exact over int16); multiplier-free `q214_to_e4m3` front-end; **spectrum bus Q2.14** (|spec| 1.46 clips Q1.15 — a real bug caught in verification). [STFT_TOP.sv]

**H19 — Mask-apply (mode-C complex multiply).** Q2.14 spec × fp16 mask → fp32 datapath → single RNE back to Q2.14 (fp32 rounding is scale-invariant → no 2^±14 scaling); mask consumed in fp16 (e4m3 costs 1.9% RMS). [mask_apply.sv]

**H20 — Chip I/O and boot streaming (no external CPU).** Weights in 16,384 beats (256 KiB burned once) → features 32 beats/frame → execute; output 512 B/frame byte-serial. [bcctn_top.sv]

**H21 — Multi-mode power measurement (6 modes, subtraction method).** FULL (bit-exact vs single-mode build) / STFT / ENC / ATTN / DEC / ISTFT; ENC/ATTN/DEC emit a 4-byte rotate-XOR signature; per-block power = difference of adjacent modes. [MULTIMODE_TAPEOUT_SPEC]

**H22 — Debug MISR.** 8-bit Galois (poly 0x1D) folds all output bytes; one byte at o_done = whole-pattern pass/fail. [CHIP.sv]

**H23 — Pad ring.** 148 pads (PDDW16*GZ): TOP 66 / RIGHT 72 / BOTTOM 5 / LEFT 5. [CHIP.sv]

### 5.6 Latency / Area / Power (measured)

**H24 — Frame-latency progression (binaural, 20 MHz).** 45,727 (first closure) → 47,279 (T6) → out_proj 4-PEG −6,144 → softmax overlap −208 → dw tap-parallel −12,096 → **28,831 cy = 1.442 ms**; cold boot **31,554 cy = 1.578 ms**. Groups: ENC 9,259 / ATTN 11,832 / DEC 7,675. The three optimizations total −39% cycles with **net −8,010 FF**. [HW_HIERARCHY]

**H25 — Flip-flop reduction.** FF pass −54.2k (ce_engine 70.7k→30.7k; design ~105k→~55k) + a further net −8k from the optimization pass; OACC/cbuf are RF-macro candidates. [t5-orchestration]

**H26 — Power operand isolation (zero cycle cost, bit-exact).** PE-array operands zeroed outside RUN (~25% of time); dw reduce frozen (~80%); per-mode path gating; mask_apply isolation; STFT datapath naturally register-isolated. [HW_HIERARCHY §3.0b]

---

## 6. Verification-Methodology Features (V-series)

**V1 — Bit-exact golden ladder.** Exhaustive unit tests (PE 15×256 / 254 e4m3 codes) → per-layer goldens (all enc/dec/attention vs deploy dumps) → chained (MHA 30..35, one i_start) → **full-frame e2e** → **boot e2e (pure chip ports)** → gate-level TB (512 B byte-exact) → 6-mode sweep. All bit-exact.

**V2 — Python bit-exact replicas as arbiter.** fp16 tree, L-Mul m4/m8, dealer, RNE e4m3 cast, DesignWare NaN semantics (measured) all replicated → RTL vs replica bit-compared; replicas cross-checked against deploy fp32 (all deltas in the known ±1 ulp class).

**V3 — Quality-level certification.** The full fp16 PPU chain is **MBSTOI-lossless** (VCTK Δ0.0000 / TIMIT −0.0001); after the FP8 preshift re-bake 0.8356/0.7897 → the "RTL numerics = deployed quality" loop closed.

**V4 — Golden sweeps catch real bugs (liveness ≠ numerics).** Multi-chunk dealer misalignment, nibble parity, three e4m3-cast bugs, a WEB-polarity silicon-killer, spec-ear timing, a 35-writeback hazard — **10+ real bugs found only by bit-exact goldens**; smoke tests missed all of them.

**V5 — Lint / synthesis sign-off.** SpyGlass 0 errors / 0 warnings; DC flow (compile_ultra -gate_clock, 20 MHz); gate-sim TB in place.

---

## 7. Quality Positioning / Competitor Features (E-series)

> **Repo disambiguation** (fixes the "this repo" clash flagged in the 2026-07-07 HW review):
> E-series raw data lives in the **algorithm repo** `/raid/R12K41024/20_pesq_finetune/results/`
> ({pesq_eval, voicebank_demand}); it is NOT in the **RTL/Journal repo**. A sync bundle for the
> Journal repo is provided (see §10-2 answer).

**E1 — VoiceBank+DEMAND standard-protocol eval (zero-shot).** 824 files, standard protocol, binaural render (az0/az70): **STOI 0.9685 (96.85) > 95.1 (CICC'26-PKU) > 91.44 (JSSC'24-SNU)** — highest among all chips reporting STOI; PESQ nb 2.89→3.01 / wb 2.00→2.27; SI-SDR 17.9→17.1 (healthy).

**E2 — PESQ↔spatial-cue trade-off ablation (dose-dependence proven).** Warm-start from deploy with increasing PESQ loss weight: w0 = 0.834 / ILD 3.33 / IPD 16.0 → w18 = 0.823/3.44/17.7 → w30 = 0.773/3.98/21.6. **Higher PESQ weight monotonically wrecks spatial cues** → "quantization preserves cues; PESQ finetuning erodes them" — turns a weakness into evidence of design intent.

**E3 — Honest PESQ framing.** wb/nb reported separately and labelled; **never** placing our nb next to CICC'26-PKU's wb 3.34; the original BCCTN paper reports no PESQ. High-PESQ variants (mix_ep19 nb 3.324) collapse SI-SDR to 11.9 and MBSTOI to 0.802 → **report deploy, not a gamed point** (all PESQ-leaning finetunes fall below the MBSTOI gate [Consultant §10-10]). Full paper-ready argumentation: **E6**.

**E4 — Quantization does not sacrifice spatial fidelity (same-test-set, same-metric A/B).** Official BCCTN FP32 ckpt (11 M params) vs our W4A8 deploy on the same 64 VCTK files, same metric: **IPD 18.0° vs 17.3° (tie), ILD 3.76 vs 4.59 dB** → publishable: "the W4A8 zero-multiplier core preserves interaural cues on par with the FP32 reference." Also proves **absolute ILD/IPD is not cross-test-set comparable** (the original paper's 0.4 dB/5° is specific to its unreleased test set) → report improvement-over-noisy and same-set A/B.

**E5 — Competitor positioning (venue-institution labels; MIT VLSID'25 dropped — not a top venue, no PDF).**

| Chip | Venue | Model / MAC style | Dataset | Reported quality | Power/energy |
|---|---|---|---|---|---|
| **JSSC'20-NTU** (Lee, Chi, Yang) | IEEE JSSC vol. 55 no. 8, Aug. 2020 | CNN mask, general MAC, 40nm | IEEE/NOISEX-92 | WAR +41% | 2.17 mW, ~4 µJ |
| **CICC'26-PKU** ⭐ (Zhi et al.) | CICC 2026 | Transformer 1.35M, **D-CIM**, 22nm | **VB+DEMAND** | **PESQ wb 3.34 / STOI 95.1** | 1.9 mW, 15.2 µJ |
| **A-SSCC'25-Tsinghua** (Wu et al.) | IEEE A-SSCC 2025, paper 20.1 | AV CNN (mono+lip), 22nm | TMSV(+visual) | PESQ 2.13 @ −10 dB | 3.64 µJ/frame |
| **JSSC'24-SNU** (Park et al.) | IEEE JSSC (ISSCC'23 ext.) | SepConv+GRU 56k, **multiplier-less**, 28nm | CHiME-2 | PESQ nb 2.73 / STOI 91.44 / SDR 14.0 | 0.74 mW |

**All four competitors are monaural with zero spatial cues** → MBSTOI/ILD/IPD uncontested; hardware differentiation: pure-digital zero-multiplier (more portable than CICC'26-PKU's D-CIM; vs JSSC'24-SNU's multiplier-less mono CNN we run the far harder binaural transformer, zero-multiplier).

**E6 — Why our PESQ does not beat CICC'26-PKU, and how to argue it (paper-ready narrative).**
Sources: `20_pesq_finetune/docs/STATUS_CHIP_PAPER_COMPARISON_POSITIONING_2026-06-28.md` §TL;DR/§4/§6; BCCTN = arXiv:2403.05393 (ICASSP'24).

*The structural reason (not a tuning failure):* our architecture is BCCTN lineage, whose design
objective is **binaural spatial-cue preservation** — the loss is `SNR + 10·STOI + ILD + 10·IPD`
(intelligibility + spatial cues; **no PESQ term**), and **the original BCCTN paper does not report
PESQ at all**. CICC'26-PKU's lineage (MP-SENet) is a PESQ specialist (FP32 wb 3.50). Chasing PESQ
is provably counterproductive for us: the E2 ablation shows PESQ-weighted finetuning erodes
MBSTOI/ILD/IPD **monotonically** (dose-dependent), and the highest-PESQ variant games the metric
(SI-SDR collapses 17.9→11.9). So the low/no-PESQ deploy point is a **design decision**, not a
shortfall.

*The argument chain (reusable from the BCCTN paper — it concedes noise-reduction supremacy and
wins on cue preservation):*
1. Binaural listening itself improves intelligibility **and** spatial awareness for the user.
2. Binaural signals carry the ILD/IPD needed for localization and binaural unmasking → the cues
   must be **preserved**, not just the noise removed.
3. The villain: monaural per-ear processing denoises *"at the cost of damaging vital binaural
   cues."*
4. Even a *good* monaural model is dangerous: *"altering the level and phase information would
   modify the spatial information … and therefore compromise the localization and spatial
   awareness of the listener."*
5. The fix: explicit ILD/IPD loss terms (our loss).
6. Data capstone (algorithm-level precedent): BiTasNet wins SegSNR yet **loses MBSTOI and
   ILD/IPD** → stronger denoising alone is not better hearing.

*Chip-paper adaptation (swap SegSNR→PESQ, BiTasNet→CICC'26-PKU, algorithm→chip):*
> "In noisy environments, binaural signals improve both intelligibility and spatial awareness,
> carrying the interaural cues (ILD/IPD) required for localization and binaural unmasking
> [Hawley'04; Beutelmann & Brand'06; Blauert'97]. **However, all prior SE accelerators are
> single-channel**: applied independently to each ear they denoise at the cost of damaging the
> interaural cues. Even an accelerator achieving high PESQ/STOI, by altering per-ear level and
> phase, modifies the spatial information and compromises the listener's localization and spatial
> awareness. We present **the first binaural SE accelerator that preserves ILD/IPD while
> denoising**, in a pure-digital zero-multiplier W4A8 core. Mirroring the algorithm-level finding
> that a stronger noise reducer can win SegSNR yet lose binaural intelligibility, the strongest
> monaural SE chip (CICC'26-PKU) leads PESQ but neither reports nor can preserve spatial cues;
> ours leads on binaural intelligibility (MBSTOI 0.836) and STOI (96.85 vs 95.1) while preserving
> ILD/IPD."

*Supporting citations (all already used by BCCTN):* Hawley, Litovsky & Culling 2004 (binaural
cocktail-party benefit); Beutelmann & Brand 2006 (binaural unmasking → intelligibility);
Blauert 1997, *Spatial Hearing* (ILD/ITD/IPD as the primary localization cues); Klasen et al. 2006
(cue-preserving binaural Wiener precedent); Andersen et al. 2018 (MBSTOI metric).

*Sentence to pre-empt the reviewer:* "This architecture follows BCCTN (ICASSP'24), whose design
objective is binaural spatial-cue preservation; the original work does not use PESQ as an
evaluation metric. We therefore report PESQ for completeness (wb and nb, labelled) and treat the
PESQ↔spatial-cue trade-off as a design-decision ablation (E2/fig12)."

---

## 8. Paper Claim List (consultant's 10 + hardware additions, merged)

1. **First binaural, cue-preserving SE accelerator** (all **four** compared chips — JSSC'20-NTU, CICC'26-PKU, A-SSCC'25-Tsinghua, JSSC'24-SNU — are monaural).
2. **Zero-multiplier neural core**: Conv/MHA/BN/PReLU/softmax all shift/add/mux; no multiplier, divider, or exp LUT.
3. **All weights on-chip**: 337,876 weights (INT4+FP8) + constants in one 256 KiB WBUF (95.7% full) → zero weight DMA at inference.
4. **Static calibration, zero runtime activation scaling**: pow2 block-float + fused exponent offset; eager==fused bit-exact (relL2=0) → **the reported MBSTOI is the silicon number**.
5. **STOI 96.85 > 95.1 (CICC'26-PKU) > 91.44 (JSSC'24-SNU)** (same-dataset direct comparison for the PKU chip); MBSTOI 0.836 + ILD/IPD as uncontested metrics.
6. **Compression**: params −91.5% (3.97 M→338 K), MACs −20~27× (29 M→1.10 M/frame; footnote the bases), attention probability transport −89.8% (512→52 bits/row), MHA weights 512 KB→24 KB (QKV).
7. **Tied complex MHA**: −50% MHA params/softmax serializations with a quality *gain* (0.8411 > 0.8397) — the architectural novelty.
8. **Quantization does not sacrifice spatial fidelity**: W4A8 vs FP32, same test set, IPD 18.0° vs 17.3°.
9. **PESQ↔spatial-cue trade-off ablation**: dose-dependent evidence for the "we chose to preserve spatial cues" narrative.
10. **Trained under the deployed arithmetic**: QAT/recovery ran directly under bit-exact hardware arithmetic; the deploy point beats the multiplier-based production stack (0.8365/0.7912 vs 0.8354/0.7880; block-float variant 0.8357/0.7898).

**Hardware additions (H/V-series; include as space allows):**
11. Precision-reconfigurable PE lane (paired 2×8b slices; one array, 4 modes, 128/64 lanes).
12. Complex-conv BN non-foldability → post-MAC PPU necessity argument.
13. 4-path KV ring (per-pass histories of the tied MHA; exactly 32 KiB).
14. Score-domain Top-4 pow2 softmax hardware (~7 cy/head, bit-exact vs deploy).
15. A+ weight-stationary binaural dataflow (WBUF reads ÷5.2, bit-exact order) + dw 5-tap parallelism (5×).
16. Preshift/residual decomposition for FP16-tree overflow (zero cost).
17. Measurement-oriented tapeout (6-mode subtraction + single-byte MISR).
18. Bit-exact verification methodology (Python-replica arbitration, down to gate level, 10+ real-bug case studies).
19. Measured frame latency 1.442 ms @ 20 MHz (29% of the 5 ms budget) — real-time met (area/power to follow after synthesis).

---

## 9. Writing Caveats (internal inconsistencies + consultant caveats, merged)

| # | Trap | Disposition |
|---|---|---|
| 1 | **Do not mix parameter bases**: dirs 01/02 used the old 512-dim (3.97 M), 03/04/06 used 256-dim, dir 07's unified report computes 2.54 M; per-directory `model_size.md` files are stale boilerplate | Always quote the `RESULTS_ARCH_PROGRESSION.md` series (3,973,288→830,300→567,132→485,332→337,876 RTL) [Consultant §10-1] |
| 2 | **WBUF narrative**: early 64/32 KB streaming story (HANDOFF §4/§9) is stale | Frozen = 256 KiB fully resident (2026-06-03 spec) [Consultant §10-2] |
| 3 | **MAC bases**: 29 M (hop-100/fft-512, own basis) vs 1.10 M (hop-80/fft-128) not directly comparable | Footnote every plot [Consultant §10-3] |
| 4 | **Group size**: dir 03 recommended gsz=256; deploy uses gsz=32; 03's two result files disagree | ✅ **Resolved (audit 2026-07-07)**: report's trend correct; `summary.md`/CSV BN-PReLU rows are pre-bugfix artifacts. Conv+MHA-restricted (n=56): per-tensor 7.63e-4 → gsz16 2.76e-4 (best) → gsz32 2.93e-4 (deployed) → gsz256 3.36e-4. Never cite "gsz256 = 0.28×" (see P4) |
| 5 | **Window Hann/Hamming**: FUSIONS.md says Hann; training uses Hamming | ✅ **Fully resolved**: RTL is Hamming, bit-exact vs deploy (T4, spec 511/512); FUSIONS.md now carries a dated erratum (2026-07-07) correcting Hann→Hamming |
| 6 | **Absolute ILD/IPD not cross-test-set comparable** (the original paper's 0.4 dB/5° is its private test set) | Report improvement-over-noisy + same-set FP32-vs-W4A8 A/B [Consultant §10-6] |
| 7 | **Dirs 15–17 all-L-Mul NOT in deploy** | Ablation/boundary only; deploy L-Mul = QK/Score×V (m8) + 12 outliers (m4) [Consultant §10-9] |
| 8 | **All PESQ-leaning finetunes fall below the gate** | Final reported model = deploy (the low/no-PESQ-pressure point) [Consultant §10-10] |
| 9 | PE count: CLAUDE.md stale 256/8 | **128 lanes (4×32)** per Param_def.sv |
| 10 | Sub-op count: inst_decoder comment "68" | **N_SUBOP = 36** |
| 11 | ~~54 vs 66 tensors~~ | ✅ **Resolved**: 54 INT4 + 12 FP8 = 66 packed tensors (63 weight layers) [Consultant §7] |
| 12 | KVBUF/WBUF: CLAUDE.md old tables 64 KB / 32 KB | 32 KiB dual-port / 256 KiB single-port resident |
| 13 | "HPC+" process qualifier not found in docs | Confirm before citing |
| 14 | `MEM_spec.md` is entirely legacy | Do not cite |
| 15 | The DMA-bound 8:1 narrative (old 32 KB WBUF premise) | Motivation contrast only, not current state |
| 16 | **Consultant §7 RTL status is stale** (lists "full-network golden compare, controller wiring, macro swap, power numbers" as remaining) | Reality: full-frame + boot e2e bit-exact, T6 real macros done, multi-mode tapeout build, latency measured (28,831 / 31,554 cy); **the only true gap = synthesis area/power** |
| 17 | **"~8-lane PPU pool" (Consultant §7)** | RTL implements a single complex PPU (throughput sufficient); the paper should describe the RTL |
| 18 | **"Attention = 77.6% of MACs" is the embed-512 variant, not deploy** | Deploy basis (E=256/16, re-profiled): 0.823 G total, 54.0/30.8/15.2 — "attention largest" holds on both bases; label the basis (see P1) |
| 19 | ~~All of dir-06 rests on n=2~~ ✅ **Re-run at n=20 (2026-07-07)** — QK^T 1049>448, 12/16 sharp heads confirmed; its IPD ~1.4 rad remains a saturated raw-STFT metric | Cite the n=20 numbers; top-4 *quality* still from 07/14; never mix the ~1.4 rad IPD with the 15–21° eval numbers (see P7) |
| 20 | ~~Dir-02 raw eval outputs gone~~ ✅ **Re-run n=500/seed 42 (2026-07-07), artifacts retained** (`results_rerun_2026-07-07/`); 29+1 tiers & 61.5%/50%/62.5% savings reproduced exactly; INT4 −0.38 dB / INT3 −4.36 dB / mixed −2.31 dB | Cite the re-run numbers; the report's absolute values (baseline 10.49) are a different eval subset — historical only (see P3) |
| 21 | **Dir-05 low-rank "+18–27%" holds only @95% energy** (negative @99%); "num_heads=32" is a doc error (actual 16); MHA 640 KB counts outer complex projections real-only (~768 KB full) | Quote with the 95% qualifier; dominance claim (MHA ≫ convs) unaffected (see P6) |
| 22 | **Dir-04 "6 dense = Q/K/V/O projections" is mislabeled** (5 conv + 1 MHA); ~~dir-03's "42/42" artifact-unsupported~~ ✅ 42/42 re-supported by the 2026-07-07 regen (`results_regen_2026-07-07/per_channel_variance.csv`) | Sparsity conclusion verified; fix the 04 wording; cite 03 from the regen dir (see P4/P5) |
| 23 | **Three MAC bases now exist**: 29 M/frame (original, hop-100), 1.10 M m-eq/frame (deploy normalized), and the P1 re-profile 0.823 G/s ≈ 4.1 M/frame (complex→4-real convention, pre-09 architecture) | None inter-divisible. Reviewers WILL divide them — every quoted MAC number must name its basis; the re-profile is for **percent shares only** (see P1) |
| 24 | **Competitor P-numbering abolished** (SPEC and STATUS conflicted; headline had cited an ambiguous "P4") | ✅ Resolved 2026-07-07: venue-institution labels everywhere (JSSC'20-NTU / CICC'26-PKU / A-SSCC'25-Tsinghua / JSSC'24-SNU); MIT VLSID'25 dropped (not a top venue, no PDF); comparison set = 4 chips |

---

## 10. Open Items

**Back to the algorithm consultant — ✅ ALL FIVE ANSWERED 2026-07-07** (full detail:
`19_algo_final/docs/RESPONSE_OPEN_ITEMS_PAPER_NUMBERS_2026-07-07.md`):
1. ✅ **Decoder-PW MAC**: same basis (conv-only module-call MAC), different scope — −62.16% = grouping alone (P1 g42); **−74.74% = grouping + top-50% skip (P2 g42+s050_topk) = the deploy config → paper cites −74.7% dec-PW local / −46.5% all-conv** (see A6).
2. ✅ **MBSTOI pair**: agreed — body cites deploy **0.8357/0.7898**; footnote HW-chain-certified **0.8356/0.7897** (Δ ≤ ±0.0001, post preshift re-bake; strengthens claim 4).
3. ✅ **Per-SNR table LOCKED**: `18_..._hw_handoff_20260530/results/apot_descale_eval/ft_pow2_ep3_full/per_snr_locked_paper_table.csv` — VCTK+TIMIT × 8 SNR × {noisy, enhanced, Δ} × {MBSTOI, PESQ, SegSNR, ILD, IPD}; per-SNR means reproduce 0.8357/0.7898 (canonical citation = the `*_averages.txt`). Still missing if wanted: per-SNR STOI/ESTOI/SI-SDR on VCTK/TIMIT (needs a re-run, ~1 GPU).
4. ✅ **E-series raw data IS in the *algorithm* repo** (`/raid/R12K41024/20_pesq_finetune/results/pesq_eval/<run>/`, 45+ runs incl. per-SNR detail, + `results/voicebank_demand/<run>/`; eval entry points in `20_pesq_finetune/code/`) — it was never in the RTL/Journal repo (19_algo_final is the frozen handoff and never carried it; the earlier "not in this repo" wording meant the Journal repo — now disambiguated in the §7 header). `E_SERIES_MANIFEST.md` (run → ckpt → metrics traceability) now provided in the sync bundle.
4b. ✅ **Audit-artifact sync (HW review 🔴2)**: all four referenced artifacts staged for the Journal repo in **`/raid/R12K41024/journal_sync_2026-07-07/`** (target: `Journal/algo_feature/`) — the 01–06 audit report, the RESPONSE doc, `paper_figures/` (SVG/EMF/PNG + CAPTIONS.md + `data/util_rows_e256.csv`), and `per_snr_locked_paper_table.csv`.
5. ✅ **Citations extracted from the PDFs** (see RESPONSE §5): Lee/Chi/Yang, JSSC vol. 55 no. 8, Aug. 2020 (**JSSC'20-NTU**); Zhi et al. (PKU), CICC 2026 (**CICC'26-PKU**); Wu et al. (Tsinghua/Fudan), A-SSCC 2025 paper 20.1 (**A-SSCC'25-Tsinghua**); Park et al. (SNU), JSSC 28nm (**JSSC'24-SNU**). ✅ **Both follow-up decisions CLOSED 2026-07-07 (user)**: (a) P-numbering **abolished** — venue-institution labels used everywhere (kills the SPEC-vs-STATUS conflict and the headline ambiguity flagged in the HW review); (b) **MIT VLSID'25 dropped** from the comparison (not a top venue; no PDF). See §0 convention block and E5.

**Hardware side:**
6. Synthesis/measurement area, power, energy-efficiency numbers (the one comparison-table gap) → backfill §0/§8-19.
7. Confirm the exact "TSMC 28nm HPC+" process name.
8. ✅ FUSIONS.md Hann→Hamming: dated erratum added at the top of `09_software_to_hardware/FUSIONS.md` (2026-07-07; original text preserved per the append-only convention).

---

*v2 consolidated and verified by Claude (2026-07-07). New source: `Journal/PAPER_FEATURE_SUMMARY_2026-07-07.md` (algorithm consultant). Other sources as in v1 (HANDOFF/ARCH18/PROG/SOFTMAX/SRAM specs/T6/MULTIMODE/HW_HIERARCHY/Param_def, and project working memory).*

*v2.1 (2026-07-07, algorithm side): P1–P7, A6, §9 (rows 4, 18–22), and §10 items 1–5 updated
with the results of the dirs-01–06 numbers audit (`RESULTS_ANALYSIS_VERIFICATION_01_06_2026-07-07.md`)
and the consultant-open-items response (`19_algo_final/docs/RESPONSE_OPEN_ITEMS_PAPER_NUMBERS_2026-07-07.md`).
Paper figures regenerated with the corrected data: `paper_figures/` (SVG + EMF for Visio + PNG;
per-figure sources & caption caveats in `paper_figures/CAPTIONS.md`).*

*v2.2 (2026-07-07, algorithm side — response to the HW review of v2.1): 🔴1 resolved by ABOLISHING
competitor P-numbering (venue-institution labels everywhere: JSSC'20-NTU / CICC'26-PKU /
A-SSCC'25-Tsinghua / JSSC'24-SNU; MIT VLSID'25 dropped per user — not a top venue) — §0 convention
block, E1/E3/E5, claims 1/5, §10-5. 🔴2 resolved via the sync bundle
`/raid/R12K41024/journal_sync_2026-07-07/` (target `Journal/algo_feature/`; incl.
`E_SERIES_MANIFEST.md`, 107 runs). 🔴3 resolved: third-MAC-basis warning added (P1, §9-23).
🟠 P3 self-contradiction reworded; "this repo" disambiguated (§7 header, §10-4). E6 added:
paper-ready PESQ narrative (BCCTN-lineage argument chain + adapted paragraph + citations).
Reproducibility re-runs on one GPU: 06 n=20 DONE (QK^T abs_max 1049 > 448 — conclusion
strengthened); 03 regen + 02 n=500 in flight, artifacts land in `03_dynamic_range/
results_regen_2026-07-07/` and `02_weights_bitwidth/results_rerun_2026-07-07/`.*
