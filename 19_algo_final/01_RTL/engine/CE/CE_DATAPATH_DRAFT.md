# CE per-layer datapath — DRAFT for review (2026-06-07)

Status: design discussion before writing CE RTL. Confidence tags:
`[OK]` from handoff/code, `[?]` needs confirm, `[TBD]` scheduling detail to design.
Sources: HANDOFF_HW_ZERO_MULTIPLIER_CORE, SRAM_RTL_MEMORY_HANDOFF_EN,
RTL_HANDOFF_PE_LANE_DATAPATH_2026-06-06, ARCHITECTURE_SUMMARY_STRICT18.

## 0. Buffers
| SRAM | role | port | feeds |
|---|---|---|---|
| WBUF 256KiB | all weight codes (INT4/FP8) + packed consts (APoT scale / BN / PReLU / act-k) | single-port | weights → PE; const region → PPU const cache (prefetch) |
| UFBUF 8KiB | current-frame activations (layer in/out, ping-pong) | dual-port | x operand → PE; FMA result → back |
| KVBUF 32KiB | K/V cache, raw e4m3, ring over 32 frames | dual-port | K/V → MHA QK / Score×V |
| SKIPBUF 4KiB | decoder skip subset (static top-50% ch) | dual-port | skip branch → decoder PW |

PPU const cache = 2KiB reg file (not SRAM); per-sub-op prefetch of APoT/BN/PReLU/k.

## 1. Generic MAC datapath (every conv/linear/proj layer)
```
WBUF[weight INT4/FP8] ─┐
                        ├─> PEG ×4 groups (pe_pair) ─> 4 group partial sums (pre-descale, FP16, saturating tree)
UFBUF / KVBUF[act] ─────┘
   ─> per-group DEALER : APoT 2^a±2^b  (fused weight descale, INCLUDES activation 2^-k)   [雷②: dealer BETWEEN tree & OACC]
   ─> OACC            : accumulate across input-groups / cycles  (complex: separate re/im accumulators)
   ─> post-MAC FMA    : BN affine α·x+β (α = APoT signed [-9,4], β = fp16 bias)  →  PReLU pow2 (±2^k, k∈[-8,0])
   ─> e4m3 cast       : block-float, uses NEXT layer's k ; RNE to match fp8_qat  [雷④]
   ─> UFBUF[out]
```
- post-MAC path is **conv-only**; attention has different post-paths (§4). [雷③: mode-select, don't hardwire BN+PReLU+cast]
- everything in dealer/FMA is shift-add (zero-multiplier).

## 1.1 Unit counts & throughput (the binding rule)

**Throughput invariant: the PE emits 4 group-partials / cycle.** `pe_pair + 32-way
tree = 1 PEG`, ×4 PEG. W4A8 / m4 (8-bit) = 128 lane = 4 PEG × 32; m8 / W4A16 (16-bit)
= 64 paired = 4 PEG × 16. **Both modes → 4 group-partials/cycle.** Every CE stage must
drain ≥ that rate at its granularity, or the PE stalls. That is the *only* sizing判準.

| stage | count | why this count | cost |
|---|---|---|---|
| **Dealer** (per-group descale) | **4** — one bolted to each PEG tree output, pre-OACC | 4 partials arrive *simultaneously* every cycle; a shared dealer serializes 4:1 → ¼ throughput. Also forced pre-OACC + per-group by 雷② | 2 shift + 1 add/sub each, trivial |
| **OACC** | **1 unit = 2 lanes (re/im) + ~4–8 acc regs** | complex = 4 real-product streams → 2 accumulators (re,im) must advance *same cycle*; 1 acc halves complex throughput. cross-group 4→2 is fixed −/+. acc regfile lets you interleave output channels / DW taps without stall | 6 small adders + regfile |
| **post-MAC FMA / PPU** | **4–8 shared lanes** (NOT dedicated BN/PReLU/cast lanes) | sizing by ratio: if descale stays in the 4 dealers, post pool = BN+PReLU+cast ≈32K/frame → `128×32K/1.40M ≈ 3` → take **4**. If descale folded into one post pool (§9.5) → **8**. | 2 shift + 1–2 add + sign-mux each |

Total post-PE shift-add ALUs ≈ **4 + 2 + 4 ≈ 10–12** vs 128 PE lanes (~10% area) and it
sustains full PE throughput. **Do NOT build dedicated BN + dedicated PReLU + dedicated
cast lanes**: MHA is 78% of compute and has *no* BN/PReLU — dedicated lanes idle through
all of attention (§9.5). One shared shift-add pool, assigned per-layer to descale OR BN
OR PReLU OR cast.

**Complex ×4 real-product spreads the FMA load thin.** Deepest layer `enc5.pw`
(Cout=128, Cin=128) = 128 ic × 4 real = 512 real-products / complex output → **4 cycle per
complex output** on 128 lanes. So FMA drain = 2 real (re+im) / 4 cycle = 0.5/cycle — even
1 fully-pipelined FMA lane keeps up. This is why deploy duty is only **0.87 (×4) / 0.69
(×3)** (`mac_profiler.py`): the post-path is never the bottleneck.

**What actually drops throughput (not the CE unit count):** (a) small-layer PE underfill
(enc0.pw Cin=1 → 1/128 lanes used; depthwise reduction=5; deconv) — a *front-end* duty
loss, fix by FIFO-drain / cap PE width to `PPU_lanes × reduction`, do NOT over-provision CE
(§9.5); (b) OACC serialization if re/im share one acc → the 2-bank fix above; (c) 雷①
FP16 tree overflow — a *correctness* cliff (worst-case 32-group sum 100,352 > 65,504 → Inf,
MBSTOI-invisible), do the >FP16 / saturating tree first.

## 2. Complex handling (encoder/decoder conv, MHA proj)
weight = w_r + i·w_i, act = x_r + i·x_i:
- `out_r = Σ(x_r·w_r − x_i·w_i)` , `out_i = Σ(x_r·w_i + x_i·w_r)`  → 4 real product streams → 2 OACC (re, im).
- BN is a **standalone per-channel affine after** the complex MAC (NOT folded into weights). conv bias folded into BN bias: re uses `b_r−b_i`, im uses `b_r+b_i`. [OK]

**[DECIDED] 4-stream → PEG mapping = PEG-per-stream** (resolves open Q#1):

```
PEG0 = fr·xr   PEG1 = fi·xi   PEG2 = fr·xi   PEG3 = fi·xr     (a common 32-ic chunk)
   ↓ dealer (per-PEG static k, see below)
OACC:  re += (P0 − P1)     im += (P2 + P3)     (accumulate across ic-chunks)
```
- 4→2 combine is fixed wiring (one sub, one add); re/im accumulators advance every cycle.
- throughput = 32 ic / cycle → 1 complex output per `⌈ic/32⌉` cycles (ic≤128 ⇒ ≤4 cycles).
- 雷⑤ `kreal/kimag` becomes **per-PEG static binding**, not runtime select (see §2.1).
- alt mapping (PEG-per-ic-group, cycle through 4 streams) has identical throughput but
  needs dynamic k-select + cross-4-cycle combine — more control. Prefer PEG-per-stream.

## 2.1 Complex activation exponent `k` — re/im (resolves the new open Q)

Activation `k = floor(log2 448/amax)` is calibrated **per-slice** (re-slice and im-slice
get their own amax). "Shared" only means they floored to the same value. From the actual
export (`activation_calibration.json`, `fused_descale_summary.json`):

| layer class | re/im `k` | note |
|---|---|---|
| Encoder dw + pw (all 6 stages) | **shared** (single `activation_pow2_exp`) | `per_branch=False` |
| Decoder depthwise-T | **shared** | `per_branch=False` |
| Decoder split-pointwise (all 6) | **separate** `kreal/kimag`; 20/24 equal, **stage 3 differs** (k_re=5, k_im=6) | `per_branch=True` |
| MHA in/out proj | **separate** (kept un-fused by policy) | `not_fused_note` |

**HW rule: always carry two static k slots (`k_real`, `k_imag`), wired per-PEG by slice**
— PEG0/PEG3 (driven by `x_real`) → `k_real`; PEG1/PEG2 (driven by `x_imag`) → `k_imag`.
When a layer is "shared", the export writes `k_real == k_imag` into both slots.
**Do NOT fork the datapath on `branches_collapse`** — it is purely a WBUF storage
optimization (store one fused table + broadcast), not a control mode. This keeps 雷⑤ as
zero-runtime-cost static wiring. (`k` is already fused into the dealer's APoT a/b exponent
via the `_fused_act_kreal/kimag` files, so the dealer just loads the right pre-fused exponent
per PEG.)

## 3. Encoder / Decoder conv layers

| Layer | PE mode | lanes | act src | weight | dealer scale gran. | post | out |
|---|---|---|---|---|---|---|---|
| Encoder DW conv (×6, k=5 causal) | W4A8 | 128 | UFBUF | WBUF INT4 | **per-out-channel** APoT [-16,-1] | BN+PReLU | UFBUF |
| Encoder PW conv (×6, 1×1) | W4A8 | 128 | UFBUF | WBUF INT4 | **per-32-input-ch** APoT [-16,-1] | BN+PReLU | UFBUF |
| Decoder DW-transpose (×6) | W4A8 | 128 | UFBUF | WBUF INT4 | per-out-channel APoT | BN+PReLU | UFBUF |
| Decoder PW INT4 (grouped, split x/skip) | W4A8 | 128 | x:UFBUF, skip:SKIPBUF | WBUF INT4 | per-32-input-ch APoT | BN+PReLU | UFBUF |
| Decoder PW **outlier** (stage 0/3/4, 12 tensors) | **FP8 L-Mul m4** | 128 | x:UFBUF, skip:SKIPBUF | WBUF FP8 e4m3 | **per-tensor** FP8 delayed-APoT descale | BN+PReLU | UFBUF |

Notes:
- Decoder PW is grouped: `groups_x=4`, `groups_skip=2`; split into `pointwise_{r,i}_{x,skip}`. [OK]
- `[?]` Decoder split-pointwise **kreal/kimag differ** per stage → dealer must pick `*_act_kreal`/`kimag` by which slice (real/imag) is feeding. [雷⑤]
- `[TBD]` DW conv 5 taps: accumulate 5 cycles in OACC before BN (per channel); BN/PReLU at last tap.

## 4. MHA (tied complex, dim=256, 16 heads, head_dim=16)
4 real attention paths (Phase A: MHA_re weights → path1 re×re + path3 re×im; Phase B: MHA_im → path2 im×im + path4 im×re). KVBUF holds the 2-paths-per-phase K/V.

| Sub-op | PE mode | lanes | operands | scale / extra | out |
|---|---|---|---|---|
| input_proj fc_r/fc_i | W4A8 | 128 | act UFBUF × INT4 | per-head APoT | UFBUF |
| internal in_proj (Q/K/V) | W4A8 | 128 | act × INT4 compact[12,64,64], qkv_groups=4 | per-head APoT | Q→scratch, **K/V→KVBUF (e4m3)** |
| QK^T | **L-Mul m8** | 64 | Q(fp16) × K(KVBUF e4m3→fp16) | **post: ×2^-2 (1/√d) + 2^-k BEFORE softmax** [雷③] | scores(fp16) → softmax |
| softmax | — (Softmax_v3 SFU, NOT PEG) | — | scores | shift Top-K pow2, output **top-4 q0.8 packet** | packet |
| Score×V | **L-Mul m8** (packet-gather) | 64 | top-4 packet(q0.8) × V(KVBUF e4m3) | `/256` = q0.8 exp offset (not APoT) | ctx(fp16) |
| out_proj | **W4A16** | 64 | ctx(fp16) × INT4 | per-head APoT, **NO activation k** | → combine complex |
| output_proj fc_r/fc_i | W4A8 | 128 | act UFBUF × INT4 | per-head APoT | UFBUF |

`[?]` Score×V packet path: top-4 gather of V rows + L-Mul m8 — likely a **dedicated small datapath**, not the 128/64 PEG (only 4 nonzero per row). Confirm whether it reuses PEG lanes or a separate unit.
`[?]` QK: K read from KVBUF is e4m3 (1B); needs fp8→fp16 at PEG input for L-Mul m8. Q is fp16 scratch.

## 5. Open questions for review
1. `[DECIDED §2]` **Complex 4-stream → PEG mapping** = PEG-per-stream; OACC = 2 banks (re += P0−P1, im += P2+P3).
2. `[?]` **Score×V / softmax**: dedicated SFU+gather vs PEG reuse? (attention is <8% of MACs.)
3. `[DECIDED §2.1]` **Dealer kreal/kimag** = two static k slots, per-PEG wiring; don't fork on `branches_collapse` (雷⑤).
4. `[DECIDED]` **DW conv 5-tap** = accumulate in OACC across cycles (no dedicated pre-accumulator). DW base ≤640/stage, channel-few, DMA-bound → not throughput-critical, not worth extra HW.
5. `[?]` **e4m3 cast rounding** mode vs `fp8_qat` (RNE?) — must be bit-exact (雷④).
6. `[TBD]` **UFBUF ping-pong / addressing** per layer; KVBUF ring write (in_proj) vs read (QK/ScoreV) phasing.
7. WBUF const **prefetch scheduling** into PPU cache per sub-op (overlap with PE MAC).

**Unit-count summary (§1.1):** Dealer ×4 (per PEG, pre-OACC) · OACC ×1 (2 re/im banks + acc regfile) · post-MAC FMA/PPU ×4–8 shared (no dedicated BN/PReLU/cast lanes). Throughput-bound by PE fill on small layers, not by these.

## 6. What's already built (PE level)
PEG (pe_pair fused 2×8b slices) + 32-way FP16 saturating tree → o_accum (pre-descale). Verified vs pe_lane×32 (16000 vec, 0 err). The dealer / OACC / post-MAC FMA / cast / buffer control in §1,§4 are the **next RTL** (CE level).
