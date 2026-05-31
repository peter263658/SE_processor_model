# Future Work: 5-20MHz Low-Power Architecture Co-Optimization

Date: 2026-05-19
Base stack: shipped Option C from `RTL_HANDOFF_FULL_STACK_2026-05-19.md`
(`P2 g42 + s050_topk` + tied MHA + W4A8 +
`shift_topk_pow2_clamp8_nomantissa` + FP16 attention accumulator +
L-Mul M4 on QK and Score x V)
Related results:
- `RESULTS_P2_SKIP_2026-05-18.md`
- `RESULTS_P3_ATTENTION_FP8_FINAL_2026-05-18.md`
- `RESULTS_P4_LMUL_FEASIBILITY_FINAL_2026-05-18.md`
- `RTL_HANDOFF_FULL_STACK_2026-05-19.md`

---

## 0. Goal

Target an always-on / low-power implementation at **5-20MHz**, rather than
max-throughput operation.

Current one-frame steady-state estimate for the P2/Option-C style compute
shape:

| Assumption | Value |
|---|---:|
| sample rate | 16kHz |
| hop | 80 samples = 5ms |
| frame budget @ 5MHz | 25k cycles |
| frame budget @ 10MHz | 50k cycles |
| frame budget @ 20MHz | 100k cycles |
| current 256 mixed PE + 64 FP16 estimate | about 5.8-6.0k cycles/frame |
| current 64 active unified PE raw MAC estimate | about 21.8k cycles/frame |

Implication: 5MHz is feasible with the current 256-lane configuration and is
**possible but tight** with 64 active unified lanes. A 64-lane design requires
high PE utilization, overlapped / separate FFT and elementwise units, and no
large SRAM stalls. Otherwise 96 or 128 active lanes are the safer 5MHz
operating points.

The architecture question is not "can the full 256-lane design meet real
time?", but rather:

1. How many PE lanes should stay active?
2. Which multipliers can be removed?
3. How much weight/SRAM traffic can be avoided?
4. Which algorithm changes reduce the current attention-projection bottleneck?

---

## 1. Important clarification: q0.8 probability path

### 1.1 What has been tested

P3 tested `b_prob_q08` as an **isolated algorithm ablation** on top of P2.
It clears the hard VCTK + TIMIT gates, but misses the newer 0.835 VCTK soft
target used by the full-stack handoff:

| Variant | RTL effect | VCTK delta vs baseline | TIMIT delta vs baseline |
|---|---|---:|---:|
| `b_prob_q08` | 8-bit probability storage | -0.0014 | -0.0008 |

Meaning:
- Softmax / shift-topk produces attention probabilities.
- Those probabilities are quantized to the q0.8 grid, i.e. multiples of
  `1/256`.
- The following `Score x V` sees quantized probabilities in the simulator.

### 1.2 Ship-status boundary

`RTL_HANDOFF_FULL_STACK_2026-05-19.md` measured:

```text
Option C + q0.8 probability + FP8 W4 scale
  VCTK 0.8326 / TIMIT 0.7889
```

This clears hard floors but fails the 0.835 VCTK target and is **not shipped**.
Therefore q0.8 probability storage is a future recovery experiment, not part
of the current RTL endpoint.

### 1.3 What has not been proven yet

The current P3 test is **not** a bit-accurate RTL proof of an integer
`q0.8 x V` datapath.

In software, q0.8 fake-quantization produces a floating tensor whose values
lie on the q0.8 grid. Depending on whether `fp16_accumulator` is enabled, the
tensor is still carried by a floating dtype. Therefore:

- Tested: quality impact of q0.8 probability values.
- Not fully tested: exact fixed-point q0.8 storage, dequant, shift-add, or
  integer/fixed multiply behavior in RTL.

### 1.4 Future experiment required

Add a bit-accurate q0.8 probability packet simulation:

```text
top-4 index: 5 bits each, range 0..31
probability: q0.8, 8 bits each
V: FP16 or FP8 depending datapath
Score x V:
  option A: dequant q0.8 to FP16 then multiply
  option B: fixed q0.8 x FP16 shift-add / small fixed multiplier
  option C: q0.8 x FP16 L-Mul-style approximate product
```

Gate:
- 20-pair sanity first: MBSTOI >= 0.8031.
- If pass, full VCTK + TIMIT.

---

## 2. Current bottleneck after P2

One steady-state frame:

| Block | MAC/frame | Current engine |
|---|---:|---|
| Encoder conv | 202,880 | mixed PE |
| Decoder conv after P2 | 122,624 | mixed PE |
| Attention projection / Linear | 1,048,576 | mixed PE |
| QK + Score x V | 18,432 | FP16 or L-Mul candidate |

The dominant cost is now **attention projection**, not QK/softmax:

| Attention projection sub-block | MAC/frame |
|---|---:|
| complex `input_proj` 256 -> 256 | 262,144 |
| MHA QKV projection, tied 2 real passes | 393,216 |
| MHA `out_proj`, tied 2 real passes | 131,072 |
| complex `output_proj` 256 -> 256 | 262,144 |
| total | 1,048,576 |

Future work should prioritize MHA projection grouping / sparsity before
spending too much effort on QK latency.

---

## 3. Phase A: Complex BN simplification + PReLU pow2 slope

### Motivation

Low risk, small compute gain, useful hardware cleanup, but with an important
complex-conv caveat:

- The model uses `NaiveComplexBatchNorm2d`, which applies independent real and
  imaginary `BatchNorm2d` modules.
- Exact folding into the existing complex conv form is **not generally valid**
  unless the real and imaginary BN affine scales are equal.
- Reason: the complex conv shares `conv_r` and `conv_i` weights across the
  real and imaginary equations:

```text
y_real = conv_r(x_real) - conv_i(x_imag)
y_imag = conv_r(x_imag) + conv_i(x_real)
```

Independent BN produces:

```text
z_real = a_real * y_real + b_real
z_imag = a_imag * y_imag + b_imag
```

If `a_real != a_imag`, there is no single folded `conv_r'` / `conv_i'` pair
that satisfies both equations exactly. Exact fold would require breaking the
complex weight sharing into four independent real kernels, which is a
different architecture.

Therefore this phase should prioritize **PReLU pow2 slope** and only treat BN
as a separate affine simplification or a constrained future experiment.

Potential hardware cleanup:

- keep BN as per-channel real/imag affine, possibly quantized to FP8 / pow2
  scale, rather than folded into conv.
- optionally add a constrained QAT experiment tying real/imag BN scales so BN
  becomes foldable.
- quantize PReLU negative slope to nearest power-of-two.
- replace runtime slope multiply with shift.

Expected compute gain is small, likely <1% of total MAC-equivalent work, but
it removes special elementwise units and simplifies the activation path.

### Experiments

| ID | Change | Train? | Purpose |
|---|---|---|---|
| A0 | PReLU slope pow2 PTQ only | no | measure direct quality loss |
| A1 | PReLU slope pow2 QAT | short 5-10ep | recover if A0 drops |
| A2 | BN affine scale quantized to FP8 / pow2, not folded | no / short QAT | reduce BN affine cost |
| A3 | tie real/imag BN affine scale, then fold | short QAT | test whether foldable BN is recoverable |
| A4 | fold only layers where `a_real ~= a_imag` | no | opportunistic exact/near-exact fold |

### Gates

1. A0/A1 20-pair sanity must pass MBSTOI >= 0.8031.
2. A2-A4 must report whether the transform is exact, approximate, or requires
   architectural change.
3. If a variant passes sanity, run full VCTK + TIMIT.

### RTL endpoint candidate

```text
Conv/Deconv -> real/imag affine BN unit or constrained folded BN -> PReLU pow2 shift
```

Do not claim "no standalone BN unit" unless A3/A4 proves an exact or
quality-safe folding route.

---

## 4. Phase B: q0.8 probability packet + Score x V datapath

### Motivation

P3 already showed q0.8 probability values are quality-safe in isolation.
The next step is to turn that into a real hardware packet format.

### Packet format

Per row, after top-k=4:

```text
for each of 4 entries:
  local_idx : 5 bits  // key position within 32-entry attention window
  prob_q08  : 8 bits
```

Storage:

```text
4 * (5 + 8) = 52 bits/row
```

This replaces dense probability storage:

```text
32 * FP16 = 512 bits/row
32 * q0.8 = 256 bits/row
```

### Experiments

| ID | Change | Train? | Purpose |
|---|---|---|---|
| B0 | reproduce `b_prob_q08` isolated result | no / existing | anchor |
| B1 | bit-accurate q0.8 packet, dequant to FP16 before Score x V | no | storage-only RTL model |
| B2 | q0.8 fixed x FP16 V with fixed-point product | short QAT if needed | true datapath |
| B3 | q0.8 packet + L-Mul Score x V | no-train first | unified PE candidate |
| B4 | stack B2/B3 with P3 `a1 + c2b` | 20-pair sanity | combined-stack risk check |

### Gates

- B1 should match `b_prob_q08` within numerical tolerance.
- B2/B3 must pass 20-pair sanity before full eval.
- Full sign-off requires VCTK + TIMIT.

---

## 5. Phase C: Unified mixed/L-Mul PE array

### Motivation

At 5-20MHz, the existing 64 FP16 multiplier island is not the main latency
bottleneck. Its main cost is area/power/design fragmentation.

Goal: merge mixed PE and FP16 attention multiplier into one mode-configurable
low-power PE.

### Candidate PE modes

| Mode | Operation | Use |
|---|---|---|
| M0 | INT4 weight x FP8 activation | W4A8 conv/linear |
| M1 | FP8 x FP8 | FP8-preserved outlier paths |
| M2 | FP16/BF16 L-Mul M4 | QK and Score x V |
| M3 | q0.8 x FP16 or q0.8 x FP8 | probability packet x V |
| M4 | bypass / add / shift | softmax and folded affine support |

### Experiments

| ID | Change | Train? | Purpose |
|---|---|---|---|
| C0 | bit-accurate L-Mul M4 simulator for QK | no | replace tensor-level P4 simulator |
| C1 | bit-accurate L-Mul M4 simulator for Score x V | no | validate no-train Score x V path |
| C2 | combined attn_m4 bit-accurate no-train | no | match P4 option B |
| C3 | combined attn_m4 + q0.8 packet | no / short QAT | unified attention datapath |
| C4 | active-lane sweep: 64 / 128 / 256 lanes at 5 / 10 / 20MHz | no | low-power operating point |

### Gates

- C0-C2 should reproduce P4 within small metric drift.
- C3 must pass 20-pair sanity before full eval.
- C4 should report cycles/frame, active duty cycle, and SRAM bandwidth.

---

## 6. Phase D: MHA Q/K/V/O grouped or block-sparse Linear

### Motivation

This is the highest-impact compute future. Current attention projection costs
about **1.05M real MAC/frame**, about 75% of current total compute.

Ideal grouped reduction:

| Grouping | Attention projection MAC/frame |
|---|---:|
| dense | 1,048,576 |
| group=2 | about 524,288 |
| group=4 | about 262,144 |
| group=8 | about 131,072 |

Quality risk rises with grouping strength.

### Candidate variants

| ID | Change | Risk | Note |
|---|---|---|---|
| D0 | QKV grouped g=2 | low-medium | start here |
| D1 | QKV grouped g=4 | medium | likely useful |
| D2 | QKV + out_proj grouped g=4 | medium-high | larger saving |
| D3 | input_proj/output_proj grouped g=4 | high | affects complex bottleneck interface |
| D4 | all attention projections grouped g=4 | high | maximum clean grouped story |
| D5 | 16x16 block-sparse MHA projection | medium-high | pairs naturally with block-mask encoding |
| D6 | head-aligned block diagonal Q/K/V | high | big saving, may reduce cross-head mixing |

### Training plan

1. Warm-start by slicing dense weights into grouped/block structures.
2. Short QAT 10ep, lr=2e-4.
3. If 20-pair sanity passes, run full VCTK.
4. Promote only best variants to TIMIT.
5. If g=4 fails but g=2 passes, try longer 30ep QAT.

### Gates

Use the same P3/P4 gates:

| Metric | Gate |
|---|---|
| VCTK MBSTOI | >= 0.8302 |
| TIMIT MBSTOI | >= 0.7869 |
| VCTK ILD | <= 3.45 |
| VCTK IPD | <= 16.74 |
| TIMIT ILD | <= 3.20 |
| TIMIT IPD | <= 21.20 |

---

## 7. Phase E: Weight encoding

### 7.1 What is suitable now

Current weights are mostly dense after W4A8. For dense weights, scalar
value-index encoding is usually not efficient because index bits dominate.

Recommended immediate formats:

#### E0. Packed W4 + FP8 scale

```text
group header:
  scale_fp8 : 8 bits
payload:
  32 signed INT4 weights packed into 16 bytes
```

Effective payload:

```text
4.25 bits/weight = 4b weight + 0.25b scale overhead
```

This is the best first encoding for current dense W4 layers.

#### E1. Structural encode for grouped decoder

Do not store implicit zeros outside grouped connectivity. The group structure
is compile-time metadata:

```text
groups_x, groups_skip, in_skip_selected, out_channels
```

No value-index pair is required for structural zeros.

#### E2. Static skip-index LUT

P2 `s050_topk` already uses static skip indices. Store a small per-stage LUT:

```text
stage -> selected skip channel IDs
```

The decoder reads only selected skip channels.

### 7.2 What becomes suitable after sparsity

#### E3. Block-mask + packed values

For a 16-weight block:

```text
mask: 16 bits
values: NZ * 4 bits
```

Break-even vs dense W4:

```text
16 + 4*NZ < 64  ->  NZ < 12
```

So it becomes useful when block sparsity is greater than about 25%.

#### E4. Value-index pair

Good only when scalar sparsity is high:

| Format | bits / nonzero | Break-even vs 4.25b dense W4 |
|---|---:|---:|
| 4b value + 8b index | 12b | needs > about 65% sparsity |
| 8b value + 8b index | 16b | needs > about 73% sparsity |
| 4b value + 4b local index | 8b | needs > about 47% sparsity |

This matches the ISSCC SpeechBMI idea, where sparse value-index pairs are
useful because the model has high sparsity. For the current BCCTN model, use
it only after MHA block-sparse or N:M sparse QAT creates enough sparsity.

### 7.3 Other encoding candidates

| Encoding | Suitable? | Comment |
|---|---|---|
| packed INT4 + FP8 scale | yes, immediate | best baseline |
| block floating / shared exponent | yes | already close to W4 + scale |
| block-mask sparse | yes, after block sparsity | better than scalar val-index for hardware |
| N:M sparsity, e.g. 2:4 or 4:8 | possible | easy PE scheduling, needs QAT |
| codebook / palette weights | possible | index selects centroid; may need LUT and QAT |
| power-of-two weights | risky | multiplier-free but quality risk |
| CSD signed-digit weights | possible but complex | useful if replacing small multipliers with shift-add |
| Huffman / entropy coding | not first choice | variable length hurts timing |
| RLE | not first choice | only works with long zero runs |
| delta encoding | not first choice | learned weights are not smoothly ordered unless reordered |
| low-rank factorization | possible | changes algorithm structure; useful for MHA Linear |

---

## 8. Phase F: Low-power operating-point sweep

### Architecture candidates

| Candidate | Active lanes | Clock | Note |
|---|---:|---:|---|
| F0 | 256 mixed + 64 FP16 | 5MHz | current over-provisioned low-frequency baseline |
| F1 | 128 unified PE | 5MHz | safer low-power candidate |
| F2 | 64 unified PE | 5MHz | minimum viable candidate |
| F3 | 64 unified PE + grouped MHA g=4 | 5MHz | likely best energy point |
| F4 | 256 unified PE burst + sleep | 5MHz | good if leakage is controlled |

### Report per candidate

1. cycles/frame
2. duty cycle under 5ms hop
3. SRAM reads/frame
4. weight bytes/frame if not fully resident
5. estimated multiplier count
6. exact vs L-Mul lane count
7. quality status

---

## 9. Recommended execution order

1. **A0-A1 PReLU pow2 cleanup**: low risk and likely useful.
2. **A2-A4 BN simplification only as a constrained experiment**: exact BN fold
   is not generally valid for current complex conv.
3. **B1-B3 q0.8 packet + Score x V datapath**: future recovery experiment,
   not ship-path work, because full-stack q0.8 was deferred.
4. **C0-C3 bit-accurate L-Mul / unified PE**: removes the FP16 multiplier
   island.
5. **D0-D4 MHA projection grouping**: highest compute and energy impact.
6. **E0-E4 weight encoding**: start dense-packed; introduce value-index only
   after sparsity exists.
7. **F0-F4 operating-point sweep**: choose 64/128/256 active-lane low-power
   target.

---

## 10. Decision log

1. q0.8 probability quantization was tested as an algorithmic value grid, not
   as a fully bit-accurate q0.8 x V RTL datapath. It is deferred from the ship
   endpoint because Option C + q0.8 + FP8 scale misses the 0.835 VCTK target.
2. At 5MHz, current compute is feasible with 256 lanes and tight with 64
   lanes. The low-power problem is area, switching activity, SRAM traffic, duty
   cycle, and utilization.
3. QK/Score x V is no longer the largest bottleneck because of window=32 and
   top-k=4.
4. The largest remaining compute block is attention projection / Linear.
5. Value-index encoding is not a good first encoding for dense W4 weights.
   It becomes attractive only after block or scalar sparsity is introduced.
6. Exact BN fold is not generally valid for the current independent real/imag
   `NaiveComplexBatchNorm2d` after complex conv. PReLU pow2 slope is the safer
   near-term cleanup.
7. The most defensible low-power story is:

```text
PReLU pow2 + constrained BN simplification
+ unified mixed/L-Mul PE
+ grouped/block-sparse MHA projection
+ packed W4 / block-sparse weight stream
```
