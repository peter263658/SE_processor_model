# Consult — Softmax score-delta precision: RTL (sq4) vs deploy (fp16). Which is canonical? (2026-06-09)

## TL;DR
The RTL softmax (`Softmax_v3.sv`, == `00_TB/SoftMax/tb_SM.py`) and the **deployed** softmax
(`_shift_topk_softmax` in `10_algo/code/DCNN/utils/complexPyTorch/complexLayers.py`) **disagree on
5/16 attention heads** (14/512 dense probability entries differ by a **factor of 2**) on the real t*=351
MHA dump. Root cause is the **precision at which the top-4 score delta is computed before `floor(1.5·x)`**,
not the QK accumulator and not the denominator rounding. Need a decision on which is the sign-off target.

## Exactly where they diverge

Deploy `_shift_topk_softmax` (clamp_min=-8, use_mantissa=False, denom_rounding="nearest"):
```
row_max = topk_scores[...,:1]                 # fp16 (QK scores are fp16)
x       = topk_scores - row_max               # delta in FULL fp16
scaled  = (x * 1.5).clamp(-8, 0)              # full fp
integer = floor(scaled)                        # <-- floor of a full-precision negative
weight  = 2**integer                           # use_mantissa=False
denom   = round_pow2( sum(weight), "nearest" )
prob    = weight / denom
```
=> Any entry even **ε below the max** has `x<0` → `floor(1.5·x) ≤ -1` → weight ≤ 2^-1.
   **Only the exact max (x=0) keeps weight 2^0.** The distribution is **peaky**.

RTL `Softmax_v3.sv` (and its golden `tb_SM.py`): quantizes the score to **sq4 = Q?.4 (1/16, truncation)
BEFORE** the delta:
```
score_q = fp16_to_sq4(score)                   # 4 frac bits, truncation
delta_q = score_q - max_q                       # integer, 1/16 units
k       = clamp( (delta_q + (delta_q>>1)) >> 4, -8, 0 )
weight  = 1 << (8 + k)
```
=> Two entries that differ by **< 1/16** both map to `score_q = max_q` → `delta_q=0` → `k=0` → both get
   weight 2^0. The distribution is **flatter** for near-max ties.

## Evidence (real t*=351 dump, real_pass, all 16 heads, fp16-tree QK scores)
- **Full-fp16-delta model == deploy fp16-tree packet golden: 16/16 heads.**
- **sq4 model (= Softmax_v3) == deploy packet golden: 11/16 heads** (5 heads differ).
- The fp16-tree QK is NOT the cause: the fp16-tree packet == the deploy fp32-QK packet for **16/16** heads
  (the ~1-ULP QK difference did not flip any packet here).

Example (head 1, top-4 by score): idx14=0.56982 (max), idx29=0.56445, idx15=0.54541, idx20=0.49902.
- deploy weights: {14:2^0, 29:2^-1, 15:2^-1, 20:2^-1}  (idx29 is ε below max → floored to -1)
- sq4 weights:    {14:2^0, 29:2^0,  15:2^-1, 20:2^-1}  (idx29 sq4-ties the max → 0)
  → deploy prob_code [128,64,64,64] vs RTL [128,128,... after its own denom]. Factor-2 on the tied entry.

## The question(s) for the algorithm side
1. **Which is canonical for tape-out sign-off** — the deploy's full-fp16 `floor(1.5·(score−max))`
   (peaky), or the RTL's sq4-domain delta (flatter near-max)?
2. The `RTL_HANDOFF_SOFTMAX_2026-05-14.md` §3.4 sweep covered **rounding policy / clamp / mantissa** but
   (as far as we can tell) **not the score-delta fixed-point precision**. Was sq4 (Q?.4) ever evaluated
   for MBSTOI, or was the deploy 0.8360 measured with full-fp delta? If sq4 was never quality-checked,
   matching the deploy (fp16 delta) is the safe sign-off target.
3. If fp16 delta is canonical: confirm the RTL should compute the top-4 `(score − row_max)` in fp16,
   then `floor(1.5·x)` (1.5x = x + x>>1 in a fixed-point with enough frac bits to preserve sub-1/16
   differences), clamp [-8,0]. We will refactor `Softmax_v3.sv` accordingly and re-issue its golden
   (`tb_SM.py` also needs the same fix — both currently use sq4).

## What is already verified (so this is the only softmax open item)
- QK datapath (L-Mul m8 + PEG fp16 tree + ×2^-2 PPU SCALE): VCS bit-exact 512/512 (both passes).
- Score×V model == fp16-tree context golden: 256/256 (both passes).
- The deploy↔fp16-tree QK packet agreement (16/16) isolates this to the softmax weight quantization.
