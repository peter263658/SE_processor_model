# Future Work: P3 Attention FP8 Datapath Recovery

Date: 2026-05-17  
Updated: 2026-05-18 after P2 sign-off and Claude review  
Workspace: `/raid/R12K41024/12_decoder_grouped_pointwise`  
Depends on: P2 `g42 + s050_topk` sign-off

This document is a handoff for Claude Code / Codex to continue after P2.
The goal is to turn the next phase into a controlled set of experiments rather
than a loose collection of hardware ideas.

---

## 0. Executive Summary

P1 already reduces decoder pointwise compute with branch-asymmetric grouped
pointwise:

```text
P1 primary: g42 = groups_x=4, groups_skip=2
VCTK MBSTOI: 0.8361 vs baseline 0.8360
TIMIT MBSTOI: 0.7923 vs reference 0.7927
decoder pointwise MAC: -62.2%
conv-only total MAC: -38.7%
```

P2 is now complete. The final RTL primary candidate is:

```text
P2 primary: g42 + s050_topk
groups_x = 4
groups_skip = 2
skip_select_ratio = 0.50
skip_select_mode = topk_l1

VCTK MBSTOI: 0.8352, delta vs baseline = -0.0008
TIMIT MBSTOI: 0.7919, delta vs reference = -0.0008
decoder pointwise MAC vs baseline: -74.74%
conv-only total MAC vs baseline: -46.52%
```

P2 result source of truth:

```text
RESULTS_P2_SKIP_2026-05-18.md
```

The P2 handoff from 2026-05-17 is historical pre-result context. Use
`RESULTS_P2_SKIP_2026-05-18.md` over `HANDOFF_P2_2026-05-17.md` whenever the
two disagree about winner/status.

After P2, P3 should attack the attention arithmetic datapath:

```text
P3 goal:
  make attention datapath arithmetic / score storage / scale handling more
  FP8- or pow2-friendly, then recover quality with short QAT finetuning.

Target:
  keep VCTK MBSTOI around 0.835-0.836, or at least within -0.005 of the
  final P2 candidate, with no TIMIT collapse and no ILD/IPD regression above
  5%.
```

Important framing:

- W4A8 still matters because it reduces DMA / SRAM traffic.
- P1/P2 are MAC-count reductions: grouped decoder pointwise and skip-select
  remove decoder pointwise work.
- P3 is datapath-precision work: bit-width, score/probability storage,
  multiplier width, and quantization-scale representation. It should not be
  reported as a new MAC-count reduction unless the report explicitly counts the
  changed attention score/probability matmuls.
- Grouped or structured MHA `Q/K/V/O` Linear is the later attention
  MAC-reduction direction. Keep that as a separate spec.
- L-Mul should be treated as a later optional arithmetic backend, not the first
  P3 change.

---

## 1. P2 Final Reference

P2 has been reported cleanly. Do not rerun P2 unless a new issue is found.

Current P2-related files:

```text
HANDOFF_DECODER_GROUPED_P2_2026-05-17.md
HANDOFF_P2_2026-05-17.md
RESULTS_P1_GROUPED_2026-05-17.md
RESULTS_P2_SKIP_2026-05-18.md
SPEC_DECODER_SPLIT_POINTWISE_2026-05-16.md
```

Current P2 scripts:

```text
scripts/make_phase2_skip_warmstarts.sh
scripts/run_phase2_skip_sweep.sh
scripts/run_phase2_full_eval.sh
scripts/run_phase2_timit.sh
scripts/count_p2_macs.py
```

Final P2 primary artifacts:

```text
P2_BEST_CKPT:
  /raid/R12K41024/12_decoder_grouped_pointwise/outputs/p2_skip/s050_topk_2026-05-17_21-27-05/logs/lightning_logs/version_0/checkpoints/epoch=27-step=27104.ckpt

P2_INDEX:
  /raid/R12K41024/12_decoder_grouped_pointwise/Checkpoints/p2_skip_indices_g42_topk_l1_r050.pt

P2 full VCTK result:
  results/p2_skip_full/s050_topk_w750/

P2 TIMIT result:
  results/p2_skip_timit/s050_topk_vctk_timit_w750/
```

P2 conservative backup:

```text
g42 + s075_topk
  VCTK MBSTOI delta vs baseline: -0.0010
  TIMIT MBSTOI delta vs reference: -0.0024
  decoder pointwise MAC vs baseline: -68.45%
```

No last-layer skip override is needed. `RESULTS_P2` shows the feared mask-head
IPD issue did not appear: `s050_topk` IPD was better than P1 `g42` on both
VCTK and TIMIT.

---

## 2. Current Code Anchors

These are the files/lines a new agent should inspect first.

Attention softmax and attention forward:

```text
DCNN/utils/complexPyTorch/complexLayers.py:678
  _canonical_softmax_mode

DCNN/utils/complexPyTorch/complexLayers.py:732
  _shift_based_softmax

DCNN/utils/complexPyTorch/complexLayers.py:783
  _shift_topk_config

DCNN/utils/complexPyTorch/complexLayers.py:832
  _shift_topk_softmax

DCNN/utils/complexPyTorch/complexLayers.py:908
  _mha_attention_forward

DCNN/utils/complexPyTorch/complexLayers.py:949
  existing fp16_accumulator cast for q/k/v

DCNN/utils/complexPyTorch/complexLayers.py:954
  QK^T / sqrt(D)

DCNN/utils/complexPyTorch/complexLayers.py:988
  Score x V
```

Attention module construction:

```text
DCNN/models/model.py:455
  MultiAttnBlock

DCNN/models/model.py:468
  ComplexMultiheadAttention construction

DCNN/models/model.py:504
  x = self.mattn(x, attn_mask=attn_mask)
```

Mixed-precision QAT wiring:

```text
DCNN/utils/mixed_precision_qat.py:199
  _make_fp8_activation_hook

DCNN/utils/mixed_precision_qat.py:211
  _make_split_pointwise_activation_hook

DCNN/utils/mixed_precision_qat.py:263
  DEFAULT_CONFIG precision routing

DCNN/utils/mixed_precision_qat.py:313
  _set_topk_on_complex_mha

DCNN/utils/mixed_precision_qat.py:345
  enable_mixed_precision_qat
```

Trainer bridge from Hydra config to QAT:

```text
DCNN/trainer.py:75
  mixed_precision_qat branch

DCNN/trainer.py:101
  enable_mixed_precision_qat(...)

DCNN/trainer.py:106
  training.fp16_accumulator is already passed through
```

Config:

```text
config/config.yaml
  model.attention_softmax.mode
  model.decoder_grouped_pointwise.*
  training.mixed_precision_qat
  training.qat_scope
  training.top_k_attention
  training.int4_group_size
```

`training.fp16_accumulator` is already consumed by the trainer even though it
is not yet listed as a default field in `config/config.yaml`. It can be passed
as a Hydra override today.

Verified on 2026-05-18:

```text
DCNN/trainer.py:106
  fp16_accumulator=training_cfg.get("fp16_accumulator", False)

Default:
  false, unless explicitly set by config/override.

P2 primary:
  frozen P2 config/hparams do not set training.fp16_accumulator, so P2 used the
  false/default A0 path.

When true:
  q/k/v are cast to FP16 after QKV projection and optional activation quant.
  QK^T, mask addition, shift-topk softmax, and Score x V then run on FP16
  tensors. The attention output is cast back to the original dtype before
  out_proj.
```

---

## 3. P3 Main Hypothesis

Current strict W4A8 uses FP8 activation fake quantization and INT4 / FP8 weight
routing, but the attention arithmetic path is still only partly hardware-like.

P3 hypothesis:

```text
After P2, a short QAT finetune can recover MBSTOI near 0.836 even when:
  - Q/K/V activations are FP8-quantized,
  - attention score storage is quantized to FP8 or fixed-point,
  - top-k softmax uses pow2 / exponent-domain normalization,
  - selected quantization scales are represented as FP8 or pow2.
```

This is a continuation of the existing hardware story:

```text
Softmax phase:
  replace expensive exp/normalization with top-k shift/pow2 softmax

P1/P2:
  reduce decoder pointwise MAC and skip bandwidth.
  Final candidate is g42 + s050_topk.

P3:
  reduce attention datapath/storage precision cost.
  This is not the same as reducing MAC count.

Later MHA Linear work:
  grouped/structured Q/K/V/O Linear can reduce MHA projection MAC count, but it
  changes topology and needs a separate spec.
```

There is also a separate future direction suggested by `RESULTS_P2`: grouped or
structured MHA `Q/K/V/O` Linear layers. That direction may have high MAC impact,
but it changes the attention projection topology and should be a separate spec.
This P3 file focuses first on datapath precision and score/probability storage.

---

## 4. P3 Experiment Order

### P3-A: Existing FP16 Accumulator Knob

This is the first experiment because it needs no code change.

```text
P3-A0: P2 best, current attention path
P3-A1: P2 best + training.fp16_accumulator=true
```

Purpose:

- Establish whether the hardware-intended FP16 QK / softmax / Score x V path
  itself changes quality.
- Use this as the reference before changing score/prob storage to FP8.

Current status:

- A0 is the real P2 primary path because the default is
  `training.fp16_accumulator=false`.
- A1 is not a no-op. It changes QK^T, mask addition, shift-topk softmax, and
  Score x V to operate on FP16 tensors inside `_mha_attention_forward`.
- Add `training.fp16_accumulator: false` to `config/config.yaml` before P3 code
  changes so future overrides are explicit and discoverable.

Example train command:

```bash
cd /raid/R12K41024/12_decoder_grouped_pointwise

P2_BEST_CKPT=/raid/R12K41024/12_decoder_grouped_pointwise/outputs/p2_skip/s050_topk_2026-05-17_21-27-05/logs/lightning_logs/version_0/checkpoints/epoch=27-step=27104.ckpt
P2_INDEX=/raid/R12K41024/12_decoder_grouped_pointwise/Checkpoints/p2_skip_indices_g42_topk_l1_r050.pt

CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n bcctn python -u train.py \
  training.train_checkpoint_path="$P2_BEST_CKPT" \
  training.n_epochs=30 \
  training.learning_rate=2e-4 \
  training.qat_scope=full \
  training.top_k_attention=4 \
  training.fp16_accumulator=true \
  model.attention_softmax.mode=shift_topk_pow2_clamp8_nomantissa \
  model.decoder_grouped_pointwise.mode=grouped \
  model.decoder_grouped_pointwise.groups_x=4 \
  model.decoder_grouped_pointwise.groups_skip=2 \
  model.decoder_grouped_pointwise.skip_select_ratio=0.5 \
  model.decoder_grouped_pointwise.skip_select_mode=topk_l1 \
  model.decoder_grouped_pointwise.skip_indices_path="$P2_INDEX" \
  hydra.run.dir=outputs/p3_attention_fp8/a1_fp16_accum_\${now:%Y-%m-%d_%H-%M-%S}
```

Learning-rate fallback rule (tied to the §4 P3-D screening flow):

- Run the 5-10 epoch screening at `learning_rate=2e-4` first.
- If screening shows no recovery trend toward the 20-pair sanity floor 0.8031,
  retry one screening pass at `learning_rate=5e-4` from the same P2 ckpt
  before killing the variant.
- Do not promote a variant to a full 30-epoch finetune until at least one LR
  has produced a recovering screening curve.

### P3-B: Attention Score / Probability Quantization

These variants need code changes in `_mha_attention_forward`.

Do not start with a full cumulative score/probability stack. The existing
`shift_topk_pow2_clamp8_nomantissa` softmax already discards most mantissa
information in the selected exponent-domain path. Score FP8 quantization may be
redundant, and it can still destabilize top-k ordering. Run the pre-flight dump
first, then run isolated ablations.

Recommended config addition:

```yaml
model:
  attention_datapath:
    enabled: false
    score_quant: "none"       # {"none", "fp8_e4m3", "fixed"}
    prob_quant: "none"        # {"none", "fp8_e4m3", "q0.8", "pow2"}
    qk_scale_mode: "pow2"     # {"pow2", "fp16", "fp8_e4m3"}
    score_fixed_int_bits: null    # derive from B0 score distribution
    score_fixed_frac_bits: null   # derive from B0 score distribution
    topk_stability_min_jaccard: 0.98
    top1_flip_max_rate: 0.01
```

#### B0. Pre-flight, no training

Before implementing/running score quantization or fixed-Q score variants, dump
P2 attention score tensors from the P2 best checkpoint and measure
quantization sensitivity offline. No training is required for B0.

Sample scope:

```text
ckpt:
  P2_BEST_CKPT from §1

subset:
  at least 8 distinct utterances from the VCTK validation set, sampled to span
  high-SNR and low-SNR conditions. Reuse the 20-pair sanity subset if
  convenient, but extend it if any individual layer has fewer than 1000 rows.

rows:
  collect >= 1000 attention rows (batch x heads x time positions) per MHA
  layer.

layers:
  dump from every MHA layer constructed in DCNN/models/model.py:455
  (MultiAttnBlock). Report per-layer statistics; do not collapse layers into a
  single global average. The gate below is evaluated on the worst-case
  layer/head, not on the mean.

mask:
  before any FP8 simulation, dump the actual additive mask value(s) used in
  `_mha_attention_forward`. Record whether the mask is literal -inf, -1e9,
  FP16 min, or another finite very-negative value. This decides whether the
  score-quant helper can pass mask values through unchanged or must clamp
  them to FP8 negative max.
```

Required measurements:

```text
score distribution (per layer, per head):
  finite min / max / p1 / p50 / p99 / p99.9
  histogram with >= 256 bins, bin width fine enough to read the
  min-resolvable step between adjacent finite scores
  saturation rate at each candidate fixed-Q (int_bits, frac_bits)

mask value audit:
  unique mask values observed, with counts per layer
  classify each as {literal_inf, finite_very_negative}
  if any finite_very_negative is observed, record min(|mask|) so the
  score-quant helper can pick a safe clamp threshold

FP8 score simulation:
  finite-only FP8 E4M3 quantization applied after mask addition and before
  top-k.
  literal -inf masks pass through unchanged.
  finite very-negative masks (if present) are clamped to FP8 negative max and
  excluded from the "finite score" range used for quantization.

top-k stability (per layer, per head):
  top-4 set Jaccard
  top-4 exact-support match rate
  top-1 flip rate
  row-max value/order flip rate
```

Gate (worst-case head across all MHA layers, not global mean):

```text
Proceed to B_score_fp8 only if, for every layer and every head:
  top-4 Jaccard >= 0.98
  top-1 flip rate <= 1%

If any single head fails:
  skip B_score_fp8 and prefer either FP16 score registers, or the fixed-Q
  format derived from the rule below.
```

Bit-width derivation for `B_score_fixed` (apply per layer, then take the
worst case across layers):

```text
int_bits  = ceil(log2(max(|p99.9|, |p0.1|))) + 1     # +1 sign bit
frac_bits = ceil(log2(1 / min_resolvable_step))      # from score histogram

Acceptance per layer/head:
  saturation_rate(int_bits) <= 1% at p99.9 of the per-layer distribution
  top-4 Jaccard with this fixed-Q simulation >= 0.98 per head
  top-1 flip rate with this fixed-Q simulation <= 1% per head

If acceptance fails:
  widen int_bits by 1 and rerun the simulation.
  if still failing at int_bits + 2, drop the fixed-Q variant for that layer
  and fall back to FP16 score registers there.
```

The earlier `score_fixed_int_bits=5` and `score_fixed_frac_bits=8` numbers
are placeholders. Fill them only after running B0 with the rule above, and
record the per-layer derived values in the `RESULTS_P3_ATTENTION_FP8` file.

Recommended ablation matrix:

| variant | change | expected risk |
|---|---|---|
| B1 | existing FP8 Q/K/V activation + FP16 accumulator | low |
| B_score_fp8 | B1 + quantize finite attention scores to FP8 before top-k | medium, gated by B0 |
| B_score_fixed | B1 + fixed-Q score simulation before shift softmax | medium, bit-width from B0 |
| B_prob_only | B1 + quantize attention probabilities before Score x V | medium |
| B_score_fp8_prob | B_score_fp8 + probability quantization | high, only if isolated tests pass |
| B_full | score quant + fixed-Q/top-k path + prob quant | highest, only after isolated passes |

`B_prob_only` is mandatory because otherwise a full-stack quality drop cannot
be attributed to score quantization, fixed-Q score path, or probability
quantization.

Probability quantization format for `B_prob_only`:

```text
primary: q0.8 fixed-point
  This is the format that the hardware probability register actually
  constrains. After shift_topk_pow2_clamp8_nomantissa, probabilities are
  already highly discrete and approximately pow2-aligned, so q0.8 is what
  meaningfully tests the storage path.

sanity: fp8_e4m3
  Expected to be near no-op because the softmax output is already
  discretized. Run only to confirm the prob-quant insertion point and
  dataflow are wired correctly. Do not draw conclusions about probability
  storage cost from this variant.

pow2:
  Optional. Run only if both q0.8 and fp8_e4m3 sanity pass and the cue gates
  hold on the 20-pair sanity subset.
```

Implementation notes:

- Add a helper that quantizes only finite scores and preserves `-inf` mask
  values. Do not FP8-quantize `-inf` directly.
- Insert score quantization after `attn = QK^T / sqrt(D)` and mask addition,
  before `_shift_topk_softmax`.
- Insert probability quantization after `_shift_topk_softmax` or
  `_approx_attention_probs`, before `torch.matmul(attn, v)`.
- Keep dense scatter semantics unchanged for correctness comparison.
- Add a dump/simulation script before the training sweep, for example
  `scripts/dump_p3_attention_scores.py`, so B0 can run without touching model
  training.

### P3-C: Scale Representation

There are two different meanings of "scale"; do not mix them.

#### C1. QK scale

Current model has:

```text
embed_dim = 256
num_heads = 16
head_dim D = 16
1 / sqrt(D) = 1 / 4
```

So the QK scale is exactly a power-of-two. Hardware should implement this as
an exponent shift, not as an FP8 multiplier. In code, it currently appears as:

```text
DCNN/utils/complexPyTorch/complexLayers.py:954
  attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)
```

For P3, add a `qk_scale_mode="pow2"` path and assert/log that `D=16`.
If a future model uses non-pow2 `D`, fall back to `fp16` or run a separate
`fp8_e4m3` ablation.

#### C2. INT4 / W4A8 quantization scales

These are the per-channel / per-group / per-head scale tensors in:

```text
DCNN/utils/mixed_precision_qat.py:70
  _scale_per_channel

DCNN/utils/mixed_precision_qat.py:77
  _scale_per_group_last_dim

DCNN/utils/mixed_precision_qat.py:136
  Int4PerHeadMHAParametrization

DCNN/utils/mixed_precision_qat.py:161
  Int4PerHeadLinearParametrization
```

Recommended scale sweep:

| variant | scale format | note |
|---|---|---|
| C2a | scale remains FP16/FP32 in fake quant | reference |
| C2b | fake-quant scale to FP8 E4M3 | likely first useful test |
| C2c | pow2-only scale | most hardware-friendly, highest risk |

Do C2b before C2c. Pow2-only scale may increase INT4 quantization error.

C2c fallback rule:

```text
Run C2c only if C2b has at least about 0.002 MBSTOI margin over both full-data
floors:
  VCTK >= 0.8322
  TIMIT >= 0.7889

If C2b is within about 0.001 of either floor, or if ILD/IPD is close to the
cue gate, stop at C2b and make C2b the P3 scale endpoint.
```

### P3-D: Compute Budget and Early Stop

Do not assume all variants get 30 epochs.

Recommended flow:

1. Run B0 pre-flight before any score-quant/fixed-Q training.
2. For each trainable variant, start with a 5-10 epoch screening finetune from
   the P2 best checkpoint at `learning_rate=2e-4`.
3. Run the 20-pair sanity evaluation after screening.
4. Promote to a full 30-epoch finetune only if the sanity result is recovering
   toward the P2 floor and no cue metric is obviously broken.
5. Kill a variant if three consecutive validation checkpoints remain below the
   20-pair MBSTOI floor of 0.8031, or if the 20-pair sanity result is more
   than 0.003 below that floor (i.e. <= 0.8001) with no recovery trend across
   the screening window. "Recovery trend" means the most recent two sanity
   evaluations are non-decreasing.

---

## 5. Suggested Code Changes

Minimal P3-A requires no code changes.

For P3-B/C, make these scoped edits:

1. `config/config.yaml`
   - Add `model.attention_datapath` defaults.
   - Add explicit `training.fp16_accumulator: false` before the first P3 run
     for discoverability and override safety.

2. `DCNN/utils/complexPyTorch/complexLayers.py`
   - Add finite-only score quantization helpers. Mask handling follows the
     B0 mask-value audit: literal -inf passes through unchanged; finite
     very-negative mask values are clamped to FP8 negative max and excluded
     from the "finite score" range used for quantization.
   - Add probability quantization helpers supporting `q0.8`, `fp8_e4m3`, and
     `pow2` modes.
   - Extend `_mha_attention_forward(...)` with an optional
     `attention_datapath` dict argument.
   - Store `self.attention_datapath = {}` in `ComplexMultiheadAttention` as
     a plain Python attribute. Do not use `register_buffer` and do not
     register a submodule, so it does not appear in `state_dict()` and
     checkpoint resume does not bring back stale datapath config.
   - Pass `self.attention_datapath` into every `_mha_attention_forward`
     call.

3. `DCNN/utils/mixed_precision_qat.py`
   - Extend `enable_mixed_precision_qat(..., attention_datapath=None)`.
   - In `_set_topk_on_complex_mha`, assign `m.attention_datapath`.
   - Optionally add fake-quant of scale tensors for C2b/C2c.

4. `DCNN/trainer.py`
   - Pass `self.config["model"].get("attention_datapath", {})` into
     `enable_mixed_precision_qat`.

5. New scripts
   - `scripts/dump_p3_attention_scores.py`
     Reuses the P2 eval ckpt loader (same path as
     `scripts/run_phase2_full_eval.sh`) to load `P2_BEST_CKPT` and
     `P2_INDEX` from §1. Hooks every MHA layer's `_mha_attention_forward`
     to capture the post-mask pre-softmax score tensor and the actual mask
     tensor. Writes per-layer / per-head histograms plus raw row samples to
     a single `.npz`, and a JSON summary the B0 gate can be evaluated
     against. Should run without any code change to the model.
   - optional: `scripts/simulate_p3_attention_quant.py`
     Consumes the B0 dump and runs offline FP8 / fixed-Q / pow2
     simulations. Emits per-layer / per-head top-k stability, saturation,
     and bit-width derivation values. Used to fill in
     `score_fixed_int_bits` / `score_fixed_frac_bits` before any training.
   - `scripts/run_phase3_attention_fp8_sweep.sh`
   - `scripts/run_phase3_attention_full_eval.sh`
   - `scripts/run_phase3_attention_timit.sh`
   - optional: `scripts/count_p3_attention_storage.py`

Keep P3 scripts structurally similar to:

```text
scripts/run_phase2_skip_sweep.sh
scripts/run_phase2_full_eval.sh
scripts/run_phase2_timit.sh
```

---

## 6. Metrics and Gates

Primary reference:

```text
P2 best candidate: g42 + s050_topk.
```

Report both:

```text
delta vs P2 best
delta vs original strict W4A8 baseline
```

Reference values:

```text
Original strict baseline:
  VCTK MBSTOI = 0.8360
  TIMIT MBSTOI = 0.7927

P1 g42:
  VCTK MBSTOI = 0.8361
  TIMIT MBSTOI = 0.7923

P2 g42+s050_topk:
  VCTK MBSTOI = 0.8352
  TIMIT MBSTOI = 0.7919
  20-pair sanity MBSTOI = 0.8081
  VCTK ILD/IPD = 3.2871 dB / 15.94 deg
  TIMIT ILD/IPD = 3.0520 dB / 20.19 deg
```

Quality gates:

```text
20-pair sanity:
  MBSTOI >= 0.8031

Full VCTK:
  target around 0.835-0.836
  hard floor: 0.8302

TIMIT:
  no collapse
  hard floor: 0.7869

Binaural cues:
  ILD/IPD not worse by more than about 5%

P2-derived cue ceilings:
  VCTK ILD <= 3.45 dB, IPD <= 16.74 deg
  TIMIT ILD <= 3.20 dB, IPD <= 21.20 deg
```

Hardware reporting:

- Score register storage: compare `32 x FP16` vs `32 x FP8`.
- Top-k compare path: note whether compare is FP16 bit-level or FP8 bit-level.
- Normalization: prefer "exponent-domain pow2 normalization" wording over
  "barrel shifter normalization" where possible.
- Report P3 as datapath/storage/multiplier-width improvement, not as decoder
  MAC reduction.
- Do not claim full-model MAC unless MHA and attention score matmuls are
  included in the counter.

Recommended result file:

```text
RESULTS_P3_ATTENTION_FP8_2026-MM-DD.md
```

Suggested sections:

1. TL;DR
2. P2 reference
3. P3 experiment matrix
4. 20-pair sanity
5. Full VCTK
6. TIMIT
7. Attention datapath hardware implications
8. Decision and next candidate
9. Repro commands

---

## 7. Later Optional Work: L-Mul Backend

Do this after P3-B/C establishes a stable FP8-ish attention datapath.

Relevant papers:

```text
Addition is All You Need for Energy-efficient Language Models
https://arxiv.org/abs/2410.00907

A Power-Efficient Hardware Implementation of L-Mul
https://arxiv.org/abs/2412.18948
```

Recommended L-Mul order:

1. FP8 x FP8 L-Mul simulator on attention and FP8-preserved decoder PW layers.
2. Optional FP16 x FP16 L-Mul simulator only for non-critical paths.
3. Do not replace all FP16 multipliers before MBSTOI/ILD/IPD validation.

Before FP16 x FP16 L-Mul, write an explicit allowlist. In this project,
"non-critical" should exclude at least:

```text
Do not start with:
  QK score generation
  top-k max/compare path
  softmax normalization path
  final mask/output head
  any path whose P3 ablation shows ILD/IPD sensitivity

Initial candidates:
  FP8 x FP8 paths first
  FP8-preserved decoder PW paths
  selected projection or Score x V paths only after the P3 attention ablations
  prove they are not cue-sensitive
```

Important wording:

```text
L-Mul can approximate FP16xFP16, but it is not exact FP16 multiplication.
Existing hardware evidence is strongest for FP8 L-Mul.
```

This is a good P4, not the first P3 task.

---

## 8. Immediate Pickup Checklist

For the next agent:

0. Add explicit `training.fp16_accumulator: false` to `config/config.yaml`.
1. Read this file.
2. Read `RESULTS_P2_SKIP_2026-05-18.md`; treat it as source of truth.
3. Use the P2 primary checkpoint and index listed in §1.
4. Run P3-A0 evaluation (no retraining) and a P3-A1 5-10 epoch screening with
   `training.fp16_accumulator=true` at `learning_rate=2e-4`. Retry once at
   `5e-4` if no recovery trend per §4 P3-D. Promote to a full 30-epoch
   finetune only after a recovering screening curve.
5. Run 20-pair sanity at the end of screening and again at the end of full
   training; if both pass, run full VCTK + TIMIT.
6. Before any score-quant / fixed-Q training, run the B0 score-distribution
   dump on every MHA layer (>= 1000 rows per layer, >= 8 utterances) with
   the mask-value audit, then evaluate the per-head worst-case top-k
   stability gate.
7. Fill in `score_fixed_int_bits`, `score_fixed_frac_bits`, the mask-handling
   policy, and the prob-quant primary format (`q0.8`) in
   `model.attention_datapath` defaults from B0 results before any P3-B
   training run.
8. Implement `model.attention_datapath` and run isolated ablations in this
   order: `B_score_fp8` (only if the B0 gate passed), `B_score_fixed`, then
   `B_prob_only` with `q0.8` as primary and `fp8_e4m3` as sanity.
9. Apply the §4 P3-D 5-10 epoch screening + kill rule before committing any
   variant to a full 30-epoch run.
10. Only after isolated B variants pass should quantization-scale FP8 / pow2
    (C2b / C2c) be attempted, with the §4 P3-C C2c fallback rule enforced.
11. Treat L-Mul and grouped / structured MHA Linear as separate follow-up
    specs.
