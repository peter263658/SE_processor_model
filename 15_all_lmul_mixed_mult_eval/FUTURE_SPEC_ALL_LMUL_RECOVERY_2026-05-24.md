# Future Spec: All-LMul Score Recovery

Date: 2026-05-24

Purpose: keep future experiments aligned on the same definitions, protocol, and
staging order while trying to recover the remaining MBSTOI gap of the all-LMul
model.

## Goal

Recover the all-LMul model toward the current production gate without confusing
diagnostic ablations with deployable hardware points.

Fixed reference numbers:

- Production gate / baseline (`d1+c3`, packet enabled, conv real multiply):
  - VCTK MBSTOI: 0.8354
  - TIMIT MBSTOI: 0.7880
  - Checkpoint:
    `/raid/R12K41024/14_low_power_arch_qat/outputs/low_power_phase_d/d1_qkv_g4_2026-05-20_02-56-03/logs/lightning_logs/version_0/checkpoints/epoch=25-step=100672.ckpt`
- v3 all-LMul m4 QAT best (`epoch=3-step=968.ckpt`):
  - VCTK MBSTOI: 0.8264
  - TIMIT MBSTOI: 0.7824
  - Checkpoint:
    `/raid/R12K41024/15_all_lmul_mixed_mult_eval/outputs/all_lmul_mixed_qat/all_lmul_m4_bitexact_qat_2026-05-22_14-13-49/logs/lightning_logs/version_0/checkpoints/epoch=3-step=968.ckpt`
- v4 all-LMul fine-tune:
  - Run directory:
    `/raid/R12K41024/15_all_lmul_mixed_mult_eval/outputs/all_lmul_mixed_qat/all_lmul_m4_bitexact_qat_2026-05-24_02-49-17`
  - Best checkpoint: to be filled after v4 completes. During the run, the best
    observed checkpoint was `epoch=5-step=1452.ckpt`.
- Gap to recover:
  - VCTK: about +0.0090
  - TIMIT: about +0.0056

## Terminology

### Production Gate

The `d1+c3` reference stack that reproduced 0.8354 / 0.7880 under the current
`w=750` eval protocol.

Important: this gate uses packet attention and real-multiply convs. It is not an
all-LMul datapath.

### All-LMul m4

The target low-power student datapath:

- Conv2d: L-Mul m4
- ConvTranspose2d: L-Mul m4
- Linear / MHA projections: L-Mul m4
- MHA QK: L-Mul m4
- MHA Score x V:
  - packet setting: packet sparse top-k path with `score_v_mode=lmul`
  - no-packet setting: dense `score_v` L-Mul unless a sparse no-q0.8 path is
    explicitly implemented

### Packet Attention

`prob_packet.enabled=true` with top-k packet slots. This is not only a softmax
quantization feature. In the current implementation it also changes Score x V
from dense attention matmul to sparse top-k gather plus weighted sum.

Therefore:

- Packet eval can be faster than no-packet eval.
- Removing packet is an architecture ablation, not simply "removing a small
  softmax block".

### No-Packet + Score x V L-Mul

Correct fair no-packet ablation:

```text
model.attention_datapath.prob_packet.enabled=false
model.attention_datapath.lmul_targets=[qk,score_v]
```

Do not run only `prob_packet.enabled=false` with `lmul_targets=[qk]`; that lets
Score x V fall back to true `torch.matmul`, which is not a fair all-LMul result.

Current caveat: this path is dense and slow in software. It is useful as an
accuracy ablation, but it is not automatically the desired hardware datapath.

### Sparse Top-K No-q0.8

Desired future ablation:

- Keep top-k sparse indices / gather dataflow.
- Do not quantize probabilities to q0.8 packet codes.
- Use top-k probability values directly in L-Mul Score x V.

This is different from full no-packet dense Score x V. It likely requires a code
path, not only Hydra overrides.

## Eval Protocol

Use this protocol for any number compared against the gate:

```text
eval.py
--paper_style_eval
--limit_pairs 750
--batch_size 1
--metric_workers 32
VCTK + TIMIT
```

`--limit_pairs 750` is a cap. VCTK uses 750 pairs under this protocol. TIMIT has
fewer available matched pairs and is expected to use its available set, e.g. 160
pairs, under the same cap.

Quick probes may use `w=200`, but those results must be labeled quick-probe and
must not be mixed with final gate numbers.

Report at least:

- VCTK MBSTOI, PESQ, SegSNR improvement, ILD error, IPD error
- TIMIT MBSTOI, PESQ, SegSNR improvement, ILD error, IPD error
- per-SNR MBSTOI when interpreting low-SNR/high-SNR behavior

## Why fp16_accumulator=false Is Tested

`training.fp16_accumulator=false` is a diagnostic ablation only.

It answers: "Is part of the remaining MBSTOI gap caused by fp16 accumulator
rounding rather than L-Mul mantissa approximation or packet probability
quantization?"

The production gate also uses the fp16 accumulator setting. This ablation checks
whether the existing fp16 accumulator choice is still harmless after stacking it
with all-LMul convs and packet attention.

Interpretation:

- If fp32 accumulation improves MBSTOI meaningfully, accumulator precision is a
  real contributor and needs hardware discussion.
- If fp32 accumulation is unchanged, keep `fp16_accumulator=true` for the target
  datapath.

Do not treat fp32 accumulation as the default production recommendation unless
it buys enough score to justify the hardware cost.

## Experiment Staging

Keep attribution clean. Do not combine multiple new mechanisms in the first run.

### Phase 0: Finish Current Runs

Wait for:

- v4 fine-tune completion and full eval of best checkpoint.
- packet vs no-packet full eval on the same v4 checkpoint.

### Phase 1: Cheap Config Ablations

Run these as separate evals on the same checkpoint:

1. Packet top-k 8

```text
training.top_k_attention=8
model.attention_datapath.prob_packet.top_k=8
```

2. Packet renormalization

```text
model.attention_datapath.prob_packet.renormalize=true
```

3. Packet probability 10-bit

```text
model.attention_datapath.prob_packet.prob_bits=10
model.attention_datapath.prob_packet.prob_scale=1024
model.attention_datapath.prob_packet.prob_max_code=1023
```

4. fp32 accumulator diagnostic

```text
training.fp16_accumulator=false
```

5. SWA / checkpoint averaging, only if enough compatible late checkpoints exist.

SWA requires several compatible late checkpoints. With Lightning
`ModelCheckpoint(save_top_k=1)`, only the current best plus `last.ckpt` may be
available, which is usually not enough. For future SWA runs, save multiple late
checkpoints via `save_top_k=N` or an `every_n_epochs` checkpoint callback.

Decision rule:

- `>= +0.005 MBSTOI`: candidate path for QAT / longer training.
- `+0.002 to +0.004`: keep as secondary; combine only after attribution is
  clear.
- `<= +0.001`: treat as noise unless it has strong hardware value.

### Phase 2: Per-Layer Heatmap

Do not rely on a naive two-model hook comparison:

```text
model_real = all true multiply
model_lmul = all L-Mul
```

That measures downstream activation divergence, not local layer responsibility.
Early-layer differences contaminate all later hooks.

Preferred first heatmap:

- For each L-Mul-capable layer, feed the same input to the same layer twice:
  - one pass through the current L-Mul operator
  - one pass through an equivalent true-multiply operator
- Compute local operator error:

```text
relative_error = MSE(out_real - out_lmul) / mean(out_real^2)
```

Output sorted table:

```text
layer_name | layer_type | rel_error | abs_mse | output_power | shape
```

Then validate top hot layers using single-layer swap quick eval:

- baseline: all-LMul
- variant A: only one selected layer is true multiply
- variant B: only one selected layer is L-Mul m5
- quick eval: `w=200`, then full eval only if the signal is large enough

Variant A answers "how much does L-Mul in this layer cost?" Variant B answers
"does one extra mantissa bit in this layer recover the cost?" Run Variant A
first for hot layers, then Variant B only when Variant A shows a meaningful
signal.

Heatmap focus rule:

- Sort by local relative error and absolute MSE.
- Treat the top 10 layers, or the smallest set explaining about 70% of
  cumulative local MSE, as the first focus list.
- A relative error above about `1e-3` is a useful initial "hot" threshold, but
  MBSTOI-sensitive layers must still be confirmed by single-layer swap eval.

### Phase 3: Training-Heavy Paths

Use only after Phase 0-2 results are available.

1. True resume from v4 best

- Use a full-state checkpoint.
- Prefer `training.resume_ckpt_path`, not weight-only warm-start.
- LR candidates: `5e-6`, `2e-6`.

2. Output-level KD

Minimal first KD path:

- teacher: production gate / `d1+c3` checkpoint
- teacher frozen, eval mode
- student: all-LMul m4 target datapath
- loss:

```text
loss = task_loss(student, clean) + lambda_kd * kd_loss(student_output, teacher_output)
```

First acceptable output-level KD losses:

```text
kd_loss = MSE(student_waveform, teacher_waveform)
```

or

```text
kd_loss = L1(STFT(student_waveform) - STFT(teacher_waveform))
```

Start with:

```text
lambda_kd = 0.1 to 0.3
```

Use output-level waveform/STFT loss first. Do not add feature KD in the first KD
run.

3. Heatmap-guided feature KD

Only after local heatmap and/or single-layer swap identifies hot layers.

- Add feature alignment only on the top 2-3 hot layers.
- Do not apply global feature KD by default; it can over-constrain the student
  and prevent L-Mul-specific compensation.

4. Learnable bias correction

Defer until baseline KD is measured.

Reason: adding learnable bias correction in the first KD run confounds
attribution.

## Bias Correction Policy

Bias correction has three possible levels:

1. Offline calibrated bias patch
   - Estimate per-output-channel delta.
   - Add it into existing conv bias.
   - Hardware unchanged.

2. Toggleable buffer
   - Store delta as a named buffer and expose a config switch.
   - Useful for ablation.
   - Deploy by folding into bias.

3. Learnable parameter
   - Train the delta during QAT.
   - Potentially stronger, but must be evaluated after baseline KD.

Expected ceiling is modest because L-Mul error is not only DC bias; it is also
input-dependent gain/variance error.

Do not treat bias correction as Tier-1 unless heatmap/KD evidence says the mean
component dominates.

## Hardware Interpretation Rules

Avoid mixing accuracy-only probes with hardware candidates.

- fp32 accumulator: diagnostic unless score gain is large enough.
- dense no-packet Score x V: accuracy ablation, likely not a preferred hardware
  point.
- packet top-k 8: plausible hardware trade-off; keeps sparse packet dataflow.
- packet renormalize / prob_bits 10: plausible packet datapath tweaks.
- sparse top-k no-q0.8: promising architecture point, but needs explicit code
  path and hardware accounting.
- selective m5: only justify after heatmap and single-layer swap show a strong
  localized signal.
- selective real multiply: last-resort area/quality trade-off, only for hot
  layers with strong MBSTOI sensitivity.

## Current Priority Order

1. Complete current v4 and packet/no-packet evals.
2. Run Phase 1 cheap config ablations.
3. Build local operator error heatmap.
4. Validate hot layers with single-layer swap quick eval.
5. If no cheap path gives enough gain, start output-level KD.
6. Add feature KD or learnable bias only after baseline KD attribution is clear.
