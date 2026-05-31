# Handoff: Decoder Grouped Pointwise + P2 Skip Subset

Date: 2026-05-17  
Workspace: `/raid/R12K41024/12_decoder_grouped_pointwise`  
Source baseline folder: `/raid/R12K41024/11_weight_tied_mha_fast_mbstoi`

This handoff summarizes the technical decisions, current code state, and next commands for continuing P2. It intentionally excludes terminal/tmux/Terminus usage discussion.

---

## 1. Starting Point

Baseline from `11_weight_tied_mha_fast_mbstoi`:

- Strict W4A8 QAT
- Tied complex MHA
- RTL-style softmax mode: `shift_topk_pow2_clamp8_nomantissa`
- `top_k_attention=4`
- Reference checkpoint:

```bash
/raid/R12K41024/11_weight_tied_mha_fast_mbstoi/outputs/finetune_strict/2026-05-10_02-40-16/logs/lightning_logs/version_0/checkpoints/epoch=45-step=44528.ckpt
```

Softmax result from previous handoff:

- VCTK MBSTOI baseline/reference: about `0.8360`
- TIMIT reference: about `0.7927`
- Softmax RTL path already considered stable.

The new experiment folder was copied to:

```bash
/raid/R12K41024/12_decoder_grouped_pointwise
```

All intended code changes are inside `12_decoder_grouped_pointwise/`.

---

## 2. Main Algorithm Idea

Original decoder pointwise path:

```text
cat([x, skip]) -> dense pointwise 1x1 -> depthwise_transpose 5x1
```

R0 split identity:

```text
pointwise(cat([x, skip])) == pointwise_x(x) + pointwise_skip(skip)
```

This exact split is not itself an algorithmic gain. It is scaffolding that exposes two branches:

- `x`: decoder-state branch
- `skip`: encoder-skip branch

P1 then applies branch-asymmetric grouped pointwise:

```text
x    -> grouped pointwise_x    with groups_x
skip -> grouped pointwise_skip with groups_skip
add  -> depthwise_transpose
```

The key novelty claim should not be "grouped convolution is novel." The defensible claim is:

> Branch-asymmetric split grouped pointwise decoder: an exact decoder cat-pointwise factorization used as a hardware-algorithm interface, enabling different structured channel mixing for decoder-state and encoder-skip branches while preserving binaural cues.

Why decoder first:

- Decoder cat pointwise is a clear hotspot.
- The split has an exact algebraic reference.
- `x` and `skip` have different semantic roles, so asymmetric grouping is meaningful.
- RTL can avoid materializing the cat tensor and run two regular grouped MAC streams plus add.

Why encoder grouped PW is not in this phase:

- Encoder pointwise has no `x/skip` branch factorization.
- Encoder is the analysis transform and affects next encoder layers, MHA input, and decoder skips.
- Grouping encoder PW without shuffle may create fixed channel islands.
- It is a reasonable future P3, but higher risk and a different claim.

---

## 3. Channel Convention

Model config has:

```yaml
kernel_num: [16, 32, 64, 128, 256, 256]
```

Internally the model uses:

```text
[2, 16, 32, 64, 128, 256, 256]
```

Because this is complex, effective complex channels are `kernel_num_internal[idx] // 2`.

Decoder stages are:

| stage | idx | in_x | in_skip | out |
|---:|---:|---:|---:|---:|
| 0 | 6 | 128 | 128 | 128 |
| 1 | 5 | 128 | 128 | 64 |
| 2 | 4 | 64 | 64 | 32 |
| 3 | 3 | 32 | 32 | 16 |
| 4 | 2 | 16 | 16 | 8 |
| 5 | 1 | 8 | 8 | 1 |

The last layer falls back to dense for grouping because `gcd(in, out)=1`.

---

## 4. Implemented R0/P1 Code

### Config

File:

```bash
config/config.yaml
```

Added:

```yaml
model:
  decoder_grouped_pointwise:
    mode: "off"
    groups_x: 1
    groups_skip: 1
    channel_shuffle: false
    skip_select_ratio: 1.0
    skip_select_mode: "topk_l1"
    skip_indices_path: null
```

Modes:

- `off`: legacy cat path
- `split`: exact split, groups forced to 1
- `grouped`: split + grouped pointwise

### Model

File:

```bash
DCNN/models/model.py
```

Implemented:

- `_effective_groups(...)`
- `_DecoderBlockSplit`
- Decoder split/grouped construction
- Group fallback per layer via `gcd(in, out)`
- Decoder forward path:

```python
skip = encoder_out[-1 - idx]
if self.skip_select_active:
    skip_indices = getattr(self, f"skip_indices_{idx}")
    skip = skip.index_select(1, skip_indices)
if self.grouped_pw_mode == "off":
    x = torch.cat([x, skip], 1)
    x = self.model[idx](x)
else:
    x = self.model[idx](x, skip)
```

### Complex Layer

File:

```bash
DCNN/utils/complexPyTorch/complexLayers.py
```

Added:

```python
ComplexDepthwiseSeparableConvTranspose2dSplit
```

It applies:

```text
apply_complex(pointwise_x, x) + apply_complex(pointwise_skip, skip)
```

then the original complex depthwise transpose.

### QAT Scale Sharing

File:

```bash
DCNN/utils/mixed_precision_qat.py
```

Important catch:

- Split branches must share the legacy cat activation quantization boundary.
- Otherwise `g11` would no longer be a valid reference because branch-specific activation scales would create quantization drift.

Implemented split pointwise pre-hook:

```text
cat([x, skip]) -> FP8 quantize once -> split back to x/skip
```

Branch conv pre-hooks are skipped for split pointwise modules to avoid double quantization.

### R0/P1 Scripts

Existing scripts:

```bash
scripts/convert_ckpt_to_grouped.py
verify_decoder_split_equivalence.py
scripts/make_phase1_warmstarts.sh
scripts/run_phase1_grouped_sweep.sh
scripts/run_phase1_full_eval.sh
scripts/run_phase1_timit.sh
scripts/count_p1_macs.py
```

Generated P1 warm-starts:

```bash
Checkpoints/p1_warmstart_g11.ckpt
Checkpoints/p1_warmstart_g21.ckpt
Checkpoints/p1_warmstart_g12.ckpt
Checkpoints/p1_warmstart_g22.ckpt
Checkpoints/p1_warmstart_g42.ckpt
Checkpoints/p1_warmstart_g44.ckpt
```

---

## 5. R0 Verification

R0 exact split equivalence passed.

Representative results:

- Small smoke: final max diff about `1.311e-06`, decoder max about `9.775e-06`
- Default 4 random 16k: final max diff about `1.073e-06`, decoder max about `1.144e-05`
- PASS threshold was final wav max diff `< 1e-5`

Interpretation:

- Exact split is numerically equivalent.
- Later quality deltas can be attributed to grouping/QAT/finetune rather than split math.

---

## 6. P1 Results Review

Result file:

```bash
RESULTS_P1_GROUPED_2026-05-17.md
```

Baseline:

- VCTK MBSTOI: `0.8360`

P1 primary candidate:

```text
g42 = groups_x=4, groups_skip=2
```

Results:

| variant | VCTK MBSTOI | delta | TIMIT MBSTOI | TIMIT delta | dec_pw MAC delta | full_conv MAC delta |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.8360 | 0 | 0.7927 | 0 | 0% | 0% |
| g42 | 0.8361 | +0.0001 | 0.7923 | -0.0004 | -62.16% | -38.70% |
| g44 | 0.8346 | -0.0014 | 0.7901 | -0.0026 | -74.59% | -46.44% |
| g22 | 0.8348 | -0.0012 | not tested | - | -49.73% | -30.96% |

Review conclusion:

- `g42` is a clean primary.
- `g44` is a backup if RTL area pressure is very high, but ILD/IPD/val_loss show capacity pressure.
- `g22` is a conservative fallback.
- `g11` control at `0.8352` establishes the split/QAT finetune noise floor around `-0.0008`.

Important reporting caveat:

- `full_conv MAC` means conv-only total MAC:

```text
encoder conv + decoder conv/deconv Conv2d/ConvTranspose2d MAC
```

It does not include:

- MHA Linear projections
- attention scores / softmax / score x V
- STFT / iSTFT
- BN / PReLU
- cat / skip index select / memory bandwidth

Use wording such as `conv-only total MAC` to avoid implying true full-model MAC.

---

## 7. What g42 Changes

Deepest layer example:

```text
x    : [B, 128, F, T]
skip : [B, 128, F, T]
out  : [B, 128, F, T]
```

The tensor shape does not change in P1.

What changes is the channel connectivity inside the pointwise weight matrix.

Original dense:

```text
out[o] sees all 128 x channels + all 128 skip channels
MAC per spatial point = 128 out * 256 in = 32768
```

g42:

```text
x branch groups=4:
  each output sees 128/4 = 32 x channels
  MAC = 128 * 32 = 4096

skip branch groups=2:
  each output sees 128/2 = 64 skip channels
  MAC = 128 * 64 = 8192

total = 12288
```

Saving:

```text
1 - 12288 / 32768 = 62.5%
```

So P1 does not reduce external channel dimensions. It reduces cross-channel coupling by turning dense all-to-all channel mixing into structured block-diagonal mixing.

P2 is the phase that actually reduces skip channel count.

---

## 8. Implemented P2 Code

P2 target:

```text
P1 best g42 + static skip subset selection
```

Implemented as group-aware fixed index selection. This matters because `g42` has `groups_skip=2`; arbitrary global top-k would scramble grouped conv layout. Each skip group keeps the same number of channels.

### P2 Model Path

File:

```bash
DCNN/models/model.py
```

Implemented:

- `skip_select_ratio < 1.0`
- `skip_select_mode in {"first_n", "topk_l1"}`
- `skip_indices_path`
- group-aware keep counts
- validation that indices are:
  - correct length
  - unique
  - in range
  - stored group-wise
- registered persistent buffers:

```text
decoder.skip_indices_0
decoder.skip_indices_1
...
decoder.skip_indices_5
```

For g42:

```text
s075 keep = [96, 96, 48, 24, 12, 6]
s050 keep = [64, 64, 32, 16, 8, 4]
```

### P2 Scripts Added

```bash
scripts/precompute_skip_indices.py
scripts/convert_ckpt_to_skip_subset.py
scripts/make_phase2_skip_warmstarts.sh
scripts/run_phase2_skip_sweep.sh
scripts/run_phase2_full_eval.sh
scripts/run_phase2_timit.sh
scripts/count_p2_macs.py
```

Purpose:

- `precompute_skip_indices.py`
  - `first_n`: group-wise first channels
  - `topk_l1`: run P1 checkpoint encoder over validation set, compute per-channel mean L1, pick group-wise top-k
- `convert_ckpt_to_skip_subset.py`
  - crop only `pointwise_{r,i}_skip` weights
  - keep x branch, depthwise, BN/PReLU, encoder, MHA unchanged
  - save skip index buffers into checkpoint
- `make_phase2_skip_warmstarts.sh`
  - batch-create P2 index files and warm-start checkpoints
- `run_phase2_skip_sweep.sh`
  - train P2 variants and run 20-pair sanity
- `run_phase2_full_eval.sh`
  - full VCTK eval
- `run_phase2_timit.sh`
  - cross-domain TIMIT eval
- `count_p2_macs.py`
  - analytical P2 MAC counter

### Smoke Artifacts

Already generated for first_n s075:

```bash
Checkpoints/p2_skip_indices_g42_first_n_r075.pt
Checkpoints/p2_warmstart_g42_first_n_r075.ckpt
```

These are smoke artifacts and can be reused or regenerated.

One-batch topk smoke artifacts were written under `/tmp`:

```bash
/tmp/p2_skip_indices_g42_topk_r075_smoke.pt
/tmp/p2_warmstart_g42_topk_r075_smoke.ckpt
```

They are not full-validation P2 artifacts.

---

## 9. P2 Smoke Checks Completed

Commands/checks performed:

### Syntax

```bash
python -m py_compile \
  DCNN/models/model.py \
  scripts/precompute_skip_indices.py \
  scripts/convert_ckpt_to_skip_subset.py \
  scripts/count_p2_macs.py
```

Passed.

### Bash Syntax

```bash
bash -n scripts/make_phase2_skip_warmstarts.sh \
        scripts/run_phase2_skip_sweep.sh \
        scripts/run_phase2_full_eval.sh \
        scripts/run_phase2_timit.sh
```

Passed.

### Shape Build

For g42+s075:

```text
skip index shapes: [(96,), (96,), (48,), (24,), (12,), (6,)]
```

### P2 MAC Counter

Command:

```bash
conda run --no-capture-output -n bcctn python scripts/count_p2_macs.py
```

Output:

| variant | ratio | dec_pw MAC | delta dec_pw | full_conv MAC | delta full_conv |
|---|---:|---:|---:|---:|---:|
| g42_s100 | 1.00 | 14,264,320 | +0.00% | 37,125,440 | +0.00% |
| g42_s075 | 0.75 | 11,895,424 | -16.61% | 34,756,544 | -6.38% |
| g42_s050 | 0.50 | 9,526,528 | -33.21% | 32,387,648 | -12.76% |

These P2 deltas are additional to the g42 P1 reference, not relative to original baseline.

### Checkpoint Surgery

Generated first_n s075 warm-start:

```bash
bash scripts/make_phase2_skip_warmstarts.sh s075_first
```

Strict load passed:

```text
strict load ok <All keys matched successfully>
```

### Topk Smoke

One-batch topk index precompute passed on CPU.

Topk smoke checkpoint conversion passed.

Strict load passed for topk smoke checkpoint.

Random forward passed:

```text
output shape = (1, 2, 15968)
finite = True
```

---

## 10. Recommended P2 Execution Order

Primary P1 checkpoint:

```bash
outputs/p1_grouped/g42_2026-05-17_04-01-16/logs/lightning_logs/version_0/checkpoints/epoch=27-step=27104.ckpt
```

The scripts default to this checkpoint.

### Step 1: Create P2 warm-starts

```bash
cd /raid/R12K41024/12_decoder_grouped_pointwise

GPU_ID=0 bash scripts/make_phase2_skip_warmstarts.sh
```

Default variants:

```text
s075_first
s075_topk
s050_first
s050_topk
```

Expected outputs:

```bash
Checkpoints/p2_skip_indices_g42_first_n_r075.pt
Checkpoints/p2_skip_indices_g42_topk_l1_r075.pt
Checkpoints/p2_skip_indices_g42_first_n_r050.pt
Checkpoints/p2_skip_indices_g42_topk_l1_r050.pt

Checkpoints/p2_warmstart_g42_first_n_r075.ckpt
Checkpoints/p2_warmstart_g42_topk_l1_r075.ckpt
Checkpoints/p2_warmstart_g42_first_n_r050.ckpt
Checkpoints/p2_warmstart_g42_topk_l1_r050.ckpt
```

### Step 2: Train + 20-pair sanity

```bash
GPU_ID=0 bash scripts/run_phase2_skip_sweep.sh \
  s075_first s075_topk s050_first s050_topk
```

Outputs:

```bash
outputs/p2_skip/
results/p2_skip_sanity/
logs/p2_skip_*.log
```

### Step 3: MAC Counter

```bash
conda run --no-capture-output -n bcctn python scripts/count_p2_macs.py
```

### Step 4: Full VCTK eval

After sanity chooses candidates:

```bash
GPU_ID=0 bash scripts/run_phase2_full_eval.sh s075_topk s050_topk
```

Run first_n full eval only if sanity suggests it is unexpectedly competitive.

### Step 5: TIMIT

Run TIMIT only on the best 1-2 P2 candidates:

```bash
GPU_ID=0 bash scripts/run_phase2_timit.sh s075_topk
```

---

## 11. P2 Decision Criteria

Reference is P1 primary:

```text
g42, groups_x=4, groups_skip=2, skip_select_ratio=1.0
```

P2 candidate should satisfy:

- VCTK MBSTOI delta within about `-0.005` of g42/P1 reference
- ILD/IPD not worse by more than about 5%
- TIMIT best candidate does not collapse
- Choose the candidate with largest MAC/bandwidth reduction among those passing quality gates

Expected risk:

- `s075_topk` is the most likely useful P2 candidate.
- `s050_topk` may be too aggressive but worth sanity.
- `first_n` is a sanity/control mode, not expected to win.

---

## 12. Suggested Result File

When P2 finishes, create:

```bash
RESULTS_P2_SKIP_2026-05-XX.md
```

Recommended sections:

1. TL;DR
2. P2 setup
3. P2 variants
4. 20-pair sanity table
5. Full VCTK table
6. MAC table
7. TIMIT table for best 1-2
8. Decision and RTL candidate
9. Repro commands

Important wording:

- Use `decoder pointwise MAC`
- Use `conv-only total MAC` instead of claiming true full-model MAC
- Explicitly say P2 skip subset reduces skip input channels and skip activation bandwidth, unlike P1 where tensor dimensions were unchanged externally.

---

## 13. Future Work Candidates

P3 candidates after P2:

- Encoder pointwise grouping sweep:
  - `encoder_pw_groups=2`, middle/late layers first
  - avoid first layer
  - monitor ILD/IPD carefully
- Width multiplier / kernel_num top-stage sweep:
  - e.g. `[16,32,64,128,256,128]` or `[16,32,64,128,192,192]`
- PReLU slope pow2 / ReLU variant
- BN affine pow2 freeze
- Full-model MAC accounting including MHA/attention/STFT

Not recommended early:

- Complex conv real/imag tying in convs due phase/IPD risk
- ConvTranspose to upsample + separable conv without full ablation
- Channel shuffle before grouped warm-start is understood, because it breaks the current block-diagonal surgery story

