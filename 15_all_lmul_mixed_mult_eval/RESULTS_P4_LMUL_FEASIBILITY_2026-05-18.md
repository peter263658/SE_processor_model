# RESULTS: P4 L-Mul Feasibility Scaffold

Date: 2026-05-18
Status: simulator + smoke validation complete; official feasibility runs pending
Base candidate: P2 `g42 + s050_topk`

The P3 future-work file originally deferred L-Mul to P4. Per the latest user
request, this repo now includes an initial L-Mul feasibility path focused on
the attention datapath.

## Scope

Implemented first:

- `QK^T` attention score generation.
- `Score x V` attention value mixing.

Not implemented yet:

- MHA `Q/K/V/O` Linear L-Mul.
- Decoder Conv/ConvTranspose L-Mul.
- Structured Q/K/V/O Linear pruning.

This separation is intentional. Replacing attention arithmetic first gives a
clean answer about the most cue-sensitive path before adding projection
topology changes.

## Implementation

The simulator lives in:

```text
DCNN/utils/complexPyTorch/complexLayers.py
```

New helpers:

- `_lmul_multiply(...)`
- `_lmul_matmul(...)`
- `_lmul_target_enabled(...)`

Config:

```yaml
model:
  attention_datapath:
    enabled: true
    lmul_enabled: true
    lmul_targets: [qk]          # or [score_v], [qk, score_v]
    lmul_mantissa_bits: 4
```

The simulator uses sign/exponent handling plus mantissa addition with the
L-Mul offset. Accumulation remains normal torch summation. This is a
feasibility simulator, not bit-exact RTL.

## Script

```text
scripts/run_p4_lmul_feasibility.sh
```

Variants:

```text
lmul_qk_m4
lmul_scorev_m4
lmul_attn_m4
lmul_attn_m3
```

Usage:

```bash
# No-training base sanity
TRAIN=0 LIMIT_PAIRS=20 bash scripts/run_p4_lmul_feasibility.sh lmul_qk_m4

# Short QAT recovery screening
EPOCHS=10 LR=2e-4 bash scripts/run_p4_lmul_feasibility.sh lmul_qk_m4
```

## Smoke Validation

Executed:

```bash
bash -n scripts/run_p4_lmul_feasibility.sh
```

Tensor-level smoke:

- `_lmul_multiply(...)`
- `_lmul_matmul(...)`
- `_mha_attention_forward(...)` with targets:
  - `[qk]`
  - `[score_v]`
  - `[qk, score_v]`

Model-init smoke confirmed Hydra passes:

```text
lmul_enabled=True
lmul_targets=['qk', 'score_v']
lmul_mantissa_bits=4
```

One-pair VCTK smoke, no retraining:

```bash
TRAIN=0 LIMIT_PAIRS=1 METRIC_WORKERS=1 BATCH_SIZE=1 \
  bash scripts/run_p4_lmul_feasibility.sh lmul_qk_m4
```

Output:

```text
results/p4_lmul_sanity/lmul_qk_m4_base_w1/
```

Smoke average:

```text
Average MBSTOI Enhanced: 0.8629
Average ILD Error Enhanced: 4.1879 dB
Average IPD Error Enhanced: 14.69 deg
```

This is not an official feasibility result because it is only one pair and the
absolute MBSTOI is not comparable to the P2 20-pair or 750-pair tables. It
does prove that `QK^T` L-Mul M4 runs end-to-end without numerical collapse in
the current P2 stack.

## Feasibility Read

Current evidence:

- Software plumbing is feasible.
- `QK^T` L-Mul M4 runs through paper-style eval.
- The first 1-pair smoke did not show an obvious attention crash.
- CPU runtime is slower because the simulator expands matmul into explicit
  product tensors; this does not reflect hardware cost.

What is still needed before saying L-Mul is quality-feasible:

1. `TRAIN=0 LIMIT_PAIRS=20` for `lmul_qk_m4`, `lmul_scorev_m4`, and
   `lmul_attn_m4`.
2. If any variant is within the 20-pair floor, run 5-10 epoch QAT recovery.
3. Promote only the best passing variant to full VCTK and TIMIT.

Initial recommendation:

- Test `lmul_qk_m4` and `lmul_scorev_m4` separately first.
- Do not start with `lmul_attn_m3`; use it only after M4 is stable.
- Do not combine L-Mul with score/prob quantization until isolated P3-B and
  isolated L-Mul both pass.

