# Low-Power Spec TODO

Run from:

```bash
cd /raid/R12K41024/14_low_power_arch_qat
```

## 1. Smoke / Syntax

```bash
python -m py_compile \
  DCNN/utils/complexPyTorch/complexLayers.py \
  DCNN/models/model.py \
  DCNN/utils/mixed_precision_qat.py \
  scripts/analyze_low_power_arch.py

bash -n scripts/run_low_power_phase_a.sh
bash -n scripts/run_low_power_phase_bc_packet_lmul.sh
bash -n scripts/run_low_power_phase_d_mha_grouping.sh
bash -n scripts/run_low_power_phase_f_analysis.sh
```

## 2. Phase A: PReLU pow2 + BN affine

Sanity first:

```bash
TRAIN=0 LIMIT_PAIRS=20 bash scripts/run_low_power_phase_a.sh \
  a0_prelu_pow2 a2_bn_pow2 a3_bn_tie_pow2
```

Short QAT recovery:

```bash
LIMIT_PAIRS=20 EPOCHS=10 LR=2e-4 bash scripts/run_low_power_phase_a.sh \
  a1_prelu_pow2_qat a2_bn_pow2 a3_bn_tie_pow2
```

Gate: 20-pair MBSTOI >= 0.8031. Report whether BN is exact, approximate, or
requires architecture changes.

## 3. Phase B/C: q0.8 packet + bit-field L-Mul

Packet / datapath sanity:

```bash
TRAIN=0 LIMIT_PAIRS=20 bash scripts/run_low_power_phase_bc_packet_lmul.sh \
  b0_prob_q08_grid b1_packet_dequant b2_packet_fixed b3_packet_lmul \
  c0_lmul_qk_bitexact c1_lmul_scorev_bitexact c2_lmul_attn_bitexact \
  c3_packet_lmul_bitexact
```

Optional QAT if a packet variant is close:

```bash
TRAIN=1 LIMIT_PAIRS=20 EPOCHS=10 LR=2e-4 bash scripts/run_low_power_phase_bc_packet_lmul.sh \
  b2_packet_fixed b3_packet_lmul c3_packet_lmul_bitexact
```

Gate: B1 should track `b_prob_q08_grid`; B2/B3/C3 need 20-pair MBSTOI >= 0.8031.

## 4. Phase D: grouped MHA projection

No-train risk screen:

```bash
TRAIN=0 LIMIT_PAIRS=20 bash scripts/run_low_power_phase_d_mha_grouping.sh \
  d0_qkv_g2 d1_qkv_g4 d2_qkv_out_g4 d3_io_g4 d4_all_g4
```

10-epoch QAT:

```bash
LIMIT_PAIRS=20 EPOCHS=10 LR=2e-4 bash scripts/run_low_power_phase_d_mha_grouping.sh \
  d0_qkv_g2 d1_qkv_g4 d2_qkv_out_g4
```

If g=4 fails but g=2 passes:

```bash
LIMIT_PAIRS=20 EPOCHS=30 LR=2e-4 bash scripts/run_low_power_phase_d_mha_grouping.sh \
  d0_qkv_g2
```

Gate: VCTK MBSTOI >= 0.8302, TIMIT MBSTOI >= 0.7869, plus ILD/IPD limits from
the spec after full eval.

## 5. Phase E/F: encoding and operating point

```bash
bash scripts/run_low_power_phase_f_analysis.sh
```

Outputs:

```text
results/low_power_analysis/f0_dense_64_128_256.json
results/low_power_analysis/f3_qkv_g4_64_128_256.json
results/low_power_analysis/f3_all_attn_g4_64_128_256.json
```

Use these to pick 64 / 96 / 128 / 256 active lanes at 5 / 10 / 20MHz.

## 6. Full Eval Promotion

Only after 20-pair sanity passes, run full VCTK with the matching checkpoint and
the same Hydra overrides from the script log. Promote to TIMIT only if VCTK
clears the spec gate.
