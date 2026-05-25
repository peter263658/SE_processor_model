# RESULTS: P3 Attention FP8 Datapath Scaffold

Date: 2026-05-18
Status: implementation + smoke validation complete; official quality runs pending
Base candidate: P2 `g42 + s050_topk`

This file records the code and script work needed to start P3. It is not a
quality sign-off yet. The only executed evaluations here are smoke tests.

## Implemented

1. Config and QAT wiring
   - Added explicit `training.fp16_accumulator: false`.
   - Added `training.int4_scale_format: fp32`.
   - Added `model.attention_datapath` with score/prob quantization defaults.
   - Passed `model.attention_datapath` through `DCNN/trainer.py` into
     `enable_mixed_precision_qat(...)`.

2. Attention datapath simulation
   - Added QK scale modes: `pow2`, `fp16`, `fp8_e4m3`.
   - Added finite-only score quantization:
     - `none`
     - `fp8_e4m3`
     - `fixed` with B0-derived `score_fixed_int_bits` /
       `score_fixed_frac_bits`.
   - Added probability quantization:
     - `q0.8`
     - `fp8_e4m3`
     - `pow2`
   - Added B0 score capture callback before score quantization and softmax.
   - Fixed a real P3-A1 dtype issue in `shift_topk` softmax: fp16 attention
     rows now keep scatter/probability dtype consistent.

3. INT4 scale representation
   - Added optional `training.int4_scale_format` support:
     - `fp32` reference
     - `fp8_e4m3`
     - `pow2`
   - Default remains `fp32` for P1/P2 equivalence.

4. Scripts
   - `scripts/dump_p3_attention_scores.py`
   - `scripts/simulate_p3_attention_quant.py`
   - `scripts/run_phase3_attention_fp8_sweep.sh`
   - `scripts/run_phase3_attention_full_eval.sh`
   - `scripts/run_phase3_attention_timit.sh`

## Smoke Validation

Executed:

```bash
python -m py_compile \
  DCNN/utils/complexPyTorch/complexLayers.py \
  DCNN/utils/mixed_precision_qat.py \
  DCNN/trainer.py \
  scripts/dump_p3_attention_scores.py \
  scripts/simulate_p3_attention_quant.py

bash -n scripts/run_phase3_attention_fp8_sweep.sh
bash -n scripts/run_phase3_attention_full_eval.sh
bash -n scripts/run_phase3_attention_timit.sh
```

Also executed tensor-level smoke tests for:

- FP16 custom attention with `shift_topk_pow2_clamp8_nomantissa`.
- Score FP8 quantization preserving literal `-inf`.
- Probability `q0.8` quantization.
- Model init with P2 grouped+skip config and `prob_quant=q0.8`.

B0 smoke:

```bash
conda run --no-capture-output -n bcctn python \
  scripts/dump_p3_attention_scores.py \
  --limit_utterances 1 \
  --max_rows_per_head 16 \
  --device cpu \
  --output_npz /tmp/p3_b0_smoke.npz \
  --summary_json /tmp/p3_b0_smoke.json

conda run --no-capture-output -n bcctn python \
  scripts/simulate_p3_attention_quant.py \
  --dump_npz /tmp/p3_b0_smoke.npz \
  --summary_json /tmp/p3_b0_smoke.json \
  --output_json /tmp/p3_b0_smoke_quant_sim.json
```

Smoke result:

- B0 dump completed and captured `mattn.mattn`.
- The 1-utterance / 16-row simulation is not a gate result.
- In that tiny sample, FP8 score quantization failed the top-k stability gate
  while fixed-Q passed. Treat this only as a warning that official B0 matters.

## Official P3 Run Order

Run from the repo root:

```bash
# A0 no-retrain reference sanity/full eval
LIMIT_PAIRS=20 bash scripts/run_phase3_attention_full_eval.sh a0_p2

# A1 5-10 epoch screening
EPOCHS=10 LR=2e-4 bash scripts/run_phase3_attention_fp8_sweep.sh a1_fp16_accum

# B0 official score dump and simulation
conda run --no-capture-output -n bcctn python scripts/dump_p3_attention_scores.py \
  --limit_utterances 8 \
  --max_rows_per_head 1200 \
  --output_npz results/p3_attention_fp8/b0_attention_scores.npz

conda run --no-capture-output -n bcctn python scripts/simulate_p3_attention_quant.py \
  --dump_npz results/p3_attention_fp8/b0_attention_scores.npz
```

Only after B0 passes should `b_score_fp8` be trained. If B0 fails FP8 score
stability, use `b_score_fixed` with the derived fixed-Q bits or keep FP16 score
registers.

