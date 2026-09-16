# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Implementation of **Binaural Speech Enhancement using Deep Complex Convolutional Transformer Networks (BCCTN)** (ICASSP 2024). The model processes stereo audio to suppress noise while preserving spatial cues (ILD and IPD).

This repo has since grown into a **hardware-aware co-design project**: the goal is a custom ASIC inference core for BCCTN. The work has moved from FP32 research → aggressive quantization (W4A8 + power-of-2 block-float) → a **strict zero-multiplier neural core** (no general multiplier in the Conv/MHA/BN/PReLU/softmax datapath) → a SystemVerilog RTL implementation with bit-exact test patterns. As of mid-2026 the algorithm/hardware handoff is essentially done and the active work is RTL bring-up and PESQ fine-tuning for a chip paper.

## Repository Structure

There are two families of directories. **The numbered `NN_*` directories are the real, current history**; the `BCCTN_*` directories are early/legacy variants.

### Numbered experiment progression (current — read these first)

Each `NN_<name>/` is a self-contained experiment building on the previous one. Roughly grouped:

| Range | Theme | Notable dirs |
|---|---|---|
| `01`–`06` | Hardware-aware **analysis** (no retraining): MAC utilization, weight bit-width, dynamic range, sparsity, conv reuse, attention precision | `01_utilization_analysis`, `03_dynamic_range`, `04_sparsity_analysis` |
| `07`–`11` | Early **QAT + MHA** variants: FP8 PTQ/QAT, mixed-precision, weight-tied MHA, fixed-point STFT | `08_all_fp8_ptq`, `09_weight_tied_mha`, `10_stft_fixpoint_ptq` |
| `12`–`14` | **Architecture** refinements: grouped/sparse decoder pointwise, aggressive QAT, low-power arch | `12_decoder_grouped_pointwise`, `14_low_power_arch_qat` |
| `15`–`17` | **L-Mul + quality recovery**: mixed multipliers, knowledge distillation, soft-L-Mul + STE | `15_all_lmul_mixed_mult_eval`, `16_kd_heatmap_recovery`, `17_soft_lmul_ste_recovery` |
| `18`–`20` | **Finalization + handoff**: W4A8 + pow2 block-float deploy, hardware handoff packages, RTL, PESQ finetune | see below |

**The most recent / active directories:**
- `18_w4a8_pow2_mixed_lmul/` — latest working model (W4A8 + pow2 mixed multiplier).
- `18_w4a8_pow2_mixed_lmul_hw_handoff_20260530/` — **source handoff package**: deploy checkpoint, full dump/export toolchain (`software_to_hardware_18/`), verification scripts. Regenerates everything in 19.
- `19_algo_final/` — **frozen final handoff** (patterns + docs, no checkpoint/toolchain) **plus the SystemVerilog RTL (`01_RTL/`) and testbenches (`00_TB/`)**.
- `20_pesq_finetune/` — current focus: PESQ fine-tuning + CE RTL implementation notes.

### Legacy variant directories (early, FP32/FP16)

`BCCTN/` (baseline), `BCCTN_sep_conv/`, `BCCTN_sep_conv_causal/`, `BCCTN_sep_conv_causal_slide_win/` (separable + causal + sliding-window — the architecture all later quant work is based on), `BCCTN_sep_conv_causal_slide_win_quant_256dim/`, `BCCTN_hardware/`, `main_model/`. Shared `Dataset/` lives at the repo root (and under `BCCTN/Dataset/`).

### Internal structure of a numbered experiment

Consistent across dirs `09`+:
- `code/DCNN/models/model.py` — encoder-decoder architecture
- `code/DCNN/models/binaural_attention_model.py` — `ComplexMHA` (in_proj → QK → softmax/top-4 → Score×V → out_proj)
- `code/DCNN/loss.py` — combined SNR + STOI + ILD + IPD loss
- `code/DCNN/trainer.py` — PyTorch Lightning `LightningModule`
- `code/DCNN/feature_extractors.py` — STFT/iSTFT wrappers
- `code/DCNN/utils/all_lmul_multiply.py` and `code/DCNN/utils/complexPyTorch/complexLayers.py` — L-Mul / fake-quant primitives
- `code/config/config.yaml` — master Hydra config (`model.yaml`/`training.yaml` defer to it)
- `code/eval.py`, `code/train.py`, `code/calibrate_activation_fp8.py` — entry points
- `code/MBSTOI/` — MBSTOI metric implementation
- `results/`, `outputs/`, `logs/`, `docs/`, `README.md`, `CHECKLIST.md`

## Environment Setup

Use `mamba` instead of `conda` for significantly faster dependency solving.

```bash
# Main environment (PyTorch 2.3.1, CUDA 12.1, Python 3.10)
mamba env create -f bcctn.yml && mamba activate bcctn

# Quantization-specific environment (PyTorch 2.2.0, CUDA 12.4, Python 3.9)
mamba env create -f bcctn_quant.yml && mamba activate bcctn_quant
```

## Common Commands

Run from inside the experiment directory's `code/` (e.g. `cd 19_algo_final/code`).

**Training / evaluation:**
```bash
python train.py                          # Hydra config/
HYDRA_FULL_ERROR=1 python train.py       # verbose errors
CUDA_VISIBLE_DEVICES=0 python eval.py --model_checkpoint <ckpt> \
    --vctk_test_dir ... --vctk_clean_dir ... --timit_test_dir ... --timit_clean_dir ... \
    --output_dir ./results --paper_style_eval --batch_size 8
bash run_eval.sh                         # wrapper with paths baked in
bash run_strict_core_eval.sh             # deploy-model (quantized) eval
```

**Quantization / hardware-specific:**
```bash
python calibrate_activation_fp8.py       # per-tensor pow2 shared-exponent calibration
bash run_strict_core_finetune.sh         # finetune the quantized deploy model
# Override the RTL-style attention / quant path via Hydra, e.g.:
python eval.py model.attention_softmax.mode=shift_topk_pow2
```

**RTL simulation (19_algo_final/00_TB/, SystemVerilog + VCS):**
```bash
cd 19_algo_final/00_TB/CE
./08_run_ce_core                         # numbered run_* scripts: 01_run_PEG .. 21_run_multiframe
# uses the compiled `simv_golden` and *_file.f Verilog file lists
```

## Key Hyperparameters (`code/config/config.yaml`)

Current canonical config (newest variants):

| Parameter | Current value | Notes |
|---|---|---|
| `model.kernel_num` | `[16,32,64,128,256,256]` | 6-stage encoder/decoder |
| `model.kernel_size` | 5 | Conv kernel (5×1 after kernel reduction) |
| `model.use_depthwise` | `true` | Depthwise separable convolution |
| `model.embed_dim` | **256** | Transformer embedding dim (was 512 in older notes) |
| `model.num_heads` | **16** | Attention heads (was 32 in older notes) |
| `model.tied_mha` | `true` | MHA_re ≡ MHA_im → halves MHA params (526K→263K) and softmax passes (4→2) |
| `model.attn_win` | 31 | Sliding attention window |
| `model.win_len` / `win_inc` / `fft_len` | 128 / 80 / 128 | STFT (256/512 also tested historically) |
| `model.rnn_layers` / `rnn_units` | 2 / 128 | CLSTM |
| `model.{snr,stoi,ild,ipd}` weights | 1 / 10 / 1 / 10 | `L = SNR + 10·STOI + ILD + 10·IPD` |
| `model.attention_softmax.mode` | `softmax` | RTL variants: `shift_topk_pow2[_floor/_ceil/_clamp8/_nomantissa]`, `taylor2`, `relu2` |
| `model.attention_datapath` / `prob_packet` | disabled by default | P3/P4 datapath + L-Mul + top-4 prob-packet simulation knobs |

## Architecture Notes

- **Encoder–decoder**, 6 complex separable-conv stages each; `ComplexConv2d` from `complexpytorch`. Shared (reused) weights for L/R channels.
- **Bottleneck**: weight-tied complex multi-head self-attention (`ComplexMHA`) over the freq-time sequence, with sliding/causal window.
- **Masks**: per-channel complex ratio masks (L and R).
- **Causal variants** use causal padding; sliding-window variants chunk the time axis for low-latency streaming.

## Quantization & Zero-Multiplier Hardware Core

The deploy model is **W4A8** with a **strict zero-multiplier core** — every multiply in Conv/MHA/BN/PReLU/softmax is realized as shift-add (CSD/APoT). Key vocabulary you will encounter in code, configs, and docs:

- **W4A8** — INT4 weights, e4m3 (FP8) block-float activations. Outlier layers (12 decoder pointwise) stay FP8 e4m3.
- **pow2 / APoT block-float** — activations are raw e4m3 `uint8` code + a per-tensor power-of-2 shared exponent (statically calibrated, no runtime amax). The `2^-k` descale is **fused** into the post-MAC weight descale (proven bit-exact: eager == fused).
- **L-Mul** — logarithmic multiply approximation (mantissa-bit add instead of full multiply), used for MHA QK / Score×V (m8) and FP8 decoder outliers (m4). See `all_lmul_multiply.py` / `complexLayers._lmul_*`.
- **CE** (Compute Engine) — the datapath spine: `PEG` (pe_pair) → dealer (fused APoT descale) → `OACC` (2-bank complex accumulator) → `PPU_fma`.
- **PPU** (Post-Processing Unit) — fixed post-MAC pipeline: FP16 saturating adder tree → dealer → OACC → BN → PReLU → cast to e4m3. Zero-multiplier (shifts/adds only).
- **Buffers**: `WBUF` (weights, 16 banks × 64-bit × 256 = 64KB), `KVBUF`, `UFBUF`, `SKIPBUF`.
- **Non-core DSP** (kept as ordinary multiply, outside the zero-mult core): STFT/iSTFT, window multiply, waveform mask application.

**Deploy gates (do not regress):** MBSTOI Enhanced **VCTK ≥ 0.835 (deploy 0.8357), TIMIT ≥ 0.788 (deploy 0.7898)**. PESQ on VoiceBank+DEMAND is the *current* optimization target (chip-paper comparison vs competitor PESQ 3.34 / STOI 95.1) — see `演算法端_品質對齊_SPEC.md`. Keep MBSTOI/ILD/IPD as the unique selling point; don't sacrifice them for PESQ.

## RTL & Hardware Patterns (19_algo_final)

- `19_algo_final/01_RTL/` — SystemVerilog 2017 (synthesizable). `Top/CHIP.sv`, `engine/CE/` (`CE_core.sv`, `PEG.sv`, `oacc.sv`, `ppu_fma.sv`), `engine/Softmax/`, `engine/FFT/`, buffer controllers (`wbuf/ kvbuf/ ufbuf/ skipbuf/`), `Define/` (params), `common/` (sram macros, FP8/FP16/FXP converters).
- `19_algo_final/00_TB/` — per-module testbenches + numbered `run_*` scripts + `simv_golden` (VCS). `06_run_PEG_w4a8` verified 0 errors over 16000 vectors; CE_core smoke passes; full bit-exact vs golden is the open item (see `CE_IMPL_NOTES.md`).
- `19_algo_final/hw_patterns_final/` — the hardware-facing payloads:
  - `tb_pattern/` — **source of truth** for RTL: packed INT4/FP8 weight codes, APoT/pow2 scale codes, fused descale, per-layer block-float activations, BN/PReLU constants, skip indices. Index: `manifest.json`, `layer_config.json`.
  - `sram_images/` — `wbuf_unified_init.bin` (full model resident in WBUF) + zero-init buffers + `memory_map.json` / `memory_map_pkg.svh` / `sram_config.vh` for RTL.
  - `debug_reference/`, `mha_module_io*/`, `mask_apply_golden/`, etc. — golden I/O `.hex` for **inspection only** (NOT the packed-weight source).

## Working Conventions (important)

- **Handoff-doc workflow.** Algorithm↔hardware correspondence is tracked as dated markdown in each dir's `docs/`, named `TYPE_TOPIC_YYYY-MM-DD.md`. Types: `HANDOFF` / `RTL_HANDOFF` (deliverables), `CONSULT` (questions to HW), `RESPONSE`/`REPLY` (answers), `REQUEST`, `RELAY` (status/blockers), `RESULTS`, `PROOF`, `STATUS`, `FUTURE_SPEC`. **Docs are append-only — never overwrite; a superseding doc links its predecessor.** Use the `/handoff-doc` skill to generate one in this template. Start any HW question from `docs/HANDOFF_HW_ZERO_MULTIPLIER_CORE.md` + `docs/ARCHITECTURE_SUMMARY_STRICT18.md`.
- **Protected artifacts.** A `PreToolUse` hook (`.claude/settings.json`) **blocks** Edit/Write to `*.ckpt`, `*.pt`, `*.pth`, and anything under `outputs/` or `Checkpoints/`. A `PostToolUse` hook runs `ast.parse` on every `*.py` you write and **fails the edit on a syntax error** — keep Python syntactically valid.
- **Numerical-equivalence verification.** When comparing two forward paths (grouped vs full pointwise, weight tying, L-Mul, FP8/W4A8 fake-quant, decoder split, softmax approximation), follow the repo's `verify_*.py` / `scripts/verify_fp16_ppu_chain.py` pattern and/or use the `numerical-equivalence-checker` agent. Report max abs diff / cosine similarity / per-layer comparison.
- **Don't regenerate from `19_algo_final`** — it has no checkpoint or toolchain; regeneration lives in `18_w4a8_pow2_mixed_lmul_hw_handoff_20260530/`.

## Known Bottlenecks (hardware target)

1. Complex 5×1 convolutions (>80% of FLOPs).
2. Attention softmax over S×S (S ≈ 157 for 1s audio at default hop) — addressed via top-4 prob-packet + pow2 softmax.
3. STFT/iSTFT FFT operations.
4. MHA weight DRAM→WBUF transfer (the original DMA bottleneck that motivated W4 + per-group scale).

## Reference Notes

- `current_progress.md` — in-flight experiment notes and the 6 hardware-aware analysis directions (T1–T4).
- `演算法端_品質對齊_SPEC.md` — quality-alignment spec for the chip paper (dataset settings, VoiceBank+DEMAND PESQ/STOI targets, per-SNR reporting).
- `README.md` — original paper-level project overview.
