# 19 algo_final — Hardware Handoff Package (BCCTN Strict Zero-Multiplier Core)

Standalone handoff package for the deployed BCCTN strict neural-inference core
(W4A8, **no general multiplier in the core datapath**). Assembled from the source
package `/raid/R12K41024/18_w4a8_pow2_mixed_lmul_hw_handoff_20260530` on 2026-06-04.

This package ships **patterns + docs**. The two things a hardware reader needs are
`hw_patterns_final/` (the patterns) and `docs/` (the handoff notes). Everything
else (`code/`, `results/`, `outputs/`, `logs/`) is reference. The training
checkpoint and the dump/export toolchain are **not** included — see
[Not Included](#not-included-lives-in-the-source-package).

## Start Here

1. `docs/HANDOFF_HW_ZERO_MULTIPLIER_CORE.md` — main hardware handoff: datapath,
   quant formats, descale realization (`.zh.md` is the Chinese version).
2. `docs/ARCHITECTURE_SUMMARY_STRICT18.md` — one-page architecture summary.
3. `docs/SRAM_RTL_MEMORY_HANDOFF_EN.md` + `docs/SRAM_MEMORY_MAP_SPEC_2026-06-03.md`
   — SRAM memory map / RTL memory handoff (current memmap = the 2026-06-03 spec).
4. `hw_patterns_final/README.md` and `hw_patterns_final/tb_pattern/README.md` —
   what each pattern folder is and how to consume it.

## Folder Map

| Path | Purpose |
|---|---|
| `hw_patterns_final/tb_pattern/` | **Testbench patterns** — packed INT4/FP8 weight codes, APoT/pow2 scale codes, the fused descale, per-layer input activations (block-float e4m3 raw `uint8` code), BN/PReLU block constants, skip indices, and top-level I/O. `manifest.json` + `layer_config.json` are the index (63 weight layer dirs, 6 skip stages). Start at its `README.md`. |
| `hw_patterns_final/sram_images/` | **SRAM init images** — `wbuf_unified_init.bin` (full model resident in WBUF), zero-init KV/UF/skip buffers, plus `memory_map.json`, `memory_map_pkg.svh`, `sram_config.vh` for RTL. Matches `docs/SRAM_*`. |
| `hw_patterns_final/debug_reference/` | **Inspection / numerical comparison only** (NOT the source of packed weights). `layer_io_fp32/` = per-module golden I/O as `.hex` text (incl. STFT/iSTFT Q1.15 DSP I/O and pre-quant outputs that are not in `tb_pattern`); `export_reference_debug/` = readable weight views. |
| `hw_patterns_final/summary.json` | Export summary: source ckpt, golden input wav, layer-call count, fused-exponent offsets. |
| `docs/` | Hardware handoff, architecture summary, SRAM memory map / RTL handoff, dynamic-range analysis, results progression, and equivalence proofs. |
| `code/` | Source snapshot (`DCNN/`, `config/`, `eval.py`, `train.py`, `calibrate_activation_fp8.py`, `MBSTOI/`, run scripts) — reference to understand or regenerate the patterns. Not required to consume them. |
| `results/apot_descale_eval/ft_pow2_ep3_full/` | Deploy-point full VCTK+TIMIT eval (CSV + averages): **VCTK 0.8357 / TIMIT 0.7898**. |
| `outputs/activation_calibration/` | Activation block-float calibration runs (per-tensor pow2 shared exponents); the `…p999_b64…正式` run is the deploy calibration. |
| `logs/` | Eval / finetune console logs (reference). |
| `scripts/make_dynamic_range_figures.py` | Regenerates `docs/analysis_figures/`. |
| `CHECKLIST.md` | Inspection checklist for this package. |

## Not Included (lives in the source package)

This package is patterns + docs, not the training/regeneration toolchain. The
following are intentionally **not** shipped here and live in
`/raid/R12K41024/18_w4a8_pow2_mixed_lmul_hw_handoff_20260530/`:

- **Deploy checkpoint** — `ckpt/strict_core_pow2_blockfloat_deploy_ep3_vctk0.8357_timit0.7898.ckpt`
  (md5 `673de90a44dc13a4770cafc5a223d161`). All patterns in `hw_patterns_final/`
  were generated from it; it is recorded as `source_ckpt` in the manifests.
- **Dump / export tooling** — `software_to_hardware_18/` (`ckpt2hwdat.py`,
  `dump_hw_config.py`, `golden_dump.py`, `build_sram_images.py`, `run_output.sh`).
- **Full raw export** — `export_out_strict18/`.

To regenerate the patterns from scratch, run that package's
`software_to_hardware_18/run_output.sh` against the deploy checkpoint above.

## Current Architecture

The deploy point is the strict core:

- W4A8 for Conv/Linear/MHA projection weights and activations.
- INT4 weight code `q in [-7, 7]`; activations are **block-float**: e4m3 raw code +
  per-tensor pow2 shared exponent (static calibrated, no runtime amax/scale). The
  activation descale fuses into the layer's post-acc weight descale as an exponent
  offset (shift-add) — eager==fused exactly, so the eval is the HW number.
- Conv pointwise scale: positive APoT `2^a +/- 2^b`, per 32 input channels.
- Conv depthwise scale: positive APoT per output channel.
- MHA projection scale: positive APoT per head.
- Decoder pointwise outliers: 12 FP8 e4m3 layers, runtime FP8xFP8 L-Mul m4.
- MHA QK and Score x V: L-Mul m8.
- Softmax: shift top-4 pow2 clamp8 no-mantissa packet path.
- BN: folded affine with effective `alpha = gamma / sqrt(var + eps)`, signed
  APoT `[-9, 4]`, conv bias fused into exported BN bias.
- PReLU: signed pow2 slope `[-8, 0]`.
- Decoder pointwise: split grouped branches, `groups_x=4`, `groups_skip=2`,
  static top-50% skip channels.
- MHA QKV projection: `qkv_groups=4`, block-diagonal 64x64 blocks.

The DSP front/back-end is not part of the zero-multiplier neural-core claim:
STFT, iSTFT, window multiply, and waveform-domain mask application remain
outside the core.

## Deploy-Point Metrics

Deploy point (pow2 block-float activations + delayed APoT weight descale,
zero-multiplier core, no runtime activation scaling; pow2 recovery-finetuned),
full eval in `results/apot_descale_eval/ft_pow2_ep3_full/`:

- VCTK MBSTOI enhanced: `0.8357`  (gate 0.835)
- TIMIT MBSTOI enhanced: `0.7898`  (gate 0.7880)

(Prior dynamic-activation deploy was `0.8365 / 0.7912`; block-float trades −0.0008
VCTK for removing all runtime activation amax/scaling and a single fused descale.
Previous exact-descale strict core was `0.8355 / 0.7902`. The deploy descale is a
delayed 2-term APoT shift-add, range `[-16,-6]`; see HANDOFF §3.4.)
