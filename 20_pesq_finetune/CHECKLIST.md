# Handoff Checklist

Use this checklist to inspect the **19 algo_final** package before sending it to
the hardware consultant. This package is **patterns + docs**; the checkpoint and
the dump/export toolchain are not shipped here (see "Not Included" below).

## Package Location

- [ ] Package root exists:
      `/raid/R12K41024/19_algo_final`
- [ ] Source package remains untouched (holds the deploy ckpt + dump tooling):
      `/raid/R12K41024/18_w4a8_pow2_mixed_lmul_hw_handoff_20260530`

## Patterns Present (`hw_patterns_final/`)

- [ ] `hw_patterns_final/README.md` and `tb_pattern/README.md` describe the layout.
- [ ] `hw_patterns_final/summary.json` exists (records `source_ckpt`, golden wav).
- [ ] `tb_pattern/manifest.json` and `tb_pattern/layer_config.json` exist (the index).
- [ ] `tb_pattern/weights/` has the per-layer weight dirs (63 layer dirs) with
      packed INT4/FP8 codes, APoT/pow2 scale codes, and the fused descale.
- [ ] `tb_pattern/blocks/` (BN APoT scale codes, fused BN bias, PReLU pow2 codes).
- [ ] `tb_pattern/activations_fp8/` (per-layer input activations as raw e4m3 `uint8`
      code) and `activations_fp8/meta.json` (block-float exponent table).
- [ ] `tb_pattern/skip_indices/` (6 decoder skip stages, `decoder_stage_0..5`) and `tb_pattern/top_level_io/`.
- [ ] `sram_images/wbuf_unified_init.bin` exists (full model resident in WBUF).
- [ ] `sram_images/` has `memory_map.json`, `memory_map_pkg.svh`, `sram_config.vh`,
      the zero-init buffers (`kvbuf`/`ufbuf`/`skipbuf`), and `standalone_verify.log`.
- [ ] `debug_reference/layer_io_fp32/` holds per-module golden I/O as `.hex` text
      (+ `meta.json`); `debug_reference/export_reference_debug/` holds readable views.
      Do NOT use `debug_reference/` as the source of packed testbench weights.

## Docs Present (`docs/`)

- [ ] `HANDOFF_HW_ZERO_MULTIPLIER_CORE.md` (+ `.zh.md`) — main hardware handoff.
- [ ] `ARCHITECTURE_SUMMARY_STRICT18.md` — architecture summary.
- [ ] `SRAM_RTL_MEMORY_HANDOFF_EN.md` (+ `_ZH.md`) and
      `SRAM_MEMORY_MAP_SPEC_2026-06-03.md` — SRAM memory map / RTL handoff.
- [ ] `DYNAMIC_RANGE_QUANTIZATION_ANALYSIS.zh.md`,
      `RESULTS_ARCH_PROGRESSION.md` (+ `.zh.md`),
      `PROOF_descale_range_equivalence.md`,
      `HANDOFF_REVIEW_NOTES.md`, `HANDOFF_SOFT_LMUL_STE_2026-05-26.md`,
      `STATUS_LOW_POWER_2026-05-19.md` present.
- [ ] `docs/analysis_figures/` present (regenerable via
      `scripts/make_dynamic_range_figures.py`).

## Reference Source (`code/`)

- [ ] `code/DCNN/` and `code/config/` are present.
- [ ] `code/eval.py`, `code/train.py`, `code/calibrate_activation_fp8.py` present.
- [ ] `code/MBSTOI/` present (eval metric).

## Deploy Result Check (deploy = pow2 block-float activations + delayed APoT weight descale)

- [ ] VCTK average file exists:
      `results/apot_descale_eval/ft_pow2_ep3_full/vctk_paper_style_results_averages.txt`
- [ ] TIMIT average file exists:
      `results/apot_descale_eval/ft_pow2_ep3_full/timit_paper_style_results_averages.txt`
- [ ] VCTK Average MBSTOI Enhanced is `0.8357` (gate 0.835).
- [ ] TIMIT Average MBSTOI Enhanced is `0.7898` (gate 0.7880).
- [ ] Activations are block-float (`calibrated_pow2_e4m3`, floor rounding); the
      block-float exponent table is in `tb_pattern/activations_fp8/meta.json` and the
      `activation_blockfloat` section of `tb_pattern/manifest.json`.
- [ ] FP8-outlier descale APoT range = `[-16,-6]`; the activation pow2 exponent `−k`
      adds onto these (delayed APoT shift-add). See HANDOFF §3.4 and
      `docs/PROOF_descale_range_equivalence.md`.

## Not Included (lives in the source package)

- [ ] Deploy checkpoint is NOT in this package; it lives at
      `18_w4a8_pow2_mixed_lmul_hw_handoff_20260530/ckpt/strict_core_pow2_blockfloat_deploy_ep3_vctk0.8357_timit0.7898.ckpt`
      (md5 `673de90a44dc13a4770cafc5a223d161`), recorded as `source_ckpt` in the manifests.
- [ ] Dump/export tooling (`software_to_hardware_18/`) and the full raw export
      (`export_out_strict18/`) live in the source package, not here.
- [ ] To regenerate: run the source package's
      `software_to_hardware_18/run_output.sh` against the deploy checkpoint.

## Known Notes

- [ ] The neural-core zero-multiplier claim excludes STFT, iSTFT, window multiply,
      and waveform-domain mask application.
- [ ] QK `1/sqrt(head_dim)` is a runtime pow2 shift in the export metadata, not
      pre-fused into Q rows.
- [ ] MHA QKV scale is computed before applying the `qkv_groups=4` block-diagonal
      mask, matching the source code.
- [ ] MHA out_proj is the W4A16 exception; QK and score×V are activation×activation
      (no weight), kept with separate exponents (not fused).
- [ ] PReLU in source is scalar per real/imag module; exported metadata records the
      actual parameter count.
