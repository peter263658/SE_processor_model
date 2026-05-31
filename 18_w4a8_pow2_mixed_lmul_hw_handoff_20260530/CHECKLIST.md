# Handoff Checklist

Use this checklist to inspect the package before sending it to the hardware
consultant.

## Package Location

- [ ] Package root exists:
      `/raid/R12K41024/18_w4a8_pow2_mixed_lmul_hw_handoff_20260530`
- [ ] Original source folder remains untouched:
      `/raid/R12K41024/18_w4a8_pow2_mixed_lmul`
- [ ] Original 09 dump folder remains untouched:
      `/raid/R12K41024/09_software_to_hardware`

## Required Content

- [ ] `README.md` is present and gives the folder map.
- [ ] `docs/HANDOFF_HW_ZERO_MULTIPLIER_CORE.md` is present.
- [ ] `docs/ARCHITECTURE_SUMMARY_STRICT18.md` is present.
- [ ] `code/DCNN/` is present.
- [ ] `code/config/` is present.
- [ ] `code/train.py` and `code/eval.py` are present.
- [ ] `09_software_to_hardware_original/` is present as historical reference.
- [ ] `software_to_hardware_18/` is present as the current dump tool.

## Checkpoint And Config

- [ ] Deploy checkpoint exists:
      `ckpt/strict_core_pow2_blockfloat_deploy_ep3_vctk0.8357_timit0.7898.ckpt`
      (pow2 block-float activations + delayed APoT weight descale; 0.8357 / 0.7898)
- [ ] Activation block-float calibration table exists:
      `ckpt/activation_calibration_p999_b64.json`
- [ ] Strict config exists:
      `ckpt/strict_core_config.yaml`
- [ ] Deploy config exists:
      `ckpt/strict_core_deploy_config.yaml`
- [ ] Training overrides exist:
      `ckpt/strict_core_overrides.yaml`
- [ ] Static skip-index file exists:
      `ckpt/p2_skip_indices_g42_topk_l1_r050.pt`
- [ ] Config contains strict-core settings:
      `int4_scale_format=apot_clip`,
      `attention_datapath.lmul_mantissa_bits=8`,
      `prob_packet.enabled=true`,
      `mha_projection.qkv_groups=4`,
      `decoder_grouped_pointwise.groups_x=4`,
      `groups_skip=2`,
      `bn_affine.scale_mode=fold_apot`,
      `prelu_pow2.enabled=true`.
- [ ] Deploy config has training-only KD disabled and points skip indices to the
      copied package path.

## Export Output Already Generated

- [ ] `export_out_strict18/manifest.json` exists.
- [ ] `export_out_strict18/layer_config.json` exists.
- [ ] `export_out_strict18/weights/` exists.
- [ ] `export_out_strict18/blocks/` exists.
- [ ] `export_out_strict18/skip_indices/` exists.
- [ ] `export_out_strict18/golden/meta.json` exists.
- [ ] Manifest load report is clean:
      `missing_count=0`, `unexpected_count=0`.
- [ ] Export consistency from this run:
      12 blocks, 63 manifest layer records, 12 FP8 outlier weight dirs,
      54 INT4 weight dirs, 6 skip-index stages.

## Dump Script Review

- [ ] Do not use `09_software_to_hardware_original/ckpt2hwdat.py` for strict 18.
      It is old 09-format reference only.
- [ ] Use `software_to_hardware_18/ckpt2hwdat.py` for strict 18 weights.
- [ ] `software_to_hardware_18/dump_hw_config.py` builds the copied strict model.
- [ ] `software_to_hardware_18/golden_dump.py` can produce waveform golden I/O.
- [ ] `software_to_hardware_18/run_output.sh` runs config dump, weight dump,
      and golden dump.
- [ ] Conv pointwise INT4 scale is grouped on input channels, group size 32.
- [ ] Conv depthwise INT4 scale is per output channel.
- [ ] MHA projection INT4 scale is per head.
- [ ] MHA QKV `qkv_groups=4` block-diagonal compact files are exported.
- [ ] BN folded scale exports APoT code for
      `gamma / sqrt(running_var + eps)`, not gamma alone.
- [ ] BN folded bias includes conv bias fusion.
- [ ] PReLU exports signed pow2 exponent/sign.
- [ ] Decoder pointwise outliers export PER-TENSOR SCALED FP8 (`code=fp8(w*448/amax)`)
      plus `fp8_scale`/`fp8_descale`/APoT descale, and are marked L-Mul m4.

## FP8 Outlier Scaled-FP8 Fix (2026-05-30)

- [ ] 12 FP8 outlier layers reconstruct (`code/scale`) to `module.weight` at maxabs = 0
      (`verify_export_equiv.py`).
- [ ] Each FP8 outlier `meta.json` has `fp8_scaled=true`, `fp8_scale`, `fp8_descale`,
      and `descale_apot_files`.
- [ ] `HANDOFF_HW_ZERO_MULTIPLIER_CORE.md` §3.4 documents scaled-FP8 + the
      zero-multiplier descale realization (pow2/APoT fold into L-Mul exponent).
- [ ] `verify_export_equiv.py` reports `modules checked: 111   diffs>=1e-6: 0`.
- [ ] `ckpt2hwdat.py` / `golden_dump.py` re-run clean (`missing=0 unexpected=0`).
- [ ] (optional) VCTK/TIMIT re-eval or fixed golden-waveform diff confirms quality.

## Re-run Export

From the package:

```bash
cd /raid/R12K41024/18_w4a8_pow2_mixed_lmul_hw_handoff_20260530/software_to_hardware_18
CONDA_ENV=bcctn ./run_output.sh ../export_out_strict18
```

Expected:

- [ ] `ckpt2hwdat.py` prints `load missing=0 unexpected=0`.
- [ ] `dump_hw_config.py` prints `blocks=12 layers=60`.
- [ ] `golden_dump.py` prints an input/output tensor shape.

## Result Check (deploy = pow2 block-float activations + delayed APoT weight descale)

- [ ] Deploy ckpt:
      `ckpt/strict_core_pow2_blockfloat_deploy_ep3_vctk0.8357_timit0.7898.ckpt`
- [ ] VCTK average file exists:
      `results/apot_descale_eval/ft_pow2_ep3_full/vctk_paper_style_results_averages.txt`
- [ ] TIMIT average file exists:
      `results/apot_descale_eval/ft_pow2_ep3_full/timit_paper_style_results_averages.txt`
- [ ] VCTK Average MBSTOI Enhanced is `0.8357` (gate 0.835).
- [ ] TIMIT Average MBSTOI Enhanced is `0.7898` (gate 0.7880).
- [ ] Activations are block-float (`BCCTN_ACTIVATION_FP8_MODE=calibrated_pow2_e4m3`,
      `BCCTN_ACTIVATION_POW2_ROUND=floor`); block-float exponents exported under
      `export_out_strict18/activation_blockfloat/` (53 boundaries).
- [ ] pow2 fusion is bit-exact: `verify_pow2_fused.py` worst relL2(fused,eager) = 0
      (eager==fused ⇒ eval == HW number).
- [ ] Re-dump used the deploy ckpt:
      `CKPT=<deploy ckpt> ./run_output.sh ../export_out_strict18` (or `export_helpers.DEFAULT_CKPT`).
- [ ] FP8-outlier descale APoT range = `[-16,-6]` (`descale_apot_exp_range` in each
      outlier `meta.json`); the activation pow2 exponent `−k` adds onto these at runtime.
- [ ] golden is the true HW path: `golden/meta.json` `outlier_descale.delayed_apot_descale=true`,
      `apot_exp_min/max = -16/-6`, `activation.fp8_mode = calibrated_pow2_e4m3`.
- [ ] Prior exact-descale point `strict_core_epoch4_step1210_val13.6105.ckpt`
      (0.8355 / 0.7902) kept for reference only.

## Known Notes

- [ ] The neural core zero-multiplier claim excludes STFT, iSTFT, window
      multiply, and waveform-domain mask application.
- [ ] QK `1/sqrt(head_dim)` is represented as a runtime pow2 shift in the 18
      export metadata, not pre-fused into Q rows as the old 09 script did.
- [ ] MHA QKV scale is computed before applying the `qkv_groups=4` mask, matching
      the current source code behavior.
- [ ] PReLU in current source is scalar per real/imag PReLU module, even though
      the handoff describes per-channel as the intended hardware format. The
      exported metadata records the actual parameter count.
