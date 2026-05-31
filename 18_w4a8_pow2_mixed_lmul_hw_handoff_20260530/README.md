# 18 Strict-Core Hardware Handoff Package

Created from `/raid/R12K41024/18_w4a8_pow2_mixed_lmul` on 2026-05-30.

This folder is the standalone reference package for the current deploy
architecture: BCCTN strict neural inference core with no general multiplier in
the core datapath.

## Folder Map

| Path | Purpose |
|---|---|
| `code/` | Final source snapshot needed to build/evaluate the 18 strict-core model. |
| `docs/` | Hardware handoff and architecture/result notes. Start with `HANDOFF_HW_ZERO_MULTIPLIER_CORE.md`. |
| `docs/HANDOFF_REVIEW_NOTES.md` | Source-vs-handoff notes found during dump-script review. |
| `ckpt/strict_core_pow2_blockfloat_deploy_ep3_vctk0.8357_timit0.7898.ckpt` | **Deploy checkpoint** (pow2 block-float activations + delayed APoT weight descale; VCTK 0.8357 / TIMIT 0.7898; **no runtime activation scaling**). |
| `ckpt/activation_calibration_p999_b64.json` | Activation block-float calibration table (per-tensor pow2 shared exponents). |
| `ckpt/strict_core_apot_descale_deploy_ep2_vctk0.8365_timit0.7912.ckpt` | Prior deploy (delayed APoT weight descale, **dynamic** activations; 0.8365 / 0.7912). |
| `ckpt/strict_core_epoch4_step1210_val13.6105.ckpt` | Earlier deploy (exact descale, 0.8355 / 0.7902); kept for reference. |
| `ckpt/strict_core_config.yaml` | Exact training Hydra config used for this checkpoint. |
| `ckpt/strict_core_deploy_config.yaml` | Package-local inference/dump config, with training-only KD disabled. |
| `ckpt/p2_skip_indices_g42_topk_l1_r050.pt` | Static decoder skip top-50% index set copied into the package. |
| `09_software_to_hardware_original/` | Original 09 dump tools, kept for history/reference only. |
| `software_to_hardware_18/` | Reviewed and updated strict-18 dump tools. Use these for this checkpoint. |
| `export_out_strict18/` | Verified dump output generated from the copied checkpoint and code. |
| `results/apot_descale_eval/ft_pow2_ep3_full/` | Full VCTK+TIMIT eval (CSV/averages) for the pow2 block-float deploy ckpt (0.8357 / 0.7898). |
| `CHECKLIST.md` | User-facing inspection checklist. |

## Current Architecture

The deploy point is the 18 strict core:

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

## Dump Weights

Use the reviewed 18 scripts:

```bash
cd /raid/R12K41024/18_w4a8_pow2_mixed_lmul_hw_handoff_20260530/software_to_hardware_18
CONDA_ENV=bcctn ./run_output.sh ../export_out_strict18
```

Or run individual steps:

```bash
cd /raid/R12K41024/18_w4a8_pow2_mixed_lmul_hw_handoff_20260530
PYTHONPATH=$PWD/code NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/mpl_cache \
  conda run --no-capture-output -n bcctn \
  python software_to_hardware_18/ckpt2hwdat.py --outdir export_out_strict18 --overwrite
```

The original `09_software_to_hardware_original/` tools are intentionally kept,
but they are not correct for the current strict-core architecture. They are
hard-wired to the old 09 model and export fp16 multiplier-style scales/BN/PReLU.

## Verified Output

The package was checked after copy:

- `software_to_hardware_18/dump_hw_config.py`: model build + ckpt load OK.
- `software_to_hardware_18/ckpt2hwdat.py`: exported weights, scales, BN/PReLU,
  skip indices, and manifest with `missing=0`, `unexpected=0`.
- `software_to_hardware_18/golden_dump.py`: generated one CPU golden waveform
  input/output dump from the copied checkpoint.
- Export consistency: 12 block records, 63 manifest layer records, 12 FP8
  outlier weight dirs, 54 INT4 weight dirs, 6 skip-index stages.

Selected checkpoint full-eval metrics:

Deploy point (pow2 block-float activations + delayed APoT weight descale,
zero-multiplier core, no runtime activation scaling; pow2 recovery-finetuned):

- VCTK MBSTOI enhanced: `0.8357`  (gate 0.835)
- TIMIT MBSTOI enhanced: `0.7898`  (gate 0.7880)

(Prior dynamic-activation deploy was `0.8365 / 0.7912`; block-float trades −0.0008
VCTK for removing all runtime activation amax/scaling and a single fused descale.)

(Previous exact-descale strict core was `0.8355 / 0.7902`. The deploy descale is a
delayed 2-term APoT shift-add, range `[-16,-6]`; see HANDOFF §3.4.)
