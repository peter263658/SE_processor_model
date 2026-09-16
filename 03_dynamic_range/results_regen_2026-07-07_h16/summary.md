# Weight Dynamic Range Analysis Summary

## Q1: Are weight distributions similar across channels/heads?

- **Conv layers**: 42 conv weight tensors analyzed per-channel.
  - 42 tensors show **high cross-channel variance** (var_max > 1e-4):
    - `model.encoder.model.0.0.pointwise_i.parametrizations.weight.original` (var_max = 5.26e-01)
    - `model.encoder.model.0.0.pointwise_r.parametrizations.weight.original` (var_max = 4.94e-01)
    - `model.encoder.model.2.0.depthwise_i.parametrizations.weight.original` (var_max = 6.74e-02)
    - `model.encoder.model.1.0.depthwise_r.parametrizations.weight.original` (var_max = 5.50e-02)
    - `model.decoder.model.4.0.depthwise_r.parametrizations.weight.original` (var_max = 5.38e-02)
    - `model.encoder.model.1.0.depthwise_i.parametrizations.weight.original` (var_max = 4.79e-02)
    - `model.decoder.model.4.0.depthwise_i.parametrizations.weight.original` (var_max = 4.59e-02)
    - `model.decoder.model.3.0.depthwise_i.parametrizations.weight.original` (var_max = 3.82e-02)
    - `model.encoder.model.4.0.depthwise_i.parametrizations.weight.original` (var_max = 3.55e-02)
    - `model.encoder.model.3.0.depthwise_r.parametrizations.weight.original` (var_max = 3.52e-02)

- **MHA layers**: 12 head-group tensors analyzed.
  - 4 groups are **uniform** (var_max ≤ 1e-3), 8 show **significant cross-head variance**.
  - High-variance head groups:
    - `model.mattn.output_proj.parametrizations.weight.original.imag` (var_max = 5.60e-03)
    - `model.mattn.mattn.MultiheadAttention_im.parametrizations.in_proj_weight.original.V` (var_max = 4.10e-03)
    - `model.mattn.mattn.MultiheadAttention_im.out_proj.parametrizations.weight.original` (var_max = 2.58e-03)
    - `model.mattn.mattn.MultiheadAttention_im.parametrizations.in_proj_weight.original.K` (var_max = 1.67e-03)
    - `model.mattn.output_proj.parametrizations.weight.original.real` (var_max = 1.63e-03)
    - `model.mattn.mattn.MultiheadAttention_im.parametrizations.in_proj_weight.original.Q` (var_max = 1.36e-03)
    - `model.mattn.input_proj.parametrizations.weight.original.imag` (var_max = 1.24e-03)
    - `model.mattn.mattn.MultiheadAttention_re.parametrizations.in_proj_weight.original.K` (var_max = 1.20e-03)

## Q2: Optimal group_size for W4 scalar factor

- Per-tensor baseline MSE: 1.314210e-03
- group_size=16: avg MSE = 9.572173e-04 (0.73x vs per-tensor)
- group_size=32: avg MSE = 1.047957e-03 (0.80x vs per-tensor)
- group_size=64: avg MSE = 1.112756e-03 (0.85x vs per-tensor)
- group_size=128: avg MSE = 1.169661e-03 (0.89x vs per-tensor)
- group_size=256: avg MSE = 1.202230e-03 (0.91x vs per-tensor)

**Recommendation**: group_size = **256** (largest size within 2× MSE of finest granularity gsz=16)

> Note: smaller group_size = more scale factors = lower MSE (better precision) but higher hardware overhead (~6% at gsz=32).

## Q3: Layers with outliers requiring FP format

- 5 of 104 weight tensors are recommended for **FP** (kurtosis > 5).
- FP-recommended layers:
  - `model.decoder.model.0.0.pointwise_r.parametrizations.weight.original` (kurtosis = 6.38)
  - `model.decoder.model.0.0.pointwise_i.parametrizations.weight.original` (kurtosis = 7.33)
  - `model.decoder.model.3.0.pointwise_r.parametrizations.weight.original` (kurtosis = 8.46)
  - `model.decoder.model.3.0.pointwise_i.parametrizations.weight.original` (kurtosis = 12.32)
  - `model.decoder.model.4.0.pointwise_i.parametrizations.weight.original` (kurtosis = 9.36)
- 99 weight tensors are recommended for **INT4**.
  - 28 can use per-tensor scale, 71 need per-channel/per-group scaling.

## Q4: Real vs Imaginary weight distributions

- 2 complex layers have **similar** real/imag distributions (std ratio within 20%, kurtosis diff < 2).
- 0 complex layers show **significant differences** between real and imaginary components.
- **Conclusion**: Real and imaginary components can share the same scale factor.

## Q5: MHA per-head weight magnitude uniformity

- `model.mattn.input_proj.parametrizations.weight.original.real`: var_max = 9.687122e-04 → uniform
- `model.mattn.input_proj.parametrizations.weight.original.imag`: var_max = 1.237443e-03 → **non-uniform**
- `model.mattn.mattn.MultiheadAttention_re.out_proj.parametrizations.weight.original`: var_max = 5.962544e-04 → uniform
- `model.mattn.mattn.MultiheadAttention_re.parametrizations.in_proj_weight.original.Q`: var_max = 8.275706e-04 → uniform
- `model.mattn.mattn.MultiheadAttention_re.parametrizations.in_proj_weight.original.K`: var_max = 1.199310e-03 → **non-uniform**
- `model.mattn.mattn.MultiheadAttention_re.parametrizations.in_proj_weight.original.V`: var_max = 4.528695e-04 → uniform
- `model.mattn.mattn.MultiheadAttention_im.out_proj.parametrizations.weight.original`: var_max = 2.581040e-03 → **non-uniform**
- `model.mattn.mattn.MultiheadAttention_im.parametrizations.in_proj_weight.original.Q`: var_max = 1.355122e-03 → **non-uniform**
- `model.mattn.mattn.MultiheadAttention_im.parametrizations.in_proj_weight.original.K`: var_max = 1.669941e-03 → **non-uniform**
- `model.mattn.mattn.MultiheadAttention_im.parametrizations.in_proj_weight.original.V`: var_max = 4.098261e-03 → **non-uniform**
- `model.mattn.output_proj.parametrizations.weight.original.real`: var_max = 1.629010e-03 → **non-uniform**
- `model.mattn.output_proj.parametrizations.weight.original.imag`: var_max = 5.598767e-03 → **non-uniform**

**Conclusion**: Some heads have significantly different weight magnitudes — **per-head scaling is recommended** for these MHA tensors.

## Group Size Sensitivity

Refer to `group_size_plot.png` for average MSE across different group sizes.
Smaller group sizes improve precision (lower MSE) at the cost of more scale factors.
