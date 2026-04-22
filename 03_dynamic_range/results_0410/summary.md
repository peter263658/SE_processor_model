# Weight Dynamic Range Analysis Summary

## Q1: Are weight distributions similar across channels/heads?

- **MHA layers**: 0 head groups are uniform, 12 show significant cross-head variance.
  - High-variance groups: model.mattn.input_proj.parametrizations.weight.original.real, model.mattn.input_proj.parametrizations.weight.original.imag, model.mattn.mattn.MultiheadAttention_re.out_proj.parametrizations.weight.original, model.mattn.mattn.MultiheadAttention_re.parametrizations.in_proj_weight.original.Q, model.mattn.mattn.MultiheadAttention_re.parametrizations.in_proj_weight.original.K

## Q2: Optimal group_size for W4 scalar factor

- Per-tensor baseline MSE: 1.355679e-03
- group_size=16: avg MSE = 9.153611e-04 (0.68x vs per-tensor)
- group_size=32: avg MSE = 8.285156e-04 (0.61x vs per-tensor)
- group_size=64: avg MSE = 7.113576e-04 (0.52x vs per-tensor)
- group_size=128: avg MSE = 5.771587e-04 (0.43x vs per-tensor)
- group_size=256: avg MSE = 3.802111e-04 (0.28x vs per-tensor)

**Recommendation**: group_size = **256** (largest size within 2x MSE of finest granularity)

## Q3: Layers with outliers requiring FP format

- 5 of 100 weight tensors are recommended for **FP** (kurtosis > 5).
- FP-recommended layers:
  - `model.decoder.model.0.0.pointwise_r.parametrizations.weight.original` (kurtosis = 6.38)
  - `model.decoder.model.0.0.pointwise_i.parametrizations.weight.original` (kurtosis = 7.33)
  - `model.decoder.model.3.0.pointwise_r.parametrizations.weight.original` (kurtosis = 8.46)
  - `model.decoder.model.3.0.pointwise_i.parametrizations.weight.original` (kurtosis = 12.32)
  - `model.decoder.model.4.0.pointwise_i.parametrizations.weight.original` (kurtosis = 9.36)
- 95 weight tensors are recommended for **INT4**.
  - 62 can use per-tensor scale, 33 need per-channel/per-group scaling.

## Q4: Real vs Imaginary weight distributions

- 2 complex layers have **similar** real/imag distributions (std ratio within 20%, kurtosis diff < 2).
- 0 complex layers show **significant differences** between real and imaginary components.
- **Conclusion**: Real and imaginary components can share the same scale factor.

## Q5: MHA per-head weight magnitude uniformity

- `model.mattn.input_proj.parametrizations.weight.original.real`: var_max = 1.365139e-03 → **non-uniform**
- `model.mattn.input_proj.parametrizations.weight.original.imag`: var_max = 1.187718e-03 → **non-uniform**
- `model.mattn.mattn.MultiheadAttention_re.out_proj.parametrizations.weight.original`: var_max = 3.238991e-03 → **non-uniform**
- `model.mattn.mattn.MultiheadAttention_re.parametrizations.in_proj_weight.original.Q`: var_max = 1.760497e-03 → **non-uniform**
- `model.mattn.mattn.MultiheadAttention_re.parametrizations.in_proj_weight.original.K`: var_max = 1.731760e-03 → **non-uniform**
- `model.mattn.mattn.MultiheadAttention_re.parametrizations.in_proj_weight.original.V`: var_max = 8.049189e-04 → **non-uniform**
- `model.mattn.mattn.MultiheadAttention_im.out_proj.parametrizations.weight.original`: var_max = 3.146075e-03 → **non-uniform**
- `model.mattn.mattn.MultiheadAttention_im.parametrizations.in_proj_weight.original.Q`: var_max = 7.681424e-04 → **non-uniform**
- `model.mattn.mattn.MultiheadAttention_im.parametrizations.in_proj_weight.original.K`: var_max = 2.127119e-03 → **non-uniform**
- `model.mattn.mattn.MultiheadAttention_im.parametrizations.in_proj_weight.original.V`: var_max = 5.132798e-03 → **non-uniform**
- `model.mattn.output_proj.parametrizations.weight.original.real`: var_max = 3.271637e-03 → **non-uniform**
- `model.mattn.output_proj.parametrizations.weight.original.imag`: var_max = 4.258458e-03 → **non-uniform**

**Conclusion**: Some heads have significantly different weight magnitudes — per-head scaling is recommended.

## Group Size Sensitivity

Refer to `group_size_plot.png` for average MSE across different group sizes. Smaller group sizes improve precision but have higher hardware overhead.
