# PROOF — FP8-outlier descale APoT range [-24,0] ≡ [-16,-6]

Auto-generated audit artifact (do not hand-edit). Regenerate with the
command at the bottom. Deploy ckpt = ep2 (`export_helpers.DEFAULT_CKPT`).

## Conclusion

For all 12 FP8-outlier descales of the deploy ckpt, the 2-term APoT value is
**bit-identical** between range `[-24,0]` and `[-16,-6]` (**max |Δ| = 0.000e+00**). The descale exponents live in [-14,-8], inside
both ranges, so neither clamps → identical values → identical model output.

Therefore the completed full eval run at `[-24,0]` (**VCTK 0.8365 / TIMIT 0.7912**, `results/apot_descale_eval/ft_ep2_full/`)
is, by construction, the score at the deployed `[-16,-6]` range. The package
default range is now `[-16,-6]` everywhere (runtime / eval / finetune / export).

**Direct corroboration:** a full VCTK+TIMIT eval run with `DD_MIN=-16 DD_MAX=-6`
explicitly (`BCCTN_OUTLIER_DELAYED_DESCALE=1`; log header confirms `exp range
[-16,-6]`) reproduces it exactly — **VCTK 0.8365 / TIMIT 0.7912**
(`results/apot_descale_eval/ft_ep2_dd16_6_proof/`).

## Per-layer equivalence (12 FP8 outlier layers)

| layer | descale | a_exp | b_exp | APoT[-24,0] | APoT[-16,-6] | \|Δ\| |
|---|---|---|---|---|---|---|
| decoder.model.0.0.pointwise_r_x | 1.4308e-03 | -9 | -11 | 1.464844e-03 | 1.464844e-03 | 0.0e+00 |
| decoder.model.0.0.pointwise_i_x | 1.3219e-03 | -10 | -11 | 1.464844e-03 | 1.464844e-03 | 0.0e+00 |
| decoder.model.0.0.pointwise_r_skip | 3.7222e-03 | -8 | -12 | 3.662109e-03 | 3.662109e-03 | 0.0e+00 |
| decoder.model.0.0.pointwise_i_skip | 5.2592e-03 | -8 | -10 | 4.882812e-03 | 4.882812e-03 | 0.0e+00 |
| decoder.model.3.0.pointwise_r_x | 2.1463e-03 | -9 | -12 | 2.197266e-03 | 2.197266e-03 | 0.0e+00 |
| decoder.model.3.0.pointwise_i_x | 2.1525e-03 | -9 | -12 | 2.197266e-03 | 2.197266e-03 | 0.0e+00 |
| decoder.model.3.0.pointwise_r_skip | 7.5716e-04 | -10 | -12 | 7.324219e-04 | 7.324219e-04 | 0.0e+00 |
| decoder.model.3.0.pointwise_i_skip | 7.7801e-04 | -10 | -12 | 7.324219e-04 | 7.324219e-04 | 0.0e+00 |
| decoder.model.4.0.pointwise_r_x | 2.7224e-03 | -9 | -10 | 2.929688e-03 | 2.929688e-03 | 0.0e+00 |
| decoder.model.4.0.pointwise_i_x | 1.8376e-03 | -9 | -13 | 1.831055e-03 | 1.831055e-03 | 0.0e+00 |
| decoder.model.4.0.pointwise_r_skip | 3.1901e-04 | -12 | -14 | 3.051758e-04 | 3.051758e-04 | 0.0e+00 |
| decoder.model.4.0.pointwise_i_skip | 3.3623e-04 | -12 | -13 | 3.662109e-04 | 3.662109e-04 | 0.0e+00 |

**max |Δ| across all 12 = 0.000e+00**

## Reproduce

```bash
cd <pkg>/software_to_hardware_18
PYTHONPATH=$PWD/../code conda run -n bcctn python - <<'EOF'
import torch, export_helpers as eh
m,_,_,_=eh.build_model(); 0
for b in eh.discover_blocks(m):
    if not b.is_outlier_fp8: continue
    for s,sub in b.pointwise_items:
        d=eh.scaled_fp8_encode(eh.raw_weight(sub).float(),torch.float8_e4m3fn)['descale']
        a=eh.apot_encode(d,-24,0,signed=False)['quantized']
        c=eh.apot_encode(d,-16,-6,signed=False)['quantized']
        assert (a-c).abs().max()==0
print('OK: [-24,0] == [-16,-6] for all outlier descales')
EOF
```
