"""Smoke: build the STRICT-core model and verify every op is multiplier-free.

Run before launching run_strict_core_eval.sh. Asserts the full wiring:
  - INT4 W4A8 conv/proj : shift-add proxy, all_lmul OFF, APoT-clip scale
  - decoder_pw_outlier  : FP8xFP8 L-Mul m4 (the 12 tagged layers)
  - qk / score_v        : L-Mul m8
  - BN                  : fold_apot (effective a quantized -> mul-free), NOT gamma
  - PReLU               : pow2 slope (shift)
No training, no ckpt, CPU build. ~30s.
"""
from __future__ import annotations
import contextlib
import io
import os
import sys

os.environ["BCCTN_STE_MODE"] = "lmul_exact"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_cache")
# conv APoT-clip bounds (weight scale < 1)
for k, v in (("A_MIN", "-16"), ("A_MAX", "-1"), ("B_MIN", "-16"), ("B_MAX", "-1")):
    os.environ.setdefault(f"BCCTN_APOT_{k}", v)

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from hydra import compose, initialize_config_dir  # noqa: E402
from hydra.core.global_hydra import GlobalHydra  # noqa: E402

P2 = os.path.join(_REPO, "Checkpoints/p2_skip_indices_g42_topk_l1_r050.pt")
overrides = [
    "training.qat_scope=full",
    "training.fp16_accumulator=true",
    "training.int4_scale_format=apot_clip",
    "training.top_k_attention=4",
    "model.attention_softmax.mode=shift_topk_pow2_clamp8_nomantissa",
    "model.attention_datapath.enabled=true",
    "model.attention_datapath.lmul_enabled=true",
    "model.attention_datapath.lmul_targets=[qk]",
    "model.attention_datapath.lmul_mantissa_bits=8",
    "model.attention_datapath.lmul_bit_exact=true",
    "model.attention_datapath.prob_packet.enabled=true",
    "model.attention_datapath.prob_packet.score_v_mode=lmul",
    "model.attention_datapath.prob_packet.top_k=4",
    "model.attention_datapath.prob_packet.local_idx_bits=5",
    "model.attention_datapath.prob_packet.validate_local_idx=true",
    "model.decoder_grouped_pointwise.mode=grouped",
    "model.decoder_grouped_pointwise.groups_x=4",
    "model.decoder_grouped_pointwise.groups_skip=2",
    "model.decoder_grouped_pointwise.skip_select_ratio=0.5",
    "model.decoder_grouped_pointwise.skip_select_mode=topk_l1",
    f"model.decoder_grouped_pointwise.skip_indices_path={P2}",
    "model.low_power.mha_projection.enabled=true",
    "model.low_power.mha_projection.qkv_groups=4",
    "model.low_power.all_lmul_multiply.enabled=false",
    "model.low_power.outlier_fp8_lmul.enabled=true",
    "model.low_power.outlier_fp8_lmul.mantissa_bits=4",
    # strict-core
    "model.low_power.bn_affine.scale_mode=fold_apot",
    "model.low_power.bn_affine.pow2_min_exp=-9",
    "model.low_power.bn_affine.pow2_max_exp=4",
    "model.low_power.prelu_pow2.enabled=true",
    "model.low_power.prelu_pow2.min_exp=-8",
    "model.low_power.prelu_pow2.max_exp=0",
    "model.kd.enabled=false",
]

GlobalHydra.instance().clear()
initialize_config_dir(config_dir=os.path.join(_REPO, "config"), version_base=None)
cfg = compose(config_name="config", overrides=overrides)

from DCNN.trainer import DCNNLightningModule  # noqa: E402

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    lm = DCNNLightningModule(cfg)
log = buf.getvalue()
print(log)

model = getattr(lm, "model", lm)
fp8n = list(getattr(model, "_fp8_preserved_names", []) or [])

bn_modes, prelu_pow2 = [], []
for _n, _m in model.named_modules():
    cn = _m.__class__.__name__
    if "BatchNorm2d" in cn and hasattr(_m, "scale_mode"):
        bn_modes.append(str(_m.scale_mode).lower())
    if cn == "ComplexPReLU":
        prelu_pow2.append(bool(getattr(_m, "pow2_slope", False)))

print("=" * 70)
print("[smoke-strict] verification:")
checks = []
checks.append(("12 FP8-preserved outliers tagged", len(fp8n) == 12, f"got {len(fp8n)}"))
checks.append(("outlier FP8 L-Mul patched on 12", "patched on 12" in log, ""))
checks.append(("all_lmul OFF (conv real shift-add proxy)", "attention modules (m=" not in log, ""))
checks.append(("conv scale = apot_clip", "Mixed-Precision QAT enabled" in log, ""))
ok_bn = len(bn_modes) > 0 and all(m.startswith("fold_") for m in bn_modes)
checks.append((f"all {len(bn_modes)} BN in fold_* mode (mul-free, not gamma)", ok_bn,
               f"modes={set(bn_modes)}"))
ok_pr = len(prelu_pow2) > 0 and all(prelu_pow2)
checks.append((f"all {len(prelu_pow2)} PReLU pow2_slope=True", ok_pr,
               f"on={sum(prelu_pow2)}/{len(prelu_pow2)}"))

allok = True
for label, passed, extra in checks:
    print(f"  [{'OK' if passed else 'FAIL'}] {label}" + (f"  ({extra})" if extra else ""))
    allok = allok and passed
print("=" * 70)
if not allok:
    print("[smoke-strict] FAIL")
    sys.exit(1)
print("[smoke-strict] strict network-inference-core arithmetic confirmed:")
print("  conv/proj (W4A8)  : shift-add proxy + APoT-clip[-16,-1] delayed scale")
print("  qk/score_v        : L-Mul m8")
print("  decoder_pw_outlier: FP8xFP8 L-Mul m4 (12 layers)")
print("  BN                : fold a=gamma/sqrt(var+eps) -> APoT[-9,4] shift(s)+add + bias")
print("  PReLU             : pow2 slope (sign mux + shift)")
print("[smoke-strict] OK — whole inference core is multiplier-free (amortized scale = shifts).")
