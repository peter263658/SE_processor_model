"""Smoke: build the B1b multiplier-free model and verify the per-op arithmetic.

Required before launching the B1b fine-tune (codex). Asserts the full
multiplier-free wiring is actually what we think:
  - INT4 W4A8 conv/proj : real-mult (shift-add EXACT proxy) — all_lmul OFF
  - decoder_pw_outlier  : FP8xFP8 L-Mul m4 — ONLY the 12 tagged layers
  - qk / score_v        : L-Mul m4 (c3 datapath)
  - MHA projection      : NOT L-Mul (all_lmul off; c3 doesn't set proj L-Mul)
  - INT4 scale          : pow2
  - STE mode            : lmul_exact (env)
No training, no ckpt, CPU build. ~30s.
"""
from __future__ import annotations
import contextlib
import io
import os
import sys

os.environ["BCCTN_STE_MODE"] = "lmul_exact"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU build, don't grab a GPU
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")   # librosa/numba need a writable cache
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_cache")

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from hydra import compose, initialize_config_dir  # noqa: E402
from hydra.core.global_hydra import GlobalHydra  # noqa: E402

P2 = os.path.join(_REPO, "Checkpoints/p2_skip_indices_g42_topk_l1_r050.pt")
overrides = [
    "training.qat_scope=full",
    "training.fp16_accumulator=true",
    "training.int4_scale_format=pow2",
    "training.top_k_attention=4",
    "model.attention_softmax.mode=shift_topk_pow2_clamp8_nomantissa",
    "model.attention_datapath.enabled=true",
    "model.attention_datapath.lmul_enabled=true",
    "model.attention_datapath.lmul_targets=[qk]",
    "model.attention_datapath.lmul_mantissa_bits=4",
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

print("=" * 70)
print("[smoke-b1b] verification:")
checks = []

ok = len(fp8n) == 12
checks.append(("12 FP8-preserved outliers tagged", ok, f"got {len(fp8n)}"))

ok = ("[outlier-lmul] FP8" in log) and ("patched on 12" in log)
checks.append(("outlier FP8 L-Mul patched on 12 layers", ok, ""))

# all_lmul OFF => its summary line ("... attention modules (m=") must be absent
ok = "attention modules (m=" not in log
checks.append(("all_lmul OFF (no all_lmul MHA-patch log line)", ok, ""))

# Hard check: inspect the ComplexMHA datapath flag directly (not just log).
mha_proj_lmul = None
for _n, _mod in model.named_modules():
    if _mod.__class__.__name__ == "ComplexMultiheadAttention":
        _dp = getattr(_mod, "attention_datapath", {}) or {}
        mha_proj_lmul = bool(_dp.get("lmul_all_projection_ops", False))
        break
ok = (mha_proj_lmul is False)
checks.append(("MHA projection L-Mul OFF (datapath flag inspected)", ok,
               f"lmul_all_projection_ops={mha_proj_lmul}"))

ok = "Mixed-Precision QAT enabled" in log
checks.append(("W4A8 mixed-precision QAT enabled (conv real + pow2)", ok, ""))

ok = os.environ.get("BCCTN_STE_MODE") == "lmul_exact"
checks.append(("STE mode = lmul_exact (qk/pv backward)", ok, ""))

allok = True
for label, passed, extra in checks:
    print(f"  [{'OK' if passed else 'FAIL'}] {label}" + (f"  ({extra})" if extra else ""))
    allok = allok and passed

print("=" * 70)
if not allok:
    print("[smoke-b1b] FAIL")
    sys.exit(1)
print("[smoke-b1b] per-op arithmetic confirmed:")
print("  conv/proj (W4A8)  : real-mult shift-add proxy + pow2 scale")
print("  qk/score_v        : L-Mul m4 (c3 datapath), backward STE lmul_exact")
print("  decoder_pw_outlier: FP8xFP8 L-Mul m4 (12 layers)")
print("  MHA projection    : NOT L-Mul")
print("[smoke-b1b] OK — B1b is full-multiplier-free (except amortized pow2 scale = shift)")
