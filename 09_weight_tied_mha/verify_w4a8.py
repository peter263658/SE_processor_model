"""Verify the W4A8 parametrization is actually applied after loading FP8 ckpt.

Checks:
  1. MHA layers carry INT4 parametrizations (attribute presence)
  2. The quantized weight ≠ the raw stored weight (fake-quant actually fires)
  3. Reconstruction error is consistent with INT4-per-head symmetric rounding
     (not with FP8)
  4. The forward pass through the MHA block produces different outputs from a
     plain FP32-weight control.
"""
import sys, os
import torch
import numpy as np
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

GlobalHydra.instance().clear()
initialize(config_path="config", version_base="1.1")
cfg = compose(config_name="config")

print("=" * 70)
print("CONFIG SANITY")
print("=" * 70)
print(f"mixed_precision_qat = {cfg.training.mixed_precision_qat}")
print(f"qat_scope           = {cfg.training.qat_scope}")
print(f"int4_group_size     = {cfg.training.int4_group_size}")
print(f"top_k_attention     = {cfg.training.top_k_attention}")

from DCNN.trainer import DCNNLightningModule

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DCNNLightningModule(cfg)
model.eval()

ckpt_path = "/raid/R12K41024/07_mixed_precision_qat/epoch=134-step=130680.ckpt"
ckpt = torch.load(ckpt_path, map_location=device)
model.load_state_dict(ckpt["state_dict"])
model = model.to(device)

print("\n" + "=" * 70)
print("PARAMETRIZATION WALK (MHA / MHA-adjacent layers only)")
print("=" * 70)

mha_like_keywords = ("mattn", "MultiheadAttention", "attention", "attn", "in_proj", "out_proj")
hits = []
for name, module in model.named_modules():
    if not hasattr(module, "parametrizations"):
        continue
    if not any(k in name for k in mha_like_keywords):
        continue
    for pname, plist in module.parametrizations.items():
        cls = type(plist[0]).__name__
        hits.append((name, pname, cls))

print(f"Found {len(hits)} MHA-related parametrized weights:")
for name, pname, cls in hits:
    print(f"  {cls:38s} <- {name}.{pname}")

print("\n" + "=" * 70)
print("QUANT FUNCTIONAL CHECK (first MHA in_proj_weight)")
print("=" * 70)
# Grab the first nn.MultiheadAttention
import torch.nn as nn
target_mha = None
target_name = None
for name, m in model.named_modules():
    if isinstance(m, nn.MultiheadAttention):
        target_mha = m
        target_name = name
        break

if target_mha is None:
    print("No nn.MultiheadAttention found!")
    sys.exit(1)

print(f"Target: {target_name}")
raw = target_mha.parametrizations.in_proj_weight.original
quantized = target_mha.in_proj_weight  # parametrization applies on access
print(f"raw    shape={tuple(raw.shape)} dtype={raw.dtype}")
print(f"quant  shape={tuple(quantized.shape)} dtype={quantized.dtype}")
print(f"raw    |w|_max = {raw.abs().max().item():.6f}")
print(f"quant  |w|_max = {quantized.abs().max().item():.6f}")

diff = (raw - quantized).abs()
print(f"|raw - quant|  max = {diff.max().item():.6e}")
print(f"|raw - quant|  mean= {diff.mean().item():.6e}")
same_mask = torch.isclose(raw, quantized, atol=1e-12)
pct_same = same_mask.float().mean().item() * 100
print(f"exact-equal elements = {pct_same:.2f}%  (W4 would be ~0%)")

# Count distinct quantized levels per head block
num_heads = cfg.model.num_heads
rows, cols = raw.shape  # 3E x E
total_groups = 3 * num_heads
head_dim = rows // total_groups
grouped = quantized.detach().reshape(total_groups, head_dim, cols)
unique_per_group = []
for g in range(total_groups):
    ug = torch.unique(grouped[g]).numel()
    unique_per_group.append(ug)
print(f"Unique values per (3H) group: min={min(unique_per_group)} "
      f"max={max(unique_per_group)} mean={np.mean(unique_per_group):.1f} "
      f"(INT4 symmetric → ≤15 per group)")

# Also check one of the ComplexInt4 per-head linears (input_proj / output_proj)
print("\n" + "=" * 70)
print("COMPLEX LINEAR CHECK (input_proj / output_proj)")
print("=" * 70)
for name, m in model.named_modules():
    if isinstance(m, nn.Linear) and hasattr(m, "parametrizations") and "weight" in m.parametrizations:
        if any(k in name for k in ("input_proj", "output_proj")):
            raw = m.parametrizations.weight.original
            q = m.weight
            dtype = raw.dtype
            d = (raw - q).abs()
            uniq = torch.unique(q.real if q.is_complex() else q).numel() if q.numel() < 500_000 else -1
            print(f"  {name:50s} dtype={dtype}  |Δ|max={d.abs().max().item():.4e} "
                  f"mean={d.abs().mean().item():.4e}  "
                  f"unique_real(≤15×groups expected)={uniq}")

print("\n" + "=" * 70)
print("FORWARD-PASS DIFF (FP32 control vs W4A8 current)")
print("=" * 70)
# Build an unquantized twin by re-instantiating the same model w/o QAT
import copy
from omegaconf import OmegaConf
cfg_fp = OmegaConf.to_container(cfg, resolve=True)
cfg_fp["training"]["mixed_precision_qat"] = False
cfg_fp["training"]["fp8_qat"] = False
cfg_fp = OmegaConf.create(cfg_fp)

# Load a pure-fp32 version
fp_model = DCNNLightningModule(cfg_fp).eval()
# Copy raw weights from ckpt — the ckpt keys will have parametrizations.weight.original,
# so we need to strip those.
remapped = {}
for k, v in ckpt["state_dict"].items():
    nk = k.replace(".parametrizations.weight.original", ".weight") \
          .replace(".parametrizations.in_proj_weight.original", ".in_proj_weight")
    remapped[nk] = v
missing, unexpected = fp_model.load_state_dict(remapped, strict=False)
print(f"FP twin load: missing={len(missing)} unexpected={len(unexpected)}")
if missing[:3]:
    print("  sample missing:", missing[:3])
if unexpected[:3]:
    print("  sample unexpected:", unexpected[:3])
fp_model = fp_model.to(device)

torch.manual_seed(0)
x = torch.randn(1, 2, 16000, device=device)
with torch.no_grad():
    y_q = model(x)
    y_fp = fp_model(x)

rel_err = (y_q - y_fp).abs().mean() / (y_fp.abs().mean() + 1e-12)
print(f"FP vs W4A8 output: |Δ|max={((y_q-y_fp).abs().max().item()):.4e}  "
      f"mean={(y_q-y_fp).abs().mean().item():.4e}  "
      f"relative={rel_err.item():.4%}")
print("(if ≈0, quantization is a no-op; W4A8 should give non-trivial deviation)")
