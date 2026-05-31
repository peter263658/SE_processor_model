"""
Custom FP8 Quantization-Aware Training (QAT) Module
====================================================
Provides FP8 Fake Quantization using PyTorch's native float8_e4m3fn / float8_e5m2
dtypes with a Straight-Through Estimator (STE) for gradient propagation.

Supports both REAL-valued and COMPLEX-valued weights and activations.
For complex tensors, real and imaginary parts are quantized independently.
"""

import json
import os

import torch
import torch.nn as nn
from torch.nn.utils.parametrize import register_parametrization


# ──────────────────────────────────────────────────────────────
# Core STE Function
# ──────────────────────────────────────────────────────────────

class FP8QuantizeSTE(torch.autograd.Function):
    """Simulates FP8 quantization in the forward pass with STE in backward."""

    @staticmethod
    def forward(ctx, x, dtype=torch.float8_e4m3fn):
        # E4M3 representable max = 448.0 ; E5M2 representable max = 57344.0
        max_val = 448.0 if dtype == torch.float8_e4m3fn else 57344.0

        # Per-tensor dynamic scaling
        current_amax = x.abs().max().clamp(min=1e-12)
        scale = max_val / current_amax

        # Scale → cast to FP8 → cast back → descale
        quantized = (x * scale).to(dtype).to(x.dtype)
        return quantized / scale

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-Through Estimator: pass gradients unchanged
        return grad_output, None


class FP8PlainQuantizeSTE(torch.autograd.Function):
    """Plain FP8 activation quantization with no per-tensor amax scaling.

    This mode is for the hardware experiment where activation A8 means direct
    e4m3/e5m2 storage. Values are saturated to the finite FP8 range before the
    cast to avoid PyTorch float8_e4m3fn overflow producing NaNs.
    """

    @staticmethod
    def forward(ctx, x, dtype=torch.float8_e4m3fn):
        max_val = 448.0 if dtype == torch.float8_e4m3fn else 57344.0
        return x.clamp(min=-max_val, max=max_val).to(dtype).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class FP8CalibratedQuantizeSTE(torch.autograd.Function):
    """FP8 activation quantization with an offline fixed per-boundary scale."""

    @staticmethod
    def forward(ctx, x, scale, dtype=torch.float8_e4m3fn):
        max_val = 448.0 if dtype == torch.float8_e4m3fn else 57344.0
        scale = scale.to(device=x.device, dtype=x.dtype).clamp(min=1e-12)
        scaled = (x * scale).clamp(min=-max_val, max=max_val)
        return scaled.to(dtype).to(x.dtype) / scale

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


_ACTIVATION_CALIB_CACHE = None
_ACTIVATION_CALIB_CACHE_PATH = None
_ACTIVATION_CALIB_STATS = {}


def reset_activation_calibration_stats():
    """Clear in-process activation calibration statistics."""
    _ACTIVATION_CALIB_STATS.clear()


def _activation_calibration_quantile():
    return float(os.environ.get("BCCTN_ACTIVATION_CALIB_QUANTILE", "0.999"))


def _record_activation_calibration(key, x):
    """Record one activation boundary for offline fixed-scale calibration.

    We store max over per-batch quantiles, plus the true max for audit. This
    keeps memory bounded while avoiding one rare outlier dominating the scale.
    """
    if key is None:
        key = "__unknown_activation__"
    if not x.is_floating_point():
        return
    with torch.no_grad():
        abs_x = x.detach().abs()
        if abs_x.numel() == 0:
            return
        batch_max = float(abs_x.max().item())
        q = _activation_calibration_quantile()
        if q >= 1.0 or abs_x.numel() == 1:
            batch_q = batch_max
        else:
            batch_q = float(torch.quantile(abs_x.float().reshape(-1), q).item())

    entry = _ACTIVATION_CALIB_STATS.setdefault(
        key,
        {
            "num_batches": 0,
            "num_elements": 0,
            "max_abs": 0.0,
            "quantile_abs_max": 0.0,
        },
    )
    entry["num_batches"] += 1
    entry["num_elements"] += int(x.numel())
    entry["max_abs"] = max(entry["max_abs"], batch_max)
    entry["quantile_abs_max"] = max(entry["quantile_abs_max"], batch_q)


def get_activation_calibration_stats(max_val=448.0):
    """Return JSON-serializable fixed activation scales from collected stats."""
    keys = {}
    for key, entry in sorted(_ACTIVATION_CALIB_STATS.items()):
        calib_amax = max(float(entry["quantile_abs_max"]), 1e-12)
        keys[key] = {
            "calib_amax": calib_amax,
            "scale": float(max_val / calib_amax),
            "descale": float(calib_amax / max_val),
            "max_abs": float(entry["max_abs"]),
            "num_batches": int(entry["num_batches"]),
            "num_elements": int(entry["num_elements"]),
            "stat": "max_of_batch_quantile",
            "quantile": _activation_calibration_quantile(),
        }
    return keys


def _load_activation_calibration_table():
    global _ACTIVATION_CALIB_CACHE, _ACTIVATION_CALIB_CACHE_PATH
    path = os.environ.get("BCCTN_ACTIVATION_CALIB_PATH", "").strip()
    if not path:
        raise RuntimeError(
            "BCCTN_ACTIVATION_FP8_MODE=calibrated_e4m3 requires "
            "BCCTN_ACTIVATION_CALIB_PATH=<calibration json>"
        )
    if _ACTIVATION_CALIB_CACHE is None or _ACTIVATION_CALIB_CACHE_PATH != path:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        _ACTIVATION_CALIB_CACHE = data.get("keys", data)
        _ACTIVATION_CALIB_CACHE_PATH = path
    return _ACTIVATION_CALIB_CACHE


def _lookup_activation_calibration_scale(key):
    if key is None:
        key = "__unknown_activation__"
    table = _load_activation_calibration_table()
    entry = table.get(key)
    if entry is None and key.startswith("model."):
        entry = table.get(key[len("model."):])
    if entry is None:
        raise KeyError(
            f"Missing calibrated activation scale for {key!r} in "
            f"{_ACTIVATION_CALIB_CACHE_PATH}"
        )
    return float(entry["scale"])


def _fp8_quantize_tensor(x, fp8_dtype):
    """Scaled FP8 quantization used by weight parametrizations and legacy activations."""
    return FP8QuantizeSTE.apply(x, fp8_dtype)


def _fp8_plain_quantize_tensor(x, fp8_dtype):
    """Plain FP8 quantization: no dynamic amax scale/descale."""
    return FP8PlainQuantizeSTE.apply(x, fp8_dtype)


def _fp8_calibrated_quantize_tensor(x, fp8_dtype, key):
    """Fixed-scale FP8 activation quantization from offline calibration."""
    scale = torch.tensor(_lookup_activation_calibration_scale(key))
    return FP8CalibratedQuantizeSTE.apply(x, scale, fp8_dtype)


def _pow2_round(scale):
    """Round a positive scale to a single power of two.

    BCCTN_ACTIVATION_POW2_ROUND = 'floor' (default) | 'nearest'.
    floor => scale_pow2 <= scale => calib_amax * scale_pow2 <= 448 (no extra
    saturation / overflow; block-floating-point 'no-overflow' convention).
    """
    import math
    mode = os.environ.get("BCCTN_ACTIVATION_POW2_ROUND", "floor").strip().lower()
    s = max(float(scale), 1e-30)
    e = math.log2(s)
    e = math.floor(e) if mode == "floor" else round(e)
    return 2.0 ** e


def _fp8_calibrated_pow2_quantize_tensor(x, fp8_dtype, key):
    """Calibrated FP8 activation quant with the scale snapped to a single pow2.

    A power-of-two per-tensor scale makes the activation a block-float (e4m3 +
    shared exponent): the descale is a pure exponent shift, so on every L-Mul
    site  LMul(code * 2^k, w) == 2^k * LMul(code, w)  EXACTLY. The eager descale
    used here is therefore numerically identical to the fused post-accumulation
    descale the hardware applies (no fused-vs-eager gap to recover).
    """
    scale = _pow2_round(_lookup_activation_calibration_scale(key))
    return FP8CalibratedQuantizeSTE.apply(x, torch.tensor(scale), fp8_dtype)


def activation_fp8_mode():
    """Runtime activation FP8 policy.

    ``scaled`` preserves the training/export behavior used so far:
        code = fp8(x * 448/amax), value = code * amax/448

    ``plain_e4m3`` / ``plain`` removes activation amax/descale:
        value = fp8(x)

    ``calibrated_e4m3`` uses offline fixed scales:
        code = fp8(x * fixed_scale[layer]), value = code / fixed_scale[layer]

    Weight parametrizations intentionally ignore this switch.
    """
    return os.environ.get("BCCTN_ACTIVATION_FP8_MODE", "scaled").strip().lower()


def _fp8_quantize_activation_tensor(x, fp8_dtype, key=None):
    mode = activation_fp8_mode()
    if mode in ("scaled", "scaled_e4m3", "scaled_fp8", "dynamic", "dynamic_scaled"):
        return _fp8_quantize_tensor(x, fp8_dtype)
    if mode in ("plain", "plain_e4m3", "plain_fp8", "raw", "raw_e4m3"):
        return _fp8_plain_quantize_tensor(x, fp8_dtype)
    if mode in ("calibrated", "calibrated_e4m3", "calib", "calib_e4m3", "fixed", "fixed_scale"):
        return _fp8_calibrated_quantize_tensor(x, fp8_dtype, key)
    if mode in ("calibrated_pow2", "calibrated_pow2_e4m3", "calib_pow2",
                "pow2", "pow2_e4m3", "blockfloat", "bfp_e4m3"):
        return _fp8_calibrated_pow2_quantize_tensor(x, fp8_dtype, key)
    if mode in ("calibrate", "calibrate_e4m3", "collect_calib", "calibration_collect"):
        _record_activation_calibration(key, x)
        return _fp8_quantize_tensor(x, fp8_dtype)
    raise ValueError(
        "Unsupported BCCTN_ACTIVATION_FP8_MODE="
        f"{mode!r}; expected 'scaled', 'plain_e4m3', 'calibrated_e4m3', "
        "or 'calibrated_pow2_e4m3'"
    )


def fp8_quantize(x, fp8_dtype, key=None):
    """
    Quantize an activation tensor to FP8, handling both real and complex dtypes.
    For complex tensors, real and imaginary parts are quantized independently.
    """
    if x.is_complex():
        real_key = f"{key}.real" if key is not None else None
        imag_key = f"{key}.imag" if key is not None else None
        real_q = _fp8_quantize_activation_tensor(x.real, fp8_dtype, real_key)
        imag_q = _fp8_quantize_activation_tensor(x.imag, fp8_dtype, imag_key)
        return torch.complex(real_q, imag_q)
    elif x.is_floating_point():
        return _fp8_quantize_activation_tensor(x, fp8_dtype, key)
    else:
        # Integer or bool tensors — leave unchanged
        return x


# ──────────────────────────────────────────────────────────────
# Weight Parametrization (real-valued weights only)
# ──────────────────────────────────────────────────────────────

class FP8Parametrization(nn.Module):
    """
    nn.utils.parametrize wrapper that fake-quantizes a real-valued
    parameter (e.g. weight) to FP8 on every forward call.
    """

    def __init__(self, dtype_str="e4m3"):
        super().__init__()
        self.dtype = (torch.float8_e4m3fn if dtype_str.lower() == "e4m3"
                      else torch.float8_e5m2)

    def forward(self, x):
        return _fp8_quantize_tensor(x, self.dtype)


# ──────────────────────────────────────────────────────────────
# Complex-weight Parametrization
# ──────────────────────────────────────────────────────────────

class FP8ComplexParametrization(nn.Module):
    """
    Similar to FP8Parametrization but for complex64 / complex128 parameters.
    Quantizes real and imaginary parts independently.
    """

    def __init__(self, dtype_str="e4m3"):
        super().__init__()
        self.fp8_dtype = (torch.float8_e4m3fn if dtype_str.lower() == "e4m3"
                          else torch.float8_e5m2)

    def forward(self, x):
        real_q = _fp8_quantize_tensor(x.real, self.fp8_dtype)
        imag_q = _fp8_quantize_tensor(x.imag, self.fp8_dtype)
        return torch.complex(real_q, imag_q)


# ──────────────────────────────────────────────────────────────
# Activation Hooks
# ──────────────────────────────────────────────────────────────

def _make_activation_hook(fp8_dtype, key=None):
    """Create a forward-pre-hook that quantizes input activations to FP8."""

    def hook(module, args):
        new_args = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                new_args.append(fp8_quantize(arg, fp8_dtype, key=key))
            else:
                new_args.append(arg)
        return tuple(new_args)

    return hook


# ──────────────────────────────────────────────────────────────
# Main Entry Point
# ──────────────────────────────────────────────────────────────

def enable_fp8_qat(model, dtype_str="e4m3"):
    """
    Traverse the model graph and inject FP8 Fake Quantization into all
    Conv2d, ConvTranspose2d, Linear, and MultiheadAttention sub-modules.

    This covers both real-valued AND complex-valued weights/activations.
    BatchNorm layers are intentionally left in FP32 (standard QAT practice).
    PReLU layers are also left untouched.

    Args:
        model:     nn.Module to modify **in-place**.
        dtype_str: "e4m3" (default) or "e5m2".

    Returns:
        The same model reference (modified in-place).
    """
    fp8_dtype = (torch.float8_e4m3fn if dtype_str.lower() == "e4m3"
                 else torch.float8_e5m2)

    print(f"🔥 Enabling FP8 QAT (Format: {dtype_str.upper()}) across the model...")
    count = 0

    for name, module in model.named_modules():
        # --- Standard real-valued layers (Conv2d, ConvTranspose2d, Linear) ---
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            if hasattr(module, 'weight') and module.weight is not None:
                register_parametrization(module, 'weight',
                                         FP8Parametrization(dtype_str))
            # Activation hook (inputs to Conv layers are always real in this model)
            module.register_forward_pre_hook(_make_activation_hook(fp8_dtype, name))
            count += 1

        elif isinstance(module, nn.Linear):
            if hasattr(module, 'weight') and module.weight is not None:
                # Check if the weight is complex (e.g., MultiAttnBlock.input_proj)
                if module.weight.is_complex():
                    register_parametrization(module, 'weight',
                                             FP8ComplexParametrization(dtype_str))
                else:
                    register_parametrization(module, 'weight',
                                             FP8Parametrization(dtype_str))
            # Activation hook (handles both real and complex inputs)
            module.register_forward_pre_hook(_make_activation_hook(fp8_dtype, name))
            count += 1

        elif isinstance(module, nn.MultiheadAttention):
            # in_proj_weight is always real-valued
            if hasattr(module, 'in_proj_weight') and module.in_proj_weight is not None:
                register_parametrization(module, 'in_proj_weight',
                                         FP8Parametrization(dtype_str))
            # out_proj is an nn.Linear and will be caught above by named_modules
            module.register_forward_pre_hook(_make_activation_hook(fp8_dtype, name))
            count += 1

    print(f"✅ FP8 QAT enabled. Wrapped {count} layers with FP8 FakeQuantization.")
    return model
