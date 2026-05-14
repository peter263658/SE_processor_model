"""
Custom FP8 Quantization-Aware Training (QAT) Module
====================================================
Provides FP8 Fake Quantization using PyTorch's native float8_e4m3fn / float8_e5m2
dtypes with a Straight-Through Estimator (STE) for gradient propagation.

Supports both REAL-valued and COMPLEX-valued weights and activations.
For complex tensors, real and imaginary parts are quantized independently.
"""

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


def _fp8_quantize_tensor(x, fp8_dtype):
    """Quantize a single real-valued tensor through the STE."""
    return FP8QuantizeSTE.apply(x, fp8_dtype)


def fp8_quantize(x, fp8_dtype):
    """
    Quantize a tensor to FP8, handling both real and complex dtypes.
    For complex tensors, real and imaginary parts are quantized independently.
    """
    if x.is_complex():
        real_q = _fp8_quantize_tensor(x.real, fp8_dtype)
        imag_q = _fp8_quantize_tensor(x.imag, fp8_dtype)
        return torch.complex(real_q, imag_q)
    elif x.is_floating_point():
        return _fp8_quantize_tensor(x, fp8_dtype)
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

def _make_activation_hook(fp8_dtype):
    """Create a forward-pre-hook that quantizes input activations to FP8."""

    def hook(module, args):
        new_args = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                new_args.append(fp8_quantize(arg, fp8_dtype))
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
            module.register_forward_pre_hook(_make_activation_hook(fp8_dtype))
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
            module.register_forward_pre_hook(_make_activation_hook(fp8_dtype))
            count += 1

        elif isinstance(module, nn.MultiheadAttention):
            # in_proj_weight is always real-valued
            if hasattr(module, 'in_proj_weight') and module.in_proj_weight is not None:
                register_parametrization(module, 'in_proj_weight',
                                         FP8Parametrization(dtype_str))
            # out_proj is an nn.Linear and will be caught above by named_modules
            module.register_forward_pre_hook(_make_activation_hook(fp8_dtype))
            count += 1

    print(f"✅ FP8 QAT enabled. Wrapped {count} layers with FP8 FakeQuantization.")
    return model
