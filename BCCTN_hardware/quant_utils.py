"""
Quantization utilities for PTQ Export Tool.
Supports INT8 and BF16 quantization with proper scale calculation and conversion.
"""

import json
import math
from pathlib import Path
from typing import List, Union, Optional, Dict, Any

import torch
import torch.nn as nn
import numpy as np


def round_nearest_even(x: torch.Tensor) -> torch.Tensor:
    """Round to nearest even (banker's rounding)."""
    # PyTorch's round() uses round-half-to-even by default
    return torch.round(x)



def clamp_to_int8(x: torch.Tensor) -> torch.Tensor:
    """Clamp tensor to INT8 range [-128, 127]."""
    return torch.clamp(x, -128, 127)


def to_twos_complement_int8(x: torch.Tensor) -> torch.Tensor:
    """Convert to two's complement INT8 representation."""
    # PyTorch tensors are already in two's complement format
    return x.to(torch.int8)


class INT8Quantizer:
    """INT8 symmetric quantization with per-channel weights and per-tensor activations."""
    
    def __init__(self):
        self.eps = 1e-8
    
    def calibrate_weights(self, weight: torch.Tensor, max_vals: Optional[List[float]] = None) -> List[float]:
        """
        Calibrate weight scales using per-channel MaxAbs.
        
        Args:
            weight: Weight tensor [out_ch, in_ch, ...]
            max_vals: Pre-computed max values (optional)
        
        Returns:
            List of scales for each output channel
        """
        if max_vals:
            max_abs_vals = torch.tensor(max_vals, device=weight.device)
        else:
            # 處理 complex 權重
            if weight.is_complex():
                # 對 complex 權重，使用 magnitude
                weight_abs = weight.abs()
            else:
                weight_abs = weight.abs()
            
            # Compute per out_channel max
            dims_to_reduce = tuple(range(1, weight_abs.dim()))
            max_abs_vals = weight_abs.amax(dim=dims_to_reduce)
        
        # Calculate scales
        scales = (max_abs_vals / 127.0).clamp(min=self.eps)
        return scales.cpu().tolist()
    
    def calibrate_activations(self, max_vals: List[float]) -> float:
        """
        Calibrate activation scale using per-tensor MaxAbs.
        
        Args:
            max_vals: List of maximum absolute values from calibration
        
        Returns:
            Single scale value
        """
        if not max_vals:
            return 1.0
        
        max_abs = max(max_vals)
        scale = max_abs / 127.0
        return max(scale, self.eps)
    
    def quantize_weights(self, weight: torch.Tensor, scales: List[float]) -> Union[torch.Tensor, dict]:
        """
        Quantize real or complex weights to int8 using given scales.
        """
        # 以 float32 建 scale，並支援 per-tensor → per-channel 擴張
        scales_tensor = torch.tensor(scales, device=weight.device, dtype=torch.float32)
        if scales_tensor.numel() == 1:
            scales_tensor = scales_tensor.expand(weight.shape[0])
        
        # Reshape scales for broadcasting
        scale_shape = [weight.shape[0]] + [1] * (weight.dim() - 1)
        scales_tensor = scales_tensor.view(scale_shape)
        
        if weight.is_complex():
            # 用實部/虛部共用的 scale
            wr = torch.clamp(torch.round(weight.real / scales_tensor), -128, 127).to(torch.int8)
            wi = torch.clamp(torch.round(weight.imag / scales_tensor), -128, 127).to(torch.int8)
            # 返回 dict 以支援 complex 分離處理
            return {'real': wr, 'imag': wi}
        else:
            return torch.clamp(torch.round(weight / scales_tensor), -128, 127).to(torch.int8)
        
    
    def quantize_activations(self, activation: torch.Tensor, scale: float) -> torch.Tensor:
        """
        Quantize activations to INT8.
        
        Args:
            activation: FP32 activation tensor
            scale: Per-tensor scale
        
        Returns:
            INT8 quantized activations
        """
        quantized = round_nearest_even(activation / scale)
        quantized = torch.clamp(quantized, -128, 127)
        
        return quantized.to(torch.int8)
    
    def dequantize_weights(self, quantized: Union[torch.Tensor, dict], scales: List[float]) -> torch.Tensor:
        """Dequantize INT8 weights back to FP32."""
        # -- NEW: complex-safe dequant --
        if isinstance(quantized, dict):
            # quantized['real'] / ['imag'] 假為 int8 tensor
            real_q = quantized['real']
            imag_q = quantized['imag']
            device = real_q.device
            scales_tensor = torch.tensor(scales, device=device, dtype=torch.float32)
            
            # 支援 per-tensor → per-channel 擴張
            if scales_tensor.numel() == 1:
                scales_tensor = scales_tensor.expand(real_q.shape[0])
            
            scale_shape = [real_q.shape[0]] + [1] * (real_q.dim() - 1)
            scales_tensor = scales_tensor.view(scale_shape)
            
            real_fp = real_q.float() * scales_tensor
            imag_fp = imag_q.float() * scales_tensor
            return torch.complex(real_fp, imag_fp)

        # 原本的實數 tensor 路徑：保持不變
        scales_tensor = torch.tensor(scales, device=quantized.device, dtype=torch.float32)
        
        # 支援 per-tensor → per-channel 擴張
        if scales_tensor.numel() == 1:
            scales_tensor = scales_tensor.expand(quantized.shape[0])
        
        scale_shape = [quantized.shape[0]] + [1] * (quantized.dim() - 1)
        scales_tensor = scales_tensor.view(scale_shape)
        return quantized.float() * scales_tensor
    
    def dequantize_activations(self, quantized: torch.Tensor, scale: float) -> torch.Tensor:
        """Dequantize INT8 activations back to FP32."""
        return quantized.float() * scale


class BF16Quantizer:
    """BF16 quantization for attention layers."""
    
    def __init__(self):
        pass
    
    def quantize(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Convert tensor to BF16.
        
        Args:
            tensor: Input tensor
        
        Returns:
            BF16 tensor (or FP32 if BF16 not supported)
        """
        if hasattr(torch, 'bfloat16') and torch.cuda.is_bf16_supported():
            return tensor.to(torch.bfloat16)
        else:
            # Fallback to FP32 but with BF16 precision simulation
            return self._simulate_bf16(tensor)
    
    def _simulate_bf16(self, tensor: torch.Tensor) -> torch.Tensor:
        """Simulate BF16 precision on FP32 tensor."""
        # BF16 has 8 exponent bits and 7 mantissa bits
        # This is a simplified simulation
        return tensor.float()
    
    def to_fp32(self, tensor: torch.Tensor) -> torch.Tensor:
        """Convert BF16 tensor to FP32."""
        return tensor.float()


class CrossDomainConverter:
    """Convert between INT8 and BF16 domains."""
    
    def __init__(self):
        self.int8_quantizer = INT8Quantizer()
        self.bf16_quantizer = BF16Quantizer()
    
    def int8_to_bf16(self, int8_tensor: torch.Tensor, scale: float) -> torch.Tensor:
        """Convert INT8 to BF16: x_bf16 = to_bf16(dequantize_int8(x_int8, scale))."""
        fp32_tensor = self.int8_quantizer.dequantize_activations(int8_tensor, scale)
        return self.bf16_quantizer.quantize(fp32_tensor)
    
    def bf16_to_int8(self, bf16_tensor: torch.Tensor, scale: float) -> torch.Tensor:
        """Convert BF16 to INT8: x_int8 = quantize_to_int8(to_fp32(x_bf16), scale)."""
        fp32_tensor = self.bf16_quantizer.to_fp32(bf16_tensor)
        return self.int8_quantizer.quantize_activations(fp32_tensor, scale)


class QuantizationManager:
    """Manages quantization parameters and file I/O."""
    
    def __init__(self):
        self.weight_params = {}
        self.activation_params = {}
    
    def save_weight_params(self, layer_name: str, scales: List[float]):
        """Save weight quantization parameters."""
        self.weight_params[layer_name] = {
            'name': layer_name,
            'type': 'weight',
            'bitwidth': 8,
            'scheme': 'symmetric',
            'weight_axis': 'out_ch',
            'scale': scales
        }
    
    def save_activation_params(self, layer_name: str, scale: float):
        """Save activation quantization parameters."""
        self.activation_params[layer_name] = {
            'name': layer_name,
            'type': 'activation',
            'bitwidth': 8,
            'scheme': 'symmetric',
            'scale': scale
        }
    
    def export_params(self, output_dir: Path):
        """Export quantization parameters to JSON files."""
        quant_dir = output_dir / 'quant_params'
        quant_dir.mkdir(exist_ok=True)
        
        # Export weight parameters
        for layer_name, params in self.weight_params.items():
            weight_file = quant_dir / f'{layer_name}_w.json'
            with open(weight_file, 'w') as f:
                json.dump(params, f, indent=2)
        
        # Export activation parameters
        for layer_name, params in self.activation_params.items():
            act_file = quant_dir / f'{layer_name}_act.json'
            with open(act_file, 'w') as f:
                json.dump(params, f, indent=2)
    
    def load_params(self, quant_dir: Path) -> Dict[str, Any]:
        """Load quantization parameters from JSON files."""
        params = {}
        
        for json_file in quant_dir.glob('*.json'):
            with open(json_file, 'r') as f:
                param_data = json.load(f)
                params[param_data['name']] = param_data
        
        return params


def get_tensor_bit_pattern_bf16(tensor: torch.Tensor) -> torch.Tensor:
    """
    Get BF16 bit pattern as uint16 tensor.
    
    Args:
        tensor: Input tensor (any dtype)
    
    Returns:
        Tensor containing BF16 bit patterns as uint16
    """
    if tensor.dtype == torch.bfloat16:
        # Direct bit pattern extraction
        return tensor.view(torch.uint16)
    else:
        # Convert to BF16 first, then get bit pattern
        bf16_tensor = tensor.to(torch.bfloat16)
        return bf16_tensor.view(torch.uint16)


def simulate_hardware_bf16_support() -> bool:
    """Check if hardware supports BF16 operations."""
    if not torch.cuda.is_available():
        return False
    
    try:
        # Try to create a BF16 tensor and perform operation
        x = torch.tensor([1.0], dtype=torch.bfloat16, device='cuda')
        y = x + x
        return True
    except:
        return False


# Quantization error analysis utilities
class QuantizationErrorAnalyzer:
    """Analyze quantization errors and provide statistics."""
    
    def __init__(self):
        self.errors = {}
    
    def analyze_layer_error(self, layer_name: str, original: torch.Tensor, 
                          quantized: torch.Tensor, dequantized: torch.Tensor) -> Dict[str, float]:
        """
        Analyze quantization error for a single layer.
        
        Args:
            layer_name: Name of the layer
            original: Original FP32 tensor
            quantized: Quantized tensor
            dequantized: Dequantized tensor
        
        Returns:
            Dictionary containing error statistics
        """
        # Calculate various error metrics
        mse = torch.mean((original - dequantized) ** 2).item()
        mae = torch.mean(torch.abs(original - dequantized)).item()
        
        # Signal-to-noise ratio
        signal_power = torch.mean(original ** 2).item()
        noise_power = torch.mean((original - dequantized) ** 2).item()
        snr = 10 * math.log10(signal_power / (noise_power + 1e-10))
        
        # Peak signal-to-noise ratio
        max_val = torch.max(torch.abs(original)).item()
        psnr = 20 * math.log10(max_val / (math.sqrt(mse) + 1e-10))
        
        # Cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(
            original.flatten(), dequantized.flatten(), dim=0
        ).item()
        
        error_stats = {
            'mse': mse,
            'mae': mae,
            'snr_db': snr,
            'psnr_db': psnr,
            'cosine_similarity': cos_sim,
            'max_abs_error': torch.max(torch.abs(original - dequantized)).item(),
            'quantization_range': torch.max(quantized).item() - torch.min(quantized).item()
        }
        
        self.errors[layer_name] = error_stats
        return error_stats
    
    def get_summary_stats(self) -> Dict[str, float]:
        """Get summary statistics across all layers."""
        if not self.errors:
            return {}
        
        all_mse = [stats['mse'] for stats in self.errors.values()]
        all_snr = [stats['snr_db'] for stats in self.errors.values()]
        all_cos_sim = [stats['cosine_similarity'] for stats in self.errors.values()]
        
        return {
            'avg_mse': np.mean(all_mse),
            'avg_snr_db': np.mean(all_snr),
            'avg_cosine_similarity': np.mean(all_cos_sim),
            'min_snr_db': np.min(all_snr),
            'max_snr_db': np.max(all_snr)
        }


# Utility functions for specific quantization scenarios
def quantize_complex_tensor(real_part: torch.Tensor, imag_part: torch.Tensor, 
                          quantizer: INT8Quantizer, scale: float) -> tuple:
    """
    Quantize complex tensor by quantizing real and imaginary parts separately.
    
    Args:
        real_part: Real part of complex tensor
        imag_part: Imaginary part of complex tensor
        quantizer: INT8Quantizer instance
        scale: Quantization scale
    
    Returns:
        Tuple of (quantized_real, quantized_imag)
    """
    quant_real = quantizer.quantize_activations(real_part, scale)
    quant_imag = quantizer.quantize_activations(imag_part, scale)
    
    return quant_real, quant_imag


def dequantize_complex_tensor(quant_real: torch.Tensor, quant_imag: torch.Tensor,
                            quantizer: INT8Quantizer, scale: float) -> torch.Tensor:
    """
    Dequantize complex tensor from separate real and imaginary parts.
    
    Args:
        quant_real: Quantized real part
        quant_imag: Quantized imaginary part
        quantizer: INT8Quantizer instance
        scale: Quantization scale
    
    Returns:
        Complex tensor
    """
    real_part = quantizer.dequantize_activations(quant_real, scale)
    imag_part = quantizer.dequantize_activations(quant_imag, scale)
    
    return torch.complex(real_part, imag_part)


def adaptive_quantization_scale(tensor: torch.Tensor, percentile: float = 99.9) -> float:
    """
    Compute adaptive quantization scale using percentile-based clipping.
    
    Args:
        tensor: Input tensor
        percentile: Percentile for clipping (default: 99.9%)
    
    Returns:
        Adaptive scale value
    """
    abs_tensor = tensor.abs()
    threshold = torch.quantile(abs_tensor, percentile / 100.0)
    scale = threshold.item() / 127.0
    
    return max(scale, 1e-8)


def mixed_precision_scale_selection(tensor: torch.Tensor, target_bits: int = 8) -> float:
    """
    Select quantization scale for mixed precision scenarios.
    
    Args:
        tensor: Input tensor
        target_bits: Target bit width
    
    Returns:
        Optimal scale for the given bit width
    """
    max_quant_val = (2 ** (target_bits - 1)) - 1  # e.g., 127 for 8-bit
    max_abs = tensor.abs().max().item()
    
    return max_abs / max_quant_val if max_abs > 0 else 1e-8


def fake_quant_activation(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    x: fp32 activation
    scale: scalar tensor (per-tensor), shape [] 或 [1]
    流程: q = RNE(x/scale) -> clamp to int8 -> dq = q*scale (fp32)
    """
    if not torch.is_tensor(scale):
        scale = torch.tensor(scale, dtype=x.dtype, device=x.device)
    q = round_nearest_even(x / scale)
    q = torch.clamp(q, -128, 127)
    x_dq = q * scale
    return x_dq
