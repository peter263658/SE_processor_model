#!/usr/bin/env python3
"""
Quantization-only module for BCCTN model
Supports mixed precision: INT8 for conv layers, FP16/FP8 for attention layers
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Union
import warnings

# Import complex layers
import DCNN.utils.complexPyTorch.complexLayers as torch_complex

# Handle pytorch_quantization import
try:
    from pytorch_quantization import nn as quant_nn
    from pytorch_quantization.tensor_quant import QuantDescriptor
    from pytorch_quantization import calib
    HAS_PYTORCH_QUANT = True
except ImportError:
    HAS_PYTORCH_QUANT = False
    warnings.warn("pytorch-quantization not installed. Using PyTorch native quantization.")


class MixedPrecisionQuantizer:
    """Mixed precision quantization manager without pruning"""
    
    def __init__(self, model: nn.Module, 
                 conv_precision: str = "int8",
                 attention_precision: str = "fp16",
                 calibration_method: str = "histogram"):
        """
        Args:
            model: Model to quantize
            conv_precision: Precision for conv layers ("int8", "fp16", "fp32")
            attention_precision: Precision for attention layers ("fp16", "fp8", "fp32")
            calibration_method: Calibration method for INT8 ("histogram", "max", "entropy")
        """
        self.model = model
        self.conv_precision = conv_precision
        self.attention_precision = attention_precision
        self.calibration_method = calibration_method
        self.quantized_layers = {}
        
        # Check FP8 availability
        self.fp8_available = self._check_fp8_support()
        if attention_precision == "fp8" and not self.fp8_available:
            print("Warning: FP8 not available, falling back to FP16 for attention")
            self.attention_precision = "fp16"
    
    def _check_fp8_support(self) -> bool:
        """Check if FP8 is supported"""
        if not torch.cuda.is_available():
            return False
        
        # Check GPU architecture (needs Hopper/H100)
        capability = torch.cuda.get_device_capability()
        if capability[0] < 9:  # Hopper is compute capability 9.0
            return False
            
        # Check PyTorch version supports FP8
        return hasattr(torch, "float8_e4m3fn")
    
    def quantize_model(self) -> nn.Module:
        """Apply mixed precision quantization to model"""
        print(f"Applying mixed precision quantization:")
        print(f"  Conv layers: {self.conv_precision}")
        print(f"  Attention layers: {self.attention_precision}")
        
        if HAS_PYTORCH_QUANT and self.conv_precision == "int8":
            self._setup_int8_quantization()
            self._replace_layers_with_quantized()
        else:
            self._apply_native_quantization()
        
        return self.model
    
    def _setup_int8_quantization(self):
        """Setup INT8 quantization using pytorch-quantization"""
        quant_desc = QuantDescriptor(calib_method=self.calibration_method)
        
        # Set quantization descriptors
        quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc)
        quant_nn.QuantConvTranspose2d.set_default_quant_desc_input(quant_desc)
        quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc)
    
    def _replace_layers_with_quantized(self):
        """Replace layers with quantized versions based on precision settings"""
        count = 0
        
        for name, module in list(self.model.named_modules()):
            # Skip complex and certain layers
            if self._should_skip_layer(name, module):
                continue
            
            # Determine target precision
            if self._is_attention_layer(name, module):
                precision = self.attention_precision
            elif self._is_conv_layer(name, module):
                precision = self.conv_precision
            else:
                continue
            
            # Replace with appropriate quantized version
            parent_name = '.'.join(name.split('.')[:-1])
            module_name = name.split('.')[-1]
            parent = self.model
            
            if parent_name:
                for part in parent_name.split('.'):
                    parent = getattr(parent, part)
            
            if isinstance(module, nn.Conv2d) and precision == "int8":
                new_module = self._create_quant_conv2d(module)
                setattr(parent, module_name, new_module)
                self.quantized_layers[name] = precision
                count += 1
                
            elif isinstance(module, nn.ConvTranspose2d) and precision == "int8":
                new_module = self._create_quant_conv_transpose2d(module)
                setattr(parent, module_name, new_module)
                self.quantized_layers[name] = precision
                count += 1
                
            elif isinstance(module, nn.Linear):
                if precision == "int8" and not self._is_attention_layer(name, module):
                    new_module = self._create_quant_linear(module)
                    setattr(parent, module_name, new_module)
                    self.quantized_layers[name] = precision
                    count += 1
                elif precision in ["fp16", "fp8"]:
                    # For FP16/FP8, we'll handle it differently
                    self._convert_to_low_precision(module, precision)
                    self.quantized_layers[name] = precision
                    count += 1
        
        print(f"Replaced {count} layers with quantized versions")
        for layer, precision in self.quantized_layers.items():
            print(f"  {layer}: {precision}")
    
    def _apply_native_quantization(self):
        """Apply PyTorch native quantization (for FP16 or when pytorch-quantization not available)"""
        if self.conv_precision == "fp16" or self.attention_precision == "fp16":
            # Convert specific layers to FP16
            for name, module in self.model.named_modules():
                if self._should_skip_layer(name, module):
                    continue
                    
                if self._is_conv_layer(name, module) and self.conv_precision == "fp16":
                    module.half()
                    self.quantized_layers[name] = "fp16"
                elif self._is_attention_layer(name, module) and self.attention_precision == "fp16":
                    module.half()
                    self.quantized_layers[name] = "fp16"
    
    def _should_skip_layer(self, name: str, module) -> bool:
        """Check if layer should skip quantization"""
        skip_keywords = ['complex', 'stft', 'istft', 'bn', 'norm', 'prelu']
        
        # Skip if module name contains skip keywords
        if any(keyword in name.lower() for keyword in skip_keywords):
            return True
        
        # Skip complex-valued layers
        if hasattr(module, 'weight') and module.weight.dtype == torch.complex64:
            return True
            
        return False
    
    def _is_attention_layer(self, name: str, module) -> bool:
        """Check if layer is part of attention mechanism"""
        attention_keywords = ['mattn', 'attention', 'multihead', 'input_proj', 'output_proj']
        return any(keyword in name.lower() for keyword in attention_keywords)
    
    def _is_conv_layer(self, name: str, module) -> bool:
        """Check if layer is a conv layer"""
        return isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)) and not self._is_attention_layer(name, module)
    
    def _create_quant_conv2d(self, module: nn.Conv2d) -> quant_nn.QuantConv2d:
        """Create INT8 quantized Conv2d"""
        quant_module = quant_nn.QuantConv2d(
            module.in_channels,
            module.out_channels,
            module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
            bias=(module.bias is not None)
        )
        
        quant_module.weight.data = module.weight.data.clone()
        if module.bias is not None:
            quant_module.bias.data = module.bias.data.clone()
            
        return quant_module
    
    def _create_quant_conv_transpose2d(self, module: nn.ConvTranspose2d) -> quant_nn.QuantConvTranspose2d:
        """Create INT8 quantized ConvTranspose2d"""
        quant_module = quant_nn.QuantConvTranspose2d(
            module.in_channels,
            module.out_channels,
            module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            output_padding=module.output_padding,
            groups=module.groups,
            bias=(module.bias is not None),
            dilation=module.dilation
        )
        
        quant_module.weight.data = module.weight.data.clone()
        if module.bias is not None:
            quant_module.bias.data = module.bias.data.clone()
            
        return quant_module
    
    def _create_quant_linear(self, module: nn.Linear) -> quant_nn.QuantLinear:
        """Create INT8 quantized Linear"""
        quant_module = quant_nn.QuantLinear(
            module.in_features,
            module.out_features,
            bias=(module.bias is not None)
        )
        
        quant_module.weight.data = module.weight.data.clone()
        if module.bias is not None:
            quant_module.bias.data = module.bias.data.clone()
            
        return quant_module
    
    def _convert_to_low_precision(self, module: nn.Module, precision: str):
        """Convert module to FP16 or FP8"""
        if precision == "fp16":
            module.half()
        elif precision == "fp8" and self.fp8_available:
            # FP8 conversion (experimental)
            if hasattr(module, 'weight'):
                module.weight.data = module.weight.data.to(torch.float8_e4m3fn)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data = module.bias.data.to(torch.float8_e4m3fn)
    
    def enable_calibration(self):
        """Enable calibration for INT8 quantization"""
        if not HAS_PYTORCH_QUANT:
            return
            
        for module in self.model.modules():
            if hasattr(module, 'enable_calib'):
                module.enable_calib()
            elif hasattr(module, '_input_quantizer'):
                module._input_quantizer.enable_calib()
            elif hasattr(module, '_weight_quantizer'):
                module._weight_quantizer.enable_calib()
    
    def disable_calibration(self):
        """Disable calibration after calibration phase"""
        if not HAS_PYTORCH_QUANT:
            return
            
        for module in self.model.modules():
            if hasattr(module, 'disable_calib'):
                module.disable_calib()
            elif hasattr(module, '_input_quantizer'):
                module._input_quantizer.disable_calib()
            elif hasattr(module, '_weight_quantizer'):
                module._weight_quantizer.disable_calib()
    
    def get_quantization_stats(self) -> Dict:
        """Get statistics about quantization"""
        stats = {
            "total_layers": len(list(self.model.modules())),
            "quantized_layers": len(self.quantized_layers),
            "int8_layers": sum(1 for p in self.quantized_layers.values() if p == "int8"),
            "fp16_layers": sum(1 for p in self.quantized_layers.values() if p == "fp16"),
            "fp8_layers": sum(1 for p in self.quantized_layers.values() if p == "fp8"),
        }
        
        # Calculate model size
        total_params = sum(p.numel() for p in self.model.parameters())
        
        # Estimate compressed size
        compressed_size = 0
        for name, module in self.model.named_modules():
            if hasattr(module, 'weight'):
                weight_params = module.weight.numel()
                if name in self.quantized_layers:
                    precision = self.quantized_layers[name]
                    if precision == "int8":
                        compressed_size += weight_params  # 1 byte per param
                    elif precision == "fp16":
                        compressed_size += weight_params * 2  # 2 bytes per param
                    elif precision == "fp8":
                        compressed_size += weight_params  # 1 byte per param
                else:
                    compressed_size += weight_params * 4  # 4 bytes per param (FP32)
        
        stats["original_size_mb"] = (total_params * 4) / (1024 * 1024)
        stats["compressed_size_mb"] = compressed_size / (1024 * 1024)
        stats["compression_ratio"] = compressed_size / (total_params * 4)
        
        return stats


def quantize_model_mixed_precision(model: nn.Module,
                                  conv_precision: str = "int8",
                                  attention_precision: str = "fp16",
                                  calibration_method: str = "histogram") -> MixedPrecisionQuantizer:
    """
    Convenience function to quantize model with mixed precision
    
    Args:
        model: Model to quantize
        conv_precision: Precision for conv layers
        attention_precision: Precision for attention layers
        calibration_method: Calibration method for INT8
    
    Returns:
        MixedPrecisionQuantizer instance
    """
    quantizer = MixedPrecisionQuantizer(
        model=model,
        conv_precision=conv_precision,
        attention_precision=attention_precision,
        calibration_method=calibration_method
    )
    
    quantizer.quantize_model()
    
    return quantizer