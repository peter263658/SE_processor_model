import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from torch.quantization import fuse_modules
import numpy as np
from typing import Dict, List, Tuple, Union

# Import your complex layers
import DCNN.utils.complexPyTorch.complexLayers as torch_complex

import warnings

# Handle pytorch_quantization import and version compatibility
# try:
#     from pytorch_quantization import nn as quant_nn
#     from pytorch_quantization.tensor_quant import QuantDescriptor, MinMaxObserver
#     from pytorch_quantization import calib
    
#     # Check if QuantConvTranspose2d exists
#     if not hasattr(quant_nn, 'QuantConvTranspose2d'):
#         quant_nn.QuantConvTranspose2d = quant_nn.QuantConv2d
#         warnings.warn("QuantConvTranspose2d not found, using QuantConv2d as fallback")
# except ImportError:
#     raise ImportError("Please install pytorch-quantization: pip install pytorch-quantization")

import warnings

try:
    # 核心量化模块
    from pytorch_quantization import nn as quant_nn
    from pytorch_quantization.tensor_quant import QuantDescriptor

    # PyTorch 自带的 MinMaxObserver，用于收集负值范围
    from torch.ao.quantization.observer import MinMaxObserver
    
    # NVIDIA 提供的 calibration 工具
    import pytorch_quantization.calib as calib

    # 兼容性：如果老版本缺少 QuantConvTranspose2d，就 fallback
    if not hasattr(quant_nn, 'QuantConvTranspose2d'):
        quant_nn.QuantConvTranspose2d = quant_nn.QuantConv2d
        warnings.warn("QuantConvTranspose2d not found, using QuantConv2d as fallback")
except ImportError as e:
    raise ImportError(
        "pytorch-quantization 子模块导入失败，请通过：\n"
        "  pip uninstall pytorch-quantization\n"
        "  pip install pytorch-quantization==2.1.3 --extra-index-url https://pypi.ngc.nvidia.com\n"
        "或\n"
        "  mamba install pytorch-quantization -c nvidia\n"
        f"来安装正版 NVIDIA 量化 toolkit\n"
        f"具体错误：{e}"
    )

# Define conv classes at module level
_CONV_CLASSES = (
    nn.Conv2d, nn.ConvTranspose2d,
    quant_nn.QuantConv2d, quant_nn.QuantConvTranspose2d,
)

 

class StructuredPruningMethod(prune.BasePruningMethod):
    """Custom structured pruning method for channels/filters"""
    PRUNING_TYPE = 'structured'
    
    def __init__(self, amount: float, dim: int = 0):
        self.amount = amount
        self.dim = dim
    
    def compute_mask(self, t, default_mask):
        """Compute structured pruning mask based on L2 norm"""
        # Calculate importance scores (L2 norm along specified dimensions)
        if self.dim == 0:  # Prune output channels
            importance = torch.norm(t.view(t.size(0), -1), p=2, dim=1)
        elif self.dim == 1:  # Prune input channels
            importance = torch.norm(t.view(t.size(1), -1), p=2, dim=1)
        else:
            raise ValueError(f"Unsupported pruning dimension: {self.dim}")
        
        # Determine number of channels to prune
        num_channels = importance.size(0)
        num_to_prune = int(self.amount * num_channels)
        
        # Get indices of least important channels
        _, indices = torch.topk(importance, num_to_prune, largest=False)
        
        # Create mask
        mask = torch.ones_like(default_mask)
        if self.dim == 0:
            mask[indices] = 0
        elif self.dim == 1:
            mask[:, indices] = 0
            
        return mask

class PruningAwareQAT:
    """Pruning-aware Quantization Aware Training manager"""
    
    def __init__(self, model: nn.Module, cfg_type: str = "int8"):
        self.model = model
        self.cfg_type = cfg_type
        self.pruned_modules = {}
        self.original_weights = {}
        self.pruning_ratios = {}
        
        # Initialize quantization
        self._initialize_qat()
    
    # Add this method to PruningAwareQAT class
    def _initialize_qat(self):
        """Initialize QAT configuration with better error handling"""
        if self.cfg_type == "int8":
            try:
                # Try MinMaxObserver first (handles negative values better)
                quant_desc = QuantDescriptor(calib_method="max")
            except (ImportError, AttributeError):
                # Fallback to histogram for older versions
                quant_desc = QuantDescriptor(calib_method="histogram")
        elif self.cfg_type == "fp8":
            try:
                if hasattr(torch, "float8_e4m3fn"):
                    quant_desc = QuantDescriptor(
                        calib_method="max",
                        num_bits=8,
                        dtype=torch.float8_e4m3fn
                    )
                else:
                    raise AttributeError("FP8 not available")
            except (AttributeError, RuntimeError):
                print("Warning: FP8 not available, falling back to INT8")
                quant_desc = QuantDescriptor(calib_method="histogram")

        # Set default quantization descriptors
        quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc)
        quant_nn.QuantConvTranspose2d.set_default_quant_desc_input(quant_desc)
        quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc)


    def apply_structured_pruning(self, pruning_config: Dict[str, float]):
        """
        Apply structured pruning to specified layers including custom complex modules
        """
        print("Applying structured pruning...")
        
        for name, module in self.model.named_modules():
            if name in pruning_config:
                pruning_ratio = pruning_config[name]
                self.pruning_ratios[name] = pruning_ratio
                
                # Store original weights
                if hasattr(module, 'weight'):
                    self.original_weights[name] = module.weight.data.clone()
                
                # Apply structured pruning based on module type
                if isinstance(module, (nn.Conv2d, quant_nn.QuantConv2d)):
                    self._prune_conv2d(module, name, pruning_ratio)
                elif isinstance(module, (nn.ConvTranspose2d, quant_nn.QuantConvTranspose2d)):
                    self._prune_conv_transpose2d(module, name, pruning_ratio)
                elif isinstance(module, (nn.Linear, quant_nn.QuantLinear)):
                    self._prune_linear(module, name, pruning_ratio)
                elif isinstance(module, torch_complex.ComplexConv2d):
                    self._prune_complex_conv2d(module, name, pruning_ratio)
                elif isinstance(module, torch_complex.ComplexConvTranspose2d):
                    self._prune_complex_conv_transpose2d(module, name, pruning_ratio)
                elif isinstance(module, torch_complex.ComplexLinear):
                    self._prune_complex_linear(module, name, pruning_ratio)
                # Handle your custom complex separable modules by pruning their internal convs
                elif isinstance(module, torch_complex.ComplexDepthwiseSeparableConv2d):
                    self._prune_complex_separable_conv2d(module, name, pruning_ratio)
                elif isinstance(module, torch_complex.ComplexDepthwiseSeparableConvTranspose2d):
                    self._prune_complex_separable_conv_transpose2d(module, name, pruning_ratio)
                
                print(f"✓ Applied {pruning_ratio:.1%} structured pruning to {name}")

    
    def _prune_conv2d(self, module: nn.Module, name: str, ratio: float):
        """Apply structured pruning to Conv2d layers"""
        # Prune output channels (filters)
        StructuredPruningMethod.apply(module, 'weight', amount=ratio, dim=0)
        self.pruned_modules[name] = module
    
    def _prune_conv_transpose2d(self, module: nn.Module, name: str, ratio: float):
        """Apply structured pruning to ConvTranspose2d layers"""
        # For transpose conv, prune input channels
        StructuredPruningMethod.apply(module, 'weight', amount=ratio, dim=1)
        self.pruned_modules[name] = module
    
    def _prune_linear(self, module: nn.Module, name: str, ratio: float):
        """Apply structured pruning to Linear layers"""
        # Prune output features
        StructuredPruningMethod.apply(module, 'weight', amount=ratio, dim=0)
        self.pruned_modules[name] = module
    
    def _prune_complex_conv2d(self, module: torch_complex.ComplexConv2d, name: str, ratio: float):
        """Apply structured pruning to Complex Conv2d layers"""
        # Prune both real and imaginary parts consistently
        StructuredPruningMethod.apply(module.conv_r, 'weight', amount=ratio, dim=0)
        StructuredPruningMethod.apply(module.conv_i, 'weight', amount=ratio, dim=0)
        self.pruned_modules[name] = module
    
    def _prune_complex_conv_transpose2d(self, module: torch_complex.ComplexConvTranspose2d, name: str, ratio: float):
        """Apply structured pruning to Complex ConvTranspose2d layers"""
        StructuredPruningMethod.apply(module.conv_tran_r, 'weight', amount=ratio, dim=1)
        StructuredPruningMethod.apply(module.conv_tran_i, 'weight', amount=ratio, dim=1)
        self.pruned_modules[name] = module
    
    def _prune_complex_linear(self, module: torch_complex.ComplexLinear, name: str, ratio: float):
        """Apply structured pruning to Complex Linear layers"""
        StructuredPruningMethod.apply(module.fc_r, 'weight', amount=ratio, dim=0)
        StructuredPruningMethod.apply(module.fc_i, 'weight', amount=ratio, dim=0)
        self.pruned_modules[name] = module

    def _prune_complex_separable_conv2d(self, module: torch_complex.ComplexDepthwiseSeparableConv2d, name: str, ratio: float):
        """Apply structured pruning to Complex Depthwise Separable Conv2d layers"""
        # Prune all four internal convolutions
        StructuredPruningMethod.apply(module.depthwise_r, 'weight', amount=ratio, dim=0)
        StructuredPruningMethod.apply(module.depthwise_i, 'weight', amount=ratio, dim=0)
        StructuredPruningMethod.apply(module.pointwise_r, 'weight', amount=ratio, dim=0)
        StructuredPruningMethod.apply(module.pointwise_i, 'weight', amount=ratio, dim=0)
        self.pruned_modules[name] = module

    def _prune_complex_separable_conv_transpose2d(self, module: torch_complex.ComplexDepthwiseSeparableConvTranspose2d, name: str, ratio: float):
        """Apply structured pruning to Complex Depthwise Separable ConvTranspose2d layers"""
        # For transpose conv, prune input channels (dim=1) for pointwise, output channels (dim=0) for depthwise
        StructuredPruningMethod.apply(module.pointwise_r, 'weight', amount=ratio, dim=1)
        StructuredPruningMethod.apply(module.pointwise_i, 'weight', amount=ratio, dim=1)
        StructuredPruningMethod.apply(module.depthwise_r, 'weight', amount=ratio, dim=1)
        StructuredPruningMethod.apply(module.depthwise_i, 'weight', amount=ratio, dim=1)
        self.pruned_modules[name] = module
    
    def convert_to_qat(self):
        """Convert model to quantization-aware training"""
        print("Converting model to QAT...")
        
        # First apply fusion (conservative)
        try:
            self.model = self._fuse_modules(self.model)
            print("✓ Module fusion completed")
        except Exception as e:
            print(f"Fusion failed: {e}, continuing...")
        
        # Replace with quantized modules
        replacement_count = self._recursive_replace_with_qat(self.model)
        print(f"✓ Replaced {replacement_count} modules with quantized versions")
        
        return self.model
    
    def _fuse_modules(self, module):
        """Conservative module fusion"""
        fusion_patterns = [
            ['conv', 'bn', 'relu'],
            ['conv', 'bn'],
            ['0', '1', '2'],
            ['0', '1'],
        ]
        
        for pattern in fusion_patterns:
            try:
                if all(hasattr(module, layer_name) for layer_name in pattern):
                    fuse_modules(module, pattern, inplace=True)
            except:
                pass
        
        for child in module.children():
            self._fuse_modules(child)
        
        return module
    
    def _recursive_replace_with_qat(self, model, path="", count=0):
        """Replace layers with quantized versions while preserving pruning"""
        for name, module in list(model.named_children()):
            current_path = f"{path}.{name}" if path else name
            
            # Updated skip logic - be more specific about what to skip
            skip_keywords = [
                'multihead', 'attention', 'batch_norm', 'bn', 'norm', 'layernorm', 'groupnorm',
                'prelu', 'stft', 'istft'
            ]
            
            # Skip complex modules that should NOT be quantized, but allow depthwise separable convs
            should_skip = (
                any(keyword in current_path.lower() for keyword in skip_keywords) or
                (hasattr(module, 'weight') and module.weight.dtype == torch.complex64) or
                # Skip specific complex modules but NOT the separable convs
                isinstance(module, (
                    torch_complex.ComplexMultiheadAttention,
                    torch_complex.ComplexBatchNorm1d,
                    torch_complex.ComplexBatchNorm2d,
                    torch_complex.NaiveComplexBatchNorm1d,
                    torch_complex.NaiveComplexBatchNorm2d,
                    torch_complex.ComplexPReLU,
                    torch_complex.ComplexReLU,
                    torch_complex.ComplexSigmoid,
                    torch_complex.ComplexTanh
                ))
            )
            
            # Don't skip Linear layers in attention (input_proj, output_proj, qkv)
            if should_skip and isinstance(module, nn.Linear) and ('proj' in name or 'qkv' in name):
                should_skip = False
            
            if should_skip:
                count = self._recursive_replace_with_qat(module, current_path, count)
                continue
            
            # Replace standard layers (this will now recurse into complex separable convs)
            if isinstance(module, (nn.Conv2d, quant_nn.QuantConv2d)) and module.weight.dtype != torch.complex64:
                new_module = self._create_quant_conv2d(module)
                setattr(model, name, new_module)
                count += 1
                print(f"✓ Replaced Conv2d: {current_path}")
                
            elif isinstance(module, (nn.ConvTranspose2d, quant_nn.QuantConvTranspose2d)) and module.weight.dtype != torch.complex64:
                new_module = self._create_quant_conv_transpose2d(module)
                setattr(model, name, new_module)
                count += 1
                print(f"✓ Replaced ConvTranspose2d: {current_path}")
                
            elif isinstance(module, (nn.Linear, quant_nn.QuantLinear)) and module.weight.dtype != torch.complex64:
                new_module = self._create_quant_linear(module)
                setattr(model, name, new_module)
                count += 1
                print(f"✓ Replaced Linear: {current_path}")
            else:
                # Recurse into child modules (this will enter your custom complex modules)
                count = self._recursive_replace_with_qat(module, current_path, count)
        
        return count
    
    def _should_skip_quantization(self, module_name: str, module_type) -> bool:
        """Determine if module should skip quantization"""
        skip_keywords = [
            'prelu', 'complex', 'stft', 'istft', 'attention', 'multihead',
            'batch_norm', 'bn', 'norm', 'layernorm', 'groupnorm'
        ]
        
        if any(keyword in module_name.lower() for keyword in skip_keywords):
            return True
            
        if hasattr(module_type, 'weight') and module_type.weight.dtype == torch.complex64:
            return True
            
        return False
    
    def _create_quant_conv2d(self, original_module: nn.Conv2d) -> quant_nn.QuantConv2d:
        """Create quantized Conv2d preserving pruning masks"""
        quant_module = quant_nn.QuantConv2d(
            original_module.in_channels,
            original_module.out_channels,
            original_module.kernel_size,
            stride=original_module.stride,
            padding=original_module.padding,
            dilation=original_module.dilation,
            groups=original_module.groups,
            bias=(original_module.bias is not None)
        )
        
        # Copy weights and pruning masks
        quant_module.weight.data = original_module.weight.data.clone()
        if original_module.bias is not None:
            quant_module.bias.data = original_module.bias.data.clone()
        
        # Preserve pruning masks if they exist
        if hasattr(original_module, 'weight_mask'):
            quant_module.register_buffer('weight_mask', original_module.weight_mask)
        
        return quant_module
    
    def _create_quant_conv_transpose2d(self, original_module: nn.ConvTranspose2d) -> quant_nn.QuantConvTranspose2d:
        """Create quantized ConvTranspose2d preserving pruning masks"""
        quant_module = quant_nn.QuantConvTranspose2d(
            original_module.in_channels,
            original_module.out_channels,
            original_module.kernel_size,
            stride=original_module.stride,
            padding=original_module.padding,
            output_padding=original_module.output_padding,
            groups=original_module.groups,
            bias=(original_module.bias is not None),
            dilation=original_module.dilation
        )
        
        quant_module.weight.data = original_module.weight.data.clone()
        if original_module.bias is not None:
            quant_module.bias.data = original_module.bias.data.clone()
            
        if hasattr(original_module, 'weight_mask'):
            quant_module.register_buffer('weight_mask', original_module.weight_mask)
        
        return quant_module
    
    def _create_quant_linear(self, original_module: nn.Linear) -> quant_nn.QuantLinear:
        """Create quantized Linear preserving pruning masks"""
        quant_module = quant_nn.QuantLinear(
            original_module.in_features,
            original_module.out_features,
            bias=(original_module.bias is not None)
        )
        
        quant_module.weight.data = original_module.weight.data.clone()
        if original_module.bias is not None:
            quant_module.bias.data = original_module.bias.data.clone()
            
        if hasattr(original_module, 'weight_mask'):
            quant_module.register_buffer('weight_mask', original_module.weight_mask)
        
        return quant_module
    
    def enable_calibration(self):
        """Enable calibration for all quantized modules"""
        for module in self.model.modules():
            try:
                if hasattr(module, 'enable_calib'):
                     module.enable_calib()
                elif hasattr(module, '_input_quantizer') and hasattr(module._input_quantizer, 'enable_calib'):
                     module._input_quantizer.enable_calib()
                elif hasattr(module, '_weight_quantizer') and hasattr(module._weight_quantizer, 'enable_calib'):
                    module._weight_quantizer.enable_calib()
                elif hasattr(module, 'fake_quant_enabled'):
                    module.fake_quant_enabled = False
            except Exception as e:
                # Silently handle API differences
                pass
    
    # Add this method to PruningAwareQAT class if missing
    def disable_calibration(self):
        """Disable calibration for all quantized modules"""
        for module in self.model.modules():
            try:
                if hasattr(module, 'disable_calib'):
                    module.disable_calib()
                elif hasattr(module, '_input_quantizer') and hasattr(module._input_quantizer, 'disable_calib'):
                    module._input_quantizer.disable_calib()
                elif hasattr(module, '_weight_quantizer') and hasattr(module._weight_quantizer, 'disable_calib'):
                    module._weight_quantizer.disable_calib()
                elif hasattr(module, 'fake_quant_enabled'):
                    module.fake_quant_enabled = True
            except Exception as e:
                # Silently handle API differences
                pass
   
    def get_compression_stats(self) -> Dict[str, float]:
        """
        Return statistics with support for custom complex separable modules
        """
        total_params = 0
        effective_params = 0
        total_flops = 0
        effective_flops = 0
        depthwise_flops = 0
        pointwise_flops = 0

        for name, module in self.model.named_modules():
            # Handle your custom complex separable modules
            if isinstance(module, (torch_complex.ComplexDepthwiseSeparableConv2d, 
                                torch_complex.ComplexDepthwiseSeparableConvTranspose2d)):
                # Process all four internal convolutions
                internal_convs = []
                if hasattr(module, 'depthwise_r'):
                    internal_convs.extend([module.depthwise_r, module.depthwise_i])
                if hasattr(module, 'pointwise_r'):
                    internal_convs.extend([module.pointwise_r, module.pointwise_i])
                
                for conv in internal_convs:
                    if hasattr(conv, 'weight') and conv.weight is not None:
                        weight = conv.weight
                        
                        # Calculate effective parameters
                        if hasattr(conv, "weight_orig"):
                            actual_weight = conv.weight_orig
                            mask = conv.weight_mask
                            nz = (mask != 0).sum().item()
                            total_count = actual_weight.numel()
                        else:
                            nz = (weight != 0).sum().item()
                            total_count = weight.numel()

                        total_params += total_count
                        effective_params += nz

                        # Calculate FLOPs
                        if weight.ndim == 4:
                            k_out, k_in, k_h, k_w = weight.shape
                            
                            # Detect type based on conv properties
                            is_depthwise = hasattr(conv, 'groups') and conv.groups == k_in and k_in > 1
                            is_pointwise = k_h == 1 and k_w == 1
                            
                            if is_depthwise:
                                flops = k_out * k_h * k_w
                                depthwise_flops += flops
                            elif is_pointwise:
                                flops = k_out * k_in
                                pointwise_flops += flops
                            else:
                                flops = k_out * k_in * k_h * k_w
                            
                            total_flops += flops
                            pruning_ratio = nz / float(total_count)
                            effective_flops += flops * pruning_ratio
                continue
                
            # Handle standard modules
            if not hasattr(module, "weight"):
                continue
            weight = module.weight
            if weight is None:
                continue

            # Calculate effective parameters for standard modules
            if hasattr(module, "weight_orig"):
                actual_weight = module.weight_orig
                mask = module.weight_mask
                nz = (mask != 0).sum().item()
                total_count = actual_weight.numel()
            else:
                nz = (weight != 0).sum().item()
                total_count = weight.numel()

            total_params += total_count
            effective_params += nz

            # Calculate FLOPs for standard conv and linear layers
            if isinstance(module, _CONV_CLASSES):
                if weight.ndim == 4:
                    k_out, k_in, k_h, k_w = weight.shape
                    flops = k_out * k_in * k_h * k_w
                    total_flops += flops
                    pruning_ratio = nz / float(total_count)
                    effective_flops += flops * pruning_ratio
                    
            elif isinstance(module, (nn.Linear, quant_nn.QuantLinear)):
                if weight.ndim == 2:
                    k_out, k_in = weight.shape
                    flops = k_out * k_in
                    total_flops += flops
                    pruning_ratio = nz / float(total_count)
                    effective_flops += flops * pruning_ratio

        compression_ratio = effective_params / float(max(total_params, 1))
        pruning_ratio = 1.0 - compression_ratio
        theoretical_speedup = total_flops / max(effective_flops, 1e-9)

        return {
            "total_parameters": int(total_params),
            "effective_parameters": int(effective_params),
            "compression_ratio": compression_ratio,
            "pruning_ratio": pruning_ratio,
            "theoretical_speedup": theoretical_speedup,
            "depthwise_flops": int(depthwise_flops),
            "pointwise_flops": int(pointwise_flops),
            "total_flops": int(total_flops),
            "effective_flops": int(effective_flops),
        }
    
    def remove_pruning_reparameterization(self):
        """Remove pruning masks and make pruning permanent"""
        for name, module in self.model.named_modules():
            if name in self.pruned_modules:
                try:
                    if hasattr(module, 'weight_mask'):
                        prune.remove(module, 'weight')
                        print(f"✓ Removed pruning mask from {name}")
                    # Handle complex modules
                    elif hasattr(module, 'conv_r') and hasattr(module.conv_r, 'weight_mask'):
                        prune.remove(module.conv_r, 'weight')
                        prune.remove(module.conv_i, 'weight')
                    elif hasattr(module, 'fc_r') and hasattr(module.fc_r, 'weight_mask'):
                        prune.remove(module.fc_r, 'weight')
                        prune.remove(module.fc_i, 'weight')
                except Exception as e:
                    print(f"Could not remove pruning from {name}: {e}")

def get_default_pruning_config(model: nn.Module, base_ratio: float = 0.3) -> Dict[str, float]:
    """
    Generate default pruning configuration for BCCTN model
    
    Args:
        model: The model to analyze
        base_ratio: Base pruning ratio (0.3 = 30% pruning)
    
    Returns:
        Dictionary mapping layer names to pruning ratios
    """
    pruning_config = {}
    
    for name, module in model.named_modules():
        # More aggressive pruning for deeper layers
        if 'encoder' in name and isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            layer_idx = len([x for x in name.split('.') if x.isdigit()])
            # Gradually increase pruning ratio for deeper layers
            ratio = min(base_ratio + layer_idx * 0.1, 0.7)
            pruning_config[name] = ratio
            
        elif 'decoder' in name and isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            # More conservative pruning for decoder
            ratio = base_ratio * 0.8
            pruning_config[name] = ratio
            
        elif 'mattn' in name and isinstance(module, nn.Linear):
            # Conservative pruning for attention layers
            ratio = base_ratio * 0.5
            pruning_config[name] = ratio
    
    return pruning_config

# Convenience functions for easy use
def initialize_pruning_aware_qat(model: nn.Module, 
                                cfg_type: str = "int8",
                                pruning_config: Dict[str, float] = None,
                                base_pruning_ratio: float = 0.3) -> PruningAwareQAT:
    """
    Initialize pruning-aware QAT for a model
    
    Args:
        model: The model to optimize
        cfg_type: Quantization type ("int8" or "fp8")
        pruning_config: Custom pruning configuration
        base_pruning_ratio: Base pruning ratio if no config provided
    
    Returns:
        PruningAwareQAT manager
    """
    # Create pruning-aware QAT manager
    pqat = PruningAwareQAT(model, cfg_type)
    
    # Use default config if none provided
    if pruning_config is None:
        pruning_config = get_default_pruning_config(model, base_pruning_ratio)
    
    # Apply structured pruning
    pqat.apply_structured_pruning(pruning_config)
    
    # Convert to QAT
    pqat.convert_to_qat()
    
    return pqat