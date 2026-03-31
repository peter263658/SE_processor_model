#!/usr/bin/env python3
"""
Enhanced BCCTN Model Weight Analyzer for Multi-Format Quantization Compatibility

Supports INT8, FP8 E4M3, and FP8 E5M2 formats with block-wise quantization analysis.
Specialized for BCCTN models with complex-valued weights.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from collections import defaultdict
import json
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Quantization Format Specifications
QUANTIZATION_FORMATS = {
    'int8': {
        'name': 'INT8 (8-bit integer)',
        'max_value': 127.0,
        'min_value': -128.0,
        'precision_bits': 7,  # 1 sign bit + 7 magnitude bits
        'dynamic_range_db': 20 * np.log10(127 / 1),  # Assuming min representable is 1
        'special_values': {'inf': False, 'nan': False},
        'use_case': 'General purpose quantization (weights & activations)',
        'quantization_type': 'symmetric',
        'zero_point': 0
    },
    'e4m3': {
        'name': 'FP8 E4M3 (4-bit exponent, 3-bit mantissa)',
        'max_value': 448.0,
        'min_value': 1/512,  # 2^(-9)
        'precision_bits': 3,
        'dynamic_range_db': 20 * np.log10(448 / (1/512)),
        'special_values': {'inf': False, 'nan': True},
        'use_case': 'Forward pass (activations and weights)',
        'quantization_type': 'floating_point'
    },
    'e5m2': {
        'name': 'FP8 E5M2 (5-bit exponent, 2-bit mantissa)', 
        'max_value': 57344.0,  # 2^15 * (1 + 3/4)
        'min_value': 1/16384,  # 2^(-14)
        'precision_bits': 2,
        'dynamic_range_db': 20 * np.log10(57344 / (1/16384)),
        'special_values': {'inf': True, 'nan': True},
        'use_case': 'Gradients and optimizer states',
        'quantization_type': 'floating_point'
    }
}

# Block quantization configurations
BLOCK_CONFIGS = {
    'per_tensor': {'block_size': None, 'description': 'Single scale per tensor'},
    'per_channel': {'block_size': 'channel', 'description': 'Per output channel'},
    'block_64': {'block_size': 64, 'description': '64-element blocks'},
    'block_128': {'block_size': 128, 'description': '128-element blocks'},
    'block_256': {'block_size': 256, 'description': '256-element blocks'},
    'block_512': {'block_size': 512, 'description': '512-element blocks'}
}

class EnhancedBCCTNWeightAnalyzer:
    """Enhanced weight analyzer for BCCTN models with INT8, FP8, and block quantization support."""
    
    def __init__(self, model_path: str, config_path: Optional[str] = None):
        self.model_path = Path(model_path)
        self.config_path = Path(config_path) if config_path else None
        self.model = None
        self.checkpoint = None
        self.weight_stats = defaultdict(dict)
        self.layer_analysis = defaultdict(dict)
        
    def load_model_and_checkpoint(self):
        """Load model and checkpoint with proper error handling."""
        print(f"Loading checkpoint from: {self.model_path}")
        
        try:
            # Load checkpoint
            self.checkpoint = torch.load(self.model_path, map_location='cpu')
            print(f"✓ Checkpoint loaded successfully")
            
            # Try to load the actual model if config is provided
            if self.config_path and self.config_path.exists():
                try:
                    from hydra import compose, initialize
                    from hydra.core.global_hydra import GlobalHydra
                    from DCNN.trainer import DCNNLightningModule
                    
                    # Initialize Hydra
                    GlobalHydra.instance().clear()
                    initialize(config_path=str(self.config_path.parent))
                    config = compose(config_name=self.config_path.name.replace('.yaml', ''))
                    
                    # Load model
                    self.model = DCNNLightningModule(config)
                    self.model.load_state_dict(self.checkpoint["state_dict"])
                    print(f"✓ Model loaded successfully")
                    
                except Exception as e:
                    print(f"Warning: Could not load model architecture: {e}")
                    print("Will analyze checkpoint weights directly")
                    
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint: {e}")
    
    def extract_weights(self) -> Dict[str, torch.Tensor]:
        """Extract all weights from model or checkpoint."""
        weights = {}
        
        if self.model is not None:
            # Extract from loaded model
            for name, param in self.model.named_parameters():
                if param.requires_grad and 'weight' in name:
                    weights[name] = param.data.detach().clone()
        else:
            # Extract from checkpoint state_dict
            state_dict = self.checkpoint.get("state_dict", self.checkpoint)
            for name, tensor in state_dict.items():
                if 'weight' in name and torch.is_tensor(tensor):
                    weights[name] = tensor.detach().clone()
        
        print(f"✓ Extracted {len(weights)} weight tensors")
        return weights
    
    def safe_tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Safely convert tensor to numpy with overflow handling."""
        try:
            # Handle complex tensors
            if tensor.dtype in [torch.complex64, torch.complex128]:
                # Use magnitude for complex tensors
                tensor_abs = tensor.abs()
            else:
                tensor_abs = tensor.abs()
            
            # Convert to float32 to avoid overflow
            if tensor_abs.dtype in [torch.float64, torch.complex128]:
                tensor_abs = tensor_abs.float()
            
            # Replace inf and nan values
            tensor_abs = torch.where(torch.isnan(tensor_abs), torch.tensor(1e-10), tensor_abs)
            tensor_abs = torch.where(torch.isinf(tensor_abs), torch.tensor(1e10), tensor_abs)
            
            # Clamp extreme values to avoid overflow
            tensor_abs = torch.clamp(tensor_abs, min=1e-20, max=1e20)
            
            # Convert to numpy
            return tensor_abs.detach().cpu().numpy().astype(np.float32)
            
        except Exception as e:
            print(f"Warning: Error converting tensor to numpy: {e}")
            # Fallback: create a safe numpy array
            if hasattr(tensor, 'shape'):
                return np.full(tensor.shape, 1e-10, dtype=np.float32)
            else:
                return np.array([1e-10], dtype=np.float32)
    
    def analyze_block_quantization(self, tensor: torch.Tensor, block_config: str) -> Dict:
        """Analyze tensor for block-wise quantization compatibility."""
        config = BLOCK_CONFIGS[block_config]
        block_size = config['block_size']
        
        try:
            # Handle complex tensors
            if tensor.dtype in [torch.complex64, torch.complex128]:
                # For complex tensors, analyze magnitude
                tensor_values = tensor.abs()
                tensor_type = 'complex'
            else:
                tensor_values = tensor
                tensor_type = 'real'
            
            # Flatten tensor for analysis
            flat_tensor = tensor_values.flatten()
            
            if block_size is None:
                # Per-tensor quantization
                blocks = [flat_tensor]
                block_info = "per_tensor"
            elif block_size == 'channel':
                # Per-channel quantization (assuming first dimension is output channels)
                if len(tensor.shape) >= 2:
                    blocks = [tensor_values[i].flatten() for i in range(tensor_values.shape[0])]
                    block_info = f"per_channel ({tensor_values.shape[0]} channels)"
                else:
                    blocks = [flat_tensor]
                    block_info = "per_tensor (single channel)"
            else:
                # Fixed block size
                num_elements = len(flat_tensor)
                num_blocks = (num_elements + block_size - 1) // block_size
                blocks = []
                for i in range(num_blocks):
                    start_idx = i * block_size
                    end_idx = min((i + 1) * block_size, num_elements)
                    blocks.append(flat_tensor[start_idx:end_idx])
                block_info = f"{num_blocks} blocks of size {block_size}"
            
            # Analyze each block
            block_stats = []
            for i, block in enumerate(blocks):
                if len(block) == 0:
                    continue
                    
                block_np = block.detach().cpu().numpy()
                # Remove zeros and very small values
                non_zero_mask = np.abs(block_np) > 1e-15
                non_zero_values = np.abs(block_np[non_zero_mask])
                
                if len(non_zero_values) == 0:
                    continue
                
                block_stat = {
                    'block_id': i,
                    'elements': len(block),
                    'non_zero_elements': len(non_zero_values),
                    'sparsity': 1 - (len(non_zero_values) / len(block)),
                    'min': float(np.min(non_zero_values)),
                    'max': float(np.max(non_zero_values)),
                    'mean': float(np.mean(non_zero_values)),
                    'std': float(np.std(non_zero_values)),
                    'dynamic_range_db': 20 * np.log10(np.max(non_zero_values) / np.max([np.min(non_zero_values), 1e-20]))
                }
                block_stats.append(block_stat)
            
            # Aggregate statistics
            if block_stats:
                dynamic_ranges = [b['dynamic_range_db'] for b in block_stats]
                sparsities = [b['sparsity'] for b in block_stats]
                
                aggregate_stats = {
                    'block_config': block_config,
                    'block_info': block_info,
                    'tensor_type': tensor_type,
                    'total_blocks': len(block_stats),
                    'avg_dynamic_range_db': float(np.mean(dynamic_ranges)),
                    'max_dynamic_range_db': float(np.max(dynamic_ranges)),
                    'min_dynamic_range_db': float(np.min(dynamic_ranges)),
                    'std_dynamic_range_db': float(np.std(dynamic_ranges)),
                    'avg_sparsity': float(np.mean(sparsities)),
                    'max_sparsity': float(np.max(sparsities)),
                    'blocks_over_threshold': {
                        'int8_40db': sum(1 for dr in dynamic_ranges if dr > 40),
                        'fp8_60db': sum(1 for dr in dynamic_ranges if dr > 60),
                        'fp8_80db': sum(1 for dr in dynamic_ranges if dr > 80)
                    },
                    'block_details': block_stats[:20]  # Store first 20 for detailed analysis
                }
                
                return aggregate_stats
            else:
                return {
                    'block_config': block_config,
                    'block_info': block_info,
                    'tensor_type': tensor_type,
                    'error': 'No valid blocks found'
                }
                
        except Exception as e:
            return {
                'block_config': block_config,
                'error': str(e)
            }
    
    def evaluate_quantization_compatibility(self, stats: Dict, format_name: str, 
                                          block_analysis: Optional[Dict] = None) -> Dict:
        """Evaluate compatibility with specific quantization format (INT8, E4M3, E5M2)."""
        if 'error' in stats or 'warning' in stats:
            return {
                'format': format_name,
                'max_value_fits': False,
                'min_value_fits': False,
                'dynamic_range_fits': False,
                'precision_loss_estimate': 1.0,
                'compatibility_score': 0,
                'recommendations': ['Cannot analyze due to tensor errors']
            }
        
        format_info = QUANTIZATION_FORMATS[format_name]
        
        # Extract key values
        abs_max = stats['abs_max']
        abs_min = stats['abs_min']
        dynamic_range_db = stats['dynamic_range_db']
        
        # For INT8, we need to consider the quantization process
        if format_name == 'int8':
            # Calculate quantization scale
            scale = abs_max / 127.0 if abs_max > 0 else 1.0
            quantization_error = scale / 2.0  # Half LSB error
            
            # Check if we can represent the range
            max_fits = abs_max <= format_info['max_value'] * scale
            min_fits = abs_min >= quantization_error  # Can represent non-zero values
            
            # For INT8, dynamic range is limited by precision
            effective_dynamic_range = 20 * np.log10(127)  # ~42 dB
            dynamic_range_fits = dynamic_range_db <= effective_dynamic_range
            
        else:  # FP8 formats
            max_fits = abs_max <= format_info['max_value']
            min_fits = abs_min >= format_info['min_value']
            dynamic_range_fits = dynamic_range_db <= format_info['dynamic_range_db']
        
        # Precision loss estimation
        precision_loss = self._estimate_precision_loss_safe(stats, format_info, format_name)
        
        # Block quantization bonus
        block_bonus = 0
        if block_analysis and 'avg_dynamic_range_db' in block_analysis:
            block_dr = block_analysis['avg_dynamic_range_db']
            if format_name == 'int8' and block_dr < 35:
                block_bonus = 15
            elif format_name in ['e4m3', 'e5m2'] and block_dr < format_info['dynamic_range_db'] * 0.8:
                block_bonus = 10
        
        # Overall compatibility score (0-100)
        compatibility_score = 0
        if max_fits:
            compatibility_score += 35
        if min_fits:
            compatibility_score += 25
        if dynamic_range_fits:
            compatibility_score += 25
        if precision_loss < 0.1:  # Less than 10% precision loss
            compatibility_score += 15
        
        # Add block quantization bonus
        compatibility_score = min(100, compatibility_score + block_bonus)
        
        return {
            'format': format_name,
            'max_value_fits': max_fits,
            'min_value_fits': min_fits,
            'dynamic_range_fits': dynamic_range_fits,
            'precision_loss_estimate': precision_loss,
            'compatibility_score': compatibility_score,
            'block_quantization_benefit': block_bonus,
            'recommendations': self._generate_recommendations_safe(
                max_fits, min_fits, dynamic_range_fits, precision_loss, 
                format_info, stats, format_name, block_analysis
            )
        }
    
    def _estimate_precision_loss_safe(self, stats: Dict, format_info: Dict, format_name: str) -> float:
        """Estimate precision loss for quantization with safe operations."""
        try:
            if format_name == 'int8':
                # For INT8, precision loss depends on quantization scale
                abs_max = stats['abs_max']
                scale = abs_max / 127.0 if abs_max > 0 else 1.0
                typical_value = stats['percentiles']['p50']
                
                if typical_value <= 0:
                    return 1.0
                
                # Quantization error as fraction of typical value
                quantization_error = scale / 2.0
                relative_precision_loss = quantization_error / typical_value
                
                return min(relative_precision_loss, 1.0)
            else:
                # For FP8, use mantissa bits
                mantissa_bits = format_info['precision_bits']
                max_representable_precision = 2 ** (-mantissa_bits)
                typical_value = stats['percentiles']['p50']
                
                if typical_value <= 0:
                    return 1.0
                    
                relative_precision_loss = max_representable_precision / typical_value
                return min(relative_precision_loss, 1.0)
        except:
            return 0.5  # Default moderate precision loss
    
    def _generate_recommendations_safe(self, max_fits: bool, min_fits: bool, 
                                     dynamic_range_fits: bool, precision_loss: float,
                                     format_info: Dict, stats: Dict, format_name: str,
                                     block_analysis: Optional[Dict] = None) -> List[str]:
        """Generate actionable recommendations with safe operations."""
        recommendations = []
        
        # Format-specific prefix
        format_prefix = {
            'int8': '🔢 INT8',
            'e4m3': '🔥 FP8-E4M3',
            'e5m2': '⚡ FP8-E5M2'
        }.get(format_name, format_name.upper())
        
        # Check for complex tensor specific recommendations
        if stats['type'] == 'complex':
            if format_name == 'int8':
                recommendations.append(
                    f"ℹ️  {format_prefix}: Complex tensor detected. Consider separate INT8 quantization "
                    f"for real/imaginary parts, or convert to magnitude-only representation."
                )
            else:
                recommendations.append(
                    f"ℹ️  {format_prefix}: Complex tensor detected. Consider separate quantization "
                    f"for magnitude and phase, or convert to magnitude-only representation."
                )
        
        # Range compatibility
        if not max_fits:
            if format_name == 'int8':
                recommendations.append(
                    f"⚠️  {format_prefix}: Values exceed quantization range. "
                    f"Max value: {stats['abs_max']:.2e}. Consider per-channel or block quantization."
                )
            else:
                recommendations.append(
                    f"⚠️  {format_prefix}: Values exceed {format_info['name']} max ({format_info['max_value']:.2e}). "
                    f"Max value: {stats['abs_max']:.2e}. Consider weight clipping or scaling."
                )
        
        if not min_fits:
            if format_name == 'int8':
                recommendations.append(
                    f"⚠️  {format_prefix}: Small values may be lost to quantization noise. "
                    f"Min value: {stats['abs_min']:.2e}. Consider higher bit precision for small weights."
                )
            else:
                recommendations.append(
                    f"⚠️  {format_prefix}: Values below {format_info['name']} min ({format_info['min_value']:.2e}). "
                    f"Min value: {stats['abs_min']:.2e}. May lose small weights to zero."
                )
        
        # Dynamic range
        if not dynamic_range_fits:
            if format_name == 'int8':
                recommendations.append(
                    f"⚠️  {format_prefix}: Dynamic range ({stats['dynamic_range_db']:.1f} dB) too large for "
                    f"effective INT8 quantization (~42 dB effective). "
                    f"Strongly recommend block-wise or per-channel quantization."
                )
            else:
                recommendations.append(
                    f"⚠️  {format_prefix}: Dynamic range ({stats['dynamic_range_db']:.1f} dB) too large for "
                    f"{format_info['name']} ({format_info['dynamic_range_db']:.1f} dB). "
                    f"Consider block-wise quantization or mixed precision."
                )
        
        # Precision loss
        if precision_loss > 0.1:
            recommendations.append(
                f"⚠️  {format_prefix}: High precision loss ({precision_loss:.1%}). "
                f"Monitor model accuracy after quantization."
            )
        
        # Success cases
        if precision_loss < 0.05 and max_fits and min_fits:
            recommendations.append(
                f"✅ {format_prefix}: Excellent compatibility!"
            )
        
        # Block quantization recommendations
        if block_analysis and 'avg_dynamic_range_db' in block_analysis:
            avg_block_dr = block_analysis['avg_dynamic_range_db']
            if format_name == 'int8' and avg_block_dr < 35:
                recommendations.append(
                    f"✅ {format_prefix}: Block quantization shows {avg_block_dr:.1f} dB avg dynamic range. "
                    f"Excellent for INT8!"
                )
            elif format_name in ['e4m3', 'e5m2'] and avg_block_dr < format_info['dynamic_range_db'] * 0.8:
                recommendations.append(
                    f"✅ {format_prefix}: Block quantization reduces dynamic range to {avg_block_dr:.1f} dB. "
                    f"Good improvement for {format_name.upper()}!"
                )
        
        # Sparsity-based recommendations
        if stats.get('sparsity', 0) > 0.5:
            recommendations.append(
                f"ℹ️  {format_prefix}: High sparsity ({stats['sparsity']:.1%}) detected. "
                f"Structured sparse quantization may be beneficial."
            )
        
        # Format-specific recommendations
        if format_name == 'int8':
            if stats['dynamic_range_db'] > 40:
                recommendations.append(
                    f"💡 {format_prefix}: Consider quantization-aware training (QAT) to improve INT8 compatibility."
                )
        
        return recommendations
    
    def analyze_tensor_statistics(self, tensor: torch.Tensor, name: str) -> Dict:
        """Compute comprehensive statistics for a tensor with robust error handling."""
        try:
            # Handle complex tensors safely
            if tensor.dtype in [torch.complex64, torch.complex128]:
                # For complex tensors, analyze both magnitude and phase
                tensor_abs = tensor.abs()
                tensor_phase = tensor.angle()
                tensor_type = 'complex'
                
                # Convert to safe numpy arrays
                abs_numpy = self.safe_tensor_to_numpy(tensor_abs)
                phase_numpy = tensor_phase.detach().cpu().numpy()
                original_numpy = None  # Complex values can't be directly converted
                
            else:
                tensor_abs = tensor.abs()
                tensor_type = 'real'
                
                # Convert to safe numpy arrays
                abs_numpy = self.safe_tensor_to_numpy(tensor_abs)
                phase_numpy = None
                original_numpy = tensor.detach().cpu().numpy().astype(np.float32)
            
            # Flatten arrays for statistics
            flat_abs = abs_numpy.flatten()
            
            # Remove zeros and very small values for better analysis
            non_zero_mask = flat_abs > 1e-15
            non_zero_values = flat_abs[non_zero_mask]
            
            if len(non_zero_values) == 0:
                return {
                    'name': name,
                    'type': tensor_type,
                    'shape': list(tensor.shape),
                    'dtype': str(tensor.dtype),
                    'total_elements': tensor.numel(),
                    'warning': 'All values are zero or near-zero'
                }
            
            # Calculate statistics with safe operations
            stats = {
                'name': name,
                'type': tensor_type,
                'shape': list(tensor.shape),
                'dtype': str(tensor.dtype),
                'total_elements': tensor.numel(),
                'non_zero_elements': len(non_zero_values),
                'sparsity': 1 - (len(non_zero_values) / tensor.numel()),
                
                # Absolute value statistics
                'abs_mean': float(np.mean(non_zero_values)),
                'abs_std': float(np.std(non_zero_values)),
                'abs_min': float(np.min(non_zero_values)),
                'abs_max': float(np.max(non_zero_values)),
                
                # Percentiles for range analysis
                'percentiles': {
                    'p1': float(np.percentile(non_zero_values, 1)),
                    'p5': float(np.percentile(non_zero_values, 5)),
                    'p25': float(np.percentile(non_zero_values, 25)),
                    'p50': float(np.percentile(non_zero_values, 50)),
                    'p75': float(np.percentile(non_zero_values, 75)),
                    'p95': float(np.percentile(non_zero_values, 95)),
                    'p99': float(np.percentile(non_zero_values, 99)),
                },
            }
            
            # Calculate dynamic range safely
            if stats['abs_max'] > 0 and stats['abs_min'] > 0:
                stats['dynamic_range_db'] = 20 * np.log10(stats['abs_max'] / stats['abs_min'])
            else:
                stats['dynamic_range_db'] = 0.0
            
            # Add original statistics for real tensors
            if tensor_type == 'real' and original_numpy is not None:
                flat_original = original_numpy.flatten()
                stats.update({
                    'mean': float(np.mean(flat_original)),
                    'std': float(np.std(flat_original)),
                    'min': float(np.min(flat_original)),
                    'max': float(np.max(flat_original)),
                })
            
            # Distribution analysis
            stats['has_outliers'] = self._detect_outliers_safe(non_zero_values)
            stats['distribution_type'] = self._analyze_distribution_safe(non_zero_values)
            
            # Complex-specific statistics
            if tensor_type == 'complex' and phase_numpy is not None:
                flat_phase = phase_numpy.flatten()
                stats['phase_stats'] = {
                    'mean': float(np.mean(flat_phase)),
                    'std': float(np.std(flat_phase)),
                    'min': float(np.min(flat_phase)),
                    'max': float(np.max(flat_phase)),
                }
            
            return stats
            
        except Exception as e:
            print(f"Error analyzing tensor {name}: {e}")
            return {
                'name': name,
                'type': 'unknown',
                'shape': list(tensor.shape) if hasattr(tensor, 'shape') else [],
                'dtype': str(tensor.dtype) if hasattr(tensor, 'dtype') else 'unknown',
                'total_elements': tensor.numel() if hasattr(tensor, 'numel') else 0,
                'error': str(e)
            }
    
    def _detect_outliers_safe(self, values: np.ndarray) -> bool:
        """Detect outliers using IQR method with safe operations."""
        try:
            q1 = np.percentile(values, 25)
            q3 = np.percentile(values, 75)
            iqr = q3 - q1
            
            if iqr <= 0:
                return False
                
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = (values < lower_bound) | (values > upper_bound)
            return float(np.sum(outliers)) / len(values) > 0.01  # More than 1% outliers
        except:
            return False
    
    def _analyze_distribution_safe(self, values: np.ndarray) -> str:
        """Analyze the distribution type of values with safe operations."""
        try:
            mean_val = np.mean(values)
            std_val = np.std(values)
            
            if mean_val <= 0 or std_val <= 0:
                return "degenerate"
            
            # Simple skewness measure
            skewness = std_val / mean_val
            
            if skewness > 2:
                return "heavy_tailed"
            elif 0.5 < skewness < 2:
                return "right_skewed" 
            elif skewness < 0.5:
                return "approximately_normal"
            else:
                return "unknown"
        except:
            return "unknown"
    
    def analyze_all_weights(self) -> Dict:
        """Perform comprehensive analysis on all weights with robust error handling."""
        weights = self.extract_weights()
        
        overall_analysis = {
            'total_layers': len(weights),
            'quantization_compatibility': {
                'int8': {'compatible_layers': 0, 'total_score': 0},
                'e4m3': {'compatible_layers': 0, 'total_score': 0},
                'e5m2': {'compatible_layers': 0, 'total_score': 0}
            },
            'block_quantization_analysis': {},
            'layer_details': {},
            'summary_stats': {
                'total_parameters': 0,
                'complex_layers': 0,
                'real_layers': 0,
                'error_layers': 0,
                'overall_dynamic_range_db': 0,
                'overall_sparsity': 0
            }
        }
        
        all_values = []
        total_elements = 0
        total_non_zero = 0
        
        print(f"\n{'='*60}")
        print(f"ANALYZING {len(weights)} WEIGHT TENSORS")
        print(f"{'='*60}")
        
        for name, tensor in weights.items():
            print(f"\nAnalyzing: {name}")
            print(f"  Shape: {tensor.shape}, Dtype: {tensor.dtype}")
            
            # Compute statistics
            stats = self.analyze_tensor_statistics(tensor, name)
            
            # Skip if problematic
            if 'warning' in stats or 'error' in stats:
                if 'warning' in stats:
                    print(f"  ⚠️  {stats['warning']}")
                if 'error' in stats:
                    print(f"  ❌ {stats['error']}")
                    overall_analysis['summary_stats']['error_layers'] += 1
                continue
            
            # Analyze block quantization for different configurations
            block_analyses = {}
            for block_config in BLOCK_CONFIGS.keys():
                block_analysis = self.analyze_block_quantization(tensor, block_config)
                block_analyses[block_config] = block_analysis
            
            # Evaluate compatibility with all quantization formats
            int8_compat = self.evaluate_quantization_compatibility(stats, 'int8', block_analyses.get('block_128'))
            e4m3_compat = self.evaluate_quantization_compatibility(stats, 'e4m3', block_analyses.get('block_128'))
            e5m2_compat = self.evaluate_quantization_compatibility(stats, 'e5m2', block_analyses.get('block_128'))
            
            # Store results
            overall_analysis['layer_details'][name] = {
                'stats': stats,
                'quantization_int8': int8_compat,
                'quantization_e4m3': e4m3_compat,
                'quantization_e5m2': e5m2_compat,
                'block_analyses': block_analyses
            }
            
            # Update counters
            if int8_compat['compatibility_score'] >= 70:
                overall_analysis['quantization_compatibility']['int8']['compatible_layers'] += 1
            if e4m3_compat['compatibility_score'] >= 70:
                overall_analysis['quantization_compatibility']['e4m3']['compatible_layers'] += 1
            if e5m2_compat['compatibility_score'] >= 70:
                overall_analysis['quantization_compatibility']['e5m2']['compatible_layers'] += 1
                
            overall_analysis['quantization_compatibility']['int8']['total_score'] += int8_compat['compatibility_score']
            overall_analysis['quantization_compatibility']['e4m3']['total_score'] += e4m3_compat['compatibility_score']
            overall_analysis['quantization_compatibility']['e5m2']['total_score'] += e5m2_compat['compatibility_score']
            
            # Collect for overall statistics
            try:
                safe_values = self.safe_tensor_to_numpy(tensor).flatten()
                all_values.extend(safe_values.tolist())
                total_elements += stats['total_elements']
                total_non_zero += stats['non_zero_elements']
            except:
                print(f"    Warning: Could not collect values for overall statistics")
            
            if stats['type'] == 'complex':
                overall_analysis['summary_stats']['complex_layers'] += 1
            else:
                overall_analysis['summary_stats']['real_layers'] += 1
            
            # Print brief summary
            print(f"  Range: [{stats['abs_min']:.2e}, {stats['abs_max']:.2e}]")
            print(f"  Dynamic Range: {stats['dynamic_range_db']:.1f} dB")
            print(f"  INT8 Score: {int8_compat['compatibility_score']}/100")
            print(f"  E4M3 Score: {e4m3_compat['compatibility_score']}/100")
            print(f"  E5M2 Score: {e5m2_compat['compatibility_score']}/100")
            
            # Show block quantization benefit
            best_block_config = None
            best_improvement = 0
            for config_name, block_analysis in block_analyses.items():
                if 'avg_dynamic_range_db' in block_analysis:
                    improvement = stats['dynamic_range_db'] - block_analysis['avg_dynamic_range_db']
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_block_config = config_name
            
            if best_block_config and best_improvement > 5:
                print(f"  Best Block Config: {best_block_config} (-{best_improvement:.1f} dB)")
            
            # Show complex-specific info
            if stats['type'] == 'complex':
                print(f"  Type: Complex tensor")
                if 'phase_stats' in stats:
                    phase_range = stats['phase_stats']['max'] - stats['phase_stats']['min']
                    print(f"  Phase Range: {phase_range:.2f} radians")
        
        # Analyze block quantization benefits across all layers
        self._analyze_global_block_benefits(overall_analysis)
        
        # Compute overall statistics
        if all_values:
            try:
                all_array = np.array(all_values, dtype=np.float32)
                non_zero_mask = all_array > 1e-15
                non_zero_values = all_array[non_zero_mask]
                
                if len(non_zero_values) > 0:
                    overall_analysis['summary_stats'].update({
                        'total_parameters': total_elements,
                        'overall_sparsity': 1 - (total_non_zero / total_elements) if total_elements > 0 else 0,
                        'overall_dynamic_range_db': 20 * np.log10(
                            float(np.max(non_zero_values) / np.max([np.min(non_zero_values), 1e-20]))
                        ),
                        'global_min': float(np.min(non_zero_values)),
                        'global_max': float(np.max(non_zero_values)),
                        'global_median': float(np.median(non_zero_values))
                    })
            except Exception as e:
                print(f"Warning: Could not compute overall statistics: {e}")
        
        # Calculate average compatibility scores
        num_layers = len(overall_analysis['layer_details'])
        if num_layers > 0:
            for format_name in ['int8', 'e4m3', 'e5m2']:
                overall_analysis['quantization_compatibility'][format_name]['avg_score'] = (
                    overall_analysis['quantization_compatibility'][format_name]['total_score'] / num_layers
                )
        
        return overall_analysis
    
    def _analyze_global_block_benefits(self, analysis: Dict):
        """Analyze global benefits of block quantization across all layers."""
        block_benefits = defaultdict(list)
        
        for layer_name, layer_details in analysis['layer_details'].items():
            base_dr = layer_details['stats']['dynamic_range_db']
            
            for block_config, block_analysis in layer_details['block_analyses'].items():
                if 'avg_dynamic_range_db' in block_analysis:
                    improvement = base_dr - block_analysis['avg_dynamic_range_db']
                    block_benefits[block_config].append({
                        'layer': layer_name,
                        'improvement_db': improvement,
                        'original_dr': base_dr,
                        'block_dr': block_analysis['avg_dynamic_range_db']
                    })
        
        # Store global block analysis
        analysis['block_quantization_analysis'] = {}
        for config_name, benefits in block_benefits.items():
            if benefits:
                improvements = [b['improvement_db'] for b in benefits]
                analysis['block_quantization_analysis'][config_name] = {
                    'avg_improvement_db': float(np.mean(improvements)),
                    'max_improvement_db': float(np.max(improvements)),
                    'min_improvement_db': float(np.min(improvements)),
                    'std_improvement_db': float(np.std(improvements)),
                    'layers_with_benefit': sum(1 for imp in improvements if imp > 5),
                    'total_layers': len(improvements),
                    'config_description': BLOCK_CONFIGS[config_name]['description']
                }
    
    def generate_enhanced_report(self, analysis: Dict, output_path: Optional[str] = None):
        """Generate a comprehensive analysis report with INT8 and block quantization."""
        report_lines = []
        
        # Header
        report_lines.extend([
            "="*80,
            "ENHANCED BCCTN MODEL WEIGHT ANALYSIS",
            "Multi-Format Quantization Compatibility (INT8 + FP8) with Block Analysis",
            "="*80,
            f"Model Path: {self.model_path}",
            f"Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ])
        
        # Summary Statistics
        summary = analysis['summary_stats']
        report_lines.extend([
            "SUMMARY STATISTICS:",
            "-" * 40,
            f"Total Parameters: {summary['total_parameters']:,}",
            f"Total Layers Analyzed: {analysis['total_layers']}",
            f"  - Real-valued layers: {summary['real_layers']}",
            f"  - Complex-valued layers: {summary['complex_layers']}",
            f"  - Error layers (skipped): {summary['error_layers']}",
            f"Overall Sparsity: {summary.get('overall_sparsity', 0):.2%}",
            f"Global Dynamic Range: {summary.get('overall_dynamic_range_db', 0):.1f} dB",
            f"Global Value Range: [{summary.get('global_min', 0):.2e}, {summary.get('global_max', 0):.2e}]",
            ""
        ])
        
        # Multi-Format Compatibility Summary
        report_lines.extend([
            "QUANTIZATION FORMAT COMPATIBILITY SUMMARY:",
            "-" * 50
        ])
        
        for format_name in ['int8', 'e4m3', 'e5m2']:
            compat = analysis['quantization_compatibility'][format_name]
            format_info = QUANTIZATION_FORMATS[format_name]
            
            report_lines.extend([
                f"{format_info['name']}:",
                f"  Use Case: {format_info['use_case']}",
                f"  Compatible layers: {compat['compatible_layers']}/{len(analysis['layer_details'])} "
                f"({compat['compatible_layers']/max(1,len(analysis['layer_details']))*100:.1f}%)",
                f"  Average compatibility score: {compat.get('avg_score', 0):.1f}/100",
                ""
            ])
        
        # Block Quantization Analysis
        if 'block_quantization_analysis' in analysis:
            report_lines.extend([
                "BLOCK QUANTIZATION BENEFITS ANALYSIS:",
                "-" * 45
            ])
            
            best_config = None
            best_improvement = 0
            
            for config_name, block_stats in analysis['block_quantization_analysis'].items():
                if 'avg_improvement_db' in block_stats:
                    avg_imp = block_stats['avg_improvement_db']
                    max_imp = block_stats['max_improvement_db']
                    benefit_layers = block_stats['layers_with_benefit']
                    total_layers = block_stats['total_layers']
                    
                    report_lines.extend([
                        f"{config_name.upper()} ({block_stats['config_description']}):",
                        f"  Average Dynamic Range Reduction: {avg_imp:.1f} dB",
                        f"  Maximum Reduction: {max_imp:.1f} dB",
                        f"  Layers with Significant Benefit (>5dB): {benefit_layers}/{total_layers} "
                        f"({benefit_layers/max(1,total_layers)*100:.1f}%)",
                        ""
                    ])
                    
                    if avg_imp > best_improvement:
                        best_improvement = avg_imp
                        best_config = config_name
            
            if best_config:
                report_lines.extend([
                    f"RECOMMENDED BLOCK CONFIGURATION: {best_config.upper()}",
                    f"  Average improvement: {best_improvement:.1f} dB dynamic range reduction",
                    ""
                ])
        
        # Complex layer analysis
        complex_layers = [name for name, details in analysis['layer_details'].items() 
                         if details['stats']['type'] == 'complex']
        if complex_layers:
            report_lines.extend([
                "COMPLEX LAYER ANALYSIS:",
                "-" * 40,
                f"Found {len(complex_layers)} complex-valued layers:",
            ])
            for name in complex_layers[:10]:  # Show first 10
                details = analysis['layer_details'][name]
                stats = details['stats']
                report_lines.append(f"  - {name}: {stats['shape']}, Range: [{stats['abs_min']:.2e}, {stats['abs_max']:.2e}]")
            
            if len(complex_layers) > 10:
                report_lines.append(f"  ... and {len(complex_layers) - 10} more")
            report_lines.append("")
        
        # Multi-format comparison for problematic layers
        problematic_layers = []
        for name, details in analysis['layer_details'].items():
            int8_score = details['quantization_int8']['compatibility_score']
            e4m3_score = details['quantization_e4m3']['compatibility_score']
            e5m2_score = details['quantization_e5m2']['compatibility_score']
            min_score = min(int8_score, e4m3_score, e5m2_score)
            if min_score < 50:  # Below 50% compatibility
                problematic_layers.append((name, min_score, details, int8_score, e4m3_score, e5m2_score))
        
        if problematic_layers:
            problematic_layers.sort(key=lambda x: x[1])  # Sort by minimum score
            report_lines.extend([
                "PROBLEMATIC LAYERS (Min Score < 50):",
                "-" * 45
            ])
            
            for name, min_score, details, int8_score, e4m3_score, e5m2_score in problematic_layers[:10]:
                stats = details['stats']
                report_lines.extend([
                    f"Layer: {name}",
                    f"  Scores: INT8={int8_score:.1f}, E4M3={e4m3_score:.1f}, E5M2={e5m2_score:.1f}",
                    f"  Range: [{stats['abs_min']:.2e}, {stats['abs_max']:.2e}]",
                    f"  Dynamic Range: {stats['dynamic_range_db']:.1f} dB",
                    f"  Shape: {stats['shape']}, Type: {stats['type']}",
                    ""
                ])
        
        # Enhanced Recommendations
        report_lines.extend([
            "ENHANCED QUANTIZATION RECOMMENDATIONS:",
            "-" * 50
        ])
        
        # Get average scores
        int8_avg = analysis['quantization_compatibility']['int8'].get('avg_score', 0)
        e4m3_avg = analysis['quantization_compatibility']['e4m3'].get('avg_score', 0)
        e5m2_avg = analysis['quantization_compatibility']['e5m2'].get('avg_score', 0)
        
        # BCCTN-specific recommendations
        if summary['complex_layers'] > summary['real_layers']:
            report_lines.extend([
                "🧠 BCCTN Model Detected - Complex-dominant architecture:",
                "   Special considerations for quantization:",
                "   - Complex numbers require magnitude/phase or real/imaginary separation",
                "   - INT8: Separate quantization for real/imaginary components",
                "   - FP8: Consider magnitude-only approximation for inference",
                "   - Block quantization especially beneficial for complex layers",
                ""
            ])
        
        # Format recommendations based on scores
        best_format = max([('INT8', int8_avg), ('E4M3', e4m3_avg), ('E5M2', e5m2_avg)], key=lambda x: x[1])
        
        report_lines.extend([
            f"PRIMARY RECOMMENDATION: {best_format[0]} (Score: {best_format[1]:.1f}/100)",
            ""
        ])
        
        if int8_avg >= 70:
            report_lines.extend([
                "✅ INT8 Quantization Recommended:",
                "   - Most widely supported format",
                "   - Good balance of compression and compatibility",
                "   - Consider per-channel or block quantization for best results",
                ""
            ])
        
        if e4m3_avg >= 70:
            report_lines.extend([
                "✅ FP8-E4M3 Quantization Suitable:",
                "   - Excellent for forward pass and inference",
                "   - Better precision than INT8 for gradual values",
                "   - Requires H100/H200 GPU support",
                ""
            ])
        
        if e5m2_avg >= 70:
            report_lines.extend([
                "✅ FP8-E5M2 Quantization Suitable:",
                "   - Best for gradients and optimizer states",
                "   - Larger dynamic range than E4M3",
                "   - Requires H100/H200 GPU support",
                ""
            ])
        
        # Block quantization recommendations
        if 'block_quantization_analysis' in analysis and best_config:
            block_stats = analysis['block_quantization_analysis'][best_config]
            if block_stats['avg_improvement_db'] > 10:
                report_lines.extend([
                    f"🔧 STRONGLY RECOMMEND Block Quantization ({best_config}):",
                    f"   - Reduces dynamic range by {block_stats['avg_improvement_db']:.1f} dB on average",
                    f"   - {block_stats['layers_with_benefit']} layers show significant benefit",
                    f"   - {block_stats['config_description']}",
                    ""
                ])
        
        # Implementation strategy
        report_lines.extend([
            "IMPLEMENTATION STRATEGY:",
            "-" * 30,
            "1. START WITH: Block quantization analysis to identify optimal block sizes",
            "2. FOR COMPLEX LAYERS: Separate real/imaginary or magnitude/phase quantization",
            "3. FOR HIGH DYNAMIC RANGE LAYERS: Use per-channel or block quantization",
            "4. MIXED PRECISION: Use different formats for different layer types",
            "5. VALIDATION: Implement quantization-aware training if accuracy drops",
            ""
        ])
        
        # Performance expectations
        if max(int8_avg, e4m3_avg, e5m2_avg) >= 80:
            report_lines.append("📈 EXPECTED OUTCOME: Minimal accuracy loss with significant speedup")
        elif max(int8_avg, e4m3_avg, e5m2_avg) >= 60:
            report_lines.append("📈 EXPECTED OUTCOME: Good compression with careful implementation")
        else:
            report_lines.append("📈 EXPECTED OUTCOME: Challenging - requires advanced techniques")
        
        # Write report
        report_text = "\n".join(report_lines)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
            print(f"\n✅ Enhanced report saved to: {output_path}")
        
        print("\n" + report_text)
        return report_text
    
    def create_enhanced_visualizations(self, analysis: Dict, output_dir: str = "./enhanced_analysis_plots"):
        """Create enhanced visualization plots including INT8 and block quantization analysis."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Set style for better looking plots
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Configure matplotlib for high-quality output
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.linewidth'] = 1.2
        plt.rcParams['lines.linewidth'] = 2
        plt.rcParams['patch.linewidth'] = 1.2
        plt.rcParams['svg.fonttype'] = 'none'
        plt.rcParams['pdf.fonttype'] = 42
        
        try:
            # 1. Multi-format compatibility comparison
            self._plot_multi_format_compatibility(analysis, output_dir)
            
            # 2. Block quantization benefits analysis
            self._plot_block_quantization_benefits(analysis, output_dir)
            
            # 3. Enhanced dynamic range distribution with all formats
            self._plot_enhanced_dynamic_range_distribution(analysis, output_dir)
            
            # 4. Complex vs Real layer analysis for all formats
            self._plot_enhanced_complex_real_comparison(analysis, output_dir)
            
            # 5. Create comprehensive summary dashboard
            self._plot_enhanced_summary_dashboard(analysis, output_dir)
            
            # 6. Layer-wise detailed analysis
            self._plot_layer_wise_detailed_analysis(analysis, output_dir)
            
            print(f"\n✅ Enhanced visualizations saved to: {output_dir}")
            print(f"📊 Generated formats: PNG (high-res), SVG (vector), PDF (vector)")
            
        except Exception as e:
            print(f"Warning: Could not create all visualizations: {e}")
    
    def _save_plot(self, output_dir: Path, filename: str, dpi: int = 300):
        """Save plot in multiple formats for different use cases."""
        base_name = filename.replace('.png', '')
        
        # High-resolution PNG for presentations
        plt.savefig(output_dir / f'{base_name}.png', dpi=dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        
        # SVG for vector editing
        plt.savefig(output_dir / f'{base_name}.svg', bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        
        # PDF for publications
        plt.savefig(output_dir / f'{base_name}.pdf', bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        
        plt.close()
    
    def _plot_multi_format_compatibility(self, analysis: Dict, output_dir: Path):
        """Plot compatibility scores for INT8, E4M3, and E5M2 formats."""
        layer_names = []
        int8_scores = []
        e4m3_scores = []
        e5m2_scores = []
        layer_types = []
        
        for name, details in analysis['layer_details'].items():
            # Truncate long layer names
            short_name = name.split('.')[-1] if '.' in name else name
            if len(short_name) > 12:
                short_name = short_name[:9] + "..."
            layer_names.append(short_name)
            
            int8_scores.append(details['quantization_int8']['compatibility_score'])
            e4m3_scores.append(details['quantization_e4m3']['compatibility_score'])
            e5m2_scores.append(details['quantization_e5m2']['compatibility_score'])
            layer_types.append(details['stats']['type'])
        
        fig, ax = plt.subplots(figsize=(16, 10))
        
        x = np.arange(len(layer_names))
        width = 0.25
        
        # Color coding with distinct colors for each format
        colors_int8 = ['#FF6B6B' if t == 'complex' else '#4ECDC4' for t in layer_types]   # Red/Teal
        colors_e4m3 = ['#C44569' if t == 'complex' else '#3742FA' for t in layer_types]  # Dark red/Blue  
        colors_e5m2 = ['#8E44AD' if t == 'complex' else '#2ECC71' for t in layer_types]  # Purple/Green
        
        bars1 = ax.bar(x - width, int8_scores, width, label='INT8', 
                      color=colors_int8, alpha=0.8, edgecolor='black', linewidth=0.8)
        bars2 = ax.bar(x, e4m3_scores, width, label='FP8-E4M3', 
                      color=colors_e4m3, alpha=0.8, edgecolor='black', linewidth=0.8)
        bars3 = ax.bar(x + width, e5m2_scores, width, label='FP8-E5M2', 
                      color=colors_e5m2, alpha=0.8, edgecolor='black', linewidth=0.8)
        
        # Styling
        ax.set_xlabel('Network Layers', fontsize=14, fontweight='bold')
        ax.set_ylabel('Quantization Compatibility Score', fontsize=14, fontweight='bold')
        ax.set_title('Multi-Format Quantization Compatibility Analysis\n(Warm colors = Complex layers, Cool colors = Real layers)', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(layer_names, rotation=45, ha='right', fontsize=10)
        ax.legend(fontsize=12, loc='upper right')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim(0, 105)
        
        # Add threshold lines
        ax.axhline(y=80, color='#2ECC71', linestyle='--', alpha=0.8, linewidth=2)
        ax.axhline(y=60, color='#F39C12', linestyle='--', alpha=0.8, linewidth=2)
        ax.axhline(y=40, color='#E74C3C', linestyle='--', alpha=0.8, linewidth=2)
        
        # Add threshold annotations
        ax.text(len(layer_names)*0.02, 82, 'Excellent (≥80)', color='#2ECC71', fontweight='bold', fontsize=10)
        ax.text(len(layer_names)*0.02, 62, 'Good (≥60)', color='#F39C12', fontweight='bold', fontsize=10)
        ax.text(len(layer_names)*0.02, 42, 'Acceptable (≥40)', color='#E74C3C', fontweight='bold', fontsize=10)
        
        plt.tight_layout()
        self._save_plot(output_dir, 'multi_format_compatibility')
    
    def _plot_block_quantization_benefits(self, analysis: Dict, output_dir: Path):
        """Plot block quantization benefits across different configurations."""
        if 'block_quantization_analysis' not in analysis:
            return
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        block_configs = []
        avg_improvements = []
        max_improvements = []
        benefit_ratios = []
        config_descriptions = []
        
        for config_name, block_stats in analysis['block_quantization_analysis'].items():
            if 'avg_improvement_db' in block_stats:
                block_configs.append(config_name.upper())
                avg_improvements.append(block_stats['avg_improvement_db'])
                max_improvements.append(block_stats['max_improvement_db'])
                benefit_ratios.append(block_stats['layers_with_benefit'] / max(1, block_stats['total_layers']) * 100)
                config_descriptions.append(block_stats['config_description'])
        
        if not block_configs:
            return
        
        # 1. Average improvement by block configuration
        bars1 = ax1.bar(block_configs, avg_improvements, alpha=0.8, 
                       color=['#3498DB', '#E74C3C', '#2ECC71', '#F39C12', '#9B59B6', '#1ABC9C'][:len(block_configs)],
                       edgecolor='black', linewidth=1)
        
        ax1.set_ylabel('Average Dynamic Range Reduction (dB)', fontsize=12, fontweight='bold')
        ax1.set_title('Block Quantization Benefits - Average Improvement', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        # Add value labels
        for bar, value in zip(bars1, avg_improvements):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{value:.1f}dB', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 2. Maximum improvement potential
        bars2 = ax2.bar(block_configs, max_improvements, alpha=0.8,
                       color=['#3498DB', '#E74C3C', '#2ECC71', '#F39C12', '#9B59B6', '#1ABC9C'][:len(block_configs)],
                       edgecolor='black', linewidth=1)
        
        ax2.set_ylabel('Maximum Dynamic Range Reduction (dB)', fontsize=12, fontweight='bold')
        ax2.set_title('Block Quantization Benefits - Maximum Potential', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')
        
        # Add value labels
        for bar, value in zip(bars2, max_improvements):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{value:.1f}dB', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 3. Percentage of layers benefiting
        bars3 = ax3.bar(block_configs, benefit_ratios, alpha=0.8,
                       color=['#3498DB', '#E74C3C', '#2ECC71', '#F39C12', '#9B59B6', '#1ABC9C'][:len(block_configs)],
                       edgecolor='black', linewidth=1)
        
        ax3.set_ylabel('Layers with Significant Benefit (%)', fontsize=12, fontweight='bold')
        ax3.set_title('Block Quantization Benefits - Coverage', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3, linestyle='--')
        ax3.set_ylim(0, 100)
        
        # Add value labels
        for bar, value in zip(bars3, benefit_ratios):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{value:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 4. Configuration recommendations
        ax4.axis('off')
        
        # Find best configuration
        best_config_idx = np.argmax(avg_improvements)
        best_config = block_configs[best_config_idx]
        best_improvement = avg_improvements[best_config_idx]
        best_coverage = benefit_ratios[best_config_idx]
        
        recommendation_text = f"""
BLOCK QUANTIZATION RECOMMENDATIONS

BEST OVERALL: {best_config}
• Average improvement: {best_improvement:.1f} dB
• Coverage: {best_coverage:.1f}% of layers benefit
• Description: {config_descriptions[best_config_idx]}

CONFIGURATION GUIDE:
• PER_TENSOR: Simplest, but limited for high dynamic range
• PER_CHANNEL: Good balance for conv layers
• BLOCK_64/128: Best for very high dynamic range layers
• BLOCK_256/512: Moderate improvement, easier implementation

IMPLEMENTATION PRIORITY:
1. Try {best_config} first
2. Fall back to PER_CHANNEL for compatibility
3. Use PER_TENSOR only if others fail
        """
        
        ax4.text(0.05, 0.95, recommendation_text, transform=ax4.transAxes, 
                fontsize=11, verticalalignment='top', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.3))
        
        plt.suptitle('Block Quantization Benefits Analysis for BCCTN Model', 
                    fontsize=16, fontweight='bold', y=0.95)
        plt.tight_layout()
        self._save_plot(output_dir, 'block_quantization_benefits')
    
    def _plot_enhanced_dynamic_range_distribution(self, analysis: Dict, output_dir: Path):
        """Plot enhanced dynamic range distribution with quantization format limits."""
        dynamic_ranges = []
        layer_types = []
        layer_names = []
        
        for name, details in analysis['layer_details'].items():
            stats = details['stats']
            dynamic_ranges.append(stats['dynamic_range_db'])
            layer_types.append(stats['type'])
            layer_names.append(name.split('.')[-1])
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Histogram with all format limits
        complex_ranges = [dr for dr, lt in zip(dynamic_ranges, layer_types) if lt == 'complex']
        real_ranges = [dr for dr, lt in zip(dynamic_ranges, layer_types) if lt == 'real']
        
        if complex_ranges:
            ax1.hist(complex_ranges, bins=20, alpha=0.7, label=f'Complex Layers (n={len(complex_ranges)})', 
                    color='#E74C3C', edgecolor='black', linewidth=1)
        if real_ranges:
            ax1.hist(real_ranges, bins=20, alpha=0.7, label=f'Real Layers (n={len(real_ranges)})', 
                    color='#3498DB', edgecolor='black', linewidth=1)
        
        # Add format limits
        ax1.axvline(42, color='#2ECC71', linestyle='--', linewidth=3, alpha=0.8, label='INT8 Effective (~42dB)')
        ax1.axvline(QUANTIZATION_FORMATS['e4m3']['dynamic_range_db'], color='#F39C12', 
                   linestyle='--', linewidth=3, alpha=0.8, label=f"FP8-E4M3 ({QUANTIZATION_FORMATS['e4m3']['dynamic_range_db']:.0f}dB)")
        ax1.axvline(QUANTIZATION_FORMATS['e5m2']['dynamic_range_db'], color='#9B59B6', 
                   linestyle='--', linewidth=3, alpha=0.8, label=f"FP8-E5M2 ({QUANTIZATION_FORMATS['e5m2']['dynamic_range_db']:.0f}dB)")
        
        ax1.set_xlabel('Dynamic Range (dB)', fontsize=13, fontweight='bold')
        ax1.set_ylabel('Number of Layers', fontsize=13, fontweight='bold')
        ax1.set_title('Dynamic Range Distribution with Format Limits', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.legend(fontsize=10, loc='upper right')
        
        # 2. Cumulative distribution
        all_ranges_sorted = sorted(dynamic_ranges)
        cumulative_pct = np.arange(1, len(all_ranges_sorted) + 1) / len(all_ranges_sorted) * 100
        
        ax2.plot(all_ranges_sorted, cumulative_pct, linewidth=3, color='#34495E', label='Cumulative Distribution')
        ax2.axvline(42, color='#2ECC71', linestyle='--', linewidth=2, alpha=0.8)
        ax2.axvline(QUANTIZATION_FORMATS['e4m3']['dynamic_range_db'], color='#F39C12', 
                   linestyle='--', linewidth=2, alpha=0.8)
        ax2.axvline(QUANTIZATION_FORMATS['e5m2']['dynamic_range_db'], color='#9B59B6', 
                   linestyle='--', linewidth=2, alpha=0.8)
        
        # Add percentage annotations
        int8_pct = sum(1 for dr in dynamic_ranges if dr <= 42) / len(dynamic_ranges) * 100
        e4m3_pct = sum(1 for dr in dynamic_ranges if dr <= QUANTIZATION_FORMATS['e4m3']['dynamic_range_db']) / len(dynamic_ranges) * 100
        e5m2_pct = sum(1 for dr in dynamic_ranges if dr <= QUANTIZATION_FORMATS['e5m2']['dynamic_range_db']) / len(dynamic_ranges) * 100
        
        ax2.text(42, int8_pct + 5, f'INT8: {int8_pct:.1f}%', color='#2ECC71', fontweight='bold', fontsize=10)
        ax2.text(QUANTIZATION_FORMATS['e4m3']['dynamic_range_db'], e4m3_pct + 5, f'E4M3: {e4m3_pct:.1f}%', 
                color='#F39C12', fontweight='bold', fontsize=10)
        ax2.text(QUANTIZATION_FORMATS['e5m2']['dynamic_range_db'], e5m2_pct + 5, f'E5M2: {e5m2_pct:.1f}%', 
                color='#9B59B6', fontweight='bold', fontsize=10)
        
        ax2.set_xlabel('Dynamic Range (dB)', fontsize=13, fontweight='bold')
        ax2.set_ylabel('Cumulative Percentage (%)', fontsize=13, fontweight='bold')
        ax2.set_title('Cumulative Distribution with Format Coverage', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_ylim(0, 105)
        
        # 3. Box plot by layer type with format comparisons
        if complex_ranges and real_ranges:
            data_by_type = [real_ranges, complex_ranges]
            labels = ['Real\nLayers', 'Complex\nLayers']
            colors = ['#3498DB', '#E74C3C']
        elif complex_ranges:
            data_by_type = [complex_ranges]
            labels = ['Complex\nLayers']
            colors = ['#E74C3C']
        elif real_ranges:
            data_by_type = [real_ranges]
            labels = ['Real\nLayers']
            colors = ['#3498DB']
        else:
            data_by_type = [[0]]
            labels = ['No Data']
            colors = ['#95A5A6']
        
        bp = ax3.boxplot(data_by_type, labels=labels, patch_artist=True, 
                        boxprops=dict(linewidth=2), whiskerprops=dict(linewidth=2),
                        capprops=dict(linewidth=2), medianprops=dict(linewidth=2, color='white'))
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Add format limits as horizontal lines
        ax3.axhline(42, color='#2ECC71', linestyle='--', linewidth=2, alpha=0.8, label='INT8')
        ax3.axhline(QUANTIZATION_FORMATS['e4m3']['dynamic_range_db'], color='#F39C12', 
                   linestyle='--', linewidth=2, alpha=0.8, label='E4M3')
        ax3.axhline(QUANTIZATION_FORMATS['e5m2']['dynamic_range_db'], color='#9B59B6', 
                   linestyle='--', linewidth=2, alpha=0.8, label='E5M2')
        
        ax3.set_ylabel('Dynamic Range (dB)', fontsize=13, fontweight='bold')
        ax3.set_title('Dynamic Range by Layer Type vs Format Limits', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3, linestyle='--')
        ax3.legend(fontsize=10)
        
        # 4. Format suitability summary
        ax4.axis('off')
        
        format_suitability = {
            'INT8': int8_pct,
            'FP8-E4M3': e4m3_pct,
            'FP8-E5M2': e5m2_pct
        }
        
        summary_text = f"""
DYNAMIC RANGE ANALYSIS SUMMARY

LAYER COVERAGE BY FORMAT:
• INT8 (≤42dB): {int8_pct:.1f}% of layers
• FP8-E4M3 (≤{QUANTIZATION_FORMATS['e4m3']['dynamic_range_db']:.0f}dB): {e4m3_pct:.1f}% of layers  
• FP8-E5M2 (≤{QUANTIZATION_FORMATS['e5m2']['dynamic_range_db']:.0f}dB): {e5m2_pct:.1f}% of layers

STATISTICS:
• Total layers: {len(dynamic_ranges)}
• Median range: {np.median(dynamic_ranges):.1f} dB
• Max range: {np.max(dynamic_ranges):.1f} dB
• Complex layers avg: {np.mean(complex_ranges) if complex_ranges else 0:.1f} dB
• Real layers avg: {np.mean(real_ranges) if real_ranges else 0:.1f} dB

RECOMMENDATIONS:
• {max(format_suitability, key=format_suitability.get)} has best coverage
• Block quantization recommended for layers >60dB
• Mixed precision may be optimal strategy
        """
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, 
                fontsize=11, verticalalignment='top', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.3))
        
        plt.suptitle('Enhanced Dynamic Range Analysis for Multi-Format Quantization', 
                    fontsize=16, fontweight='bold', y=0.95)
        plt.tight_layout()
        self._save_plot(output_dir, 'enhanced_dynamic_range_distribution')
    
    def _plot_enhanced_complex_real_comparison(self, analysis: Dict, output_dir: Path):
        """Create enhanced comparison between complex and real layers for all formats."""
        complex_layers = {}
        real_layers = {}
        
        for name, details in analysis['layer_details'].items():
            stats = details['stats']
            if stats['type'] == 'complex':
                complex_layers[name] = details
            else:
                real_layers[name] = details
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Multi-format compatibility comparison
        if complex_layers and real_layers:
            # Extract scores for all formats
            complex_int8 = [details['quantization_int8']['compatibility_score'] for details in complex_layers.values()]
            complex_e4m3 = [details['quantization_e4m3']['compatibility_score'] for details in complex_layers.values()]
            complex_e5m2 = [details['quantization_e5m2']['compatibility_score'] for details in complex_layers.values()]
            
            real_int8 = [details['quantization_int8']['compatibility_score'] for details in real_layers.values()]
            real_e4m3 = [details['quantization_e4m3']['compatibility_score'] for details in real_layers.values()]
            real_e5m2 = [details['quantization_e5m2']['compatibility_score'] for details in real_layers.values()]
            
            # Create grouped bar chart
            x = np.arange(3)  # INT8, E4M3, E5M2
            width = 0.35
            
            complex_avgs = [np.mean(complex_int8), np.mean(complex_e4m3), np.mean(complex_e5m2)]
            real_avgs = [np.mean(real_int8), np.mean(real_e4m3), np.mean(real_e5m2)]
            complex_stds = [np.std(complex_int8), np.std(complex_e4m3), np.std(complex_e5m2)]
            real_stds = [np.std(real_int8), np.std(real_e4m3), np.std(real_e5m2)]
            
            bars1 = ax1.bar(x - width/2, complex_avgs, width, yerr=complex_stds, 
                           label=f'Complex Layers (n={len(complex_layers)})', 
                           color='#E74C3C', alpha=0.8, capsize=5, edgecolor='black', linewidth=1)
            bars2 = ax1.bar(x + width/2, real_avgs, width, yerr=real_stds,
                           label=f'Real Layers (n={len(real_layers)})', 
                           color='#3498DB', alpha=0.8, capsize=5, edgecolor='black', linewidth=1)
            
            ax1.set_xlabel('Quantization Format', fontsize=13, fontweight='bold')
            ax1.set_ylabel('Average Compatibility Score', fontsize=13, fontweight='bold')
            ax1.set_title('Complex vs Real Layer Compatibility\n(All Quantization Formats)', fontsize=14, fontweight='bold')
            ax1.set_xticks(x)
            ax1.set_xticklabels(['INT8', 'FP8-E4M3', 'FP8-E5M2'])
            ax1.legend(fontsize=11)
            ax1.grid(True, alpha=0.3, linestyle='--')
            ax1.set_ylim(0, 100)
            
            # Add value labels
            for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
                ax1.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + complex_stds[i] + 2,
                        f'{complex_avgs[i]:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
                ax1.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + real_stds[i] + 2,
                        f'{real_avgs[i]:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 2. Format preference scatter plot
        if complex_layers and real_layers:
            # Create 3D-style scatter showing all three formats
            fig2d = plt.figure(figsize=(10, 8))
            ax2_3d = fig2d.add_subplot(111, projection='3d')
            
            # Complex layers
            ax2_3d.scatter(complex_int8, complex_e4m3, complex_e5m2, 
                          c='red', s=60, alpha=0.7, label=f'Complex ({len(complex_layers)})')
            # Real layers  
            ax2_3d.scatter(real_int8, real_e4m3, real_e5m2,
                          c='blue', s=60, alpha=0.7, label=f'Real ({len(real_layers)})')
            
            ax2_3d.set_xlabel('INT8 Score')
            ax2_3d.set_ylabel('E4M3 Score')
            ax2_3d.set_zlabel('E5M2 Score')
            ax2_3d.set_title('3D Format Preference Analysis')
            ax2_3d.legend()
            
            plt.savefig(output_dir / 'format_preference_3d.png', dpi=300, bbox_inches='tight')
            plt.close(fig2d)
            
            # 2D version for main plot - best two formats
            best_formats = ['INT8', 'E4M3', 'E5M2']
            format_avgs = [(np.mean(complex_int8) + np.mean(real_int8))/2,
                          (np.mean(complex_e4m3) + np.mean(real_e4m3))/2,
                          (np.mean(complex_e5m2) + np.mean(real_e5m2))/2]
            
            sorted_formats = sorted(zip(best_formats, format_avgs, 
                                      [(complex_int8, real_int8), (complex_e4m3, real_e4m3), (complex_e5m2, real_e5m2)]), 
                                  key=lambda x: x[1], reverse=True)
            
            format1_name, _, (complex1, real1) = sorted_formats[0]
            format2_name, _, (complex2, real2) = sorted_formats[1]
            
            ax2.scatter(complex1, complex2, alpha=0.7, s=60, color='#E74C3C', 
                       label=f'Complex ({len(complex1)})', edgecolor='black', linewidth=1)
            ax2.scatter(real1, real2, alpha=0.7, s=60, color='#3498DB', 
                       label=f'Real ({len(real1)})', edgecolor='black', linewidth=1)
            
            ax2.plot([0, 100], [0, 100], 'gray', linestyle='--', alpha=0.5, linewidth=2)
            ax2.set_xlabel(f'{format1_name} Compatibility Score', fontsize=12, fontweight='bold')
            ax2.set_ylabel(f'{format2_name} Compatibility Score', fontsize=12, fontweight='bold')
            ax2.set_title(f'Best Format Preference: {format1_name} vs {format2_name}', fontsize=13, fontweight='bold')
            ax2.grid(True, alpha=0.3, linestyle='--')
            ax2.legend(fontsize=11)
            ax2.set_xlim(0, 100)
            ax2.set_ylim(0, 100)
        
        # 3. Parameter efficiency analysis
        if complex_layers and real_layers:
            complex_params = [details['stats']['total_elements'] for details in complex_layers.values()]
            real_params = [details['stats']['total_elements'] for details in real_layers.values()]
            
            # Compute parameter efficiency (score per million parameters)
            complex_efficiency = []
            real_efficiency = []
            
            for i, details in enumerate(complex_layers.values()):
                params = details['stats']['total_elements'] / 1e6  # millions
                avg_score = (details['quantization_int8']['compatibility_score'] + 
                           details['quantization_e4m3']['compatibility_score'] + 
                           details['quantization_e5m2']['compatibility_score']) / 3
                complex_efficiency.append(avg_score / max(params, 0.001))
            
            for i, details in enumerate(real_layers.values()):
                params = details['stats']['total_elements'] / 1e6  # millions
                avg_score = (details['quantization_int8']['compatibility_score'] + 
                           details['quantization_e4m3']['compatibility_score'] + 
                           details['quantization_e5m2']['compatibility_score']) / 3
                real_efficiency.append(avg_score / max(params, 0.001))
            
            ax3.scatter(complex_params, complex_efficiency, alpha=0.7, s=60, color='#E74C3C', 
                       label=f'Complex Layers', edgecolor='black', linewidth=1)
            ax3.scatter(real_params, real_efficiency, alpha=0.7, s=60, color='#3498DB', 
                       label=f'Real Layers', edgecolor='black', linewidth=1)
            
            ax3.set_xlabel('Layer Parameters (count)', fontsize=12, fontweight='bold')
            ax3.set_ylabel('Quantization Efficiency\n(Avg Score / Million Params)', fontsize=12, fontweight='bold')
            ax3.set_title('Quantization Efficiency vs Layer Size', fontsize=13, fontweight='bold')
            ax3.set_xscale('log')
            ax3.grid(True, alpha=0.3, linestyle='--')
            ax3.legend(fontsize=11)
        
        # 4. Detailed comparison summary
        ax4.axis('off')
        
        if complex_layers and real_layers:
            # Calculate comprehensive statistics
            complex_total = len(complex_layers)
            real_total = len(real_layers)
            total_layers = complex_total + real_total
            
            # Best format for each type
            complex_best_format = max([('INT8', np.mean(complex_int8)), 
                                     ('E4M3', np.mean(complex_e4m3)), 
                                     ('E5M2', np.mean(complex_e5m2))], key=lambda x: x[1])
            real_best_format = max([('INT8', np.mean(real_int8)), 
                                  ('E4M3', np.mean(real_e4m3)), 
                                  ('E5M2', np.mean(real_e5m2))], key=lambda x: x[1])
            
            summary_text = f"""
COMPLEX vs REAL LAYER ANALYSIS

LAYER DISTRIBUTION:
• Complex layers: {complex_total} ({complex_total/total_layers*100:.1f}%)
• Real layers: {real_total} ({real_total/total_layers*100:.1f}%)

BEST FORMAT BY TYPE:
• Complex layers: {complex_best_format[0]} ({complex_best_format[1]:.1f} avg score)
• Real layers: {real_best_format[0]} ({real_best_format[1]:.1f} avg score)

QUANTIZATION STRATEGY:
• {'Complex-dominant model' if complex_total > real_total else 'Balanced model'}
• {'Separate real/imag quantization recommended' if complex_total > real_total*0.5 else 'Standard quantization suitable'}
• {'Block quantization especially beneficial' if complex_total > 0 else 'Per-channel quantization sufficient'}

IMPLEMENTATION NOTES:
• Complex layers may need special handling
• Consider magnitude-only inference for speed
• Mixed precision likely optimal approach
            """
            
            ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, 
                    fontsize=11, verticalalignment='top', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.3))
        
        plt.suptitle('Enhanced Complex vs Real Layer Analysis for Multi-Format Quantization', 
                    fontsize=16, fontweight='bold', y=0.95)
        plt.tight_layout()
        self._save_plot(output_dir, 'enhanced_complex_real_comparison')
    
    def _plot_enhanced_summary_dashboard(self, analysis: Dict, output_dir: Path):
        """Create a comprehensive summary dashboard with all quantization formats."""
        fig = plt.figure(figsize=(20, 14))
        
        # Create a more complex grid layout
        gs = fig.add_gridspec(4, 5, hspace=0.3, wspace=0.3)
        
        # Extract data
        total_layers = len(analysis['layer_details'])
        complex_layers = analysis['summary_stats']['complex_layers']
        real_layers = analysis['summary_stats']['real_layers']
        
        int8_avg = analysis['quantization_compatibility']['int8'].get('avg_score', 0)
        e4m3_avg = analysis['quantization_compatibility']['e4m3'].get('avg_score', 0)
        e5m2_avg = analysis['quantization_compatibility']['e5m2'].get('avg_score', 0)
        
        # 1. Model Overview (top-left)
        ax1 = fig.add_subplot(gs[0, :2])
        categories = ['Complex\nLayers', 'Real\nLayers']
        values = [complex_layers, real_layers]
        colors = ['#E74C3C', '#3498DB']
        
        wedges, texts, autotexts = ax1.pie(values, labels=categories, colors=colors, autopct='%1.1f%%',
                                          startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
        ax1.set_title(f'BCCTN Model Composition\n(Total: {total_layers} layers)', 
                     fontsize=14, fontweight='bold')
        
        # 2. Multi-format compatibility summary (top-right)
        ax2 = fig.add_subplot(gs[0, 2:])
        formats = ['INT8', 'FP8-E4M3', 'FP8-E5M2']
        scores = [int8_avg, e4m3_avg, e5m2_avg]
        colors_bar = ['#2ECC71', '#F39C12', '#9B59B6']
        
        bars = ax2.bar(formats, scores, color=colors_bar, alpha=0.8, edgecolor='black', linewidth=2)
        
        ax2.set_ylabel('Average Compatibility Score', fontsize=12, fontweight='bold')
        ax2.set_title('Multi-Format Quantization Compatibility', fontsize=14, fontweight='bold')
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3, linestyle='--')
        
        # Add score labels
        for bar, score in zip(bars, scores):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{score:.1f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # Add threshold lines
        ax2.axhline(y=80, color='#2ECC71', linestyle='--', linewidth=2, alpha=0.7)
        ax2.axhline(y=60, color='#F39C12', linestyle='--', linewidth=2, alpha=0.7)
        ax2.text(0.02, 82, 'Excellent', fontsize=10, fontweight='bold', color='#2ECC71')
        ax2.text(0.02, 62, 'Good', fontsize=10, fontweight='bold', color='#F39C12')
        
        # 3. Dynamic Range Analysis with all format limits (middle-left)
        ax3 = fig.add_subplot(gs[1, :3])
        
        dynamic_ranges = []
        layer_types = []
        for details in analysis['layer_details'].values():
            dynamic_ranges.append(details['stats']['dynamic_range_db'])
            layer_types.append(details['stats']['type'])
        
        complex_ranges = [dr for dr, lt in zip(dynamic_ranges, layer_types) if lt == 'complex']
        real_ranges = [dr for dr, lt in zip(dynamic_ranges, layer_types) if lt == 'real']
        
        if complex_ranges and real_ranges:
            bp = ax3.boxplot([real_ranges, complex_ranges], labels=['Real Layers', 'Complex Layers'], 
                           patch_artist=True, boxprops=dict(linewidth=2))
            bp['boxes'][0].set_facecolor('#3498DB')
            bp['boxes'][1].set_facecolor('#E74C3C')
            for box in bp['boxes']:
                box.set_alpha(0.7)
        
        # Add all format limits
        ax3.axhline(y=42, color='#2ECC71', linestyle='--', linewidth=2, alpha=0.8, label='INT8 (~42dB)')
        ax3.axhline(y=QUANTIZATION_FORMATS['e4m3']['dynamic_range_db'], color='#F39C12', 
                   linestyle='--', linewidth=2, alpha=0.8, label=f"E4M3 ({QUANTIZATION_FORMATS['e4m3']['dynamic_range_db']:.0f}dB)")
        ax3.axhline(y=QUANTIZATION_FORMATS['e5m2']['dynamic_range_db'], color='#9B59B6', 
                   linestyle='--', linewidth=2, alpha=0.8, label=f"E5M2 ({QUANTIZATION_FORMATS['e5m2']['dynamic_range_db']:.0f}dB)")
        
        ax3.set_ylabel('Dynamic Range (dB)', fontsize=12, fontweight='bold')
        ax3.set_title('Dynamic Range Distribution vs Format Limits', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3, linestyle='--')
        ax3.legend(fontsize=10)
        
        # 4. Quantization Recommendation (middle-right)
        ax4 = fig.add_subplot(gs[1, 3:])
        ax4.axis('off')
        
        # Generate recommendation based on best performing format
        best_format = max([('INT8', int8_avg), ('FP8-E4M3', e4m3_avg), ('FP8-E5M2', e5m2_avg)], key=lambda x: x[1])
        
        if complex_layers > real_layers:
            model_type = "Complex-Dominant BCCTN"
            special_note = "• Requires complex-aware quantization\n• Consider magnitude-only inference\n• Block-wise quantization recommended\n• Separate real/imaginary quantization"
        else:
            model_type = "Mixed BCCTN"
            special_note = "• Standard quantization applicable\n• Monitor complex layers separately\n• Mixed precision may be beneficial\n• Per-channel quantization recommended"
        
        # Performance assessment
        best_score = best_format[1]
        if best_score >= 80:
            recommendation = f"🎉 EXCELLENT {best_format[0]} Compatibility"
            color = '#2ECC71'
        elif best_score >= 60:
            recommendation = f"✅ GOOD {best_format[0]} Compatibility"  
            color = '#F39C12'
        else:
            recommendation = f"⚠️ CHALLENGING for {best_format[0]}"
            color = '#E74C3C'
        
        recommendation_text = f"""
{recommendation}

Model Type: {model_type}
Best Format: {best_format[0]}
Best Score: {best_score:.1f}/100

Format Rankings:
1. {best_format[0]}: {best_score:.1f}/100
2. Others: See detailed analysis

Special Considerations:
{special_note}
        """
        
        ax4.text(0.05, 0.95, recommendation_text, transform=ax4.transAxes, 
                fontsize=12, verticalalignment='top', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.2))
        
        # 5. Block quantization benefits summary (third row, left)
        ax5 = fig.add_subplot(gs[2, :2])
        
        if 'block_quantization_analysis' in analysis:
            block_configs = []
            improvements = []
            
            for config_name, block_stats in analysis['block_quantization_analysis'].items():
                if 'avg_improvement_db' in block_stats:
                    block_configs.append(config_name.upper())
                    improvements.append(block_stats['avg_improvement_db'])
            
            if block_configs:
                bars_block = ax5.bar(block_configs, improvements, 
                                   color=['#3498DB', '#E74C3C', '#2ECC71', '#F39C12', '#9B59B6'][:len(block_configs)],
                                   alpha=0.8, edgecolor='black', linewidth=1)
                
                ax5.set_ylabel('Avg Improvement (dB)', fontsize=11, fontweight='bold')
                ax5.set_title('Block Quantization Benefits', fontsize=12, fontweight='bold')
                ax5.tick_params(axis='x', rotation=45)
                ax5.grid(True, alpha=0.3, linestyle='--')
                
                # Highlight best configuration
                best_idx = np.argmax(improvements)
                bars_block[best_idx].set_edgecolor('gold')
                bars_block[best_idx].set_linewidth(3)
        
        # 6. Layer-wise compatibility heatmap (third row, right)
        ax6 = fig.add_subplot(gs[2, 2:])
        
        # Create compatibility matrix
        layer_names = [name.split('.')[-1][:8] for name in list(analysis['layer_details'].keys())[:15]]  # First 15 layers
        compatibility_matrix = []
        
        for name in list(analysis['layer_details'].keys())[:15]:
            details = analysis['layer_details'][name]
            int8_score = details['quantization_int8']['compatibility_score']
            e4m3_score = details['quantization_e4m3']['compatibility_score'] 
            e5m2_score = details['quantization_e5m2']['compatibility_score']
            compatibility_matrix.append([int8_score, e4m3_score, e5m2_score])
        
        if compatibility_matrix:
            im = ax6.imshow(compatibility_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
            ax6.set_xticks(range(3))
            ax6.set_xticklabels(['INT8', 'E4M3', 'E5M2'])
            ax6.set_yticks(range(len(layer_names)))
            ax6.set_yticklabels(layer_names, fontsize=9)
            ax6.set_title('Layer Compatibility Heatmap\n(First 15 layers)', fontsize=12, fontweight='bold')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax6, shrink=0.8)
            cbar.set_label('Compatibility Score', fontsize=10, fontweight='bold')
        
        # 7. Format comparison radar chart (bottom left)
        ax7 = fig.add_subplot(gs[3, :2], projection='polar')
        
        # Metrics for radar chart
        metrics = ['Avg Score', 'Complex Layer Score', 'Real Layer Score', 'Coverage %']
        
        # Calculate metrics for each format
        int8_metrics = [
            int8_avg,
            np.mean([d['quantization_int8']['compatibility_score'] for d in analysis['layer_details'].values() 
                    if d['stats']['type'] == 'complex']) if complex_layers > 0 else 0,
            np.mean([d['quantization_int8']['compatibility_score'] for d in analysis['layer_details'].values() 
                    if d['stats']['type'] == 'real']) if real_layers > 0 else 0,
            analysis['quantization_compatibility']['int8']['compatible_layers'] / max(1, total_layers) * 100
        ]
        
        e4m3_metrics = [
            e4m3_avg,
            np.mean([d['quantization_e4m3']['compatibility_score'] for d in analysis['layer_details'].values() 
                    if d['stats']['type'] == 'complex']) if complex_layers > 0 else 0,
            np.mean([d['quantization_e4m3']['compatibility_score'] for d in analysis['layer_details'].values() 
                    if d['stats']['type'] == 'real']) if real_layers > 0 else 0,
            analysis['quantization_compatibility']['e4m3']['compatible_layers'] / max(1, total_layers) * 100
        ]
        
        e5m2_metrics = [
            e5m2_avg,
            np.mean([d['quantization_e5m2']['compatibility_score'] for d in analysis['layer_details'].values() 
                    if d['stats']['type'] == 'complex']) if complex_layers > 0 else 0,
            np.mean([d['quantization_e5m2']['compatibility_score'] for d in analysis['layer_details'].values() 
                    if d['stats']['type'] == 'real']) if real_layers > 0 else 0,
            analysis['quantization_compatibility']['e5m2']['compatible_layers'] / max(1, total_layers) * 100
        ]
        
        # Number of variables
        N = len(metrics)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Complete the circle
        
        # Plot each format
        int8_metrics += int8_metrics[:1]
        e4m3_metrics += e4m3_metrics[:1]
        e5m2_metrics += e5m2_metrics[:1]
        
        ax7.plot(angles, int8_metrics, 'o-', linewidth=2, label='INT8', color='#2ECC71')
        ax7.fill(angles, int8_metrics, alpha=0.25, color='#2ECC71')
        ax7.plot(angles, e4m3_metrics, 'o-', linewidth=2, label='E4M3', color='#F39C12')
        ax7.fill(angles, e4m3_metrics, alpha=0.25, color='#F39C12')
        ax7.plot(angles, e5m2_metrics, 'o-', linewidth=2, label='E5M2', color='#9B59B6')
        ax7.fill(angles, e5m2_metrics, alpha=0.25, color='#9B59B6')
        
        ax7.set_xticks(angles[:-1])
        ax7.set_xticklabels(metrics, fontsize=10)
        ax7.set_ylim(0, 100)
        ax7.set_title('Multi-Format Performance Radar', fontsize=12, fontweight='bold', pad=20)
        ax7.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax7.grid(True)
        
        # 8. Implementation roadmap (bottom right)
        ax8 = fig.add_subplot(gs[3, 2:])
        ax8.axis('off')
        
        # Generate implementation roadmap based on analysis
        roadmap_text = f"""
IMPLEMENTATION ROADMAP

PHASE 1: PREPARATION
✓ Analyze model with this tool
✓ Identify best format: {best_format[0]}
✓ Plan block quantization strategy

PHASE 2: IMPLEMENTATION  
• Start with {best_format[0]} quantization
• Use {'block quantization' if 'block_quantization_analysis' in analysis else 'per-channel quantization'}
• Handle complex layers separately
• Implement mixed precision if needed

PHASE 3: VALIDATION
• Compare accuracy vs baseline
• Measure inference speedup
• Fine-tune quantization parameters
• Consider QAT if accuracy drops >2%

EXPECTED RESULTS:
• Compression: ~4x for INT8, ~2x for FP8
• Speedup: 2-3x on compatible hardware
• Accuracy loss: {'<1%' if best_score >= 80 else '<3%' if best_score >= 60 else 'Variable'}
        """
        
        ax8.text(0.05, 0.95, roadmap_text, transform=ax8.transAxes, 
                fontsize=11, verticalalignment='top', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.3))
        
        plt.suptitle('Enhanced BCCTN Model Multi-Format Quantization Analysis Dashboard', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        self._save_plot(output_dir, 'enhanced_summary_dashboard')
    
    def _plot_layer_wise_detailed_analysis(self, analysis: Dict, output_dir: Path):
        """Create detailed layer-wise analysis with block quantization insights."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Layer complexity vs compatibility
        layer_complexities = []
        avg_compatibilities = []
        layer_types = []
        layer_names = []
        
        for name, details in analysis['layer_details'].items():
            stats = details['stats']
            # Use parameter count as complexity measure
            complexity = np.log10(stats['total_elements'] + 1)
            layer_complexities.append(complexity)
            
            # Average compatibility across all formats
            avg_compat = (details['quantization_int8']['compatibility_score'] + 
                         details['quantization_e4m3']['compatibility_score'] + 
                         details['quantization_e5m2']['compatibility_score']) / 3
            avg_compatibilities.append(avg_compat)
            layer_types.append(stats['type'])
            layer_names.append(name.split('.')[-1][:10])
        
        # Color by type
        colors = ['#E74C3C' if t == 'complex' else '#3498DB' for t in layer_types]
        scatter = ax1.scatter(layer_complexities, avg_compatibilities, c=colors, s=80, alpha=0.7, edgecolor='black')
        
        ax1.set_xlabel('Layer Complexity (log10 parameters)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Average Compatibility Score', fontsize=12, fontweight='bold')
        ax1.set_title('Layer Complexity vs Quantization Compatibility', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        # Add trend line
        z = np.polyfit(layer_complexities, avg_compatibilities, 1)
        p = np.poly1d(z)
        ax1.plot(sorted(layer_complexities), p(sorted(layer_complexities)), "r--", alpha=0.8, linewidth=2)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='#E74C3C', label='Complex Layers'),
                          Patch(facecolor='#3498DB', label='Real Layers')]
        ax1.legend(handles=legend_elements, fontsize=11)
        
        # 2. Block quantization benefit vs layer characteristics
        if 'block_quantization_analysis' in analysis:
            layer_benefits = []
            layer_sparsities = []
            layer_dynamic_ranges = []
            
            for name, details in analysis['layer_details'].items():
                stats = details['stats']
                
                # Find best block configuration improvement
                best_improvement = 0
                for block_config, block_analysis in details['block_analyses'].items():
                    if 'avg_dynamic_range_db' in block_analysis:
                        improvement = stats['dynamic_range_db'] - block_analysis['avg_dynamic_range_db']
                        best_improvement = max(best_improvement, improvement)
                
                layer_benefits.append(best_improvement)
                layer_sparsities.append(stats['sparsity'])
                layer_dynamic_ranges.append(stats['dynamic_range_db'])
            
            # Plot sparsity vs block benefit
            colors2 = ['#E74C3C' if t == 'complex' else '#3498DB' for t in layer_types]
            scatter2 = ax2.scatter(layer_sparsities, layer_benefits, c=colors2, s=80, alpha=0.7, edgecolor='black')
            
            ax2.set_xlabel('Layer Sparsity', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Block Quantization Benefit (dB)', fontsize=12, fontweight='bold')
            ax2.set_title('Sparsity vs Block Quantization Benefit', fontsize=13, fontweight='bold')
            ax2.grid(True, alpha=0.3, linestyle='--')
            
            # Add trend line
            if len(layer_sparsities) > 1:
                z2 = np.polyfit(layer_sparsities, layer_benefits, 1)
                p2 = np.poly1d(z2)
                ax2.plot(sorted(layer_sparsities), p2(sorted(layer_sparsities)), "r--", alpha=0.8, linewidth=2)
        
        # 3. Format preference by layer characteristics
        format_preferences = []
        dynamic_ranges = []
        
        for name, details in analysis['layer_details'].items():
            stats = details['stats']
            scores = {
                'INT8': details['quantization_int8']['compatibility_score'],
                'E4M3': details['quantization_e4m3']['compatibility_score'],
                'E5M2': details['quantization_e5m2']['compatibility_score']
            }
            
            best_format = max(scores, key=scores.get)
            format_preferences.append(best_format)
            dynamic_ranges.append(stats['dynamic_range_db'])
        
        # Create format preference plot
        format_colors = {'INT8': '#2ECC71', 'E4M3': '#F39C12', 'E5M2': '#9B59B6'}
        
        for fmt in ['INT8', 'E4M3', 'E5M2']:
            fmt_ranges = [dr for dr, pref in zip(dynamic_ranges, format_preferences) if pref == fmt]
            fmt_indices = [i for i, pref in enumerate(format_preferences) if pref == fmt]
            
            if fmt_ranges:
                ax3.scatter([fmt] * len(fmt_ranges), fmt_ranges, 
                           color=format_colors[fmt], s=60, alpha=0.7, edgecolor='black')
        
        ax3.set_ylabel('Dynamic Range (dB)', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Preferred Format', fontsize=12, fontweight='bold')
        ax3.set_title('Format Preference vs Dynamic Range', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3, linestyle='--')
        
        # Add format capability lines
        ax3.axhline(y=42, color='#2ECC71', linestyle='--', alpha=0.7, label='INT8 limit')
        ax3.axhline(y=QUANTIZATION_FORMATS['e4m3']['dynamic_range_db'], color='#F39C12', 
                   linestyle='--', alpha=0.7, label='E4M3 limit')
        ax3.axhline(y=QUANTIZATION_FORMATS['e5m2']['dynamic_range_db'], color='#9B59B6', 
                   linestyle='--', alpha=0.7, label='E5M2 limit')
        ax3.legend(fontsize=10)
        
        # 4. Quantization difficulty assessment
        ax4.axis('off')
        
        # Analyze quantization difficulty
        easy_layers = sum(1 for avg in avg_compatibilities if avg >= 80)
        medium_layers = sum(1 for avg in avg_compatibilities if 60 <= avg < 80)
        hard_layers = sum(1 for avg in avg_compatibilities if avg < 60)
        
        # Complex layer analysis
        complex_avg = np.mean([avg for avg, t in zip(avg_compatibilities, layer_types) if t == 'complex']) if complex_layers > 0 else 0
        real_avg = np.mean([avg for avg, t in zip(avg_compatibilities, layer_types) if t == 'real']) if real_layers > 0 else 0
        
        # Best block configuration
        best_block_config = None
        best_block_improvement = 0
        
        if 'block_quantization_analysis' in analysis:
            for config_name, block_stats in analysis['block_quantization_analysis'].items():
                if 'avg_improvement_db' in block_stats:
                    if block_stats['avg_improvement_db'] > best_block_improvement:
                        best_block_improvement = block_stats['avg_improvement_db']
                        best_block_config = config_name
        
        difficulty_text = f"""
LAYER-WISE QUANTIZATION ASSESSMENT

DIFFICULTY DISTRIBUTION:
• Easy (≥80): {easy_layers}/{total_layers} layers ({easy_layers/total_layers*100:.1f}%)
• Medium (60-79): {medium_layers}/{total_layers} layers ({medium_layers/total_layers*100:.1f}%)
• Hard (<60): {hard_layers}/{total_layers} layers ({hard_layers/total_layers*100:.1f}%)

LAYER TYPE PERFORMANCE:
• Complex layers avg: {complex_avg:.1f}/100
• Real layers avg: {real_avg:.1f}/100
• Complexity penalty: {real_avg - complex_avg:.1f} points

BLOCK QUANTIZATION IMPACT:
• Best configuration: {best_block_config or 'N/A'}
• Average improvement: {best_block_improvement:.1f} dB
• Recommended for {sum(1 for b in layer_benefits if b > 5) if 'layer_benefits' in locals() else 0} layers

OPTIMIZATION STRATEGY:
• Focus on {hard_layers} problematic layers first
• Use block quantization for high dynamic range
• Consider mixed precision approach
• {'Complex layers need special attention' if complex_layers > 0 else 'Standard quantization suitable'}
        """
        
        ax4.text(0.05, 0.95, difficulty_text, transform=ax4.transAxes, 
                fontsize=11, verticalalignment='top', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
        
        plt.suptitle('Detailed Layer-wise Quantization Analysis', 
                    fontsize=16, fontweight='bold', y=0.95)
        plt.tight_layout()
        self._save_plot(output_dir, 'layer_wise_detailed_analysis')
    
    def save_enhanced_analysis_json(self, analysis: Dict, output_path: str):
        """Save enhanced analysis results to JSON for further processing."""
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        # Add metadata
        enhanced_analysis = convert_numpy_types(analysis)
        enhanced_analysis['metadata'] = {
            'version': '2.0',
            'features': ['INT8', 'FP8-E4M3', 'FP8-E5M2', 'block_quantization'],
            'analysis_time': datetime.now().isoformat(),
            'model_path': str(self.model_path),
            'formats_analyzed': list(QUANTIZATION_FORMATS.keys()),
            'block_configs_analyzed': list(BLOCK_CONFIGS.keys())
        }
        
        with open(output_path, 'w') as f:
            json.dump(enhanced_analysis, f, indent=2, default=str)
        
        print(f"✅ Enhanced analysis results saved to: {output_path}")


def main():
    """Main function with enhanced command line interface."""
    parser = argparse.ArgumentParser(
        description="Enhanced BCCTN model weight analyzer for multi-format quantization compatibility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic enhanced analysis with all formats
  python enhanced_bcctn_analyzer.py --model_path model.ckpt
  
  # Full analysis with all outputs  
  python enhanced_bcctn_analyzer.py --model_path model.ckpt --config_path config.yaml \\
    --output_dir ./enhanced_analysis --save_plots --save_json --save_report
  
  # Focus on specific format
  python enhanced_bcctn_analyzer.py --model_path model.ckpt --focus_format int8
        """
    )
    
    parser.add_argument("--model_path", required=True, type=str,
                        help="Path to model checkpoint (.ckpt file)")
    parser.add_argument("--config_path", type=str,
                        help="Path to model config file (for loading full model)")
    parser.add_argument("--output_dir", type=str, default="./enhanced_quantization_analysis",
                        help="Output directory for results")
    parser.add_argument("--save_plots", action="store_true",
                        help="Generate and save enhanced visualization plots")
    parser.add_argument("--save_json", action="store_true",
                        help="Save enhanced analysis results as JSON")
    parser.add_argument("--save_report", action="store_true",
                        help="Save detailed text report")
    parser.add_argument("--focus_format", choices=['int8', 'e4m3', 'e5m2'], 
                        help="Focus analysis on specific quantization format")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Initialize enhanced analyzer
        print("🔍 Initializing Enhanced BCCTN Weight Analyzer...")
        print("📊 Formats: INT8, FP8-E4M3, FP8-E5M2")
        print("🧮 Features: Multi-format comparison, Block quantization analysis")
        
        analyzer = EnhancedBCCTNWeightAnalyzer(args.model_path, args.config_path)
        
        # Load model and checkpoint
        analyzer.load_model_and_checkpoint()
        
        # Perform enhanced analysis
        print("\n🧮 Performing comprehensive multi-format analysis...")
        analysis = analyzer.analyze_all_weights()
        
        # Generate outputs
        if args.save_report or not any([args.save_plots, args.save_json]):
            # Generate enhanced report by default
            report_path = output_dir / "enhanced_quantization_compatibility_report.txt"
            analyzer.generate_enhanced_report(analysis, str(report_path))
        
        if args.save_plots:
            print("\n📊 Creating enhanced visualizations...")
            plot_dir = output_dir / "enhanced_plots"
            analyzer.create_enhanced_visualizations(analysis, str(plot_dir))
        
        if args.save_json:
            json_path = output_dir / "enhanced_analysis_results.json"
            analyzer.save_enhanced_analysis_json(analysis, str(json_path))
        
        # Print enhanced final summary
        print(f"\n{'='*80}")
        print("ENHANCED QUANTIZATION ANALYSIS SUMMARY")
        print(f"{'='*80}")
        
        int8_avg = analysis['quantization_compatibility']['int8'].get('avg_score', 0)
        e4m3_avg = analysis['quantization_compatibility']['e4m3'].get('avg_score', 0)
        e5m2_avg = analysis['quantization_compatibility']['e5m2'].get('avg_score', 0)
        complex_layers = analysis['summary_stats']['complex_layers']
        total_layers = len(analysis['layer_details'])
        
        print(f"Model Analysis Summary:")
        print(f"  - Total layers: {total_layers}")
        print(f"  - Complex layers: {complex_layers} ({complex_layers/max(1,total_layers)*100:.1f}%)")
        print(f"  - INT8 average score: {int8_avg:.1f}/100")
        print(f"  - FP8-E4M3 average score: {e4m3_avg:.1f}/100")
        print(f"  - FP8-E5M2 average score: {e5m2_avg:.1f}/100")
        print()
        
        # Format ranking
        format_scores = [('INT8', int8_avg), ('FP8-E4M3', e4m3_avg), ('FP8-E5M2', e5m2_avg)]
        format_scores.sort(key=lambda x: x[1], reverse=True)
        
        print("📊 FORMAT RANKING:")
        for i, (fmt, score) in enumerate(format_scores):
            emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
            print(f"   {emoji} {fmt}: {score:.1f}/100")
        print()
        
        # Block quantization summary
        if 'block_quantization_analysis' in analysis:
            best_block = max(analysis['block_quantization_analysis'].items(), 
                           key=lambda x: x[1].get('avg_improvement_db', 0) if 'avg_improvement_db' in x[1] else 0)
            if best_block[1].get('avg_improvement_db', 0) > 5:
                print(f"🔧 BLOCK QUANTIZATION BENEFIT:")
                print(f"   Best config: {best_block[0].upper()}")
                print(f"   Average improvement: {best_block[1]['avg_improvement_db']:.1f} dB")
                print(f"   Layers benefiting: {best_block[1]['layers_with_benefit']}/{best_block[1]['total_layers']}")
                print()
        
        # Final recommendation
        best_format, best_score = format_scores[0]
        
        if complex_layers > total_layers * 0.5:
            print("🧠 COMPLEX-DOMINANT BCCTN MODEL DETECTED")
            print("   Specialized quantization strategies required:")
            print("   → Separate real/imaginary or magnitude/phase quantization")
            print("   → Block quantization especially beneficial")
            print("   → Consider magnitude-only inference mode")
            print()
        
        if best_score >= 80:
            print(f"🎉 EXCELLENT: {best_format} quantization highly recommended!")
            print("   Expected: <1% accuracy loss, significant speedup")
        elif best_score >= 60:
            print(f"✅ GOOD: {best_format} quantization recommended with careful tuning")
            print("   Expected: <3% accuracy loss, good speedup")
        else:
            print(f"⚠️  CHALLENGING: {best_format} quantization possible but requires optimization")
            print("   Consider: QAT, mixed precision, or advanced techniques")
        
        print(f"\n📁 All enhanced results saved to: {output_dir}")
        print("🚀 Ready for quantization implementation!")
        
    except Exception as e:
        print(f"❌ Error during enhanced analysis: {e}")
        import traceback
        if args.verbose:
            traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())