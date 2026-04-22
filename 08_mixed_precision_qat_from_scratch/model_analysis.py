import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict, OrderedDict
from tqdm import tqdm
import json
import os
from typing import Dict, List, Tuple, Optional, Any
import warnings
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

class UnifiedBCCTNAnalyzer:
    """
    Unified analyzer for BCCTN model combining:
    - Quantization compatibility analysis (weights + activations)
    - Causal sliding window attention memory analysis
    - Computational efficiency evaluation
    - Hardware deployment recommendations
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.weight_stats = {}
        self.input_stats = {}
        self.activation_stats = {}
        self.layer_stats = []
        self.memory_analysis = {}
        self.computational_analysis = {}
        self.overall_analysis = {}
        
        # Quantization format configurations
        self.quantization_formats = {
            'int8': {
                'bits': 8,
                'signed': True,
                'range': (-128, 127),
                'optimal_range_db': 48,
                'name': 'INT8'
            },
            'e4m3': {
                'bits': 8,
                'mantissa': 3,
                'exponent': 4,
                'range': (-448, 448),
                'optimal_range_db': 60,
                'name': 'FP8 E4M3'
            },
            'e5m2': {
                'bits': 8,
                'mantissa': 2,
                'exponent': 5,
                'range': (-57344, 57344),
                'optimal_range_db': 90,
                'name': 'FP8 E5M2'
            }
        }
    
    # ================== Utility Functions ==================
    
    def bytes_per_elem(self, dtype: torch.dtype) -> int:
        """Calculate bytes per element for different data types"""
        if dtype == torch.complex64:
            return 8  # 2×float32
        if dtype == torch.complex128:
            return 16  # 2×float64
        if dtype in (torch.float64, torch.float32, torch.float16, torch.bfloat16):
            return torch.finfo(dtype).bits // 8
        return torch.iinfo(dtype).bits // 8
    
    # ================== Layer Statistics Collection ==================
    
    def collect_layer_stats(self):
        """Collect detailed statistics for each layer"""
        self.layer_stats = []
        
        for name, module in self.model.named_modules():
            if name == '':
                continue
                
            params = list(module.parameters(recurse=False))
            if not params:
                continue
                
            dtype = params[0].dtype
            n_params = sum(p.numel() for p in params)
            
            # Calculate memory usage
            bpe = self.bytes_per_elem(dtype)
            size_mb = n_params * bpe / 1024**2
            
            # Get quantization format configuration for more precise calculation
            is_complex = "complex" in str(dtype).lower()
            
            # More precise quantized memory calculation based on format
            # Default to INT8, but this could be parameterized later
            format_bits = 8  # Could be configurable: 8 for INT8, etc.
            bpe_q = (format_bits / 8) * (2 if is_complex else 1)
            size_mb_q = n_params * bpe_q / 1024**2
            
            layer_info = {
                'layer': name,
                'cls': module.__class__.__name__,
                'n': n_params,
                'dtype': str(dtype),
                'mb': size_mb,
                'mb_q8': size_mb_q,
                'is_complex': is_complex,
                'module_type': self._classify_module_type(module),
                'quantization_bytes_per_element': bpe_q
            }
            
            self.layer_stats.append(layer_info)
        
        return self.layer_stats
    
    def _classify_module_type(self, module):
        """Classify module type for analysis"""
        class_name = module.__class__.__name__.lower()
        
        if 'attention' in class_name or 'multihead' in class_name:
            return 'attention'
        elif 'conv' in class_name:
            return 'convolution'
        elif 'linear' in class_name:
            return 'linear'
        elif 'complex' in class_name:
            return 'complex'
        elif 'norm' in class_name or 'batch' in class_name:
            return 'normalization'
        else:
            return 'other'
    
    # ================== Input Data Statistics ==================
    
    def collect_input_stats(self, data_loader, max_batches: int = 50) -> Dict[str, float]:
        """Collect input data statistics for quantization compatibility analysis"""
        print("Collecting input data statistics...")
        
        all_values = []
        batch_count = 0
        
        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(data_loader, desc="Analyzing input data")):
                if batch_count >= max_batches:
                    break
                
                # Handle different batch formats
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0]
                else:
                    inputs = batch
                
                if not isinstance(inputs, torch.Tensor):
                    continue
                
                # Flatten and collect values
                flattened = inputs.flatten().cpu().numpy()
                all_values.extend(flattened)
                batch_count += 1
        
        if not all_values:
            print("Warning: No input data collected!")
            return self._get_default_input_stats()
        
        all_values = np.array(all_values)
        
        # Calculate comprehensive statistics
        input_stats = {
            'global_min': float(np.min(all_values)),
            'global_max': float(np.max(all_values)),
            'global_mean': float(np.mean(all_values)),
            'global_std': float(np.std(all_values)),
            'p1': float(np.percentile(all_values, 1)),
            'p5': float(np.percentile(all_values, 5)),
            'p25': float(np.percentile(all_values, 25)),
            'p50': float(np.percentile(all_values, 50)),
            'p75': float(np.percentile(all_values, 75)),
            'p95': float(np.percentile(all_values, 95)),
            'p99': float(np.percentile(all_values, 99)),
            'sample_count': len(all_values)
        }
        
        print(f"✓ Collected statistics from {batch_count} batches, {len(all_values):,} samples")
        print(f"  Range: [{input_stats['global_min']:.6f}, {input_stats['global_max']:.6f}]")
        
        self.input_stats = input_stats
        return input_stats
    
    def _get_default_input_stats(self):
        """Return default input statistics if collection fails"""
        return {
            'global_min': 0.0, 'global_max': 1.0, 'global_mean': 0.5, 'global_std': 0.1,
            'p1': 0.0, 'p5': 0.0, 'p25': 0.25, 'p50': 0.5, 'p75': 0.75, 'p95': 1.0, 'p99': 1.0,
            'sample_count': 0
        }
    
    # ================== NEW: Activation Statistics Collection ==================
    
    def collect_activation_stats(self, data_loader, max_batches: int = 20) -> Dict[str, Any]:
        """
        Collect activation statistics from model layers during inference.
        This is crucial for activation quantization analysis.
        """
        print("\nCollecting activation statistics from model layers...")
        
        activation_hooks = {}
        activation_data = defaultdict(list)
        
        def create_hook(layer_name):
            def hook_fn(module, input, output):
                # Handle different output types
                if isinstance(output, torch.Tensor):
                    # Store a sample of activations to avoid memory issues
                    sample = output.detach().cpu().flatten()
                    # Randomly sample if too large
                    if len(sample) > 10000:
                        indices = torch.randperm(len(sample))[:10000]
                        sample = sample[indices]
                    activation_data[layer_name].append(sample.numpy())
                elif isinstance(output, (list, tuple)):
                    # Handle attention outputs (attention_output, attention_weights)
                    for i, out in enumerate(output):
                        if isinstance(out, torch.Tensor):
                            sample = out.detach().cpu().flatten()
                            if len(sample) > 10000:
                                indices = torch.randperm(len(sample))[:10000]
                                sample = sample[indices]
                            activation_data[f"{layer_name}_output_{i}"].append(sample.numpy())
            return hook_fn
        
        # Register hooks for key layers
        target_modules = [
            'encoder', 'decoder', 'mattn', 'attention', 'conv', 'linear', 
            'ComplexConv', 'ComplexLinear', 'MultiAttnBlock'
        ]
        
        hook_handles = []
        for name, module in self.model.named_modules():
            # Register hooks for specific module types
            if any(target in name or target in module.__class__.__name__ for target in target_modules):
                handle = module.register_forward_hook(create_hook(name))
                hook_handles.append(handle)
                activation_hooks[name] = module.__class__.__name__
        
        print(f"  Registered hooks for {len(activation_hooks)} layers")
        
        # Run inference to collect activations
        self.model.eval()
        batch_count = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(data_loader, desc="Collecting activations")):
                if batch_count >= max_batches:
                    break
                
                try:
                    # Handle different batch formats
                    if isinstance(batch, (list, tuple)):
                        inputs = batch[0]
                    else:
                        inputs = batch
                    
                    if not isinstance(inputs, torch.Tensor):
                        continue
                    
                    # Forward pass to trigger hooks
                    _ = self.model(inputs)
                    batch_count += 1
                    
                except Exception as e:
                    print(f"Warning: Failed to process batch {batch_idx}: {e}")
                    continue
        
        # Remove hooks
        for handle in hook_handles:
            handle.remove()
        
        # Process collected activation data
        activation_stats = {}
        
        for layer_name, samples_list in activation_data.items():
            if not samples_list:
                continue
                
            # Concatenate all samples for this layer
            all_samples = np.concatenate(samples_list)
            
            if len(all_samples) == 0:
                continue
            
            # Handle complex values if present
            if all_samples.dtype == np.complex64 or all_samples.dtype == np.complex128:
                # Split into real and imaginary parts
                real_samples = all_samples.real
                imag_samples = all_samples.imag
                all_samples = np.concatenate([real_samples, imag_samples])
            
            # Calculate statistics
            abs_samples = np.abs(all_samples)
            
            # Remove outliers for better analysis
            p1, p99 = np.percentile(abs_samples, [1, 99])
            filtered_samples = abs_samples[(abs_samples >= p1) & (abs_samples <= p99)]
            
            stats = {
                'layer_name': layer_name,
                'module_class': activation_hooks.get(layer_name, 'Unknown'),
                'sample_count': len(all_samples),
                'abs_min': float(np.min(abs_samples)),
                'abs_max': float(np.max(abs_samples)),
                'abs_mean': float(np.mean(abs_samples)),
                'abs_std': float(np.std(abs_samples)),
                'abs_median': float(np.median(abs_samples)),
                'dynamic_range_db': 20 * np.log10(np.max(filtered_samples) / max(np.min(filtered_samples[filtered_samples > 0]), 1e-12)),
                'percentiles': {
                    'p1': float(np.percentile(abs_samples, 1)),
                    'p5': float(np.percentile(abs_samples, 5)),
                    'p25': float(np.percentile(abs_samples, 25)),
                    'p50': float(np.percentile(abs_samples, 50)),
                    'p75': float(np.percentile(abs_samples, 75)),
                    'p95': float(np.percentile(abs_samples, 95)),
                    'p99': float(np.percentile(abs_samples, 99))
                },
                'sparsity': float(np.sum(abs_samples < 1e-6) / len(abs_samples) * 100),
                'zero_ratio': float(np.sum(abs_samples == 0) / len(abs_samples) * 100)
            }
            
            activation_stats[layer_name] = stats
        
        print(f"✓ Collected activation statistics from {len(activation_stats)} layers")
        print(f"  Processed {batch_count} batches")
        
        self.activation_stats = activation_stats
        return activation_stats
    
    def analyze_activation_quantization_compatibility(self, activation_stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze activation quantization compatibility for all formats
        """
        print("Analyzing activation quantization compatibility...")
        
        if not activation_stats:
            print("Warning: No activation statistics available")
            return {}
        
        layer_compatibility = {}
        format_scores = defaultdict(list)
        
        # Analyze each layer for each quantization format
        for layer_name, stats in activation_stats.items():
            layer_results = {}
            
            for fmt in ['int8', 'e4m3', 'e5m2']:
                try:
                    compat = self.evaluate_quantization_compatibility(stats, fmt)
                    layer_results[fmt] = compat
                    format_scores[fmt].append(compat['compatibility_score'])
                except Exception as e:
                    print(f"Warning: Failed to analyze {layer_name} for {fmt}: {e}")
                    layer_results[fmt] = {
                        'compatibility_score': 0.0,
                        'recommendation': 'Error',
                        'recommendation_detail': f'Analysis failed: {e}'
                    }
                    format_scores[fmt].append(0.0)
            
            layer_compatibility[layer_name] = layer_results
        
        # Calculate format averages
        format_averages = {}
        for fmt in ['int8', 'e4m3', 'e5m2']:
            scores = format_scores[fmt]
            if scores:
                format_averages[fmt] = {
                    'average_score': np.mean(scores),
                    'min_score': np.min(scores),
                    'max_score': np.max(scores),
                    'std_score': np.std(scores),
                    'layer_count': len(scores)
                }
            else:
                format_averages[fmt] = {
                    'average_score': 0.0,
                    'min_score': 0.0,
                    'max_score': 0.0,
                    'std_score': 0.0,
                    'layer_count': 0
                }
        
        return {
            'layer_compatibility': layer_compatibility,
            'format_averages': format_averages,
            'total_layers_analyzed': len(activation_stats)
        }
    
    def identify_problematic_activation_layers(self, activation_compat: Dict[str, Any], threshold: float = 50.0) -> Dict[str, List[Dict[str, Any]]]:
        """
        Identify layers with poor activation quantization compatibility
        """
        problematic_layers = defaultdict(list)
        
        if 'layer_compatibility' not in activation_compat:
            return problematic_layers
        
        for layer_name, layer_results in activation_compat['layer_compatibility'].items():
            for fmt in ['int8', 'e4m3', 'e5m2']:
                if fmt in layer_results:
                    score = layer_results[fmt]['compatibility_score']
                    if score < threshold:
                        problematic_layers[fmt].append({
                            'layer': layer_name,
                            'score': score,
                            'recommendation': layer_results[fmt]['recommendation']
                        })
        
        # Sort by score (worst first)
        for fmt in problematic_layers:
            problematic_layers[fmt].sort(key=lambda x: x['score'])
        
        return problematic_layers
    
    # ================== Quantization Analysis ==================
    
    def evaluate_quantization_compatibility(self, stats: Dict[str, Any], format_name: str) -> Dict[str, Any]:
        """Evaluate quantization compatibility for given statistics and format"""
        if format_name not in self.quantization_formats:
            raise ValueError(f"Unknown quantization format: {format_name}")
        
        format_config = self.quantization_formats[format_name]
        
        # Calculate metrics
        abs_min = abs(stats.get('abs_min', stats.get('global_min', 0)))
        abs_max = abs(stats.get('abs_max', stats.get('global_max', 1)))
        
        # Dynamic range in dB
        if 'dynamic_range_db' in stats:
            dynamic_range_db = stats['dynamic_range_db']
        else:
            dynamic_range_db = 20 * np.log10(abs_max / max(abs_min, 1e-12))
        
        # Range utilization (clamped to 0-100)
        format_range = format_config['range']
        max_representable = max(abs(format_range[0]), abs(format_range[1]))
        range_utilization = max(0, min(abs_max / max_representable * 100, 100))
        
        # Dynamic range compatibility (clamped to 0-100)
        optimal_range = format_config['optimal_range_db']
        range_efficiency = max(0, min(dynamic_range_db / optimal_range * 100, 100))
        
        # Precision loss estimation (clamped to 0-100)
        if format_name == 'int8':
            scale = abs_max / 127
            precision_loss = max(0, min(20 * np.log10(scale) if scale > 0 else 0, 100))
        else:
            mantissa_bits = format_config.get('mantissa', 3)
            precision_loss = max(0, min((23 - mantissa_bits) * 0.3, 100))
        
        # Overall compatibility score with proper weighting and clamping
        scores = [
            max(0, min(range_utilization / 50, 1.0)) * 25,      # Range utilization (target 50%+)
            max(0, min(range_efficiency / 80, 1.0)) * 35,       # Dynamic range efficiency (target 80%+)
            max(0, min(1.0 - precision_loss / 100, 1.0)) * 40   # Precision preservation (lower loss better)
        ]
        
        compatibility_score = max(0, min(sum(scores), 100))  # Force 0-100 range
        
        # Determine recommendation
        if compatibility_score >= 80:
            recommendation = "Excellent"
            recommendation_detail = "Highly suitable for this quantization format"
        elif compatibility_score >= 65:
            recommendation = "Good"
            recommendation_detail = "Suitable with minor quality loss"
        elif compatibility_score >= 50:
            recommendation = "Fair"
            recommendation_detail = "Usable but may have noticeable quality impact"
        else:
            recommendation = "Poor"
            recommendation_detail = "Not recommended, significant quality loss expected"
        
        return {
            'compatibility_score': compatibility_score,
            'recommendation': recommendation,
            'recommendation_detail': recommendation_detail,
            'metrics': {
                'range_utilization': max(0, min(range_utilization, 100)),
                'range_efficiency': max(0, min(range_efficiency, 100)),
                'precision_loss_db': max(0, min(precision_loss, 100)),
                'dynamic_range_db': dynamic_range_db,
                'max_representable': max_representable,
                'abs_max': abs_max
            }
        }
    
    def analyze_layer_weights(self, layer_name: str, weights: torch.Tensor) -> Dict[str, Any]:
        """Analyze weights of a single layer"""
        if weights.numel() == 0:
            return None
        
        # Handle complex weights
        if weights.dtype == torch.complex64 or weights.dtype == torch.complex128:
            weights_real = weights.real.flatten().cpu().numpy()
            weights_imag = weights.imag.flatten().cpu().numpy()
            weights_combined = np.concatenate([weights_real, weights_imag])
        else:
            weights_combined = weights.flatten().cpu().numpy()
        
        # Calculate statistics
        abs_weights = np.abs(weights_combined)
        
        stats = {
            'layer_name': layer_name,
            'shape': list(weights.shape),
            'dtype': str(weights.dtype),
            'parameter_count': weights.numel(),
            'abs_min': float(np.min(abs_weights)),
            'abs_max': float(np.max(abs_weights)),
            'abs_mean': float(np.mean(abs_weights)),
            'abs_std': float(np.std(abs_weights)),
            'dynamic_range_db': 20 * np.log10(np.max(abs_weights) / max(np.min(abs_weights[abs_weights > 0]), 1e-12)),
            'percentiles': {
                'p1': float(np.percentile(abs_weights, 1)),
                'p5': float(np.percentile(abs_weights, 5)),
                'p50': float(np.percentile(abs_weights, 50)),
                'p95': float(np.percentile(abs_weights, 95)),
                'p99': float(np.percentile(abs_weights, 99))
            },
            'sparsity': float(np.sum(abs_weights < 1e-6) / len(abs_weights) * 100),
            'is_complex': weights.dtype in [torch.complex64, torch.complex128]
        }
        
        return stats
    
    # ================== Causal Sliding Window Analysis ==================
    
    def analyze_kv_cache_memory(self, window_size: int = 64, fft_len: int = 512, 
                               embed_dim: int = 512, batch_size: int = 1, hop_length: int = 100):
        """Analyze KV Cache memory requirements for Causal Sliding Window Attention"""
        
        # Get model parameters or use defaults
        kernel_num = getattr(self.model, 'kernel_num', [16, 32, 64, 128, 256, 256])
        full_kernel_num = [2] + kernel_num
        
        # Correct STFT output calculation
        freq_bins = fft_len // 2 + 1  # Standard STFT frequency bins
        freq_bins_after_dc = freq_bins - 1  # Remove DC component as in BCCTN
        
        # Calculate downsampling through conv layers
        num_downsamples = len(full_kernel_num) - 1  # 6 downsamples
        final_freq_bins = freq_bins_after_dc // (2 ** num_downsamples)
        final_freq_bins = max(1, final_freq_bins)  # Ensure at least 1 frequency bin
        
        # Calculate flatten_size correctly
        final_channels = kernel_num[-1]  # 256
        base_flat = (final_channels // 2) * final_freq_bins
        flatten_size = base_flat * 2  # binaural multiplication
        
        print(f"\nDetailed KV Cache Memory Calculation:")
        print(f"  FFT length: {fft_len}")
        print(f"  Initial freq bins: {freq_bins}")
        print(f"  After DC removal: {freq_bins_after_dc}")
        print(f"  Downsamples: {num_downsamples}")
        print(f"  Final freq bins: {final_freq_bins}")
        print(f"  Final channels: {final_channels}")
        print(f"  Base flatten size: {base_flat}")
        print(f"  Binaural flatten size: {flatten_size}")
        
        # Attention parameters
        num_heads = getattr(self.model, 'num_heads', 32)
        head_dim = embed_dim // num_heads
        
        print(f"  Attention heads: {num_heads}")
        print(f"  Head dimension: {head_dim}")
        print(f"  Window size: {window_size} frames")
        
        # KV Cache memory calculation (corrected)
        k_cache_elements = batch_size * num_heads * window_size * head_dim
        v_cache_elements = batch_size * num_heads * window_size * head_dim
        total_kv_elements = k_cache_elements + v_cache_elements
        
        # Memory size calculation (complex64 = 8 bytes per element)
        bytes_per_element = 8  # complex64
        kv_cache_bytes = total_kv_elements * bytes_per_element
        kv_cache_kb = kv_cache_bytes / 1024
        kv_cache_mb = kv_cache_bytes / (1024 ** 2)
        
        # Compare with full sequence memory
        full_seq_len = 400  # Typical audio sequence length
        full_seq_kv_elements = batch_size * num_heads * full_seq_len * head_dim * 2
        full_seq_bytes = full_seq_kv_elements * bytes_per_element
        full_seq_mb = full_seq_bytes / (1024**2)
        
        # Memory saving
        memory_saving = (full_seq_bytes - kv_cache_bytes) / full_seq_bytes * 100
        
        # Calculate realistic latency
        hop_ms = hop_length / 16  # Convert to ms at 16kHz
        cache_duration_ms = window_size * hop_ms
        
        # Estimate processing latency components
        fft_cycles_per_frame = fft_len * np.log2(fft_len)
        attention_cycles_per_frame = num_heads * window_size * head_dim
        conv_cycles_per_frame = sum(kernel_num) * final_freq_bins
        
        processing_freq_ghz = 1.0
        fft_latency_ms = fft_cycles_per_frame / (processing_freq_ghz * 1e6)
        attention_latency_ms = attention_cycles_per_frame / (processing_freq_ghz * 1e6)
        conv_latency_ms = conv_cycles_per_frame / (processing_freq_ghz * 1e6)
        
        total_processing_latency_ms = fft_latency_ms + attention_latency_ms + conv_latency_ms
        
        analysis = {
            'window_size': window_size,
            'flatten_size': flatten_size,
            'num_heads': num_heads,
            'head_dim': head_dim,
            'freq_bins_final': final_freq_bins,
            'kv_cache_elements': total_kv_elements,
            'kv_cache_bytes': kv_cache_bytes,
            'kv_cache_kb': kv_cache_kb,
            'kv_cache_mb': kv_cache_mb,
            'full_seq_mb': full_seq_mb,
            'memory_saving_percent': memory_saving,
            'cache_duration_ms': cache_duration_ms,
            'processing_latency_ms': total_processing_latency_ms,
            'total_latency_ms': cache_duration_ms + total_processing_latency_ms,
            'hardware_suitability': self._assess_hardware_suitability(kv_cache_kb),
            'latency_breakdown': {
                'hop_latency_ms': hop_ms,
                'cache_latency_ms': cache_duration_ms,
                'fft_latency_ms': fft_latency_ms,
                'attention_latency_ms': attention_latency_ms,
                'conv_latency_ms': conv_latency_ms,
                'total_processing_ms': total_processing_latency_ms
            }
        }
        
        self.memory_analysis = analysis
        return analysis
    
    def _assess_hardware_suitability(self, cache_kb: float) -> str:
        """Assess hardware cache suitability"""
        if cache_kb < 16:
            return "L1 Cache (Excellent)"
        elif cache_kb < 256:
            return "L2 Cache (Very Good)"
        elif cache_kb < 8192:
            return "L3 Cache (Good)"
        else:
            return "Main Memory (Moderate)"
    
    def analyze_attention_computational_savings(self, window_size: int = 64, seq_len: int = 400):
        """Analyze computational savings from Causal + Sliding Window"""
        
        # Full attention matrix: seq_len x seq_len
        full_attn_ops = seq_len * seq_len
        
        # Causal mask: only lower triangle is valid
        causal_attn_ops = seq_len * (seq_len + 1) // 2
        
        # Sliding window causal: each position sees at most window_size past positions
        sliding_causal_ops = 0
        for i in range(seq_len):
            effective_context = min(i + 1, window_size)
            sliding_causal_ops += effective_context
        
        # Calculate savings
        causal_saving = (1 - causal_attn_ops / full_attn_ops) * 100
        sliding_saving = (1 - sliding_causal_ops / full_attn_ops) * 100
        combined_saving = (1 - sliding_causal_ops / causal_attn_ops) * 100
        
        # Include multi-head and embedding dimension factors
        num_heads = getattr(self.model, 'num_heads', 32)
        embed_dim = getattr(self.model, 'embed_dim', 512)
        head_dim = embed_dim // num_heads
        
        # Calculate actual computational complexity
        full_complexity = num_heads * seq_len * seq_len * head_dim
        causal_complexity = num_heads * causal_attn_ops * head_dim
        sliding_complexity = num_heads * sliding_causal_ops * head_dim
        
        analysis = {
            'sequence_length': seq_len,
            'window_size': window_size,
            'num_heads': num_heads,
            'head_dim': head_dim,
            'operations': {
                'full_attention': full_attn_ops,
                'causal_attention': causal_attn_ops,
                'sliding_window_causal': sliding_causal_ops
            },
            'computational_complexity': {
                'full_attention': full_complexity,
                'causal_attention': causal_complexity,
                'sliding_window_causal': sliding_complexity
            },
            'savings_percent': {
                'causal_vs_full': causal_saving,
                'sliding_vs_full': sliding_saving,
                'sliding_vs_causal': combined_saving
            },
            'efficiency_ratio': {
                'causal_efficiency': causal_complexity / full_complexity,
                'sliding_efficiency': sliding_complexity / full_complexity
            }
        }
        
        self.computational_analysis = analysis
        return analysis
    
    # ================== Weight Analysis ==================
    
    def analyze_weight_quantization_compatibility(self) -> Dict[str, Any]:
        """Analyze weight quantization compatibility for all layers"""
        print("Analyzing weight quantization compatibility...")
        
        layer_weight_stats = {}
        layer_compatibility = {}
        format_scores = defaultdict(list)
        
        # Collect weight statistics for each layer
        for name, module in self.model.named_modules():
            if name == '':
                continue
                
            for param_name, param in module.named_parameters(recurse=False):
                layer_full_name = f"{name}.{param_name}" if name else param_name
                
                if param.requires_grad:  # Only analyze trainable parameters
                    weight_stats = self.analyze_layer_weights(layer_full_name, param)
                    if weight_stats:
                        layer_weight_stats[layer_full_name] = weight_stats
                        
                        # Analyze compatibility for each format
                        layer_results = {}
                        for fmt in ['int8', 'e4m3', 'e5m2']:
                            try:
                                compat = self.evaluate_quantization_compatibility(weight_stats, fmt)
                                layer_results[fmt] = compat
                                format_scores[fmt].append(compat['compatibility_score'])
                            except Exception as e:
                                print(f"Warning: Failed to analyze weights {layer_full_name} for {fmt}: {e}")
                                layer_results[fmt] = {
                                    'compatibility_score': 0.0,
                                    'recommendation': 'Error',
                                    'recommendation_detail': f'Analysis failed: {e}'
                                }
                                format_scores[fmt].append(0.0)
                        
                        layer_compatibility[layer_full_name] = layer_results
        
        # Calculate format averages
        format_averages = {}
        for fmt in ['int8', 'e4m3', 'e5m2']:
            scores = format_scores[fmt]
            if scores:
                format_averages[fmt] = {
                    'average_score': np.mean(scores),
                    'min_score': np.min(scores),
                    'max_score': np.max(scores),
                    'std_score': np.std(scores),
                    'layer_count': len(scores)
                }
            else:
                format_averages[fmt] = {
                    'average_score': 0.0,
                    'min_score': 0.0,
                    'max_score': 0.0,
                    'std_score': 0.0,
                    'layer_count': 0
                }
        
        self.weight_stats = {
            'layer_stats': layer_weight_stats,
            'layer_compatibility': layer_compatibility,
            'format_averages': format_averages,
            'total_layers_analyzed': len(layer_weight_stats)
        }
        
        return self.weight_stats
    
    # ================== Unified Analysis and Reporting ==================
    
    def run_complete_analysis(self, data_loader, output_dir: str = "bcctn_analysis", 
                            max_batches_input: int = 50, max_batches_activation: int = 20,
                            window_size: int = 64):
        """Run complete unified analysis"""
        
        print("="*80)
        print("🔬 BCCTN Unified Analysis - Starting Complete Analysis")
        print("="*80)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Layer Statistics
        print("\n📊 Step 1: Collecting layer statistics...")
        layer_stats = self.collect_layer_stats()
        
        # 2. Input Data Analysis
        print("\n📈 Step 2: Analyzing input data...")
        input_stats = self.collect_input_stats(data_loader, max_batches_input)
        input_compatibility = {}
        for fmt in ['int8', 'e4m3', 'e5m2']:
            input_compatibility[fmt] = self.evaluate_quantization_compatibility(input_stats, fmt)
        
        # 3. Weight Analysis
        print("\n⚖️ Step 3: Analyzing model weights...")
        weight_analysis = self.analyze_weight_quantization_compatibility()
        
        # 4. Activation Analysis
        print("\n🧠 Step 4: Analyzing model activations...")
        activation_stats = self.collect_activation_stats(data_loader, max_batches_activation)
        activation_compatibility = self.analyze_activation_quantization_compatibility(activation_stats)
        
        # 5. Memory Analysis
        print("\n💾 Step 5: Analyzing memory requirements...")
        memory_analysis = self.analyze_kv_cache_memory(window_size=window_size)
        
        # 6. Computational Analysis
        print("\n⚡ Step 6: Analyzing computational efficiency...")
        computational_analysis = self.analyze_attention_computational_savings(window_size=window_size)
        
        # 7. Generate Overall Assessment
        print("\n🎯 Step 7: Generating overall assessment...")
        overall_assessment = self.generate_overall_assessment(
            input_compatibility, weight_analysis, activation_compatibility,
            memory_analysis, computational_analysis
        )
        
        # 8. Save Results
        print("\n💾 Step 8: Saving analysis results...")
        self.save_analysis_results(output_dir, {
            'layer_stats': layer_stats,
            'input_stats': input_stats,
            'input_compatibility': input_compatibility,
            'weight_analysis': weight_analysis,
            'activation_stats': activation_stats,
            'activation_compatibility': activation_compatibility,
            'memory_analysis': memory_analysis,
            'computational_analysis': computational_analysis,
            'overall_assessment': overall_assessment
        })
        
        # 9. Generate Report
        print("\n📝 Step 9: Generating analysis report...")
        self.generate_analysis_report(output_dir, overall_assessment)
        
        print(f"\n✅ Complete analysis finished! Results saved to: {output_dir}")
        return overall_assessment
    
    def generate_overall_assessment(self, input_compatibility, weight_analysis, 
                                  activation_compatibility, memory_analysis, 
                                  computational_analysis) -> Dict[str, Any]:
        """Generate overall assessment and recommendations"""
        
        # Calculate overall quantization scores
        overall_scores = {}
        for fmt in ['int8', 'e4m3', 'e5m2']:
            scores = []
            
            # Input compatibility score
            if fmt in input_compatibility:
                scores.append(input_compatibility[fmt]['compatibility_score'])
            
            # Weight compatibility average
            if 'format_averages' in weight_analysis and fmt in weight_analysis['format_averages']:
                scores.append(weight_analysis['format_averages'][fmt]['average_score'])
            
            # Activation compatibility average
            if 'format_averages' in activation_compatibility and fmt in activation_compatibility['format_averages']:
                scores.append(activation_compatibility['format_averages'][fmt]['average_score'])
            
            overall_scores[fmt] = {
                'average_score': np.mean(scores) if scores else 0.0,
                'component_scores': scores,
                'recommendation': self._get_recommendation_from_score(np.mean(scores) if scores else 0.0)
            }
        
        # Find best quantization format
        best_format = max(overall_scores.keys(), key=lambda k: overall_scores[k]['average_score'])
        
        # Memory efficiency assessment
        memory_efficiency = self._assess_memory_efficiency(memory_analysis)
        
        # Computational efficiency assessment
        computational_efficiency = self._assess_computational_efficiency(computational_analysis)
        
        # Hardware deployment recommendations
        deployment_recommendations = self._generate_deployment_recommendations(
            overall_scores, memory_analysis, computational_analysis
        )
        
        assessment = {
            'quantization_analysis': {
                'format_scores': overall_scores,
                'best_format': best_format,
                'best_score': overall_scores[best_format]['average_score']
            },
            'memory_efficiency': memory_efficiency,
            'computational_efficiency': computational_efficiency,
            'deployment_recommendations': deployment_recommendations,
            'overall_readiness_score': self._calculate_overall_readiness_score(
                overall_scores, memory_efficiency, computational_efficiency
            )
        }
        
        return assessment
    
    def _get_recommendation_from_score(self, score: float) -> str:
        """Get recommendation based on compatibility score"""
        if score >= 80:
            return "Excellent - Highly Recommended"
        elif score >= 65:
            return "Good - Recommended"
        elif score >= 50:
            return "Fair - Acceptable with Monitoring"
        else:
            return "Poor - Not Recommended"
    
    def _assess_memory_efficiency(self, memory_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess memory efficiency"""
        cache_mb = memory_analysis.get('kv_cache_mb', 0)
        memory_saving = memory_analysis.get('memory_saving_percent', 0)
        hardware_suitability = memory_analysis.get('hardware_suitability', 'Unknown')
        
        if cache_mb < 1:
            efficiency_rating = "Excellent"
            efficiency_score = 95
        elif cache_mb < 16:
            efficiency_rating = "Very Good"
            efficiency_score = 85
        elif cache_mb < 64:
            efficiency_rating = "Good"
            efficiency_score = 70
        else:
            efficiency_rating = "Moderate"
            efficiency_score = 50
        
        return {
            'cache_size_mb': cache_mb,
            'memory_saving_percent': memory_saving,
            'hardware_suitability': hardware_suitability,
            'efficiency_rating': efficiency_rating,
            'efficiency_score': efficiency_score
        }
    
    def _assess_computational_efficiency(self, computational_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess computational efficiency"""
        sliding_saving = computational_analysis.get('savings_percent', {}).get('sliding_vs_full', 0)
        sliding_efficiency = computational_analysis.get('efficiency_ratio', {}).get('sliding_efficiency', 1.0)
        
        if sliding_saving > 80:
            efficiency_rating = "Excellent"
            efficiency_score = 95
        elif sliding_saving > 60:
            efficiency_rating = "Very Good"
            efficiency_score = 85
        elif sliding_saving > 40:
            efficiency_rating = "Good"
            efficiency_score = 70
        else:
            efficiency_rating = "Moderate"
            efficiency_score = 50
        
        return {
            'computational_saving_percent': sliding_saving,
            'efficiency_ratio': sliding_efficiency,
            'efficiency_rating': efficiency_rating,
            'efficiency_score': efficiency_score
        }
    
    def _generate_deployment_recommendations(self, overall_scores, memory_analysis, 
                                           computational_analysis) -> Dict[str, Any]:
        """Generate hardware deployment recommendations"""
        
        # Get best quantization format
        best_format = max(overall_scores.keys(), key=lambda k: overall_scores[k]['average_score'])
        best_score = overall_scores[best_format]['average_score']
        
        # Memory requirements
        cache_kb = memory_analysis.get('kv_cache_kb', 0)
        total_latency = memory_analysis.get('total_latency_ms', 0)
        
        # Generate recommendations
        recommendations = {
            'quantization': {
                'recommended_format': best_format,
                'expected_quality': self._get_recommendation_from_score(best_score),
                'alternative_formats': [fmt for fmt, data in overall_scores.items() 
                                      if data['average_score'] > 50 and fmt != best_format]
            },
            'hardware_targets': [],
            'optimization_suggestions': [],
            'deployment_strategy': {}
        }
        
        # Hardware target recommendations
        if cache_kb < 16:
            recommendations['hardware_targets'].extend([
                "Edge devices with limited memory",
                "Mobile processors",
                "Embedded systems"
            ])
        elif cache_kb < 256:
            recommendations['hardware_targets'].extend([
                "Mid-range GPUs",
                "ARM-based servers",
                "Edge AI accelerators"
            ])
        else:
            recommendations['hardware_targets'].extend([
                "High-end GPUs",
                "Server-class hardware",
                "Cloud deployment"
            ])
        
        # Optimization suggestions
        if best_score < 70:
            recommendations['optimization_suggestions'].append(
                "Consider mixed-precision training to improve quantization compatibility"
            )
        
        if memory_analysis.get('memory_saving_percent', 0) < 60:
            recommendations['optimization_suggestions'].append(
                "Consider reducing window size for better memory efficiency"
            )
        
        if total_latency > 50:
            recommendations['optimization_suggestions'].append(
                "Consider model pruning or knowledge distillation for lower latency"
            )
        
        # Deployment strategy
        if best_score > 80 and cache_kb < 64:
            recommendations['deployment_strategy'] = {
                'readiness': 'Production Ready',
                'confidence': 'High',
                'recommended_approach': 'Direct deployment with monitoring'
            }
        elif best_score > 60:
            recommendations['deployment_strategy'] = {
                'readiness': 'Near Production',
                'confidence': 'Medium',
                'recommended_approach': 'Deploy with extensive testing and fallback'
            }
        else:
            recommendations['deployment_strategy'] = {
                'readiness': 'Development/Testing',
                'confidence': 'Low',
                'recommended_approach': 'Optimization required before production'
            }
        
        return recommendations
    
    def _calculate_overall_readiness_score(self, overall_scores, memory_efficiency, 
                                         computational_efficiency) -> float:
        """Calculate overall deployment readiness score"""
        
        # Get best quantization score
        best_quant_score = max(data['average_score'] for data in overall_scores.values())
        
        # Weighted average of all factors
        weights = {
            'quantization': 0.4,
            'memory': 0.3,
            'computation': 0.3
        }
        
        readiness_score = (
            weights['quantization'] * best_quant_score +
            weights['memory'] * memory_efficiency['efficiency_score'] +
            weights['computation'] * computational_efficiency['efficiency_score']
        )
        
        return readiness_score
    
    def save_analysis_results(self, output_dir: str, results: Dict[str, Any]):
        """Save analysis results to files"""
        
        # Save complete results as JSON
        results_path = os.path.join(output_dir, "complete_analysis.json")
        with open(results_path, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            json_results = self._convert_for_json(results)
            json.dump(json_results, f, indent=2)
        
        # Save layer statistics as CSV
        if results.get('layer_stats'):
            df = pd.DataFrame(results['layer_stats'])
            df.to_csv(os.path.join(output_dir, "layer_statistics.csv"), index=False)
        
        # Save quantization compatibility summary
        compatibility_summary = []
        
        # Input compatibility
        for fmt, data in results.get('input_compatibility', {}).items():
            compatibility_summary.append({
                'component': 'Input Data',
                'format': fmt,
                'score': data.get('compatibility_score', 0),
                'recommendation': data.get('recommendation', 'Unknown')
            })
        
        # Weight compatibility
        weight_analysis = results.get('weight_analysis', {})
        for fmt, data in weight_analysis.get('format_averages', {}).items():
            compatibility_summary.append({
                'component': 'Model Weights',
                'format': fmt,
                'score': data.get('average_score', 0),
                'recommendation': self._get_recommendation_from_score(data.get('average_score', 0))
            })
        
        # Activation compatibility
        activation_analysis = results.get('activation_compatibility', {})
        for fmt, data in activation_analysis.get('format_averages', {}).items():
            compatibility_summary.append({
                'component': 'Model Activations',
                'format': fmt,
                'score': data.get('average_score', 0),
                'recommendation': self._get_recommendation_from_score(data.get('average_score', 0))
            })
        
        if compatibility_summary:
            df_compat = pd.DataFrame(compatibility_summary)
            df_compat.to_csv(os.path.join(output_dir, "quantization_compatibility_summary.csv"), index=False)
    
    def _convert_for_json(self, obj):
        """Convert numpy types to Python types for JSON serialization"""
        if isinstance(obj, dict):
            return {key: self._convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj
    
    def generate_analysis_report(self, output_dir: str, assessment: Dict[str, Any]):
        """Generate a comprehensive analysis report"""
        
        report_path = os.path.join(output_dir, "BCCTN_Analysis_Report.md")
        
        with open(report_path, 'w') as f:
            f.write("# BCCTN Model Analysis Report\n\n")
            f.write(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Executive Summary
            f.write("## 🎯 Executive Summary\n\n")
            readiness_score = assessment.get('overall_readiness_score', 0)
            f.write(f"**Overall Readiness Score: {readiness_score:.1f}/100**\n\n")
            
            # Quantization Analysis
            quant_analysis = assessment.get('quantization_analysis', {})
            best_format = quant_analysis.get('best_format', 'Unknown')
            best_score = quant_analysis.get('best_score', 0)
            
            f.write(f"**Recommended Quantization Format: {best_format.upper()} (Score: {best_score:.1f})**\n\n")
            
            # Memory and Computational Efficiency
            memory_eff = assessment.get('memory_efficiency', {})
            comp_eff = assessment.get('computational_efficiency', {})
            
            f.write(f"**Memory Efficiency: {memory_eff.get('efficiency_rating', 'Unknown')} "
                   f"({memory_eff.get('efficiency_score', 0):.1f}/100)**\n\n")
            f.write(f"**Computational Efficiency: {comp_eff.get('efficiency_rating', 'Unknown')} "
                   f"({comp_eff.get('efficiency_score', 0):.1f}/100)**\n\n")
            
            # Detailed Analysis Sections
            f.write("## 📊 Detailed Analysis\n\n")
            
            # Quantization Compatibility
            f.write("### Quantization Compatibility\n\n")
            format_scores = quant_analysis.get('format_scores', {})
            for fmt, data in format_scores.items():
                score = data.get('average_score', 0)
                recommendation = data.get('recommendation', 'Unknown')
                f.write(f"- **{fmt.upper()}**: {score:.1f}/100 - {recommendation}\n")
            f.write("\n")
            
            # Memory Analysis
            f.write("### Memory Analysis\n\n")
            if hasattr(self, 'memory_analysis'):
                mem = self.memory_analysis
                f.write(f"- **KV Cache Size**: {mem.get('kv_cache_mb', 0):.2f} MB\n")
                f.write(f"- **Memory Saving**: {mem.get('memory_saving_percent', 0):.1f}% vs full attention\n")
                f.write(f"- **Hardware Suitability**: {mem.get('hardware_suitability', 'Unknown')}\n")
                f.write(f"- **Processing Latency**: {mem.get('processing_latency_ms', 0):.2f} ms\n\n")
            
            # Computational Analysis
            f.write("### Computational Analysis\n\n")
            if hasattr(self, 'computational_analysis'):
                comp = self.computational_analysis
                savings = comp.get('savings_percent', {})
                f.write(f"- **Computational Saving**: {savings.get('sliding_vs_full', 0):.1f}% vs full attention\n")
                f.write(f"- **Window Size**: {comp.get('window_size', 0)} frames\n")
                f.write(f"- **Attention Heads**: {comp.get('num_heads', 0)}\n\n")
            
            # Deployment Recommendations
            f.write("### 🚀 Deployment Recommendations\n\n")
            deploy_rec = assessment.get('deployment_recommendations', {})
            
            deployment_strategy = deploy_rec.get('deployment_strategy', {})
            f.write(f"**Deployment Readiness**: {deployment_strategy.get('readiness', 'Unknown')}\n\n")
            f.write(f"**Confidence Level**: {deployment_strategy.get('confidence', 'Unknown')}\n\n")
            f.write(f"**Recommended Approach**: {deployment_strategy.get('recommended_approach', 'Unknown')}\n\n")
            
            # Hardware Targets
            hardware_targets = deploy_rec.get('hardware_targets', [])
            if hardware_targets:
                f.write("**Suitable Hardware Platforms**:\n")
                for target in hardware_targets:
                    f.write(f"- {target}\n")
                f.write("\n")
            
            # Optimization Suggestions
            opt_suggestions = deploy_rec.get('optimization_suggestions', [])
            if opt_suggestions:
                f.write("**Optimization Suggestions**:\n")
                for suggestion in opt_suggestions:
                    f.write(f"- {suggestion}\n")
                f.write("\n")
            
            f.write("---\n\n")
            f.write("*This report was generated by the BCCTN Unified Analyzer*\n")
        
        print(f"📝 Analysis report saved to: {report_path}")


def main():
    """
    Main function to demonstrate the unified BCCTN analyzer
    """
    import argparse
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description="BCCTN Unified Model Analyzer")
    parser.add_argument("--model_checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--config_path", default="./config", help="Path to config directory")
    parser.add_argument("--data_path", required=True, help="Path to test dataset")
    parser.add_argument("--output_dir", default="./bcctn_analysis", help="Output directory")
    parser.add_argument("--window_size", type=int, default=64, help="Attention window size")
    parser.add_argument("--max_batches_input", type=int, default=50, help="Max batches for input analysis")
    parser.add_argument("--max_batches_activation", type=int, default=20, help="Max batches for activation analysis")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for analysis")
    
    args = parser.parse_args()
    
    print("🔬 BCCTN Unified Analyzer")
    print("="*50)
    
    try:
        # Initialize Hydra and load model
        from hydra import compose, initialize
        from hydra.core.global_hydra import GlobalHydra
        
        # Clear any existing Hydra instance
        GlobalHydra.instance().clear()
        initialize(config_path=args.config_path)
        config = compose(config_name="config")
        
        # Load model
        print(f"📥 Loading model from: {args.model_checkpoint}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Import model class
        try:
            from DCNN.trainer import DCNNLightningModule
        except ImportError:
            print("❌ Error: Cannot import DCNN modules. Please ensure the DCNN package is in your PYTHONPATH")
            return
        
        model = DCNNLightningModule(config)
        model.eval()
        
        # Load checkpoint
        checkpoint = torch.load(args.model_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        model = model.to(device)
        
        print(f"✅ Model loaded successfully on {device}")
        
        # Create data loader
        print(f"📊 Setting up data loader from: {args.data_path}")
        
        try:
            from DCNN.datasets.base_dataset import BaseDataset
        except ImportError:
            print("❌ Error: Cannot import dataset classes. Please ensure the DCNN package is in your PYTHONPATH")
            return
        
        # Create a simple dataset for analysis
        clean_path = Path(args.data_path) / "clean_testset"
        noisy_path = Path(args.data_path) / "noisy_testset"
        
        if not clean_path.exists() or not noisy_path.exists():
            print(f"❌ Error: Dataset paths not found:")
            print(f"   Clean: {clean_path}")
            print(f"   Noisy: {noisy_path}")
            return
        
        dataset = BaseDataset(str(noisy_path), str(clean_path), mono=False)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=False,
            drop_last=False,
            num_workers=0  # Avoid multiprocessing issues during analysis
        )
        
        print(f"✅ Data loader created with {len(dataset)} samples")
        
        # Initialize analyzer
        print("🔧 Initializing BCCTN Unified Analyzer...")
        analyzer = UnifiedBCCTNAnalyzer(model)
        
        # Run complete analysis
        assessment = analyzer.run_complete_analysis(
            dataloader,
            output_dir=args.output_dir,
            max_batches_input=args.max_batches_input,
            max_batches_activation=args.max_batches_activation,
            window_size=args.window_size
        )
        
        # Print summary
        print("\n🎉 Analysis Complete!")
        print("="*50)
        print("📋 Summary:")
        print(f"   Overall Readiness Score: {assessment.get('overall_readiness_score', 0):.1f}/100")
        
        quant_analysis = assessment.get('quantization_analysis', {})
        best_format = quant_analysis.get('best_format', 'Unknown')
        best_score = quant_analysis.get('best_score', 0)
        print(f"   Best Quantization Format: {best_format.upper()} ({best_score:.1f}/100)")
        
        memory_eff = assessment.get('memory_efficiency', {})
        print(f"   Memory Efficiency: {memory_eff.get('efficiency_rating', 'Unknown')}")
        
        comp_eff = assessment.get('computational_efficiency', {})
        print(f"   Computational Efficiency: {comp_eff.get('efficiency_rating', 'Unknown')}")
        
        deploy_rec = assessment.get('deployment_recommendations', {})
        deployment_strategy = deploy_rec.get('deployment_strategy', {})
        print(f"   Deployment Readiness: {deployment_strategy.get('readiness', 'Unknown')}")
        
        print(f"\n📁 Results saved to: {args.output_dir}")
        print("   📄 Complete results: complete_analysis.json")
        print("   📊 Layer statistics: layer_statistics.csv")
        print("   📈 Compatibility summary: quantization_compatibility_summary.csv")
        print("   📝 Analysis report: BCCTN_Analysis_Report.md")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())