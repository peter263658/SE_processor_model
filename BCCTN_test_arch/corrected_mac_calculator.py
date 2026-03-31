#!/usr/bin/env python3
"""
Corrected MAC Calculator Based on Actual Model Structure
Accounts for the real complex layer implementation using separate real/imaginary components
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
import math


class CorrectedMACCounter:
    """MAC counter that matches the actual complex layer implementation"""
    
    def __init__(self):
        self.mac_count = 0
        self.layer_macs = {}
    
    def reset(self):
        self.mac_count = 0
        self.layer_macs = {}
    
    def count_real_conv2d_macs(self, input_shape: Tuple[int, ...], 
                              output_shape: Tuple[int, ...],
                              kernel_size: Tuple[int, int],
                              in_channels: int, out_channels: int,
                              is_depthwise: bool = False) -> int:
        """
        Count MAC operations for REAL convolution (not complex)
        This matches the actual implementation with separate _r and _i layers
        """
        batch_size, _, out_h, out_w = output_shape
        kernel_h, kernel_w = kernel_size
        
        if is_depthwise:
            # Depthwise: each input channel convolved separately
            macs_per_output = kernel_h * kernel_w
            total_outputs = batch_size * in_channels * out_h * out_w
        else:
            # Regular convolution
            macs_per_output = in_channels * kernel_h * kernel_w
            total_outputs = batch_size * out_channels * out_h * out_w
            
        return macs_per_output * total_outputs
    
    def count_complex_depthwise_separable_macs(self, input_shape: Tuple[int, ...],
                                             output_shape: Tuple[int, ...],
                                             kernel_size: Tuple[int, int],
                                             in_channels: int, out_channels: int) -> int:
        """
        Count MACs for complex depthwise separable conv using actual implementation:
        - depthwise_r, depthwise_i (2 separate real depthwise convs)
        - pointwise_r, pointwise_i (2 separate real pointwise convs)
        """
        batch_size, _, out_h, out_w = output_shape
        kernel_h, kernel_w = kernel_size
        
        # 1. Depthwise convolution: 2 separate real convolutions
        depthwise_macs_r = self.count_real_conv2d_macs(
            input_shape, (batch_size, in_channels, out_h, out_w),
            kernel_size, in_channels, in_channels, is_depthwise=True
        )
        depthwise_macs_i = depthwise_macs_r  # Same for imaginary part
        
        # 2. Pointwise convolution: 2 separate real 1x1 convolutions
        pointwise_macs_r = self.count_real_conv2d_macs(
            (batch_size, in_channels, out_h, out_w), output_shape,
            (1, 1), in_channels, out_channels, is_depthwise=False
        )
        pointwise_macs_i = pointwise_macs_r  # Same for imaginary part
        
        return depthwise_macs_r + depthwise_macs_i + pointwise_macs_r + pointwise_macs_i
    
    def count_complex_regular_conv_macs(self, input_shape: Tuple[int, ...],
                                       output_shape: Tuple[int, ...],
                                       kernel_size: Tuple[int, int],
                                       in_channels: int, out_channels: int) -> int:
        """
        Count MACs for complex regular conv using actual implementation:
        - conv_r, conv_i (2 separate real convolutions)
        """
        # 2 separate real convolutions
        real_macs = self.count_real_conv2d_macs(
            input_shape, output_shape, kernel_size, in_channels, out_channels
        )
        return real_macs * 2  # Real and imaginary parts
    
    def count_complex_attention_macs(self, embed_dim: int, num_heads: int,
                                   seq_len: int, batch_size: int) -> int:
        """
        Count MACs for complex attention using actual implementation:
        - MultiheadAttention_re, MultiheadAttention_im (2 separate real attention blocks)
        """
        # Calculate MACs for one real attention block
        head_dim = embed_dim // num_heads
        
        # Q, K, V projections (3 linear layers)
        qkv_macs = 3 * batch_size * seq_len * embed_dim * embed_dim
        
        # Attention computation: Q @ K^T
        attn_macs = batch_size * num_heads * seq_len * seq_len * head_dim
        
        # Attention @ V  
        attn_v_macs = batch_size * num_heads * seq_len * seq_len * head_dim
        
        # Output projection
        out_proj_macs = batch_size * seq_len * embed_dim * embed_dim
        
        single_attention_macs = qkv_macs + attn_macs + attn_v_macs + out_proj_macs
        
        # 2 separate attention blocks (real and imaginary)
        return single_attention_macs * 2
    
    def count_fft_macs(self, n: int, batch_size: int, num_channels: int = 1) -> int:
        """Count MAC operations for FFT"""
        # FFT complexity: O(n * log2(n))
        # Each complex multiplication = 4 real MACs + 2 real additions
        fft_ops_per_transform = n * math.log2(n) * 6  # 4 mults + 2 adds
        total_transforms = batch_size * num_channels
        return int(fft_ops_per_transform * total_transforms)


class ActualModelMACCalculator:
    """MAC calculator based on your actual model structure"""
    
    def __init__(self, config: Dict = None):
        self.counter = CorrectedMACCounter()
        
        # Configuration from your actual model output
        self.config = config or {
            'fft_len': 512,           # From your output
            'win_len': 400,           # From your output  
            'win_inc': 100,           # From your output
            'kernel_num': [16, 32, 64, 128, 256, 256],  # From your output
            'kernel_size': 5,
            'embed_dim': 512,
            'num_heads': 32,
            'binaural': True,
            'use_depthwise': True,
            'batch_size': 1,
            'duration_seconds': 2.0,
            'sample_rate': 16000
        }
        
        if config:
            self.config.update(config)
            
        self._calculate_dimensions()
    
    def _calculate_dimensions(self):
        """Calculate dimensions based on actual model"""
        self.fft_len = self.config['fft_len']
        self.win_inc = self.config['win_inc'] 
        self.sample_rate = self.config['sample_rate']
        self.duration = self.config['duration_seconds']
        
        # Calculate number of frames
        total_samples = int(self.duration * self.sample_rate)
        self.num_frames = (total_samples - self.config['win_len']) // self.win_inc + 1
        
        # Frequency bins (remove DC component as per your model)
        self.freq_bins = self.fft_len // 2  # 256 bins for FFT=512
        
        print(f"Actual Model Dimensions:")
        print(f"  Total samples: {total_samples}")
        print(f"  Number of frames: {self.num_frames}")  
        print(f"  Frequency bins: {self.freq_bins}")
        print(f"  Binaural channels: {2 if self.config['binaural'] else 1}")
    
    def calculate_stft_macs(self) -> int:
        """Calculate STFT MACs based on actual implementation"""
        batch_size = self.config['batch_size']
        num_channels = 2 if self.config['binaural'] else 1
        
        # FFT for each frame  
        total_ffts = batch_size * num_channels * self.num_frames
        fft_macs = self.counter.count_fft_macs(self.fft_len, total_ffts, 1)
        
        # Windowing operations
        windowing_macs = batch_size * num_channels * self.num_frames * self.config['win_len']
        
        total_macs = fft_macs + windowing_macs
        self.counter.layer_macs['STFT'] = total_macs
        return total_macs
    
    def calculate_encoder_macs(self) -> int:
        """Calculate encoder MACs using actual depthwise separable structure"""
        batch_size = self.config['batch_size']
        kernel_num = [1] + [k//2 for k in self.config['kernel_num']]  # Convert to real channels
        kernel_size = self.config['kernel_size']
        
        total_macs = 0
        current_h, current_w = self.freq_bins, self.num_frames
        
        # For binaural, we have separate encoders
        num_encoders = 2 if self.config['binaural'] else 1
        
        print(f"Encoder layers analysis:")
        
        for layer_idx in range(len(kernel_num) - 1):
            in_channels = kernel_num[layer_idx]
            out_channels = kernel_num[layer_idx + 1]
            
            # Output dimensions after conv with stride=2 in frequency
            out_h = current_h // 2
            out_w = current_w
            
            # Depthwise separable convolution MACs (using corrected calculation)
            conv_macs = self.counter.count_complex_depthwise_separable_macs(
                (batch_size, in_channels, current_h, current_w),
                (batch_size, out_channels, out_h, out_w),
                (kernel_size, 2),
                in_channels, out_channels
            )
            
            # Batch normalization (2 BN layers: bn_r + bn_i, ~2 MACs per element each)
            bn_macs = batch_size * out_channels * out_h * out_w * 4  # 2 BN layers * 2 MACs each
            
            # PReLU activation (2 PReLU layers: r_prelu + i_prelu, 1 MAC each)
            prelu_macs = batch_size * out_channels * out_h * out_w * 2
            
            layer_macs = (conv_macs + bn_macs + prelu_macs) * num_encoders
            total_macs += layer_macs
            
            print(f"  Layer {layer_idx}: {in_channels}->{out_channels}, {conv_macs:,d} conv MACs")
            self.counter.layer_macs[f'Encoder_Layer_{layer_idx}'] = layer_macs
            
            current_h, current_w = out_h, out_w
        
        return total_macs
    
    def calculate_attention_macs(self) -> int:
        """Calculate attention MACs using actual complex attention structure"""
        batch_size = self.config['batch_size']
        embed_dim = self.config['embed_dim']
        num_heads = self.config['num_heads']
        
        # Calculate sequence length from final encoder dimensions
        final_channels = self.config['kernel_num'][-1] // 2  # Real channels
        final_h = self.freq_bins // (2 ** len(self.config['kernel_num']))
        final_w = self.num_frames
        
        input_size = final_channels * final_h
        if self.config['binaural']:
            input_size *= 2  # Concatenated left and right
            
        seq_len = final_w
        
        print(f"Attention dimensions: input_size={input_size}, seq_len={seq_len}")
        
        # Input projection (complex linear layer = 2 real linear layers)
        input_proj_macs = 2 * batch_size * seq_len * input_size * embed_dim
        
        # Complex multi-head attention (2 separate real attention blocks)
        attention_macs = self.counter.count_complex_attention_macs(
            embed_dim, num_heads, seq_len, batch_size
        )
        
        # Output projection (complex linear layer = 2 real linear layers)
        output_proj_macs = 2 * batch_size * seq_len * embed_dim * input_size
        
        total_macs = input_proj_macs + attention_macs + output_proj_macs
        self.counter.layer_macs['Attention'] = total_macs
        return total_macs
    
    def calculate_decoder_macs(self) -> int:
        """Calculate decoder MACs using actual depthwise separable structure"""
        batch_size = self.config['batch_size']
        kernel_num = [k//2 for k in self.config['kernel_num']]  # Convert to real channels
        kernel_size = self.config['kernel_size']
        
        total_macs = 0
        
        # Start from final encoder dimensions
        current_h = self.freq_bins // (2 ** len(kernel_num))
        current_w = self.num_frames
        
        num_decoders = 2 if self.config['binaural'] else 1
        
        print(f"Decoder layers analysis:")
        
        # Decoder layers (reverse of encoder)
        for layer_idx in range(len(kernel_num) - 1, 0, -1):
            # Skip connections double the input channels
            in_channels = kernel_num[layer_idx] * 2  # Skip connection
            out_channels = kernel_num[layer_idx - 1]
            
            # Output dimensions after transpose conv
            out_h = current_h * 2
            out_w = current_w
            
            # Depthwise separable transpose convolution
            # From your model: pointwise first, then depthwise
            
            # Pointwise conv (1x1) to change channels
            pointwise_macs = 2 * self.counter.count_real_conv2d_macs(
                (batch_size, in_channels, current_h, current_w),
                (batch_size, out_channels, current_h, current_w),
                (1, 1), in_channels, out_channels
            )
            
            # Depthwise transposed conv
            depthwise_macs = 2 * self.counter.count_real_conv2d_macs(
                (batch_size, out_channels, current_h, current_w),
                (batch_size, out_channels, out_h, out_w),
                (kernel_size, 2), out_channels, out_channels, is_depthwise=True
            )
            
            conv_macs = pointwise_macs + depthwise_macs
            
            # Batch norm and activation (except last layer)
            if layer_idx != 1:
                bn_macs = batch_size * out_channels * out_h * out_w * 4
                prelu_macs = batch_size * out_channels * out_h * out_w * 2
            else:
                bn_macs = prelu_macs = 0
            
            layer_macs = (conv_macs + bn_macs + prelu_macs) * num_decoders
            total_macs += layer_macs
            
            print(f"  Layer {layer_idx}: {in_channels}->{out_channels}, {conv_macs:,d} conv MACs")
            self.counter.layer_macs[f'Decoder_Layer_{layer_idx}'] = layer_macs
            
            current_h, current_w = out_h, out_w
        
        return total_macs
    
    def calculate_istft_macs(self) -> int:
        """Calculate ISTFT MACs"""
        batch_size = self.config['batch_size']
        num_channels = 2 if self.config['binaural'] else 1
        
        # IFFT for each frame
        total_iffts = batch_size * num_channels * self.num_frames
        ifft_macs = self.counter.count_fft_macs(self.fft_len, total_iffts, 1)
        
        # Overlap-add operations
        overlap_add_macs = batch_size * num_channels * self.num_frames * self.config['win_len']
        
        total_macs = ifft_macs + overlap_add_macs
        self.counter.layer_macs['ISTFT'] = total_macs
        return total_macs
    
    def calculate_total_macs(self) -> Dict:
        """Calculate total MAC operations based on actual model structure"""
        self.counter.reset()
        
        print("Calculating MACs for Actual Model Structure...")
        print("=" * 50)
        
        stft_macs = self.calculate_stft_macs()
        encoder_macs = self.calculate_encoder_macs()
        attention_macs = self.calculate_attention_macs()
        decoder_macs = self.calculate_decoder_macs()
        istft_macs = self.calculate_istft_macs()
        
        total_macs = stft_macs + encoder_macs + attention_macs + decoder_macs + istft_macs
        
        breakdown = {
            'STFT': stft_macs,
            'Encoder': encoder_macs,
            'Attention': attention_macs,
            'Decoder': decoder_macs,
            'ISTFT': istft_macs,
            'Total': total_macs,
            'Layer_Details': self.counter.layer_macs
        }
        
        return breakdown
    
    def print_results(self, breakdown: Dict):
        """Print corrected MAC analysis results"""
        print("\nCorrected MAC Operations Analysis")
        print("=" * 50)
        
        print(f"\nMain Components:")
        for component in ['STFT', 'Encoder', 'Attention', 'Decoder', 'ISTFT']:
            macs = breakdown[component]
            percentage = (macs / breakdown['Total']) * 100
            print(f"  {component:12s}: {macs:15,d} MACs ({percentage:5.1f}%)")
        
        print(f"\nTotal MACs: {breakdown['Total']:,d}")
        print(f"Total GMACs: {breakdown['Total'] / 1e9:.3f}")
        
        # Performance estimates with corrected values
        print(f"\nPerformance Estimates:")
        print(f"  At 1 GMAC/s:  {breakdown['Total'] / 1e9:.3f} seconds")
        print(f"  At 10 GMAC/s: {breakdown['Total'] / 1e10:.4f} seconds")
        print(f"  At 100 GMAC/s: {breakdown['Total'] / 1e11:.5f} seconds")
        
        # Real-time factor
        rtf_10gmac = (breakdown['Total'] / 1e10) / self.config['duration_seconds']
        rtf_100gmac = (breakdown['Total'] / 1e11) / self.config['duration_seconds']
        
        print(f"\nReal-time Factor (RTF):")
        print(f"  At 10 GMAC/s:  {rtf_10gmac:.4f}")
        print(f"  At 100 GMAC/s: {rtf_100gmac:.5f}")


def compare_original_vs_corrected():
    """Compare original estimation vs corrected calculation"""
    
    print("COMPARISON: Original vs Corrected MAC Calculation")
    print("=" * 60)
    
    # Configuration from actual model
    config = {
        'fft_len': 512,
        'win_len': 400,
        'win_inc': 100,
        'kernel_num': [16, 32, 64, 128, 256, 256],
        'kernel_size': 5,
        'embed_dim': 512,
        'num_heads': 32,
        'binaural': True,
        'use_depthwise': True,
        'batch_size': 1,
        'duration_seconds': 2.0,
        'sample_rate': 16000
    }
    
    # Calculate with corrected method
    corrected_calc = ActualModelMACCalculator(config)
    corrected_results = corrected_calc.calculate_total_macs()
    corrected_calc.print_results(corrected_results)
    
    print(f"\n" + "="*60)
    print("ANALYSIS OF DIFFERENCES")
    print("="*60)
    
    print(f"The corrected MAC calculation accounts for:")
    print(f"1. ✓ Complex ops as 2 separate real ops (not 4x multiplier)")
    print(f"2. ✓ Actual depthwise + pointwise structure")
    print(f"3. ✓ Correct attention implementation (2 separate blocks)")
    print(f"4. ✓ Proper FFT sizing and frame calculations")
    
    corrected_total = corrected_results['Total']
    print(f"\nCorrected Total MACs: {corrected_total:,d} ({corrected_total/1e9:.2f} GMACs)")
    
    # Real-time feasibility with corrected numbers
    min_throughput = corrected_total / config['duration_seconds'] / 1e9
    print(f"Minimum throughput for real-time: {min_throughput:.1f} GMAC/s")
    
    if min_throughput < 5:
        print("✓ Feasible on mobile CPU")
    elif min_throughput < 15:
        print("✓ Feasible on desktop CPU")
    elif min_throughput < 50:
        print("✓ Requires mobile GPU")
    else:
        print("⚠ Requires desktop GPU")
    
    return corrected_results


if __name__ == "__main__":
    results = compare_original_vs_corrected()
