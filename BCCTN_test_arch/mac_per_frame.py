#!/usr/bin/env python3
"""
MAC Calculator Per Frame
Calculates MAC operations for processing a single frame of audio
"""

import torch
import numpy as np
from typing import Dict, Tuple
import math


class MACPerFrameCounter:
    """MAC counter for single frame processing"""
    
    def __init__(self):
        self.mac_count = 0
        self.layer_macs = {}
    
    def reset(self):
        self.mac_count = 0
        self.layer_macs = {}
    
    def count_real_conv2d_macs_per_frame(self, input_shape: Tuple[int, ...], 
                                        output_shape: Tuple[int, ...],
                                        kernel_size: Tuple[int, int],
                                        in_channels: int, out_channels: int,
                                        is_depthwise: bool = False) -> int:
        """
        Count MAC operations for REAL convolution per frame
        Note: input/output shapes should NOT include time dimension multiplication
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
    
    def count_fft_macs_per_frame(self, n: int, batch_size: int, num_channels: int = 1) -> int:
        """Count MAC operations for FFT per frame"""
        # FFT complexity: O(n * log2(n))
        # Each complex multiplication = 4 real MACs + 2 real additions
        fft_ops_per_transform = n * math.log2(n) * 6  # 4 mults + 2 adds
        total_transforms = batch_size * num_channels
        return int(fft_ops_per_transform * total_transforms)


class ActualModelMACPerFrame:
    """MAC calculator per frame based on your actual model structure"""
    
    def __init__(self, config: Dict = None):
        self.counter = MACPerFrameCounter()
        
        # Configuration from your actual model
        self.config = config or {
            'fft_len': 256,           # From your output
            'win_len': 256,           # From your output  
            'win_inc': 80,           # From your output (hop size)
            'kernel_num': [16, 32, 64, 128, 256, 256],  # From your output
            'kernel_size': 5,
            'embed_dim': 512,
            'num_heads': 32,
            'binaural': True,
            'use_depthwise': True,
            'batch_size': 1,
            'sample_rate': 16000
        }
        
        if config:
            self.config.update(config)
            
        self._calculate_dimensions()
    
    def _calculate_dimensions(self):
        """Calculate dimensions for single frame processing"""
        self.fft_len = self.config['fft_len']
        self.win_inc = self.config['win_inc']
        
        # For single frame analysis, we consider the processing of one frame
        # Frequency bins (remove DC component as per your model)
        self.freq_bins = self.fft_len // 2  # 256 bins for FFT=512
        
        # For streaming/real-time processing, each frame processes one time step
        # But the model sees a sequence of frames in its internal buffer
        # Let's assume a reasonable buffer size for the transformer
        self.buffer_frames = 32  # Typical buffer size for real-time processing
        
        print(f"Single Frame Processing Dimensions:")
        print(f"  FFT length: {self.fft_len}")
        print(f"  Frequency bins: {self.freq_bins}")
        print(f"  Hop size: {self.win_inc} samples")
        print(f"  Transformer buffer: {self.buffer_frames} frames")
        print(f"  Binaural channels: {2 if self.config['binaural'] else 1}")
    
    def calculate_stft_macs_per_frame(self) -> int:
        """Calculate STFT MACs for processing one frame"""
        batch_size = self.config['batch_size']
        num_channels = 2 if self.config['binaural'] else 1
        
        # For each new frame, we need:
        # 1. One FFT per channel
        fft_macs = self.counter.count_fft_macs_per_frame(self.fft_len, batch_size, num_channels)
        
        # 2. Windowing operation for the new frame
        windowing_macs = batch_size * num_channels * self.config['win_len']
        
        total_macs = fft_macs + windowing_macs
        self.counter.layer_macs['STFT_per_frame'] = total_macs
        return total_macs
    
    def calculate_encoder_macs_per_frame(self) -> int:
        """Calculate encoder MACs for one frame (full feature extraction)"""
        batch_size = self.config['batch_size']
        kernel_num = [1] + [k//2 for k in self.config['kernel_num']]  # Convert to real channels
        kernel_size = self.config['kernel_size']
        
        total_macs = 0
        current_h, current_w = self.freq_bins, 1  # Process one time step
        
        # For binaural, we have separate encoders
        num_encoders = 2 if self.config['binaural'] else 1
        
        print(f"\nEncoder per-frame analysis:")
        
        for layer_idx in range(len(kernel_num) - 1):
            in_channels = kernel_num[layer_idx]
            out_channels = kernel_num[layer_idx + 1]
            
            # Output dimensions after conv with stride=2 in frequency
            out_h = current_h // 2
            out_w = current_w  # Time dimension stays 1 for per-frame
            
            # Depthwise separable convolution MACs
            # 1. Depthwise conv (separate _r and _i)
            depthwise_macs_r = self.counter.count_real_conv2d_macs_per_frame(
                (batch_size, in_channels, current_h, current_w),
                (batch_size, in_channels, out_h, out_w),
                (kernel_size, 2), in_channels, in_channels, is_depthwise=True
            )
            depthwise_macs_i = depthwise_macs_r
            
            # 2. Pointwise conv (separate _r and _i)
            pointwise_macs_r = self.counter.count_real_conv2d_macs_per_frame(
                (batch_size, in_channels, out_h, out_w),
                (batch_size, out_channels, out_h, out_w),
                (1, 1), in_channels, out_channels, is_depthwise=False
            )
            pointwise_macs_i = pointwise_macs_r
            
            conv_macs = depthwise_macs_r + depthwise_macs_i + pointwise_macs_r + pointwise_macs_i
            
            # Batch normalization (2 BN layers: bn_r + bn_i, ~2 MACs per element each)
            bn_macs = batch_size * out_channels * out_h * out_w * 4
            
            # PReLU activation (2 PReLU layers: r_prelu + i_prelu, 1 MAC each)
            prelu_macs = batch_size * out_channels * out_h * out_w * 2
            
            layer_macs = (conv_macs + bn_macs + prelu_macs) * num_encoders
            total_macs += layer_macs
            
            print(f"  Layer {layer_idx}: {in_channels}->{out_channels}, {layer_macs:,d} MACs")
            self.counter.layer_macs[f'Encoder_Layer_{layer_idx}_per_frame'] = layer_macs
            
            current_h, current_w = out_h, out_w
        
        return total_macs
    
    def calculate_attention_macs_per_frame(self) -> int:
        """
        Calculate attention MACs per frame
        Note: Attention processes a buffer of frames, not just one
        """
        batch_size = self.config['batch_size']
        embed_dim = self.config['embed_dim']
        num_heads = self.config['num_heads']
        
        # Calculate dimensions after encoder
        final_channels = self.config['kernel_num'][-1] // 2  # Real channels
        final_h = self.freq_bins // (2 ** len(self.config['kernel_num']))
        
        input_size = final_channels * final_h
        if self.config['binaural']:
            input_size *= 2  # Concatenated left and right
            
        # For streaming, attention processes a buffer of frames
        seq_len = self.buffer_frames
        
        print(f"\nAttention per-frame analysis:")
        print(f"  Input size: {input_size}")
        print(f"  Sequence length (buffer): {seq_len}")
        print(f"  Embed dim: {embed_dim}")
        
        # Input projection (complex linear = 2 real linear layers)
        input_proj_macs = 2 * batch_size * seq_len * input_size * embed_dim
        
        # Multi-head attention computation
        head_dim = embed_dim // num_heads
        
        # Q, K, V projections (2 separate attention blocks: _re and _im)
        qkv_macs = 2 * 3 * batch_size * seq_len * embed_dim * embed_dim
        
        # Attention computation: Q @ K^T (2 blocks)
        attn_macs = 2 * batch_size * num_heads * seq_len * seq_len * head_dim
        
        # Attention @ V (2 blocks)
        attn_v_macs = 2 * batch_size * num_heads * seq_len * seq_len * head_dim
        
        # Output projection (2 blocks)
        attn_out_macs = 2 * batch_size * seq_len * embed_dim * embed_dim
        
        attention_core_macs = qkv_macs + attn_macs + attn_v_macs + attn_out_macs
        
        # Output projection (complex linear = 2 real linear layers)
        output_proj_macs = 2 * batch_size * seq_len * embed_dim * input_size
        
        total_macs = input_proj_macs + attention_core_macs + output_proj_macs
        
        # Amortize over the buffer size (MACs per frame)
        macs_per_frame = total_macs // seq_len
        
        print(f"  Total attention MACs for buffer: {total_macs:,d}")
        print(f"  Amortized MACs per frame: {macs_per_frame:,d}")
        
        self.counter.layer_macs['Attention_per_frame'] = macs_per_frame
        return macs_per_frame
    
    def calculate_decoder_macs_per_frame(self) -> int:
        """Calculate decoder MACs for one frame"""
        batch_size = self.config['batch_size']
        kernel_num = [k//2 for k in self.config['kernel_num']]  # Convert to real channels
        kernel_size = self.config['kernel_size']
        
        total_macs = 0
        
        # Start from final encoder dimensions
        current_h = self.freq_bins // (2 ** len(kernel_num))
        current_w = 1  # One time step
        
        num_decoders = 2 if self.config['binaural'] else 1
        
        print(f"\nDecoder per-frame analysis:")
        
        # Decoder layers (reverse of encoder)
        for layer_idx in range(len(kernel_num) - 1, 0, -1):
            # Skip connections double the input channels
            in_channels = kernel_num[layer_idx] * 2  # Skip connection
            out_channels = kernel_num[layer_idx - 1]
            
            # Output dimensions after transpose conv
            out_h = current_h * 2
            out_w = current_w
            
            # Depthwise separable transpose convolution
            # 1. Pointwise conv (1x1) to change channels (_r and _i)
            pointwise_macs_r = self.counter.count_real_conv2d_macs_per_frame(
                (batch_size, in_channels, current_h, current_w),
                (batch_size, out_channels, current_h, current_w),
                (1, 1), in_channels, out_channels
            )
            pointwise_macs_i = pointwise_macs_r
            
            # 2. Depthwise transposed conv (_r and _i)
            depthwise_macs_r = self.counter.count_real_conv2d_macs_per_frame(
                (batch_size, out_channels, current_h, current_w),
                (batch_size, out_channels, out_h, out_w),
                (kernel_size, 2), out_channels, out_channels, is_depthwise=True
            )
            depthwise_macs_i = depthwise_macs_r
            
            conv_macs = pointwise_macs_r + pointwise_macs_i + depthwise_macs_r + depthwise_macs_i
            
            # Batch norm and activation (except last layer)
            if layer_idx != 1:
                bn_macs = batch_size * out_channels * out_h * out_w * 4
                prelu_macs = batch_size * out_channels * out_h * out_w * 2
            else:
                bn_macs = prelu_macs = 0
            
            layer_macs = (conv_macs + bn_macs + prelu_macs) * num_decoders
            total_macs += layer_macs
            
            print(f"  Layer {layer_idx}: {in_channels}->{out_channels}, {layer_macs:,d} MACs")
            self.counter.layer_macs[f'Decoder_Layer_{layer_idx}_per_frame'] = layer_macs
            
            current_h, current_w = out_h, out_w
        
        return total_macs
    
    def calculate_istft_macs_per_frame(self) -> int:
        """Calculate ISTFT MACs for one frame output"""
        batch_size = self.config['batch_size']
        num_channels = 2 if self.config['binaural'] else 1
        
        # For each frame output, we need:
        # 1. One IFFT per channel
        ifft_macs = self.counter.count_fft_macs_per_frame(self.fft_len, batch_size, num_channels)
        
        # 2. Overlap-add operation
        overlap_add_macs = batch_size * num_channels * self.config['win_len']
        
        total_macs = ifft_macs + overlap_add_macs
        self.counter.layer_macs['ISTFT_per_frame'] = total_macs
        return total_macs
    
    def calculate_total_macs_per_frame(self) -> Dict:
        """Calculate total MAC operations per frame"""
        self.counter.reset()
        
        print("Calculating MACs Per Frame...")
        print("=" * 50)
        
        stft_macs = self.calculate_stft_macs_per_frame()
        encoder_macs = self.calculate_encoder_macs_per_frame()
        attention_macs = self.calculate_attention_macs_per_frame()
        decoder_macs = self.calculate_decoder_macs_per_frame()
        istft_macs = self.calculate_istft_macs_per_frame()
        
        total_macs = stft_macs + encoder_macs + attention_macs + decoder_macs + istft_macs
        
        breakdown = {
            'STFT_per_frame': stft_macs,
            'Encoder_per_frame': encoder_macs,
            'Attention_per_frame': attention_macs,
            'Decoder_per_frame': decoder_macs,
            'ISTFT_per_frame': istft_macs,
            'Total_per_frame': total_macs,
            'Layer_Details': self.counter.layer_macs
        }
        
        return breakdown
    
    def print_results(self, breakdown: Dict):
        """Print per-frame MAC analysis results"""
        print(f"\nMAC Operations Per Frame Analysis")
        print("=" * 50)
        
        total_per_frame = breakdown['Total_per_frame']
        
        print(f"\nPer-Frame Component Breakdown:")
        components = ['STFT_per_frame', 'Encoder_per_frame', 'Attention_per_frame', 
                     'Decoder_per_frame', 'ISTFT_per_frame']
        for component in components:
            macs = breakdown[component]
            percentage = (macs / total_per_frame) * 100
            comp_name = component.replace('_per_frame', '')
            print(f"  {comp_name:12s}: {macs:12,d} MACs ({percentage:5.1f}%)")
        
        print(f"\nTotal MACs per frame: {total_per_frame:,d}")
        print(f"Total MACs per frame: {total_per_frame / 1e6:.3f} MMACs")
        
        # Calculate throughput requirements
        hop_size_ms = self.config['win_inc'] / self.config['sample_rate'] * 1000
        frames_per_second = 1000 / hop_size_ms
        macs_per_second = total_per_frame * frames_per_second
        
        print(f"\nReal-time Requirements:")
        print(f"  Hop size: {self.config['win_inc']} samples ({hop_size_ms:.1f} ms)")
        print(f"  Frames per second: {frames_per_second:.1f}")
        print(f"  MACs per second: {macs_per_second:,.0f} ({macs_per_second/1e9:.3f} GMAC/s)")
        
        # Hardware feasibility
        print(f"\nHardware Feasibility:")
        hardware_specs = [
            ("Mobile CPU", 1e9),
            ("Desktop CPU", 10e9),
            ("Mobile GPU", 50e9),
            ("Desktop GPU", 100e9),
        ]
        
        for hw_name, throughput in hardware_specs:
            utilization = (macs_per_second / throughput) * 100
            status = "✓" if utilization < 80 else "⚠" if utilization < 150 else "✗"
            print(f"  {hw_name:12s} ({throughput/1e9:3.0f} GMAC/s): {utilization:5.1f}% utilization {status}")


def main():
    """Run per-frame MAC analysis"""
    
    print("BCCTN MAC Analysis Per Frame")
    print("Based on Actual Model Structure")
    print("=" * 60)
    
    # Configuration from your actual model
    config = {
        'fft_len': 256,
        'win_len': 256,
        'win_inc': 80,
        'kernel_num': [16, 32, 64, 128, 256, 256],
        'kernel_size': 5,
        'embed_dim': 512,
        'num_heads': 32,
        'binaural': True,
        'use_depthwise': True,
        'batch_size': 1,
        'sample_rate': 16000
    }
    
    # Calculate per-frame MACs
    calculator = ActualModelMACPerFrame(config)
    results = calculator.calculate_total_macs_per_frame()
    calculator.print_results(results)
    
    print(f"\n" + "="*60)
    print("KEY INSIGHTS")
    print("="*60)
    
    total_per_frame = results['Total_per_frame']
    hop_size_ms = config['win_inc'] / config['sample_rate'] * 1000
    
    print(f"✓ Per-frame processing: {total_per_frame:,d} MACs")
    print(f"✓ Frame interval: {hop_size_ms:.1f} ms")
    print(f"✓ Real-time throughput needed: {total_per_frame * 1000 / hop_size_ms / 1e9:.2f} GMAC/s")
    
    if total_per_frame < 1e6:
        print(f"✓ Very efficient per-frame processing")
    elif total_per_frame < 5e6:
        print(f"✓ Moderate per-frame processing")
    else:
        print(f"⚠ High per-frame processing requirements")
    
    return results


if __name__ == "__main__":
    results = main()
