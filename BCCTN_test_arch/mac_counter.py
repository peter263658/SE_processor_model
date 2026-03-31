"""
MAC Operations Counter for Binaural Complex Convolutional Transformer Networks (BCCTN)
Calculates the total MAC count for full inference with FFT=256
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
import math


class MACCounter:
    """Calculate MAC operations for neural network inference"""
    
    def __init__(self):
        self.mac_count = 0
        self.layer_macs = {}
        self.detailed_breakdown = {}
    
    def reset(self):
        """Reset MAC counter"""
        self.mac_count = 0
        self.layer_macs = {}
        self.detailed_breakdown = {}
    
    def count_conv2d_macs(self, input_shape: Tuple[int, ...], 
                         output_shape: Tuple[int, ...],
                         kernel_size: Tuple[int, int],
                         in_channels: int, out_channels: int,
                         is_complex: bool = False, is_depthwise: bool = False) -> int:
        """
        Count MAC operations for 2D convolution
        
        Args:
            input_shape: (batch, channels, height, width)
            output_shape: (batch, channels, height, width) 
            kernel_size: (kernel_h, kernel_w)
            in_channels: Number of input channels
            out_channels: Number of output channels
            is_complex: Whether this is complex convolution
            is_depthwise: Whether this is depthwise convolution
            
        Returns:
            Number of MAC operations
        """
        batch_size, _, out_h, out_w = output_shape
        kernel_h, kernel_w = kernel_size
        
        if is_depthwise:
            # Depthwise convolution: each input channel is convolved separately
            macs_per_output = kernel_h * kernel_w  # Only 1 input channel per output channel
            total_outputs = batch_size * in_channels * out_h * out_w  # in_channels outputs
            base_macs = macs_per_output * total_outputs
        else:
            # Regular convolution
            macs_per_output = in_channels * kernel_h * kernel_w
            total_outputs = batch_size * out_channels * out_h * out_w
            base_macs = macs_per_output * total_outputs
        
        # Complex convolution requires 4x real convolutions
        # (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
        # This requires 4 real multiplications and 2 real additions
        if is_complex:
            base_macs *= 4
            
        return base_macs
    
    def count_depthwise_separable_macs(self, input_shape: Tuple[int, ...],
                                     output_shape: Tuple[int, ...],
                                     kernel_size: Tuple[int, int],
                                     in_channels: int, out_channels: int,
                                     is_complex: bool = False) -> int:
        """
        Count MAC operations for depthwise separable convolution
        
        Args:
            input_shape: (batch, channels, height, width)
            output_shape: (batch, channels, height, width)
            kernel_size: (kernel_h, kernel_w) for depthwise conv
            in_channels: Number of input channels
            out_channels: Number of output channels  
            is_complex: Whether this is complex convolution
            
        Returns:
            Number of MAC operations
        """
        batch_size, _, out_h, out_w = output_shape
        kernel_h, kernel_w = kernel_size
        
        # Depthwise convolution: each input channel convolved separately
        depthwise_macs = self.count_conv2d_macs(
            input_shape, (batch_size, in_channels, out_h, out_w),
            kernel_size, in_channels, in_channels, is_complex, is_depthwise=True
        )
        
        # Pointwise convolution: 1x1 conv to change channel dimension
        pointwise_macs = self.count_conv2d_macs(
            (batch_size, in_channels, out_h, out_w), output_shape,
            (1, 1), in_channels, out_channels, is_complex, is_depthwise=False
        )
        
        return depthwise_macs + pointwise_macs
    
    def count_linear_macs(self, input_features: int, output_features: int, 
                         batch_size: int, seq_len: int = 1, 
                         is_complex: bool = False) -> int:
        """
        Count MAC operations for linear layer
        
        Args:
            input_features: Number of input features
            output_features: Number of output features
            batch_size: Batch size
            seq_len: Sequence length (for attention)
            is_complex: Whether this is complex linear layer
            
        Returns:
            Number of MAC operations
        """
        base_macs = batch_size * seq_len * input_features * output_features
        
        if is_complex:
            base_macs *= 4  # Complex multiplication
            
        return base_macs
    
    def count_attention_macs(self, embed_dim: int, num_heads: int, 
                           seq_len: int, batch_size: int,
                           is_complex: bool = False) -> int:
        """
        Count MAC operations for multi-head attention
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            seq_len: Sequence length
            batch_size: Batch size
            is_complex: Whether this is complex attention
            
        Returns:
            Number of MAC operations
        """
        head_dim = embed_dim // num_heads
        
        # Q, K, V projections
        qkv_macs = 3 * self.count_linear_macs(embed_dim, embed_dim, batch_size, seq_len, is_complex)
        
        # Attention computation: Q @ K^T
        attn_macs = batch_size * num_heads * seq_len * seq_len * head_dim
        
        # Attention @ V
        attn_v_macs = batch_size * num_heads * seq_len * seq_len * head_dim
        
        # Output projection
        out_proj_macs = self.count_linear_macs(embed_dim, embed_dim, batch_size, seq_len, is_complex)
        
        total_macs = qkv_macs + attn_macs + attn_v_macs + out_proj_macs
        
        if is_complex:
            # Additional complexity for complex attention operations
            total_macs = int(total_macs * 1.5)  # Approximate factor for complex operations
            
        return total_macs
    
    def count_fft_macs(self, n: int, batch_size: int, num_channels: int = 1) -> int:
        """
        Count MAC operations for FFT
        
        Args:
            n: FFT size
            batch_size: Batch size
            num_channels: Number of channels
            
        Returns:
            Number of MAC operations
        """
        # FFT complexity: O(n * log2(n))
        # Each complex multiplication counts as 4 real MACs
        fft_ops_per_transform = n * math.log2(n) * 4
        total_transforms = batch_size * num_channels
        
        return int(fft_ops_per_transform * total_transforms)


class BCCTNMACCalculator:
    """MAC calculator specifically for BCCTN model"""
    
    def __init__(self, config: Dict = None):
        self.counter = MACCounter()
        
        # Updated BCCTN configuration to match your new settings
        self.config = config or {
            'fft_len': 256,           # Updated from 512 to 256
            'win_len': 256,           # Updated from 400 to 256
            'win_inc': 80,            # Updated from 100 to 80
            'kernel_num': [2, 16, 32, 64, 128, 256, 256],  # Updated from [2, 16, 32, 64, 128, 256, 256]
            'kernel_size': 5,
            'embed_dim': 512,
            'num_heads': 32,
            'rnn_units': 128,
            'binaural': True,
            'batch_size': 1,  # Single inference
            'duration_seconds': 2.0,  # 2 seconds audio
            'sample_rate': 16000,
            'use_depthwise': True     # New parameter for depthwise separable conv
        }
        
        # Override with provided config
        if config:
            self.config.update(config)
        
        # Calculate input dimensions
        self._calculate_dimensions()
    
    def _calculate_dimensions(self):
        """Calculate various dimensions based on config"""
        self.fft_len = self.config['fft_len']
        self.win_inc = self.config['win_inc']
        self.sample_rate = self.config['sample_rate']
        self.duration = self.config['duration_seconds']
        
        # Calculate number of frames
        total_samples = int(self.duration * self.sample_rate)
        self.num_frames = (total_samples - self.config['win_len']) // self.win_inc + 1
        
        # Frequency bins (one-sided spectrum)
        self.freq_bins = self.fft_len // 2 + 1 - 1  # Remove DC component as per your code
        
        print(f"Input dimensions:")
        print(f"  Total samples: {total_samples}")
        print(f"  Number of frames: {self.num_frames}")
        print(f"  Frequency bins: {self.freq_bins}")
        print(f"  Binaural channels: {2 if self.config['binaural'] else 1}")
    
    def calculate_stft_macs(self) -> int:
        """Calculate MAC operations for STFT"""
        batch_size = self.config['batch_size']
        num_channels = 2 if self.config['binaural'] else 1
        
        # FFT for each frame
        total_ffts = batch_size * num_channels * self.num_frames
        fft_macs = self.counter.count_fft_macs(self.fft_len, total_ffts, 1)
        
        # Windowing operations (multiply by window function)
        windowing_macs = batch_size * num_channels * self.num_frames * self.config['win_len']
        
        total_macs = fft_macs + windowing_macs
        self.counter.layer_macs['STFT'] = total_macs
        
        return total_macs
    
    def calculate_encoder_macs(self) -> int:
        """Calculate MAC operations for encoder with depthwise separable conv support"""
        batch_size = self.config['batch_size']
        kernel_num = self.config['kernel_num']
        kernel_size = self.config['kernel_size']
        use_depthwise = self.config.get('use_depthwise', True)
        
        total_macs = 0
        current_h, current_w = self.freq_bins, self.num_frames
        
        # For binaural, we have separate encoders for left and right
        num_encoders = 2 if self.config['binaural'] else 1
        
        print(f"Encoder using {'depthwise separable' if use_depthwise else 'regular'} convolutions")
        
        for layer_idx in range(len(kernel_num) - 1):
            in_channels = kernel_num[layer_idx] // 2  # Complex channels
            out_channels = kernel_num[layer_idx + 1] // 2
            
            # Output dimensions after conv with stride=2 in frequency
            out_h = current_h // 2
            out_w = current_w  # No stride in time
            
            if use_depthwise:
                # Depthwise separable convolution MACs
                conv_macs = self.counter.count_depthwise_separable_macs(
                    (batch_size, in_channels, current_h, current_w),
                    (batch_size, out_channels, out_h, out_w),
                    (kernel_size, 2),  # kernel size from your code
                    in_channels, out_channels,
                    is_complex=True
                )
            else:
                # Regular convolution MACs
                conv_macs = self.counter.count_conv2d_macs(
                    (batch_size, in_channels, current_h, current_w),
                    (batch_size, out_channels, out_h, out_w),
                    (kernel_size, 2),  # kernel size from your code
                    in_channels, out_channels,
                    is_complex=True
                )
            
            # Batch normalization (approximately 2 MACs per element)
            bn_macs = batch_size * out_channels * out_h * out_w * 2
            
            # PReLU activation (1 MAC per element)
            prelu_macs = batch_size * out_channels * out_h * out_w
            
            layer_macs = (conv_macs + bn_macs + prelu_macs) * num_encoders
            total_macs += layer_macs
            
            conv_type = "DepthwiseSep" if use_depthwise else "Regular"
            self.counter.layer_macs[f'Encoder_{conv_type}_Layer_{layer_idx}'] = layer_macs
            
            # Update dimensions for next layer
            current_h, current_w = out_h, out_w
        
        return total_macs
    
    def calculate_attention_macs(self) -> int:
        """Calculate MAC operations for transformer attention"""
        batch_size = self.config['batch_size']
        embed_dim = self.config['embed_dim']
        num_heads = self.config['num_heads']
        
        # Calculate flattened sequence length
        # After encoder, we have concatenated left and right features
        final_channels = self.config['kernel_num'][-1]  # 256 from config
        final_h = self.freq_bins // (2 ** (len(self.config['kernel_num']) - 1))
        final_w = self.num_frames
        
        # Input size to attention (flattened spatial dimensions)
        input_size = final_channels * final_h
        seq_len = final_w
        
        if self.config['binaural']:
            input_size *= 2  # Concatenated left and right
        
        print(f"Attention input size: {input_size}, seq_len: {seq_len}")
        
        # Input projection to embed_dim
        input_proj_macs = self.counter.count_linear_macs(
            input_size, embed_dim, batch_size, seq_len, is_complex=True
        )
        
        # Multi-head attention
        attention_macs = self.counter.count_attention_macs(
            embed_dim, num_heads, seq_len, batch_size, is_complex=True
        )
        
        # Output projection back to input size
        output_proj_macs = self.counter.count_linear_macs(
            embed_dim, input_size, batch_size, seq_len, is_complex=True
        )
        
        total_macs = input_proj_macs + attention_macs + output_proj_macs
        self.counter.layer_macs['Attention'] = total_macs
        
        return total_macs
    
    def calculate_decoder_macs(self) -> int:
        """Calculate MAC operations for decoder with depthwise separable conv support"""
        batch_size = self.config['batch_size']
        kernel_num = self.config['kernel_num']
        kernel_size = self.config['kernel_size']
        use_depthwise = self.config.get('use_depthwise', True)
        
        total_macs = 0
        
        # Start from the final encoder dimensions
        current_h = self.freq_bins // (2 ** (len(kernel_num) - 1))
        current_w = self.num_frames
        
        # For binaural, we have separate decoders for left and right
        num_decoders = 2 if self.config['binaural'] else 1
        
        print(f"Decoder using {'depthwise separable' if use_depthwise else 'regular'} transposed convolutions")
        
        # Decoder layers (reverse of encoder)
        for layer_idx in range(len(kernel_num) - 1, 0, -1):
            # Skip connections double the input channels
            in_channels = kernel_num[layer_idx]  # Includes skip connection
            out_channels = kernel_num[layer_idx - 1] // 2
            
            # Output dimensions after transpose conv with stride=2 in frequency
            out_h = current_h * 2
            out_w = current_w
            
            if use_depthwise:
                # For depthwise separable transposed conv:
                # 1. First pointwise conv to change channels
                # 2. Then depthwise transposed conv
                
                # Pointwise conv (1x1) to change channels from in_channels to out_channels
                pointwise_macs = self.counter.count_conv2d_macs(
                    (batch_size, in_channels, current_h, current_w),
                    (batch_size, out_channels, current_h, current_w),
                    (1, 1), in_channels, out_channels, is_complex=True
                )
                
                # Depthwise transposed conv to upsample
                depthwise_macs = self.counter.count_conv2d_macs(
                    (batch_size, out_channels, current_h, current_w),
                    (batch_size, out_channels, out_h, out_w),
                    (kernel_size, 2), out_channels, out_channels, 
                    is_complex=True, is_depthwise=True
                )
                
                conv_macs = pointwise_macs + depthwise_macs
            else:
                # Regular transpose convolution MACs
                conv_macs = self.counter.count_conv2d_macs(
                    (batch_size, in_channels, current_h, current_w),
                    (batch_size, out_channels, out_h, out_w),
                    (kernel_size, 2),
                    in_channels, out_channels,
                    is_complex=True
                )
            
            # Batch normalization and activation (except for last layer)
            if layer_idx != 1:
                bn_macs = batch_size * out_channels * out_h * out_w * 2
                prelu_macs = batch_size * out_channels * out_h * out_w
            else:
                bn_macs = prelu_macs = 0
            
            layer_macs = (conv_macs + bn_macs + prelu_macs) * num_decoders
            total_macs += layer_macs
            
            conv_type = "DepthwiseSep" if use_depthwise else "Regular"
            self.counter.layer_macs[f'Decoder_{conv_type}_Layer_{layer_idx}'] = layer_macs
            
            # Update dimensions for next layer
            current_h, current_w = out_h, out_w
        
        return total_macs
    
    def calculate_istft_macs(self) -> int:
        """Calculate MAC operations for ISTFT"""
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
        """Calculate total MAC operations for full inference"""
        self.counter.reset()
        
        print("Calculating MAC operations for BCCTN inference...")
        print("=" * 50)
        
        # Calculate MACs for each component
        stft_macs = self.calculate_stft_macs()
        encoder_macs = self.calculate_encoder_macs()
        attention_macs = self.calculate_attention_macs()
        decoder_macs = self.calculate_decoder_macs()
        istft_macs = self.calculate_istft_macs()
        
        total_macs = stft_macs + encoder_macs + attention_macs + decoder_macs + istft_macs
        
        # Create detailed breakdown
        breakdown = {
            'STFT': stft_macs,
            'Encoder': encoder_macs,
            'Attention': attention_macs,
            'Decoder': decoder_macs,
            'ISTFT': istft_macs,
            'Total': total_macs
        }
        
        # Add layer-wise breakdown
        breakdown['Layer_Details'] = self.counter.layer_macs
        
        return breakdown
    
    def print_results(self, breakdown: Dict):
        """Print detailed MAC analysis results"""
        print("\nMAC Operations Analysis Results")
        print("=" * 50)
        
        print(f"\nMain Components:")
        for component in ['STFT', 'Encoder', 'Attention', 'Decoder', 'ISTFT']:
            macs = breakdown[component]
            percentage = (macs / breakdown['Total']) * 100
            print(f"  {component:12s}: {macs:15,d} MACs ({percentage:5.1f}%)")
        
        print(f"\nTotal MACs: {breakdown['Total']:,d}")
        print(f"Total GMACs: {breakdown['Total'] / 1e9:.3f}")
        
        # Performance estimates
        print(f"\nPerformance Estimates:")
        print(f"  At 1 GMAC/s:  {breakdown['Total'] / 1e9:.3f} seconds")
        print(f"  At 10 GMAC/s: {breakdown['Total'] / 1e10:.4f} seconds")
        print(f"  At 100 GMAC/s: {breakdown['Total'] / 1e11:.5f} seconds")
        
        # Real-time factor (for 2-second audio)
        rtf_1gmac = (breakdown['Total'] / 1e9) / self.config['duration_seconds']
        rtf_10gmac = (breakdown['Total'] / 1e10) / self.config['duration_seconds']
        rtf_100gmac = (breakdown['Total'] / 1e11) / self.config['duration_seconds']
        
        print(f"\nReal-time Factor (RTF):")
        print(f"  At 1 GMAC/s:   {rtf_1gmac:.3f}")
        print(f"  At 10 GMAC/s:  {rtf_10gmac:.4f}")
        print(f"  At 100 GMAC/s: {rtf_100gmac:.5f}")
        
        print(f"\nLayer-wise Breakdown:")
        for layer_name, macs in breakdown['Layer_Details'].items():
            percentage = (macs / breakdown['Total']) * 100
            print(f"  {layer_name:20s}: {macs:12,d} MACs ({percentage:5.1f}%)")


# Example usage and testing
def main():
    """Main function to demonstrate MAC calculation"""
    
    # Configuration for BCCTN model with FFT=256
    config = {
        'fft_len': 256,
        'win_len': 400,
        'win_inc': 100,
        'kernel_num': [2, 16, 32, 64, 128, 256, 256],
        'kernel_size': 5,
        'embed_dim': 512,
        'num_heads': 32,
        'rnn_units': 128,
        'binaural': True,
        'batch_size': 1,
        'duration_seconds': 2.0,
        'sample_rate': 16000
    }
    
    # Create calculator and compute MACs
    calculator = BCCTNMACCalculator(config)
    breakdown = calculator.calculate_total_macs()
    calculator.print_results(breakdown)
    
    # Additional analysis
    print(f"\nAdditional Analysis:")
    print(f"  Model type: {'Binaural' if config['binaural'] else 'Monaural'}")
    print(f"  Audio duration: {config['duration_seconds']} seconds")
    print(f"  FFT size: {config['fft_len']}")
    print(f"  Sample rate: {config['sample_rate']} Hz")
    
    return breakdown


if __name__ == "__main__":
    results = main()
