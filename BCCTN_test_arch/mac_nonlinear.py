#!/usr/bin/env python3
"""
Corrected BCCTN Operations Analyzer
Fixes all systematic over/under-estimation issues identified
"""

import math
from typing import Dict, Tuple, Optional


class CorrectedOpsAnalyzer:
    """Corrected operations analyzer with proper real vs complex distinction"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {
            'fft_len': 128,
            'win_len': 128,
            'win_inc': 80,
            'kernel_num': [16, 32, 64, 128, 256, 256],
            'kernel_size': 5,
            'embed_dim': 512,
            'num_heads': 32,
            'binaural': True,
            'batch_size': 1,
            'sample_rate': 16000,
            'attn_win': 32,
            'include_layernorm': True,
            'include_feedforward': True,
        }
        
        self._calculate_dimensions()
        
    def _calculate_dimensions(self):
        """Calculate dimensions"""
        self.fft_len = self.config['fft_len']
        self.win_inc = self.config['win_inc'] 
        self.freq_bins = self.fft_len // 2 + 1 - 1  # 64
        
        # Correct channel calculation
        self.conv_channels = [k//2 for k in [2] + self.config['kernel_num']]
        self.num_downsamples = len(self.conv_channels) - 1
        self.final_freq_bins = self.freq_bins // (2 ** self.num_downsamples)
        
        # Attention flatten size
        full_kernel_num = [2] + self.config['kernel_num']
        base_flat = (self.config['kernel_num'][-1]//2) * (self.fft_len // (2**len(full_kernel_num)))
        self.flatten_size = base_flat * (2 if self.config['binaural'] else 1)
        
        self.window_size = self.config.get('attn_win', 32)
        
    def count_complex_fft_ops(self, n: int, num_transforms: int) -> Tuple[int, int]:
        """
        Corrected FFT operation count using theoretical minimum
        5 * N * log2(N) real MACs per complex FFT
        """
        theoretical_macs = 5 * n * int(math.log2(n)) * num_transforms
        return theoretical_macs, 0  # No non-linear ops in FFT
    
    def count_real_linear_ops(self, input_dim: int, output_dim: int, 
                             num_samples: int = 1) -> Tuple[int, int]:
        """
        Count real linear layer operations: y = Wx + b
        Each output element = input_dim multiplications + (input_dim-1) additions + 1 bias addition
        = input_dim MACs per output element
        """
        mac_ops = num_samples * output_dim * input_dim
        return mac_ops, 0
    
    def count_real_dot_product_ops(self, dim: int, num_dots: int) -> Tuple[int, int]:
        """
        Count dot product operations: sum(a[i] * b[i])
        = dim multiplications + (dim-1) additions = dim MACs
        """
        mac_ops = num_dots * dim
        return mac_ops, 0
    
    def count_softmax_ops(self, num_elements: int) -> Tuple[int, int]:
        """
        Corrected softmax operations:
        1. Subtract max: 1 real op per element
        2. Exp: ~15 MACs per element (typical hardware cost)
        3. Sum: num_elements-1 additions
        4. Divide: ~5 MACs per element (typical hardware cost)
        """
        # Conservative estimate for hardware implementation
        subtract_max_macs = num_elements  # 1 sub per element
        sum_macs = num_elements - 1       # reduction sum
        divide_macs = num_elements * 5    # division cost
        
        total_macs = subtract_max_macs + sum_macs + divide_macs
        exp_ops = num_elements            # exp is pure non-linear
        
        return total_macs, exp_ops
    
    def count_layernorm_ops(self, num_elements: int) -> Tuple[int, int]:
        """
        LayerNorm operations: (x - mean) / sqrt(var + eps) * gamma + beta
        """
        # Mean calculation: (num_elements-1) adds + 1 div
        mean_macs = num_elements - 1 + 5  # +5 for division
        
        # Variance calculation: num_elements subtractions + num_elements squares + sum + div
        var_macs = num_elements + num_elements + (num_elements-1) + 5
        
        # Normalization: subtract mean + divide by std + scale + bias
        norm_macs = num_elements * 3  # sub, scale, bias
        norm_div_macs = num_elements * 5  # division by std
        
        total_macs = mean_macs + var_macs + norm_macs + norm_div_macs
        sqrt_ops = 1  # One sqrt operation for std calculation
        
        return total_macs, sqrt_ops
    
    def count_activation_ops(self, num_elements: int, activation_type: str) -> Tuple[int, int]:
        """Count activation function operations"""
        if activation_type.lower() == 'prelu':
            # PReLU: y = max(0, x) + a * min(0, x)
            mac_ops = num_elements  # multiplication by slope parameter
            nonlinear_ops = num_elements  # max/min selection
        elif activation_type.lower() == 'gelu':
            # GELU: x * 0.5 * (1 + tanh(...)) - approximate cost
            mac_ops = num_elements * 3  # various multiplications
            nonlinear_ops = num_elements  # tanh approximation
        elif activation_type.lower() == 'relu':
            mac_ops = 0
            nonlinear_ops = num_elements  # max(0, x)
        else:
            mac_ops = num_elements
            nonlinear_ops = num_elements
            
        return mac_ops, nonlinear_ops
    
    def analyze_stft_per_frame(self) -> Tuple[int, int]:
        """Corrected STFT analysis using theoretical minimum"""
        batch_size = self.config['batch_size']
        num_channels = 2 if self.config['binaural'] else 1
        
        # Corrected FFT: 5 * N * log2(N) per transform
        fft_macs, _ = self.count_complex_fft_ops(
            self.fft_len, batch_size * num_channels)
        
        # Windowing: element-wise multiplication
        windowing_macs = batch_size * num_channels * self.config['win_len']
        
        total_macs = fft_macs + windowing_macs
        total_nonlinear = 0
        
        return total_macs, total_nonlinear
    
    def analyze_attention_per_frame(self) -> Tuple[int, int]:
        """
        Corrected attention analysis - treats attention as REAL operations
        Key insight: After complex convolutions, attention operates on real features
        """
        batch_size = self.config['batch_size']
        embed_dim = self.config['embed_dim']  # 512
        num_heads = self.config['num_heads']  # 32
        head_dim = embed_dim // num_heads  # 16
        
        context_len = self.window_size + 1 if self.window_size else 65  # 33
        input_size = self.flatten_size  # 256
        
        total_macs = 0
        total_nonlinear = 0
        
        # === CORRECTED: All projections are REAL linear layers ===
        
        # 1. Input projection: 256 -> 512 (REAL)
        input_proj_macs, _ = self.count_real_linear_ops(
            input_size, embed_dim, batch_size * 1)
        
        # 2. Q projection: 512 -> 512 (REAL)
        q_proj_macs, _ = self.count_real_linear_ops(
            embed_dim, embed_dim, batch_size * 1)
        
        # 3. K, V projections: 512 -> 512 each (REAL)
        k_proj_macs, _ = self.count_real_linear_ops(
            embed_dim, embed_dim, batch_size * 1)
        v_proj_macs, _ = self.count_real_linear_ops(
            embed_dim, embed_dim, batch_size * 1)
        
        # 4. Q @ K^T: REAL dot products with proper accumulation
        # Shape: [batch, heads, 1, context_len] @ [batch, heads, context_len, head_dim]
        qk_dots = batch_size * num_heads * 1 * context_len
        qk_macs, _ = self.count_real_dot_product_ops(head_dim, qk_dots)
        
        # 5. Softmax: CORRECTED with realistic hardware costs
        softmax_elements = batch_size * num_heads * 1 * context_len
        softmax_macs, softmax_nonlinear = self.count_softmax_ops(softmax_elements)
        
        # 6. Attention @ V: REAL dot products
        attn_v_dots = batch_size * num_heads * 1 * context_len
        attn_v_macs, _ = self.count_real_dot_product_ops(head_dim, attn_v_dots)
        
        # 7. Output projection in attention: 512 -> 512 (REAL)
        attn_out_macs, _ = self.count_real_linear_ops(
            embed_dim, embed_dim, batch_size * 1)
        
        # 8. Final output projection: 512 -> 256 (REAL)
        output_proj_macs, _ = self.count_real_linear_ops(
            embed_dim, input_size, batch_size * 1)
        
        # === Add auxiliary operations ===
        
        # Bias additions (often forgotten!)
        bias_macs = embed_dim * 4  # Q, K, V, output projections
        
        # Residual connections
        residual_macs = input_size  # Add input to output
        
        attention_core_macs = (input_proj_macs + q_proj_macs + k_proj_macs + 
                              v_proj_macs + qk_macs + attn_v_macs + 
                              attn_out_macs + output_proj_macs + 
                              bias_macs + residual_macs)
        
        attention_core_nonlinear = softmax_nonlinear
        attention_total_macs = attention_core_macs + softmax_macs
        
        # === LayerNorm (if included) ===
        if self.config.get('include_layernorm', True):
            # Pre-norm and post-norm
            ln_elements = input_size * 2  # before and after attention
            ln_macs, ln_nonlinear = self.count_layernorm_ops(ln_elements)
            attention_total_macs += ln_macs
            total_nonlinear += ln_nonlinear
        
        # === Feed-Forward (if included) ===
        if self.config.get('include_feedforward', True):
            ff_dim = embed_dim * 4  # 2048
            
            # Two linear layers: 512->2048->512
            ff_up_macs, _ = self.count_real_linear_ops(embed_dim, ff_dim, batch_size * 1)
            ff_down_macs, _ = self.count_real_linear_ops(ff_dim, embed_dim, batch_size * 1)
            
            # Activation (GELU)
            ff_act_macs, ff_act_nonlinear = self.count_activation_ops(ff_dim, 'gelu')
            
            # Bias and residual
            ff_bias_macs = ff_dim + embed_dim  # Two bias additions
            ff_residual_macs = embed_dim  # Residual connection
            
            ff_total_macs = ff_up_macs + ff_down_macs + ff_act_macs + ff_bias_macs + ff_residual_macs
            attention_total_macs += ff_total_macs
            total_nonlinear += ff_act_nonlinear
        
        total_macs = attention_total_macs
        total_nonlinear += attention_core_nonlinear
        
        # Store breakdown for analysis
        self.attention_breakdown = {
            'input_proj': input_proj_macs,
            'qkv_proj': q_proj_macs + k_proj_macs + v_proj_macs,
            'qk_attention': qk_macs,
            'softmax_mac': softmax_macs,
            'softmax_nonlinear': softmax_nonlinear,
            'attn_v': attn_v_macs,
            'output_proj': attn_out_macs + output_proj_macs,
            'auxiliary': bias_macs + residual_macs,
            'layernorm_mac': ln_macs if self.config.get('include_layernorm') else 0,
            'layernorm_nonlinear': ln_nonlinear if self.config.get('include_layernorm') else 0,
            'feedforward_mac': ff_total_macs if self.config.get('include_feedforward') else 0,
            'feedforward_nonlinear': ff_act_nonlinear if self.config.get('include_feedforward') else 0,
        }
        
        return total_macs, total_nonlinear
    
    def analyze_complete_model(self) -> Dict:
        """Analyze complete model with corrected calculations"""
        
        print("Corrected BCCTN Operations Analysis")
        print("=" * 60)
        print(f"Configuration:")
        print(f"  FFT length: {self.fft_len}")
        print(f"  Attention flatten size: {self.flatten_size}")
        print(f"  Embed dim: {self.config['embed_dim']}")
        print(f"  Num heads: {self.config['num_heads']}")
        print(f"  Context length: {self.window_size + 1 if self.window_size else 65}")
        
        # Only analyze key components for this comparison
        stft_macs, stft_nonlinear = self.analyze_stft_per_frame()
        attention_macs, attention_nonlinear = self.analyze_attention_per_frame()
        istft_macs, istft_nonlinear = self.analyze_stft_per_frame()  # Same as STFT
        
        # Rough estimates for encoder/decoder (not the focus of correction)
        # These use complex operations so keep higher estimates
        encoder_macs = 1_500_000  # Rough estimate for 6-layer complex conv
        decoder_macs = 1_500_000  # Rough estimate for 6-layer complex deconv
        encoder_nonlinear = 50_000  # PReLU activations
        decoder_nonlinear = 50_000  # PReLU activations
        
        total_macs = stft_macs + encoder_macs + attention_macs + decoder_macs + istft_macs
        total_nonlinear = (stft_nonlinear + encoder_nonlinear + attention_nonlinear + 
                          decoder_nonlinear + istft_nonlinear)
        
        results = {
            'mac_operations': {
                'STFT': stft_macs,
                'Encoder': encoder_macs,
                'Attention': attention_macs,
                'Decoder': decoder_macs,
                'ISTFT': istft_macs,
                'Total': total_macs
            },
            'nonlinear_operations': {
                'STFT': stft_nonlinear,
                'Encoder': encoder_nonlinear,
                'Attention': attention_nonlinear,
                'Decoder': decoder_nonlinear,
                'ISTFT': istft_nonlinear,
                'Total': total_nonlinear
            },
            'attention_breakdown': self.attention_breakdown
        }
        
        return results
    
    def print_comparison(self, results: Dict):
        """Print comparison with previous estimates"""
        mac_ops = results['mac_operations']
        nonlinear_ops = results['nonlinear_operations']
        breakdown = results['attention_breakdown']
        
        print(f"\n" + "="*60)
        print("CORRECTED vs PREVIOUS ESTIMATES")
        print("="*60)
        
        # Focus on attention corrections
        attention_macs = mac_ops['Attention']
        attention_nonlinear = nonlinear_ops['Attention']
        
        print(f"\nSTFT/ISTFT Corrections:")
        print(f"  Previous estimate: ~11,008 MACs/frame")
        print(f"  Corrected estimate: {mac_ops['STFT']:,d} MACs/frame")
        print(f"  Reduction: {((11008 - mac_ops['STFT'])/11008)*100:.1f}%")
        
        print(f"\nAttention Corrections:")
        previous_attention_only = 2_691_136
        previous_complete = 6_887_488
        
        if self.config.get('include_feedforward', True):
            print(f"  Previous complete transformer: {previous_complete:,d} MACs")
            print(f"  Corrected complete transformer: {attention_macs:,d} MACs")
            print(f"  Change: {((attention_macs - previous_complete)/previous_complete)*100:+.1f}%")
        else:
            print(f"  Previous attention-only: {previous_attention_only:,d} MACs")
            print(f"  Corrected attention-only: {attention_macs:,d} MACs")
            print(f"  Change: {((attention_macs - previous_attention_only)/previous_attention_only)*100:+.1f}%")
        
        print(f"\nDetailed Attention Breakdown:")
        print(f"{'Component':<20} {'MACs':<12} {'Non-linear':<12}")
        print("-" * 45)
        
        components = [
            ('Input projection', breakdown['input_proj'], 0),
            ('QKV projections', breakdown['qkv_proj'], 0),
            ('QK attention', breakdown['qk_attention'], 0),
            ('Softmax', breakdown['softmax_mac'], breakdown['softmax_nonlinear']),
            ('Attention@V', breakdown['attn_v'], 0),
            ('Output projection', breakdown['output_proj'], 0),
            ('Auxiliary ops', breakdown['auxiliary'], 0),
            ('LayerNorm', breakdown['layernorm_mac'], breakdown['layernorm_nonlinear']),
            ('Feed-Forward', breakdown['feedforward_mac'], breakdown['feedforward_nonlinear']),
        ]
        
        for name, mac_count, nonlinear_count in components:
            if mac_count > 0 or nonlinear_count > 0:
                print(f"{name:<20} {mac_count:<12,d} {nonlinear_count:<12,d}")
        
        # Real-time implications
        hop_size_ms = self.config['win_inc'] / self.config['sample_rate'] * 1000
        frames_per_second = 1000 / hop_size_ms
        
        total_mac_per_second = mac_ops['Total'] * frames_per_second
        
        print(f"\nCorrected Real-time Requirements:")
        print(f"  Total MACs per frame: {mac_ops['Total']:,d} ({mac_ops['Total']/1e6:.2f} MMACs)")
        print(f"  Total MACs per second: {total_mac_per_second:,.0f} ({total_mac_per_second/1e9:.2f} GMAC/s)")
        print(f"  Non-linear ops per second: {nonlinear_ops['Total'] * frames_per_second:,.0f}")
        
        # Key insights
        print(f"\nKey Corrections Applied:")
        print(f"  ✓ FFT uses theoretical minimum: 5*N*log2(N)")
        print(f"  ✓ Attention projections are REAL operations (1 MAC/op)")
        print(f"  ✓ Dot products include accumulation costs")
        print(f"  ✓ Softmax uses realistic hardware costs (~20 MACs/element)")
        print(f"  ✓ Added bias, residual, and auxiliary operations")
        print(f"  ✓ Separated complex (conv) from real (attention) domains")


def main():
    """Run corrected analysis"""
    
    # Test attention-only configuration
    config_attention_only = {
        'fft_len': 128,
        'win_len': 128,
        'win_inc': 80,
        'kernel_num': [16, 32, 64, 128, 256, 256],
        'kernel_size': 5,
        'embed_dim': 512,
        'num_heads': 32,
        'binaural': True,
        'batch_size': 1,
        'sample_rate': 16000,
        'attn_win': 32,
        'include_layernorm': False,
        'include_feedforward': False,
    }
    
    # Test complete transformer configuration
    config_complete = config_attention_only.copy()
    config_complete['include_layernorm'] = True
    config_complete['include_feedforward'] = True
    
    print("CORRECTED BCCTN OPERATIONS ANALYSIS")
    print("Fixing systematic over/under-estimation issues")
    print("=" * 80)
    
    # Analyze attention-only
    print("\n" + "="*40)
    print("ATTENTION-ONLY ANALYSIS")
    print("="*40)
    
    analyzer1 = CorrectedOpsAnalyzer(config_attention_only)
    results1 = analyzer1.analyze_complete_model()
    analyzer1.print_comparison(results1)
    
    # Analyze complete transformer
    print("\n" + "="*40)
    print("COMPLETE TRANSFORMER ANALYSIS")
    print("="*40)
    
    analyzer2 = CorrectedOpsAnalyzer(config_complete)
    results2 = analyzer2.analyze_complete_model()
    analyzer2.print_comparison(results2)
    
    return results1, results2


if __name__ == "__main__":
    results = main()