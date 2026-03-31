#!/usr/bin/env python3
"""
Configuration Test for Your BCCTN Model
Tests MAC calculations with your exact configuration and compares regular vs depthwise separable convolutions
"""

import numpy as np
from typing import Dict


def test_your_configuration():
    """Test MAC calculations with your exact configuration"""
    
    # Import the calculator (you may need to adjust the import path)
    try:
        from mac_counter import BCCTNMACCalculator
    except ImportError:
        print("Error: Could not import BCCTNMACCalculator.")
        print("Make sure the mac_counter.py file is in the same directory.")
        return None
    
    # Your exact configuration
    your_config = {
        'fft_len': 256,           # Updated from 512
        'win_len': 256,           # Updated from 400
        'win_inc': 80,            # Updated from 100  
        'kernel_num': [16, 32, 64, 128, 256, 256],  # Your kernel numbers
        'kernel_size': 5,
        'embed_dim': 512,
        'num_heads': 32,
        'rnn_units': 128,
        'binaural': True,
        'batch_size': 1,
        'duration_seconds': 2.0,
        'sample_rate': 16000,
        'use_depthwise': True     # Your new depthwise separable conv
    }
    
    print("Testing Your BCCTN Configuration")
    print("=" * 50)
    print(f"FFT Length: {your_config['fft_len']}")
    print(f"Window Length: {your_config['win_len']} samples")
    print(f"Hop Length: {your_config['win_inc']} samples") 
    print(f"Kernel Numbers: {your_config['kernel_num']}")
    print(f"Using Depthwise Separable Conv: {your_config['use_depthwise']}")
    print(f"Binaural: {your_config['binaural']}")
    
    # Test with depthwise separable convolutions
    print(f"\n1. Analysis with Depthwise Separable Convolutions")
    print("-" * 50)
    
    calculator_depthwise = BCCTNMACCalculator(your_config)
    results_depthwise = calculator_depthwise.calculate_total_macs()
    
    # Test with regular convolutions for comparison
    print(f"\n2. Analysis with Regular Convolutions (for comparison)")
    print("-" * 50)
    
    config_regular = your_config.copy()
    config_regular['use_depthwise'] = False
    
    calculator_regular = BCCTNMACCalculator(config_regular)
    results_regular = calculator_regular.calculate_total_macs()
    
    # Compare the two approaches
    print(f"\n3. Comparison: Depthwise vs Regular Convolutions")
    print("-" * 50)
    
    depthwise_total = results_depthwise['Total']
    regular_total = results_regular['Total']
    
    savings = regular_total - depthwise_total
    savings_percent = (savings / regular_total) * 100
    
    print(f"Regular Conv Total:    {regular_total:15,d} MACs ({regular_total/1e9:.3f} GMACs)")
    print(f"Depthwise Sep Total:   {depthwise_total:15,d} MACs ({depthwise_total/1e9:.3f} GMACs)")
    print(f"Savings:               {savings:15,d} MACs ({savings_percent:.1f}% reduction)")
    
    # Component-wise comparison
    print(f"\nComponent-wise Comparison:")
    components = ['STFT', 'Encoder', 'Attention', 'Decoder', 'ISTFT']
    
    for comp in components:
        reg_macs = results_regular.get(comp, 0)
        dep_macs = results_depthwise.get(comp, 0)
        
        if reg_macs > 0:
            comp_savings = (reg_macs - dep_macs) / reg_macs * 100
            print(f"  {comp:12s}: Regular={reg_macs:10,d}, Depthwise={dep_macs:10,d}, Savings={comp_savings:5.1f}%")
        else:
            print(f"  {comp:12s}: Regular={reg_macs:10,d}, Depthwise={dep_macs:10,d}")
    
    # Real-time analysis
    print(f"\n4. Real-time Performance Analysis")
    print("-" * 50)
    
    audio_duration = your_config['duration_seconds']
    
    # Hardware scenarios
    hardware_scenarios = [
        ("Mobile CPU (ARM)", 1e9),
        ("Desktop CPU", 10e9),
        ("Mobile GPU", 50e9), 
        ("Desktop GPU", 100e9),
        ("High-end GPU", 500e9)
    ]
    
    print(f"Audio duration: {audio_duration} seconds")
    print(f"\nReal-time Factor Analysis:")
    print("(RTF < 1.0 means real-time capable)")
    print(f"{'Hardware':<15} {'Throughput':<12} {'Regular RTF':<12} {'Depthwise RTF':<14} {'Status'}")
    print("-" * 80)
    
    for hardware, throughput in hardware_scenarios:
        reg_time = regular_total / throughput
        dep_time = depthwise_total / throughput
        reg_rtf = reg_time / audio_duration
        dep_rtf = dep_time / audio_duration
        
        reg_status = "✓" if reg_rtf < 1.0 else "✗"
        dep_status = "✓" if dep_rtf < 1.0 else "✗"
        
        print(f"{hardware:<15} {throughput/1e9:>8.0f} GMAC/s {reg_rtf:>10.4f} {reg_status}  {dep_rtf:>12.4f} {dep_status}")
    
    # Memory analysis
    print(f"\n5. Memory Usage Estimation")
    print("-" * 50)
    
    # Rough parameter estimation for your model
    def estimate_parameters(config):
        kernel_num = [2] + config['kernel_num']  # Add input channels
        total_params = 0
        
        # Encoder parameters
        for i in range(len(kernel_num) - 1):
            in_ch = kernel_num[i] // 2
            out_ch = kernel_num[i + 1] // 2
            
            if config.get('use_depthwise', False):
                # Depthwise: in_ch * kernel_size * 2 (for complex)
                # Pointwise: in_ch * out_ch (for complex, multiply by 4)
                depthwise_params = in_ch * config['kernel_size'] * 2 * 4  # Complex
                pointwise_params = in_ch * out_ch * 4  # Complex 1x1 conv
                layer_params = depthwise_params + pointwise_params
            else:
                # Regular conv: in_ch * out_ch * kernel_size * 2 * 4 (complex)
                layer_params = in_ch * out_ch * config['kernel_size'] * 2 * 4
            
            total_params += layer_params
        
        # Decoder parameters (similar to encoder)
        total_params *= 2  # Encoder + Decoder
        
        # Attention parameters
        embed_dim = config['embed_dim']
        input_size = (kernel_num[-1] // 2) * (config['fft_len'] // 2**(len(kernel_num))) 
        if config['binaural']:
            input_size *= 2
            
        attention_params = (
            input_size * embed_dim * 4 +  # Input projection (complex)
            embed_dim * embed_dim * 3 * 4 +  # Q, K, V projections (complex)
            embed_dim * input_size * 4     # Output projection (complex)
        )
        
        total_params += attention_params
        
        return total_params
    
    reg_params = estimate_parameters(config_regular)
    dep_params = estimate_parameters(your_config)
    
    # Memory in MB (float32 = 4 bytes, complex = 8 bytes)
    reg_memory = reg_params * 8 / (1024 * 1024)  # Complex parameters
    dep_memory = dep_params * 8 / (1024 * 1024)
    
    print(f"Regular Conv Parameters:    {reg_params:10,d} ({reg_memory:.1f} MB)")
    print(f"Depthwise Sep Parameters:   {dep_params:10,d} ({dep_memory:.1f} MB)")
    print(f"Parameter Reduction:        {(reg_params-dep_params)/reg_params*100:.1f}%")
    
    # Summary and recommendations
    print(f"\n6. Summary and Recommendations")
    print("-" * 50)
    
    print(f"✓ Depthwise separable convolutions reduce MACs by {savings_percent:.1f}%")
    print(f"✓ Your configuration ({depthwise_total/1e9:.2f} GMACs) is more efficient than regular conv ({regular_total/1e9:.2f} GMACs)")
    
    # Find minimum hardware requirement
    min_throughput_dep = depthwise_total / audio_duration / 1e9  # In GMAC/s
    min_throughput_reg = regular_total / audio_duration / 1e9
    
    print(f"✓ Minimum throughput for real-time:")
    print(f"  - Regular conv: {min_throughput_reg:.1f} GMAC/s")
    print(f"  - Depthwise sep: {min_throughput_dep:.1f} GMAC/s")
    
    if min_throughput_dep < 10:
        print(f"✓ Feasible on desktop CPUs with depthwise separable conv")
    elif min_throughput_dep < 50:
        print(f"✓ Requires mobile GPU or better with depthwise separable conv")
    else:
        print(f"⚠ Requires desktop GPU even with depthwise separable conv")
    
    # Additional optimizations
    print(f"\n7. Additional Optimization Suggestions")
    print("-" * 50)
    
    attention_percent = (results_depthwise['Attention'] / depthwise_total) * 100
    
    if attention_percent > 30:
        print(f"• Attention is still dominant ({attention_percent:.1f}%). Consider:")
        print(f"  - Reducing num_heads from 32 to 16")
        print(f"  - Reducing embed_dim from 512 to 256")
    
    print(f"• Current savings from depthwise separable conv: {savings_percent:.1f}%")
    print(f"• Model is {'already quite efficient' if depthwise_total < 500e6 else 'moderately complex'}")
    
    return {
        'depthwise': results_depthwise,
        'regular': results_regular,
        'savings_percent': savings_percent,
        'config': your_config
    }


def analyze_fft_impact():
    """Analyze the impact of different FFT sizes with your configuration"""
    
    try:
        from mac_counter import BCCTNMACCalculator
    except ImportError:
        print("Error: Could not import BCCTNMACCalculator.")
        return None
    
    print("\n" + "="*60)
    print("FFT Size Impact Analysis")
    print("="*60)
    
    base_config = {
        'win_len': 256,
        'win_inc': 80,
        'kernel_num': [16, 32, 64, 128, 256, 256],
        'kernel_size': 5,
        'embed_dim': 512,
        'num_heads': 32,
        'rnn_units': 128,
        'binaural': True,
        'batch_size': 1,
        'duration_seconds': 2.0,
        'sample_rate': 16000,
        'use_depthwise': True
    }
    
    fft_sizes = [128, 256, 512, 1024]
    results = {}
    
    print(f"Testing FFT sizes: {fft_sizes}")
    print(f"All with depthwise separable convolutions\n")
    
    for fft_size in fft_sizes:
        config = base_config.copy()
        config['fft_len'] = fft_size
        
        calculator = BCCTNMACCalculator(config)
        breakdown = calculator.calculate_total_macs()
        
        results[fft_size] = breakdown['Total']
        rtf = (breakdown['Total'] / 100e9) / config['duration_seconds']  # @ 100 GMAC/s
        
        print(f"FFT={fft_size:4d}: {breakdown['Total']:12,d} MACs ({breakdown['Total']/1e9:.3f} GMACs, RTF={rtf:.4f})")
    
    # Analyze scaling
    print(f"\nFFT Scaling Analysis (relative to FFT=256):")
    base_macs = results[256]
    for fft_size, macs in results.items():
        if fft_size != 256:
            scaling_factor = macs / base_macs
            print(f"FFT={fft_size} vs FFT=256: {scaling_factor:.2f}x MACs")
    
    return results


def test_configuration_variants():
    """Test different configuration variants"""
    
    try:
        from mac_counter import BCCTNMACCalculator
    except ImportError:
        print("Error: Could not import BCCTNMACCalculator.")
        return None
    
    print("\n" + "="*60)
    print("Configuration Variants Analysis")
    print("="*60)
    
    # Base configuration (your current one)
    base_config = {
        'fft_len': 256,
        'win_len': 256,
        'win_inc': 80,
        'kernel_num': [16, 32, 64, 128, 256, 256],
        'kernel_size': 5,
        'embed_dim': 512,
        'num_heads': 32,
        'rnn_units': 128,
        'binaural': True,
        'batch_size': 1,
        'duration_seconds': 2.0,
        'sample_rate': 16000,
        'use_depthwise': True
    }
    
    variants = [
        ("Your Current Config", base_config),
        ("Lightweight (fewer channels)", {**base_config, 'kernel_num': [8, 16, 32, 64, 128, 128]}),
        ("Reduced Attention", {**base_config, 'embed_dim': 256, 'num_heads': 16}),
        ("Monaural", {**base_config, 'binaural': False}),
        ("High Quality", {**base_config, 'kernel_num': [32, 64, 128, 256, 512, 512]}),
        ("Minimal Attention", {**base_config, 'embed_dim': 128, 'num_heads': 8}),
    ]
    
    results = {}
    
    print(f"{'Configuration':<20} {'Total MACs':<15} {'GMACs':<8} {'RTF@100G':<10} {'Real-time?'}")
    print("-" * 70)
    
    for name, config in variants:
        calculator = BCCTNMACCalculator(config)
        breakdown = calculator.calculate_total_macs()
        
        total_macs = breakdown['Total']
        gmacs = total_macs / 1e9
        rtf = (total_macs / 100e9) / config['duration_seconds']
        real_time = "✓" if rtf < 1.0 else "✗"
        
        results[name] = breakdown
        
        print(f"{name:<20} {total_macs:>12,d} {gmacs:>6.2f} {rtf:>8.4f} {real_time:>10}")
    
    # Find the most efficient configuration
    min_macs = min(results.values(), key=lambda x: x['Total'])['Total']
    best_config = next(name for name, result in results.items() if result['Total'] == min_macs)
    
    print(f"\nMost efficient configuration: {best_config}")
    print(f"Efficiency gain over your current config: {(results['Your Current Config']['Total'] / min_macs - 1) * 100:.1f}%")
    
    return results


def create_optimization_report(test_results):
    """Create a detailed optimization report"""
    
    print("\n" + "="*60)
    print("OPTIMIZATION REPORT FOR YOUR BCCTN CONFIGURATION")
    print("="*60)
    
    if not test_results:
        print("No test results available for report generation.")
        return
    
    depthwise_results = test_results['depthwise']
    regular_results = test_results['regular']
    savings_percent = test_results['savings_percent']
    config = test_results['config']
    
    print(f"\n1. CURRENT CONFIGURATION SUMMARY")
    print("-" * 40)
    print(f"FFT Length: {config['fft_len']}")
    print(f"Window Length: {config['win_len']} samples ({config['win_len']/16:.1f}ms @ 16kHz)")
    print(f"Hop Length: {config['win_inc']} samples ({config['win_inc']/16:.1f}ms @ 16kHz)")
    print(f"Kernel Numbers: {config['kernel_num']}")
    print(f"Attention: {config['embed_dim']} embed_dim, {config['num_heads']} heads")
    print(f"Convolution Type: {'Depthwise Separable' if config['use_depthwise'] else 'Regular'}")
    print(f"Binaural Processing: {config['binaural']}")
    
    print(f"\n2. COMPUTATIONAL COMPLEXITY")
    print("-" * 40)
    total_macs = depthwise_results['Total']
    print(f"Total MACs: {total_macs:,d}")
    print(f"Total GMACs: {total_macs/1e9:.3f}")
    print(f"Depthwise savings: {savings_percent:.1f}% vs regular convolutions")
    
    # Component breakdown
    components = ['STFT', 'Encoder', 'Attention', 'Decoder', 'ISTFT']
    print(f"\nComponent Breakdown:")
    for comp in components:
        macs = depthwise_results.get(comp, 0)
        percentage = (macs / total_macs) * 100
        print(f"  {comp:12s}: {macs:10,d} MACs ({percentage:5.1f}%)")
    
    print(f"\n3. HARDWARE REQUIREMENTS")
    print("-" * 40)
    
    audio_duration = config['duration_seconds']
    min_throughput = total_macs / audio_duration / 1e9  # GMAC/s for real-time
    
    print(f"Minimum throughput for real-time: {min_throughput:.1f} GMAC/s")
    
    if min_throughput < 5:
        hw_category = "Mobile CPU capable"
    elif min_throughput < 15:
        hw_category = "Desktop CPU capable"
    elif min_throughput < 50:
        hw_category = "Mobile GPU required"
    elif min_throughput < 150:
        hw_category = "Desktop GPU required"
    else:
        hw_category = "High-end GPU required"
    
    print(f"Hardware category: {hw_category}")
    
    print(f"\n4. OPTIMIZATION RECOMMENDATIONS")
    print("-" * 40)
    
    # Analyze bottlenecks
    attention_macs = depthwise_results.get('Attention', 0)
    encoder_macs = depthwise_results.get('Encoder', 0)
    decoder_macs = depthwise_results.get('Decoder', 0)
    
    attention_percent = (attention_macs / total_macs) * 100
    encoder_percent = (encoder_macs / total_macs) * 100
    decoder_percent = (decoder_macs / total_macs) * 100
    
    print(f"Current bottlenecks:")
    if attention_percent > 35:
        print(f"  • Attention dominates ({attention_percent:.1f}%)")
        print(f"    - Consider reducing num_heads from {config['num_heads']} to {config['num_heads']//2}")
        print(f"    - Consider reducing embed_dim from {config['embed_dim']} to {config['embed_dim']//2}")
    
    if encoder_percent > 30:
        print(f"  • Encoder is expensive ({encoder_percent:.1f}%)")
        print(f"    - Already using depthwise separable conv ✓")
        print(f"    - Consider reducing channel numbers")
    
    if decoder_percent > 30:
        print(f"  • Decoder is expensive ({decoder_percent:.1f}%)")
        print(f"    - Already using depthwise separable conv ✓")
        print(f"    - Consider weight sharing between layers")
    
    print(f"\nNext optimization steps:")
    print(f"  1. ✓ Already using depthwise separable convolutions ({savings_percent:.1f}% savings)")
    
    if attention_percent > 25:
        estimated_attention_savings = attention_percent * 0.75  # Assume 75% reduction possible
        print(f"  2. Optimize attention: potential {estimated_attention_savings:.1f}% total savings")
    
    if config['binaural']:
        print(f"  3. Consider monaural version for ~40% MAC reduction (if acceptable)")
    
    print(f"  4. Consider quantization (INT8) for ~2x speedup with minimal accuracy loss")
    print(f"  5. Consider pruning for additional 20-30% reduction")
    
    print(f"\n5. DEPLOYMENT RECOMMENDATIONS")
    print("-" * 40)
    
    if min_throughput < 10:
        print(f"✓ Suitable for real-time deployment on modern mobile devices")
        print(f"✓ Can run on CPU with optimization")
    elif min_throughput < 50:
        print(f"✓ Suitable for real-time deployment on mobile GPU")
        print(f"✓ Consider TensorRT or similar optimization frameworks")
    else:
        print(f"⚠ Requires powerful GPU for real-time deployment")
        print(f"⚠ Consider further model optimization")
    
    print(f"\nFramework recommendations:")
    print(f"  • Use mixed precision (FP16) for ~2x speedup")
    print(f"  • Use TensorRT/ONNX Runtime for inference optimization")
    print(f"  • Consider batch processing for throughput applications")
    
    return True


def main():
    """Main function to run all tests"""
    
    print("BCCTN Configuration Analysis Suite")
    print("Testing your exact configuration with depthwise separable convolutions")
    print("="*80)
    
    # Run main configuration test
    test_results = test_your_configuration()
    
    if test_results:
        # Run additional analyses
        fft_results = analyze_fft_impact()
        variant_results = test_configuration_variants()
        
        # Generate optimization report
        create_optimization_report(test_results)
        
        print(f"\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        print(f"Your depthwise separable BCCTN configuration:")
        print(f"  • Total MACs: {test_results['depthwise']['Total']:,d}")
        print(f"  • Efficiency gain: {test_results['savings_percent']:.1f}% vs regular conv")
        print(f"  • Real-time capable: Check hardware requirements above")
        
    else:
        print("Test failed. Please check that mac_counter.py is available.")


if __name__ == "__main__":
    main()
        