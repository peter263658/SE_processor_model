#!/usr/bin/env python3
"""
Quick Test Script for Your BCCTN Configuration
Simple script to quickly test MAC calculations with your exact settings
"""

def quick_mac_test():
    """Quick MAC test with your configuration"""
    
    print("Quick MAC Test for Your BCCTN Configuration")
    print("=" * 50)
    
    # Your exact configuration from the config file
    your_config = {
        # FFT settings
        'fft_len': 256,           # Changed from 512 to 256
        'win_len': 256,           # Changed from 400 to 256  
        'win_inc': 80,            # Changed from 100 to 80
        'win_type': 'hann',
        
        # Model architecture
        'kernel_num': [16, 32, 64, 128, 256, 256],  # Your kernel numbers
        'kernel_size': 5,
        'embed_dim': 512,
        'num_heads': 32,
        'rnn_units': 128,
        'rnn_layers': 2,
        'bidirectional': False,
        
        # Processing settings
        'binaural': True,
        'use_depthwise': True,    # New: depthwise separable convolutions
        'masking_mode': 'E',
        
        # Test settings
        'batch_size': 1,
        'duration_seconds': 2.0,
        'sample_rate': 16000
    }
    
    print("Configuration:")
    print(f"  FFT Length: {your_config['fft_len']}")
    print(f"  Window: {your_config['win_len']} samples, Hop: {your_config['win_inc']} samples")
    print(f"  Kernel Numbers: {your_config['kernel_num']}")
    print(f"  Attention: {your_config['embed_dim']} dim, {your_config['num_heads']} heads")
    print(f"  Depthwise Separable: {your_config['use_depthwise']}")
    print(f"  Binaural: {your_config['binaural']}")
    
    # Try to import and run the calculator
    try:
        # You might need to adjust this import based on your file structure
        import sys
        import os
        
        # Add current directory to path if needed
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.append(current_dir)
        
        from mac_counter import BCCTNMACCalculator
        
        print(f"\n" + "="*50)
        print("MAC Calculation Results")
        print("="*50)
        
        # Calculate MACs
        calculator = BCCTNMACCalculator(your_config)
        results = calculator.calculate_total_macs()
        
        # Print results
        total_macs = results['Total']
        total_gmacs = total_macs / 1e9
        
        print(f"\nMain Results:")
        print(f"  Total MACs: {total_macs:,d}")
        print(f"  Total GMACs: {total_gmacs:.3f}")
        
        # Component breakdown
        print(f"\nComponent Breakdown:")
        components = ['STFT', 'Encoder', 'Attention', 'Decoder', 'ISTFT']
        for comp in components:
            macs = results.get(comp, 0)
            percentage = (macs / total_macs) * 100 if total_macs > 0 else 0
            print(f"  {comp:12s}: {macs:12,d} MACs ({percentage:5.1f}%)")
        
        # Real-time analysis
        print(f"\nReal-time Analysis:")
        audio_duration = your_config['duration_seconds']
        
        # Common hardware throughputs (MACs per second)
        hardware_specs = [
            ("Mobile CPU", 1e9),
            ("Desktop CPU", 10e9), 
            ("Mobile GPU", 50e9),
            ("Desktop GPU", 100e9),
        ]
        
        print(f"  Audio duration: {audio_duration} seconds")
        print(f"  Hardware requirements for real-time processing:")
        
        for hw_name, throughput in hardware_specs:
            processing_time = total_macs / throughput
            rtf = processing_time / audio_duration
            status = "✓ Real-time" if rtf < 1.0 else "✗ Too slow"
            print(f"    {hw_name:12s} ({throughput/1e9:3.0f} GMAC/s): RTF = {rtf:.4f} {status}")
        
        # Minimum throughput needed
        min_throughput_gmacs = total_macs / audio_duration / 1e9
        print(f"  Minimum required: {min_throughput_gmacs:.1f} GMAC/s for real-time")
        
        # Memory estimation
        print(f"\nMemory Estimation:")
        
        # Rough parameter count estimation
        def estimate_model_size(config):
            kernel_num = [2] + config['kernel_num']  # Add input channels
            total_params = 0
            
            # Encoder/Decoder parameters (complex, so 4x real parameters)
            for i in range(len(kernel_num) - 1):
                in_ch = kernel_num[i] // 2
                out_ch = kernel_num[i + 1] // 2
                
                if config.get('use_depthwise', False):
                    # Depthwise + Pointwise
                    depthwise_params = in_ch * config['kernel_size'] * 2 * 4
                    pointwise_params = in_ch * out_ch * 4
                    layer_params = depthwise_params + pointwise_params
                else:
                    # Regular conv
                    layer_params = in_ch * out_ch * config['kernel_size'] * 2 * 4
                
                total_params += layer_params
            
            # Double for encoder + decoder
            total_params *= 2
            
            # Attention parameters
            embed_dim = config['embed_dim']
            # Rough estimate for attention parameters
            attention_params = embed_dim * embed_dim * 8  # Q, K, V, O projections
            total_params += attention_params
            
            # Binaural factor
            if config['binaural']:
                total_params *= 1.8  # Not exactly 2x due to shared attention
            
            return total_params
        
        estimated_params = estimate_model_size(your_config)
        model_memory_mb = estimated_params * 4 / (1024 * 1024)  # 4 bytes per float32
        
        print(f"  Estimated parameters: {estimated_params:,d}")
        print(f"  Model memory: {model_memory_mb:.1f} MB")
        print(f"  Total memory (with activations): ~{model_memory_mb * 3:.1f} MB")
        
        # Summary
        print(f"\n" + "="*50)
        print("SUMMARY")
        print("="*50)
        
        if min_throughput_gmacs < 10:
            feasibility = "✓ Feasible on desktop CPU"
        elif min_throughput_gmacs < 50:
            feasibility = "✓ Feasible on mobile GPU"
        elif min_throughput_gmacs < 150:
            feasibility = "✓ Feasible on desktop GPU"
        else:
            feasibility = "⚠ Requires high-end GPU"
        
        print(f"Computational Complexity: {total_gmacs:.2f} GMACs")
        print(f"Real-time Feasibility: {feasibility}")
        print(f"Memory Footprint: {model_memory_mb:.1f} MB")
        
        # Optimization suggestions
        attention_percent = (results.get('Attention', 0) / total_macs) * 100
        
        print(f"\nOptimization Suggestions:")
        print(f"✓ Already using depthwise separable convolutions")
        
        if attention_percent > 30:
            print(f"• Attention dominates ({attention_percent:.1f}%) - consider reducing heads/embed_dim")
        if total_gmacs > 1.0:
            print(f"• Consider quantization (INT8) for ~2x speedup")
        if your_config['binaural']:
            print(f"• Monaural version would reduce complexity by ~40%")
        
        return results
        
    except ImportError as e:
        print(f"\nError: Could not import MAC calculator: {e}")
        print(f"Make sure you have the mac_counter.py file in the same directory.")
        print(f"You can copy the MAC calculator code into a file named 'mac_counter.py'")
        return None
        
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        return None


if __name__ == "__main__":
    quick_mac_test()
