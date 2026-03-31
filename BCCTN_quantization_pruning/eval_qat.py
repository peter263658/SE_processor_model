#!/usr/bin/env python3
"""
Evaluation script for Pruning-aware QAT optimized models

This script evaluates and compares:
1. Original full-precision model
2. Pruned + Quantized model
3. Performance metrics and compression statistics

Usage:
    python eval_qat.py --original original_model.ckpt --optimized optimized_model.ckpt
    python eval_qat.py --optimized optimized_model.ckpt --benchmark
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
import time
import psutil
import os
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

from DCNN.trainer import DCNNLightningModule
from DCNN.datasets import create_torch_dataloader
from DCNN.quantization import PruningAwareQAT

# Import evaluation functions from eval.py
try:
    from eval import (safe_mbstoi, safe_pesq, fwsegsnr, 
                     compute_ild_error_with_mask, compute_ipd_error_with_mask)
except ImportError:
    # If eval.py is in a different location, adjust the import path
    import sys
    sys.path.append('..')  # Adjust path as needed
    from eval import (safe_mbstoi, safe_pesq, fwsegsnr,
                     compute_ild_error_with_mask, compute_ipd_error_with_mask)


class ModelBenchmark:
    """Comprehensive model benchmarking and comparison"""
    
    def __init__(self, device: str = "auto"):
        self.device = torch.device("cuda" if torch.cuda.is_available() and device != "cpu" else "cpu")
        print(f"Using device: {self.device}")
    
    def load_model(self, checkpoint_path: str, config_path: str = None) -> Tuple[nn.Module, Dict]:
        """Load model from checkpoint"""
        print(f"Loading model from: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Extract config from checkpoint or load separately
        if 'config' in checkpoint:
            config = checkpoint['config']
        elif config_path:
            from omegaconf import OmegaConf
            config = OmegaConf.load(config_path)
        else:
            raise ValueError("No config found in checkpoint and no config_path provided")
        
        # Initialize model
        model = DCNNLightningModule(config)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        model = model.to(self.device)
        
        # Extract compression stats if available
        compression_stats = checkpoint.get('compression_stats', {})
        
        return model, compression_stats
    
    def measure_inference_time(self, model: nn.Module, input_tensor: torch.Tensor, 
                             num_warmup: int = 10, num_runs: int = 100) -> Dict[str, float]:
        """Measure model inference time"""
        print("Measuring inference time...")
        
        model.eval()
        input_tensor = input_tensor.to(self.device)
        
        # Warmup runs
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = model(input_tensor)
        
        # Synchronize GPU
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        
        # Measure inference time
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.perf_counter()
                _ = model(input_tensor)
                
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                
                end_time = time.perf_counter()
                times.append(end_time - start_time)
        
        return {
            "mean_time": np.mean(times),
            "std_time": np.std(times),
            "min_time": np.min(times),
            "max_time": np.max(times),
            "median_time": np.median(times)
        }
    
    def measure_memory_usage(self, model: nn.Module, input_tensor: torch.Tensor) -> Dict[str, float]:
        """Measure memory usage during inference"""
        print("Measuring memory usage...")
        
        model.eval()
        input_tensor = input_tensor.to(self.device)
        
        # Clear cache
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        # Measure memory before
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        if self.device.type == "cuda":
            gpu_memory_before = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        
        # Run inference
        with torch.no_grad():
            output = model(input_tensor)
        
        # Measure memory after
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        
        result = {
            "cpu_memory_mb": memory_after - memory_before,
            "model_size_mb": sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
        }
        
        if self.device.type == "cuda":
            gpu_memory_after = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            gpu_memory_peak = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
            gpu_memory_reserved = torch.cuda.max_memory_reserved() / 1024 / 1024  # MB
            
            result.update({
                "gpu_memory_mb": gpu_memory_after - gpu_memory_before,
                "gpu_memory_peak_mb": gpu_memory_peak,
                "gpu_memory_reserved_mb": gpu_memory_reserved
            })
        
        return result
    
    def evaluate_audio_quality(self, model: nn.Module, test_dataloader, 
                              num_samples: int = 100) -> Dict[str, float]:
        """Evaluate audio quality metrics"""
        print(f"Evaluating audio quality on {num_samples} samples...")
        
        model.eval()
        metrics = {
            'mbstoi': [],
            'pesq': [],
            'segsnr': [],
            'ild_error': [],
            'ipd_error': []
        }
        
        processed_samples = 0
        
        with torch.no_grad():
            for batch_idx, (noisy, clean) in enumerate(test_dataloader):
                if processed_samples >= num_samples:
                    break
                
                noisy = noisy.to(self.device)
                enhanced = model(noisy)
                
                # Move to CPU for evaluation
                noisy_np = noisy.cpu().numpy()
                clean_np = clean.numpy()
                enhanced_np = enhanced.cpu().numpy()
                
                # Process each sample in batch
                for i in range(min(noisy.size(0), num_samples - processed_samples)):
                    try:
                        # Extract channels
                        clean_l = clean_np[i, 0]
                        clean_r = clean_np[i, 1]
                        enhanced_l = enhanced_np[i, 0]
                        enhanced_r = enhanced_np[i, 1]
                        
                        # Calculate metrics
                        mbstoi = safe_mbstoi(clean_l, clean_r, enhanced_l, enhanced_r)
                        if mbstoi is not None:
                            metrics['mbstoi'].append(mbstoi)
                        
                        pesq = safe_pesq(clean_l, clean_r, enhanced_l, enhanced_r)
                        if pesq is not None:
                            metrics['pesq'].append(pesq)
                        
                        # SegSNR improvement
                        segsnr_l = fwsegsnr(clean_l, noisy_np[i, 0], enhanced_l)
                        segsnr_r = fwsegsnr(clean_r, noisy_np[i, 1], enhanced_r)
                        metrics['segsnr'].append((segsnr_l + segsnr_r) / 2)
                        
                        # ILD error
                        ild_error = compute_ild_error_with_mask(clean_l, clean_r, enhanced_l, enhanced_r)
                        metrics['ild_error'].append(ild_error)
                        
                        # IPD error
                        ipd_error = compute_ipd_error_with_mask(clean_l, clean_r, enhanced_l, enhanced_r)
                        metrics['ipd_error'].append(ipd_error * 180 / np.pi)  # Convert to degrees
                        
                        processed_samples += 1
                        
                    except Exception as e:
                        print(f"Error processing sample {processed_samples}: {e}")
                        continue
        
        # Calculate average metrics
        avg_metrics = {}
        for key, values in metrics.items():
            if values:
                avg_metrics[f"avg_{key}"] = np.mean(values)
                avg_metrics[f"std_{key}"] = np.std(values)
            else:
                avg_metrics[f"avg_{key}"] = 0.0
                avg_metrics[f"std_{key}"] = 0.0
        
        return avg_metrics
    
    def benchmark_model(self, model: nn.Module, compression_stats: Dict, 
                       test_dataloader, sample_input: torch.Tensor) -> Dict:
        """Complete benchmark of a model"""
        print(f"\n{'='*50}")
        print("RUNNING COMPREHENSIVE MODEL BENCHMARK")
        print(f"{'='*50}")
        
        # Model statistics
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Performance measurements
        inference_times = self.measure_inference_time(model, sample_input)
        memory_usage = self.measure_memory_usage(model, sample_input)
        audio_quality = self.evaluate_audio_quality(model, test_dataloader)
        
        # Compile results
        results = {
            # Model statistics
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            **compression_stats,
            
            # Performance metrics
            **inference_times,
            **memory_usage,
            **audio_quality
        }
        
        return results
    
    def compare_models(self, original_results: Dict, optimized_results: Dict) -> Dict:
        """Compare original and optimized model results"""
        print(f"\n{'='*50}")
        print("MODEL COMPARISON ANALYSIS")
        print(f"{'='*50}")
        
        comparison = {}
        
        # Model size comparison
        original_params = original_results["total_parameters"]
        optimized_params = optimized_results.get("effective_parameters", optimized_results["total_parameters"])
        
        comparison["parameter_reduction"] = original_params - optimized_params
        comparison["parameter_reduction_ratio"] = optimized_params / original_params
        comparison["compression_ratio"] = optimized_results.get("compression_ratio", 1.0)
        
        # Speed comparison
        original_time = original_results["mean_time"]
        optimized_time = optimized_results["mean_time"]
        comparison["speedup"] = original_time / optimized_time
        
        # Memory comparison
        original_memory = original_results["model_size_mb"]
        optimized_memory = optimized_results["model_size_mb"]
        comparison["memory_reduction"] = original_memory - optimized_memory
        comparison["memory_reduction_ratio"] = optimized_memory / original_memory
        
        # Quality comparison
        quality_metrics = ["avg_mbstoi", "avg_pesq", "avg_segsnr"]
        for metric in quality_metrics:
            if metric in original_results and metric in optimized_results:
                original_val = original_results[metric]
                optimized_val = optimized_results[metric]
                comparison[f"{metric}_drop"] = original_val - optimized_val
                comparison[f"{metric}_retention"] = optimized_val / original_val if original_val != 0 else 1.0
        
        return comparison
    
    def print_results(self, results: Dict, title: str):
        """Pretty print benchmark results"""
        print(f"\n{title}")
        print("-" * len(title))
        
        # Model statistics
        print(f"Total Parameters: {results['total_parameters']:,}")
        if 'effective_parameters' in results:
            print(f"Effective Parameters: {results['effective_parameters']:,}")
        print(f"Model Size: {results['model_size_mb']:.2f} MB")
        
        # Performance metrics
        print(f"\nInference Time: {results['mean_time']*1000:.2f} ± {results['std_time']*1000:.2f} ms")
        
        if 'gpu_memory_peak_mb' in results:
            print(f"GPU Memory Peak: {results['gpu_memory_peak_mb']:.2f} MB")
        
        # Audio quality metrics
        print(f"\nAudio Quality Metrics:")
        print(f"  MBSTOI: {results.get('avg_mbstoi', 0):.4f} ± {results.get('std_mbstoi', 0):.4f}")
        print(f"  PESQ: {results.get('avg_pesq', 0):.3f} ± {results.get('std_pesq', 0):.3f}")
        print(f"  SegSNR: {results.get('avg_segsnr', 0):.2f} ± {results.get('std_segsnr', 0):.2f} dB")
        print(f"  ILD Error: {results.get('avg_ild_error', 0):.3f} ± {results.get('std_ild_error', 0):.3f} dB")
        print(f"  IPD Error: {results.get('avg_ipd_error', 0):.2f} ± {results.get('std_ipd_error', 0):.2f} deg")
    
    def print_comparison(self, comparison: Dict):
        """Pretty print model comparison"""
        print(f"\n{'='*50}")
        print("MODEL COMPARISON SUMMARY")
        print(f"{'='*50}")
        
        # Compression metrics
        print(f"Parameter Reduction: {comparison['parameter_reduction']:,} ({(1-comparison['parameter_reduction_ratio'])*100:.1f}%)")
        print(f"Compression Ratio: {comparison['compression_ratio']:.3f}")
        print(f"Memory Reduction: {comparison['memory_reduction']:.2f} MB ({(1-comparison['memory_reduction_ratio'])*100:.1f}%)")
        print(f"Speedup: {comparison['speedup']:.2f}x")
        
        # Quality retention
        print(f"\nQuality Retention:")
        if 'avg_mbstoi_retention' in comparison:
            print(f"  MBSTOI: {comparison['avg_mbstoi_retention']*100:.1f}% (drop: {comparison['avg_mbstoi_drop']:.4f})")
        if 'avg_pesq_retention' in comparison:
            print(f"  PESQ: {comparison['avg_pesq_retention']*100:.1f}% (drop: {comparison['avg_pesq_drop']:.3f})")
        if 'avg_segsnr_retention' in comparison:
            print(f"  SegSNR: {comparison['avg_segsnr_retention']*100:.1f}% (drop: {comparison['avg_segsnr_drop']:.2f} dB)")
    
    def save_results(self, results: Dict, filepath: str):
        """Save benchmark results to CSV"""
        df = pd.DataFrame([results])
        df.to_csv(filepath, index=False)
        print(f"Results saved to: {filepath}")


def create_sample_input(batch_size: int = 1, channels: int = 2, 
                       duration: float = 2.0, sample_rate: int = 16000) -> torch.Tensor:
    """Create sample input tensor for benchmarking"""
    samples = int(duration * sample_rate)
    return torch.randn(batch_size, channels, samples)


def main():
    parser = argparse.ArgumentParser(description="Evaluate QAT optimized models")
    parser.add_argument("--original", type=str, help="Path to original model checkpoint")
    parser.add_argument("--optimized", type=str, required=True, help="Path to optimized model checkpoint")
    parser.add_argument("--config", type=str, help="Path to model config file")
    parser.add_argument("--test_data", type=str, help="Path to test dataset")
    parser.add_argument("--output_dir", type=str, default="./benchmark_results", help="Output directory for results")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of test samples to evaluate")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Device to use")
    parser.add_argument("--benchmark_only", action="store_true", help="Only benchmark optimized model")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize benchmark
    benchmark = ModelBenchmark(device=args.device)
    
    # Create sample input for performance testing
    sample_input = create_sample_input()
    
    # Load test dataloader if specified
    test_dataloader = None
    if args.test_data:
        from hydra import compose, initialize
        from hydra.core.global_hydra import GlobalHydra
        
        # Load test data
        GlobalHydra.instance().clear()
        initialize(config_path="../config")
        config = compose(config_name="config")
        config["dataset"]["noisy_test_dataset_dir"] = args.test_data
        
        from DCNN.datasets import create_torch_dataloader
        test_dataloader = create_torch_dataloader(config, "test")
    else:
        # Create dummy dataloader for testing
        from torch.utils.data import TensorDataset, DataLoader
        dummy_data = TensorDataset(
            torch.randn(args.num_samples, 2, 32000),  # noisy
            torch.randn(args.num_samples, 2, 32000)   # clean
        )
        test_dataloader = DataLoader(dummy_data, batch_size=8, shuffle=False)
    
    # Load and benchmark optimized model
    print("Loading optimized model...")
    optimized_model, optimized_compression_stats = benchmark.load_model(args.optimized, args.config)
    optimized_results = benchmark.benchmark_model(
        optimized_model, optimized_compression_stats, test_dataloader, sample_input)
    
    benchmark.print_results(optimized_results, "OPTIMIZED MODEL RESULTS")
    
    # Save optimized model results
    optimized_results_path = output_dir / "optimized_model_benchmark.csv"
    benchmark.save_results(optimized_results, str(optimized_results_path))
    
    # Load and benchmark original model if provided
    if args.original and not args.benchmark_only:
        print("\nLoading original model...")
        original_model, original_compression_stats = benchmark.load_model(args.original, args.config)
        original_results = benchmark.benchmark_model(
            original_model, original_compression_stats, test_dataloader, sample_input)
        
        benchmark.print_results(original_results, "ORIGINAL MODEL RESULTS")
        
        # Save original model results
        original_results_path = output_dir / "original_model_benchmark.csv"
        benchmark.save_results(original_results, str(original_results_path))
        
        # Compare models
        comparison = benchmark.compare_models(original_results, optimized_results)
        benchmark.print_comparison(comparison)
        
        # Save comparison results
        comparison_path = output_dir / "model_comparison.csv"
        benchmark.save_results(comparison, str(comparison_path))
    
    # Generate summary report
    generate_summary_report(output_dir, optimized_results, 
                           original_results if args.original and not args.benchmark_only else None)
    
    print(f"\nBenchmark complete! Results saved to: {output_dir}")


def generate_summary_report(output_dir: Path, optimized_results: Dict, original_results: Dict = None):
    """Generate a comprehensive summary report"""
    report_path = output_dir / "benchmark_summary.txt"
    
    with open(report_path, 'w') as f:
        f.write("PRUNING-AWARE QAT MODEL BENCHMARK SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        # Optimized model summary
        f.write("OPTIMIZED MODEL PERFORMANCE:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Total Parameters: {optimized_results['total_parameters']:,}\n")
        
        if 'effective_parameters' in optimized_results:
            f.write(f"Effective Parameters: {optimized_results['effective_parameters']:,}\n")
            compression = optimized_results.get('compression_ratio', 1.0)
            f.write(f"Compression Ratio: {compression:.3f}\n")
            f.write(f"Model Size Reduction: {(1-compression)*100:.1f}%\n")
        
        f.write(f"Model Size: {optimized_results['model_size_mb']:.2f} MB\n")
        f.write(f"Inference Time: {optimized_results['mean_time']*1000:.2f} ms\n")
        f.write(f"MBSTOI: {optimized_results.get('avg_mbstoi', 0):.4f}\n")
        f.write(f"PESQ: {optimized_results.get('avg_pesq', 0):.3f}\n")
        f.write(f"SegSNR: {optimized_results.get('avg_segsnr', 0):.2f} dB\n\n")
        
        # Comparison if original model provided
        if original_results:
            f.write("OPTIMIZATION GAINS:\n")
            f.write("-" * 20 + "\n")
            
            original_params = original_results["total_parameters"]
            optimized_params = optimized_results.get("effective_parameters", optimized_results["total_parameters"])
            param_reduction = (1 - optimized_params / original_params) * 100
            
            speedup = original_results["mean_time"] / optimized_results["mean_time"]
            memory_reduction = (1 - optimized_results["model_size_mb"] / original_results["model_size_mb"]) * 100
            
            f.write(f"Parameter Reduction: {param_reduction:.1f}%\n")
            f.write(f"Memory Reduction: {memory_reduction:.1f}%\n")
            f.write(f"Speedup: {speedup:.2f}x\n")
            
            # Quality retention
            mbstoi_retention = (optimized_results.get('avg_mbstoi', 0) / original_results.get('avg_mbstoi', 1)) * 100
            f.write(f"MBSTOI Retention: {mbstoi_retention:.1f}%\n")
        
        f.write("\nRECOMMendations:\n")
        f.write("-" * 15 + "\n")
        
        # Provide recommendations based on results
        compression_ratio = optimized_results.get('compression_ratio', 1.0)
        if compression_ratio < 0.5:
            f.write("✓ Excellent compression achieved (>50% reduction)\n")
        elif compression_ratio < 0.7:
            f.write("✓ Good compression achieved (30-50% reduction)\n")
        else:
            f.write("- Moderate compression (consider more aggressive pruning)\n")
        
        mbstoi_score = optimized_results.get('avg_mbstoi', 0)
        if mbstoi_score > 0.85:
            f.write("✓ Excellent audio quality maintained\n")
        elif mbstoi_score > 0.75:
            f.write("✓ Good audio quality maintained\n")
        else:
            f.write("- Audio quality may need improvement\n")
        
        if original_results:
            speedup = original_results["mean_time"] / optimized_results["mean_time"]
            if speedup > 2.0:
                f.write("✓ Significant speedup achieved\n")
            elif speedup > 1.5:
                f.write("✓ Good speedup achieved\n")
            else:
                f.write("- Consider further optimization for speed\n")
    
    print(f"Summary report saved to: {report_path}")


if __name__ == "__main__":
    main()