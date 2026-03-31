#!/usr/bin/env python3
"""
Complete test script for quantization experiments on BCCTN model
Tests different quantization configurations and measures performance
"""

import torch
import torch.nn as nn
import numpy as np
import time
import argparse
from pathlib import Path
import json
import pandas as pd
from typing import Dict, List, Tuple

from DCNN.trainer import DCNNLightningModule
from DCNN.datasets import create_torch_dataloader
from DCNN.quantization_only import MixedPrecisionQuantizer, quantize_model_mixed_precision
from eval import safe_mbstoi, fwsegsnr, compute_ild_error_with_mask, compute_ipd_error_with_mask

# Import necessary modules
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra


class QuantizationTester:
    """Test different quantization configurations"""
    
    def __init__(self, base_model_path: str, config_path: str = "config"):
        self.base_model_path = base_model_path
        self.config_path = config_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load configuration
        self._load_config()
        
        # Results storage
        self.results = []
        
    def _load_config(self):
        """Load model configuration"""
        GlobalHydra.instance().clear()
        initialize(config_path=self.config_path)
        self.config = compose(config_name="config")
    
    def load_base_model(self) -> nn.Module:
        """Load the base model"""
        print(f"Loading base model from: {self.base_model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.base_model_path, map_location=self.device)
        
        # Initialize model
        model = DCNNLightningModule(self.config)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        model = model.to(self.device)
        
        return model
    
    def test_configuration(self, conv_precision: str, attention_precision: str,
                          test_dataloader, num_samples: int = 100):
        """Test a specific quantization configuration"""
        print(f"\n{'='*60}")
        print(f"Testing: Conv={conv_precision}, Attention={attention_precision}")
        print(f"{'='*60}")
        
        # Load fresh model for each test
        model = self.load_base_model()
        
        # Apply quantization
        start_time = time.time()
        quantizer = quantize_model_mixed_precision(
            model.model,
            conv_precision=conv_precision,
            attention_precision=attention_precision
        )
        quantization_time = time.time() - start_time
        
        # Get quantization stats
        quant_stats = quantizer.get_quantization_stats()
        
        # Calibrate if using INT8
        if conv_precision == "int8":
            print("Running calibration...")
            self._run_calibration(model, test_dataloader, num_batches=10)
        
        # Measure inference performance
        print("Measuring inference performance...")
        perf_stats = self._measure_inference_performance(model, test_dataloader, num_samples=20)
        
        # Measure audio quality
        print("Evaluating audio quality...")
        quality_stats = self._evaluate_audio_quality(model, test_dataloader, num_samples=num_samples)
        
        # Compile results
        result = {
            "conv_precision": conv_precision,
            "attention_precision": attention_precision,
            "quantization_time": quantization_time,
            **quant_stats,
            **perf_stats,
            **quality_stats
        }
        
        self.results.append(result)
        
        # Print summary
        self._print_result_summary(result)
        
        return result
    
    def _run_calibration(self, model: nn.Module, dataloader, num_batches: int = 10):
        """Run calibration for INT8 quantization"""
        model.eval()
        
        # Enable calibration
        for module in model.modules():
            if hasattr(module, 'enable_calib'):
                module.enable_calib()
        
        # Run calibration batches
        with torch.no_grad():
            for i, (noisy, clean) in enumerate(dataloader):
                if i >= num_batches:
                    break
                    
                noisy = noisy.to(self.device)
                _ = model(noisy)
        
        # Disable calibration
        for module in model.modules():
            if hasattr(module, 'disable_calib'):
                module.disable_calib()
    
    def _measure_inference_performance(self, model: nn.Module, dataloader, 
                                     num_samples: int = 20) -> Dict:
        """Measure inference performance metrics"""
        model.eval()
        
        # Create sample input
        sample_batch = next(iter(dataloader))[0][:1].to(self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = model(sample_batch)
        
        # Measure latency
        latencies = []
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        
        with torch.no_grad():
            for _ in range(num_samples):
                start = time.perf_counter()
                _ = model(sample_batch)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                end = time.perf_counter()
                latencies.append((end - start) * 1000)  # ms
        
        # Measure throughput
        batch_sizes = [1, 4, 8, 16]
        throughputs = {}
        
        for bs in batch_sizes:
            if bs > len(dataloader.dataset):
                continue
                
            batch = next(iter(dataloader))[0][:bs].to(self.device)
            
            with torch.no_grad():
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()
                
                for _ in range(10):
                    _ = model(batch)
                    
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                end = time.perf_counter()
                
                throughputs[f"throughput_bs{bs}"] = (bs * 10) / (end - start)
        
        return {
            "latency_mean_ms": np.mean(latencies),
            "latency_std_ms": np.std(latencies),
            "latency_min_ms": np.min(latencies),
            "latency_max_ms": np.max(latencies),
            **throughputs
        }
    
    def _evaluate_audio_quality(self, model: nn.Module, dataloader, 
                              num_samples: int = 100) -> Dict:
        """Evaluate audio quality metrics"""
        model.eval()
        
        metrics = {
            'mbstoi': [],
            'segsnr': [],
            'ild_error': [],
            'ipd_error': []
        }
        
        processed = 0
        
        with torch.no_grad():
            for noisy, clean in dataloader:
                if processed >= num_samples:
                    break
                    
                noisy = noisy.to(self.device)
                enhanced = model(noisy)
                
                # Process each sample
                noisy_np = noisy.cpu().numpy()
                clean_np = clean.numpy()
                enhanced_np = enhanced.cpu().numpy()
                
                for i in range(min(noisy.size(0), num_samples - processed)):
                    try:
                        # MBSTOI
                        mbstoi = safe_mbstoi(
                            clean_np[i, 0], clean_np[i, 1],
                            enhanced_np[i, 0], enhanced_np[i, 1]
                        )
                        if mbstoi is not None:
                            metrics['mbstoi'].append(mbstoi)
                        
                        # SegSNR
                        segsnr_l = fwsegsnr(clean_np[i, 0], noisy_np[i, 0], enhanced_np[i, 0])
                        segsnr_r = fwsegsnr(clean_np[i, 1], noisy_np[i, 1], enhanced_np[i, 1])
                        metrics['segsnr'].append((segsnr_l + segsnr_r) / 2)
                        
                        # ILD error
                        ild_error = compute_ild_error_with_mask(
                            clean_np[i, 0], clean_np[i, 1],
                            enhanced_np[i, 0], enhanced_np[i, 1]
                        )
                        metrics['ild_error'].append(ild_error)
                        
                        # IPD error
                        ipd_error = compute_ipd_error_with_mask(
                            clean_np[i, 0], clean_np[i, 1],
                            enhanced_np[i, 0], enhanced_np[i, 1]
                        )
                        metrics['ipd_error'].append(ipd_error * 180 / np.pi)
                        
                        processed += 1
                        
                    except Exception as e:
                        print(f"Error processing sample: {e}")
                        continue
        
        # Calculate statistics
        stats = {}
        for metric, values in metrics.items():
            if values:
                stats[f"{metric}_mean"] = np.mean(values)
                stats[f"{metric}_std"] = np.std(values)
            else:
                stats[f"{metric}_mean"] = 0.0
                stats[f"{metric}_std"] = 0.0
        
        return stats
    
    def _print_result_summary(self, result: Dict):
        """Print summary of test results"""
        print(f"\nConfiguration: Conv={result['conv_precision']}, Attention={result['attention_precision']}")
        print(f"Model size: {result['original_size_mb']:.2f} MB → {result['compressed_size_mb']:.2f} MB")
        print(f"Compression ratio: {result['compression_ratio']:.3f}")
        print(f"Latency: {result['latency_mean_ms']:.2f} ± {result['latency_std_ms']:.2f} ms")
        print(f"MBSTOI: {result.get('mbstoi_mean', 0):.4f} ± {result.get('mbstoi_std', 0):.4f}")
        print(f"SegSNR: {result.get('segsnr_mean', 0):.2f} ± {result.get('segsnr_std', 0):.2f} dB")
    
    def run_all_tests(self, test_configurations: List[Tuple[str, str]], 
                     test_dataloader, num_samples: int = 100):
        """Run all test configurations"""
        print("Starting quantization tests...")
        print(f"Device: {self.device}")
        
        for conv_prec, attn_prec in test_configurations:
            try:
                self.test_configuration(conv_prec, attn_prec, test_dataloader, num_samples)
            except Exception as e:
                print(f"Error testing {conv_prec}/{attn_prec}: {e}")
                continue
        
        return self.results
    
    def save_results(self, output_path: str):
        """Save test results"""
        # Save as CSV
        df = pd.DataFrame(self.results)
        csv_path = Path(output_path).with_suffix('.csv')
        df.to_csv(csv_path, index=False)
        print(f"Results saved to: {csv_path}")
        
        # Save as JSON
        json_path = Path(output_path).with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Generate comparison plot
        self._plot_results(df, Path(output_path).parent)
    
    def _plot_results(self, df: pd.DataFrame, output_dir: Path):
        """Generate comparison plots"""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Model size comparison
            ax = axes[0, 0]
            x_labels = [f"{row['conv_precision']}/{row['attention_precision']}" 
                       for _, row in df.iterrows()]
            ax.bar(x_labels, df['compressed_size_mb'])
            ax.set_ylabel('Model Size (MB)')
            ax.set_title('Model Size Comparison')
            ax.tick_params(axis='x', rotation=45)
            
            # Latency comparison
            ax = axes[0, 1]
            ax.bar(x_labels, df['latency_mean_ms'], yerr=df['latency_std_ms'])
            ax.set_ylabel('Latency (ms)')
            ax.set_title('Inference Latency')
            ax.tick_params(axis='x', rotation=45)
            
            # MBSTOI comparison
            ax = axes[1, 0]
            ax.bar(x_labels, df['mbstoi_mean'], yerr=df['mbstoi_std'])
            ax.set_ylabel('MBSTOI')
            ax.set_title('Audio Quality (MBSTOI)')
            ax.tick_params(axis='x', rotation=45)
            
            # SegSNR comparison
            ax = axes[1, 1]
            ax.bar(x_labels, df['segsnr_mean'], yerr=df['segsnr_std'])
            ax.set_ylabel('SegSNR (dB)')
            ax.set_title('Signal-to-Noise Ratio')
            ax.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plot_path = output_dir / 'quantization_comparison.png'
            plt.savefig(plot_path, dpi=150)
            plt.close()
            print(f"Comparison plot saved to: {plot_path}")
            
        except ImportError:
            print("Matplotlib not available, skipping plots")


def main():
    parser = argparse.ArgumentParser(description="Test quantization configurations for BCCTN")
    parser.add_argument("--model", required=True, help="Path to base model checkpoint")
    parser.add_argument("--test_data", required=True, help="Path to test dataset")
    parser.add_argument("--output", default="./quantization_results", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to test")
    parser.add_argument("--configs", nargs='+', 
                        default=["fp32/fp32", "int8/fp16", "int8/fp32", "fp16/fp16"],
                        help="Quantization configs to test (format: conv/attention)")
    
    args = parser.parse_args()
    
    # Parse configurations
    test_configs = []
    for config_str in args.configs:
        conv, attn = config_str.split('/')
        test_configs.append((conv, attn))
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize tester
    tester = QuantizationTester(args.model)
    
    # Load test data
    from hydra import compose, initialize
    from hydra.core.global_hydra import GlobalHydra
    
    GlobalHydra.instance().clear()
    initialize(config_path="../config")
    config = compose(config_name="config")
    config["dataset"]["noisy_test_dataset_dir"] = args.test_data
    
    from DCNN.datasets import create_torch_dataloader
    test_dataloader = create_torch_dataloader(config, "test")
    
    # Run tests
    results = tester.run_all_tests(test_configs, test_dataloader, args.num_samples)
    
    # Save results
    tester.save_results(str(output_dir / "quantization_test_results"))
    
    # Print summary
    print("\n" + "="*60)
    print("QUANTIZATION TEST SUMMARY")
    print("="*60)
    
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    
    # Find best configuration
    if not df.empty and 'mbstoi_mean' in df.columns:
        best_config = df.loc[df['mbstoi_mean'].idxmax()]
        print(f"\nBest configuration (by MBSTOI):")
        print(f"  Conv: {best_config['conv_precision']}, Attention: {best_config['attention_precision']}")
        print(f"  MBSTOI: {best_config['mbstoi_mean']:.4f}")
        print(f"  Latency: {best_config['latency_mean_ms']:.2f} ms")
        print(f"  Compression: {best_config['compression_ratio']:.3f}")


if __name__ == "__main__":
    main()