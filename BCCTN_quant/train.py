#!/usr/bin/env python3
"""
Training script for quantization-only experiments (no pruning)
Supports mixed precision quantization for BCCTN model

Usage:
    # Basic quantization training
    python train_quantization_only.py
    
    # With custom config
    python train_quantization_only.py --config config_quantization.yaml
    
    # Different precision settings
    python train_quantization_only.py --conv_precision int8 --attention_precision fp16
    
    # Using pre-trained model
    python train_quantization_only.py --pretrained /path/to/pretrained.ckpt
"""

import hydra
import torch
import os
import warnings
import argparse
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

from DCNN.datasets import create_torch_dataloaders
from DCNN.trainer_quantization import QuantizationAwareTrainer, create_quantization_config

warnings.simplefilter('ignore')
torch.cuda.empty_cache()


def validate_quantization_config(config: DictConfig):
    """Validate quantization configuration parameters"""
    quant_config = config.get("quantization", {})
    
    # Validate precision settings
    conv_precision = quant_config.get("conv_precision", "int8")
    attention_precision = quant_config.get("attention_precision", "fp16")
    
    valid_precisions = ["int8", "fp16", "fp8", "fp32"]
    
    if conv_precision not in valid_precisions:
        raise ValueError(f"Invalid conv_precision: {conv_precision}. Must be one of {valid_precisions}")
    
    if attention_precision not in valid_precisions:
        raise ValueError(f"Invalid attention_precision: {attention_precision}. Must be one of {valid_precisions}")
    
    # Check FP8 requirements
    if attention_precision == "fp8":
        if not torch.cuda.is_available():
            raise ValueError("FP8 quantization requires CUDA")
        
        gpu_name = torch.cuda.get_device_name()
        if "H100" not in gpu_name and "H800" not in gpu_name:
            print(f"Warning: FP8 is optimized for Hopper GPUs (H100/H800), detected: {gpu_name}")
    
    print("✓ Quantization configuration validation passed")


def load_pretrained_model(trainer, checkpoint_path: str):
    """Load pretrained model weights"""
    print(f"Loading pretrained model from: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        trainer._lightning_module.load_state_dict(checkpoint["state_dict"])
        print("✓ Pretrained model loaded successfully")
        return True
    except Exception as e:
        print(f"Error loading pretrained model: {e}")
        return False


def analyze_model_before_quantization(trainer):
    """Analyze model characteristics before quantization"""
    model = trainer._lightning_module.model
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Analyze layer types
    conv_layers = sum(1 for m in model.modules() if isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)))
    linear_layers = sum(1 for m in model.modules() if isinstance(m, torch.nn.Linear))
    attention_layers = sum(1 for name, m in model.named_modules() if 'mattn' in name.lower() or 'attention' in name.lower())
    
    print("=== Model Analysis Before Quantization ===")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Conv/ConvTranspose layers: {conv_layers}")
    print(f"Linear layers: {linear_layers}")
    print(f"Attention layers: {attention_layers}")
    print(f"Model size (float32): {total_params * 4 / 1024 / 1024:.2f} MB")
    
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "conv_layers": conv_layers,
        "linear_layers": linear_layers,
        "attention_layers": attention_layers
    }


def analyze_model_after_quantization(trainer, original_stats):
    """Analyze model characteristics after quantization"""
    if not trainer._lightning_module.quantizer:
        print("No quantizer found, skipping post-quantization analysis")
        return
    
    stats = trainer._lightning_module.quantizer.get_quantization_stats()
    
    print("\n=== Model Analysis After Quantization ===")
    print(f"Original parameters: {original_stats['total_params']:,}")
    print(f"Total layers: {stats['total_layers']}")
    print(f"Quantized layers: {stats['quantized_layers']}")
    print(f"INT8 layers: {stats['int8_layers']}")
    print(f"FP16 layers: {stats['fp16_layers']}")
    print(f"FP8 layers: {stats['fp8_layers']}")
    
    # Model size comparison
    original_size_mb = stats['original_size_mb']
    compressed_size_mb = stats['compressed_size_mb']
    compression_ratio = stats['compression_ratio']
    size_reduction = (1 - compression_ratio) * 100
    
    print(f"Original model size: {original_size_mb:.2f} MB")
    print(f"Quantized model size: {compressed_size_mb:.2f} MB")
    print(f"Compression ratio: {compression_ratio:.3f}")
    print(f"Size reduction: {size_reduction:.1f}%")


def create_export_directory(config):
    """Create directory for exporting quantized models"""
    export_path = config.get("quantization", {}).get("export_path", "./quantized_models/")
    export_dir = Path(export_path)
    export_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectory with timestamp
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = export_dir / f"quantization_run_{timestamp}"
    run_dir.mkdir(exist_ok=True)
    
    return run_dir


@hydra.main(config_path="config", config_name="config_quantization", version_base="1.1")
def train_quantization_only(config: DictConfig):
    """
    Main training function for quantization-only experiments
    
    Args:
        config: Hydra configuration with quantization settings
    """
    print("=" * 60)
    print("QUANTIZATION-ONLY TRAINING FOR BCCTN MODEL")
    print("=" * 60)
    
    # Validate configuration
    validate_quantization_config(config)
    
    # Create export directory
    export_dir = create_export_directory(config)
    print(f"Export directory: {export_dir}")
    
    # Load datasets
    print("\nLoading datasets...")
    dataset_train, dataset_val, dataset_test = create_torch_dataloaders(config)
    print(f"Training samples: {len(dataset_train.dataset)}")
    print(f"Validation samples: {len(dataset_val.dataset)}")
    print(f"Test samples: {len(dataset_test.dataset)}")
    
    # Initialize quantization-aware trainer
    print("\nInitializing Quantization-Aware trainer...")
    trainer = QuantizationAwareTrainer(config)
    
    # Analyze model before quantization
    original_stats = analyze_model_before_quantization(trainer)
    
    # Load pretrained model if specified
    pretrained_path = config.get("training", {}).get("train_checkpoint_path")
    if pretrained_path:
        load_pretrained_model(trainer, pretrained_path)
        # Skip normal training phase if using pretrained model
        config["quantization"]["phases"]["skip_normal_training"] = True
    
    # Get quantization settings
    conv_precision = config["quantization"]["conv_precision"]
    attention_precision = config["quantization"]["attention_precision"]
    
    # Run quantization training
    print(f"\nStarting quantization training...")
    print(f"Conv precision: {conv_precision}")
    print(f"Attention precision: {attention_precision}")
    
    try:
        # Use multi-phase training if enabled
        if config["quantization"]["phases"]["enable_multi_phase"]:
            stats = trainer.fit_with_quantization_phases(
                dataset_train, 
                dataset_val,
                conv_precision=conv_precision,
                attention_precision=attention_precision
            )
        else:
            # Simple quantization and fine-tuning
            stats = trainer.quantize_and_finetune(
                dataset_train,
                dataset_val,
                conv_precision=conv_precision,
                attention_precision=attention_precision,
                num_epochs=config["quantization"]["phases"]["finetune_epochs"]
            )
    except Exception as e:
        print(f"Training failed: {e}")
        raise
    
    # Analyze model after quantization
    analyze_model_after_quantization(trainer, original_stats)
    
    # Test the quantized model
    print("\nTesting quantized model...")
    try:
        trainer.test(dataset_test)
    except Exception as e:
        print(f"Testing failed: {e}")
    
    # Export quantized model
    if config.get("quantization", {}).get("export_quantized_model", True):
        model_name = f"quantized_bcctn_{conv_precision}_{attention_precision}.ckpt"
        export_path = export_dir / model_name
        print(f"\nExporting quantized model to: {export_path}")
        trainer.export_for_deployment(str(export_path))
        
        # Save configuration used
        config_path = export_dir / "quantization_config.yaml"
        with open(config_path, 'w') as f:
            OmegaConf.save(config=config, f=f)
        print(f"Configuration saved to: {config_path}")
    
    # Run benchmark tests if configured
    if config.get("quantization_test", {}).get("enable", False):
        print("\n" + "=" * 60)
        print("RUNNING QUANTIZATION BENCHMARK TESTS")
        print("=" * 60)
        
        test_configs = []
        for test_config in config["quantization_test"]["test_configurations"]:
            test_configs.append((test_config["conv"], test_config["attention"]))
        
        benchmark_results = trainer.benchmark_quantization(dataset_test, test_configs)
        
        # Save benchmark results
        import pandas as pd
        df = pd.DataFrame(benchmark_results)
        benchmark_path = export_dir / "quantization_benchmark.csv"
        df.to_csv(benchmark_path, index=False)
        print(f"Benchmark results saved to: {benchmark_path}")
    
    print("\n" + "=" * 60)
    print("QUANTIZATION TRAINING COMPLETE!")
    print("=" * 60)


def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(description="Quantization-only training for BCCTN")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--conv_precision", type=str, choices=["int8", "fp16", "fp32"],
                       help="Convolution layer precision")
    parser.add_argument("--attention_precision", type=str, choices=["fp16", "fp8", "fp32"],
                       help="Attention layer precision")
    parser.add_argument("--pretrained", type=str, help="Path to pretrained model")
    parser.add_argument("--export_path", type=str, help="Export path for quantized model")
    
    args = parser.parse_args()
    
    # Override config with command line arguments
    if args.config:
        os.environ["HYDRA_CONFIG_PATH"] = os.path.dirname(args.config)
        os.environ["HYDRA_CONFIG_NAME"] = os.path.basename(args.config).replace('.yaml', '')
    
    # Set environment variables for argument overrides
    overrides = []
    if args.conv_precision:
        overrides.append(f"quantization.conv_precision={args.conv_precision}")
    if args.attention_precision:
        overrides.append(f"quantization.attention_precision={args.attention_precision}")
    if args.pretrained:
        overrides.append(f"training.train_checkpoint_path={args.pretrained}")
    if args.export_path:
        overrides.append(f"quantization.export_path={args.export_path}")
    
    # Set Hydra overrides
    if overrides:
        import sys
        sys.argv.extend(overrides)
    
    # Run training
    train_quantization_only()


if __name__ == "__main__":
    # Set environment variables for optimal performance
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    
    # Enable mixed precision if available
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    
    main()