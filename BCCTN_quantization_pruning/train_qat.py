#!/usr/bin/env python3
"""
Pruning-aware Quantization Aware Training (QAT) for BCCTN Model

This script implements structured pruning combined with quantization aware training
to achieve maximum model compression while maintaining performance.

Usage:
    python train_qat.py --config config/config_qat.yaml
    python train_qat.py --pretrained path/to/pretrained.ckpt
"""

import hydra
import torch
import os
import warnings
from omegaconf import DictConfig
from pathlib import Path

from DCNN.datasets import create_torch_dataloaders
from DCNN.trainer import PruningAwareQATTrainer
from DCNN.quantization import PruningAwareQAT

warnings.simplefilter('ignore')
torch.cuda.empty_cache()


def validate_qat_config(config: DictConfig):
    """Validate QAT configuration parameters"""
    if not config.get("quantization", {}).get("enable_pruning_qat", False):
        raise ValueError("Pruning-aware QAT is not enabled in config. Set quantization.enable_pruning_qat=true")
    
    qat_config = config["quantization"]
    
    # Validate pruning ratio
    base_ratio = qat_config.get("base_pruning_ratio", 0.3)
    if not 0.0 <= base_ratio <= 0.9:
        raise ValueError(f"base_pruning_ratio must be between 0.0 and 0.9, got {base_ratio}")
    
    # Validate quantization type
    qat_type = qat_config.get("type", "int8")
    if qat_type not in ["int8", "fp8"]:
        raise ValueError(f"Quantization type must be 'int8' or 'fp8', got {qat_type}")
    
    # Check GPU requirements for FP8
    if qat_type == "fp8":
        if not torch.cuda.is_available():
            raise ValueError("FP8 quantization requires CUDA")
        
        gpu_name = torch.cuda.get_device_name()
        if "H100" not in gpu_name and "H800" not in gpu_name:
            print(f"Warning: FP8 is optimized for Hopper GPUs (H100/H800), detected: {gpu_name}")
    
    print("✓ QAT configuration validation passed")


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


def analyze_model_before_optimization(trainer):
    """Analyze model characteristics before optimization"""
    model = trainer._lightning_module.model
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Analyze layer types
    conv_layers = sum(1 for m in model.modules() if isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)))
    linear_layers = sum(1 for m in model.modules() if isinstance(m, torch.nn.Linear))
    
    print("=== Model Analysis Before Optimization ===")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Conv/ConvTranspose layers: {conv_layers}")
    print(f"Linear layers: {linear_layers}")
    print(f"Model size (float32): {total_params * 4 / 1024 / 1024:.2f} MB")
    
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "conv_layers": conv_layers,
        "linear_layers": linear_layers
    }


def analyze_model_after_optimization(trainer, original_stats):
    """Analyze model characteristics after optimization"""
    if not trainer._lightning_module.pqat_manager:
        print("No QAT manager found, skipping post-optimization analysis")
        return
    
    stats = trainer._lightning_module.pqat_manager.get_compression_stats()
    
    print("\n=== Model Analysis After Optimization ===")
    print(f"Original parameters: {original_stats['total_params']:,}")
    print(f"Effective parameters: {stats['effective_parameters']:,}")
    print(f"Parameters reduction: {original_stats['total_params'] - stats['effective_parameters']:,}")
    print(f"Compression ratio: {stats['compression_ratio']:.3f}")
    print(f"Pruning ratio: {stats['pruning_ratio']:.1%}")
    print(f"Theoretical speedup: {stats['theoretical_speedup']:.2f}x")
    
    # Estimate model size reduction
    original_size_mb = original_stats['total_params'] * 4 / 1024 / 1024  # FP32
    compressed_size_mb = stats['effective_parameters'] * 1 / 1024 / 1024  # INT8 (1 byte per param)
    size_reduction = (1 - compressed_size_mb / original_size_mb) * 100
    
    print(f"Original model size: {original_size_mb:.2f} MB (FP32)")
    print(f"Optimized model size: {compressed_size_mb:.2f} MB (INT8)")
    print(f"Model size reduction: {size_reduction:.1f}%")


def create_export_directory(config):
    """Create directory for exporting optimized models"""
    export_path = config.get("optimization", {}).get("export_path", "./optimized_models/")
    export_dir = Path(export_path)
    export_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectory with timestamp
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = export_dir / f"qat_run_{timestamp}"
    run_dir.mkdir(exist_ok=True)
    
    return run_dir


@hydra.main(config_path="config", config_name="config", version_base="1.1")
def train_pruning_aware_qat(config: DictConfig):
    """
    Main training function for Pruning-aware QAT
    
    Args:
        config: Hydra configuration with QAT settings
    """
    print("=" * 60)
    print("PRUNING-AWARE QUANTIZATION AWARE TRAINING (QAT)")
    print("=" * 60)
    
    # Validate configuration
    validate_qat_config(config)
    
    # Create export directory
    export_dir = create_export_directory(config)
    print(f"Export directory: {export_dir}")
    
    # Load datasets
    print("\nLoading datasets...")
    dataset_train, dataset_val, dataset_test = create_torch_dataloaders(config)
    print(f"Training samples: {len(dataset_train.dataset)}")
    print(f"Validation samples: {len(dataset_val.dataset)}")
    print(f"Test samples: {len(dataset_test.dataset)}")
    
    # Initialize pruning-aware QAT trainer
    print("\nInitializing Pruning-aware QAT trainer...")
    trainer = PruningAwareQATTrainer(config)
    
    # Analyze model before optimization
    original_stats = analyze_model_before_optimization(trainer)
    
    # Note: Pretrained weights are now loaded in DCNNLightningModule.__init__
    # Skip normal training phase if using pretrained model
    if config.get("training", {}).get("train_checkpoint_path"):
        config["quantization"]["skip_normal_training"] = True
    
    # Run multi-phase training
    print("\nStarting multi-phase pruning-aware QAT training...")
    try:
        trainer.fit_with_qat_phases(dataset_train, val_dataloaders=dataset_val)
    except Exception as e:
        print(f"Training failed: {e}")
        raise
    
    # Analyze model after optimization
    analyze_model_after_optimization(trainer, original_stats)
    
    # Test the optimized model
    print("\nTesting optimized model...")
    try:
        trainer.test(dataset_test)
    except Exception as e:
        print(f"Testing failed: {e}")
    
    # Export optimized model
    if config.get("optimization", {}).get("export_optimized_model", True):
        export_path = export_dir / "optimized_bcctn_model.ckpt"
        print(f"\nExporting optimized model to: {export_path}")
        trainer.export_for_deployment(str(export_path))
        
        # Save configuration used
        config_path = export_dir / "optimization_config.yaml"
        with open(config_path, 'w') as f:
            from omegaconf import OmegaConf
            OmegaConf.save(config=config, f=f)
        print(f"Configuration saved to: {config_path}")
    
    print("\n" + "=" * 60)
    print("PRUNING-AWARE QAT TRAINING COMPLETE!")
    print("=" * 60)


def benchmark_model_performance(model_path: str, test_dataloader):
    """Benchmark the performance of the optimized model"""
    print(f"\nBenchmarking model: {model_path}")
    
    # Load model
    checkpoint = torch.load(model_path, map_location='cpu')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # TODO: Implement benchmarking logic
    # - Measure inference time
    # - Measure memory usage  
    # - Calculate MBSTOI, SegSNR, etc.
    
    print("Benchmark complete!")


if __name__ == "__main__":
    # Set environment variables for optimal performance
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    
    # Enable mixed precision if available
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    
    train_pruning_aware_qat()