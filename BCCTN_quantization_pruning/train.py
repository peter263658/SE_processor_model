#!/usr/bin/env python3
"""
Enhanced Training Script with Pruning-aware QAT Support

This script can run either:
1. Normal training (when QAT is disabled in config)
2. Pruning-aware QAT training (when enabled in config)

Usage:
    # Normal training
    python train.py
    
    # QAT training
    python train.py --enable_qat
    
    # QAT with custom config
    python train.py --config config/config_qat.yaml
"""

import hydra
import torch
import os
import warnings
from omegaconf import DictConfig
from pathlib import Path

from DCNN.datasets import create_torch_dataloaders
from DCNN.trainer import PruningAwareQATTrainer

from train_qat import (validate_qat_config, load_pretrained_model, 
                      analyze_model_before_optimization, analyze_model_after_optimization,
                      create_export_directory, train_pruning_aware_qat)


warnings.simplefilter('ignore')
torch.cuda.empty_cache()


@hydra.main(config_path="config", config_name="config", version_base="1.1")
def train(config: DictConfig):
    """
    Enhanced training function that supports both normal and QAT training
    
    Args:
        config: Hydra configuration
    """
    # Check if QAT is enabled
    enable_qat = config.get("quantization", {}).get("enable_pruning_qat", False)
    
    if enable_qat:
        print("=" * 60)
        print("PRUNING-AWARE QUANTIZATION AWARE TRAINING (QAT)")
        print("=" * 60)
        
        # Run QAT training
        train_pruning_aware_qat(config)
    else:
        print("=" * 60)
        print("STANDARD TRAINING")
        print("=" * 60)
        
        # Run standard training
        train_standard(config)


def train_standard(config: DictConfig):
    """Standard training procedure"""
    dataset_train, dataset_val, dataset_test = create_torch_dataloaders(config)
    
    from DCNN.trainer import DCNNTrainer
    trainer = DCNNTrainer(config)

    trainer.fit(dataset_train, val_dataloaders=dataset_val)
    trainer.test(dataset_test)
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
    
    # Load pretrained model if specified
    pretrained_path = config.get("training", {}).get("train_checkpoint_path")
    if pretrained_path:
        load_pretrained_model(trainer, pretrained_path)
        # Skip normal training phase if using pretrained model
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
        torch.set_float32_matmul_precision('high')
    train()