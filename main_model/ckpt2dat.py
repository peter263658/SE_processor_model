#!/usr/bin/env python3
"""
Checkpoint to DAT File Exporter
================================

This script reads a PyTorch checkpoint file and exports each layer's parameters
to individual .dat files named according to the layer names.

Usage:
    python checkpoint_to_dat_exporter.py --checkpoint path/to/model.ckpt --output_dir ./dat_files

Features:
- Handles both .ckpt and .pth checkpoint formats
- Supports complex-valued parameters (splits real/imaginary parts)
- Creates organized directory structure
- Exports weights and biases separately
- Provides detailed logging and statistics
"""

import os
import sys
import argparse
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, Union
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('export.log')
    ]
)
logger = logging.getLogger(__name__)


class CheckpointExporter:
    """Exports PyTorch checkpoint parameters to .dat files"""
    
    def __init__(self, output_dir: str = "./dat_files"):
        """
        Initialize the exporter
        
        Args:
            output_dir: Directory to save .dat files
        """
        self.output_dir = Path(output_dir)
        self.precision = 6  # Default decimal precision
        self.elements_per_row = 8  # Default elements per row
        self.stats = {
            'total_layers': 0,
            'total_parameters': 0,
            'complex_layers': 0,
            'exported_files': 0
        }
        
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load checkpoint from file
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Dictionary containing model state
        """
        try:
            checkpoint_path = Path(checkpoint_path)
            
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
            
            logger.info(f"Loading checkpoint from: {checkpoint_path}")
            
            # Load checkpoint with CPU mapping to avoid GPU dependency
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                    logger.info("Found 'state_dict' in checkpoint")
                elif 'model' in checkpoint:
                    state_dict = checkpoint['model']
                    logger.info("Found 'model' in checkpoint")
                else:
                    # Assume the checkpoint itself is the state_dict
                    state_dict = checkpoint
                    logger.info("Using checkpoint as state_dict directly")
            else:
                raise ValueError(f"Unsupported checkpoint format: {type(checkpoint)}")
            
            logger.info(f"Loaded checkpoint with {len(state_dict)} parameter tensors")
            return state_dict
            
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            raise
    
    def sanitize_filename(self, name: str) -> str:
        """
        Sanitize layer name for use as filename
        
        Args:
            name: Original layer name
            
        Returns:
            Sanitized filename
        """
        # Remove problematic characters for filename
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            name = name.replace(char, '_')
        
        # Replace dots with underscores for nested module names
        name = name.replace('.', '_')
        
        # Remove model prefix if present
        if name.startswith('model_'):
            name = name[6:]
        
        return name
    
    def export_tensor_to_dat(self, tensor: torch.Tensor, filepath: Path, 
                           tensor_name: str = "") -> None:
        """
        Export a tensor to human-readable .dat file
        
        Args:
            tensor: PyTorch tensor to export
            filepath: Output file path
            tensor_name: Name of the tensor for logging
        """
        try:
            # Ensure output directory exists
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert tensor to numpy
            if tensor.is_complex():
                # Handle complex tensors by splitting real and imaginary parts
                real_part = tensor.real.detach().cpu().numpy()
                imag_part = tensor.imag.detach().cpu().numpy()
                
                # Save real part as text
                real_filepath = filepath.with_suffix('.real.dat')
                self._save_array_as_text(real_part, real_filepath)
                logger.info(f"Exported {tensor_name} (real part) to {real_filepath}")
                
                # Save imaginary part as text
                imag_filepath = filepath.with_suffix('.imag.dat')
                self._save_array_as_text(imag_part, imag_filepath)
                logger.info(f"Exported {tensor_name} (imag part) to {imag_filepath}")
                
                self.stats['complex_layers'] += 1
                self.stats['exported_files'] += 2
                
                # Also save shape information
                shape_filepath = filepath.with_suffix('.shape.txt')
                with open(shape_filepath, 'w') as f:
                    f.write(f"Shape: {list(tensor.shape)}\n")
                    f.write(f"Dtype: complex64\n")
                    f.write(f"Real file: {real_filepath.name}\n")
                    f.write(f"Imag file: {imag_filepath.name}\n")
                
            else:
                # Handle real tensors
                numpy_array = tensor.detach().cpu().numpy()
                
                # Save as human-readable text
                self._save_array_as_text(numpy_array, filepath)
                logger.info(f"Exported {tensor_name} to {filepath}")
                self.stats['exported_files'] += 1
                
                # Save shape information
                shape_filepath = filepath.with_suffix('.shape.txt')
                with open(shape_filepath, 'w') as f:
                    f.write(f"Shape: {list(tensor.shape)}\n")
                    f.write(f"Dtype: {numpy_array.dtype}\n")
                    f.write(f"Data file: {filepath.name}\n")
            
            # Update statistics
            self.stats['total_parameters'] += tensor.numel()
            
        except Exception as e:
            logger.error(f"Error exporting tensor {tensor_name}: {e}")
            raise
    
    def _save_array_as_text(self, array: np.ndarray, filepath: Path) -> None:
        """
        Save numpy array as human-readable text file
        
        Args:
            array: NumPy array to save
            filepath: Output file path
        """
        with open(filepath, 'w') as f:
            # Write header with shape information
            f.write(f"# Shape: {list(array.shape)}\n")
            f.write(f"# Total elements: {array.size}\n")
            f.write(f"# Data format: text (precision: {self.precision})\n")
            f.write("# Data starts below:\n")
            
            # Flatten array for easier processing
            flat_array = array.flatten()
            
            # Write data in rows of reasonable length
            for i in range(0, len(flat_array), self.elements_per_row):
                row_data = flat_array[i:i + self.elements_per_row]
                
                # Format numbers with specified precision
                formatted_numbers = [f"{num:.{self.precision}f}" for num in row_data]
                
                # Write row
                f.write(" ".join(formatted_numbers))
                f.write("\n")
            
            # Write footer with array statistics
            f.write("\n# Statistics:\n")
            f.write(f"# Min: {array.min():.{self.precision}f}\n")
            f.write(f"# Max: {array.max():.{self.precision}f}\n")
            f.write(f"# Mean: {array.mean():.{self.precision}f}\n")
            f.write(f"# Std: {array.std():.{self.precision}f}\n")
    
    def export_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """
        Export entire state dictionary to .dat files
        
        Args:
            state_dict: Model state dictionary
        """
        logger.info("Starting export process...")
        
        # Group parameters by layer
        layer_groups = self._group_parameters_by_layer(state_dict)
        
        for layer_name, parameters in layer_groups.items():
            logger.info(f"Processing layer: {layer_name}")
            
            # Create layer directory
            layer_dir = self.output_dir / self.sanitize_filename(layer_name)
            layer_dir.mkdir(parents=True, exist_ok=True)
            
            # Export each parameter in this layer
            for param_name, tensor in parameters.items():
                # Create filename based on layer and parameter name
                filename = f"{param_name}.dat"
                filepath = layer_dir / filename
                
                # Export tensor
                full_name = f"{layer_name}.{param_name}"
                self.export_tensor_to_dat(tensor, filepath, full_name)
            
            self.stats['total_layers'] += 1
        
        # Create summary file
        self._create_export_summary()
        
        logger.info("Export process completed successfully!")
    
    def _group_parameters_by_layer(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Group parameters by their layer names
        
        Args:
            state_dict: Model state dictionary
            
        Returns:
            Dictionary grouped by layer names
        """
        layer_groups = {}
        
        for param_name, tensor in state_dict.items():
            # Split parameter name to extract layer name and parameter type
            parts = param_name.split('.')
            
            if len(parts) >= 2:
                # Extract layer name (everything except the last part)
                layer_name = '.'.join(parts[:-1])
                param_type = parts[-1]  # weight, bias, etc.
            else:
                # Handle single-level parameter names
                layer_name = "root"
                param_type = param_name
            
            # Initialize layer group if not exists
            if layer_name not in layer_groups:
                layer_groups[layer_name] = {}
            
            # Add parameter to layer group
            layer_groups[layer_name][param_type] = tensor
        
        return layer_groups
    
    def _create_export_summary(self) -> None:
        """Create a summary file with export statistics"""
        summary_file = self.output_dir / "export_summary.txt"
        
        with open(summary_file, 'w') as f:
            f.write("Checkpoint Export Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total layers exported: {self.stats['total_layers']}\n")
            f.write(f"Total parameters: {self.stats['total_parameters']:,}\n")
            f.write(f"Complex layers: {self.stats['complex_layers']}\n")
            f.write(f"Generated .dat files: {self.stats['exported_files']}\n")
            f.write(f"Output directory: {self.output_dir.absolute()}\n\n")
            
            # Calculate total size
            total_size = sum(f.stat().st_size for f in self.output_dir.rglob('*.dat'))
            f.write(f"Total exported data size: {total_size / (1024*1024):.2f} MB\n")
        
        logger.info(f"Export summary saved to: {summary_file}")


def main():
    """Main function to handle command line interface"""
    parser = argparse.ArgumentParser(
        description="Export PyTorch checkpoint parameters to .dat files"
    )
    parser.add_argument(
        "--checkpoint", "-c",
        required=True,
        help="Path to checkpoint file (.ckpt or .pth)"
    )
    parser.add_argument(
        "--output_dir", "-o",
        default="./dat_files",
        help="Output directory for .dat files (default: ./dat_files)"
    )
    parser.add_argument(
        "--precision", "-p",
        type=int,
        default=6,
        help="Number of decimal places for text output (default: 6)"
    )
    parser.add_argument(
        "--elements_per_row", "-r",
        type=int,
        default=8,
        help="Number of elements per row in text output (default: 8)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize exporter with custom settings
        exporter = CheckpointExporter(args.output_dir)
        exporter.precision = args.precision
        exporter.elements_per_row = args.elements_per_row
        
        # Load checkpoint
        state_dict = exporter.load_checkpoint(args.checkpoint)
        
        # Export to .dat files
        exporter.export_state_dict(state_dict)
        
        # Print final statistics
        stats = exporter.stats
        print("\n" + "=" * 50)
        print("EXPORT COMPLETED SUCCESSFULLY")
        print("=" * 50)
        print(f"Layers exported: {stats['total_layers']}")
        print(f"Parameters exported: {stats['total_parameters']:,}")
        print(f"Complex layers: {stats['complex_layers']}")
        print(f"Generated files: {stats['exported_files']}")
        print(f"Output directory: {Path(args.output_dir).absolute()}")
        
    except Exception as e:
        logger.error(f"Export failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()