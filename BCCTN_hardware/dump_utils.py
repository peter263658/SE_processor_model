"""
Dump utilities for PTQ Export Tool.
Handles tensor serialization to hex format with zigzag ordering and complex number support.
"""

import json
from pathlib import Path
from typing import List, Tuple, Union, Optional, Any, Dict

import torch
import numpy as np


def zigzag_indices_2d(height: int, width: int) -> List[Tuple[int, int]]:
    """
    Generate zigzag indices for 2D tensor traversal.
    
    Pattern: (0,0),(0,1),(0,2),(1,2),(1,1),(1,0),(2,0),(2,1),(2,2) for 3x3
    
    Args:
        height: Height dimension
        width: Width dimension
    
    Returns:
        List of (h, w) coordinate tuples in zigzag order
    """
    indices = []
    
    # Traverse diagonals
    for diagonal in range(height + width - 1):
        coords = []
        
        # Collect coordinates on this diagonal
        for h in range(height):
            w = diagonal - h
            if 0 <= w < width:
                coords.append((h, w))
        
        # Reverse every other diagonal to create zigzag
        if diagonal % 2 == 1:
            coords.reverse()
        
        indices.extend(coords)
    
    return indices


def flatten_tensor_zigzag(tensor: torch.Tensor) -> torch.Tensor:
    """
    Flatten tensor using zigzag ordering for spatial dimensions.
    
    For Conv weights [out,in,kH,kW]: out→in→Zigzag(kH,kW)
    For Features [N,C,H,W] (N=1): C→Zigzag(H,W)
    
    Args:
        tensor: Input tensor
    
    Returns:
        Flattened tensor with zigzag ordering
    """
    shape = tensor.shape
    
    if len(shape) == 4:  # [N,C,H,W] or [out,in,kH,kW]
        if shape[0] == 1:  # Feature tensor [1,C,H,W]
            N, C, H, W = shape
            zigzag_idx = zigzag_indices_2d(H, W)
            
            # Reorder spatial dimensions
            flattened = []
            for c in range(C):
                for h, w in zigzag_idx:
                    flattened.append(tensor[0, c, h, w])
            
            return torch.stack(flattened)
        
        else:  # Conv weight [out,in,kH,kW]
            out_ch, in_ch, kH, kW = shape
            zigzag_idx = zigzag_indices_2d(kH, kW)
            
            flattened = []
            for o in range(out_ch):
                for i in range(in_ch):
                    for h, w in zigzag_idx:
                        flattened.append(tensor[o, i, h, w])
            
            return torch.stack(flattened)
    
    elif len(shape) == 3:  # [C,H,W]
        C, H, W = shape
        zigzag_idx = zigzag_indices_2d(H, W)
        
        flattened = []
        for c in range(C):
            for h, w in zigzag_idx:
                flattened.append(tensor[c, h, w])
        
        return torch.stack(flattened)
    
    elif len(shape) == 2:  # [H,W]
        H, W = shape
        zigzag_idx = zigzag_indices_2d(H, W)
        
        flattened = []
        for h, w in zigzag_idx:
            flattened.append(tensor[h, w])
        
        return torch.stack(flattened)
    
    else:
        # For other shapes, use regular flattening
        return tensor.flatten()


def tensor_to_hex_int8(tensor: torch.Tensor) -> List[str]:
    """
    Convert INT8 tensor to hexadecimal strings.
    
    Args:
        tensor: INT8 tensor
    
    Returns:
        List of hex strings (2 digits, uppercase, no 0x prefix)
    """
    # Ensure tensor is on CPU and int8
    tensor_cpu = tensor.cpu().to(torch.int8)
    
    # Convert to numpy for easier manipulation
    numpy_arr = tensor_cpu.numpy()
    
    # Convert each value to hex
    hex_values = []
    for val in numpy_arr.flat:
        # Handle two's complement: -128 to 127 -> 0x80 to 0x7F, 0xFF
        if val < 0:
            # Two's complement conversion
            hex_val = format(256 + val, '02X')
        else:
            hex_val = format(val, '02X')
        
        hex_values.append(hex_val)
    
    return hex_values


def tensor_to_hex_bf16(tensor: torch.Tensor) -> List[str]:
    """
    Convert BF16 tensor to hexadecimal strings.
    
    Args:
        tensor: BF16 tensor (or FP32 tensor to be converted)
    
    Returns:
        List of hex strings (4 digits, uppercase, big-endian)
    """
    # Convert to BF16 if needed
    if tensor.dtype != torch.bfloat16:
        if hasattr(torch, 'bfloat16'):
            tensor = tensor.to(torch.bfloat16)
        else:
            # Simulate BF16 by truncating FP32 mantissa
            tensor = simulate_bf16_precision(tensor)
    
    # Get bit patterns
    if tensor.dtype == torch.bfloat16:
        # Extract raw bits as uint16
        uint16_tensor = tensor.view(torch.uint16)
        numpy_arr = uint16_tensor.cpu().numpy()
    else:
        # Fallback for simulated BF16
        numpy_arr = tensor.cpu().numpy()
        # Convert float32 to simulated BF16 bit pattern
        numpy_arr = np.array([float32_to_bf16_bits(x) for x in numpy_arr.flat], dtype=np.uint16)
    
    # Convert to big-endian hex strings
    hex_values = []
    for val in numpy_arr.flat:
        # Big-endian format (most significant byte first)
        hex_val = format(val, '04X')
        hex_values.append(hex_val)
    
    return hex_values


def simulate_bf16_precision(tensor: torch.Tensor) -> torch.Tensor:
    """Simulate BF16 precision by truncating mantissa bits."""
    # BF16: 1 sign + 8 exponent + 7 mantissa bits
    # FP32: 1 sign + 8 exponent + 23 mantissa bits
    # Truncate 16 mantissa bits (23-7=16)
    
    numpy_arr = tensor.cpu().numpy().astype(np.float32)
    
    # Bit manipulation to simulate BF16 truncation
    uint32_view = numpy_arr.view(np.uint32)
    
    # Truncate lower 16 bits (mantissa truncation)
    bf16_simulated = uint32_view & 0xFFFF0000
    
    # Convert back to float32
    result = bf16_simulated.view(np.float32)
    
    return torch.from_numpy(result)


def float32_to_bf16_bits(value: float) -> int:
    """Convert float32 value to BF16 bit pattern."""
    # Get FP32 bit representation
    fp32_bits = np.array([value], dtype=np.float32).view(np.uint32)[0]
    
    # BF16 is the upper 16 bits of FP32
    bf16_bits = (fp32_bits >> 16) & 0xFFFF
    
    return int(bf16_bits)


def write_tensor_hex(file_path: Path, tensor: torch.Tensor, is_complex: bool = False) -> None:
    """
    Write tensor to hex file with proper formatting.
    
    Args:
        file_path: Output file path
        tensor: Tensor to write
        is_complex: Whether tensor represents complex numbers
    """
    # Flatten with zigzag ordering
    flat_tensor = flatten_tensor_zigzag(tensor)
    
    # Determine tensor type and convert to hex
    if tensor.dtype == torch.int8:
        hex_values = tensor_to_hex_int8(flat_tensor)
    elif tensor.dtype in [torch.bfloat16, torch.float32, torch.float16]:
        hex_values = tensor_to_hex_bf16(flat_tensor)
    else:
        raise ValueError(f"Unsupported tensor dtype: {tensor.dtype}")
    
    # Write to file
    with open(file_path, 'w') as f:
        for hex_val in hex_values:
            f.write(hex_val + '\n')


def write_complex_tensor_hex(file_path_base: Path, complex_tensor: torch.Tensor) -> None:
    """
    Write complex tensor to separate real and imaginary hex files.
    
    Args:
        file_path_base: Base file path (without extension)
        complex_tensor: Complex tensor
    """
    real_part = complex_tensor.real
    imag_part = complex_tensor.imag
    
    # Generate file names
    real_file = file_path_base.with_suffix('.real.hex')
    imag_file = file_path_base.with_suffix('.imag.hex')
    
    # Alternative naming convention
    if 'r.hex' in str(file_path_base) or 'i.hex' in str(file_path_base):
        base_name = str(file_path_base).replace('r.hex', '').replace('i.hex', '')
        real_file = Path(f"{base_name}r.hex")
        imag_file = Path(f"{base_name}i.hex")
    
    # Write real and imaginary parts
    write_tensor_hex(real_file, real_part)
    write_tensor_hex(imag_file, imag_part)


def write_shape_file(file_path: Path, shape: Tuple[int, ...]) -> None:
    """
    Write tensor shape to .shape.txt file.
    
    Args:
        file_path: Output file path
        shape: Tensor shape tuple
    """
    shape_str = ','.join(map(str, shape))
    
    with open(file_path, 'w') as f:
        f.write(shape_str)


def write_json_file(file_path: Path, data: Dict[str, Any]) -> None:
    """
    Write data to JSON file with proper formatting.
    
    Args:
        file_path: Output file path
        data: Data to write
    """
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2, default=str)


class DumpManager:
    """Manages all file dumping operations."""
    
    def __init__(self, output_root: str, run_id: str, dump_complex_split: bool = True):
        self.output_root = Path(output_root)
        self.run_id = run_id
        self.dump_complex_split = dump_complex_split
        self.output_dir = self.output_root / run_id
        
        # Create subdirectories
        self.weights_dir = self.output_dir / 'weights'
        self.activations_dir = self.output_dir / 'activations'
        self.shapes_dir = self.output_dir / 'shapes'
        self.quant_params_dir = self.output_dir / 'quant_params'
        self.streaming_dir = self.output_dir / 'streaming'
    
    def setup_directories(self) -> None:
        """Create all necessary output directories."""
        directories = [
            self.output_dir,
            self.weights_dir,
            self.activations_dir,
            self.shapes_dir,
            self.quant_params_dir,
            self.streaming_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def dump_weights(self, layer_name: str, weight_tensor: torch.Tensor, 
                    scales: Optional[List[float]] = None) -> None:
        """
        Dump weight tensor to hex file.
        
        Args:
            layer_name: Name of the layer
            weight_tensor: Weight tensor
            scales: Quantization scales (optional)
        """
        # Check if tensor is complex
        is_complex = weight_tensor.dtype in [torch.complex64, torch.complex128]
        
        if is_complex and self.dump_complex_split:
            # Split complex tensor
            base_path = self.weights_dir / f"{layer_name}_w"
            write_complex_tensor_hex(base_path, weight_tensor)
            
            # Write shapes for real and imaginary parts
            real_shape_file = self.shapes_dir / f"{layer_name}_w_real.shape.txt"
            imag_shape_file = self.shapes_dir / f"{layer_name}_w_imag.shape.txt"
            write_shape_file(real_shape_file, weight_tensor.real.shape)
            write_shape_file(imag_shape_file, weight_tensor.imag.shape)
        else:
            # Single tensor
            weight_file = self.weights_dir / f"{layer_name}_w.hex"
            write_tensor_hex(weight_file, weight_tensor, is_complex)
            
            # Write shape
            shape_file = self.shapes_dir / f"{layer_name}_w.shape.txt"
            write_shape_file(shape_file, weight_tensor.shape)
        
        # Write quantization parameters if provided
        if scales:
            quant_file = self.quant_params_dir / f"{layer_name}_w.json"
            quant_data = {
                'name': layer_name,
                'type': 'weight',
                'bitwidth': 8 if weight_tensor.dtype == torch.int8 else 16,
                'scheme': 'symmetric',
                'weight_axis': 'out_ch',
                'scale': scales
            }
            write_json_file(quant_file, quant_data)
    
    def dump_activations(self, layer_name: str, activation_tensor: torch.Tensor,
                        frame_idx: Optional[int] = None, is_input: bool = True) -> None:
        """
        Dump activation tensor to hex file.
        
        Args:
            layer_name: Name of the layer
            activation_tensor: Activation tensor
            frame_idx: Frame index for streaming (None for single dump)
            is_input: Whether this is input or output activation
        """
        # Determine file name
        io_suffix = 'in' if is_input else 'out'
        
        if frame_idx is not None:
            # Streaming mode
            base_name = f"{layer_name}_{io_suffix}_f{frame_idx:06d}"
            output_dir = self.streaming_dir
        else:
            # Single dump mode
            base_name = f"{layer_name}_{io_suffix}"
            output_dir = self.activations_dir
        
        # Check if tensor is complex
        is_complex = activation_tensor.dtype in [torch.complex64, torch.complex128]
        
        if is_complex and self.dump_complex_split:
            # Split complex tensor
            real_file = output_dir / f"{base_name}r.hex"
            imag_file = output_dir / f"{base_name}i.hex"
            
            write_tensor_hex(real_file, activation_tensor.real)
            write_tensor_hex(imag_file, activation_tensor.imag)
            
            # Write shapes
            real_shape_file = self.shapes_dir / f"{base_name}r.shape.txt"
            imag_shape_file = self.shapes_dir / f"{base_name}i.shape.txt"
            write_shape_file(real_shape_file, activation_tensor.real.shape)
            write_shape_file(imag_shape_file, activation_tensor.imag.shape)
        else:
            # Single tensor
            act_file = output_dir / f"{base_name}.hex"
            write_tensor_hex(act_file, activation_tensor, is_complex)
            
            # Write shape
            shape_file = self.shapes_dir / f"{base_name}.shape.txt"
            write_shape_file(shape_file, activation_tensor.shape)
    
    def dump_attention_outputs(self, layer_name: str, attention_map: torch.Tensor,
                             qkv_outputs: Dict[str, torch.Tensor], frame_idx: Optional[int] = None) -> None:
        """
        Dump attention-specific outputs.
        
        Args:
            layer_name: Name of the attention layer
            attention_map: Attention weights after softmax
            qkv_outputs: Dictionary containing Q, K, V projection outputs
            frame_idx: Frame index for streaming
        """
        # Dump attention map
        self.dump_activations(layer_name + '_attn_map', attention_map, frame_idx, is_input=False)
        
        # Dump Q, K, V projections
        for projection_name, tensor in qkv_outputs.items():
            full_name = f"{layer_name}_{projection_name}_proj"
            self.dump_activations(full_name, tensor, frame_idx, is_input=False)
    
    def get_output_paths(self) -> Dict[str, Path]:
        """Get all output directory paths."""
        return {
            'root': self.output_dir,
            'weights': self.weights_dir,
            'activations': self.activations_dir,
            'shapes': self.shapes_dir,
            'quant_params': self.quant_params_dir,
            'streaming': self.streaming_dir
        }


# Utility functions for testing and validation
def validate_hex_file(file_path: Path, expected_count: Optional[int] = None) -> bool:
    """
    Validate hex file format.
    
    Args:
        file_path: Path to hex file
        expected_count: Expected number of hex values (optional)
    
    Returns:
        True if file is valid
    """
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Check each line
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            
            # Check hex format
            if not all(c in '0123456789ABCDEF' for c in line):
                print(f"Invalid hex character in line {i+1}: {line}")
                return False
            
            # Check length (2 for INT8, 4 for BF16)
            if len(line) not in [2, 4]:
                print(f"Invalid hex length in line {i+1}: {line} (length: {len(line)})")
                return False
        
        # Check expected count
        valid_lines = [line.strip() for line in lines if line.strip()]
        if expected_count is not None and len(valid_lines) != expected_count:
            print(f"Expected {expected_count} values, found {len(valid_lines)}")
            return False
        
        return True
        
    except Exception as e:
        print(f"Error validating {file_path}: {e}")
        return False


def hex_file_to_tensor(file_path: Path, shape: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    """
    Load tensor from hex file.
    
    Args:
        file_path: Path to hex file
        shape: Target tensor shape
        dtype: Target tensor dtype
    
    Returns:
        Loaded tensor
    """
    with open(file_path, 'r') as f:
        hex_values = [line.strip() for line in f.readlines() if line.strip()]
    
    # Convert hex to numbers
    if dtype == torch.int8:
        values = []
        for hex_val in hex_values:
            int_val = int(hex_val, 16)
            # Convert from unsigned to signed int8
            if int_val > 127:
                int_val = int_val - 256
            values.append(int_val)
        
        tensor = torch.tensor(values, dtype=torch.int8)
    
    elif dtype in [torch.bfloat16, torch.float32]:
        values = []
        for hex_val in hex_values:
            uint16_val = int(hex_val, 16)
            # Convert BF16 bits to float
            fp32_bits = uint16_val << 16
            float_val = np.array([fp32_bits], dtype=np.uint32).view(np.float32)[0]
            values.append(float_val)
        
        tensor = torch.tensor(values, dtype=dtype)
    
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    
    # Reshape to target shape
    return tensor.reshape(shape)
