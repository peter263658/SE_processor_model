# PTQ Export Tool for BCCTN

Post-Training Quantization (PTQ) export tool for Binaural Complex Convolutional Transformer Networks with mixed precision INT8/BF16 quantization and streaming output generation.

## Features

- **Mixed Precision Quantization**
  - INT8 for Conv/Linear layers (per-channel weights, per-tensor activations)
  - BF16 for Attention layers
  - Symmetric quantization with round-to-nearest-even (RNE)
  - Cross-domain conversion between INT8 ↔ BF16

- **Tensor Export Format**
  - Hexadecimal output (big-endian, uppercase, no 0x prefix)
  - Zigzag ordering for spatial dimensions
  - Complex tensor support (split real/imaginary files)
  - Per-frame streaming dumps

- **Attention Cache Management**
  - Configurable attention window size
  - Causal masking support
  - K/V cache with sliding window

- **Comprehensive Output**
  - Layer weights and activations
  - Quantization parameters (JSON)
  - Tensor shapes
  - Evaluation metrics (MBSTOI)

## Installation

```bash
# Install dependencies
pip install torch torchaudio numpy
```

## Quick Start

### Basic Usage

```bash
python ptq_export.py \
    --model-import DCNN/models/binaural_attention_model.py \
    --model-class BinauralAttentionDCNN \
    --weights path/to/checkpoint.ckpt \
    --calib-dir path/to/calibration/audio \
    --eval-dir path/to/evaluation/audio \
    --output-root artifacts \
    --run-id my_experiment
```

### With Custom Settings

```bash
python ptq_export.py \
    --model-import path/to/model.py \
    --model-class MyModel \
    --weights model.ckpt \
    --calib-dir calibration_data/ \
    --eval-dir test_data/ \
    --output-root results/ \
    --run-id experiment_001 \
    --attn-window 64 \
    --stream-dump per-frame \
    --device cuda \
    --eval-entry eval:compute_mbstoi
```

## Command Line Arguments

### Required Arguments
- `--model-import`: Path to Python file containing model class
- `--model-class`: Name of model class to instantiate
- `--weights`: Path to FP32 checkpoint file
- `--calib-dir`: Directory containing calibration audio files (.wav)
- `--eval-dir`: Directory containing evaluation audio files (.wav)

### Optional Arguments
- `--output-root`: Output root directory (default: `artifacts`)
- `--run-id`: Run identifier (default: timestamp)
- `--stream-dump`: Streaming mode (`per-frame` or `full`, default: `per-frame`)
- `--attn-window`: Attention window size (int or `none`, default: `32`)
- `--device`: Compute device (`cuda`, `cpu`, `auto`, default: `auto`)
- `--eval-entry`: Evaluation function (`module:function`, default: `eval:compute_mbstoi`)
- `--dump-complex-split`: Split complex tensors (default: `true`)
- `--seed`: Random seed (default: `42`)

## Output Directory Structure

```
artifacts/
└── {run_id}/
    ├── report.json              # Execution report with metrics
    ├── weights/                 # Quantized weight tensors
    │   ├── 00_conv_encoder_w.hex
    │   └── 01_attention_w.hex
    ├── activations/             # Activation tensors (full dump)
    │   ├── 00_conv_encoder_in.hex
    │   └── 00_conv_encoder_out.hex
    ├── streaming/               # Per-frame streaming dumps
    │   ├── 00_conv_encoder_in_f000001.hex
    │   └── 00_conv_encoder_out_f000001.hex
    ├── shapes/                  # Tensor shape files
    │   ├── 00_conv_encoder_w.shape.txt
    │   └── 00_conv_encoder_in.shape.txt
    └── quant_params/           # Quantization parameters
        ├── 00_conv_encoder_w.json
        └── 00_conv_encoder_act.json
```

## File Formats

### Hex Files
- **Format**: Big-endian, uppercase hex, one value per line
- **INT8**: 2 digits (e.g., `7F`, `80`, `FF`)
- **BF16**: 4 digits (e.g., `3F80` for 1.0)
- **Complex**: Separate `*r.hex` and `*i.hex` files

### Shape Files
- **Format**: Comma-separated dimensions (e.g., `1,128,64,64`)

### JSON Files
- **Quantization params**: Scale values and metadata
- **Report**: Execution summary with metrics

### Zigzag Ordering

Spatial dimensions (H,W) or (kH,kW) use zigzag traversal:
```
3×3 example: (0,0)→(0,1)→(0,2)→(1,2)→(1,1)→(1,0)→(2,0)→(2,1)→(2,2)
```

## Model Integration

### Model Requirements
Your model file should export the model class:

```python
# my_model.py
import torch.nn as nn

class MyBinauralModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Model definition
    
    def forward(self, x):
        # x: [batch, channels, samples]
        return enhanced_audio
```

### Evaluation Function
Custom evaluation function format:

```python
# eval.py
def compute_mbstoi(model, eval_dir, device):
    """
    Evaluate model and return metrics.
    
    Args:
        model: Quantized model
        eval_dir: Directory with evaluation files
        device: Compute device
    
    Returns:
        dict: {'mbstoi': float, ...}
    """
    # Your evaluation logic
    return {'mbstoi': 0.85}
```

## Quantization Details

### INT8 Quantization
- **Weights**: Per-channel symmetric, `scale = max_abs / 127`
- **Activations**: Per-tensor symmetric, `scale = max_abs / 127`
- **Formula**: `x_int8 = clamp(round_even(x_fp32 / scale), -128, 127)`

### BF16 Quantization
- **Format**: IEEE 754 bfloat16 (1 sign + 8 exp + 7 mantissa bits)
- **Fallback**: FP32 simulation if BF16 not supported
- **Usage**: Attention layers only

### Cross-Domain Conversion
- **INT8 → BF16**: `x_bf16 = to_bf16(dequantize_int8(x_int8, scale))`
- **BF16 → INT8**: `x_int8 = quantize_to_int8(to_fp32(x_bf16), scale)`

## Examples

### Basic BCCTN Evaluation

```bash
# Evaluate BCCTN model
python ptq_export.py \
    --model-import DCNN/models/binaural_attention_model.py \
    --model-class BinauralAttentionDCNN \
    --weights checkpoints/best_model.ckpt \
    --calib-dir datasets/vctk_calib/ \
    --eval-dir datasets/vctk_test/ \
    --run-id bcctn_int8_bf16
```

### Streaming with Custom Window

```bash
# Streaming mode with 64-frame attention window
python ptq_export.py \
    --model-import my_model.py \
    --model-class MyModel \
    --weights model.ckpt \
    --calib-dir calib/ \
    --eval-dir test/ \
    --attn-window 64 \
    --stream-dump per-frame \
    --run-id streaming_test
```

### CPU-only Quantization

```bash
# Force CPU processing
python ptq_export.py \
    --model-import model.py \
    --model-class Model \
    --weights model.ckpt \
    --calib-dir calib/ \
    --eval-dir test/ \
    --device cpu \
    --run-id cpu_quant
```

## Output Analysis

### Report File (report.json)
```json
{
  "run_id": "experiment_001",
  "env": {
    "torch": "2.0.0",
    "cuda": "11.8",
    "device": "cuda:0"
  },
  "layer_order": ["00_conv_encoder", "01_attention", ...],
  "metrics": {
    "mbstoi_fp32": 0.93,
    "mbstoi_ptq": 0.91,
    "delta": -0.02
  }
}
```

### Quantization Parameters
```json
{
  "name": "00_conv_encoder",
  "type": "conv",
  "bitwidth": 8,
  "scheme": "symmetric",
  "weight_axis": "out_ch",
  "scale": [0.0234, 0.0187, ...]
}
```

## Troubleshooting

### Common Issues

1. **Model Loading Error**
   ```
   ✗ Error loading model: Class MyModel not found in module
   ```
   - Ensure model class is correctly exported
   - Check import path and class name

2. **CUDA Out of Memory**
   ```
   RuntimeError: CUDA out of memory
   ```
   - Use `--device cpu` or reduce calibration dataset size

3. **Invalid Hex Format**
   ```
   Invalid hex character in line 42: ZZ
   ```
   - Check tensor conversion and file writing logic

4. **MBSTOI Calculation Failed**
   ```
   ⚠️ Evaluation failed with eval:compute_mbstoi
   ```
   - Verify evaluation function exists and is accessible
   - Check evaluation data format

### Debug Tips

- Use smaller calibration/evaluation datasets for testing
- Check intermediate outputs in `activations/` directory
- Validate hex files with provided validation functions
- Monitor GPU memory usage during processing

## License

This tool is designed for research and development purposes. Please ensure compliance with your model's license terms.
