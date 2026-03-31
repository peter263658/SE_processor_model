#!/bin/bash

# Script to run quantization tests on BCCTN model

# Set paths
MODEL_CHECKPOINT="/raid/R12K41024/BCCTN_quant/model.ckpt"  # Update this
TEST_DATA_DIR=""/raid/R12K41024/BCCTN/Dataset/noisy_testset""  # Update this
OUTPUT_DIR="./quantization_test_results_$(date +%Y%m%d_%H%M%S)"

# Create output directory
mkdir -p $OUTPUT_DIR

# Test configurations
# Format: "conv_precision/attention_precision"
CONFIGS=(
    "fp32/fp32"      # Baseline
    "int8/fp16"      # INT8 conv, FP16 attention
    "int8/fp32"      # INT8 conv, FP32 attention
    "fp16/fp16"      # FP16 for all
    "int8/fp8"       # INT8 conv, FP8 attention (if supported)
)

echo "Starting Quantization Tests"
echo "=========================="
echo "Model: $MODEL_CHECKPOINT"
echo "Test Data: $TEST_DATA_DIR"
echo "Output: $OUTPUT_DIR"
echo ""

# Run quantization tests
python test_quantization.py \
    --model "$MODEL_CHECKPOINT" \
    --test_data "$TEST_DATA_DIR" \
    --output "$OUTPUT_DIR" \
    --num_samples 100 \
    --configs "${CONFIGS[@]}"

# Check if tests completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "Tests completed successfully!"
    echo "Results saved to: $OUTPUT_DIR"
    
    # Display results summary
    echo ""
    echo "Results Summary:"
    echo "================"
    cat "$OUTPUT_DIR/quantization_test_results.csv" | column -t -s ','
else
    echo "Tests failed! Check the error messages above."
    exit 1
fi