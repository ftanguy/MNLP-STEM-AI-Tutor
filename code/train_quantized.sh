#!/bin/bash

# =================================================================
# Bash Script to "Train" (Quantize) the Final Quantized Model
#
# This script executes the Python script that performs Post-Training
# Quantization on a base model and saves the result to a local
# directory.
#
# Usage:
# 1. Make sure you have the required libraries installed:
#    pip install torch transformers bitsandbytes accelerate
#
# 2. Run the script:
#    ./train_quantized.sh
# =================================================================

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
# The original, unquantized model from your teammate.
ORIGINAL_MODEL="Nbenmo/MNLP_M3_mcqa_model"

# The local directory where the final quantized model will be saved.
# The script will create this folder.
OUTPUT_DIR="./train_quantized/quantized_model_output"

# --- Script Execution ---
echo "Starting quantization process..."
echo "Original Model: $ORIGINAL_MODEL"
echo "Local Output Directory: $OUTPUT_DIR"

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run the Python script with the configured arguments
python train_quantized/train.py \
    --original_model_id "$ORIGINAL_MODEL" \
    --output_dir "$OUTPUT_DIR"

echo "Script finished successfully. Quantized model saved in $OUTPUT_DIR"
