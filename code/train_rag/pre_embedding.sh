#!/bin/bash
#SBATCH --job-name=rag_embeddings
#SBATCH --partition=normal                  
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8                
#SBATCH --gpus=1                         
#SBATCH --mem=100G                         
#SBATCH --time=04:00:00                 
#SBATCH --environment=mnlp

# Fix OpenBLAS threading issues
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Set CUDA environment
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1


echo "Environment setup complete"
echo "OpenBLAS threads: $OPENBLAS_NUM_THREADS"
echo "CUDA device: $CUDA_VISIBLE_DEVICES"

set -e  # Exit on any error

# =============================================================================
# CONFIGURATION - Modify these variables as needed
# =============================================================================

# Input and output paths
INPUT_FILE="./doc_embeddings/unembed_top_100k_chunks.parquet"
OUTPUT_FILE="./doc_embeddings/embeddings_top_100k_chunks_3.npy"


# Model parameters
BATCH_SIZE=32          # Adjust based on your GPU memory
MAX_LENGTH=256         # Token limit for all-MiniLM-L6-v2
OUTPUT_FORMAT="numpy"  # Options: numpy, parquet, csv



echo "Configuration:"
echo "- Batch size: $BATCH_SIZE"
echo "- Max length: $MAX_LENGTH"
echo "- Output format: $OUTPUT_FORMAT"
echo ""

# Option 1: Process Top 100K chunks only
if [ -f "$INPUT_FILE" ]; then
    echo ""
    echo "Starting embedding generation for top 100K chunks..."
    
    python3 "/doc_embeddings/embed.py" \
        --input "$INPUT_FILE" \
        --output "$OUTPUT_FILE" \
        --batch_size "$BATCH_SIZE" \
        --max_length "$MAX_LENGTH" \
        --format "$OUTPUT_FORMAT" 
    
    echo ""
    echo "Top 100K embeddings completed!"
else
    echo "Warning: Top 100K chunks file not found at $INPUT_FILE"
fi

echo ""


