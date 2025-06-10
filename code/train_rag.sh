#!/bin/bash
#SBATCH --job-name=train_rag
#SBATCH --partition=normal                    
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8                 
#SBATCH --gres=gpu:2  
#SBATCH --mem=100G                         
#SBATCH --time=04:00:00                           
#SBATCH --environment=mnlp

set -e  # Exit on any error

# Fix OpenBLAS warning by limiting threads
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OMP_NUM_THREADS=8

# Data paths
TRAIN_DATA="./train_data.json"
VAL_DATA="./validation_data.json"
DOCS_PATH="./unembed_top_100k_chunks.parquet"

# Model and output paths
CONFIG_PATH="./rag_model.yaml"
OUTPUT_DIR="./RAG_checkpoint"
FINAL_MODEL_DIR="./final_rag_model"

# Training parameters
NUM_EPOCHS=4
BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=4
LEARNING_RATE=5e-5
WARMUP_STEPS=500

PRECOMPUTED_EMBEDDINGS="./embeddings_top_100k_chunks_3.npy"  # Leave empty if creating new embeddings

echo "=== RAG Model Training Setup ==="

# Check CUDA availability
echo "Checking CUDA availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}')"

echo "=== Environment Variables Set ==="
echo "OPENBLAS_NUM_THREADS: $OPENBLAS_NUM_THREADS"
echo "MKL_NUM_THREADS: $MKL_NUM_THREADS"
echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"

echo "=== Embedding Configuration ==="
echo "PRECOMPUTED_EMBEDDINGS: $PRECOMPUTED_EMBEDDINGS"
if [[ -f "$PRECOMPUTED_EMBEDDINGS" ]]; then
    echo "✓ Precomputed embeddings file exists"
    echo "File size: $(du -sh "$PRECOMPUTED_EMBEDDINGS" | cut -f1)"
else
    echo "✗ Warning: Precomputed embeddings file not found!"
fi

echo "=== Starting RAG Model Training ==="
echo "Training will begin in 5 seconds... (Ctrl+C to cancel)"
sleep 5

# Log training start time
START_TIME=$(date)
echo "Training started at: $START_TIME"

# Execute the training command
python3 "rag_trainer.py" \
    --config_path "$CONFIG_PATH" \
    --train_data "$TRAIN_DATA" \
    --val_data "$VAL_DATA" \
    --docs_path "$DOCS_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --final_model_dir "$FINAL_MODEL_DIR" \
    --num_epochs "$NUM_EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --learning_rate "$LEARNING_RATE" \
    --warmup_steps "$WARMUP_STEPS" \
    --precomputed_embeddings "$PRECOMPUTED_EMBEDDINGS"

echo "=== Post-Training Information ==="
echo "Checkpoints directory: $OUTPUT_DIR"
echo "Final model directory: $FINAL_MODEL_DIR"


# Check model size
if [[ -d "$FINAL_MODEL_DIR" ]]; then
    MODEL_SIZE=$(du -sh "$FINAL_MODEL_DIR" | cut -f1)
    echo "Final model size: $MODEL_SIZE"
fi

echo "=== Training Complete ==="