#!/bin/bash
set -e

echo "=================================================="
echo "Starting DPO Model Training"
echo "=================================================="

# --- Configuration ---
REQUIREMENTS_FILE="./train_dpo/requirements.txt"

BASE_SFT_MODEL="Nbenmo/M3_SFT"
EPOCHS=1
BATCH_SIZE=1
GRAD_ACC_STEPS=16
LEARNING_RATE=2e-5

DATA_DIR="train_dpo/data/dpo"
MODEL_CHECKPOINTS_DIR="train_dpo/models/dpo_temp_checkpoints"
FINAL_MODEL_DIR="train_dpo/models/dpo_final_model"

TRAIN_DATA_FILE="$DATA_DIR/stem_dpo_train.jsonl"
EVAL_DATA_FILE="$DATA_DIR/stem_dpo_eval.jsonl"

MAX_PREP_TRAIN_SAMPLES=157675
MAX_PREP_EVAL_SAMPLES=2985
MAX_TRAIN_SAMPLES=30000
MAX_EVAL_SAMPLES=1500


# --- Step 1: Install Dependencies ---
echo "--> STEP 1: Installing dependencies from $REQUIREMENTS_FILE..."
pip install -r "$REQUIREMENTS_FILE"
echo "Dependencies are installed."

# --- Step 2: Prepare Datasets ---
echo -e "\n--> STEP 2: Preparing datasets..."
mkdir -p "$DATA_DIR"

if [ -f "$TRAIN_DATA_FILE" ]; then
    echo "Training data already exists. Skipping."
else
    echo "Creating filtered training data..."
    python ./train_dpo/prepare_data.py \
        --dataset_name "argilla/ultrafeedback-multi-binarized-preferences-cleaned" \
        --split_name "train" \
        --output_path "$TRAIN_DATA_FILE" \
        --max_samples "$MAX_PREP_TRAIN_SAMPLES"
fi

if [ -f "$EVAL_DATA_FILE" ]; then
    echo "Evaluation data already exists. Skipping."
else
    echo "Creating filtered evaluation data..."
    python ./train_dpo/prepare_data.py \
        --dataset_name "allenai/reward-bench" \
        --split_name "filtered" \
        --output_path "$EVAL_DATA_FILE" \
        --max_samples "$MAX_PREP_EVAL_SAMPLES"
fi
echo "Dataset preparation complete."

# --- Step 3: Run DPO Training ---
echo -e "\n--> STEP 3: Run DPO Training..."
mkdir -p "$MODEL_CHECKPOINTS_DIR"
mkdir -p "$FINAL_MODEL_DIR"

python ./train_dpo/train.py \
    --base_model_name "$BASE_SFT_MODEL" \
    --train_data_path "$TRAIN_DATA_FILE" \
    --eval_data_path "$EVAL_DATA_FILE" \
    --output_dir "$MODEL_CHECKPOINTS_DIR" \
    --final_model_path "$FINAL_MODEL_DIR" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --grad_acc_steps "$GRAD_ACC_STEPS" \
    --learning_rate "$LEARNING_RATE" \
    --max_train_samples "$MAX_TRAIN_SAMPLES" \
    --max_eval_samples "$MAX_EVAL_SAMPLES"

echo "=================================================="
echo "DONE"
echo "=================================================="