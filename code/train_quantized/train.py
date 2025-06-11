# File: train_quantized/train.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import argparse
import os

def main():
    """
    This script loads a pre-trained model, applies a specific BitsAndBytes
    quantization configuration, and saves the resulting quantized model
    and tokenizer to a local directory.
    """
    # --- 1. Setup Argument Parser ---
    parser = argparse.ArgumentParser(description="Quantize a model and save it locally.")
    parser.add_argument(
        "--original_model_id",
        type=str,
        required=True,
        help="The Hugging Face Hub ID of the base model to quantize."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The local directory where the quantized model will be saved."
    )
    args = parser.parse_args()

    # --- 2. Define the Quantization Configuration ---
    # This is the best configuration you found from your experiments.
    # It's hardcoded here for reproducibility.
    final_bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    print("--- Using Best Quantization Config: 4-bit NF4 with Double Quantization ---")


    # --- 3. Main Quantization and Save Logic ---
    try:
        # Load the tokenizer from the original model
        print(f"\n[Step 1/3] Loading tokenizer from: {args.original_model_id}")
        tokenizer = AutoTokenizer.from_pretrained(args.original_model_id, use_fast=False)
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        print("Tokenizer loaded successfully.")

        # Load the original model and apply the quantization config on the fly
        print(f"\n[Step 2/3] Loading and quantizing model: {args.original_model_id}")
        model = AutoModelForCausalLM.from_pretrained(
            args.original_model_id,
            quantization_config=final_bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        )
        print("Model loaded and quantized successfully.")

        # Save the final quantized model and tokenizer to the specified local directory
        print(f"\n[Step 3/3] Saving final model and tokenizer to: {args.output_dir}")
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f"\nSUCCESS: Final model and tokenizer saved to {args.output_dir}")

    except Exception as e:
        print(f"An error occurred during the process: {e}")

if __name__ == "__main__":
    main()