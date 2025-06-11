import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
import torch
from torch.nn.utils.rnn import pad_sequence
import pyarrow.parquet as pq
import pandas as pd
from datasets import Dataset, load_dataset
import matplotlib.pyplot as plt

# Custom data collator for supervised fine-tuning (SFT)
class SFTDataCollator:
    def __call__(self, batch):
        input_ids_list = [torch.tensor(b["input_ids"]) for b in batch]
        input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id)
        attention_mask = (input_ids != tokenizer.pad_token_id).long()
        labels = input_ids.clone()
        for i, b in enumerate(batch):
            prompt_len = b["prompt_len"]
            # Set prompt tokens in labels to -100 so they are ignored during loss calculation
            labels[i, :prompt_len] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

# Function to format a raw example into prompt-response format using special qwen tokens
def format_example(example):
    user_text = example["prompt"]
    prompt = (
        f"<|im_start|>user\n{user_text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    full_text = prompt + example["response"] + "<|im_end|>"
    prompt_len = len(tokenizer(prompt)["input_ids"])
    return {"prompt": prompt, "full_text": full_text, "prompt_len": prompt_len}

# Tokenization function
def tokenize_func(ex):
    # Tokenize the full example
    tok = tokenizer(ex["full_text"], truncation=True)
    tok["prompt_len"] = ex["prompt_len"]
    return tok

if __name__ == "__main__":

    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True, help="Model name or path")
    parser.add_argument("--use_hf_dataset", action='store_true', help="Specify if loading dataset from Hugging Face Hub")
    parser.add_argument("--dataset_name", type=str, default=None, help="Name of the Hugging Face dataset to load")
    parser.add_argument("--data_path", type=str, default=None, help="Path to local dataset (Parquet)")
    parser.add_argument("--output_dir", required=True, help="Directory to save the model")

    args = parser.parse_args()

    # Load tokenizer and model from the Hugging Face model hub
    print("Loading model")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float32, device_map="auto"
    )
    print("Loaded!")

    # Load dataset from Hugging Face Hub or a local Parquet file
    print("Loading data")
    if args.use_hf_dataset:
        if not args.dataset_name:
            raise ValueError("You must provide --dataset_name when using Hugging Face dataset.")
        ds = load_dataset(args.dataset_name, split="train")
    else:
        if not args.data_path:
            raise ValueError("You must provide --data_path when using a local dataset.")
        df = pd.read_parquet(args.data_path).dropna(subset=["prompt", "response"])
        ds = Dataset.from_pandas(df)
    print("Loaded!")

    # Preprocess dataset: format and tokenize
    ds = ds.filter(lambda example: 
        isinstance(example['prompt'], str) and
        isinstance(example['response'], str) and
        len(example['prompt'].strip()) > 0 and 
        len(example['response'].strip()) > 0
    )
    ds = ds.map(format_example, remove_columns=['response'])  
    ds = ds.map(tokenize_func, remove_columns=ds.column_names)
    # Filter examples with too-long prompts
    ds = ds.filter(lambda example: example["prompt_len"] <= 512)

    # Set up the data collator
    data_collator = SFTDataCollator()

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="qwen_sft",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=2, # Change epending on GPU RAM
        gradient_accumulation_steps=4,
        logging_steps=20,
        save_steps=0,
        report_to=[],  # Disable reporting
        disable_tqdm=False,
        remove_unused_columns=False,
        learning_rate=3e-5,
        fp16=False,  # we want 32bits
        bf16=False
    )

    # Initialize the Trainer object
    trainer = Trainer(
        model=model,
        train_dataset=ds,
        data_collator=data_collator,
        args=training_args,
    )

    # Enable gradient checkpointing to reduce memory usage
    model.gradient_checkpointing_enable()
    # Start training
    print("Starting training")
    trainer.train()

    # Extract and plot loss over time
    loss_history = [log["loss"] for log in trainer.state.log_history if "loss" in log]
    plt.figure(figsize=(6,4))
    plt.plot(range(len(loss_history)), loss_history, marker='o', label="Training loss")
    plt.xlabel("Log Step")
    plt.ylabel("Loss")
    plt.title("Loss during SFT training")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"loss_curve_{args.model_name}.png")

    # Save the trained model and tokenizer locally
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"Saving model to {args.output_dir}. Saving loss curve to loss_curve_{args.model_name}.png")
