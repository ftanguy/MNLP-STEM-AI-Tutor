import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset
from transformers.trainer_utils import set_seed

def train_and_save(args):
    """
    Main function to run DPO training on a LoRA-adapted model, then merge.
    """
    set_seed(args.seed)

    # Load datasets
    print(f"FIY: Loading training data from: {args.train_data_path}")
    train_dataset = load_dataset("json", data_files=args.train_data_path, split="train")

    print(f"FIY: Loading eval data from: {args.eval_data_path}")
    eval_dataset = load_dataset("json", data_files=args.eval_data_path, split="train")

    if args.max_train_samples and args.max_train_samples < len(train_dataset):
        print(f"FIY: Using {args.max_train_samples} samples for training.")
        train_dataset = train_dataset.shuffle(seed=args.seed).select(range(args.max_train_samples))
        
    if args.max_eval_samples and args.max_eval_samples < len(eval_dataset):
        print(f"FIY: Using {args.max_eval_samples} samples for eval.")
        eval_dataset = eval_dataset.shuffle(seed=args.seed).select(range(args.max_eval_samples))

    print(f"FIY: Training samples: {len(train_dataset)}, Evaluation samples: {len(eval_dataset)}")
    
    # Load Model
    model_dtype = torch.float16
    print(f"FIY: Loading base model '{args.base_model_name}' ")

    # Load model to be trained
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name, device_map="auto", trust_remote_code=True, torch_dtype=model_dtype
    )
    model.config.use_cache = False

    model_ref = AutoModelForCausalLM.from_pretrained(
        args.base_model_name, device_map="auto", trust_remote_code=True, torch_dtype=model_dtype,
    )
    model_ref.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Apply Lora
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    print("FIY: Applying LoRA adapters...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Configure DPO
    dpo_config = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_acc_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        beta=0.1,
        loss_type="sigmoid",
        max_prompt_length=384,
        max_length=1024,
        bf16=(model_dtype == torch.bfloat16),
        fp16=(model_dtype == torch.float16),
        eval_strategy="steps",
        eval_steps=50,
        logging_steps=10,
        remove_unused_columns=False,
        report_to="none",
        seed=args.seed,
    )

    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=model_ref,
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=None, 
    )

    #  Train model
    print("FIY: Starting DPO training...")
    dpo_trainer.train()
    print("FIY: DPO training completed")
    
    #  Merge adapters 
    final_adapter_path = os.path.join(args.output_dir, "final_dpo_checkpoint")
    dpo_trainer.save_model(final_adapter_path)
    print(f"FIY: DPO adapters saved to {final_adapter_path}")

    del model, model_ref, dpo_trainer
    torch.cuda.empty_cache()

    print(f"FIY: Reloading base model '{args.base_model_name}' for merging...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name,
        torch_dtype=model_dtype,
        trust_remote_code=True,
        device_map="cpu",
    )

    print(f"FIY: Loading LoRA adapters from {final_adapter_path}...")
    peft_model = PeftModel.from_pretrained(base_model, final_adapter_path)

    print("FIY: Merging LoRA adapters into the base model...")
    merged_model = peft_model.merge_and_unload()
    
    os.makedirs(args.final_model_path, exist_ok=True)
    
    print(f"FIY: Saving final merged model to {args.final_model_path}...")
    merged_model.save_pretrained(args.final_model_path)
    tokenizer.save_pretrained(args.final_model_path)
    print(f"WORKED: Merged model and tokenizer saved to {args.final_model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a DPO model, merge adapters, and save.")
    parser.add_argument("--base_model_name", type=str, default="Nbenmo/M3_SFT", help="Base SFT model from Hugging Face Hub.")
    parser.add_argument("--train_data_path", type=str, required=True, help="Path to the prepared .jsonl training data.")
    parser.add_argument("--eval_data_path", type=str, required=True, help="Path to the prepared .jsonl evaluation data.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save temporary checkpoints and adapters.")
    parser.add_argument("--final_model_path", type=str, required=True, help="Final directory to save the merged model.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=1, help="Per-device training batch size.")
    parser.add_argument("--grad_acc_steps", type=int, default=16, help="Gradient accumulation steps.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Peak learning rate.")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Maximum number of samples to use for training.")
    parser.add_argument("--max_eval_samples", type=int, default=None, help="Maximum number of samples to use for evaluation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    
    args = parser.parse_args()
    train_and_save(args)