from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
import torch
from torch.nn.utils.rnn import pad_sequence
import pyarrow.parquet as pq
import pandas as pd
from datasets import Dataset
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
import sys

if __name__ == "__main__":
    model_name = sys.argv[0]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, device_map="auto"
    )

    ds = Dataset.from_pandas(pd.read_parquet(sys.argv[1]).dropna())

    def format_example(example):
        user_text = example["prompt"]
        prompt = (
            f"<|im_start|>user\n{user_text}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        full_text = prompt + example["response"] + "<|im_end|>"

        prompt_len = len(tokenizer(prompt)["input_ids"])
        return {"prompt": prompt, "full_text": full_text, "prompt_len": prompt_len}

    ds = ds.map(format_example, remove_columns=['response'])

    def tokenize_func(ex):
        tok = tokenizer(ex["full_text"], truncation=True)
        tok["prompt_len"] = ex["prompt_len"]          # keep for masking later
        return tok

    ds = ds.map(tokenize_func, remove_columns=ds.column_names)

    ds = ds.filter(lambda example: example["prompt_len"] <= 512)


    class SFTDataCollator:
        def __call__(self, batch):
            input_ids_list = [torch.tensor(b["input_ids"]) for b in batch]
            input_ids = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id)
            attention_mask = (input_ids != tokenizer.pad_token_id).long()
            labels = input_ids.clone()
            for i, b in enumerate(batch):
                prompt_len = b["prompt_len"]
                labels[i, :prompt_len] = -100
            return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    data_collator = SFTDataCollator()

    training_args = TrainingArguments(
        output_dir="qwen_sft_syn",
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        logging_steps=20,
        save_steps=0,
        report_to=[],
        disable_tqdm=False,
        remove_unused_columns=False,
        learning_rate=1e-5,
        fp16=False,
        bf16=False
    )

    trainer = Trainer(
        model=model,
        train_dataset=ds,
        data_collator=data_collator,
        args=training_args,
    )

    model.gradient_checkpointing_enable()
    trainer.train()

    # Grab only the loss entries
    loss_history = [log["loss"] for log in trainer.state.log_history if "loss" in log]

    plt.figure(figsize=(6,4))
    plt.plot(range(len(loss_history)), loss_history, marker='o', label="Training loss")
    plt.xlabel("Log Step")
    plt.ylabel("Loss")
    plt.title("Loss during SFT training")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Optionally save it
    plt.savefig("loss_curve.png")

    output_dir = "model_output"

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    name = sys.argv[2]

    model.push_to_hub(name, safe_serialization=True)
    tokenizer.push_to_hub(name)
