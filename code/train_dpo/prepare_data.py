import json
import re
import argparse
from datasets import load_dataset
import random
import os
import sys

STEM_KEYWORDS = [
    "math", "mathematics", "mathematical", "algebra", "calculus", "geometry", "statistics", "probability", "calculate", "matrix", "decimal", "sum",
    "science", "scientific", "physics", "physicist", "quantum", "relativity", "thermodynamics", "optics", "acoustic",
    "chemistry", "chemical", "organic", "inorganic", "biochemistry", "molecule", "element", "reaction",
    "biology", "biological", "genetic", "dna", "rna", "protein", "cell", "evolution", "ecology", "botany", "zoology", "photosynthesis", "medical",
    "engineering", "engineer", "mechanical", "electrical", "civil", "aerospace", "software", "biomedical", "circuit", "engine",
    "technology", "technological",
    "computer science", "cs", "code", "coding", "program", "programming", "algorithm", "algorithmic",
    "software development", "web development", "app development", "machine learning", "artificial intelligence", "ai", "ml",
    "data", "data science", "data analysis", "database", "dataset",
    "python", "java", "c++", "javascript", "sql", "ruby", "swift", "kotlin", "php", "rust", "golang",
    "astronomy", "astrophysics", "cosmology", "geology", "earth science", "meteorology",
    "robotics", "nanotechnology", "cryptography", "cybersecurity",
    "equation", "variable", "function", "theorem", "proof", "hypothesis", "experiment", "laboratory", "research", "solve", "puzzle"
]

def extract_content(example):
    if "chosen" in example and isinstance(example["chosen"], list):
        prompt = example.get("prompt")
        chosen_text = next((msg['content'] for msg in example['chosen'] if msg['role'] == 'assistant'), None)
        rejected_text = next((msg['content'] for msg in example['rejected'] if msg['role'] == 'assistant'), None)
        return prompt, chosen_text, rejected_text
    
    elif "chosen" in example and isinstance(example["chosen"], str):
        prompt = example.get("prompt")
        chosen_text = example.get("chosen")
        rejected_text = example.get("rejected")
        return prompt, chosen_text, rejected_text

    return None, None, None

def filter_and_save_data(args):
    print(f"FIY: Loding dataset '{args.dataset_name}' with split '{args.split_name}'.")
    dataset = load_dataset(args.dataset_name, split=args.split_name)
    print(f"FIY: Loaded {len(dataset)} total sample.")

    if args.max_samples and args.max_samples < len(dataset):
        print(f"FIY: Selecting a random subset of {args.max_samples} samples for processing.")
        dataset = dataset.shuffle(seed=args.seed).select(range(args.max_samples))

    filtered_examples = []
    
    print("FIY: Filtering examples for STEM content...")
    for example in dataset:
        prompt_text, chosen_text, rejected_text = extract_content(example)
        
        if not all([isinstance(prompt_text, str), isinstance(chosen_text, str), isinstance(rejected_text, str)]):
            continue
        
        if not prompt_text.strip() or chosen_text == rejected_text:
            continue

        prompt_lower = prompt_text.lower()
        is_stem = any(re.search(r"\b" + re.escape(kw.lower()) + r"\b", prompt_lower) for kw in STEM_KEYWORDS)

        if is_stem:
            filtered_examples.append({
                "prompt": prompt_text,
                "chosen": chosen_text,
                "rejected": rejected_text
            })

    print(f"FIY: Found {len(filtered_examples)} valid STEM examples.")


    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    print(f"FIY: Saving {len(filtered_examples)} examples to {args.output_path}...")
    with open(args.output_path, 'w', encoding='utf-8') as f_out:
        for example in filtered_examples:
            f_out.write(json.dumps(example) + '\n')
    
    print(f"Worked: Saved filtered dataset to {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter a Hugging Face dataset for STEM content.")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the source dataset on Hugging Face Hub.")
    parser.add_argument("--split_name", type=str, required=True, help="Dataset split to process.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output .jsonl file.")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to process from the source dataset for testing.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling.")
    
    args = parser.parse_args()
    filter_and_save_data(args)