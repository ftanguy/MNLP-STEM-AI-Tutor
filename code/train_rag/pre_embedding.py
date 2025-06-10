import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import json
import argparse
from tqdm import tqdm
import time
import gc
import os


def mean_pooling(model_output, attention_mask):
    """Mean Pooling - Take attention mask into account for correct averaging"""
    token_embeddings = model_output[0]  # First element contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def setup_environment():
    """Setup environment variables and check GPU availability"""
    # Set environment variables for better stability
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    # Check GPU availability
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Clear GPU cache
        torch.cuda.empty_cache()
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    return device


def load_and_preprocess_data(parquet_path):
    """Load parquet file and extract text content"""
    print(f"Loading data from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    
    # Extract text content (handle JSON format if needed)
    if isinstance(df.iloc[0]['text'], str):
        try:
            # Try to parse as JSON
            df['parsed'] = df['text'].apply(
                lambda x: json.loads(x) if x.startswith('{') else {'text': x}
            )
            texts = df['parsed'].apply(lambda x: x.get('text', '')).tolist()
        except:
            # If not JSON, treat as plain text
            texts = df['text'].tolist()
    else:
        texts = df['text'].tolist()
    
    print(f"Loaded {len(texts)} documents")
    return texts


def generate_embeddings_batch(texts, model, tokenizer, batch_size=32, max_length=256, device='cuda'):
    """Generate embeddings for a list of texts using batching with improved memory management"""
    model.eval()
    all_embeddings = []
    
    print(f"Generating embeddings with batch size {batch_size}...")
    print(f"Using device: {device}")
    print(f"Max sequence length: {max_length}")
    
    # Reduce batch size if GPU memory is limited
    if device.type == 'cuda':
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if gpu_memory_gb < 8:  # If less than 8GB GPU memory
            batch_size = min(batch_size, 16)
            print(f"Reduced batch size to {batch_size} due to limited GPU memory")
    
    # Process in batches
    for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
        batch_texts = texts[i:i + batch_size]
        
        try:
            # Tokenize batch
            encoded_input = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            ).to(device)
            
            # Generate embeddings
            with torch.no_grad():
                model_output = model(**encoded_input)
            
            # Perform mean pooling
            batch_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
            
            # Normalize embeddings
            batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
            
            # Move to CPU and convert to numpy
            batch_embeddings = batch_embeddings.cpu().numpy()
            all_embeddings.append(batch_embeddings)
            
            # Clear GPU memory after each batch
            del encoded_input, model_output, batch_embeddings
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            
            # Force garbage collection every 100 batches
            if i % (100 * batch_size) == 0:
                gc.collect()
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"GPU out of memory at batch {i//batch_size}. Trying with smaller batch...")
                # Clear memory and retry with smaller batch
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                gc.collect()
                
                # Process this batch one by one
                for single_text in batch_texts:
                    encoded_input = tokenizer(
                        [single_text],
                        padding=True,
                        truncation=True,
                        max_length=max_length,
                        return_tensors='pt'
                    ).to(device)
                    
                    with torch.no_grad():
                        model_output = model(**encoded_input)
                    
                    single_embedding = mean_pooling(model_output, encoded_input['attention_mask'])
                    single_embedding = F.normalize(single_embedding, p=2, dim=1)
                    single_embedding = single_embedding.cpu().numpy()
                    all_embeddings.append(single_embedding)
                    
                    del encoded_input, model_output, single_embedding
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
            else:
                raise e
    
    # Concatenate all embeddings
    embeddings = np.vstack(all_embeddings)
    print(f"Generated embeddings shape: {embeddings.shape}")
    
    return embeddings


def save_embeddings(embeddings, output_path, format='numpy'):
    """Save embeddings to file"""
    print(f"Saving embeddings to {output_path}...")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if format == 'numpy':
        np.save(output_path, embeddings)
    elif format == 'parquet':
        # Convert to DataFrame and save as parquet
        embedding_df = pd.DataFrame(embeddings)
        embedding_df.to_parquet(output_path, index=False)
    elif format == 'csv':
        # Convert to DataFrame and save as CSV
        embedding_df = pd.DataFrame(embeddings)
        embedding_df.to_csv(output_path, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    print(f"Embeddings saved successfully!")


def main():
    parser = argparse.ArgumentParser(description='Generate document embeddings using all-MiniLM-L6-v2')
    parser.add_argument('--input', required=True, help='Input parquet file path')
    parser.add_argument('--output', required=True, help='Output file path')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for processing (default: 32)')
    parser.add_argument('--max_length', type=int, default=256, help='Maximum sequence length (default: 256)')
    parser.add_argument('--format', choices=['numpy', 'parquet', 'csv'], default='numpy', 
                        help='Output format (default: numpy)')
    
    args = parser.parse_args()
    
    # Setup environment and device
    device = setup_environment()
    
    print(f"Starting embedding generation...")
    print(f"Input file: {args.input}")
    print(f"Output file: {args.output}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max length: {args.max_length}")
    print(f"Output format: {args.format}")
    
    # Load model and tokenizer
    print("\nLoading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    model.to(device)
    
    print(f"Model loaded on {device}")
    
    # Load and preprocess data
    texts = load_and_preprocess_data(args.input)
    
    # Generate embeddings
    start_time = time.time()
    embeddings = generate_embeddings_batch(
        texts=texts,
        model=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device
    )
    end_time = time.time()
    
    print(f"\nEmbedding generation completed in {end_time - start_time:.2f} seconds")
    print(f"Average time per document: {(end_time - start_time) / len(texts):.4f} seconds")
    
    # Save embeddings
    save_embeddings(embeddings, args.output, args.format)
    
    # Print summary
    print(f"\n=== SUMMARY ===")
    print(f"Documents processed: {len(texts):,}")
    print(f"Embedding dimensions: {embeddings.shape[1]}")
    print(f"Total processing time: {end_time - start_time:.2f} seconds")
    print(f"Memory usage: {embeddings.nbytes / (1024**3):.2f} GB")
    
    # Final GPU memory cleanup
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        print(f"GPU memory cleared")


if __name__ == "__main__":
    main()