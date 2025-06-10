import pandas as pd
import json
from transformers import AutoTokenizer
from typing import List, Dict
import math

def load_and_chunk_rag_documents(
    parquet_path: str,
    max_tokens: int = 256,
    output_path_all: str = "./doc_embeddings/unembed_all_chunks.parquet",
    output_path_100k: str = "./doc_embeddings/unembed_top_100k_chunks.parquet"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load RAG documents from parquet and chunk them for all-MiniLM-L6-v2 (256 token limit)
    Creates two files: one with all chunks, one with top 100k longest chunks
    """
    
    # Load the tokenizer for all-MiniLM-L6-v2
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    
    # Load the parquet file
    print(f"Loading parquet file from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    
    # Parse JSON if the text field contains JSON strings
    if isinstance(df.iloc[0]['text'], str):
        try:
            # Try to parse as JSON
            df['parsed'] = df['text'].apply(lambda x: json.loads(x) if x.startswith('{') else {'text': x, 'source': 'unknown'})
            df['text_content'] = df['parsed'].apply(lambda x: x.get('text', ''))
            df['source'] = df['parsed'].apply(lambda x: x.get('source', 'unknown'))
        except:
            # If not JSON, treat as plain text
            df['text_content'] = df['text']
            df['source'] = 'unknown'
    else:
        df['text_content'] = df['text']
        df['source'] = 'unknown'
    
    print(f"Loaded {len(df)} documents")
    
    chunked_documents = []
    total_chunks = 0
    
    for idx, row in df.iterrows():
        text = row['text_content']
        source = row.get('source', 'unknown')
        
        # Tokenize the text
        tokens = tokenizer.encode(text, add_special_tokens=False)
        
        if len(tokens) <= max_tokens:
            # Document fits in one chunk
            chunked_documents.append({
                'text': text,
                'source': source,
                'original_doc_id': idx,
                'chunk_id': 0,
                'token_count': len(tokens),
                'total_chunks_for_doc': 1
            })
            total_chunks += 1
        else:
            # Need to split into multiple chunks (no overlap)
            chunks = create_separated_chunks(
                tokens, 
                tokenizer, 
                max_tokens=max_tokens
            )
            
            for chunk_idx, chunk_text in enumerate(chunks):
                chunk_tokens = tokenizer.encode(chunk_text, add_special_tokens=False)
                chunked_documents.append({
                    'text': chunk_text,
                    'source': source,
                    'original_doc_id': idx,
                    'chunk_id': chunk_idx,
                    'token_count': len(chunk_tokens),
                    'total_chunks_for_doc': len(chunks)
                })
                total_chunks += 1
        
        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx + 1} documents, created {total_chunks} chunks so far...")
    
    # Create DataFrame with all chunked documents
    all_chunks_df = pd.DataFrame(chunked_documents)
    
    # Save all chunks
    all_chunks_df.to_parquet(output_path_all, index=False)
    print(f"All chunks saved to: {output_path_all}")
    
    # Create top 100k longest chunks
    top_100k_df = all_chunks_df.nlargest(100000, 'token_count').copy()
    top_100k_df.to_parquet(output_path_100k, index=False)
    print(f"Top 100k longest chunks saved to: {output_path_100k}")
    
    print(f"\nChunking complete!")
    print(f"Original documents: {len(df)}")
    print(f"Total chunks created: {total_chunks}")
    print(f"Average chunks per document: {total_chunks / len(df):.2f}")
    
    # Print statistics for all chunks
    print(f"\n=== ALL CHUNKS STATISTICS ===")
    print(f"Total chunks: {len(all_chunks_df)}")
    print(f"Mean tokens per chunk: {all_chunks_df['token_count'].mean():.1f}")
    print(f"Max tokens per chunk: {all_chunks_df['token_count'].max()}")
    print(f"Min tokens per chunk: {all_chunks_df['token_count'].min()}")
    
    # Print statistics for top 100k chunks
    print(f"\n=== TOP 100K LONGEST CHUNKS STATISTICS ===")
    print(f"Total chunks: {len(top_100k_df)}")
    print(f"Mean tokens per chunk: {top_100k_df['token_count'].mean():.1f}")
    print(f"Max tokens per chunk: {top_100k_df['token_count'].max()}")
    print(f"Min tokens per chunk: {top_100k_df['token_count'].min()}")
    print(f"Token cutoff for top 100k: {top_100k_df['token_count'].min()}")
    
    # ADD THIS LINE: Print total token counts for both files
    print(f"\n=== TOTAL TOKEN COUNTS ===")
    print(f"All chunks total tokens: {all_chunks_df['token_count'].sum():,}")
    print(f"Top 100k chunks total tokens: {top_100k_df['token_count'].sum():,}")
    print(f"Top 100k represents {top_100k_df['token_count'].sum() / all_chunks_df['token_count'].sum() * 100:.1f}% of all tokens")
    
    return all_chunks_df, top_100k_df

def create_separated_chunks(
    tokens: List[int], 
    tokenizer, 
    max_tokens: int = 256
) -> List[str]:
    """
    Create exactly separated chunks from token list (no overlap)
    
    Args:
        tokens: List of token IDs
        tokenizer: Tokenizer instance
        max_tokens: Maximum tokens per chunk
    
    Returns:
        List of chunk texts
    """
    chunks = []
    
    for i in range(0, len(tokens), max_tokens):
        # Extract chunk tokens
        chunk_tokens = tokens[i:i + max_tokens]
        
        # Decode back to text
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text)
    
    return chunks

def analyze_document_lengths(parquet_path: str) -> None:
    """
    Analyze the token distribution in your documents
    """
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    
    # Load and parse data
    df = pd.read_parquet(parquet_path)
    
    if isinstance(df.iloc[0]['text'], str):
        try:
            df['parsed'] = df['text'].apply(lambda x: json.loads(x) if x.startswith('{') else {'text': x})
            df['text_content'] = df['parsed'].apply(lambda x: x.get('text', ''))
        except:
            df['text_content'] = df['text']
    else:
        df['text_content'] = df['text']
    
    # Calculate token counts
    df['token_count'] = df['text_content'].apply(
        lambda x: len(tokenizer.encode(x, add_special_tokens=False))
    )
    
    print("Document length analysis:")
    print(f"Total documents: {len(df)}")
    print(f"Mean tokens: {df['token_count'].mean():.1f}")
    print(f"Median tokens: {df['token_count'].median():.1f}")
    print(f"Max tokens: {df['token_count'].max()}")
    print(f"Documents > 256 tokens: {(df['token_count'] > 256).sum()} ({(df['token_count'] > 256).mean()*100:.1f}%)")
    
    # Show distribution
    print(f"\nToken distribution:")
    print(df['token_count'].describe())

# Example usage
if __name__ == "__main__":
    # Replace with your parquet file path
    parquet_file = "/capstor/scratch/cscs/inesaltemir/MNLP/rag_docs/rag_documents/rag_documents.parquet"
    
    # First, analyze your documents
    print("=== Document Analysis ===")
    analyze_document_lengths(parquet_file)
    
    print("\n=== Creating Chunks ===")
    # Create chunks with 256 token limit - no overlap, two output files
    all_chunks_df, top_100k_df = load_and_chunk_rag_documents(
        parquet_path=parquet_file,
        max_tokens=256                    # all-MiniLM-L6-v2 limit
    )
    
    # Show sample of chunked data
    print("\n=== Sample from ALL chunks ===")
    print(all_chunks_df.head(3)[['text', 'source', 'chunk_id', 'token_count']].to_string())
    
    print("\n=== Sample from TOP 100K chunks ===")
    print(top_100k_df.head(3)[['text', 'source', 'chunk_id', 'token_count']].to_string())