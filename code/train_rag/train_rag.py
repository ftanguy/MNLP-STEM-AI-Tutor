import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    get_linear_schedule_with_warmup
)
import pandas as pd
import numpy as np
import yaml
import json
import logging
import argparse
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import os
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def mean_pooling(model_output, attention_mask):
    """
    Mean Pooling - Take attention mask into account for correct averaging
    This is the standard approach for sentence transformers
    """
    token_embeddings = model_output[0]  # First element contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class GeneratorOnlyRAGTrainer:
    def __init__(self, generator, encoder, generator_tokenizer, encoder_tokenizer, documents, k=5, 
                 precomputed_embeddings_path=None):
        """
        
        Args:
            generator: The generator model to train (e.g., Qwen, Llama)
            encoder: The frozen encoder for queries/documents
            generator_tokenizer: Tokenizer for the generator model
            encoder_tokenizer: Tokenizer for the encoder model
            documents: List of document strings
            k: Number of documents to retrieve
            precomputed_embeddings_path: Optional path to precomputed embeddings
        """
        self.generator = generator  # TRAINABLE
        self.encoder = encoder  # FROZEN
        self.generator_tokenizer = generator_tokenizer
        self.encoder_tokenizer = encoder_tokenizer
        self.documents = documents
        self.k = k
        
        self.device = next(self.encoder.parameters()).device
        logger.info(f"Using device: {self.device}")
        
        # Handle embeddings
        if precomputed_embeddings_path:
            logger.info(f"Loading pre-computed embeddings from {precomputed_embeddings_path}")
            self.doc_embeddings = self._load_precomputed_embeddings(precomputed_embeddings_path)
            self._validate_embeddings()
        else:
            logger.info("Computing document embeddings...")
            self.doc_embeddings = self._encode_documents_batch()
        
        # Freeze the encoder completely
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        logger.info(f"Final document embeddings shape: {self.doc_embeddings.shape}")
        
    def _load_precomputed_embeddings(self, embeddings_path: str) -> torch.Tensor:
        """Load pre-computed embeddings from numpy file"""
        try:
            if embeddings_path.endswith('.npy'):
                embeddings = np.load(embeddings_path)
            elif embeddings_path.endswith('.npz'):
                data = np.load(embeddings_path)
                embeddings = data['embeddings']
            else:
                raise ValueError(f"Unsupported file format: {embeddings_path}")
                
            return torch.from_numpy(embeddings).float()
        except Exception as e:
            logger.error(f"Error loading embeddings from {embeddings_path}: {e}")
            raise
    
    def _validate_embeddings(self):
        """Validate that embeddings match document count"""
        if len(self.documents) != self.doc_embeddings.shape[0]:
            logger.warning(f"Document count ({len(self.documents)}) != embedding count ({self.doc_embeddings.shape[0]})")
            min_count = min(len(self.documents), self.doc_embeddings.shape[0])
            self.documents = self.documents[:min_count]
            self.doc_embeddings = self.doc_embeddings[:min_count]
            logger.info(f"Trimmed both to {min_count} items")
        
    def _encode_documents_batch(self, batch_size=32):
        """Encode documents in batches using mean pooling"""
        all_embeddings = []
        
        for i in tqdm(range(0, len(self.documents), batch_size), desc="Encoding documents"):
            batch_docs = self.documents[i:i+batch_size]
            
            # Use encoder tokenizer for documents
            tokenized = self.encoder_tokenizer(
                batch_docs, 
                padding=True, 
                truncation=True, 
                max_length=512,
                return_tensors="pt"
            )
            
            tokenized = {k: v.to(self.device) for k, v in tokenized.items()}
            
            with torch.no_grad():
                model_output = self.encoder(**tokenized)
                embeddings = mean_pooling(model_output, tokenized['attention_mask'])
                embeddings = F.normalize(embeddings, p=2, dim=1)
                all_embeddings.append(embeddings.cpu())
        
        return torch.cat(all_embeddings, dim=0)
    
    def retrieve_documents(self, query_embeddings):
        """Retrieve top-k documents for each query"""
        # Ensure embeddings are on the same device as queries
        doc_embeddings = self.doc_embeddings.to(query_embeddings.device)
        
        # Compute similarities (cosine similarity since embeddings are normalized)
        similarities = torch.matmul(query_embeddings, doc_embeddings.T)
        top_k_indices = similarities.topk(self.k, dim=-1).indices
        
        return top_k_indices
    

    def forward(self, batch):
        """Forward pass for RAG training """ 
        with torch.no_grad():
            tokenized_queries = self.encoder_tokenizer(
                batch['questions'], 
                padding=True, 
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            tokenized_queries = {k: v.to(self.device) for k, v in tokenized_queries.items()}
            
            model_output = self.encoder(**tokenized_queries)
            query_embeddings = mean_pooling(model_output, tokenized_queries['attention_mask'])
            query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
            
        # 2. Retrieve documents 
        with torch.no_grad():
            top_k_indices = self.retrieve_documents(query_embeddings)
            
        # 3. Create RAG prompts (WITHOUT answers - this is the input)
        rag_prompts = []
        for i, question in enumerate(batch['questions']):
            retrieved_docs = [self.documents[idx.item()] for idx in top_k_indices[i]]
            context = " ".join(retrieved_docs[:5])
            rag_prompt = f"Question: {question}\nContext: {context}\nAnswer:"
            rag_prompts.append(rag_prompt)
        
        # 4. Create full sequences (prompt + answer for training)
        full_sequences = []
        for prompt, answer in zip(rag_prompts, batch['answers']):
            full_sequence = f"{prompt} {answer}"
            full_sequences.append(full_sequence)
            
        # 5. Tokenize everything
        generator_device = next(self.generator.parameters()).device
        
        # Tokenize the prompts to know where they end
        tokenized_prompts = self.generator_tokenizer(
            rag_prompts,
            padding=True,
            truncation=True,
            max_length=3584,  # Leave room for answers
            return_tensors="pt"
        )
        
        # Tokenize the full sequences
        tokenized_full = self.generator_tokenizer(
            full_sequences,
            padding=True,
            truncation=True,
            max_length=4096,
            return_tensors="pt"
        )
        
        # Move to device
        tokenized_prompts = {k: v.to(generator_device) for k, v in tokenized_prompts.items()}
        tokenized_full = {k: v.to(generator_device) for k, v in tokenized_full.items()}
        
        # 6. Create labels that only compute loss on answer tokens
        input_ids = tokenized_full['input_ids']
        attention_mask = tokenized_full['attention_mask']
        prompt_lengths = tokenized_prompts['attention_mask'].sum(dim=1)  # Length of each prompt
        
        # Create labels - same as input_ids but mask out prompt tokens
        labels = input_ids.clone()
        
        # Mask prompt tokens (set to -100 so they're ignored in loss computation)
        for i, prompt_len in enumerate(prompt_lengths):
            labels[i, :prompt_len] = -100
        
        # 7. Forward pass
        outputs = self.generator(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        return outputs.loss


class RAGDataset(Dataset):
    def __init__(self, data_path: str):
        """Load dataset from JSON or parquet file"""
        if data_path.endswith('.json'):
            with open(data_path, 'r') as f:
                self.data = json.load(f)
        elif data_path.endswith('.parquet'):
            self.data = pd.read_parquet(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
        
    def __len__(self):
        if isinstance(self.data, list):
            return len(self.data)
        return len(self.data)
    
    def __getitem__(self, idx):
        if isinstance(self.data, list):
            item = self.data[idx]
        else:
            item = self.data.iloc[idx]
        return {
            'questions': item['question'],
            'answers': item['answer']
        }


class RAGTrainerWrapper(Trainer):
    def __init__(self, rag_model, *args, **kwargs):
        self.rag_model = rag_model
        super().__init__(*args, **kwargs)
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Custom loss computation using RAG model"""
        
        loss = self.rag_model.forward(inputs)
        return (loss, None) if return_outputs else loss

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_documents(docs_path: str, num_chunks: int = 100000) -> List[str]:
    """Load documents from parquet file"""
    logger.info(f"Loading documents from {docs_path}")
    df = pd.read_parquet(docs_path)
    documents = df['text'].tolist()[:num_chunks]
    logger.info(f"Loaded {len(documents)} documents")
    return documents

def setup_models_and_tokenizer(config: Dict[str, Any]):
    """Initialize models and tokenizer from config"""
    model_args = config['model']['base_params']['model_args']
    embedding_model = config['model']['rag_params']['embedding_model']
    
    # Parse generator model name
    model_name = model_args.split('pretrained=')[1].split(',')[0]
    
    logger.info(f"Loading generator model: {model_name}")
    generator = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if config['model']['base_params']['dtype'] == 'float16' else torch.float32,
        device_map="auto"  # Automatically places model on available GPU(s)
    )
    
    logger.info(f"Loading encoder model: {embedding_model}")
    encoder = AutoModel.from_pretrained(embedding_model)
    
    # Move encoder to GPU if available
    if torch.cuda.is_available():
        encoder = encoder.cuda()
        logger.info(f"Moved encoder to GPU: {next(encoder.parameters()).device}")
    
    # Use Qwen tokenizer for M3_SFT generator (built on Qwen)
    qwen_base_model = "Qwen/Qwen3-0.6B-Base"
    logger.info(f"Loading Qwen tokenizer for generator: {qwen_base_model}")
    generator_tokenizer = AutoTokenizer.from_pretrained(qwen_base_model)
    if generator_tokenizer.pad_token is None:
        generator_tokenizer.pad_token = generator_tokenizer.eos_token
    
    # Use encoder's own tokenizer
    logger.info(f"Loading encoder tokenizer: {embedding_model}")
    encoder_tokenizer = AutoTokenizer.from_pretrained(embedding_model)
    if encoder_tokenizer.pad_token is None:
        encoder_tokenizer.pad_token = encoder_tokenizer.eos_token
    
    return generator, encoder, generator_tokenizer, encoder_tokenizer

def create_precomputed_embeddings(documents: List[str], encoder_model, encoder_tokenizer, 
                                 output_path: str, batch_size: int = 32):
    """Create and save pre-computed document embeddings"""
    logger.info(f"Creating pre-computed embeddings for {len(documents)} documents")
    
    device = next(encoder_model.parameters()).device
    embeddings = []
    
    for i in tqdm(range(0, len(documents), batch_size), desc="Computing embeddings"):
        batch = documents[i:i+batch_size]
        tokenized = encoder_tokenizer(
            batch, 
            padding=True, 
            truncation=True, 
            max_length=512, 
            return_tensors="pt"
        )
        tokenized = {k: v.to(device) for k, v in tokenized.items()}
        
        with torch.no_grad():
            model_output = encoder_model(**tokenized)
            batch_embeddings = mean_pooling(model_output, tokenized['attention_mask'])
            batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
            embeddings.append(batch_embeddings.cpu().numpy())
    
    all_embeddings = np.vstack(embeddings)
    
    # Save embeddings
    if output_path.endswith('.npz'):
        np.savez_compressed(output_path, embeddings=all_embeddings)
    else:
        np.save(output_path, all_embeddings)
    
    logger.info(f"Saved {all_embeddings.shape} embeddings to {output_path}")
    return all_embeddings.shape

def collate_fn(batch):
    """Custom collate function for DataLoader"""
    questions = [item['questions'] for item in batch]
    answers = [item['answers'] for item in batch]
    
    return {
        'questions': questions,
        'answers': answers
    }

def main():
    parser = argparse.ArgumentParser(description="Train RAG model (simplified - encoder always present)")
    parser.add_argument("--config_path", type=str, default="rag_model.yaml", help="Path to config file")
    parser.add_argument("--train_data", type=str, required=True, help="Path to training data")
    parser.add_argument("--val_data", type=str, required=True, help="Path to validation data")
    parser.add_argument("--docs_path", type=str, required=True, help="Path to documents")
    parser.add_argument("--precomputed_embeddings", type=str, help="Path to pre-computed embeddings (.npy or .npz)")
    parser.add_argument("--create_embeddings", action="store_true", help="Create and save embeddings before training")
    parser.add_argument("--embeddings_output", type=str, default="doc_embeddings.npy", help="Output path for created embeddings")
    parser.add_argument("--output_dir", type=str, default="./rag_checkpoints", help="Output directory for checkpoints")
    parser.add_argument("--final_model_dir", type=str, default="./final_rag_model", help="Final model directory")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Warmup steps")
    parser.add_argument("--logging_steps", type=int, default=100, help="Logging frequency")
    parser.add_argument("--eval_steps", type=int, default=500, help="Evaluation frequency")
    parser.add_argument("--save_steps", type=int, default=1000, help="Save frequency")
    
    args = parser.parse_args()
    
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
        logger.info(f"Current GPU: {torch.cuda.current_device()}")
        logger.info(f"GPU name: {torch.cuda.get_device_name()}")
    
    # Load config and setup
    config = load_config(args.config_path)
    documents = load_documents(args.docs_path, config['model']['rag_params']['num_chunks'])
    generator, encoder, generator_tokenizer, encoder_tokenizer = setup_models_and_tokenizer(config)
    
    # Handle embedding creation if requested
    if args.create_embeddings:
        create_precomputed_embeddings(
            documents, encoder, encoder_tokenizer, 
            args.embeddings_output, batch_size=32
        )
        logger.info(f"Embeddings created at {args.embeddings_output}")
        if args.precomputed_embeddings is None:
            args.precomputed_embeddings = args.embeddings_output
    
    # Initialize RAG trainer
    rag_model = GeneratorOnlyRAGTrainer(
        generator=generator,
        encoder=encoder,
        generator_tokenizer=generator_tokenizer,
        encoder_tokenizer=encoder_tokenizer,
        documents=documents,
        k=config['model']['rag_params']['top_k'],
        precomputed_embeddings_path=args.precomputed_embeddings
    )
    
    # Load datasets
    train_dataset = RAGDataset(args.train_data)
    val_dataset = RAGDataset(args.val_data)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=0.01,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=4,
        fp16=True,
        remove_unused_columns=False,
        report_to=None,
    )
    
    # Initialize trainer
    trainer = RAGTrainerWrapper(
        rag_model=rag_model,
        model=generator,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
    )
    
    # Train the model
    logger.info("Starting training...")
    trainer.train()
    
    # Save the final model
    logger.info(f"Saving final model to {args.final_model_dir}")
    trainer.save_model(args.final_model_dir)
    generator_tokenizer.save_pretrained(args.final_model_dir)
    
    # Save config and metadata
    config_save_path = f"{args.final_model_dir}/rag_config.yaml"
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f)
    
    if args.precomputed_embeddings:
        embedding_info = {
            'embeddings_path': args.precomputed_embeddings,
            'num_documents': len(documents),
            'embedding_dim': rag_model.doc_embeddings.shape[1]
        }
        with open(f"{args.final_model_dir}/embedding_info.json", 'w') as f:
            json.dump(embedding_info, f, indent=2)
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()