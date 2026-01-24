import torch
import logging
import numpy as np
from typing import List, Union
from transformers import AutoTokenizer, AutoModel
from utils import embedding_cache, compute_text_hash, monitor_performance, batch_items

# === Logging ===
logger = logging.getLogger("embeddings")

# === Model Configuration ===
MODEL_NAME = "intfloat/e5-small-v2"  # 384-dim output, small size

# === GPU Detection ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"🚀 Using device: {device}")

# === Load Model & Tokenizer Once ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.to(device)  # Move model to GPU if available
model.eval()  # Disable dropout, etc.

# === Generate Single Embedding with Caching ===
@monitor_performance("get_embedding")
def get_embedding(text: str, use_cache: bool = True) -> list:
    """
    Generate a 384-dimensional vector embedding for the input text.
    Supports caching to avoid recomputing embeddings for identical text.
    
    Args:
        text (str): The input sentence or paragraph.
        use_cache (bool): Whether to use cache (default: True)
    
    Returns:
        list: Embedding vector as a list of floats.
    """
    text = text.strip()
    
    # Check cache first
    if use_cache:
        cache_key = compute_text_hash(text)
        cached = embedding_cache.get(cache_key)
        if cached is not None:
            logger.debug(f"Cache hit for text hash: {cache_key[:8]}...")
            return cached
    
    # Preprocess input (per E5 model's requirement)
    if not text.startswith("query:") and not text.startswith("passage:"):
        text = "passage: " + text

    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move to GPU if available

    # Generate embeddings (no gradients)
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
        
        # Move back to CPU for storage
        embedding = embedding.cpu().tolist()
    
    # Store in cache
    if use_cache:
        embedding_cache.put(cache_key, embedding)
    
    return embedding


# === Batch Embedding Generation ===
@monitor_performance("get_embeddings_batch")
def get_embeddings_batch(texts: List[str], batch_size: int = 32, use_cache: bool = True) -> List[list]:
    """
    Generate embeddings for multiple texts efficiently using batching.
    
    Args:
        texts (List[str]): List of text strings to embed
        batch_size (int): Number of texts to process in each batch
        use_cache (bool): Whether to use cache (default: True)
    
    Returns:
        List[list]: List of embedding vectors
    """
    embeddings = []
    texts_to_compute = []
    indices_to_compute = []
    
    # Check cache for each text
    for idx, text in enumerate(texts):
        text = text.strip()
        if use_cache:
            cache_key = compute_text_hash(text)
            cached = embedding_cache.get(cache_key)
            if cached is not None:
                embeddings.append(cached)
                continue
        
        texts_to_compute.append(text)
        indices_to_compute.append(idx)
        embeddings.append(None)  # Placeholder
    
    logger.info(f"Computing embeddings for {len(texts_to_compute)}/{len(texts)} texts (cache hits: {len(texts) - len(texts_to_compute)})")
    
    # Compute embeddings in batches
    computed_embeddings = []
    for batch in batch_items(texts_to_compute, batch_size):
        # Preprocess batch
        processed_batch = []
        for text in batch:
            if not text.startswith("query:") and not text.startswith("passage:"):
                text = "passage: " + text
            processed_batch.append(text)
        
        # Tokenize batch
        inputs = tokenizer(
            processed_batch,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            batch_embeddings = outputs.last_hidden_state.mean(dim=1)
            
            # Move to CPU and convert to list
            batch_embeddings = batch_embeddings.cpu().numpy()
            computed_embeddings.extend([emb.tolist() for emb in batch_embeddings])
    
    # Fill in computed embeddings and cache them
    for idx, orig_idx in enumerate(indices_to_compute):
        embedding = computed_embeddings[idx]
        embeddings[orig_idx] = embedding
        
        # Cache the computed embedding
        if use_cache:
            cache_key = compute_text_hash(texts[orig_idx].strip())
            embedding_cache.put(cache_key, embedding)
    
    return embeddings


# === Get Cache Statistics ===
def get_cache_stats() -> dict:
    """Return embedding cache statistics."""
    return embedding_cache.stats()
