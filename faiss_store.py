"""
FAISS-based vector store for fast similarity search.
Alternative to PostgreSQL pgvector for high-performance retrieval.
"""

import os
import pickle
import logging
import numpy as np
from typing import List, Tuple, Optional
import faiss
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_DIM, TOP_K_CHUNKS
from utils import monitor_performance, compute_text_hash

# === Logging ===
logger = logging.getLogger("faiss_store")

# === FAISS Vector Store ===
class FAISSVectorStore:
    """
    FAISS-based vector store with metadata management.
    Uses IVFPQ for efficient approximate nearest neighbor search.
    """
    
    def __init__(self, dimension: int = EMBEDDING_DIM, index_type: str = "IVF"):
        """
        Initialize FAISS vector store.
        
        Args:
            dimension: Embedding vector dimension
            index_type: Type of FAISS index ("Flat", "IVF", or "HNSW")
        """
        self.dimension = dimension
        self.index_type = index_type
        self.index = None
        self.chunks = []  # Store actual text chunks
        self.metadata = []  # Store document names and other metadata
        self.text_hashes = set()  # For deduplication
        self.model = SentenceTransformer("intfloat/e5-small-v2")
        
        self._initialize_index()
    
    def _initialize_index(self):
        """Initialize FAISS index based on type."""
        if self.index_type == "Flat":
            # Exact search (slower but accurate)
            self.index = faiss.IndexFlatL2(self.dimension)
        elif self.index_type == "IVF":
            # Inverted File Index for faster search
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)  # 100 clusters
            self.index.nprobe = 10  # Search in 10 nearest clusters
        elif self.index_type == "HNSW":
            # Hierarchical Navigable Small World for very fast search
            self.index = faiss.IndexHNSWFlat(self.dimension, 32)
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
        
        logger.info(f"✅ Initialized FAISS index (type={self.index_type}, dim={self.dimension})")
    
    @monitor_performance("faiss_add_chunks")
    def add_chunks(self, chunks: List[str], document_name: str, batch_size: int = 100):
        """
        Add chunks to FAISS index with deduplication.
        
        Args:
            chunks: List of text chunks
            document_name: Source document name
            batch_size: Batch size for encoding
        """
        if not chunks:
            return
        
        added_count = 0
        skipped_count = 0
        
        # Process in batches
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            # Check for duplicates
            batch_to_add = []
            for chunk in batch:
                text_hash = compute_text_hash(chunk)
                if text_hash not in self.text_hashes:
                    batch_to_add.append(chunk)
                    self.text_hashes.add(text_hash)
                    added_count += 1
                else:
                    skipped_count += 1
            
            if not batch_to_add:
                continue
            
            # Encode batch
            embeddings = self.model.encode(batch_to_add, normalize_embeddings=True, show_progress_bar=False)
            embeddings = np.array(embeddings, dtype=np.float32)
            
            # Train index if using IVF and this is first batch
            if self.index_type == "IVF" and not self.index.is_trained:
                if embeddings.shape[0] >= 100:
                    logger.info("Training FAISS IVF index...")
                    self.index.train(embeddings)
                else:
                    logger.warning("Not enough data to train IVF index, switching to Flat")
                    self.index = faiss.IndexFlatL2(self.dimension)
            
            # Add to index
            if self.index_type != "IVF" or self.index.is_trained:
                self.index.add(embeddings)
                self.chunks.extend(batch_to_add)
                self.metadata.extend([{"document_name": document_name}] * len(batch_to_add))
        
        logger.info(f"✅ Added {added_count} chunks to FAISS index, skipped {skipped_count} duplicates")
    
    @monitor_performance("faiss_search")
    def search(self, query: str, k: int = TOP_K_CHUNKS, document_name: Optional[str] = None) -> List[str]:
        """
        Search for top-k most similar chunks.
        
        Args:
            query: Query text
            k: Number of results to return
            document_name: Optional filter by document name
        
        Returns:
            List of relevant chunk texts
        """
        if self.index.ntotal == 0:
            logger.warning("FAISS index is empty")
            return []
        
        # Encode query
        query_embedding = self.model.encode(query, normalize_embeddings=True)
        query_embedding = np.array([query_embedding], dtype=np.float32)
        
        # Search - retrieve more if filtering by document
        search_k = k * 10 if document_name else k
        distances, indices = self.index.search(query_embedding, min(search_k, self.index.ntotal))
        
        # Get results
        results = []
        for idx in indices[0]:
            if idx < len(self.chunks):
                # Filter by document if specified
                if document_name and self.metadata[idx]["document_name"] != document_name:
                    continue
                results.append(self.chunks[idx])
                if len(results) >= k:
                    break
        
        return results
    
    def save(self, filepath: str):
        """Save FAISS index and metadata to disk."""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, f"{filepath}.index")
        
        # Save metadata
        with open(f"{filepath}.meta", "wb") as f:
            pickle.dump({
                "chunks": self.chunks,
                "metadata": self.metadata,
                "text_hashes": self.text_hashes,
                "dimension": self.dimension,
                "index_type": self.index_type
            }, f)
        
        logger.info(f"✅ Saved FAISS index to {filepath}")
    
    def load(self, filepath: str):
        """Load FAISS index and metadata from disk."""
        # Load FAISS index
        self.index = faiss.read_index(f"{filepath}.index")
        
        # Load metadata
        with open(f"{filepath}.meta", "rb") as f:
            data = pickle.load(f)
            self.chunks = data["chunks"]
            self.metadata = data["metadata"]
            self.text_hashes = data["text_hashes"]
            self.dimension = data["dimension"]
            self.index_type = data["index_type"]
        
        logger.info(f"✅ Loaded FAISS index from {filepath} ({self.index.ntotal} vectors)")
    
    def clear(self):
        """Clear all data from the index."""
        self._initialize_index()
        self.chunks = []
        self.metadata = []
        self.text_hashes = set()
        logger.info("✅ FAISS index cleared")
    
    def get_stats(self) -> dict:
        """Get index statistics."""
        return {
            "total_vectors": self.index.ntotal,
            "dimension": self.dimension,
            "index_type": self.index_type,
            "total_chunks": len(self.chunks),
            "unique_hashes": len(self.text_hashes)
        }


# === Global FAISS Store Instance ===
_faiss_store = None

def get_faiss_store() -> FAISSVectorStore:
    """Get or create global FAISS store instance."""
    global _faiss_store
    if _faiss_store is None:
        _faiss_store = FAISSVectorStore(index_type="IVF")
    return _faiss_store
