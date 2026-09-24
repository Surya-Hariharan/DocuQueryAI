import os
import logging
import asyncio
import psycopg2
from psycopg2 import pool
import numpy as np
from typing import List, Set
from sentence_transformers import SentenceTransformer
from config import (
    DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME, DB_TABLE,
    EMBEDDING_DIM, TOP_K_CHUNKS
)
from utils import monitor_performance, compute_text_hash, batch_items, retry_on_failure

# === Logging ===
logger = logging.getLogger("db_vector_store")

# Load embedding model once
model = SentenceTransformer("intfloat/e5-small-v2")

# === Connection Pool for Better Performance ===
connection_pool = None

def init_connection_pool(minconn=1, maxconn=10):
    """Initialize PostgreSQL connection pool."""
    global connection_pool
    
    db_url = os.getenv("DATABASE_URL")
    
    try:
        if db_url:
            # Parse DATABASE_URL for connection pool
            if "sslmode" not in db_url:
                if "?" in db_url:
                    db_url += "&sslmode=allow"
                else:
                    db_url += "?sslmode=allow"
            connection_pool = pool.SimpleConnectionPool(minconn, maxconn, db_url)
        else:
            connection_pool = pool.SimpleConnectionPool(
                minconn, maxconn,
                host=DB_HOST,
                port=DB_PORT,
                user=DB_USER,
                password=DB_PASSWORD,
                dbname=DB_NAME,
                sslmode="allow",
                connect_timeout=30
            )
        logger.info(f"✅ Connection pool initialized (min={minconn}, max={maxconn})")
    except Exception as e:
        logger.error(f"Failed to initialize connection pool: {e}")
        raise

# === Get Connection from Pool ===
@retry_on_failure(max_retries=3, delay=1.0)
def get_db_connection():
    """Get connection from pool."""
    if connection_pool is None:
        init_connection_pool()
    return connection_pool.getconn()

def return_db_connection(conn):
    """Return connection to pool."""
    if connection_pool:
        connection_pool.putconn(conn)

# === DB Initialization with Indexes ===
@monitor_performance("init_db")
def init_db():
    """Initialize database with optimized schema and indexes."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # Enable pgvector extension
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # Create table with hash column for deduplication
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {DB_TABLE} (
                    id SERIAL PRIMARY KEY,
                    chunk_text TEXT NOT NULL,
                    text_hash VARCHAR(64) UNIQUE NOT NULL,
                    embedding VECTOR({EMBEDDING_DIM}) NOT NULL,
                    document_name TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Create indexes for better query performance
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{DB_TABLE}_document_name 
                ON {DB_TABLE}(document_name);
            """)
            
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{DB_TABLE}_text_hash 
                ON {DB_TABLE}(text_hash);
            """)
            
            # Create IVFFLAT index for faster vector similarity search
            # Note: This requires sufficient data; add after inserting documents
            try:
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{DB_TABLE}_embedding_ivfflat
                    ON {DB_TABLE} USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100);
                """)
            except Exception as e:
                logger.warning(f"Could not create IVFFLAT index (may need more data): {e}")
            
        conn.commit()
        logger.info("✅ Database initialized with optimized schema")
    finally:
        return_db_connection(conn)

# === Compute Embedding ===
@monitor_performance("compute_embedding")
def get_embedding(text: str) -> np.ndarray:
    """Generate normalized embedding for text."""
    return model.encode(text.strip(), normalize_embeddings=True)

# === Batch Upsert Chunks with Deduplication ===
@monitor_performance("upsert_chunks_batch")
def upsert_chunks(chunks: List[str], document_name: str, batch_size: int = 100):
    """
    Upsert chunks with batch processing and deduplication.
    
    Args:
        chunks: List of text chunks
        document_name: Name of the source document
        batch_size: Number of chunks to process per batch
    """
    if not chunks:
        logger.warning("No chunks to upsert")
        return
    
    conn = get_db_connection()
    try:
        inserted_count = 0
        skipped_count = 0
        
        # Process in batches
        for batch in batch_items(chunks, batch_size):
            # Compute embeddings for batch
            embeddings = [get_embedding(chunk) for chunk in batch]
            
            # Prepare data with hashes
            batch_data = []
            for chunk, emb in zip(batch, embeddings):
                text_hash = compute_text_hash(chunk)
                batch_data.append((chunk, text_hash, emb.tolist(), document_name))
            
            # Batch insert with conflict handling (deduplication)
            with conn.cursor() as cur:
                for chunk_text, text_hash, embedding, doc_name in batch_data:
                    try:
                        cur.execute(
                            f"""
                            INSERT INTO {DB_TABLE} (chunk_text, text_hash, embedding, document_name)
                            VALUES (%s, %s, %s, %s)
                            ON CONFLICT (text_hash) DO NOTHING;
                            """,
                            (chunk_text, text_hash, embedding, doc_name)
                        )
                        if cur.rowcount > 0:
                            inserted_count += 1
                        else:
                            skipped_count += 1
                    except Exception as e:
                        logger.error(f"Error inserting chunk: {e}")
            
            conn.commit()
        
        logger.info(f"✅ Upserted {inserted_count} chunks, skipped {skipped_count} duplicates for '{document_name}'")
    finally:
        return_db_connection(conn)

# === Query Top-K Relevant Chunks with Caching ===
@monitor_performance("query_top_k_chunks")
def query_top_k_chunks(query: str, k: int = TOP_K_CHUNKS, document_name: str = None) -> List[str]:
    """
    Query top-k most relevant chunks using vector similarity.
    
    Args:
        query: User query text
        k: Number of top chunks to retrieve
        document_name: Optional filter by document name
    
    Returns:
        List of relevant chunk texts
    """
    query_embedding = get_embedding(query)
    
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            if document_name:
                # Filter by document
                cur.execute(
                    f"""
                    SELECT chunk_text, (embedding <-> %s::vector) as distance
                    FROM {DB_TABLE}
                    WHERE document_name = %s
                    ORDER BY distance
                    LIMIT %s;
                    """,
                    (query_embedding.tolist(), document_name, k)
                )
            else:
                # Search all documents
                cur.execute(
                    f"""
                    SELECT chunk_text, (embedding <-> %s::vector) as distance
                    FROM {DB_TABLE}
                    ORDER BY distance
                    LIMIT %s;
                    """,
                    (query_embedding.tolist(), k)
                )
            results = cur.fetchall()
        
        return [r[0] for r in results]
    finally:
        return_db_connection(conn)

# === Get Document Count ===
def get_document_count() -> int:
    """Get total number of stored chunks."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {DB_TABLE};")
            count = cur.fetchone()[0]
        return count
    finally:
        return_db_connection(conn)

# === Clear Database ===
def clear_database():
    """Clear all data from the database."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(f"TRUNCATE TABLE {DB_TABLE} RESTART IDENTITY;")
        conn.commit()
        logger.info("✅ Database cleared")
    finally:
        return_db_connection(conn)
