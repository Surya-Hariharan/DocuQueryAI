import os
import psycopg2
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
from config import (
    DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME, DB_TABLE,
    EMBEDDING_DIM, TOP_K_CHUNKS
)

# Load embedding model once
model = SentenceTransformer("intfloat/e5-small-v2")

# === PostgreSQL Connection ===
def get_db_connection():
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        # Append sslmode if not present
        if "sslmode" not in db_url:
            if "?" in db_url:
                db_url += "&sslmode=require"
            else:
                db_url += "?sslmode=require"
        return psycopg2.connect(db_url)
    else:
        return psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            dbname=DB_NAME,
            sslmode="require"  # Required for Render PostgreSQL
        )

# === DB Initialization ===
def init_db():
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {DB_TABLE} (
                    id SERIAL PRIMARY KEY,
                    chunk_text TEXT,
                    embedding VECTOR({EMBEDDING_DIM}),
                    document_name TEXT
                );
            """)
        conn.commit()

# === Compute Embedding ===
def get_embedding(text: str) -> np.ndarray:
    return model.encode(text.strip(), normalize_embeddings=True)

# === Upsert Chunks ===
def upsert_chunks(chunks: List[str], document_name: str):
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            for chunk in chunks:
                emb = get_embedding(chunk)
                cur.execute(
                    f"""
                    INSERT INTO {DB_TABLE} (chunk_text, embedding, document_name)
                    VALUES (%s, %s, %s);
                    """,
                    (chunk, emb.tolist(), document_name)
                )
        conn.commit()

# === Query Top-K Relevant Chunks ===
def query_top_k_chunks(query: str, k: int = TOP_K_CHUNKS) -> List[str]:
    query_embedding = get_embedding(query)
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT chunk_text
                FROM {DB_TABLE}
                ORDER BY embedding <-> %s::vector
                LIMIT %s;
                """,
                (query_embedding.tolist(), k)
            )
            results = cur.fetchall()
    return [r[0] for r in results]
