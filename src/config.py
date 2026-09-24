import os
from dotenv import load_dotenv

# === Load environment variables from .env file ===
# config.py lives one directory below the repo root (src/config.py);
# point load_dotenv at the repo-root .env explicitly rather than relying on
# its default search behavior, which varies by python-dotenv version and by
# the process's current working directory.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
load_dotenv(os.path.join(_REPO_ROOT, ".env"))

# === API Keys ===
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
BEARER_TOKEN = os.getenv("BEARER_TOKEN")

if not GROQ_API_KEY or not BEARER_TOKEN:
    raise EnvironmentError("GROQ_API_KEY and BEARER_TOKEN must be set in .env file")

# === LLM Model ===
LLM_MODEL = os.getenv("LLM_MODEL", "llama3-8b-8192")

# === PDF Folder Configuration ===
PDF_FOLDER = os.path.join(_REPO_ROOT, os.getenv("PDF_FOLDER", "pdfs"))

# === Chunking Config (optimized for token-based chunking) ===
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 512))  # Max tokens per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))  # Overlap in tokens
MIN_CHUNK_LENGTH = int(os.getenv("MIN_CHUNK_LENGTH", 10))  # Min tokens
TOP_K_CHUNKS = int(os.getenv("TOP_K_CHUNKS", 5))  # Increased for better context

# === Embedding Vector Dimension (must match your embedding model and pgvector column) ===
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", 384))

# === Answer Generation Context Budget ===
# Approximate, not exact: we don't have a local tokenizer for the Groq LLM
# (LLM_MODEL), so context length is budgeted with a conservative ~4-chars-
# per-token estimate rather than a real tokenizer count for that model.
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", 3000))

# === PostgreSQL DB Config ===
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_TABLE = os.getenv("DB_TABLE", "document_chunks")

# === Upload / Download Safety ===
MAX_DOWNLOAD_BYTES = int(os.getenv("MAX_DOWNLOAD_BYTES", 26_214_400))  # 25 MB default

# === Document Identity ===
# No real multi-tenant auth exists yet (Tier 3 work) — this is the single
# tenant every document/chunk is scoped under until that lands, but the
# column/parameter already exists everywhere so adding real tenants later
# doesn't require touching the schema or call sites again.
DEFAULT_TENANT_ID = os.getenv("DEFAULT_TENANT_ID", "default")

# === Performance Optimization Config ===
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))  # Batch size for embedding generation
CACHE_SIZE = int(os.getenv("CACHE_SIZE", 5000))  # LRU cache size for embeddings
USE_GPU = os.getenv("USE_GPU", "true").lower() == "true"  # Enable GPU acceleration

# === Connection Pool Config ===
DB_POOL_MIN = int(os.getenv("DB_POOL_MIN", 2))
DB_POOL_MAX = int(os.getenv("DB_POOL_MAX", 10))

# Ensure all critical DB fields are set
if not all([DB_NAME, DB_USER, DB_PASSWORD]):
    raise EnvironmentError("Database configuration incomplete in .env file")
