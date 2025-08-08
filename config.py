import os
from dotenv import load_dotenv

# === Load environment variables from .env file ===
load_dotenv()

# === API Keys ===
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
BEARER_TOKEN = os.getenv("BEARER_TOKEN")

if not GROQ_API_KEY or not BEARER_TOKEN:
    raise EnvironmentError("GROQ_API_KEY and BEARER_TOKEN must be set in .env file")

# === LLM Model ===
LLM_MODEL = os.getenv("LLM_MODEL", "llama3-8b-8192")

# === PDF Folder Configuration ===
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
PDF_FOLDER = os.path.join(ROOT_DIR, os.getenv("PDF_FOLDER", "pdfs"))

# === Chunking Config ===
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))
MIN_CHUNK_LENGTH = int(os.getenv("MIN_CHUNK_LENGTH", 50))
TOP_K_CHUNKS = int(os.getenv("TOP_K_CHUNKS", 3))

# === Embedding Vector Dimension (must match your embedding model and pgvector column) ===
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", 384))

# === PostgreSQL DB Config ===
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_TABLE = os.getenv("DB_TABLE", "document_chunks")

# Ensure all critical DB fields are set
if not all([DB_NAME, DB_USER, DB_PASSWORD]):
    raise EnvironmentError("Database configuration incomplete in .env file")