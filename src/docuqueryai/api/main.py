import os
import uuid
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse
import requests
import uvicorn

from fastapi import FastAPI, Request, HTTPException, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

from docuqueryai.config import GROQ_API_KEY, BEARER_TOKEN, TOP_K_CHUNKS, MAX_DOWNLOAD_BYTES, DB_POOL_MIN, DB_POOL_MAX
from docuqueryai.ingestion.pipeline import parse_document
from docuqueryai.ingestion.file_type_detector import detect_file_type
from docuqueryai.retrieval.context_builder import build_context
from docuqueryai.generation.answer_generator import generate_answer, looks_grounded, get_token_usage_stats
from docuqueryai.retrieval.pg_vector_store import PgVectorStore
from docuqueryai.utils import monitor_performance, query_cache, compute_text_hash, compute_bytes_hash, embedding_cache
from docuqueryai.retrieval.embeddings import get_cache_stats
from docuqueryai.security.url_safety import safe_fetch

# === Vector Store (single shared instance) ===
vector_store = PgVectorStore()

# === Logging Setup ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("main")

# === Thread Pool for CPU-bound tasks ===
executor = ThreadPoolExecutor(max_workers=4)

# === FastAPI Models ===
class QueryRequest(BaseModel):
    documents: str  # URL of the document (PDF, DOCX, XLSX, or CSV — format is detected from its bytes)
    questions: List[str]

class SourceRef(BaseModel):
    source: int
    document_id: str
    source_uri: Optional[str] = None
    page_number: Optional[int] = None
    sheet_name: Optional[str] = None

class AnswerWithSources(BaseModel):
    answer: str
    sources: List[SourceRef] = []
    low_confidence: bool = False

class QueryResponse(BaseModel):
    answers: List[AnswerWithSources]

class UploadResponse(BaseModel):
    document_id: str
    chunks_stored: int

class AskRequest(BaseModel):
    questions: List[str]

class HealthResponse(BaseModel):
    status: str
    details: dict

# === FastAPI App Init ===
app = FastAPI(
    title="DocuQueryAI - Optimized RAG System",
    description="Production-ready RAG system with async processing, caching, and GPU acceleration",
    version="3.0.0"
)

# === CORS Middleware ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Optional Bearer Token Verification ===
def verify_token(request: Request):
    expected = f"Bearer {BEARER_TOKEN}"
    token = request.headers.get("Authorization")
    if not token or token != expected:
        logger.warning("Unauthorized access attempt.")
        raise HTTPException(status_code=401, detail="Invalid token")
    return True

# === Startup: Initialize DB and Connection Pool ===
@app.on_event("startup")
async def on_startup():
    """Initialize database and connection pool on startup."""
    try:
        vector_store.init_connection_pool(minconn=DB_POOL_MIN, maxconn=DB_POOL_MAX)
        vector_store.init_schema()
        logger.info("✅ Database and connection pool initialized")
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        raise

# === Shutdown: Cleanup ===
@app.on_event("shutdown")
async def on_shutdown():
    """Cleanup on shutdown."""
    executor.shutdown(wait=True)
    logger.info("✅ Shutdown complete")

# === Health Check Endpoint ===
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Enhanced health check with cache statistics."""
    status = "healthy"
    details = {}
    
    try:
        # Test database connection
        doc_count = await asyncio.get_event_loop().run_in_executor(
            executor, vector_store.count
        )
        details["database"] = "healthy"
        details["total_chunks"] = doc_count
        
        # Get cache statistics
        emb_stats = get_cache_stats()
        details["embedding_cache"] = emb_stats
        details["query_cache"] = query_cache.stats()
        details["token_usage"] = get_token_usage_stats()
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        status = "degraded"
        details["error"] = str(e)
    
    return HealthResponse(status=status, details=details)


def _log_prefix(request_id: str, document_id: Optional[str] = None) -> str:
    """A consistent [req=... doc=...] prefix so one request's log lines can
    be grepped out of a multi-worker, concurrent-request log stream."""
    return f"[req={request_id} doc={document_id}]" if document_id else f"[req={request_id}]"


# === Shared ingestion helper (used by both the URL-fetch and upload paths) ===
async def _ingest_bytes(
    loop: asyncio.AbstractEventLoop,
    content: bytes,
    source_uri: str,
    source_type: str,
    request_id: str,
    filename_hint: Optional[str] = None
) -> tuple:
    """
    Resolve (or create) the document's identity, parse it through the
    format-agnostic pipeline, and store its chunks. Returns
    (document_id, chunks_stored).
    """
    checksum = compute_bytes_hash(content)
    document_id = await loop.run_in_executor(
        executor,
        lambda: vector_store.get_or_create_document(source_uri, source_type=source_type, checksum=checksum)
    )
    prefix = _log_prefix(request_id, document_id)

    try:
        chunks = await loop.run_in_executor(executor, parse_document, content, filename_hint)
    except ValueError as e:
        logger.warning(f"{prefix} Could not parse document: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    if chunks:
        await loop.run_in_executor(executor, vector_store.upsert_chunks, chunks, document_id)

    logger.info(f"{prefix} Ingested {len(chunks)} chunks from '{source_uri}'")
    return document_id, len(chunks)


# === Shared question-answering helper (used by both /hackrx/run and /documents/{id}/ask) ===
async def _answer_questions(
    loop: asyncio.AbstractEventLoop,
    questions: List[str],
    document_id: str,
    cache_namespace: str,
    request_id: str
) -> List[AnswerWithSources]:
    prefix = _log_prefix(request_id, document_id)

    def _error_answer() -> AnswerWithSources:
        return AnswerWithSources(answer="Sorry, an error occurred while answering this question.", sources=[], low_confidence=False)

    async def answer_question(question: str) -> AnswerWithSources:
        if not question.strip():
            return AnswerWithSources(answer="Please provide a valid question.", sources=[], low_confidence=False)

        cache_key = compute_text_hash(f"{cache_namespace}:{question}")
        cached = query_cache.get(cache_key)
        if cached:
            logger.info(f"{prefix} 🎯 Cache hit for question: {question[:50]}...")
            return cached

        try:
            # Retrieve relevant chunks — scoped to the document just ingested,
            # not the entire cross-document table.
            retrieved = await loop.run_in_executor(
                executor,
                vector_store.query_top_k,
                question,
                TOP_K_CHUNKS,
                document_id
            )

            context, sources = build_context(retrieved)
            answer = await loop.run_in_executor(
                executor,
                generate_answer,
                question,
                context
            )
            low_confidence = not looks_grounded(answer, context)
            if low_confidence:
                logger.warning(f"{prefix} Low-confidence answer for question: {question[:50]!r}")
        except Exception as e:
            logger.error(f"{prefix} Failed to answer question {question[:50]!r}: {e}")
            return _error_answer()

        result = AnswerWithSources(
            answer=answer,
            sources=[SourceRef(**s) for s in sources],
            low_confidence=low_confidence
        )
        query_cache.put(cache_key, result)
        return result

    # A failure in one question must not take down the answers already
    # computed for the others.
    answers = await asyncio.gather(
        *[answer_question(q) for q in questions],
        return_exceptions=True
    )
    return [a if isinstance(a, AnswerWithSources) else _error_answer() for a in answers]


# === Main Document QA Endpoint (URL fetch + ingest + answer, in one call) ===
@app.post("/hackrx/run", response_model=QueryResponse)
@monitor_performance("hackrx_run_endpoint")
async def process_queries(req: QueryRequest, _=Depends(verify_token)):
    """
    Fetch a document from a URL, ingest it (any supported format — PDF,
    DOCX, XLSX, CSV — detected from its actual bytes, not the URL), and
    answer the given questions against it.
    """
    request_id = uuid.uuid4().hex[:12]
    prefix = _log_prefix(request_id)
    logger.info(f"{prefix} 📨 Received {len(req.questions)} questions for document: {req.documents}")

    if not req.documents.startswith("http"):
        raise HTTPException(status_code=400, detail="Only http(s) document URLs are supported.")

    loop = asyncio.get_event_loop()

    try:
        logger.info(f"{prefix} 📥 Downloading remote document...")
        content = await loop.run_in_executor(
            executor,
            safe_fetch,
            req.documents,
            MAX_DOWNLOAD_BYTES
        )
    except ValueError as e:
        logger.warning(f"{prefix} Blocked document URL {req.documents!r}: {e}")
        raise HTTPException(status_code=400, detail="The provided document URL is not allowed.")
    except requests.RequestException as e:
        logger.error(f"{prefix} Failed to download remote document: {e}")
        raise HTTPException(status_code=400, detail="Could not process the document URL.")

    filename_hint = os.path.basename(urlparse(req.documents).path) or None
    document_id, chunk_count = await _ingest_bytes(loop, content, req.documents, "url", request_id, filename_hint)

    answers = await _answer_questions(loop, req.questions, document_id, cache_namespace=req.documents, request_id=request_id)

    logger.info(f"{_log_prefix(request_id, document_id)} ✅ All questions answered")
    return QueryResponse(answers=answers)


# === Upload Endpoint (multipart, any supported format) ===
@app.post("/documents/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...), _=Depends(verify_token)):
    """
    Upload a document file directly (PDF, DOCX, XLSX, or CSV). The format is
    verified from its actual bytes, not trusted from the filename or the
    browser-supplied content-type header.
    """
    request_id = uuid.uuid4().hex[:12]
    content = await file.read()

    if len(content) > MAX_DOWNLOAD_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"File exceeds the maximum allowed size of {MAX_DOWNLOAD_BYTES} bytes."
        )

    file_type = detect_file_type(content, file.filename)
    if file_type == "unknown":
        raise HTTPException(status_code=400, detail="Unsupported or unrecognized file format.")

    loop = asyncio.get_event_loop()
    source_uri = file.filename or f"upload-{compute_bytes_hash(content)[:12]}"

    document_id, chunk_count = await _ingest_bytes(loop, content, source_uri, "upload", request_id, file.filename)

    return UploadResponse(document_id=document_id, chunks_stored=chunk_count)


# === Ask Endpoint (question a previously ingested document by its id) ===
@app.post("/documents/{document_id}/ask", response_model=QueryResponse)
async def ask_document(document_id: str, req: AskRequest, _=Depends(verify_token)):
    """Answer questions against a document previously ingested via /hackrx/run or /documents/upload."""
    request_id = uuid.uuid4().hex[:12]
    loop = asyncio.get_event_loop()
    answers = await _answer_questions(loop, req.questions, document_id, cache_namespace=document_id, request_id=request_id)
    return QueryResponse(answers=answers)

# === Root Endpoint ===
@app.get("/")
async def root():
    return {
        "message": "DocuQueryAI - Production-Ready RAG System", 
        "version": "3.0.0",
        "features": [
            "Async processing",
            "GPU acceleration",
            "Intelligent caching",
            "Batch embedding",
            "Connection pooling",
            "Deduplication"
        ]
    }


# === Cache Statistics Endpoint ===
@app.get("/stats")
async def get_stats(_=Depends(verify_token)):
    """Get system statistics including cache performance."""
    doc_count = await asyncio.get_event_loop().run_in_executor(
        executor, vector_store.count
    )
    
    return {
        "total_chunks": doc_count,
        "embedding_cache": get_cache_stats(),
        "query_cache": query_cache.stats(),
        "token_usage": get_token_usage_stats()
    }


# === Clear Cache Endpoint ===
@app.post("/cache/clear")
async def clear_cache(_=Depends(verify_token)):
    """Clear all caches."""
    embedding_cache.clear()
    query_cache.clear()
    logger.info("✅ All caches cleared")
    return {"message": "Caches cleared successfully"}