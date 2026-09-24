import os
import sys
import logging
import tempfile
import asyncio
from concurrent.futures import ThreadPoolExecutor
import requests
import uvicorn

# Add root directory to Python path so we can import from parent folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, Request, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

from config import GROQ_API_KEY, BEARER_TOKEN, TOP_K_CHUNKS
from parser import parse_single_pdf_file
from answer_generator import generate_answer
from db_vector_store import init_db, upsert_chunks, query_top_k_chunks, get_document_count, init_connection_pool
from utils import monitor_performance, query_cache, compute_text_hash, embedding_cache
from embeddings import get_cache_stats

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
    documents: str  # URL of the PDF
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

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
        init_connection_pool(minconn=2, maxconn=10)
        init_db()
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
            executor, get_document_count
        )
        details["database"] = "healthy"
        details["total_chunks"] = doc_count
        
        # Get cache statistics
        emb_stats = get_cache_stats()
        details["embedding_cache"] = emb_stats
        details["query_cache"] = query_cache.stats()
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        status = "degraded"
        details["error"] = str(e)
    
    return HealthResponse(status=status, details=details)


# === Background Task: Process Document ===
async def process_document_background(pdf_url: str, document_id: str):
    """Background task to process and store document chunks."""
    try:
        logger.info(f"📥 Background processing: {document_id}")
        
        # Download PDF
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            executor,
            lambda: requests.get(pdf_url)
        )
        response.raise_for_status()
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(response.content)
            tmp_file_path = tmp_file.name
        
        # Parse PDF
        chunks = await loop.run_in_executor(
            executor,
            parse_single_pdf_file,
            tmp_file_path
        )
        os.unlink(tmp_file_path)
        
        # Store in database
        if chunks:
            await loop.run_in_executor(
                executor,
                upsert_chunks,
                chunks,
                document_id
            )
            logger.info(f"✅ Background processing complete: {document_id} ({len(chunks)} chunks)")
        else:
            logger.warning(f"No chunks extracted from {document_id}")
            
    except Exception as e:
        logger.error(f"Background processing failed for {document_id}: {e}")

# === Main PDF QA Endpoint (Async with Caching) ===
@app.post("/hackrx/run", response_model=QueryResponse)
@monitor_performance("hackrx_run_endpoint")
async def process_queries(req: QueryRequest, _=Depends(verify_token)):
    """
    Async endpoint to process PDF queries with caching and background processing.
    """
    logger.info(f"📨 Received {len(req.questions)} questions for document: {req.documents}")

    # === 1. Validate PDF URL ===
    if not req.documents.startswith("http"):
        raise HTTPException(status_code=400, detail="Only online PDF URLs are supported.")

    # === 2. Download and Parse PDF (Async) ===
    try:
        logger.info("📥 Downloading remote PDF...")
        loop = asyncio.get_event_loop()
        
        # Download in thread pool
        response = await loop.run_in_executor(
            executor,
            lambda: requests.get(req.documents)
        )
        response.raise_for_status()

        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(response.content)
            tmp_file_path = tmp_file.name

        # Parse PDF in thread pool
        chunks = await loop.run_in_executor(
            executor,
            parse_single_pdf_file,
            tmp_file_path
        )
        os.unlink(tmp_file_path)
        logger.info(f"✅ Remote PDF parsed successfully ({len(chunks)} chunks)")
        
    except Exception as e:
        logger.error(f"Failed to download/parse remote PDF: {str(e)}")
        raise HTTPException(status_code=400, detail="Could not process online PDF.")

    # === 3. Store Embeddings in DB (Async) ===
    if chunks:
        await loop.run_in_executor(
            executor,
            upsert_chunks,
            chunks,
            req.documents
        )
        logger.info("✅ Chunks stored in database")

    # === 4. Answer Questions with Caching ===
    answers = []
    
    # Process questions concurrently
    async def answer_question(question: str) -> str:
        if not question.strip():
            return "Please provide a valid question."
        
        # Check query cache
        cache_key = compute_text_hash(f"{req.documents}:{question}")
        cached_answer = query_cache.get(cache_key)
        if cached_answer:
            logger.info(f"🎯 Cache hit for question: {question[:50]}...")
            return cached_answer
        
        # Retrieve relevant chunks
        top_chunks = await loop.run_in_executor(
            executor,
            query_top_k_chunks,
            question,
            TOP_K_CHUNKS
        )
        
        # Generate answer
        context = " ".join(top_chunks)
        answer = await loop.run_in_executor(
            executor,
            generate_answer,
            question,
            context
        )
        
        # Cache the answer
        query_cache.put(cache_key, answer)
        
        return answer
    
    # Process all questions concurrently
    answers = await asyncio.gather(*[answer_question(q) for q in req.questions])

    logger.info("✅ All questions answered")
    return QueryResponse(answers=list(answers))

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
        executor, get_document_count
    )
    
    return {
        "total_chunks": doc_count,
        "embedding_cache": get_cache_stats(),
        "query_cache": query_cache.stats()
    }


# === Clear Cache Endpoint ===
@app.post("/cache/clear")
async def clear_cache(_=Depends(verify_token)):
    """Clear all caches."""
    embedding_cache.clear()
    query_cache.clear()
    logger.info("✅ All caches cleared")
    return {"message": "Caches cleared successfully"}