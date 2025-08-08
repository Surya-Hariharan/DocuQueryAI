import os
import sys
import logging
import tempfile
import requests
import uvicorn

# Add root directory to Python path so we can import from parent folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

from config import GROQ_API_KEY, BEARER_TOKEN, TOP_K_CHUNKS
from parser import parse_single_pdf_file
from answer_generator import generate_answer
from db_vector_store import init_db, upsert_chunks, query_top_k_chunks

# === Logging Setup ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

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
    title="HackRx Insurance Query System",
    description="LLM-powered insurance QA using Groq API and pgvector",
    version="2.0.0"
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

# === Startup: Initialize DB ===
@app.on_event("startup")
async def on_startup():
    init_db()
    logger.info("pgvector database initialized.")

# === Health Check Endpoint ===
@app.get("/health", response_model=HealthResponse)
async def health_check():
    status = "healthy"
    try:
        _ = query_top_k_chunks("test", k=1)  # Dummy vector test
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        status = "degraded"
    return HealthResponse(status=status, details={"pgvector": status})

# === Main PDF QA Endpoint ===
@app.post("/hackrx/run", response_model=QueryResponse)
async def process_queries(req: QueryRequest, _=Depends(verify_token)):
    logger.info(f"Received {len(req.questions)} questions for document: {req.documents}")

    # === 1. Handle Remote PDF ===
    if not req.documents.startswith("http"):
        raise HTTPException(status_code=400, detail="Only online PDF URLs are supported.")

    try:
        logger.info("Downloading remote PDF...")
        response = requests.get(req.documents)
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(response.content)
            tmp_file_path = tmp_file.name

        chunks = parse_single_pdf_file(tmp_file_path)
        os.unlink(tmp_file_path)
        logger.info("Remote PDF parsed successfully.")
    except Exception as e:
        logger.error(f"Failed to download/parse remote PDF: {str(e)}")
        raise HTTPException(status_code=400, detail="Could not process online PDF.")

    # === 2. Store Embeddings in DB ===
    upsert_chunks(chunks, req.documents)

    # === 3. Answer Each Question ===
    answers = []
    for question in req.questions:
        if not question.strip():
            answers.append("Please provide a valid question.")
            continue

        top_chunks = query_top_k_chunks(question, k=TOP_K_CHUNKS)
        context = " ".join(top_chunks)
        answer = generate_answer(question, context)
        answers.append(answer)

    logger.info("All questions answered.")
    return QueryResponse(answers=answers)

# === Root Endpoint ===
@app.get("/")
async def root():
    return {"message": "HackRx Insurance Query System is live.", "version": "2.0.0"}