import os
import re
import logging
from typing import List, Tuple
import PyPDF2
from transformers import AutoTokenizer
from config import PDF_FOLDER, CHUNK_SIZE
from db_vector_store import upsert_chunks
from utils import monitor_performance

# === Logging Setup ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("parser")

# === Chunking Config ===
CHUNK_OVERLAP = 100
MIN_CHUNK_LENGTH = 50

# === Load Tokenizer for Accurate Token Counting ===
MODEL_NAME = "intfloat/e5-small-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
MAX_TOKENS_PER_CHUNK = 512  # Model's max sequence length

# === Extract Text from PDF File ===
@monitor_performance("extract_text_from_pdf")
def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF file with error handling."""
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for i, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                except Exception as page_err:
                    logger.warning(f"Failed to extract page {i} in {pdf_path}: {page_err}")
    except Exception as e:
        logger.error(f"Could not open PDF file {pdf_path}: {e}")
    return text.strip()

# === Count Tokens Using Model's Tokenizer ===
def count_tokens(text: str) -> int:
    """Count actual tokens using the embedding model's tokenizer."""
    return len(tokenizer.encode(text, add_special_tokens=True))

# === Split Text into Sentence-Aware, Token-Counted Chunks ===
@monitor_performance("split_text_into_chunks")
def split_text_into_chunks(
    text: str,
    max_tokens: int = MAX_TOKENS_PER_CHUNK - 50,  # Leave room for "passage:" prefix
    overlap_tokens: int = 50
) -> List[str]:
    """
    Split text into chunks based on token count with sentence-aware boundaries.
    
    Args:
        text: Input text to chunk
        max_tokens: Maximum tokens per chunk
        overlap_tokens: Number of overlapping tokens between chunks
    
    Returns:
        List of text chunks
    """
    # Split into paragraphs first
    paragraphs = re.split(r'\n{2,}', text)
    chunks = []
    current_chunk = ""
    current_tokens = 0
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        # Split paragraph into sentences
        sentences = re.split(r'(?<=[.!?])\s+', para)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_tokens = count_tokens(sentence)
            
            # If single sentence exceeds max, split it by words
            if sentence_tokens > max_tokens:
                words = sentence.split()
                for word in words:
                    word_with_space = word + " "
                    word_tokens = count_tokens(word_with_space)
                    
                    if current_tokens + word_tokens > max_tokens:
                        if current_chunk.strip():
                            chunks.append(current_chunk.strip())
                        
                        # Start new chunk with overlap
                        if overlap_tokens > 0 and current_chunk:
                            overlap_text = get_last_n_tokens(current_chunk, overlap_tokens)
                            current_chunk = overlap_text + word_with_space
                            current_tokens = count_tokens(current_chunk)
                        else:
                            current_chunk = word_with_space
                            current_tokens = word_tokens
                    else:
                        current_chunk += word_with_space
                        current_tokens += word_tokens
            
            # Normal sentence processing
            elif current_tokens + sentence_tokens > max_tokens:
                # Save current chunk
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap
                if overlap_tokens > 0 and current_chunk:
                    overlap_text = get_last_n_tokens(current_chunk, overlap_tokens)
                    current_chunk = overlap_text + sentence + " "
                    current_tokens = count_tokens(current_chunk)
                else:
                    current_chunk = sentence + " "
                    current_tokens = sentence_tokens
            else:
                current_chunk += sentence + " "
                current_tokens += sentence_tokens
    
    # Add final chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    # Filter chunks by minimum length
    chunks = [chunk for chunk in chunks if count_tokens(chunk) >= 10]
    
    logger.info(f"Created {len(chunks)} chunks with avg {sum(count_tokens(c) for c in chunks) / len(chunks):.1f} tokens" if chunks else "No chunks created")
    
    return chunks

# === Get Last N Tokens from Text ===
def get_last_n_tokens(text: str, n_tokens: int) -> str:
    """Extract the last n tokens from text for overlap."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= n_tokens:
        return text
    
    overlap_tokens = tokens[-n_tokens:]
    overlap_text = tokenizer.decode(overlap_tokens, skip_special_tokens=True)
    return overlap_text + " "

# === Process All PDFs in a Folder ===
@monitor_performance("parse_pdfs_in_folder")
def parse_pdfs_in_folder(folder_path: str = PDF_FOLDER) -> Tuple[List[str], List[str]]:
# === Process All PDFs in a Folder ===
@monitor_performance("parse_pdfs_in_folder")
def parse_pdfs_in_folder(folder_path: str = PDF_FOLDER) -> Tuple[List[str], List[str]]:
    """Process all PDF files in a folder and store chunks in database."""
    all_chunks = []
    doc_ids = []

    if not os.path.exists(folder_path):
        logger.warning(f"Folder '{folder_path}' does not exist.")
        return [], []

    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]
    logger.info(f"Found {len(pdf_files)} PDFs in '{folder_path}'")

    for fname in pdf_files:
        path = os.path.join(folder_path, fname)
        logger.info(f"Processing: {fname}")
        text = extract_text_from_pdf(path)

        if not text:
            logger.warning(f"No text extracted in: {fname}")
            continue

        chunks = split_text_into_chunks(text)

        document_name = os.path.basename(path)
        upsert_chunks(chunks, document_name)

        all_chunks.extend(chunks)
        doc_ids.extend([fname] * len(chunks))

        logger.info(f"✅ {fname}: {len(chunks)} chunks extracted and stored")

    return all_chunks, doc_ids

# === Process a Single PDF File (e.g., downloaded from URL) ===
@monitor_performance("parse_single_pdf_file")
def parse_single_pdf_file(file_path: str) -> List[str]:
    """Parse a single PDF file and return chunks without storing."""
    logger.info(f"Parsing single PDF file: {file_path}")
    text = extract_text_from_pdf(file_path)

    if not text:
        logger.warning("No text found in single PDF.")
        return []

    chunks = split_text_into_chunks(text)
    logger.info(f"✅ Extracted {len(chunks)} chunks from single PDF")
    
    return chunks

    document_name = os.path.basename(file_path)
    upsert_chunks(chunks, document_name)

    logger.info(f"Extracted {len(chunks)} chunks from single PDF and stored")
    return chunks