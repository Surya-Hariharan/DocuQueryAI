import os
import re
import logging
from typing import List, Tuple
import PyPDF2
from config import PDF_FOLDER, CHUNK_SIZE
from db_vector_store import upsert_chunks

# === Logging Setup ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("parser")

# === Chunking Config ===
CHUNK_OVERLAP = 100
MIN_CHUNK_LENGTH = 50

# === Extract Text from PDF File ===
def extract_text_from_pdf(pdf_path: str) -> str:
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

# === Split Text into Overlapping Chunks ===
def split_text_into_chunks(text: str, max_length: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    paragraphs = re.split(r'\n{2,}', text)
    chunks = []

    for para in paragraphs:
        sentences = re.split(r'(?<=[.!?]) +', para.strip())
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= max_length:
                current_chunk += sentence + " "
            else:
                chunks.append(current_chunk.strip())
                overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                current_chunk = overlap_text + sentence + " "

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

    return [chunk for chunk in chunks if len(chunk) >= MIN_CHUNK_LENGTH]

# === Process All PDFs in a Folder ===
def parse_pdfs_in_folder(folder_path: str = PDF_FOLDER) -> Tuple[List[str], List[str]]:
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

        logger.info(f"{fname}: {len(chunks)} chunks extracted and stored")

    return all_chunks, doc_ids

# === Process a Single PDF File (e.g., downloaded from URL) ===
def parse_single_pdf_file(file_path: str) -> List[str]:
    logger.info(f"Parsing single PDF file: {file_path}")
    text = extract_text_from_pdf(file_path)

    if not text:
        logger.warning("No text found in single PDF.")
        return []

    chunks = split_text_into_chunks(text)

    document_name = os.path.basename(file_path)
    upsert_chunks(chunks, document_name)

    logger.info(f"Extracted {len(chunks)} chunks from single PDF and stored")
    return chunks