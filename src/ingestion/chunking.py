import re
import logging
from typing import List, BinaryIO
import PyPDF2
from transformers import AutoTokenizer
from src.utils import monitor_performance

# === Logging Setup ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chunking")

# === Load Tokenizer for Accurate Token Counting ===
MODEL_NAME = "intfloat/e5-small-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
MAX_TOKENS_PER_CHUNK = 512  # Model's max sequence length

# === Extract Per-Page Text from an Open PDF Stream ===
@monitor_performance("extract_pages_from_pdf")
def extract_pages_from_pdf(stream: BinaryIO) -> List[str]:
    """
    Extract text page-by-page from an already-open PDF stream (a file
    handle or an in-memory BytesIO — anything PyPDF2.PdfReader accepts).

    Returns one string per page (empty string for a page that failed to
    extract), so callers can still attach a page number to each result
    even when some pages come back blank.
    """
    pages: List[str] = []
    try:
        reader = PyPDF2.PdfReader(stream)
    except Exception as e:
        logger.error(f"Could not read PDF stream: {e}")
        return pages

    for i, page in enumerate(reader.pages):
        try:
            page_text = page.extract_text() or ""
        except Exception as page_err:
            logger.warning(f"Failed to extract page {i}: {page_err}")
            page_text = ""
        pages.append(page_text)

    return pages

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

    This is the shared leaf-level chunker: every format-specific parser
    (PDF, DOCX, XLSX, CSV) hands its section text through this same function
    rather than each reimplementing chunking.

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
