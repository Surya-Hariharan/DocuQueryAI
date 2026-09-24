"""
Assembles retrieved chunks into a source-tagged prompt context — replacing
the old bare `" ".join(top_chunks)`, which gave the LLM no way to tell
where one retrieved passage ended and another began, let alone which
document either came from.
"""

from typing import List, Tuple

from docuqueryai.config import MAX_CONTEXT_TOKENS
from docuqueryai.ingestion.document_model import RetrievedChunk


def build_context(chunks: List[RetrievedChunk], max_tokens: int = MAX_CONTEXT_TOKENS) -> Tuple[str, List[dict]]:
    """
    Build a source-tagged context string, truncated by an approximate token
    budget (see config.MAX_CONTEXT_TOKENS for why this is an estimate, not
    an exact count for the generation model).

    Returns (context_text, sources), where `sources` is a list of
    {source, document_id, source_uri, page_number, sheet_name} dicts — one
    per [Source N] tag actually included in the context — so the API layer
    can return real citations instead of discarding this metadata.
    """
    blocks = []
    sources = []
    total_tokens = 0

    for i, chunk in enumerate(chunks, start=1):
        label = _label_for(i, chunk)
        block = f"{label}\n{chunk.text}"
        block_tokens = _approx_tokens(block)

        if blocks and total_tokens + block_tokens > max_tokens:
            break

        blocks.append(block)
        sources.append({
            "source": i,
            "document_id": chunk.document_id,
            "source_uri": chunk.source_uri,
            "page_number": chunk.page_number,
            "sheet_name": chunk.sheet_name,
        })
        total_tokens += block_tokens

    context_text = "\n\n---\n\n".join(blocks)
    return context_text, sources


def _label_for(index: int, chunk: RetrievedChunk) -> str:
    location = [chunk.source_uri or chunk.document_id]
    if chunk.page_number:
        location.append(f"page {chunk.page_number}")
    if chunk.sheet_name:
        location.append(f"sheet '{chunk.sheet_name}'")
    return f"[Source {index} — {', '.join(location)}]"


def _approx_tokens(text: str) -> int:
    """
    Rough token estimate (~4 characters/token for English) rather than an
    exact count — we don't have a local tokenizer for the Groq generation
    model, and reusing the embedding model's tokenizer here would measure
    the wrong vocabulary while looking precise. Good enough to keep the
    prompt well within budget without claiming false accuracy.
    """
    return max(1, len(text) // 4)
