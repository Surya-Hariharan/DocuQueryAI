"""
The single embedding entrypoint for the whole application.

Every other module that needs a vector — ingestion, retrieval, and any
future vector-store backend — imports from here instead of loading its own
copy of the model. This is deliberate: previously three separate modules
each instantiated their own SentenceTransformer("intfloat/e5-small-v2"),
with inconsistent (or missing) query/passage prefixing and caching.

intfloat/e5-small-v2 is an asymmetric retrieval model: it expects a
"query: " prefix on questions and a "passage: " prefix on document text.
Using the same encoding for both (as the old code did) measurably hurts
retrieval quality, so embed_query() and embed_passage() are kept distinct
rather than exposing one generic embed(text) function.
"""

import logging
from typing import List, Optional

from sentence_transformers import SentenceTransformer

from docuqueryai.config import USE_GPU, BATCH_SIZE
from docuqueryai.utils import embedding_cache, compute_text_hash, monitor_performance

logger = logging.getLogger("embeddings")

MODEL_NAME = "intfloat/e5-small-v2"

_requested_device = "cuda" if USE_GPU else "cpu"
try:
    model = SentenceTransformer(MODEL_NAME, device=_requested_device)
except Exception as e:
    if _requested_device != "cpu":
        logger.warning(f"Could not load embedding model on '{_requested_device}' ({e}); falling back to CPU")
        model = SentenceTransformer(MODEL_NAME, device="cpu")
    else:
        raise

logger.info(f"🚀 Embedding model '{MODEL_NAME}' loaded on device: {model.device}")


def _cache_key(prefixed_text: str) -> str:
    return compute_text_hash(prefixed_text)


def _embed_single(prefixed_text: str, use_cache: bool) -> list:
    if use_cache:
        cached = embedding_cache.get(_cache_key(prefixed_text))
        if cached is not None:
            return cached

    embedding = model.encode(prefixed_text, normalize_embeddings=True).tolist()

    if use_cache:
        embedding_cache.put(_cache_key(prefixed_text), embedding)

    return embedding


@monitor_performance("embed_query")
def embed_query(text: str, use_cache: bool = True) -> list:
    """Embed a user question, using E5's 'query: ' convention."""
    return _embed_single(f"query: {text.strip()}", use_cache)


@monitor_performance("embed_passage")
def embed_passage(text: str, use_cache: bool = True) -> list:
    """Embed a single document chunk, using E5's 'passage: ' convention."""
    return _embed_single(f"passage: {text.strip()}", use_cache)


@monitor_performance("embed_passages_batch")
def embed_passages_batch(
    texts: List[str],
    batch_size: Optional[int] = None,
    use_cache: bool = True
) -> List[list]:
    """
    Embed many document chunks efficiently: cache hits are served without
    touching the model, and the rest are encoded in one batched call rather
    than once per chunk.
    """
    batch_size = batch_size or BATCH_SIZE
    prefixed_texts = [f"passage: {t.strip()}" for t in texts]

    results: List[Optional[list]] = [None] * len(texts)
    to_compute_texts = []
    to_compute_indices = []

    for idx, prefixed in enumerate(prefixed_texts):
        cached = embedding_cache.get(_cache_key(prefixed)) if use_cache else None
        if cached is not None:
            results[idx] = cached
        else:
            to_compute_texts.append(prefixed)
            to_compute_indices.append(idx)

    if to_compute_texts:
        computed = model.encode(
            to_compute_texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        for idx, prefixed, emb in zip(to_compute_indices, to_compute_texts, computed):
            emb_list = emb.tolist()
            results[idx] = emb_list
            if use_cache:
                embedding_cache.put(_cache_key(prefixed), emb_list)

    return results


def get_cache_stats() -> dict:
    """Return embedding cache statistics."""
    return embedding_cache.stats()
