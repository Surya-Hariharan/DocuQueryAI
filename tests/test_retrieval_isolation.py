"""
Integration test for the single highest-impact bug the audit found:
retrieval must never return chunks from a document other than the one
being queried.

This needs a real, reachable PostgreSQL + pgvector instance and the full
ML dependency stack (sentence-transformers/torch) to embed real text — it
is NOT run by default. To run it:

    1. Point DB_HOST/DB_NAME/DB_USER/DB_PASSWORD (or DATABASE_URL) at a
       disposable test database — never your production one, since this
       test truncates its tables.
    2. Set RUN_DB_INTEGRATION_TESTS=1
    3. pytest tests/test_retrieval_isolation.py -v

It's kept separate from the rest of the suite specifically so the fast,
dependency-light tests never require a live database to run.
"""

import os

import pytest

if not os.getenv("RUN_DB_INTEGRATION_TESTS"):
    pytest.skip(
        "DB integration tests are opt-in — set RUN_DB_INTEGRATION_TESTS=1 against a disposable test database to run them",
        allow_module_level=True
    )

pytest.importorskip("psycopg2", reason="requires the full app dependency stack")
pytest.importorskip("sentence_transformers", reason="requires the full app dependency stack")

from src.retrieval.pg_vector_store import PgVectorStore  # noqa: E402
from src.ingestion.document_model import Chunk  # noqa: E402


@pytest.fixture()
def store():
    s = PgVectorStore()
    s.init_connection_pool(minconn=1, maxconn=2)
    s.init_schema()
    s.clear()
    yield s
    s.clear()


def test_query_scoped_to_one_document_never_returns_the_other(store):
    doc_a_id = store.get_or_create_document("test://document-a.pdf")
    doc_b_id = store.get_or_create_document("test://document-b.pdf")

    store.upsert_chunks([Chunk(text="The quarterly revenue for the widget division was $4.2 million.")], doc_a_id)
    store.upsert_chunks([Chunk(text="The company's headquarters relocated to Austin in March.")], doc_b_id)

    results_a = store.query_top_k("widget division revenue", k=5, document_id=doc_a_id)
    results_b = store.query_top_k("headquarters relocation", k=5, document_id=doc_b_id)

    assert all(r.document_id == doc_a_id for r in results_a)
    assert all(r.document_id == doc_b_id for r in results_b)
    assert not any("widget" in r.text.lower() for r in results_b)
    assert not any("austin" in r.text.lower() for r in results_a)


def test_dedup_is_scoped_per_document_not_global(store):
    # Identical chunk text in two different documents used to collapse into
    # one row under the old global UNIQUE(text_hash) constraint — the
    # second document's copy was silently dropped.
    doc_a_id = store.get_or_create_document("test://dup-a.pdf")
    doc_b_id = store.get_or_create_document("test://dup-b.pdf")

    identical_text = "Section 12: Force majeure clause applies to natural disasters."
    store.upsert_chunks([Chunk(text=identical_text)], doc_a_id)
    store.upsert_chunks([Chunk(text=identical_text)], doc_b_id)

    results_a = store.query_top_k("force majeure", k=5, document_id=doc_a_id)
    results_b = store.query_top_k("force majeure", k=5, document_id=doc_b_id)

    assert len(results_a) == 1
    assert len(results_b) == 1
