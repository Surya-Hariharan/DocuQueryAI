"""
Regression test for the Tier-1 fix: one failing question inside a batch
must not take down the answers already computed for the others.

This module needs the full application import chain (fastapi, psycopg2,
sentence-transformers/torch via pg_vector_store -> embeddings, groq, etc.)
since docuqueryai/api/main.py wires all of it together at import time. It
will skip cleanly in a lighter environment; run it in a venv with
requirements.txt installed to actually exercise it.
"""

import asyncio
from unittest.mock import patch

import pytest

pytest.importorskip("fastapi", reason="requires the full app dependency stack")
pytest.importorskip("psycopg2", reason="requires the full app dependency stack")
pytest.importorskip("sentence_transformers", reason="requires the full app dependency stack")

import docuqueryai.api.main as main  # noqa: E402  (import must follow the importorskip guards above)
from docuqueryai.ingestion.document_model import RetrievedChunk  # noqa: E402


def _fake_chunk(text="some relevant context"):
    return RetrievedChunk(text=text, document_id="doc-1", source_uri="policy.pdf", page_number=1)


def test_one_failing_question_does_not_take_down_the_others():
    """
    3 questions: two succeed, the middle one's retrieval raises. The
    response must still contain 3 answers, with the failing one replaced
    by a friendly error message — not a 500 that discards everything.
    """
    def fake_query_top_k_sync(query, k, document_id=None, tenant_id=None):
        if "boom" in query:
            raise RuntimeError("simulated retrieval failure")
        return [_fake_chunk()]

    with patch.object(main.vector_store, "query_top_k", side_effect=fake_query_top_k_sync), \
         patch.object(main, "generate_answer", return_value="A grounded answer (Source 1)."), \
         patch.object(main, "looks_grounded", return_value=True):

        loop = asyncio.new_event_loop()
        try:
            answers = loop.run_until_complete(
                main._answer_questions(
                    loop,
                    questions=["question one", "boom question", "question three"],
                    document_id="doc-1",
                    cache_namespace="test-namespace",
                    request_id="test-req-1"
                )
            )
        finally:
            loop.close()

    assert len(answers) == 3
    assert answers[0].answer == "A grounded answer (Source 1)."
    assert "error occurred" in answers[1].answer
    assert answers[2].answer == "A grounded answer (Source 1)."


def test_empty_question_short_circuits_without_hitting_retrieval():
    with patch.object(main.vector_store, "query_top_k") as mock_query:
        loop = asyncio.new_event_loop()
        try:
            answers = loop.run_until_complete(
                main._answer_questions(
                    loop,
                    questions=["   "],
                    document_id="doc-1",
                    cache_namespace="test-namespace",
                    request_id="test-req-2"
                )
            )
        finally:
            loop.close()

    assert answers[0].answer == "Please provide a valid question."
    mock_query.assert_not_called()
