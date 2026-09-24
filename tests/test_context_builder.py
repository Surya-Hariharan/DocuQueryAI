from src.retrieval.context_builder import build_context
from src.ingestion.document_model import RetrievedChunk


def _chunk(text, document_id="doc1", source_uri="policy.pdf", page_number=None, sheet_name=None):
    return RetrievedChunk(text=text, document_id=document_id, source_uri=source_uri, page_number=page_number, sheet_name=sheet_name)


def test_builds_labeled_sources_in_order():
    chunks = [
        _chunk("first chunk text", page_number=1),
        _chunk("second chunk text", page_number=2),
    ]
    context, sources = build_context(chunks, max_tokens=10_000)

    assert "[Source 1 — policy.pdf, page 1]" in context
    assert "[Source 2 — policy.pdf, page 2]" in context
    assert context.index("Source 1") < context.index("Source 2")
    assert "first chunk text" in context
    assert "second chunk text" in context


def test_sources_list_matches_context_tags():
    chunks = [_chunk("text a", page_number=3), _chunk("text b", sheet_name="Q4")]
    _, sources = build_context(chunks, max_tokens=10_000)

    assert sources[0]["source"] == 1
    assert sources[0]["page_number"] == 3
    assert sources[1]["source"] == 2
    assert sources[1]["sheet_name"] == "Q4"


def test_falls_back_to_document_id_when_no_source_uri():
    chunks = [_chunk("some text", document_id="abc123", source_uri=None)]
    context, sources = build_context(chunks, max_tokens=10_000)

    assert "abc123" in context
    assert sources[0]["source_uri"] is None
    assert sources[0]["document_id"] == "abc123"


def test_empty_chunk_list_returns_empty_context():
    context, sources = build_context([], max_tokens=1000)
    assert context == ""
    assert sources == []


def test_truncates_by_token_budget_but_always_includes_first_chunk():
    # A budget too small for even one block should still include the first
    # chunk (avoids returning zero context just because the very first
    # retrieved chunk alone exceeds a tiny budget).
    long_chunk = _chunk("word " * 500)
    context, sources = build_context([long_chunk], max_tokens=1)
    assert len(sources) == 1

    # But a second chunk that would overflow the budget must be dropped.
    chunks = [_chunk("word " * 500), _chunk("this should be dropped")]
    context, sources = build_context(chunks, max_tokens=1)
    assert len(sources) == 1
    assert "this should be dropped" not in context
