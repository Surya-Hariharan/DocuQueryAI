"""
Deterministic regression tests against the small, committed synthetic
fixtures in tests/fixtures/ (see generate_fixtures.py for provenance).
Complements the in-memory-built-fixture tests elsewhere in the suite with
on-disk files that don't depend on any particular test file constructing
them correctly at runtime.
"""

import os

import pytest

pytest.importorskip("transformers", reason="chunking.py loads a HF tokenizer at import time")

from src.ingestion.pipeline import parse_document
from src.ingestion.chunking import split_text_into_chunks

FIXTURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


def _read(*parts: str) -> bytes:
    with open(os.path.join(FIXTURES_DIR, *parts), "rb") as f:
        return f.read()


class TestPdfFixtures:
    def test_normal_pdf_produces_real_extractable_text(self):
        chunks = parse_document(_read("pdf", "normal.pdf"), filename="normal.pdf")
        assert len(chunks) >= 1
        joined = " ".join(c.text for c in chunks)
        assert "DocuQueryAI Test Fixture" in joined
        assert all(c.page_number == 1 for c in chunks)

    def test_empty_pdf_produces_no_chunks_without_raising(self):
        chunks = parse_document(_read("pdf", "empty.pdf"), filename="empty.pdf")
        assert chunks == []

    def test_encrypted_pdf_raises_a_clear_controlled_error(self):
        with pytest.raises(ValueError, match="password"):
            parse_document(_read("pdf", "encrypted.pdf"), filename="encrypted.pdf")


class TestDocxFixture:
    def test_structured_docx_preserves_heading_paragraphs_and_table(self):
        pytest.importorskip("docx", reason="python-docx not installed")
        chunks = parse_document(_read("docx", "structured.docx"), filename="structured.docx")

        types = [c.chunk_type for c in chunks]
        assert "heading" in types
        assert "paragraph" in types
        assert "table" in types

        heading_chunk = next(c for c in chunks if c.chunk_type == "heading")
        assert heading_chunk.text == "Q4 Financial Summary"

        table_chunk = next(c for c in chunks if c.chunk_type == "table")
        assert "Approved" in table_chunk.text


class TestShortContentTextFixture:
    """
    Plain .txt is not a supported ingestible format yet (only pdf/docx/xlsx/
    csv are registered parsers — see file_type_detector.SUPPORTED_TYPES), so
    this fixture is used as a data source for the leaf chunker directly,
    not routed through parse_document(). It documents the realistic short
    values the audit called out, in one place, rather than duplicating the
    literal list across test files.
    """

    def test_every_line_of_realistic_short_content_survives(self):
        text = _read("text", "short_content.txt").decode("utf-8")
        lines = [line for line in text.splitlines() if line.strip()]
        assert len(lines) == 8  # matches the fixture's known content

        for line in lines:
            result = split_text_into_chunks(line)
            assert result == [line], f"Expected {line!r} to survive intact, got {result!r}"
