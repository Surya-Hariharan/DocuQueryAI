"""
Tests for parse_document()'s controlled-error contract: every failure mode
it can produce must be a ValueError with a clear, specific message — never
a raw exception from whatever library a parser happens to use internally.
Callers up the stack (api/main.py's _ingest_bytes) rely on ValueError being
the one signal that means "clean 400, not a 500."
"""

import io
import zipfile

import pytest

pytest.importorskip("transformers", reason="chunking.py loads a HF tokenizer at import time")

from docuqueryai.ingestion.pipeline import parse_document


def test_unrecognized_format_raises_value_error():
    with pytest.raises(ValueError, match="format"):
        parse_document(b"random unrecognizable bytes", filename="file.xyz")


def test_empty_bytes_raise_value_error():
    with pytest.raises(ValueError):
        parse_document(b"", filename="empty.xyz")


def test_password_protected_pdf_raises_value_error_with_clear_message():
    pytest.importorskip("PyPDF2")
    import PyPDF2

    writer = PyPDF2.PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.encrypt(user_password="secret")
    buf = io.BytesIO()
    writer.write(buf)

    with pytest.raises(ValueError, match="password"):
        parse_document(buf.getvalue(), filename="encrypted.pdf")


def test_corrupted_docx_raises_value_error_not_a_raw_library_exception():
    # A ZIP with the right magic bytes and a "word/" member (so it's
    # correctly detected as DOCX) but invalid internal structure — this
    # used to leak a raw KeyError from python-docx's zip handling.
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("word/document.xml", "not valid docx xml at all <<<")

    with pytest.raises(ValueError, match="DOCX"):
        parse_document(buf.getvalue(), filename="corrupt.docx")


def test_corrupted_xlsx_raises_value_error_not_a_raw_library_exception():
    pytest.importorskip("openpyxl")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("xl/workbook.xml", "not valid workbook xml at all <<<")

    with pytest.raises(ValueError, match="XLSX"):
        parse_document(buf.getvalue(), filename="corrupt.xlsx")


def test_genuinely_corrupt_pdf_degrades_silently_not_an_error():
    # Unlike DOCX/XLSX, a corrupt (not encrypted) PDF has an existing,
    # deliberate silent-degrade contract (empty result, no exception) —
    # this pipeline-level fix must not change that.
    chunks = parse_document(b"%PDF-1.4\nthis is not a real pdf body", filename="corrupt.pdf")
    assert chunks == []
