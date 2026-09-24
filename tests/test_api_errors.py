"""
Tests malformed/problematic documents through the actual HTTP API boundary
(not just by calling parser functions directly), confirming the API always
returns a controlled JSON error — never a raw Python traceback — for bad
input. The database layer is mocked out (no real Postgres needed); the
parsing/chunking layer is exercised for real.
"""

import io

import pytest

pytest.importorskip("fastapi", reason="requires the full app dependency stack")
pytest.importorskip("psycopg2", reason="requires the full app dependency stack")
pytest.importorskip("sentence_transformers", reason="requires the full app dependency stack")

from unittest.mock import patch

from fastapi.testclient import TestClient

import src.api.main as main

client = TestClient(main.app)
AUTH = {"Authorization": f"Bearer {main.BEARER_TOKEN}"}


def _upload(filename: str, content: bytes):
    return client.post(
        "/documents/upload",
        headers=AUTH,
        files={"file": (filename, content)}
    )


class TestUploadEndpointErrorHandling:
    def test_unsupported_file_type_returns_clean_400(self):
        response = _upload("notes.xyz", b"whatever this is, it's not a supported format")
        assert response.status_code == 400
        body = response.json()
        assert "detail" in body
        assert "Traceback" not in body["detail"]
        assert "File " not in body["detail"]  # not a Python stack frame line

    def test_empty_upload_returns_clean_400(self):
        response = _upload("empty.pdf", b"")
        assert response.status_code == 400

    def test_oversized_upload_returns_clean_400(self):
        from src.config import MAX_DOWNLOAD_BYTES
        oversized = b"%PDF-1.4\n" + (b"x" * (MAX_DOWNLOAD_BYTES + 1))
        response = _upload("huge.pdf", oversized)
        assert response.status_code == 400
        assert "size" in response.json()["detail"].lower()

    @patch.object(main.vector_store, "get_or_create_document", return_value="fake-doc-id")
    def test_password_protected_pdf_returns_clean_400_not_500(self, _mock_doc):
        import PyPDF2
        writer = PyPDF2.PdfWriter()
        writer.add_blank_page(width=200, height=200)
        writer.encrypt(user_password="secret")
        buf = io.BytesIO()
        writer.write(buf)

        response = _upload("encrypted.pdf", buf.getvalue())
        assert response.status_code == 400
        assert "password" in response.json()["detail"].lower()

    @patch.object(main.vector_store, "get_or_create_document", return_value="fake-doc-id")
    def test_corrupted_docx_returns_clean_400_not_500(self, _mock_doc):
        import zipfile
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("word/document.xml", "not valid docx xml <<<")

        response = _upload("corrupt.docx", buf.getvalue())
        assert response.status_code == 400
        body = response.json()
        assert "detail" in body
        assert "KeyError" not in body["detail"]  # the raw library exception must not leak through

    @patch.object(main.vector_store, "upsert_chunks", return_value=None)
    @patch.object(main.vector_store, "get_or_create_document", return_value="fake-doc-id")
    def test_genuinely_corrupt_pdf_degrades_to_200_with_zero_chunks(self, _mock_doc, _mock_upsert):
        # Matches the existing, deliberate silent-degrade contract for a
        # corrupt-but-not-encrypted PDF (see test_pipeline.py) — the API
        # should reflect that as a successful upload with nothing extracted,
        # not an error, since there's no way to distinguish "corrupt" from
        # "genuinely empty" at this layer.
        response = _upload("corrupt.pdf", b"%PDF-1.4\nnot a real pdf body")
        assert response.status_code == 200
        assert response.json()["chunks_stored"] == 0


class TestAskEndpointRequiresAuth:
    def test_missing_bearer_token_is_rejected(self):
        response = client.post("/documents/some-id/ask", json={"questions": ["What?"]})
        assert response.status_code == 401

    def test_wrong_bearer_token_is_rejected(self):
        response = client.post(
            "/documents/some-id/ask",
            headers={"Authorization": "Bearer wrong-token"},
            json={"questions": ["What?"]}
        )
        assert response.status_code == 401
