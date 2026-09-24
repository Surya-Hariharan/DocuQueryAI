"""
Parser edge-case tests. src.ingestion.parsers imports
src.ingestion.chunking, which loads the e5-small-v2 tokenizer at
module import time (a real network fetch on first run, or a local HF
cache) — so this whole module needs `transformers` available and
reachable, not just `PyPDF2`/`python-docx`/`openpyxl`. It will skip cleanly
without that; run it in a fully provisioned environment to actually
exercise it.
"""

import io

import pytest

pytest.importorskip("transformers", reason="chunking.py loads a HF tokenizer at import time")
pytest.importorskip("docx", reason="python-docx not installed")
pytest.importorskip("openpyxl", reason="openpyxl not installed")

import PyPDF2  # noqa: E402

from src.ingestion.parsers import PdfParser, DocxParser, XlsxParser, CsvParser  # noqa: E402


class TestPdfParser:
    def setup_method(self):
        self.parser = PdfParser()

    def test_empty_bytes_yields_no_sections(self):
        doc = self.parser.parse(b"")
        assert doc.sections == []
        assert doc.source_type == "pdf"

    def test_corrupt_pdf_yields_no_sections_without_raising(self):
        doc = self.parser.parse(b"%PDF-1.4\nthis is not a real pdf body")
        assert doc.sections == []

    def test_blank_page_pdf_yields_no_text_sections(self):
        writer = PyPDF2.PdfWriter()
        writer.add_blank_page(width=200, height=200)
        buf = io.BytesIO()
        writer.write(buf)

        doc = self.parser.parse(buf.getvalue())
        # A blank page has no extractable text, so no section should be
        # emitted for it (sections only include non-empty page text).
        assert doc.sections == []

    def test_password_protected_pdf_raises_a_clear_controlled_error(self):
        writer = PyPDF2.PdfWriter()
        writer.add_blank_page(width=200, height=200)
        writer.encrypt(user_password="secret")
        buf = io.BytesIO()
        writer.write(buf)

        # A PDF that genuinely requires a password is a distinct, actionable
        # failure — not the same as a merely corrupt/unreadable file — so it
        # must raise (a plain PyPDF2.errors.FileNotDecryptedError leaking
        # out uncaught was the confirmed bug; ValueError is the controlled,
        # caller-facing replacement, consistent with how the rest of the
        # ingestion pipeline signals a clean 400 rather than a 500).
        with pytest.raises(ValueError, match="password"):
            self.parser.parse(buf.getvalue())

    def test_permissions_only_encrypted_pdf_is_still_readable(self):
        # A PDF encrypted only to restrict printing/copying (an owner
        # password with an empty user password) has no real content
        # protection — PyPDF2 can decrypt it with an empty password, and it
        # must NOT be treated the same as a genuinely password-protected file.
        writer = PyPDF2.PdfWriter()
        writer.add_blank_page(width=200, height=200)
        writer.encrypt(user_password="", owner_password="ownersecret")
        buf = io.BytesIO()
        writer.write(buf)

        doc = self.parser.parse(buf.getvalue())  # should not raise
        assert doc.sections == []  # still a blank page — no text to extract, but no error either


class TestDocxParser:
    def _build_docx(self, add_table=False):
        import docx
        document = docx.Document()
        document.add_heading("Section Title", level=1)
        document.add_paragraph("This is a normal paragraph.")
        if add_table:
            table = document.add_table(rows=2, cols=2)
            table.cell(0, 0).text = "Header A"
            table.cell(0, 1).text = "Header B"
            table.cell(1, 0).text = "Value 1"
            table.cell(1, 1).text = "Value 2"
        buf = io.BytesIO()
        document.save(buf)
        return buf.getvalue()

    def test_heading_and_paragraph_preserved_as_distinct_sections(self):
        data = self._build_docx()
        doc = DocxParser().parse(data)

        types = [s.section_type.value for s in doc.sections]
        assert "heading" in types
        assert "paragraph" in types

    def test_table_rows_grouped_into_one_table_section(self):
        data = self._build_docx(add_table=True)
        doc = DocxParser().parse(data)

        table_sections = [s for s in doc.sections if s.section_type.value == "table"]
        assert len(table_sections) == 1
        assert "Header A" in table_sections[0].text
        assert "Value 2" in table_sections[0].text

    def test_empty_docx_yields_no_sections(self):
        import docx
        document = docx.Document()
        buf = io.BytesIO()
        document.save(buf)

        doc = DocxParser().parse(buf.getvalue())
        assert doc.sections == []


class TestXlsxParser:
    def _build_xlsx(self):
        import openpyxl
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Q4"
        ws.append(["Region", "Revenue"])
        ws.append(["West", 4200000])
        ws.append(["East", None])  # a row with a missing value shouldn't crash
        buf = io.BytesIO()
        wb.save(buf)
        return buf.getvalue()

    def test_rows_paired_with_column_headers_not_raw_concatenation(self):
        data = self._build_xlsx()
        doc = XlsxParser().parse(data)

        assert len(doc.sections) == 2
        assert doc.sections[0].text == "Region: West; Revenue: 4200000"
        assert doc.sections[0].sheet_name == "Q4"

    def test_row_with_none_cell_does_not_crash(self):
        data = self._build_xlsx()
        doc = XlsxParser().parse(data)
        # "East" row has a None revenue — should still produce a section,
        # not raise, and not silently drop the row entirely.
        assert any("East" in s.text for s in doc.sections)


class TestCsvParser:
    def test_rows_paired_with_column_headers(self):
        data = b"Region,Revenue\nWest,4200000\nEast,3100000\n"
        doc = CsvParser().parse(data)

        assert len(doc.sections) == 2
        assert doc.sections[0].text == "Region: West; Revenue: 4200000"

    def test_empty_csv_yields_no_sections(self):
        doc = CsvParser().parse(b"")
        assert doc.sections == []

    def test_header_only_csv_yields_no_sections(self):
        doc = CsvParser().parse(b"Region,Revenue\n")
        assert doc.sections == []
