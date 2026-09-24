"""
Pure logic, no external dependencies — this module should always be
collectible and runnable, in any environment.
"""

import io
import zipfile

from src.ingestion.file_type_detector import detect_file_type


def _zip_with_entries(*entry_names: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name in entry_names:
            zf.writestr(name, "content")
    return buf.getvalue()


def test_detects_pdf_by_magic_bytes():
    data = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n1 0 obj\n<< >>\nendobj"
    assert detect_file_type(data) == "pdf"


def test_pdf_detection_ignores_filename_lie():
    data = b"%PDF-1.4\n..."
    assert detect_file_type(data, filename="not_a_pdf.txt") == "pdf"


def test_detects_docx_by_zip_member():
    data = _zip_with_entries("word/document.xml", "[Content_Types].xml")
    assert detect_file_type(data) == "docx"


def test_detects_xlsx_by_zip_member():
    data = _zip_with_entries("xl/workbook.xml", "[Content_Types].xml")
    assert detect_file_type(data) == "xlsx"


def test_docx_detection_ignores_filename_lie():
    data = _zip_with_entries("word/document.xml")
    assert detect_file_type(data, filename="report.xlsx") == "docx"


def test_detects_csv_by_content_sniff():
    data = b"name,age,city\nAlice,30,NYC\nBob,25,LA\n"
    assert detect_file_type(data) == "csv"


def test_detects_csv_by_extension_when_content_ambiguous():
    data = b"just one column of plain text\nwith no commas at all\n"
    assert detect_file_type(data, filename="notes.csv") == "csv"


def test_empty_bytes_is_unknown():
    assert detect_file_type(b"") == "unknown"


def test_garbage_bytes_with_no_filename_is_unknown():
    data = bytes([0x00, 0x01, 0x02, 0xFF, 0xFE, 0xFD] * 10)
    assert detect_file_type(data) == "unknown"


def test_bad_zip_signature_but_corrupt_archive_is_unknown():
    # Starts with the ZIP magic bytes but isn't a valid archive.
    data = b"PK\x03\x04" + b"\x00" * 20
    assert detect_file_type(data) == "unknown"
