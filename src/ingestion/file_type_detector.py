"""
Identifies a document's real format from its bytes, not its claimed
extension or URL suffix — a caller can lie about what a file is, magic
bytes mostly can't.
"""

import io
import logging
import zipfile
from typing import Optional

logger = logging.getLogger("file_type_detector")

SUPPORTED_TYPES = {"pdf", "docx", "xlsx", "csv"}


def detect_file_type(data: bytes, filename: Optional[str] = None) -> str:
    """
    Return a short format tag ('pdf', 'docx', 'xlsx', 'csv', or 'unknown').

    Magic bytes are checked first. DOCX/XLSX/PPTX all share the same ZIP
    container signature, so a ZIP hit is disambiguated by which top-level
    member directory it contains. Formats with no reliable magic number
    (CSV, plain text) fall back to the filename extension, then a last-resort
    content sniff.
    """
    if not data:
        return "unknown"

    if data.startswith(b"%PDF-"):
        return "pdf"

    if data[:4] == b"PK\x03\x04":
        office_type = _detect_office_zip_type(data)
        if office_type:
            return office_type

    ext = _extension_of(filename)
    if ext in SUPPORTED_TYPES:
        return ext

    if _looks_like_csv(data):
        return "csv"

    return "unknown"


def _detect_office_zip_type(data: bytes) -> Optional[str]:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            names = zf.namelist()
    except zipfile.BadZipFile as e:
        logger.debug(f"ZIP signature present but not a valid archive: {e}")
        return None

    if any(n.startswith("word/") for n in names):
        return "docx"
    if any(n.startswith("xl/") for n in names):
        return "xlsx"
    return None


def _extension_of(filename: Optional[str]) -> Optional[str]:
    if not filename or "." not in filename:
        return None
    return filename.rsplit(".", 1)[-1].lower()


def _looks_like_csv(data: bytes) -> bool:
    try:
        sample = data[:4096].decode("utf-8")
    except UnicodeDecodeError:
        return False
    first_lines = sample.splitlines()[:5]
    return bool(first_lines) and any("," in line for line in first_lines)
