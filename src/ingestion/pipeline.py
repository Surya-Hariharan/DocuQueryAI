"""
The single ingestion pipeline every caller goes through — the live API's
URL-fetch path and the standalone folder-scan CLI utility alike. There is
deliberately only one path from "raw bytes" to "list of chunks ready for
storage"; adding a format means extending parsers.py, not adding a second
pipeline.
"""

import logging
import os
from typing import List, Optional

from src.config import PDF_FOLDER
from src.ingestion.document_model import Chunk
from src.ingestion.parsers import get_parser
from src.ingestion.file_type_detector import detect_file_type
from src.ingestion.chunking import split_text_into_chunks

logger = logging.getLogger("pipeline")


def parse_document(data: bytes, filename: Optional[str] = None) -> List[Chunk]:
    """
    Detect the format, parse into a CanonicalDocument, then run each
    section's text through the shared leaf-level chunker — carrying that
    section's page/sheet/type metadata onto every chunk it produces.

    Raises ValueError if the format can't be identified or has no
    registered parser.
    """
    file_type = detect_file_type(data, filename)
    if file_type == "unknown":
        raise ValueError("Could not identify the document format (unsupported or unrecognized file type)")

    parser = get_parser(file_type)
    canonical_doc = parser.parse(data)

    chunks: List[Chunk] = []
    for section in canonical_doc.sections:
        for sub_text in split_text_into_chunks(section.text):
            chunks.append(Chunk(
                text=sub_text,
                page_number=section.page_number,
                sheet_name=section.sheet_name,
                chunk_type=section.section_type.value
            ))

    logger.info(f"Parsed {file_type} document into {len(chunks)} chunks from {len(canonical_doc.sections)} sections")
    return chunks


def parse_files_in_folder(folder_path: str = PDF_FOLDER, vector_store=None):
    """
    Standalone/CLI utility: ingest every supported file in a local folder.
    Not used by the live API. Pass an already-initialized `vector_store` to
    reuse its connection pool; if omitted, a new one is created here.
    """
    from src.retrieval.pg_vector_store import PgVectorStore  # deferred: avoids importing the DB layer for API-side parsing

    if vector_store is None:
        vector_store = PgVectorStore()
        vector_store.init_connection_pool()
        vector_store.init_schema()

    if not os.path.exists(folder_path):
        logger.warning(f"Folder '{folder_path}' does not exist.")
        return

    for fname in os.listdir(folder_path):
        path = os.path.join(folder_path, fname)
        if not os.path.isfile(path):
            continue

        with open(path, "rb") as f:
            data = f.read()

        try:
            chunks = parse_document(data, filename=fname)
        except ValueError as e:
            logger.warning(f"Skipping '{fname}': {e}")
            continue

        if not chunks:
            logger.warning(f"No content extracted from '{fname}'")
            continue

        document_id = vector_store.get_or_create_document(fname, source_type="file")
        vector_store.upsert_chunks(chunks, document_id)
        logger.info(f"✅ '{fname}': {len(chunks)} chunks extracted and stored")
