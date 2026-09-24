"""
Format-agnostic representation every parser produces, and the leaf-level
chunk record that finally reaches storage.

The point of this layer is to stop flattening structure at parse time:
a PDF page, a DOCX heading, a spreadsheet row, all carry their own
identity (page number, sheet name, section type) through chunking instead
of collapsing into anonymous text the moment they're read.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class SectionType(str, Enum):
    PARAGRAPH = "paragraph"
    HEADING = "heading"
    TABLE = "table"
    ROW = "row"  # a single spreadsheet/CSV row


@dataclass
class DocumentSection:
    """One structural unit of a parsed document, before leaf-level chunking."""
    text: str
    section_type: SectionType = SectionType.PARAGRAPH
    page_number: Optional[int] = None
    sheet_name: Optional[str] = None
    position: int = 0


@dataclass
class CanonicalDocument:
    """What every format-specific parser produces, regardless of source format."""
    sections: List[DocumentSection] = field(default_factory=list)
    source_type: str = "unknown"  # 'pdf' | 'docx' | 'xlsx' | 'csv'


@dataclass
class Chunk:
    """A leaf-level piece of text ready for embedding and storage, with the
    structural metadata of the section it came from preserved."""
    text: str
    page_number: Optional[int] = None
    sheet_name: Optional[str] = None
    chunk_type: str = SectionType.PARAGRAPH.value


@dataclass
class RetrievedChunk:
    """A chunk coming back from retrieval, carrying enough metadata to cite
    where it came from — the whole reason retrieval doesn't just return
    bare strings."""
    text: str
    document_id: str
    source_uri: Optional[str] = None
    page_number: Optional[int] = None
    sheet_name: Optional[str] = None
    chunk_type: str = SectionType.PARAGRAPH.value
