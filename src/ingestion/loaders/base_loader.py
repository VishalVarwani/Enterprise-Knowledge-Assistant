"""
base_loader.py
--------------
Abstract base class for all document loaders.

Why an ABC:
  - Enforces a consistent interface: every loader MUST return the same
    data structure (list of Document dicts). The ingestion pipeline
    doesn't care if it's talking to a PDF or web loader.
  - Makes adding new source types (Confluence, Notion, S3) a matter of
    subclassing without touching pipeline.py.
  - Enables dependency injection and easy mocking in tests.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class RawDocument:
    """
    Canonical output of any loader.

    Fields:
        content      : Full text content of the document.
        source_path  : Original file path or URL.
        source_type  : One of 'pdf' | 'docx' | 'web' | 'text'.
        name         : Human-friendly display name.
        metadata     : Arbitrary key-value pairs (author, page_count, etc.).
                       Stored as JSONB in Supabase for flexible querying.
        file_hash    : SHA-256 of the source bytes; used to detect re-uploads.
    """
    content: str
    source_path: str
    source_type: str
    name: str
    metadata: dict = field(default_factory=dict)
    file_hash: Optional[str] = None

    def is_empty(self) -> bool:
        """A document with only whitespace provides no retrieval value."""
        return not self.content or not self.content.strip()

    def word_count(self) -> int:
        return len(self.content.split())

    def char_count(self) -> int:
        return len(self.content)


class BaseLoader(ABC):
    """
    Abstract loader. All concrete loaders inherit this.

    Contract:
        load(source) → list[RawDocument]

    A single source can produce multiple documents (e.g., a ZIP of PDFs),
    so the return type is always a list, even for single-file loaders.
    """

    @abstractmethod
    def load(self, source: str | Path) -> list[RawDocument]:
        """
        Load the source and return a list of RawDocuments.

        Args:
            source: File path (str or Path) or URL string.

        Returns:
            List of RawDocument objects. May be empty if loading fails.

        Raises:
            ValueError: If source is unreachable or unsupported format.
        """
        ...

    @abstractmethod
    def can_handle(self, source: str | Path) -> bool:
        """
        Returns True if this loader can process the given source.
        Used by the router in pipeline.py to select the right loader.
        """
        ...

    def _compute_hash(self, data: bytes) -> str:
        """SHA-256 hash of raw bytes. Used for deduplication."""
        import hashlib
        return hashlib.sha256(data).hexdigest()
