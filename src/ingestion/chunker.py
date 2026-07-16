"""
chunker.py
----------
Text chunking for the ingestion pipeline.

Why chunking matters for RAG:
  Embedding a full 50-page document into one vector averages all concepts
  together — the query "what is the refund policy" gets diluted by content
  about shipping, returns, contact info, etc. Chunks create fine-grained
  retrieval targets.

Chunking strategy: Recursive Character Splitting
  Priority list of separators, tried in order:
    1. \\n\\n  (paragraph break — best natural boundary)
    2. \\n    (line break)
    3. . (sentence end)
    4. (space) (word break — last resort)

  This is the same strategy LangChain's RecursiveCharacterTextSplitter uses,
  but implemented here so we have full control and no LangChain dependency.

Size calibration:
  - CHUNK_SIZE = 512 tokens ≈ 350–400 words ≈ 1–2 dense paragraphs
  - Voyage AI's voyage-3-lite was trained on sequences up to 4096 tokens,
    but shorter chunks = more retrieval precision = less context noise.
    512 tokens is the empirically optimal point for enterprise KB RAG
    (based on the original RAG paper and BEIR benchmarks).
  - CHUNK_OVERLAP = 50 tokens: prevents a query that hits the boundary of
    two chunks from missing relevant context.

Token counting:
  - We use a simple word-based proxy: 1 word ≈ 1.3 tokens (GPT tokenizer)
  - This avoids importing tiktoken/transformers just for chunking
  - Exact token counts (from Voyage API response) are stored separately
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger

from .loaders.base_loader import RawDocument


# Separators tried in priority order for recursive splitting
SEPARATORS = [
    "\n\n--- Page",   # PDF page breaks (from PDFLoader)
    "\n\n# ",         # H1 headings (from DOCXLoader)
    "\n\n## ",        # H2 headings
    "\n\n### ",       # H3 headings
    "\n\n",           # Paragraph breaks
    "\n",             # Line breaks
    ". ",             # Sentence ends
    " ",              # Word breaks (last resort)
]


@dataclass
class Chunk:
    """
    A single text chunk ready for embedding and storage.

    Fields:
        content       : The text content of this chunk.
        chunk_index   : Position within the source document (0-based).
        document_id   : Will be set after database insert.
        token_estimate: Rough token count (word-based proxy).
        metadata      : Inherits from source doc; extended with chunk-level info.
    """
    content: str
    chunk_index: int
    document_id: Optional[str] = None
    token_estimate: int = 0
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.token_estimate == 0:
            self.token_estimate = estimate_tokens(self.content)


def estimate_tokens(text: str) -> int:
    """
    Estimate token count from word count.

    Rule of thumb: 1 English word ≈ 1.3 BPE tokens.
    Good enough for chunking decisions; exact counts come from the API.
    """
    word_count = len(text.split())
    return int(word_count * 1.3)


class TextChunker:
    """
    Recursive character text splitter with overlap.

    Args:
        chunk_size   : Target max tokens per chunk.
        chunk_overlap: Tokens of overlap between consecutive chunks.
        min_chunk_length: Discard chunks shorter than this (noise elimination).
        separators   : Priority-ordered list of split characters.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_chunk_length: int = 50,
        separators: list[str] = SEPARATORS,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_length = min_chunk_length
        self.separators = separators

    def chunk_document(self, doc: RawDocument) -> list[Chunk]:
        """
        Split a RawDocument into chunks.

        Returns:
            List of Chunk objects, empty if document has no valid content.
        """
        if doc.is_empty():
            logger.warning(f"Empty document, skipping: {doc.name}")
            return []

        text = doc.content

        # Split into raw text segments
        segments = self._split_recursive(text, self.separators)

        # Apply overlap: slide a window over segments
        chunks_text = self._merge_with_overlap(segments)

        # Filter and build Chunk objects
        chunks: list[Chunk] = []
        for idx, text_chunk in enumerate(chunks_text):
            text_chunk = text_chunk.strip()
            if len(text_chunk) < self.min_chunk_length:
                continue  # Skip near-empty chunks (page numbers, headers)

            # Inherit document-level metadata, add chunk-level fields
            chunk_meta = {
                **doc.metadata,
                "source_name": doc.name,
                "source_type": doc.source_type,
                "source_path": doc.source_path,
                "chunk_index": idx,
            }

            # Try to extract page number if present in text
            page_match = re.search(r"--- Page (\d+) ---", text_chunk)
            if page_match:
                chunk_meta["page_number"] = int(page_match.group(1))
                # Remove page marker from chunk text
                text_chunk = re.sub(r"--- Page \d+ ---\n?", "", text_chunk).strip()

            if len(text_chunk) < self.min_chunk_length:
                continue

            chunks.append(
                Chunk(
                    content=text_chunk,
                    chunk_index=len(chunks),  # Reindex after filtering
                    metadata=chunk_meta,
                )
            )

        logger.debug(
            f"Chunked '{doc.name}': {doc.char_count()} chars → {len(chunks)} chunks"
        )
        return chunks

    def chunk_documents(self, docs: list[RawDocument]) -> list[list[Chunk]]:
        """Chunk multiple documents. Returns list of chunk lists (one per doc)."""
        return [self.chunk_document(doc) for doc in docs]

    def _split_recursive(self, text: str, separators: list[str]) -> list[str]:
        """
        Recursively split text using the separator priority list.

        If the text fits within chunk_size, return as-is.
        Otherwise, try the first separator. If it produces too-large segments,
        recurse with the next separator.
        """
        if estimate_tokens(text) <= self.chunk_size:
            return [text]

        if not separators:
            # No more separators: hard split by words
            return self._hard_split(text)

        separator = separators[0]
        remaining = separators[1:]

        parts = text.split(separator)
        if len(parts) == 1:
            # Separator not found; try next
            return self._split_recursive(text, remaining)

        # Re-attach separator to maintain content (except for hard splits)
        result: list[str] = []
        for i, part in enumerate(parts):
            if i > 0 and separator.strip():
                part = separator + part
            part = part.strip()
            if not part:
                continue

            if estimate_tokens(part) <= self.chunk_size:
                result.append(part)
            else:
                # Part still too large; recurse
                result.extend(self._split_recursive(part, remaining))

        return result

    def _hard_split(self, text: str) -> list[str]:
        """Split by words when no separator works (last resort)."""
        words = text.split()
        # Convert chunk_size tokens back to words (÷ 1.3)
        words_per_chunk = int(self.chunk_size / 1.3)
        chunks = []
        for i in range(0, len(words), words_per_chunk):
            chunks.append(" ".join(words[i : i + words_per_chunk]))
        return chunks

    def _merge_with_overlap(self, segments: list[str]) -> list[str]:
        """
        Merge small segments and add overlap between chunks.

        Why overlap: If a sentence spans a chunk boundary, both chunks
        should contain it. Without overlap, a query matching that sentence
        might retrieve neither chunk with sufficient context.

        Algorithm:
          - Accumulate segments into a buffer until chunk_size reached
          - When buffer is full, emit as chunk
          - Backtrack by chunk_overlap tokens for the next chunk start
        """
        if not segments:
            return []

        chunks: list[str] = []
        buffer: list[str] = []
        buffer_tokens = 0

        for segment in segments:
            seg_tokens = estimate_tokens(segment)

            if buffer_tokens + seg_tokens > self.chunk_size and buffer:
                # Emit current buffer as a chunk
                chunks.append("\n\n".join(buffer))

                # Build overlap: keep tail of buffer that fits within overlap limit
                overlap_buffer: list[str] = []
                overlap_tokens = 0
                for s in reversed(buffer):
                    s_tokens = estimate_tokens(s)
                    if overlap_tokens + s_tokens > self.chunk_overlap:
                        break
                    overlap_buffer.insert(0, s)
                    overlap_tokens += s_tokens

                buffer = overlap_buffer
                buffer_tokens = overlap_tokens

            buffer.append(segment)
            buffer_tokens += seg_tokens

        # Emit remaining buffer
        if buffer:
            chunks.append("\n\n".join(buffer))

        return chunks


class Chunker(TextChunker):
    """
    Convenience subclass of TextChunker with a simpler interface.

    Used for quick chunking of raw text strings (e.g. in tests or scripts)
    without needing to construct a full RawDocument object.

    Usage:
        chunker = Chunker(chunk_size=512, chunk_overlap=50)
        chunks = chunker.chunk("Long document text...", doc_id="doc-1", doc_name="Policy")
    """

    def chunk(
        self,
        text: str,
        doc_id: str = "",
        doc_name: str = "Unnamed",
    ) -> list[Chunk]:
        """
        Chunk raw text without a RawDocument wrapper.

        Args:
            text    : The document text to chunk.
            doc_id  : Optional document ID (populated after DB insert in production).
            doc_name: Human-readable document name for metadata.

        Returns:
            List of Chunk objects; empty if text is blank.
        """
        if not text or not text.strip():
            return []

        from .loaders.base_loader import RawDocument
        doc = RawDocument(
            content=text,
            name=doc_name,
            source_path="",
            source_type="text",
            metadata={"doc_id": doc_id},
        )
        chunks = self.chunk_document(doc)
        # Attach doc_id if provided
        if doc_id:
            for c in chunks:
                c.document_id = doc_id
        return chunks

    def _estimate_tokens(self, text: str) -> int:
        """Instance method exposing token estimator (used in tests)."""
        return estimate_tokens(text)
