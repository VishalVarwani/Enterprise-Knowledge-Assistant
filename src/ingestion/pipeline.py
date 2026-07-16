"""
pipeline.py
-----------
Ingestion pipeline: the main entry point for adding documents to the KB.

Flow:
  1. Route source to appropriate loader (PDF / DOCX / Web)
  2. Load → RawDocument(s)
  3. Chunk → list[Chunk]
  4. Embed chunks in batches → list[list[float]]
  5. Upsert document + chunks to Supabase (with deduplication by file hash)
  6. Return ingestion summary

Design decisions:
  - Single class (IngestionPipeline) coordinates all steps.
    No global state; all dependencies injected for testability.
  - Deduplication by SHA-256 hash: re-ingesting the same file is a no-op.
    This is important for enterprise KB where the same policy PDF might be
    uploaded by multiple users.
  - Chunks embedded AFTER all chunking is done so we can batch across
    an entire document (better Voyage API throughput).
  - Upsert (not insert) at the chunk level: allows incremental update
    if the document is modified and re-ingested.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import psycopg2
import psycopg2.extras
from loguru import logger
from supabase import create_client, Client

from src.config.settings import get_settings
from .chunker import Chunk, TextChunker
from .embedder import VoyageEmbedder
from .loaders.base_loader import RawDocument
from .loaders.pdf_loader import PDFLoader
from .loaders.docx_loader import DOCXLoader
from .loaders.web_loader import WebLoader


@dataclass
class IngestionResult:
    """Summary returned after ingesting one source."""
    source_name: str
    source_type: str
    document_id: Optional[str]
    chunks_created: int
    chunks_skipped: int
    was_duplicate: bool
    error: Optional[str] = None
    success: bool = True

    def __str__(self) -> str:
        if self.error:
            return f"[FAILED] {self.source_name}: {self.error}"
        status = "DUPLICATE" if self.was_duplicate else "OK"
        return (
            f"[{status}] {self.source_name} | "
            f"{self.chunks_created} chunks | "
            f"doc_id={self.document_id}"
        )


class IngestionPipeline:
    """
    Orchestrates the full document ingestion flow.

    Usage:
        pipeline = IngestionPipeline()
        result = pipeline.ingest_file("/path/to/doc.pdf")
        result = pipeline.ingest_url("https://company.com/policy")
    """

    def __init__(
        self,
        supabase_client: Optional[Client] = None,
        embedder: Optional[VoyageEmbedder] = None,
        chunker: Optional[TextChunker] = None,
    ):
        self.settings = get_settings()

        # Allow injection for testing; create defaults otherwise
        self.supabase = supabase_client or create_client(
            self.settings.SUPABASE_URL,
            self.settings.SUPABASE_SERVICE_KEY,
        )
        self.embedder = embedder or VoyageEmbedder()
        self.chunker = chunker or TextChunker(
            chunk_size=self.settings.CHUNK_SIZE,
            chunk_overlap=self.settings.CHUNK_OVERLAP,
            min_chunk_length=self.settings.CHUNK_MIN_LENGTH,
        )

        # Loaders (order matters: more specific first)
        self._loaders = [PDFLoader(), DOCXLoader(), WebLoader()]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest_file(self, file_path: Union[str, Path]) -> IngestionResult:
        """Ingest a local file (PDF or DOCX)."""
        path = Path(file_path)
        loader = self._get_loader(str(path))
        if not loader:
            return IngestionResult(
                source_name=path.name,
                source_type="unknown",
                document_id=None,
                chunks_created=0,
                chunks_skipped=0,
                was_duplicate=False,
                error=f"No loader available for {path.suffix}",
                success=False,
            )

        try:
            docs = loader.load(path)
        except Exception as e:
            logger.error(f"Load failed for {path}: {e}")
            return IngestionResult(
                source_name=path.name,
                source_type="unknown",
                document_id=None,
                chunks_created=0,
                chunks_skipped=0,
                was_duplicate=False,
                error=str(e),
                success=False,
            )

        return self._ingest_documents(docs)

    def ingest_url(self, url: str) -> IngestionResult:
        """Ingest a web page."""
        loader = WebLoader()
        try:
            docs = loader.load(url)
        except Exception as e:
            logger.error(f"Web load failed for {url}: {e}")
            return IngestionResult(
                source_name=url,
                source_type="web",
                document_id=None,
                chunks_created=0,
                chunks_skipped=0,
                was_duplicate=False,
                error=str(e),
                success=False,
            )
        return self._ingest_documents(docs)

    def ingest_bytes(
        self,
        data: bytes,
        filename: str,
    ) -> IngestionResult:
        """
        Ingest from raw bytes (file upload via API).
        Selects loader by filename extension.
        """
        suffix = Path(filename).suffix.lower()
        if suffix == ".pdf":
            loader = PDFLoader()
            docs = loader.load_from_bytes(data, filename)
        elif suffix in (".docx", ".doc"):
            loader = DOCXLoader()
            docs = loader.load_from_bytes(data, filename)
        else:
            return IngestionResult(
                source_name=filename,
                source_type="unknown",
                document_id=None,
                chunks_created=0,
                chunks_skipped=0,
                was_duplicate=False,
                error=f"Unsupported file type: {suffix}",
                success=False,
            )
        return self._ingest_documents(docs)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_loader(self, source: str):
        """Return the first loader that can handle this source."""
        for loader in self._loaders:
            if loader.can_handle(source):
                return loader
        return None

    def _ingest_documents(self, docs: list[RawDocument]) -> IngestionResult:
        """
        Core ingestion logic for a list of RawDocuments.

        Usually one document per source; web loaders may return one doc,
        PDF loaders always return one. Handles the common case of one doc.
        """
        if not docs:
            return IngestionResult(
                source_name="unknown",
                source_type="unknown",
                document_id=None,
                chunks_created=0,
                chunks_skipped=0,
                was_duplicate=False,
                error="Loader returned no documents",
                success=False,
            )

        # Take first document (multi-doc support is future work)
        doc = docs[0]

        # --- Deduplication check ---
        if doc.file_hash:
            existing = self._find_by_hash(doc.file_hash)
            if existing:
                logger.info(f"Duplicate detected: {doc.name} (hash={doc.file_hash[:8]}...)")
                return IngestionResult(
                    source_name=doc.name,
                    source_type=doc.source_type,
                    document_id=existing["id"],
                    chunks_created=0,
                    chunks_skipped=0,
                    was_duplicate=True,
                )

        # --- Chunk ---
        chunks = self.chunker.chunk_document(doc)
        if not chunks:
            return IngestionResult(
                source_name=doc.name,
                source_type=doc.source_type,
                document_id=None,
                chunks_created=0,
                chunks_skipped=0,
                was_duplicate=False,
                error="No chunks produced (document may be empty or too short)",
                success=False,
            )

        # --- Embed all chunks in one batch ---
        chunk_texts = [c.content for c in chunks]
        logger.info(f"Embedding {len(chunk_texts)} chunks for '{doc.name}'...")
        embeddings = self.embedder.embed_documents(chunk_texts)

        # --- Upsert to Supabase ---
        try:
            doc_id = self._upsert_document(doc, len(chunks))
            skipped = self._upsert_chunks(doc_id, chunks, embeddings)
        except Exception as e:
            logger.error(f"Database upsert failed for '{doc.name}': {e}")
            return IngestionResult(
                source_name=doc.name,
                source_type=doc.source_type,
                document_id=None,
                chunks_created=0,
                chunks_skipped=0,
                was_duplicate=False,
                error=f"DB error: {e}",
                success=False,
            )

        result = IngestionResult(
            source_name=doc.name,
            source_type=doc.source_type,
            document_id=doc_id,
            chunks_created=len(chunks) - skipped,
            chunks_skipped=skipped,
            was_duplicate=False,
        )
        logger.info(str(result))
        return result

    def _find_by_hash(self, file_hash: str) -> Optional[dict]:
        """Check if a document with this hash already exists."""
        response = (
            self.supabase.table("documents")
            .select("id, name")
            .eq("file_hash", file_hash)
            .limit(1)
            .execute()
        )
        return response.data[0] if response.data else None

    def _upsert_document(self, doc: RawDocument, total_chunks: int) -> str:
        """
        Insert or update the document record.

        Returns the document UUID.
        """
        doc_data = {
            "name": doc.name,
            "source_type": doc.source_type,
            "source_path": doc.source_path,
            "file_hash": doc.file_hash,
            "total_chunks": total_chunks,
            "metadata": doc.metadata,
        }

        # Upsert on source_path + file_hash combination
        response = (
            self.supabase.table("documents")
            .upsert(doc_data, on_conflict="source_path,file_hash")
            .execute()
        )
        return response.data[0]["id"]

    def _upsert_chunks(
        self,
        doc_id: str,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> int:
        """
        Bulk upsert chunks with their embeddings.

        Returns count of skipped chunks (empty/invalid).

        We use psycopg2 directly for the embedding upsert because
        Supabase REST (PostgREST) doesn't support pgvector type casting.
        The vector must be sent as a properly formatted string.
        """
        settings = get_settings()
        skipped = 0

        conn = psycopg2.connect(settings.DATABASE_URL)
        try:
            with conn.cursor() as cur:
                for chunk, embedding in zip(chunks, embeddings):
                    if not chunk.content.strip():
                        skipped += 1
                        continue

                    # Format vector as Postgres array literal: [0.1, 0.2, ...]
                    vector_str = "[" + ",".join(map(str, embedding)) + "]"

                    cur.execute(
                        """
                        INSERT INTO chunks
                            (document_id, content, chunk_index, token_count,
                             embedding, metadata)
                        VALUES
                            (%s, %s, %s, %s, %s::vector, %s)
                        ON CONFLICT (document_id, chunk_index)
                        DO UPDATE SET
                            content     = EXCLUDED.content,
                            token_count = EXCLUDED.token_count,
                            embedding   = EXCLUDED.embedding,
                            metadata    = EXCLUDED.metadata;
                        """,
                        (
                            doc_id,
                            chunk.content,
                            chunk.chunk_index,
                            chunk.token_estimate,
                            vector_str,
                            psycopg2.extras.Json(chunk.metadata),
                        ),
                    )

            conn.commit()
        finally:
            conn.close()

        return skipped
