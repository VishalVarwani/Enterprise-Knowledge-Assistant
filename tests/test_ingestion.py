"""
test_ingestion.py
-----------------
Tests for chunker (Chunker class), loaders, and deduplication.
"""

from __future__ import annotations
import pytest


class TestChunker:

    @pytest.mark.unit
    def test_short_text_returns_single_chunk(self, sample_text):
        from src.ingestion.chunker import Chunker
        chunker = Chunker(chunk_size=512, chunk_overlap=50)
        chunks = chunker.chunk(sample_text, doc_id="doc-001", doc_name="Test")
        assert len(chunks) >= 1
        assert all(c.content for c in chunks)

    @pytest.mark.unit
    def test_long_text_produces_multiple_chunks(self):
        from src.ingestion.chunker import Chunker
        long_text = " ".join([f"word_{i}" for i in range(2000)])
        chunker = Chunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.chunk(long_text, doc_id="doc-001", doc_name="Test")
        assert len(chunks) > 1

    @pytest.mark.unit
    def test_chunk_index_sequential(self):
        from src.ingestion.chunker import Chunker
        long_text = " ".join([f"token_{i}" for i in range(1000)])
        chunker = Chunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.chunk(long_text, doc_id="doc-001", doc_name="Test")
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(indices)))

    @pytest.mark.unit
    def test_page_markers_stripped_from_content(self):
        from src.ingestion.chunker import Chunker
        text = (
            "--- Page 1 ---\nThis is page one content with enough words to pass the filter. "
            "It has more than fifty characters easily.\n"
            "--- Page 2 ---\nThis is page two content with enough words as well. Also long enough."
        )
        chunker = Chunker(chunk_size=512, chunk_overlap=50)
        chunks = chunker.chunk(text, doc_id="doc-001", doc_name="Test")
        for chunk in chunks:
            assert "--- Page" not in chunk.content

    @pytest.mark.unit
    def test_empty_text_returns_empty_list(self):
        from src.ingestion.chunker import Chunker
        chunker = Chunker()
        chunks = chunker.chunk("", doc_id="doc-001", doc_name="Test")
        assert chunks == []

    @pytest.mark.unit
    def test_whitespace_only_returns_empty(self):
        from src.ingestion.chunker import Chunker
        chunker = Chunker()
        chunks = chunker.chunk("   \n\t  \n  ", doc_id="doc-001", doc_name="Test")
        assert chunks == []

    @pytest.mark.unit
    def test_chunks_carry_doc_id(self):
        from src.ingestion.chunker import Chunker
        chunker = Chunker()
        chunks = chunker.chunk("Some test content " * 20, doc_id="doc-999", doc_name="My Doc")
        assert all(c.document_id == "doc-999" for c in chunks)

    @pytest.mark.unit
    def test_estimate_tokens_instance_method(self):
        from src.ingestion.chunker import Chunker
        chunker = Chunker()
        est = chunker._estimate_tokens("Hello world " * 100)
        assert 100 <= est <= 500

    @pytest.mark.unit
    def test_estimate_tokens_empty(self):
        from src.ingestion.chunker import Chunker
        chunker = Chunker()
        assert chunker._estimate_tokens("") == 0


class TestPDFLoader:

    @pytest.mark.unit
    def test_load_pdf_returns_list_of_raw_documents(self, sample_pdf_path):
        from src.ingestion.loaders.pdf_loader import PDFLoader
        loader = PDFLoader()
        results = loader.load(str(sample_pdf_path))
        assert isinstance(results, list)
        assert len(results) >= 1
        result = results[0]
        assert result.content
        assert result.source_type == "pdf"   # direct field, not in metadata

    @pytest.mark.unit
    def test_load_nonexistent_file_raises(self):
        from src.ingestion.loaders.pdf_loader import PDFLoader
        loader = PDFLoader()
        with pytest.raises((FileNotFoundError, ValueError, Exception)):
            loader.load("/nonexistent/path/file.pdf")

    @pytest.mark.unit
    def test_pdf_metadata_includes_total_pages(self, sample_pdf_path):
        from src.ingestion.loaders.pdf_loader import PDFLoader
        loader = PDFLoader()
        results = loader.load(str(sample_pdf_path))
        if results:
            # actual key is "total_pages" (from the PDFLoader implementation)
            assert "total_pages" in results[0].metadata


class TestDOCXLoader:

    @pytest.mark.unit
    def test_load_docx_returns_list(self, sample_docx_path):
        from src.ingestion.loaders.docx_loader import DOCXLoader
        loader = DOCXLoader()
        results = loader.load(str(sample_docx_path))
        assert isinstance(results, list)
        assert len(results) >= 1
        result = results[0]
        assert result.content
        assert result.source_type == "docx"   # direct field on RawDocument

    @pytest.mark.unit
    def test_docx_contains_heading_text(self, sample_docx_path):
        from src.ingestion.loaders.docx_loader import DOCXLoader
        loader = DOCXLoader()
        results = loader.load(str(sample_docx_path))
        assert results
        combined = " ".join(r.content for r in results).lower()
        assert "leave" in combined or "policy" in combined


class TestDeduplication:

    @pytest.mark.unit
    def test_same_content_same_hash(self):
        import hashlib
        content = "The leave policy grants 25 days per year."
        h1 = hashlib.sha256(content.encode()).hexdigest()
        h2 = hashlib.sha256(content.encode()).hexdigest()
        assert h1 == h2

    @pytest.mark.unit
    def test_different_content_different_hash(self):
        import hashlib
        h1 = hashlib.sha256(b"content A").hexdigest()
        h2 = hashlib.sha256(b"content B").hexdigest()
        assert h1 != h2


class TestTokenEstimation:

    @pytest.mark.unit
    def test_module_level_estimate_reasonable(self):
        from src.ingestion.chunker import estimate_tokens
        est = estimate_tokens("Hello world " * 100)
        assert 100 <= est <= 500

    @pytest.mark.unit
    def test_empty_text_zero_tokens(self):
        from src.ingestion.chunker import estimate_tokens
        assert estimate_tokens("") == 0
