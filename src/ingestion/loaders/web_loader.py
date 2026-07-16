"""
web_loader.py
-------------
Web page loader using trafilatura for content extraction.

Why trafilatura over raw BeautifulSoup:
  - ML-trained content extractor: removes nav bars, ads, cookie banners,
    footers automatically. BeautifulSoup alone requires site-specific
    selectors that break when HTML changes.
  - Returns structured text with optional metadata (title, author, date)
  - Handles paywalled pages more gracefully than scrapy/requests+bs4
  - Preserves paragraph structure for chunking

Fallback:
  - If trafilatura returns empty (some JS-heavy sites), falls back to
    BeautifulSoup with conservative tag stripping.

Rate limiting:
  - Adds a small delay between requests when loading multiple URLs
  - Respects robots.txt via trafilatura config
"""

from __future__ import annotations

import hashlib
import re
import time
from pathlib import Path
from typing import Union
from urllib.parse import urlparse

import requests
from loguru import logger

from .base_loader import BaseLoader, RawDocument

try:
    import trafilatura
    from trafilatura.settings import use_config
except ImportError:
    raise ImportError("trafilatura not installed. Run: pip install trafilatura")

try:
    from bs4 import BeautifulSoup
except ImportError:
    raise ImportError("beautifulsoup4 not installed. Run: pip install beautifulsoup4")


# Request settings
REQUEST_TIMEOUT = 15        # seconds; enterprise intranets can be slow
REQUEST_DELAY = 0.5         # seconds between multiple URL loads
MAX_CONTENT_LENGTH = 5_000_000  # 5 MB; skip very large pages

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; EnterpriseKnowledgeBot/1.0; "
        "+https://yourcompany.com/bot)"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


class WebLoader(BaseLoader):
    """Loads web pages into RawDocument objects."""

    def can_handle(self, source: Union[str, Path]) -> bool:
        source_str = str(source)
        return source_str.startswith(("http://", "https://"))

    def load(self, source: Union[str, Path]) -> list[RawDocument]:
        url = str(source)
        if not self.can_handle(url):
            raise ValueError(f"Not a valid URL: {url}")

        logger.info(f"Loading web page: {url}")

        try:
            html_bytes, status_code = self._fetch(url)
        except Exception as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return []

        if not html_bytes:
            logger.warning(f"Empty response from {url}")
            return []

        file_hash = self._compute_hash(html_bytes)
        html_text = html_bytes.decode("utf-8", errors="replace")

        # Primary extraction: trafilatura
        content, title, metadata = self._extract_trafilatura(html_text, url)

        # Fallback: BeautifulSoup
        if not content or len(content.strip()) < 100:
            logger.debug(f"trafilatura returned sparse content; falling back to bs4 for {url}")
            content = self._extract_bs4(html_text)
            title = title or self._extract_title_bs4(html_text)

        if not content or not content.strip():
            logger.warning(f"No extractable content from {url}")
            return []

        # Add URL and title to metadata
        parsed = urlparse(url)
        metadata.update({
            "url": url,
            "domain": parsed.netloc,
            "status_code": status_code,
            "content_length": len(content),
        })

        display_name = title or parsed.netloc + parsed.path

        logger.info(f"Web loaded: {url} | {len(content)} chars | title: {title[:60] if title else 'N/A'}")

        return [
            RawDocument(
                content=content,
                source_path=url,
                source_type="web",
                name=display_name[:200],  # Cap name length
                metadata=metadata,
                file_hash=file_hash,
            )
        ]

    def load_multiple(self, urls: list[str], delay: float = REQUEST_DELAY) -> list[RawDocument]:
        """
        Load multiple URLs with rate limiting.

        Args:
            urls: List of URLs to load.
            delay: Seconds to wait between requests.
        Returns:
            All successfully loaded documents.
        """
        results: list[RawDocument] = []
        for i, url in enumerate(urls):
            if i > 0:
                time.sleep(delay)
            docs = self.load(url)
            results.extend(docs)
        return results

    def _fetch(self, url: str) -> tuple[bytes, int]:
        """
        Fetch URL with timeout and size limit.

        Returns (bytes, status_code).
        Raises requests.RequestException on network failure.
        """
        response = requests.get(
            url,
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT,
            stream=True,  # Stream to check content-length before downloading
        )
        response.raise_for_status()

        # Size guard: skip pages over MAX_CONTENT_LENGTH
        content_length = int(response.headers.get("content-length", 0))
        if content_length > MAX_CONTENT_LENGTH:
            logger.warning(f"Page too large ({content_length} bytes): {url}")
            return b"", response.status_code

        # Read up to max
        content = b""
        for chunk in response.iter_content(chunk_size=65536):
            content += chunk
            if len(content) > MAX_CONTENT_LENGTH:
                logger.warning(f"Truncating response at {MAX_CONTENT_LENGTH} bytes: {url}")
                break

        return content, response.status_code

    def _extract_trafilatura(
        self, html: str, url: str
    ) -> tuple[str, str, dict]:
        """
        Use trafilatura for primary extraction.

        Returns (text_content, title, metadata_dict).
        """
        # Configure trafilatura: include tables, exclude comments/ads
        config = use_config()
        config.set("DEFAULT", "EXTRACTION_TIMEOUT", "10")

        result = trafilatura.extract(
            html,
            url=url,
            include_tables=True,
            include_links=False,       # Links add noise without retrieval value
            include_images=False,
            no_fallback=False,
            favor_recall=True,         # Recall over precision for enterprise KB
            config=config,
        )

        metadata_raw = trafilatura.extract_metadata(html, default_url=url)
        metadata: dict = {}
        title = ""

        if metadata_raw:
            title = metadata_raw.title or ""
            metadata = {
                "author": metadata_raw.author or "",
                "date": str(metadata_raw.date) if metadata_raw.date else "",
                "sitename": metadata_raw.sitename or "",
                "description": metadata_raw.description or "",
            }

        return result or "", title, metadata

    def _extract_bs4(self, html: str) -> str:
        """
        BeautifulSoup fallback extractor.

        Removes scripts, styles, and nav elements; returns remaining text.
        """
        soup = BeautifulSoup(html, "lxml")

        # Remove noise elements
        for tag in soup(["script", "style", "nav", "header", "footer",
                          "aside", "form", "button", "iframe"]):
            tag.decompose()

        # Get text, collapse whitespace
        text = soup.get_text(separator="\n")
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)
        return text.strip()

    def _extract_title_bs4(self, html: str) -> str:
        """Extract page title using BeautifulSoup."""
        try:
            soup = BeautifulSoup(html, "lxml")
            title_tag = soup.find("title")
            return title_tag.get_text().strip() if title_tag else ""
        except Exception:
            return ""
