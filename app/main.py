"""
main.py (Streamlit)
-------------------
Enterprise Knowledge Assistant UI.

UI sections:
  1. Chat interface — query input, answer display, source citations
  2. Document Manager — upload files / ingest URLs, list documents
  3. Admin Panel    — cache stats, system health, guardrail logs

Run:
    streamlit run app/main.py

Design: clean professional theme. Dark sidebar, white main area.
Consistent with enterprise SaaS aesthetics (not colorful/playful).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import httpx
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import get_settings

# --- Page config (must be first Streamlit call) ---
st.set_page_config(
    page_title="Enterprise Knowledge Assistant",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS ---
st.markdown("""
<style>
/* Main layout */
.main .block-container { padding-top: 2rem; }

/* Chat message styling */
.user-message {
    background: #f0f2f6;
    border-left: 4px solid #4f46e5;
    padding: 12px 16px;
    border-radius: 0 8px 8px 0;
    margin: 8px 0;
}
.assistant-message {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-left: 4px solid #059669;
    padding: 12px 16px;
    border-radius: 0 8px 8px 0;
    margin: 8px 0;
}
.blocked-message {
    background: #fff7ed;
    border-left: 4px solid #f59e0b;
    padding: 12px 16px;
    border-radius: 0 8px 8px 0;
    margin: 8px 0;
}

/* Source card */
.source-card {
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    border-radius: 6px;
    padding: 8px 12px;
    margin: 4px 0;
    font-size: 0.85em;
}

/* Metric display */
.metric-good { color: #059669; font-weight: 600; }
.metric-warn { color: #d97706; font-weight: 600; }
.metric-bad  { color: #dc2626; font-weight: 600; }

/* Sidebar header */
section[data-testid="stSidebar"] .block-container { padding-top: 1rem; }

/* Hide streamlit branding */
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

settings = get_settings()
API_BASE = f"http://localhost:{settings.API_PORT}"


# ============================================================
# API Client
# ============================================================

def api_query(question: str, filter_doc_ids: list[str] | None = None) -> dict:
    """Call the /query endpoint."""
    with httpx.Client(timeout=60.0) as client:
        resp = client.post(
            f"{API_BASE}/query",
            json={"query": question, "filter_doc_ids": filter_doc_ids, "use_cache": True},
        )
        resp.raise_for_status()
        return resp.json()


def api_upload_file(data: bytes, filename: str) -> dict:
    with httpx.Client(timeout=120.0) as client:
        resp = client.post(
            f"{API_BASE}/ingest/file",
            files={"file": (filename, data)},
        )
        resp.raise_for_status()
        return resp.json()


def api_ingest_url(url: str) -> dict:
    with httpx.Client(timeout=60.0) as client:
        resp = client.post(f"{API_BASE}/ingest/url", json={"url": url})
        resp.raise_for_status()
        return resp.json()


def api_list_docs() -> dict:
    with httpx.Client(timeout=15.0) as client:
        resp = client.get(f"{API_BASE}/ingest/documents")
        resp.raise_for_status()
        return resp.json()


def api_delete_doc(doc_id: str) -> dict:
    with httpx.Client(timeout=15.0) as client:
        resp = client.delete(f"{API_BASE}/ingest/documents/{doc_id}")
        resp.raise_for_status()
        return resp.json()


def api_health() -> dict:
    with httpx.Client(timeout=5.0) as client:
        resp = client.get(f"{API_BASE}/health")
        resp.raise_for_status()
        return resp.json()


def api_cache_stats() -> dict:
    with httpx.Client(timeout=5.0) as client:
        resp = client.get(f"{API_BASE}/cache/stats")
        resp.raise_for_status()
        return resp.json()


def api_clear_cache() -> None:
    with httpx.Client(timeout=10.0) as client:
        client.post(f"{API_BASE}/cache/clear")


# ============================================================
# Sidebar
# ============================================================

def render_sidebar():
    with st.sidebar:
        st.image("https://via.placeholder.com/200x50/4f46e5/ffffff?text=EKA", width=200)
        st.markdown("**Enterprise Knowledge Assistant**")
        st.caption(f"v{settings.APP_VERSION} · {settings.ENVIRONMENT}")
        st.divider()

        page = st.radio(
            "Navigation",
            ["💬 Chat", "📄 Documents", "⚙️ Admin"],
            label_visibility="collapsed",
        )

        st.divider()

        # System status
        st.caption("System Status")
        try:
            health = api_health()
            db_ok = health["database"] == "healthy"
            st.markdown(
                f"{'🟢' if db_ok else '🔴'} Database: `{health['database']}`"
            )
            st.markdown(
                f"🟢 LLM: `{health['details'].get('llm_provider', 'unknown')}`"
            )
            st.markdown(
                f"🟢 Embedder: `{health['details'].get('voyage_model', 'unknown')}`"
            )
        except Exception:
            st.markdown("🔴 API not reachable — is `uvicorn api.main:app` running?")

        return page


# ============================================================
# Chat Page
# ============================================================

def render_chat():
    st.title("💬 Enterprise Knowledge Assistant")
    st.caption(
        "Ask questions about company policies, procedures, and documentation. "
        "All answers are grounded in your uploaded documents."
    )

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display history
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="user-message">🧑 {msg["content"]}</div>',
                unsafe_allow_html=True,
            )
        elif msg["role"] == "assistant":
            css_class = "blocked-message" if msg.get("blocked") else "assistant-message"
            st.markdown(
                f'<div class="{css_class}">🤖 {msg["content"]}</div>',
                unsafe_allow_html=True,
            )
            if msg.get("sources"):
                _render_sources(msg["sources"])
            if msg.get("metadata"):
                _render_response_metadata(msg["metadata"])

    # Query input
    query = st.chat_input("Ask a question about company policies or procedures...")

    if query:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": query})
        st.markdown(
            f'<div class="user-message">🧑 {query}</div>',
            unsafe_allow_html=True,
        )

        with st.spinner("Searching knowledge base..."):
            try:
                start = time.perf_counter()
                result = api_query(query)
                client_latency = int((time.perf_counter() - start) * 1000)

                answer = result["answer"]
                sources = result.get("sources", [])
                blocked = result.get("was_blocked", False)

                css_class = "blocked-message" if blocked else "assistant-message"
                st.markdown(
                    f'<div class="{css_class}">🤖 {answer}</div>',
                    unsafe_allow_html=True,
                )

                if sources and not blocked:
                    _render_sources(sources)

                metadata = {
                    "cached": result.get("was_cached", False),
                    "blocked": blocked,
                    "block_reason": result.get("block_reason"),
                    "model": result.get("model", ""),
                    "latency_ms": result.get("latency_ms", client_latency),
                    "faithfulness_score": result.get("faithfulness_score"),
                    "grounding_score": result.get("grounding_score"),
                    "tokens": result.get("prompt_tokens", 0) + result.get("completion_tokens", 0),
                }
                _render_response_metadata(metadata)

                # Store in history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "blocked": blocked,
                    "sources": sources if not blocked else [],
                    "metadata": metadata,
                })

            except httpx.ConnectError:
                st.error("Cannot connect to API. Start with: `uvicorn api.main:app --reload`")
            except httpx.HTTPStatusError as e:
                st.error(f"API error {e.response.status_code}: {e.response.text}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")

    # Clear chat button
    if st.session_state.messages:
        if st.button("Clear conversation", key="clear_chat"):
            st.session_state.messages = []
            st.rerun()


def _render_sources(sources: list[dict]):
    """Render source citation cards below an answer."""
    if not sources:
        return
    with st.expander(f"📚 Sources ({len(sources)} documents)", expanded=False):
        for i, src in enumerate(sources, 1):
            score = src.get("rerank_score", 0) or 0
            page = f", p. {src['page_number']}" if src.get("page_number") else ""
            type_icon = {"pdf": "📄", "docx": "📝", "web": "🌐"}.get(
                src.get("doc_source_type", ""), "📄"
            )
            st.markdown(
                f'<div class="source-card">'
                f'{type_icon} <strong>[{i}] {src["document_name"]}</strong>{page} '
                f'<span style="color: #9ca3af">· relevance: {score:.2f}</span>'
                f"</div>",
                unsafe_allow_html=True,
            )


def _render_response_metadata(meta: dict):
    """Render small metadata row: cache/latency/scores."""
    parts = []

    if meta.get("cached"):
        parts.append("⚡ Cached")
    if meta.get("blocked"):
        parts.append(f"🛡️ Blocked: {meta.get('block_reason', '')}")
    if meta.get("latency_ms"):
        parts.append(f"⏱️ {meta['latency_ms']}ms")
    if meta.get("faithfulness_score") is not None:
        f_score = meta["faithfulness_score"]
        color = "metric-good" if f_score >= 0.8 else "metric-warn" if f_score >= 0.5 else "metric-bad"
        parts.append(f'<span class="{color}">🎯 Faithfulness: {f_score:.2f}</span>')
    if meta.get("tokens"):
        parts.append(f"🔢 {meta['tokens']} tokens")

    if parts:
        st.markdown(
            f'<div style="font-size: 0.78em; color: #9ca3af; margin-top: 4px">'
            + " · ".join(parts)
            + "</div>",
            unsafe_allow_html=True,
        )


# ============================================================
# Document Manager Page
# ============================================================

def render_documents():
    st.title("📄 Document Manager")

    tab_upload, tab_url, tab_list = st.tabs(["Upload File", "Ingest URL", "View Documents"])

    with tab_upload:
        st.subheader("Upload PDF or DOCX")
        uploaded = st.file_uploader(
            "Choose a file",
            type=["pdf", "docx"],
            help="Supported: PDF (.pdf) and Word documents (.docx)",
        )
        if uploaded and st.button("Ingest Document", type="primary"):
            with st.spinner(f"Ingesting {uploaded.name}..."):
                try:
                    result = api_upload_file(uploaded.getvalue(), uploaded.name)
                    if result.get("was_duplicate"):
                        st.info(f"ℹ️ Duplicate: **{uploaded.name}** was already ingested.")
                    else:
                        st.success(
                            f"✅ Ingested **{uploaded.name}** · "
                            f"{result['chunks_created']} chunks created"
                        )
                except Exception as e:
                    st.error(f"Upload failed: {e}")

    with tab_url:
        st.subheader("Ingest Web Page")
        url = st.text_input("URL", placeholder="https://company.com/policy-page")
        if url and st.button("Ingest URL", type="primary"):
            with st.spinner(f"Fetching {url}..."):
                try:
                    result = api_ingest_url(url)
                    if result.get("was_duplicate"):
                        st.info(f"ℹ️ Duplicate: this URL was already ingested.")
                    else:
                        st.success(
                            f"✅ Ingested **{result['document_name']}** · "
                            f"{result['chunks_created']} chunks"
                        )
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")

    with tab_list:
        st.subheader("Knowledge Base Documents")
        if st.button("Refresh"):
            st.rerun()
        try:
            data = api_list_docs()
            docs = data.get("documents", [])
            if not docs:
                st.info("No documents ingested yet. Upload some files to get started.")
            else:
                st.caption(f"{data['total']} document(s) in knowledge base")
                for doc in docs:
                    type_icon = {"pdf": "📄", "docx": "📝", "web": "🌐"}.get(
                        doc.get("source_type", ""), "📄"
                    )
                    col1, col2, col3 = st.columns([5, 2, 1])
                    with col1:
                        st.markdown(f"{type_icon} **{doc['name']}**")
                        st.caption(f"{doc['source_path'][:80]}")
                    with col2:
                        st.caption(f"{doc['total_chunks']} chunks")
                        st.caption(doc["created_at"][:10])
                    with col3:
                        if st.button("🗑️", key=f"del_{doc['id']}", help="Delete document"):
                            try:
                                api_delete_doc(doc["id"])
                                st.success("Deleted")
                                st.rerun()
                            except Exception as e:
                                st.error(str(e))
                    st.divider()
        except Exception as e:
            st.error(f"Could not load documents: {e}")


# ============================================================
# Admin Panel Page
# ============================================================

def render_admin():
    st.title("⚙️ Admin Panel")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("System Health")
        try:
            health = api_health()
            for key, val in health.items():
                if key == "details":
                    continue
                icon = "🟢" if "healthy" in str(val) or val == "healthy" else "🔴"
                st.markdown(f"{icon} **{key}**: `{val}`")
            if health.get("details"):
                with st.expander("Details"):
                    st.json(health["details"])
        except Exception as e:
            st.error(f"Health check failed: {e}")

    with col2:
        st.subheader("Cache")
        try:
            stats = api_cache_stats()
            st.metric("Local cache entries", stats["local_cache_size"])
            st.metric("TTL (seconds)", stats["ttl_seconds"])
            st.markdown(
                f"Redis: {'🟢 Connected' if stats['redis_available'] else '🔴 Not available (using local)'}"
            )
            if st.button("🗑️ Clear Cache", type="secondary"):
                api_clear_cache()
                st.success("Cache cleared")
        except Exception as e:
            st.error(f"Cache stats failed: {e}")

    st.divider()
    st.subheader("Configuration")
    config = {
        "LLM Provider": settings.LLM_PROVIDER,
        "Voyage Model": settings.VOYAGE_MODEL,
        "Chunk Size (tokens)": settings.CHUNK_SIZE,
        "Retrieval Top-K": settings.RETRIEVAL_TOP_K,
        "Rerank Top-N": settings.RERANK_TOP_N,
        "Topic Similarity Threshold": settings.TOPIC_SIMILARITY_THRESHOLD,
        "Faithfulness Threshold": settings.FAITHFULNESS_THRESHOLD,
        "Cache TTL (seconds)": settings.CACHE_TTL_SECONDS,
    }
    for k, v in config.items():
        st.markdown(f"**{k}**: `{v}`")


# ============================================================
# App Entry Point
# ============================================================

def main():
    page = render_sidebar()

    if "💬 Chat" in page:
        render_chat()
    elif "📄 Documents" in page:
        render_documents()
    elif "⚙️ Admin" in page:
        render_admin()


if __name__ == "__main__":
    main()
