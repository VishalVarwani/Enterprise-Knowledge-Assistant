"""
app/main.py — Enterprise Knowledge Assistant UI
Redesigned: dark slate theme, explicit text colours, proper chat bubbles.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path
import re

import httpx
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config.settings import get_settings

st.set_page_config(
    page_title="EKA · Enterprise Knowledge Assistant",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ── Reset & base ─────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
    background: #0d1117 !important;
    color: #e6edf3 !important;
}

/* ── Sidebar ──────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #161b22 !important;
    border-right: 1px solid #30363d;
}
[data-testid="stSidebar"] * { color: #e6edf3 !important; }
[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }

/* ── Main container ───────────────────────────────────────── */
.main .block-container {
    padding: 2rem 3rem 4rem 3rem;
    max-width: 900px;
}

/* ── Page title ───────────────────────────────────────────── */
h1 { color: #e6edf3 !important; font-weight: 700; letter-spacing: -0.5px; }
h2, h3 { color: #c9d1d9 !important; }

/* ── Chat bubbles ─────────────────────────────────────────── */
.bubble-user {
    background: #1f2937;
    color: #f0f6fc !important;
    border-left: 3px solid #58a6ff;
    padding: 12px 16px;
    border-radius: 0 10px 10px 0;
    margin: 12px 0 4px 0;
    font-size: 0.95rem;
    line-height: 1.6;
}
.bubble-assistant {
    background: #161b22;
    color: #e6edf3 !important;
    border: 1px solid #30363d;
    border-left: 3px solid #3fb950;
    padding: 14px 18px;
    border-radius: 0 10px 10px 0;
    margin: 4px 0 4px 0;
    font-size: 0.95rem;
    line-height: 1.7;
}
.bubble-blocked {
    background: #1a1206;
    color: #e6edf3 !important;
    border-left: 3px solid #d29922;
    padding: 12px 16px;
    border-radius: 0 10px 10px 0;
    margin: 4px 0;
    font-size: 0.95rem;
}
.bubble-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #8b949e !important;
    margin-bottom: 2px;
}

/* ── Source cards ─────────────────────────────────────────── */
.src-card {
    display: flex;
    align-items: center;
    gap: 10px;
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 8px 14px;
    margin: 5px 0;
    font-size: 0.84rem;
    color: #c9d1d9 !important;
}
.src-rank {
    background: #21262d;
    color: #58a6ff !important;
    border-radius: 4px;
    padding: 1px 6px;
    font-size: 0.75rem;
    font-weight: 700;
    min-width: 22px;
    text-align: center;
}
.src-name { color: #e6edf3 !important; font-weight: 600; }
.src-page { color: #8b949e !important; font-size: 0.8rem; }
.src-score {
    margin-left: auto;
    background: #1f2937;
    border-radius: 12px;
    padding: 2px 10px;
    font-size: 0.78rem;
    font-weight: 600;
    color: #3fb950 !important;
}

/* ── Meta bar ─────────────────────────────────────────────── */
.meta-bar {
    display: flex;
    gap: 14px;
    flex-wrap: wrap;
    font-size: 0.76rem;
    color: #8b949e !important;
    margin: 6px 0 14px 4px;
    align-items: center;
}
.meta-item { display: flex; align-items: center; gap: 4px; }
.meta-good { color: #3fb950 !important; font-weight: 600; }
.meta-warn { color: #d29922 !important; font-weight: 600; }
.meta-bad  { color: #f85149 !important; font-weight: 600; }
.pill-cached {
    background: #0c2d6b;
    color: #79c0ff !important;
    border-radius: 10px;
    padding: 1px 8px;
    font-size: 0.72rem;
    font-weight: 600;
}

/* ── Status dots ──────────────────────────────────────────── */
.dot-green { color: #3fb950 !important; }
.dot-red   { color: #f85149 !important; }

/* ── Inputs ───────────────────────────────────────────────── */
[data-testid="stChatInput"] > div {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 12px !important;
}
[data-testid="stChatInput"] textarea {
    color: #e6edf3 !important;
    background: transparent !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: #484f58 !important; }

/* ── Expander (sources) ───────────────────────────────────── */
[data-testid="stExpander"] {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 10px !important;
}
[data-testid="stExpander"] summary { color: #8b949e !important; font-size: 0.82rem; }

/* ── Buttons ──────────────────────────────────────────────── */
[data-testid="stButton"] button {
    background: #21262d !important;
    color: #c9d1d9 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
}
[data-testid="stButton"] button:hover {
    background: #30363d !important;
    border-color: #58a6ff !important;
    color: #e6edf3 !important;
}

/* ── File uploader ────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    background: #161b22 !important;
    border: 1px dashed #30363d !important;
    border-radius: 10px !important;
    color: #8b949e !important;
}

/* ── Tabs ─────────────────────────────────────────────────── */
[data-testid="stTabs"] [role="tablist"] {
    background: transparent !important;
    border-bottom: 1px solid #30363d;
}
[data-testid="stTabs"] button[role="tab"] {
    color: #8b949e !important;
    font-size: 0.88rem;
}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    color: #e6edf3 !important;
    border-bottom: 2px solid #58a6ff !important;
}

/* ── Metrics ──────────────────────────────────────────────── */
[data-testid="stMetric"] { background: #161b22; border-radius: 8px; padding: 12px; }
[data-testid="stMetricLabel"] { color: #8b949e !important; font-size: 0.78rem !important; }
[data-testid="stMetricValue"] { color: #e6edf3 !important; }

/* ── Divider ──────────────────────────────────────────────── */
hr { border-color: #21262d !important; }

/* ── Radio ────────────────────────────────────────────────── */
[data-testid="stRadio"] label { color: #c9d1d9 !important; font-size: 0.9rem; }

/* ── Info / success / error alerts ───────────────────────── */
[data-testid="stAlert"] { border-radius: 8px !important; }

/* ── Caption ──────────────────────────────────────────────── */
[data-testid="stCaptionContainer"] { color: #8b949e !important; }

/* ── Hide streamlit footer ────────────────────────────────── */
footer { visibility: hidden; }
#MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

settings = get_settings()
API_BASE = f"http://localhost:{settings.API_PORT}"


# ── Helpers ────────────────────────────────────────────────────────────────

def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


# ── API client ─────────────────────────────────────────────────────────────

def _get(path: str, timeout: float = 10.0) -> dict:
    with httpx.Client(timeout=timeout) as c:
        r = c.get(f"{API_BASE}{path}")
        r.raise_for_status()
        return r.json()


def _post(path: str, **kwargs) -> dict:
    timeout = kwargs.pop("timeout", 60.0)
    with httpx.Client(timeout=timeout) as c:
        r = c.post(f"{API_BASE}{path}", **kwargs)
        r.raise_for_status()
        return r.json()


def _delete(path: str) -> dict:
    with httpx.Client(timeout=15.0) as c:
        r = c.delete(f"{API_BASE}{path}")
        r.raise_for_status()
        return r.json()


# ── Sidebar ────────────────────────────────────────────────────────────────

def render_sidebar() -> str:
    with st.sidebar:
        st.markdown(
            '<div style="font-size:1.3rem;font-weight:700;color:#e6edf3;'
            'letter-spacing:-0.3px;padding:4px 0 2px 0;">🏢 EKA</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="font-size:0.75rem;color:#8b949e;">Enterprise Knowledge Assistant'
            f' · v{settings.APP_VERSION}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

        page = st.radio(
            "nav",
            ["💬  Chat", "📄  Documents", "⚙️  Admin"],
            label_visibility="collapsed",
        )

        st.markdown("<hr style='margin:16px 0'>", unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size:0.72rem;font-weight:600;letter-spacing:0.06em;'
            'text-transform:uppercase;color:#484f58;margin-bottom:8px;">System</div>',
            unsafe_allow_html=True,
        )

        try:
            h = _get("/health", timeout=4.0)
            db = h.get("database", "")
            llm = h.get("details", {}).get("llm_provider", "—")
            emb = h.get("details", {}).get("voyage_model", "—")
            ok = "healthy" in str(db)
            st.markdown(
                f'<div style="font-size:0.82rem;line-height:2;">'
                f'<span class="{"dot-green" if ok else "dot-red"}">{"●" if ok else "●"}</span>'
                f' Database<br>'
                f'<span class="dot-green">●</span> LLM: <code>{llm}</code><br>'
                f'<span class="dot-green">●</span> Embedder: <code>{emb}</code>'
                f'</div>',
                unsafe_allow_html=True,
            )
        except Exception:
            st.markdown(
                '<div style="font-size:0.82rem;color:#f85149;">● API offline</div>',
                unsafe_allow_html=True,
            )

        return page


# ── Chat ───────────────────────────────────────────────────────────────────

def render_chat():
    st.markdown(
        '<h1 style="margin-bottom:4px;">Enterprise Knowledge Assistant</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="color:#8b949e;font-size:0.88rem;margin-bottom:24px;">'
        'Answers are grounded in your uploaded documents — every claim is cited.'
        '</div>',
        unsafe_allow_html=True,
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # ── Render history ─────────────────────────────────────────────────────
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="bubble-label">You</div>'
                f'<div class="bubble-user">{msg["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            bubble = "bubble-blocked" if msg.get("blocked") else "bubble-assistant"
            st.markdown(
                f'<div class="bubble-label">Assistant</div>'
                f'<div class="{bubble}">{msg["content"]}</div>',
                unsafe_allow_html=True,
            )
            if msg.get("sources"):
                _render_sources(msg["sources"])
            if msg.get("meta"):
                _render_meta(msg["meta"])

    # ── Input ──────────────────────────────────────────────────────────────
    query = st.chat_input("Ask about company policies, procedures, benefits…")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        st.markdown(
            f'<div class="bubble-label">You</div>'
            f'<div class="bubble-user">{query}</div>',
            unsafe_allow_html=True,
        )

        with st.spinner(""):
            try:
                t0 = time.perf_counter()
                result = _post(
                    "/query",
                    json={"query": query, "use_cache": True},
                    timeout=90.0,
                )
                latency_ms = int((time.perf_counter() - t0) * 1000)

                raw_answer = result.get("answer", "")
                # Strip [Source: ...] and [SOURCE N: ...] tags — these are for internal citation
                # tracking, not for users to read
                answer = re.sub(r'\[Source[^\]]*\]', '', raw_answer).strip()
                answer = re.sub(r'\s+', ' ', answer)  # clean up extra spaces left behind
                sources = result.get("sources", [])
                blocked = result.get("was_blocked", False)

                bubble = "bubble-blocked" if blocked else "bubble-assistant"
                st.markdown(
                    f'<div class="bubble-label">Assistant</div>'
                    f'<div class="{bubble}">{answer}</div>',
                    unsafe_allow_html=True,
                )

                if sources and not blocked:
                    _render_sources(sources)

                meta = {
                    "cached":            result.get("was_cached", False),
                    "blocked":           blocked,
                    "block_reason":      result.get("block_reason", ""),
                    "latency_ms":        result.get("latency_ms", latency_ms),
                    "faithfulness":      result.get("faithfulness_score"),
                    "grounding":         result.get("grounding_score"),
                    "tokens":            (result.get("prompt_tokens", 0) or 0)
                                       + (result.get("completion_tokens", 0) or 0),
                    "model":             result.get("model", ""),
                }
                _render_meta(meta)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "blocked": blocked,
                    "sources": sources if not blocked else [],
                    "meta": meta,
                })

            except httpx.ConnectError:
                st.error("API is not running. Start it with `uvicorn api.main:app --reload`")
            except httpx.HTTPStatusError as e:
                st.error(f"API returned {e.response.status_code}: {e.response.text[:300]}")
            except Exception as e:
                st.error(f"Error: {e}")

    if st.session_state.messages:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Clear conversation"):
            st.session_state.messages = []
            st.rerun()


def _render_sources(sources: list[dict]):
    if not sources:
        return
    with st.expander(f"📎 {len(sources)} source{'s' if len(sources) != 1 else ''} used", expanded=False):
        for i, src in enumerate(sources, 1):
            raw = src.get("rerank_score", 0) or 0
            pct = _sigmoid(raw)
            page_str = f", p. {src['page_number']}" if src.get("page_number") else ""
            icon = {"pdf": "📄", "docx": "📝", "web": "🌐"}.get(
                src.get("doc_source_type", ""), "📄"
            )
            name = src.get("document_name", "Unknown")
            st.markdown(
                f'<div class="src-card">'
                f'<span class="src-rank">{i}</span>'
                f'{icon}&nbsp;'
                f'<span class="src-name">{name}</span>'
                f'<span class="src-page">{page_str}</span>'
                f'<span class="src-score">{pct:.0%} match</span>'
                f'</div>',
                unsafe_allow_html=True,
            )


def _render_meta(meta: dict):
    parts = []

    if meta.get("cached"):
        parts.append('<span class="pill-cached">⚡ Cached</span>')

    if meta.get("blocked"):
        reason = meta.get("block_reason", "")
        parts.append(f'<span class="meta-warn">🛡 Blocked{": " + reason if reason else ""}</span>')

    ms = meta.get("latency_ms")
    if ms:
        parts.append(f'<span class="meta-item">⏱ {ms:,}ms</span>')

    f_score = meta.get("faithfulness")
    if f_score is not None:
        cls = "meta-good" if f_score >= 0.8 else "meta-warn" if f_score >= 0.5 else "meta-bad"
        parts.append(f'<span class="{cls}">🎯 Faith: {f_score:.0%}</span>')

    g_score = meta.get("grounding")
    if g_score is not None and g_score > 0:
        cls = "meta-good" if g_score >= 0.7 else "meta-warn"
        parts.append(f'<span class="{cls}">📌 Grounding: {g_score:.0%}</span>')

    tokens = meta.get("tokens")
    if tokens:
        parts.append(f'<span class="meta-item" style="color:#484f58;">{tokens:,} tokens</span>')

    model = meta.get("model", "")
    if model:
        parts.append(f'<span class="meta-item" style="color:#484f58;">{model}</span>')

    if parts:
        st.markdown(
            '<div class="meta-bar">' + "".join(parts) + "</div>",
            unsafe_allow_html=True,
        )


# ── Documents ──────────────────────────────────────────────────────────────

def render_documents():
    st.markdown('<h1>Document Manager</h1>', unsafe_allow_html=True)
    st.markdown(
        '<div style="color:#8b949e;font-size:0.88rem;margin-bottom:24px;">'
        'Upload PDFs or DOCX files — the app chunks, embeds, and indexes them automatically.'
        '</div>',
        unsafe_allow_html=True,
    )

    tab_up, tab_url, tab_list = st.tabs(["  Upload file  ", "  Ingest URL  ", "  View documents  "])

    with tab_up:
        st.markdown("<br>", unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Drop a PDF or Word document here",
            type=["pdf", "docx"],
            label_visibility="visible",
        )
        if uploaded:
            st.markdown(
                f'<div style="color:#8b949e;font-size:0.82rem;margin:6px 0 12px 0;">'
                f'Ready: <strong style="color:#e6edf3">{uploaded.name}</strong>'
                f' ({len(uploaded.getvalue()) / 1024:.0f} KB)</div>',
                unsafe_allow_html=True,
            )
            if st.button("Ingest document →", type="primary"):
                with st.spinner(f"Processing {uploaded.name}…"):
                    try:
                        r = _post(
                            "/ingest/file",
                            files={"file": (uploaded.name, uploaded.getvalue())},
                            timeout=120.0,
                        )
                        if r.get("was_duplicate"):
                            st.info(f"Already in knowledge base — no changes made.")
                        else:
                            st.success(f"✓ Ingested · {r['chunks_created']} chunks indexed")
                    except Exception as e:
                        st.error(f"Failed: {e}")

    with tab_url:
        st.markdown("<br>", unsafe_allow_html=True)
        url = st.text_input(
            "Web page URL",
            placeholder="https://company.com/hr-policy",
            label_visibility="visible",
        )
        if url and st.button("Ingest URL →", type="primary"):
            with st.spinner("Fetching and indexing…"):
                try:
                    r = _post("/ingest/url", json={"url": url})
                    if r.get("was_duplicate"):
                        st.info("Already in knowledge base.")
                    else:
                        st.success(f"✓ Ingested · {r['chunks_created']} chunks indexed")
                except Exception as e:
                    st.error(f"Failed: {e}")

    with tab_list:
        st.markdown("<br>", unsafe_allow_html=True)
        col_r, col_s = st.columns([1, 6])
        with col_r:
            if st.button("↻ Refresh"):
                st.rerun()

        try:
            data = _get("/ingest/documents")
            docs = data.get("documents", [])
            if not docs:
                st.markdown(
                    '<div style="color:#8b949e;text-align:center;padding:40px;">'
                    'No documents yet. Upload one in the Upload tab.'
                    '</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div style="color:#8b949e;font-size:0.8rem;margin-bottom:12px;">'
                    f'{data["total"]} document(s) in knowledge base</div>',
                    unsafe_allow_html=True,
                )
                for doc in docs:
                    icon = {"pdf": "📄", "docx": "📝", "web": "🌐"}.get(
                        doc.get("source_type", ""), "📄"
                    )
                    col1, col2, col3, col4 = st.columns([5, 2, 2, 1])
                    with col1:
                        st.markdown(
                            f'<div style="color:#e6edf3;font-weight:600;font-size:0.9rem;">'
                            f'{icon} {doc["name"]}</div>'
                            f'<div style="color:#484f58;font-size:0.76rem;">'
                            f'{doc.get("source_path","")[:70]}</div>',
                            unsafe_allow_html=True,
                        )
                    with col2:
                        st.markdown(
                            f'<div style="color:#8b949e;font-size:0.8rem;">'
                            f'{doc["total_chunks"]} chunks</div>',
                            unsafe_allow_html=True,
                        )
                    with col3:
                        st.markdown(
                            f'<div style="color:#8b949e;font-size:0.8rem;">'
                            f'{doc.get("created_at","")[:10]}</div>',
                            unsafe_allow_html=True,
                        )
                    with col4:
                        if st.button("🗑", key=f"del_{doc['id']}"):
                            try:
                                _delete(f"/ingest/documents/{doc['id']}")
                                st.rerun()
                            except Exception as e:
                                st.error(str(e))
                    st.markdown(
                        '<hr style="margin:8px 0;border-color:#21262d;">',
                        unsafe_allow_html=True,
                    )
        except Exception as e:
            st.error(f"Could not load documents: {e}")


# ── Admin ──────────────────────────────────────────────────────────────────

def render_admin():
    st.markdown('<h1>Admin</h1>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            '<div style="font-size:0.85rem;font-weight:600;color:#c9d1d9;'
            'margin-bottom:12px;">System health</div>',
            unsafe_allow_html=True,
        )
        try:
            h = _get("/health")
            rows = [(k, v) for k, v in h.items() if k != "details"]
            for k, v in rows:
                ok = "healthy" in str(v) or v == "healthy"
                dot = '<span style="color:#3fb950">●</span>' if ok else '<span style="color:#f85149">●</span>'
                st.markdown(
                    f'<div style="font-size:0.85rem;padding:4px 0;">'
                    f'{dot} <strong style="color:#c9d1d9">{k}</strong>: '
                    f'<code style="color:#8b949e">{v}</code></div>',
                    unsafe_allow_html=True,
                )
            if h.get("details"):
                with st.expander("Details"):
                    st.json(h["details"])
        except Exception as e:
            st.error(f"Health check failed: {e}")

    with col2:
        st.markdown(
            '<div style="font-size:0.85rem;font-weight:600;color:#c9d1d9;'
            'margin-bottom:12px;">Query cache</div>',
            unsafe_allow_html=True,
        )
        try:
            s = _get("/cache/stats")
            c1, c2 = st.columns(2)
            c1.metric("Cached queries", s.get("local_cache_size", 0))
            c2.metric("TTL", f"{s.get('ttl_seconds', 0)}s")
            redis = s.get("redis_available", False)
            st.markdown(
                f'<div style="font-size:0.82rem;margin:8px 0;">'
                f'Redis: {"<span style=\'color:#3fb950\'>● connected</span>" if redis else "<span style=\'color:#d29922\'>● using local fallback</span>"}'
                f'</div>',
                unsafe_allow_html=True,
            )
            if st.button("Clear cache"):
                with httpx.Client(timeout=10.0) as c:
                    c.post(f"{API_BASE}/cache/clear")
                st.success("Cache cleared")
        except Exception as e:
            st.error(f"Cache stats failed: {e}")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:0.85rem;font-weight:600;color:#c9d1d9;'
        'margin-bottom:12px;">Active configuration</div>',
        unsafe_allow_html=True,
    )
    cfg = {
        "LLM provider":              settings.LLM_PROVIDER,
        "Embedding model":           settings.VOYAGE_MODEL,
        "Chunk size":                f"{settings.CHUNK_SIZE} tokens",
        "Retrieval top-K":           settings.RETRIEVAL_TOP_K,
        "Rerank top-N":              settings.RERANK_TOP_N,
        "Topic threshold":           settings.TOPIC_SIMILARITY_THRESHOLD,
        "Faithfulness threshold":    settings.FAITHFULNESS_THRESHOLD,
        "Cache TTL":                 f"{settings.CACHE_TTL_SECONDS}s",
    }
    rows_html = "".join(
        f'<div style="display:flex;justify-content:space-between;padding:6px 0;'
        f'border-bottom:1px solid #21262d;font-size:0.84rem;">'
        f'<span style="color:#8b949e">{k}</span>'
        f'<code style="color:#58a6ff">{v}</code></div>'
        for k, v in cfg.items()
    )
    st.markdown(
        f'<div style="background:#161b22;border:1px solid #30363d;'
        f'border-radius:10px;padding:4px 16px;">{rows_html}</div>',
        unsafe_allow_html=True,
    )


# ── Entry point ────────────────────────────────────────────────────────────

def main():
    page = render_sidebar()
    if "Chat" in page:
        render_chat()
    elif "Documents" in page:
        render_documents()
    elif "Admin" in page:
        render_admin()


if __name__ == "__main__":
    main()