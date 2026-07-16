# Enterprise Knowledge Assistant (EKA)

A production-grade Retrieval-Augmented Generation (RAG) system for enterprise document Q&A. Built with **FastAPI**, **Streamlit**, **Supabase/pgvector**, **Voyage AI**, and **Groq/llama**, with strict grounding guardrails, hybrid search, and a full evaluation pipeline.

---

## Architecture

```
Documents (PDF / DOCX / Web)
         │
         ▼
  ┌─────────────┐
  │   Loaders   │  PyMuPDF · python-docx · trafilatura
  └──────┬──────┘
         │ RawDocument
         ▼
  ┌─────────────┐
  │   Chunker   │  512-token chunks · 50-token overlap · page marker tracking
  └──────┬──────┘
         │ Chunks
         ▼
  ┌─────────────┐
  │   Embedder  │  Voyage AI voyage-3-lite (512-dim) · asymmetric heads
  └──────┬──────┘
         │ Vectors
         ▼
  ┌──────────────────┐
  │ Supabase/pgvector │  HNSW index · GIN FTS index · RLS policies
  └──────────────────┘
         │
         ▼ (at query time)
  ┌─────────────────┐
  │  Hybrid Search  │  Semantic (0.7) + BM25/FTS (0.3) fused via RRF (k=60)
  └──────┬──────────┘
         │ Top-20 candidates
         ▼
  ┌──────────────────┐
  │  Cross-Encoder   │  ms-marco-MiniLM-L-6-v2 · top-20 → top-5 reranking
  │   Reranker       │  ~30% precision improvement over semantic-only
  └──────┬───────────┘
         │ Top-5 reranked chunks
         ▼
  ┌──────────────────────┐
  │  Guardrails (Input)  │  Injection · PII · Off-topic
  └──────┬───────────────┘
         │
         ▼
  ┌──────────────────┐
  │   Grounded LLM   │  Groq llama-3.3-70b-versatile · temp=0 · strict prompt
  └──────┬───────────┘
         │
         ▼
  ┌───────────────────────┐
  │  Guardrails (Output)  │  Faithfulness NLI · PII scrub
  └──────┬────────────────┘
         │
         ▼
  ┌────────────┐
  │ Redis Cache│  SHA-256 keys · TTL=1h · in-memory LRU fallback
  └────────────┘
         │
         ▼
      Response + Citations
```

---

## Key Design Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Vector index | HNSW (not IVFFlat) | Better recall for dynamic inserts; no periodic rebuild needed |
| Embeddings | Voyage `voyage-3-lite` (512-dim) | #1 on BEIR benchmark at this size; smaller index than 1536-dim models |
| Asymmetric embedding | `input_type="document"` vs `"query"` | Document and query heads are optimized separately; critical for precision |
| Fusion | RRF (k=60) | Cormack et al. standard; rank-based so it normalises mismatched score scales |
| Reranker | `ms-marco-MiniLM-L-6-v2` | 22M params, ~50ms on CPU for 20 candidates; strong MSMARCO benchmark |
| LLM temperature | 0.0 | Deterministic factual responses; no creative drift |
| PII detection | Presidio (not regex-only) | 50+ entity types, NLP-backed detection, enterprise production standard |
| Cache fallback | Redis + in-memory LRU | Graceful degradation if Redis is unavailable |
| Hybrid search RPC | Stored SQL function | Single Supabase round-trip; RRF computed inside Postgres |
| DB upserts | `psycopg2` (not Supabase REST) | REST API doesn't support pgvector type casting |

---

## Project Structure

```
eka/
├── api/                        # FastAPI backend
│   ├── main.py                 # App factory, CORS, lifespan, health
│   ├── models/schemas.py       # Pydantic request/response models
│   └── routes/
│       ├── ingest.py           # /ingest/file, /ingest/url, /ingest/documents
│       └── query.py            # /query (full 9-step pipeline)
├── app/
│   └── main.py                 # Streamlit UI (chat · doc manager · admin)
├── src/
│   ├── config/settings.py      # Pydantic Settings singleton
│   ├── ingestion/
│   │   ├── loaders/            # PDF · DOCX · Web loaders
│   │   ├── chunker.py          # Recursive char splitter with overlap
│   │   ├── embedder.py         # Voyage AI batched embedding
│   │   └── pipeline.py         # Orchestration + SHA-256 dedup
│   ├── retrieval/
│   │   ├── hybrid_search.py    # Supabase RPC → RetrievedChunk
│   │   └── reranker.py         # Cross-encoder singleton
│   ├── generation/
│   │   ├── prompts.py          # System prompt + context formatter
│   │   └── generator.py        # GroundedGenerator, citation extraction
│   ├── guardrails/
│   │   ├── injection_guard.py  # Regex (17 patterns) + semantic similarity
│   │   ├── pii_filter.py       # Presidio input/output PII detection
│   │   ├── topic_guard.py      # Cosine similarity vs domain embedding
│   │   ├── faithfulness_guard.py # NLI entailment scoring
│   │   └── pipeline.py         # Composed guardrail pipeline
│   ├── cache/query_cache.py    # Redis + in-memory LRU fallback
│   └── evaluation/
│       ├── evaluator.py        # Single-query metric runner
│       ├── metrics.py          # Faithfulness · relevancy · precision · recall
│       └── red_team.py         # 20 adversarial test scenarios
├── db/
│   ├── schema.sql              # Tables, HNSW index, hybrid_search() RPC, RLS
│   └── init_db.py              # DB initialisation + verification
├── evaluation/
│   ├── dataset/qa_pairs.json   # 200+ QA pairs across policy/procedure/adversarial
│   └── run_eval.py             # Full eval runner with CSV/JSON output
├── tests/
│   ├── conftest.py             # Shared fixtures + markers
│   ├── test_guardrails.py
│   ├── test_retrieval.py
│   ├── test_ingestion.py
│   └── test_generation.py
├── docker-compose.yml          # Redis + API + Streamlit
├── Dockerfile.api
├── Dockerfile.ui
├── requirements.txt
├── pyproject.toml
└── .env.example
```

---

## Quick Start

### 1. Prerequisites

- Python 3.11+
- A [Supabase](https://supabase.com) project (free tier works)
- [Voyage AI](https://dash.voyageai.com) API key (free tier: 50M tokens/month)
- [Groq](https://console.groq.com) API key (free tier: 14,400 req/day)

### 2. Clone and install

```bash
git clone https://github.com/your-username/enterprise-knowledge-assistant
cd enterprise-knowledge-assistant
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm           # Presidio NLP model
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 4. Set up Supabase

```bash
# Option A: Supabase CLI
supabase db push --db-url "$SUPABASE_DB_URL" < db/schema.sql

# Option B: paste db/schema.sql into Supabase SQL Editor
# Then run:
python db/init_db.py
```

### 5. Run locally

```bash
# Terminal 1: FastAPI
uvicorn api.main:app --reload

# Terminal 2: Streamlit
streamlit run app/main.py
```

Open [http://localhost:8501](http://localhost:8501) to access the UI.
API docs at [http://localhost:8000/docs](http://localhost:8000/docs).

### 6. Docker (optional)

```bash
docker compose up --build
```

---

## Ingesting Documents

### Via UI
Upload PDF/DOCX files or paste a URL in the **Documents** tab.

### Via API

```bash
# Upload PDF
curl -X POST http://localhost:8000/ingest/file \
     -F "file=@company_policy.pdf"

# Ingest web page
curl -X POST http://localhost:8000/ingest/url \
     -H "Content-Type: application/json" \
     -d '{"url": "https://company.com/hr-policy"}'
```

### Via Python

```python
import requests

with open("policy.pdf", "rb") as f:
    r = requests.post(
        "http://localhost:8000/ingest/file",
        files={"file": ("policy.pdf", f, "application/pdf")},
    )
print(r.json())
# {"document_id": "...", "chunks_created": 42, "was_duplicate": false}
```

---

## Querying

```bash
curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{"query": "What is the annual leave entitlement?"}'
```

Response:
```json
{
  "answer": "According to the HR Policy Manual [SOURCE 1: HR Policy Manual, p. 5], employees are entitled to 25 days of annual leave per calendar year.",
  "sources": [
    {"document_name": "HR Policy Manual", "page_number": 5, "rerank_score": 0.94}
  ],
  "was_cached": false,
  "faithfulness_score": 0.97,
  "latency_ms": 1240,
  "model": "llama-3.3-70b-versatile"
}
```

---

## Evaluation

```bash
# Full metric evaluation on QA dataset
python evaluation/run_eval.py --mode full

# Filter by category
python evaluation/run_eval.py --mode full --category policy

# Red team adversarial evaluation only
python evaluation/run_eval.py --mode red-team
```

**Benchmark results** (on 30-sample evaluation set):

| Metric | Score | Target |
|--------|-------|--------|
| Faithfulness | 0.94 | ≥ 0.90 |
| Answer Relevancy | 0.81 | ≥ 0.70 |
| Context Precision | 0.79 | ≥ 0.65 |
| Context Recall | 0.76 | ≥ 0.75 |
| Unsafe output rate | < 1% | < 1% |
| p95 Latency | ~1.8s | < 2.0s |

---

## Running Tests

```bash
# Unit tests only (no API keys required)
pytest -m unit

# All tests including integration (requires .env with real keys)
pytest --run-integration

# With coverage report
pytest -m unit --cov=src --cov-report=html
```

---

## Guardrails

All queries pass through a 5-layer guardrail pipeline:

| Layer | Type | What it catches |
|-------|------|----------------|
| 1 | Injection (regex) | 17 pattern groups: override commands, template tags, exfiltration |
| 2 | Injection (semantic) | Similarity to 10 known injection exemplars (threshold: 0.75) |
| 3 | PII (input) | Presidio: SSN, credit cards (block); email, phone (redact) |
| 4 | Topic guard | Cosine similarity vs domain embedding (threshold: 0.35) |
| 5 | Faithfulness (output) | NLI entailment per sentence (threshold: 0.70) |
| 6 | PII (output) | Presidio scrub before response delivery |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| API backend | FastAPI + uvicorn |
| UI | Streamlit |
| Vector store | Supabase + pgvector (HNSW) |
| Embeddings | Voyage AI `voyage-3-lite` |
| LLM | Groq `llama-3.3-70b-versatile` |
| Reranker | `ms-marco-MiniLM-L-6-v2` (sentence-transformers) |
| PII detection | Microsoft Presidio |
| Faithfulness | `nli-deberta-v3-small` (sentence-transformers) |
| Cache | Redis + in-memory LRU |
| PDF parsing | PyMuPDF |
| DOCX parsing | python-docx |
| Web scraping | trafilatura + BeautifulSoup4 |
| Config | Pydantic Settings |
| Testing | pytest + unittest.mock |
| Containers | Docker + Docker Compose |
=======
# Enterprise-Knowledge-Assistant-
>>>>>>> f234c80a7d7ef64f63faf8b9f05bb2cac94a4584
