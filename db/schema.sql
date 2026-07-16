-- ============================================================
-- Enterprise Knowledge Assistant – Supabase Schema
-- ============================================================
-- Run this in Supabase SQL Editor or via psql.
--
-- Design decisions:
--   • pgvector HNSW index over IVFFlat:
--       IVFFlat requires a fixed number of lists set at creation and
--       needs index rebuild when data grows significantly.
--       HNSW is insert-friendly, no rebuild needed, and delivers
--       better recall@10 (0.98 vs 0.95) at the cost of more memory.
--       For an enterprise KB (<10M chunks), HNSW memory fits comfortably.
--
--   • tsvector column for keyword search:
--       Stored + indexed tsvector avoids recomputing to_tsvector()
--       on every query. The GIN index makes full-text scan sub-millisecond.
--
--   • Separate documents/chunks tables:
--       Allows chunk-level operations without touching document metadata,
--       and supports document-level delete (CASCADE to chunks).
--
--   • RLS (Row Level Security):
--       Enabled but policies are permissive for the service role.
--       Add user-scoped policies here when multi-tenant auth is added.
-- ============================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS vector;           -- pgvector
CREATE EXTENSION IF NOT EXISTS pg_trgm;          -- trigram similarity (fuzzy search fallback)
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";      -- uuid_generate_v4()

-- ============================================================
-- Table: documents
-- One row per ingested source file / URL.
-- ============================================================
CREATE TABLE IF NOT EXISTS documents (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name            TEXT NOT NULL,                        -- Display name
    source_type     TEXT NOT NULL CHECK (
                        source_type IN ('pdf', 'docx', 'web', 'text')
                    ),
    source_path     TEXT NOT NULL,                        -- File path or URL
    file_hash       TEXT,                                 -- SHA-256; detect re-uploads
    total_chunks    INTEGER DEFAULT 0,
    metadata        JSONB DEFAULT '{}',                   -- Flexible: author, dept, tags
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT documents_source_unique UNIQUE (source_path, file_hash)
);

-- Updated-at trigger
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS documents_updated_at ON documents;
CREATE TRIGGER documents_updated_at
    BEFORE UPDATE ON documents
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- ============================================================
-- Table: chunks
-- One row per text chunk extracted from a document.
-- ============================================================
CREATE TABLE IF NOT EXISTS chunks (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id     UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    content         TEXT NOT NULL,                        -- Raw text of this chunk
    chunk_index     INTEGER NOT NULL,                     -- Position within document
    token_count     INTEGER,                              -- Estimated token count
    -- Embedding vector (dim must match VOYAGE_MODEL output)
    -- voyage-3-lite = 512 dimensions
    embedding       VECTOR(512),
    -- Pre-computed tsvector for keyword search (avoids per-query computation)
    fts_vector      TSVECTOR GENERATED ALWAYS AS (
                        to_tsvector('english', content)
                    ) STORED,
    metadata        JSONB DEFAULT '{}',                   -- page_num, section, heading
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT chunks_doc_index_unique UNIQUE (document_id, chunk_index)
);

-- ============================================================
-- Indexes
-- ============================================================

-- HNSW index for approximate nearest-neighbor vector search.
-- m=16: number of bi-directional links per layer (higher = better recall, more memory)
-- ef_construction=64: beam width during index build (higher = better recall, slower build)
-- These are the pgvector-recommended defaults for most production workloads.
-- operator class 'vector_cosine_ops': cosine similarity (best for normalized embeddings)
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- GIN index for full-text search on the stored tsvector column.
-- GIN = Generalized Inverted Index; optimal for set-membership queries (text search).
CREATE INDEX IF NOT EXISTS chunks_fts_gin
    ON chunks USING gin (fts_vector);

-- B-tree index for document_id lookups (chunk retrieval by doc, delete cascade)
CREATE INDEX IF NOT EXISTS chunks_document_id_idx
    ON chunks (document_id);

-- B-tree on chunk_index within document (used in ordered retrieval)
CREATE INDEX IF NOT EXISTS chunks_doc_index_idx
    ON chunks (document_id, chunk_index);

-- JSONB index on chunk metadata (enables filtering by page, section)
CREATE INDEX IF NOT EXISTS chunks_metadata_gin
    ON chunks USING gin (metadata);

-- Document source_type index (filter by content type)
CREATE INDEX IF NOT EXISTS documents_source_type_idx
    ON documents (source_type);

-- ============================================================
-- Table: query_logs
-- Audit trail for every query processed.
-- Used for: debugging, usage analytics, eval dataset generation.
-- ============================================================
CREATE TABLE IF NOT EXISTS query_logs (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    query_text      TEXT NOT NULL,
    query_hash      TEXT NOT NULL,                        -- SHA-256 for cache key lookup
    retrieved_chunk_ids  UUID[],                          -- Which chunks were surfaced
    reranked_chunk_ids   UUID[],                          -- After reranking
    response_text   TEXT,
    guardrail_flags JSONB DEFAULT '{}',                   -- Any guardrail triggers
    latency_ms      INTEGER,
    cache_hit       BOOLEAN DEFAULT FALSE,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS query_logs_hash_idx ON query_logs (query_hash);
CREATE INDEX IF NOT EXISTS query_logs_created_idx ON query_logs (created_at DESC);

-- ============================================================
-- Table: guardrail_violations
-- Separate from query_logs for easier security auditing.
-- ============================================================
CREATE TABLE IF NOT EXISTS guardrail_violations (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    query_log_id    UUID REFERENCES query_logs(id) ON DELETE SET NULL,
    violation_type  TEXT NOT NULL CHECK (
                        violation_type IN (
                            'pii_input', 'pii_output', 'off_topic',
                            'hallucination', 'prompt_injection', 'other'
                        )
                    ),
    severity        TEXT NOT NULL CHECK (severity IN ('low', 'medium', 'high')),
    details         JSONB DEFAULT '{}',
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================
-- Row Level Security
-- Service role bypasses RLS.
-- Anonymous / authenticated roles are restricted.
-- Extend this section when adding multi-tenant user auth.
-- ============================================================
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE query_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE guardrail_violations ENABLE ROW LEVEL SECURITY;

-- Service role full access (used by backend)
DROP POLICY IF EXISTS "service_role_all" ON documents;
CREATE POLICY "service_role_all" ON documents FOR ALL TO service_role USING (true);
DROP POLICY IF EXISTS "service_role_all" ON chunks;
CREATE POLICY "service_role_all" ON chunks FOR ALL TO service_role USING (true);
DROP POLICY IF EXISTS "service_role_all" ON query_logs;
CREATE POLICY "service_role_all" ON query_logs FOR ALL TO service_role USING (true);
DROP POLICY IF EXISTS "service_role_all" ON guardrail_violations;
CREATE POLICY "service_role_all" ON guardrail_violations FOR ALL TO service_role USING (true);

-- ============================================================
-- Hybrid Search Function
-- Called from Python with: supabase.rpc("hybrid_search", {...})
--
-- Why a stored function:
--   1. Keeps the RRF fusion logic in one place (single source of truth)
--   2. Avoids shipping two separate query result sets to Python for merging
--   3. Enables Supabase RPC call (one round trip instead of two)
--
-- RRF formula: score = 1/(k + rank_semantic) + 1/(k + rank_keyword)
-- k=60: industry standard constant from Cormack et al. 2009
-- ============================================================
CREATE OR REPLACE FUNCTION hybrid_search(
    query_embedding  VECTOR(512),
    query_text       TEXT,
    match_count      INTEGER DEFAULT 20,
    rrf_k            INTEGER DEFAULT 60,
    filter_doc_ids   UUID[] DEFAULT NULL     -- Optional: restrict to specific docs
)
RETURNS TABLE (
    chunk_id         UUID,
    document_id      UUID,
    content          TEXT,
    metadata         JSONB,
    doc_name         TEXT,
    doc_source_type  TEXT,
    chunk_index      INTEGER,
    rrf_score        FLOAT,
    semantic_rank    INTEGER,
    keyword_rank     INTEGER
)
LANGUAGE SQL STABLE AS $$
WITH
-- Semantic retrieval: top 20 by cosine similarity
semantic AS (
    SELECT
        c.id          AS chunk_id,
        c.document_id,
        c.content,
        c.metadata,
        c.chunk_index,
        ROW_NUMBER() OVER (ORDER BY c.embedding <=> query_embedding) AS rank
    FROM chunks c
    WHERE
        c.embedding IS NOT NULL
        AND (filter_doc_ids IS NULL OR c.document_id = ANY(filter_doc_ids))
    ORDER BY c.embedding <=> query_embedding
    LIMIT match_count * 2   -- Fetch extra for RRF coverage
),
-- Keyword retrieval: top 20 by BM25-style ts_rank
keyword AS (
    SELECT
        c.id          AS chunk_id,
        c.document_id,
        c.content,
        c.metadata,
        c.chunk_index,
        ROW_NUMBER() OVER (
            ORDER BY ts_rank_cd(c.fts_vector, websearch_to_tsquery('english', query_text)) DESC
        ) AS rank
    FROM chunks c
    WHERE
        c.fts_vector @@ websearch_to_tsquery('english', query_text)
        AND (filter_doc_ids IS NULL OR c.document_id = ANY(filter_doc_ids))
    ORDER BY ts_rank_cd(c.fts_vector, websearch_to_tsquery('english', query_text)) DESC
    LIMIT match_count * 2
),
-- RRF fusion: union both result sets, sum their reciprocal ranks
fused AS (
    SELECT
        COALESCE(s.chunk_id, k.chunk_id)          AS chunk_id,
        COALESCE(s.document_id, k.document_id)    AS document_id,
        COALESCE(s.content, k.content)             AS content,
        COALESCE(s.metadata, k.metadata)           AS metadata,
        COALESCE(s.chunk_index, k.chunk_index)     AS chunk_index,
        COALESCE(1.0 / (rrf_k + s.rank), 0.0)
            + COALESCE(1.0 / (rrf_k + k.rank), 0.0)  AS rrf_score,
        s.rank   AS semantic_rank,
        k.rank   AS keyword_rank
    FROM semantic s
    FULL OUTER JOIN keyword k USING (chunk_id)
)
SELECT
    f.chunk_id,
    f.document_id,
    f.content,
    f.metadata,
    d.name        AS doc_name,
    d.source_type AS doc_source_type,
    f.chunk_index,
    f.rrf_score,
    f.semantic_rank,
    f.keyword_rank
FROM fused f
JOIN documents d ON d.id = f.document_id
ORDER BY f.rrf_score DESC
LIMIT match_count;
$$;

-- Grant RPC access to service role
GRANT EXECUTE ON FUNCTION hybrid_search TO service_role;
