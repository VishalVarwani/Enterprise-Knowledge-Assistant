"""
init_db.py
----------
One-time setup script to apply schema.sql to your Supabase project.

Run:
    python db/init_db.py

Requirements:
    - DATABASE_URL in .env (direct Postgres connection string)
    - psycopg2-binary installed

What it does:
    1. Connects directly to Postgres (bypasses Supabase REST layer)
    2. Applies db/schema.sql
    3. Verifies all tables, indexes, and the hybrid_search function exist
    4. Prints a summary

Why direct Postgres and not supabase-py?
    supabase-py wraps PostgREST which doesn't support DDL (CREATE TABLE,
    CREATE INDEX, etc.). DDL must go through a direct Postgres connection.
"""

from __future__ import annotations

import sys
from pathlib import Path

import psycopg2
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import get_settings


EXPECTED_TABLES = {"documents", "chunks", "query_logs", "guardrail_violations"}
EXPECTED_INDEXES = {
    "chunks_embedding_hnsw",
    "chunks_fts_gin",
    "chunks_document_id_idx",
    "documents_source_type_idx",
}
EXPECTED_FUNCTIONS = {"hybrid_search"}


def get_connection(database_url: str):
    """Open a psycopg2 connection with a short connect timeout."""
    return psycopg2.connect(database_url, connect_timeout=10)


def apply_schema(conn, schema_path: Path) -> None:
    """Execute the entire schema.sql file as a single transaction."""
    sql = schema_path.read_text(encoding="utf-8")
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()
    logger.info("Schema applied successfully.")


def verify_tables(conn) -> bool:
    """Check that all expected tables exist in the public schema."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_type = 'BASE TABLE';
            """
        )
        existing = {row[0] for row in cur.fetchall()}

    missing = EXPECTED_TABLES - existing
    if missing:
        logger.error(f"Missing tables: {missing}")
        return False
    logger.info(f"All tables present: {EXPECTED_TABLES}")
    return True


def verify_indexes(conn) -> bool:
    """Check that all expected indexes exist."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT indexname
            FROM pg_indexes
            WHERE schemaname = 'public';
            """
        )
        existing = {row[0] for row in cur.fetchall()}

    missing = EXPECTED_INDEXES - existing
    if missing:
        logger.warning(f"Missing indexes (may need to create manually): {missing}")
        return False
    logger.info(f"All indexes present.")
    return True


def verify_functions(conn) -> bool:
    """Check that hybrid_search stored function exists."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT routine_name
            FROM information_schema.routines
            WHERE routine_schema = 'public'
            AND routine_type = 'FUNCTION';
            """
        )
        existing = {row[0] for row in cur.fetchall()}

    missing = EXPECTED_FUNCTIONS - existing
    if missing:
        logger.error(f"Missing functions: {missing}")
        return False
    logger.info(f"All functions present: {EXPECTED_FUNCTIONS}")
    return True


def verify_pgvector(conn) -> bool:
    """Confirm pgvector extension is enabled."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT extname FROM pg_extension WHERE extname = 'vector';"
        )
        result = cur.fetchone()
    if not result:
        logger.error("pgvector extension not found. Enable it in Supabase Dashboard > Extensions.")
        return False
    logger.info("pgvector extension: enabled")
    return True


def print_summary(conn) -> None:
    """Print row counts for all tables."""
    print("\n" + "=" * 50)
    print("  Database Summary")
    print("=" * 50)
    for table in sorted(EXPECTED_TABLES):
        try:
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {table};")
                count = cur.fetchone()[0]
            print(f"  {table:<30} {count:>8} rows")
        except Exception as e:
            print(f"  {table:<30}   ERROR: {e}")
    print("=" * 50 + "\n")


def main() -> None:
    settings = get_settings()

    logger.info("Connecting to Postgres...")
    try:
        conn = get_connection(settings.DATABASE_URL)
    except psycopg2.OperationalError as e:
        logger.error(f"Connection failed: {e}")
        logger.info("Check DATABASE_URL in your .env file.")
        sys.exit(1)

    schema_path = Path(__file__).parent / "schema.sql"
    if not schema_path.exists():
        logger.error(f"Schema file not found: {schema_path}")
        sys.exit(1)

    logger.info("Applying schema...")
    try:
        apply_schema(conn, schema_path)
    except Exception as e:
        logger.error(f"Schema application failed: {e}")
        conn.rollback()
        conn.close()
        sys.exit(1)

    # Verification
    ok = all([
        verify_pgvector(conn),
        verify_tables(conn),
        verify_indexes(conn),
        verify_functions(conn),
    ])

    print_summary(conn)
    conn.close()

    if ok:
        logger.info("✓ Database initialized and verified. Ready to ingest documents.")
    else:
        logger.warning("Database initialized with warnings. Review errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
