import os
import uuid
import logging
import psycopg2
from psycopg2 import pool
from typing import List, Optional

from src.config import (
    DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME, DB_TABLE,
    EMBEDDING_DIM, DEFAULT_TENANT_ID
)
from src.utils import monitor_performance, compute_text_hash, batch_items, retry_on_failure
from src.retrieval.embeddings import embed_query, embed_passages_batch
from src.retrieval.vector_store import VectorStore
from src.ingestion.document_model import Chunk, RetrievedChunk

# === Logging ===
logger = logging.getLogger("pg_vector_store")

DOCUMENTS_TABLE = "documents"


class PgVectorStore(VectorStore):
    """
    PostgreSQL + pgvector implementation of the VectorStore contract.

    Embeddings are always produced by the single shared `embeddings` module
    (never a locally-loaded model), so this class only owns document
    identity, storage, and retrieval — not encoding.
    """

    def __init__(self):
        self.connection_pool = None

    # === Connection Pool ===
    def init_connection_pool(self, minconn: int = 2, maxconn: int = 10):
        """Initialize PostgreSQL connection pool."""
        db_url = os.getenv("DATABASE_URL")

        try:
            if db_url:
                if "sslmode" not in db_url:
                    db_url += ("&" if "?" in db_url else "?") + "sslmode=allow"
                self.connection_pool = pool.ThreadedConnectionPool(minconn, maxconn, db_url)
            else:
                self.connection_pool = pool.ThreadedConnectionPool(
                    minconn, maxconn,
                    host=DB_HOST,
                    port=DB_PORT,
                    user=DB_USER,
                    password=DB_PASSWORD,
                    dbname=DB_NAME,
                    sslmode="allow",
                    connect_timeout=30
                )
            logger.info(f"✅ Connection pool initialized (min={minconn}, max={maxconn})")
        except Exception as e:
            logger.error(f"Failed to initialize connection pool: {e}")
            raise

    @retry_on_failure(max_retries=3, delay=1.0)
    def _get_connection(self):
        """Get a connection from the pool."""
        if self.connection_pool is None:
            self.init_connection_pool()
        return self.connection_pool.getconn()

    def _return_connection(self, conn):
        """Return a connection to the pool."""
        if self.connection_pool:
            self.connection_pool.putconn(conn)

    # === Schema ===
    @monitor_performance("init_schema")
    def init_schema(self):
        """
        Initialize (or migrate) the database schema. Every statement here is
        written to be safe to re-run on every app startup, including against
        a database that already has data from before this schema existed.
        """
        conn = self._get_connection()
        try:
            # --- Stage 1: extension + tables (guaranteed-safe, idempotent) ---
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {DOCUMENTS_TABLE} (
                        id TEXT PRIMARY KEY,
                        tenant_id TEXT NOT NULL DEFAULT '{DEFAULT_TENANT_ID}',
                        source_type TEXT NOT NULL DEFAULT 'url',
                        source_uri TEXT NOT NULL,
                        checksum VARCHAR(64),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE (tenant_id, source_uri)
                    );
                """)

                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {DB_TABLE} (
                        id SERIAL PRIMARY KEY,
                        chunk_text TEXT NOT NULL,
                        text_hash VARCHAR(64) NOT NULL,
                        embedding VECTOR({EMBEDDING_DIM}) NOT NULL,
                        document_name TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
            conn.commit()

            # --- Stage 2: bring an older table up to the current shape ---
            with conn.cursor() as cur:
                cur.execute(f"ALTER TABLE {DB_TABLE} ADD COLUMN IF NOT EXISTS document_id TEXT;")
                cur.execute(f"ALTER TABLE {DB_TABLE} ADD COLUMN IF NOT EXISTS tenant_id TEXT NOT NULL DEFAULT '{DEFAULT_TENANT_ID}';")
                cur.execute(f"ALTER TABLE {DB_TABLE} ADD COLUMN IF NOT EXISTS chunk_index INT;")
                cur.execute(f"ALTER TABLE {DB_TABLE} ADD COLUMN IF NOT EXISTS page_number INT;")
                cur.execute(f"ALTER TABLE {DB_TABLE} ADD COLUMN IF NOT EXISTS chunk_type TEXT DEFAULT 'paragraph';")
                cur.execute(f"ALTER TABLE {DB_TABLE} ADD COLUMN IF NOT EXISTS sheet_name TEXT;")
                # document_name used to be the only identifier and NOT NULL;
                # it's now a legacy/denormalized convenience column only —
                # new rows are identified by document_id instead.
                cur.execute(f"ALTER TABLE {DB_TABLE} ALTER COLUMN document_name DROP NOT NULL;")
            conn.commit()

            # --- Stage 3: backfill documents + document_id for pre-existing rows ---
            with conn.cursor() as cur:
                cur.execute(f"""
                    INSERT INTO {DOCUMENTS_TABLE} (id, tenant_id, source_type, source_uri)
                    SELECT DISTINCT md5('{DEFAULT_TENANT_ID}:' || document_name), '{DEFAULT_TENANT_ID}', 'url', document_name
                    FROM {DB_TABLE}
                    WHERE document_name IS NOT NULL AND document_id IS NULL
                    ON CONFLICT (tenant_id, source_uri) DO NOTHING;
                """)
                cur.execute(f"""
                    UPDATE {DB_TABLE}
                    SET document_id = md5('{DEFAULT_TENANT_ID}:' || document_name)
                    WHERE document_id IS NULL AND document_name IS NOT NULL;
                """)
                # Best-effort backfill of chunk_index from insertion order,
                # for rows that predate the column.
                cur.execute(f"""
                    WITH ordered AS (
                        SELECT id, ROW_NUMBER() OVER (PARTITION BY document_id ORDER BY id) - 1 AS rn
                        FROM {DB_TABLE}
                        WHERE chunk_index IS NULL AND document_id IS NOT NULL
                    )
                    UPDATE {DB_TABLE} c
                    SET chunk_index = ordered.rn
                    FROM ordered
                    WHERE c.id = ordered.id;
                """)
            conn.commit()

            # --- Stage 4: dedup scoped per document, not globally ---
            # The old schema had a single global UNIQUE(text_hash), which
            # silently dropped identical chunk text belonging to a different
            # document. Postgres names an inline column UNIQUE constraint
            # "<table>_<column>_key" by default, which is what's dropped here.
            with conn.cursor() as cur:
                cur.execute(f"ALTER TABLE {DB_TABLE} DROP CONSTRAINT IF EXISTS {DB_TABLE}_text_hash_key;")
                cur.execute(f"""
                    CREATE UNIQUE INDEX IF NOT EXISTS idx_{DB_TABLE}_document_text_hash
                    ON {DB_TABLE}(document_id, text_hash);
                """)
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{DB_TABLE}_document_id
                    ON {DB_TABLE}(document_id);
                """)
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{DB_TABLE}_tenant_id
                    ON {DB_TABLE}(tenant_id);
                """)
            conn.commit()

            # --- Stage 5: best-effort extras, each isolated by a savepoint so
            # a failure here can't roll back stages 1-4 ---
            with conn.cursor() as cur:
                cur.execute("SAVEPOINT sp_fk")
                try:
                    cur.execute(f"""
                        ALTER TABLE {DB_TABLE}
                        ADD CONSTRAINT fk_{DB_TABLE}_document
                        FOREIGN KEY (document_id) REFERENCES {DOCUMENTS_TABLE}(id);
                    """)
                    cur.execute("RELEASE SAVEPOINT sp_fk")
                except Exception as e:
                    cur.execute("ROLLBACK TO SAVEPOINT sp_fk")
                    logger.debug(f"Skipping document_id foreign key (likely already present): {e}")

                # IVFFLAT index using the L2 operator class (vector_l2_ops) —
                # this matches the "<->" operator used in query_top_k below.
                # An index built with vector_cosine_ops is ignored by the
                # planner for an ORDER BY using "<->", silently falling back
                # to a full sequential scan on every query. Drop-then-create
                # (rather than IF NOT EXISTS) because a database provisioned
                # before this fix already has an index of this name built
                # with the wrong operator class.
                cur.execute("SAVEPOINT sp_ivfflat")
                try:
                    cur.execute(f"DROP INDEX IF EXISTS idx_{DB_TABLE}_embedding_ivfflat;")
                    cur.execute(f"""
                        CREATE INDEX idx_{DB_TABLE}_embedding_ivfflat
                        ON {DB_TABLE} USING ivfflat (embedding vector_l2_ops)
                        WITH (lists = 100);
                    """)
                    cur.execute("RELEASE SAVEPOINT sp_ivfflat")
                except Exception as e:
                    cur.execute("ROLLBACK TO SAVEPOINT sp_ivfflat")
                    logger.warning(f"Could not (re)create IVFFLAT index (may need more data): {e}")
            conn.commit()

            logger.info("✅ Database schema initialized/migrated")
        finally:
            self._return_connection(conn)

    # === Document identity ===
    def get_or_create_document(
        self,
        source_uri: str,
        tenant_id: str = DEFAULT_TENANT_ID,
        source_type: str = "url",
        checksum: Optional[str] = None
    ) -> str:
        """Resolve a source URI to a stable document_id, creating (or checksum-refreshing) a document record."""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT id FROM {DOCUMENTS_TABLE} WHERE tenant_id = %s AND source_uri = %s;",
                    (tenant_id, source_uri)
                )
                row = cur.fetchone()
                if row:
                    document_id = row[0]
                    if checksum:
                        cur.execute(
                            f"UPDATE {DOCUMENTS_TABLE} SET checksum = %s WHERE id = %s;",
                            (checksum, document_id)
                        )
                        conn.commit()
                    return document_id

                document_id = uuid.uuid4().hex
                cur.execute(
                    f"""
                    INSERT INTO {DOCUMENTS_TABLE} (id, tenant_id, source_type, source_uri, checksum)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (tenant_id, source_uri) DO NOTHING;
                    """,
                    (document_id, tenant_id, source_type, source_uri, checksum)
                )
                conn.commit()

                if cur.rowcount == 0:
                    # A concurrent request created it first.
                    cur.execute(
                        f"SELECT id FROM {DOCUMENTS_TABLE} WHERE tenant_id = %s AND source_uri = %s;",
                        (tenant_id, source_uri)
                    )
                    row = cur.fetchone()
                    return row[0]

                return document_id
        finally:
            self._return_connection(conn)

    # === VectorStore contract ===
    @monitor_performance("upsert_chunks_batch")
    def upsert_chunks(
        self,
        chunks: List[Chunk],
        document_id: str,
        tenant_id: str = DEFAULT_TENANT_ID,
        batch_size: int = 100
    ) -> None:
        """Embed and upsert Chunk records against an existing document_id, deduplicating within that document."""
        if not chunks:
            logger.warning("No chunks to upsert")
            return

        conn = self._get_connection()
        try:
            inserted_count = 0
            skipped_count = 0
            indexed_chunks = list(enumerate(chunks))

            for batch in batch_items(indexed_chunks, batch_size):
                texts = [chunk.text for _, chunk in batch]
                # One batched embedding call per batch, via the shared encoder.
                embeddings = embed_passages_batch(texts)

                batch_data = [
                    (
                        chunk.text,
                        compute_text_hash(chunk.text),
                        emb,
                        document_id,
                        tenant_id,
                        chunk_index,
                        chunk.page_number,
                        chunk.chunk_type,
                        chunk.sheet_name
                    )
                    for (chunk_index, chunk), emb in zip(batch, embeddings)
                ]

                with conn.cursor() as cur:
                    for chunk_text, text_hash, embedding, doc_id, ten_id, chunk_index, page_number, chunk_type, sheet_name in batch_data:
                        try:
                            cur.execute(
                                f"""
                                INSERT INTO {DB_TABLE}
                                    (chunk_text, text_hash, embedding, document_id, tenant_id,
                                     chunk_index, page_number, chunk_type, sheet_name)
                                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                                ON CONFLICT (document_id, text_hash) DO NOTHING;
                                """,
                                (chunk_text, text_hash, embedding, doc_id, ten_id, chunk_index, page_number, chunk_type, sheet_name)
                            )
                            if cur.rowcount > 0:
                                inserted_count += 1
                            else:
                                skipped_count += 1
                        except Exception as e:
                            logger.error(f"Error inserting chunk: {e}")

                conn.commit()

            logger.info(f"✅ Upserted {inserted_count} chunks, skipped {skipped_count} duplicates for document_id '{document_id}'")
        finally:
            self._return_connection(conn)

    @monitor_performance("query_top_k")
    def query_top_k(
        self,
        query: str,
        k: int,
        document_id: Optional[str] = None,
        tenant_id: str = DEFAULT_TENANT_ID
    ) -> List[RetrievedChunk]:
        """
        Query top-k most relevant chunks using vector similarity (L2 distance
        on normalized vectors, equivalent in ranking to cosine similarity).

        Returns RetrievedChunk records (joined against `documents` for a
        human-readable source_uri) rather than bare strings, since the whole
        point of retrieval scoping is to be able to cite where an answer
        came from.
        """
        query_embedding = embed_query(query)

        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                select = f"""
                    SELECT c.chunk_text, c.document_id, d.source_uri, c.page_number,
                           c.sheet_name, c.chunk_type, (c.embedding <-> %s::vector) as distance
                    FROM {DB_TABLE} c
                    LEFT JOIN {DOCUMENTS_TABLE} d ON c.document_id = d.id
                """
                if document_id:
                    cur.execute(
                        select + " WHERE c.document_id = %s AND c.tenant_id = %s ORDER BY distance LIMIT %s;",
                        (query_embedding, document_id, tenant_id, k)
                    )
                else:
                    cur.execute(
                        select + " WHERE c.tenant_id = %s ORDER BY distance LIMIT %s;",
                        (query_embedding, tenant_id, k)
                    )
                results = cur.fetchall()

            return [
                RetrievedChunk(
                    text=row[0],
                    document_id=row[1],
                    source_uri=row[2],
                    page_number=row[3],
                    sheet_name=row[4],
                    chunk_type=row[5] or "paragraph"
                )
                for row in results
            ]
        finally:
            self._return_connection(conn)

    def delete_document(self, document_id: str, tenant_id: str = DEFAULT_TENANT_ID) -> None:
        """Remove every chunk and the document record for a given document_id."""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"DELETE FROM {DB_TABLE} WHERE document_id = %s AND tenant_id = %s;",
                    (document_id, tenant_id)
                )
                cur.execute(
                    f"DELETE FROM {DOCUMENTS_TABLE} WHERE id = %s AND tenant_id = %s;",
                    (document_id, tenant_id)
                )
            conn.commit()
            logger.info(f"✅ Deleted document '{document_id}' and its chunks")
        finally:
            self._return_connection(conn)

    def count(self) -> int:
        """Total number of stored chunks."""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {DB_TABLE};")
                return cur.fetchone()[0]
        finally:
            self._return_connection(conn)

    # === Maintenance (not part of the shared interface) ===
    def clear(self) -> None:
        """Clear all data from the chunks and documents tables."""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(f"TRUNCATE TABLE {DB_TABLE} RESTART IDENTITY;")
                cur.execute(f"TRUNCATE TABLE {DOCUMENTS_TABLE} CASCADE;")
            conn.commit()
            logger.info("✅ Database cleared")
        finally:
            self._return_connection(conn)
