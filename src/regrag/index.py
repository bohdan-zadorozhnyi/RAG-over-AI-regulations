"""Step 4b. Build load-dependent indexes after loading.

Migrations own tables, constraints and btree indexes. This module owns the
expensive indexes whose quality or cost depends on the data being present:
GIN on chunks.tsv and HNSW on embeddings.vector. One place, built last.
"""

import psycopg

from regrag.config import settings

CREATE_GIN = "CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING GIN (tsv)"

CREATE_HNSW = """
CREATE INDEX IF NOT EXISTS embeddings_vector_idx
    ON embeddings USING hnsw (vector vector_cosine_ops)
    WITH (m = 16, ef_construction = 64)
"""

ANALYZE = "ANALYZE chunks, embeddings"


def build_indexes() -> None:
    """Idempotent: IF NOT EXISTS on both, then refresh planner statistics."""
    with psycopg.connect(settings.database_url, autocommit=True) as conn:
        conn.execute(CREATE_GIN)
        conn.execute(CREATE_HNSW)
        conn.execute(ANALYZE)
