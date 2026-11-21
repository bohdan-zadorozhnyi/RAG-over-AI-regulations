"""Step 4b. Build load-dependent indexes after loading.

Migrations own tables, constraints and btree indexes. This module owns the
expensive indexes whose quality or cost depends on the data being present:
GIN on chunks.tsv and HNSW on embeddings.vector. One place, built last.
"""

from psycopg import sql

from regrag.config import settings
from regrag.db import connect

CREATE_GIN = "CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING GIN (tsv)"

# DDL cannot take bind parameters, so the build settings are composed in as
# SQL literals from config.
CREATE_HNSW = sql.SQL("""
CREATE INDEX IF NOT EXISTS embeddings_vector_idx
    ON embeddings USING hnsw (vector vector_cosine_ops)
    WITH (m = {m}, ef_construction = {ef_construction})
""")

ANALYZE = "ANALYZE chunks, embeddings"


def build_indexes() -> None:
    """Idempotent: IF NOT EXISTS on both, then refresh planner statistics.

    IF NOT EXISTS means changed HNSW settings take effect only after the index
    is dropped (DROP INDEX embeddings_vector_idx) and rebuilt.
    """
    create_hnsw = CREATE_HNSW.format(
        m=sql.Literal(settings.hnsw_m),
        ef_construction=sql.Literal(settings.hnsw_ef_construction),
    )
    with connect() as conn:
        conn.execute(CREATE_GIN)
        conn.execute(create_hnsw)
        conn.execute(ANALYZE)
