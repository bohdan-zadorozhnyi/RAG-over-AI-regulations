CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id           BIGSERIAL PRIMARY KEY,
    celex        TEXT NOT NULL UNIQUE,
    title        TEXT NOT NULL,
    source_url   TEXT NOT NULL,
    retrieved_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    checksum     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sections (
    id            BIGSERIAL PRIMARY KEY,
    document_id   BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    path          TEXT NOT NULL,
    kind          TEXT NOT NULL CHECK (kind IN ('recital', 'article', 'annex')),
    article_no    TEXT,
    paragraph_no  TEXT,
    heading       TEXT,
    text          TEXT NOT NULL,
    UNIQUE (document_id, path)
);

CREATE TABLE IF NOT EXISTS chunks (
    id           BIGSERIAL PRIMARY KEY,
    document_id  BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    section_id   BIGINT NOT NULL REFERENCES sections(id) ON DELETE CASCADE,
    strategy     TEXT NOT NULL,
    ordinal      INT NOT NULL,
    header       TEXT NOT NULL,
    text         TEXT NOT NULL,
    token_count  INT NOT NULL,
    content_hash TEXT NOT NULL,
    tsv          TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', header || ' ' || text)) STORED,
    UNIQUE (strategy, content_hash)
);

-- VECTOR(1024) is fixed because HNSW needs a fixed dimension. A new model of
-- the same dimension is a new row; a model of a different dimension needs a
-- new migration (a new column or table), not a new row.
CREATE TABLE IF NOT EXISTS embeddings (
    chunk_id BIGINT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    model    TEXT NOT NULL,
    dim      INT NOT NULL,
    vector   VECTOR(1024) NOT NULL,
    PRIMARY KEY (chunk_id, model)
);

CREATE TABLE IF NOT EXISTS eval_questions (
    id                TEXT PRIMARY KEY,
    question          TEXT NOT NULL,
    expected_articles TEXT[] NOT NULL,
    category          TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS eval_runs (
    id          BIGSERIAL PRIMARY KEY,
    run_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    git_sha     TEXT,
    config_hash TEXT NOT NULL,
    strategy    TEXT NOT NULL,
    metrics     JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS eval_results (
    run_id         BIGINT NOT NULL REFERENCES eval_runs(id) ON DELETE CASCADE,
    question_id    TEXT NOT NULL REFERENCES eval_questions(id),
    first_hit_rank INT,
    hit_at_5       BOOLEAN NOT NULL,
    retrieved      JSONB NOT NULL,
    PRIMARY KEY (run_id, question_id)
);

CREATE TABLE IF NOT EXISTS query_log (
    id         BIGSERIAL PRIMARY KEY,
    at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    question   TEXT NOT NULL,
    latency_ms INT NOT NULL,
    retrieved  JSONB NOT NULL,
    answer     TEXT,
    cost_usd   NUMERIC(10, 6)
);
