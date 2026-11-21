# reg-rag

Retrieval system over EU AI and data regulation. Answers questions with
article-level citations. Built as a re-runnable data pipeline with
retrieval quality measured in CI.

Read `@PLAN.md` for the full build plan, schema and step order.

## Commands

- `make db` — start Postgres, apply pending migrations
- `make ingest` — download source documents to `data/raw/`
- `make parse` — raw documents to `sections` rows
- `make chunk` — sections to `chunks` rows
- `make embed` — generate and store embeddings
- `make index` — build HNSW and GIN indexes
- `make ask Q="..."` — retrieve and answer
- `make eval` — run the eval set, write `eval-report.md`
- `make check` — ruff, mypy, pytest

## Conventions

- Python 3.12 (pinned in `.python-version`), dependencies via `uv`. Never edit `uv.lock` by hand.
- `psycopg` 3 with raw SQL. No ORM. SQL lives in `sql/` or in module-level
  constants, never inline in a function body.
- Migrations are numbered files in `sql/`, tracked in `schema_migrations`.
  Never edit an applied migration; add a new one. Migrations own tables,
  constraints and btree indexes; load-dependent indexes (GIN, HNSW) live in
  `index.py`.
- Every pipeline stage is a module in `src/regrag/` exposing one entry
  function, and a `cli.py` subcommand. Stages read and write the database or
  `data/`, never call each other directly.
- Every stage must be idempotent. Re-running must not duplicate rows.
  Use content hashes, not timestamps, to decide what to skip.
- Type hints on every public function. `mypy --strict` must pass.
- Config comes from `config.py` reading environment variables. No literals
  for model names, dimensions, chunk sizes or `k` values anywhere else.

## Rules

- Never commit anything from `data/raw/` or `data/processed/`.
- Never hardcode an AWS region, bucket name or connection string.
- The embedding model name and vector dimension are stored on every
  embeddings row. Changing to a model of the same dimension means a new row,
  not an update. The column is `VECTOR(1024)` because HNSW needs a fixed
  dimension, so a model of a different dimension needs a new migration.
- Retrieval changes require a `make eval` run before the commit, and the
  metrics go in the commit message. The initial implementation is exempt —
  there is no baseline yet; the first measured metrics land with step 6.
- Do not add a dependency without saying why in the commit message.
  `langchain-text-splitters` is present only as an eval baseline.
- Answers must cite article numbers. If retrieved chunks do not support an
  answer, the correct output is a refusal, not a hedge.

## Out of scope

No web UI, no self-hosted Langfuse (Langfuse Cloud or `query_log`), no authentication, no multi-tenancy, no streaming ingestion.
Resist adding them.
