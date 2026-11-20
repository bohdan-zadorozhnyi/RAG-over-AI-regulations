-- Bootstrap. Applied unconditionally by `make migrate` before any other file,
-- so it must stay idempotent.
SET client_min_messages = warning;
CREATE TABLE IF NOT EXISTS schema_migrations (
    filename   TEXT PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
