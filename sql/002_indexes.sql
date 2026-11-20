-- Btree indexes only. Load-dependent indexes (GIN on chunks.tsv, HNSW on
-- embeddings.vector) are built after loading by `make index` (regrag.index).
CREATE INDEX IF NOT EXISTS chunks_document_idx ON chunks (document_id);
CREATE INDEX IF NOT EXISTS sections_article_idx ON sections (document_id, article_no);
