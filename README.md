# reg-rag

Retrieval over EU AI and data regulation — the AI Act, GDPR and DORA — that
answers questions with article-level citations.

Built because an answer about regulation is worthless without a traceable
source. That single constraint decides the architecture: retrieval rather
than fine-tuning, metadata carried through every stage, and retrieval quality
measured on every change rather than eyeballed.

## Architecture

TODO step 9: diagram.

Sources (EUR-Lex XHTML) → parse to sections → section-aware chunking →
embeddings → Postgres with pgvector and full-text search → hybrid retrieval
with reciprocal rank fusion → answer with citations. Evaluation runs in CI.

## Results

TODO step 9: fill from eval-report.md.

| Strategy | hit@5 | MRR | Refusal accuracy |
|---|---|---|---|
| section_aware | — | — | — |
| recursive (baseline) | — | — | — |

## Running it

```bash
make db        # Postgres + pgvector, schema applied
make all       # ingest, parse, chunk, embed, index
make ask Q="what must a provider of a high-risk AI system document?"
make eval
```

## Design notes

TODO step 9. Cover, honestly:

- Why hybrid retrieval rather than pure vector — article references are
  lexical, obligations are semantic, and the eval shows the gap
- Why section-aware chunking, and by how much it beat the recursive baseline
- Why recitals are retrievable but filterable
- How the eval set was built and what it does not cover

## Limitations

TODO step 9. Be specific. What this does not handle: synthesis across more
than a couple of provisions, applying a requirement to a concrete system,
amendments after the retrieval date, and anything outside the three
regulations indexed. This is a retrieval tool, not a compliance opinion.
